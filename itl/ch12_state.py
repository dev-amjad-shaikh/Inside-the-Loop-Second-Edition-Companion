"""Chapter 12 helpers: a small run store in SQLite.

Every change to run state goes through RunStore.transaction(). Inside one
transaction the store checks who is writing (ownership), whether the writer
still holds the run (fencing), whether the change is a legal transition for the
record, appends one event per change and updates the current-state tables. The
events are the history; the tables are a projection kept in step with them in
the same transaction, and fold() rebuilds the same projection from events alone.
"""
import hashlib
import json
import os
import shutil
import sqlite3

from itl.core import FIXTURES, Workspace

SCHEMA_VERSION = 2

OP_STATES = ("prepared", "awaiting_approval", "dispatched", "completed",
             "failed_before_effect", "uncertain", "cancelled", "skipped")
RUN_STATES = ("running", "waiting_for_person", "needs_verification", "completed",
              "failed", "cancelled")

SCHEMA = f"""
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs(
  run_id TEXT PRIMARY KEY,
  task_revision INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN {RUN_STATES}),
  step INTEGER NOT NULL DEFAULT 0,
  owner TEXT, fence INTEGER NOT NULL DEFAULT 0, lease_expires INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1);
CREATE TABLE task_revisions(
  run_id TEXT NOT NULL REFERENCES runs, revision INTEGER NOT NULL,
  text TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY(run_id, revision));
CREATE TABLE model_calls(
  call_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs, ordinal INTEGER NOT NULL,
  task_revision INTEGER NOT NULL, input_sha TEXT NOT NULL, input_messages INTEGER NOT NULL,
  response TEXT NOT NULL);
CREATE TABLE operations(
  op_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs,
  model_call_id TEXT NOT NULL REFERENCES model_calls, tool TEXT NOT NULL,
  arguments TEXT NOT NULL, task_revision INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN {OP_STATES}),
  covers TEXT, fence INTEGER, version INTEGER NOT NULL DEFAULT 1);
CREATE TABLE observations(
  op_id TEXT PRIMARY KEY REFERENCES operations, status TEXT NOT NULL, result TEXT NOT NULL);
CREATE TABLE artifact_versions(
  path TEXT NOT NULL, version_no INTEGER NOT NULL, content TEXT NOT NULL,
  produced_by TEXT REFERENCES operations, PRIMARY KEY(path, version_no));
CREATE TABLE events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
  run_id TEXT NOT NULL, v INTEGER NOT NULL, type TEXT NOT NULL, entity TEXT NOT NULL,
  writer TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE snapshots(
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, schema_version INTEGER NOT NULL,
  state TEXT NOT NULL, state_sha TEXT NOT NULL, PRIMARY KEY(run_id, seq));
CREATE TABLE refusals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, writer TEXT NOT NULL, type TEXT NOT NULL,
  entity TEXT NOT NULL, reason TEXT NOT NULL, payload TEXT NOT NULL);
"""

# One table decides who may write what. Nothing else in the program writes run state.
OWNERS = {
    "run.created": "control",
    "lease.claimed": "control",
    "task.revised": "control",
    "run.status": "control",
    "run.completed": "control",
    "model_call.recorded": "model_adapter",
    "op.prepared": "control",
    "op.cancelled": "control",
    "op.skipped": "control",
    "op.awaiting_approval": "policy",
    "op.dispatched": "executor",
    "op.completed": "executor",
    "op.failed_before_effect": "executor",
    "op.uncertain": "reconciler",
    "observation.recorded": "executor",
    "artifact.versioned": "executor",
}
# Writes that change the run on a worker's behalf must carry the current fence.
FENCED = {"run.status", "run.completed", "model_call.recorded", "op.prepared", "op.cancelled",
          "op.skipped", "op.dispatched", "op.completed", "op.failed_before_effect",
          "observation.recorded", "artifact.versioned"}
# Legal operation transitions: event type -> states it may start from.
OP_FROM = {
    "op.awaiting_approval": {"prepared"},
    "op.dispatched": {"prepared", "awaiting_approval"},
    "op.completed": {"dispatched"},
    "op.failed_before_effect": {"prepared", "dispatched"},
    "op.uncertain": {"dispatched"},
    "op.cancelled": {"prepared", "awaiting_approval"},
    "op.skipped": {"prepared"},
}


class Refused(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Crash(Exception):
    pass


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj):
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()[:12]


def text_version(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class DirWorkspace(Workspace):
    """The chapter-3 workspace, placed in a directory we keep so a crash leaves it on disk."""

    def __init__(self, root, fixture="report", fresh=True):
        self.root = os.path.abspath(root)
        if fresh:
            shutil.rmtree(self.root, ignore_errors=True)
            shutil.copytree(os.path.join(FIXTURES, fixture), self.root)

    def close(self):
        pass


def connect(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class RunStore:
    """The single module that writes run state.

    policy controls which guards run, so the experiments can show what each guard prevents:
      "naive"   ownership only
      "version" ownership + expected record version
      "fenced"  ownership + expected record version + current lease fence (the default)
    """

    def __init__(self, path, create=False, policy="fenced"):
        self.path = path
        if create and os.path.exists(path):
            os.remove(path)
        self.conn = connect(path)
        self.policy = policy
        if create:
            self.conn.executescript(SCHEMA)
            self.conn.execute("INSERT INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            self.conn.execute("INSERT INTO meta VALUES('clock', '0')")
        found = int(self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0])
        if found > SCHEMA_VERSION:
            raise Refused(f"store schema {found} is newer than this build ({SCHEMA_VERSION})")

    # ---- time, read helpers -------------------------------------------------
    def now(self):
        return int(self.conn.execute("SELECT value FROM meta WHERE key='clock'").fetchone()[0])

    def advance(self, ticks):
        self.conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + ? WHERE key='clock'", (ticks,))

    def run(self, run_id):
        cur = self.conn.execute("SELECT run_id, task_revision, status, step, owner, fence, lease_expires, "
                                "version FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], row)) if row else None

    def op(self, op_id):
        cur = self.conn.execute("SELECT * FROM operations WHERE op_id=?", (op_id,))
        row = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], row)) if row else None

    def artifact_head(self, path):
        row = self.conn.execute("SELECT version_no, content FROM artifact_versions WHERE path=? "
                                "ORDER BY version_no DESC LIMIT 1", (path,)).fetchone()
        return (row[0], row[1]) if row else (0, None)

    def counts(self):
        out = {}
        for t in ("runs", "task_revisions", "model_calls", "operations", "observations",
                  "artifact_versions", "events", "snapshots", "refusals"):
            out[t] = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return out

    def op_states(self):
        return dict(self.conn.execute("SELECT state, COUNT(*) FROM operations GROUP BY state "
                                      "ORDER BY state").fetchall())

    # ---- the one write path ---------------------------------------------------
    def transaction(self, writer, run_id, holder=None):
        return _Tx(self, writer, run_id, holder)

    def record_refusal(self, writer, etype, entity, reason, payload):
        self.conn.execute("INSERT INTO refusals(writer, type, entity, reason, payload) VALUES(?,?,?,?,?)",
                          (writer, etype, entity, reason, canon(payload)))


class _Tx:
    """BEGIN IMMEDIATE ... COMMIT. Any Refused rolls back everything in the block."""

    def __init__(self, store, writer, run_id, holder):
        self.s, self.writer, self.run_id, self.holder = store, writer, run_id, holder
        self.appended = []

    def __enter__(self):
        self.s.conn.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.s.conn.execute("COMMIT")
            return False
        self.s.conn.execute("ROLLBACK")
        if exc_type is Refused and getattr(exc, "record", None):
            self.s.record_refusal(*exc.record)
        return False

    def append(self, etype, entity, payload, expect_version=None, event_id=None):
        s, conn = self.s, self.s.conn

        def refuse(reason):
            e = Refused(reason)
            e.record = (self.writer, etype, entity, reason, payload)
            raise e
        # 1. ownership
        owner = OWNERS.get(etype)
        if owner is None or owner != self.writer:
            refuse(f"{self.writer} may not write {etype} (owner: {owner})")
        # 2. fencing
        if s.policy == "fenced" and etype in FENCED:
            run = s.run(self.run_id)
            if self.holder is None or run is None or run["owner"] != self.holder[0] \
                    or run["fence"] != self.holder[1]:
                cur = (run["owner"], run["fence"]) if run else None
                refuse(f"stale holder {self.holder}; current lease {cur}")
        # 3. transition legality and expected version
        if etype in OP_FROM:
            op = s.op(entity)
            if op is None:
                refuse(f"unknown operation {entity}")
            if s.policy in ("version", "fenced") and op["state"] not in OP_FROM[etype]:
                refuse(f"{etype} not allowed from state {op['state']}")
            if s.policy in ("version", "fenced") and expect_version is not None \
                    and op["version"] != expect_version:
                refuse(f"operation {entity} is at version {op['version']}, writer expected {expect_version}")
        # 4. append the event (duplicate event ids are ignored, not applied twice)
        eid = event_id or f"{entity}:{etype}"
        cur = conn.execute("INSERT OR IGNORE INTO events(event_id, run_id, v, type, entity, writer, payload) "
                           "VALUES(?,?,?,?,?,?,?)",
                           (eid, self.run_id, SCHEMA_VERSION, etype, entity, self.writer, canon(payload)))
        if cur.rowcount == 0:
            return False
        apply_to_tables(conn, self.run_id, etype, entity, payload)
        self.appended.append(eid)
        return True


def apply_to_tables(conn, run_id, etype, entity, p):
    if etype == "run.created":
        conn.execute("INSERT INTO runs(run_id, task_revision, status) VALUES(?,?,?)",
                     (run_id, 1, "running"))
        conn.execute("INSERT INTO task_revisions VALUES(?,?,?,?)", (run_id, 1, p["text"], p["source"]))
    elif etype == "lease.claimed":
        conn.execute("UPDATE runs SET owner=?, fence=?, lease_expires=?, version=version+1 WHERE run_id=?",
                     (p["owner"], p["fence"], p["expires"], run_id))
    elif etype == "task.revised":
        conn.execute("INSERT INTO task_revisions VALUES(?,?,?,?)", (run_id, p["revision"], p["text"], p["source"]))
        conn.execute("UPDATE runs SET task_revision=?, version=version+1 WHERE run_id=?", (p["revision"], run_id))
    elif etype in ("run.status", "run.completed"):
        conn.execute("UPDATE runs SET status=?, version=version+1 WHERE run_id=?", (p["status"], run_id))
    elif etype == "model_call.recorded":
        conn.execute("INSERT INTO model_calls VALUES(?,?,?,?,?,?,?)",
                     (entity, run_id, p["ordinal"], p["task_revision"], p["input_sha"],
                      p["input_messages"], canon(p["response"])))
        conn.execute("UPDATE runs SET step=?, version=version+1 WHERE run_id=?", (p["ordinal"], run_id))
    elif etype == "op.prepared":
        conn.execute("INSERT INTO operations(op_id, run_id, model_call_id, tool, arguments, task_revision, "
                     "state) VALUES(?,?,?,?,?,?,?)",
                     (entity, run_id, p["model_call"], p["tool"], canon(p["arguments"]),
                      p["task_revision"], "prepared"))
    elif etype.startswith("op."):
        state = etype[3:]
        if etype == "op.dispatched":
            conn.execute("UPDATE operations SET state=?, covers=?, fence=?, version=version+1 WHERE op_id=?",
                         (state, canon(p["covers"]), p["fence"], entity))
        else:
            conn.execute("UPDATE operations SET state=?, version=version+1 WHERE op_id=?", (state, entity))
    elif etype == "observation.recorded":
        conn.execute("INSERT INTO observations VALUES(?,?,?)", (entity, p["status"], canon(p["result"])))
    elif etype == "artifact.versioned":
        conn.execute("INSERT INTO artifact_versions VALUES(?,?,?,?)",
                     (p["path"], p["version_no"], p["content"], p.get("produced_by")))
    else:
        raise ValueError(etype)


# ---- the fold: current state from events alone ---------------------------------
def empty_state():
    return {"run": None, "tasks": {}, "model_calls": {}, "operations": {}, "observations": {},
            "artifacts": {}}


UPCAST = {1: {"op.finished": "op.completed"}}   # event types renamed between schema 1 and 2


def upcast(event):
    v, etype = event["v"], event["type"]
    if v > SCHEMA_VERSION:
        raise Refused(f"event {event['event_id']} has schema {v}; this build reads up to {SCHEMA_VERSION}")
    while v < SCHEMA_VERSION:
        etype = UPCAST.get(v, {}).get(etype, etype)
        v += 1
    return dict(event, v=v, type=etype)


def fold(events, state=None):
    st = json.loads(canon(state)) if state is not None else empty_state()
    for raw in events:
        e = upcast(raw)
        t, ent, p = e["type"], e["entity"], e["payload"]
        run = st["run"]
        if t == "run.created":
            st["run"] = {"status": "running", "task_revision": 1, "step": 0, "owner": None, "fence": 0}
            st["tasks"]["1"] = p["text"]
        elif t == "lease.claimed":
            run["owner"], run["fence"] = p["owner"], p["fence"]
        elif t == "task.revised":
            st["tasks"][str(p["revision"])] = p["text"]
            run["task_revision"] = p["revision"]
        elif t in ("run.status", "run.completed"):
            run["status"] = p["status"]
        elif t == "model_call.recorded":
            st["model_calls"][ent] = {"task_revision": p["task_revision"], "input_sha": p["input_sha"]}
            run["step"] = p["ordinal"]
        elif t == "op.prepared":
            st["operations"][ent] = {"tool": p["tool"], "state": "prepared", "covers": None,
                                     "task_revision": p["task_revision"]}
        elif t.startswith("op."):
            st["operations"][ent]["state"] = t[3:]
            if t == "op.dispatched":
                st["operations"][ent]["covers"] = p["covers"]
        elif t == "observation.recorded":
            st["observations"][ent] = p["status"]
        elif t == "artifact.versioned":
            st["artifacts"].setdefault(p["path"], []).append(text_version(p["content"]))
    return st


def read_events(conn, run_id, after=0, upto=None):
    q = "SELECT seq, event_id, v, type, entity, writer, payload FROM events WHERE run_id=? AND seq>?"
    args = [run_id, after]
    if upto is not None:
        q += " AND seq<=?"
        args.append(upto)
    out = []
    for seq, eid, v, t, ent, w, p in conn.execute(q + " ORDER BY seq", args):
        out.append({"seq": seq, "event_id": eid, "v": v, "type": t, "entity": ent, "writer": w,
                    "payload": json.loads(p)})
    return out


def state_from_tables(conn, run_id):
    """The same projection, read from the current-state tables instead of the events."""
    st = empty_state()
    r = conn.execute("SELECT status, task_revision, step, owner, fence FROM runs WHERE run_id=?",
                     (run_id,)).fetchone()
    st["run"] = {"status": r[0], "task_revision": r[1], "step": r[2], "owner": r[3], "fence": r[4]}
    for rev, text in conn.execute("SELECT revision, text FROM task_revisions WHERE run_id=?", (run_id,)):
        st["tasks"][str(rev)] = text
    for cid, trev, isha in conn.execute("SELECT call_id, task_revision, input_sha FROM model_calls "
                                        "WHERE run_id=?", (run_id,)):
        st["model_calls"][cid] = {"task_revision": trev, "input_sha": isha}
    for oid, tool, state, covers, trev in conn.execute(
            "SELECT op_id, tool, state, covers, task_revision FROM operations WHERE run_id=?", (run_id,)):
        st["operations"][oid] = {"tool": tool, "state": state,
                                 "covers": json.loads(covers) if covers else None, "task_revision": trev}
    for oid, status in conn.execute("SELECT o.op_id, o.status FROM observations o JOIN operations p "
                                    "ON o.op_id=p.op_id WHERE p.run_id=?", (run_id,)):
        st["observations"][oid] = status
    for path, content in conn.execute("SELECT path, content FROM artifact_versions ORDER BY path, version_no"):
        st["artifacts"].setdefault(path, []).append(text_version(content))
    return st
