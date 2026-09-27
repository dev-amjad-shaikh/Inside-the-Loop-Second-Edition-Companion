"""Chapter 12: execution state.

A run store in SQLite (itl/ch12_state.py) drives the Chapter 3 reporting run with
real file edits and real test runs, then five experiments:

  1 transaction boundaries: a process is killed at five points around one edit;
    the store and the workspace are reopened and inspected
  2 competing updates: a lost update without a conditional write, and a stale
    worker whose late write is stopped only by a fence
  3 reconstruction: current state folded from events, checked against a snapshot
    and the tables; a rebuild that invokes no tool; a replay that does
  4 the interface as a projection: status from records against status from text
  5 ownership: writes attempted by components that do not own the record

    python3 ch12_state.py --output runs/ch12

The output directory is replaced if it exists.
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import InputError, ScriptedModel, check_args, report_tools, write_run  # noqa: E402
from itl.ch12_state import (DirWorkspace, Refused, RunStore, canon, connect, fold, read_events,  # noqa: E402
                            sha, state_from_tables, text_version, SCHEMA_VERSION)

RUN = "run-12"
SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: settings.json, report.py, test_report.py.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")
CORRECTION = "Keep the public function signature unchanged."
FILES = ("report.py", "settings.json", "test_report.py")
LEASE_TICKS = 30

NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''
OLD_DOC = '    """created_at is an aware UTC datetime; requested_day is a local date."""'
NEW_DOC = ('    """created_at is an aware UTC datetime; requested_day is a date in the\n'
           '    report timezone."""')


def call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def reply(content="", *calls):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def responses():
    return [
        reply("", call("c1", "read_file", path="settings.json"), call("c2", "read_file", path="report.py"),
              call("c3", "read_file", path="test_report.py")),
        reply("", call("c4", "replace_text", path="test_report.py",
                       old='\n\nif __name__ == "__main__":', new="\n" + NEW_TEST)),
        reply("", call("c5", "run_tests", tests="test_report.py")),
        reply("", call("c6", "run_tests", tests=["test_report.py"])),
        reply("", call("c7", "replace_text", path="report.py",
                       old="    return created_at.date() == requested_day",
                       new="    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
                           "    return local_time.date() == requested_day")),
        reply("", call("c8", "run_tests", tests=["test_report.py"])),
        reply("Both tests pass. Exporting the corrected report for September 11.",
              call("c9", "export_report", day="2026-09-11", destination="finance-share")),
        reply("The public signature is unchanged. All tests passed.",
              call("c10", "replace_text", path="report.py", old=OLD_DOC, new=NEW_DOC)),
        reply("", call("c11", "run_tests", tests=["test_report.py"])),
        reply("The filter used the UTC calendar date; it now converts to the report timezone first. "
              "The signature is unchanged, both tests pass on the current revision, and nothing was "
              "exported or deployed.")]


class Stop(Exception):
    pass


class Runtime:
    """A durable version of the Chapter 3 loop. It keeps no run state in memory:
    each model input is rebuilt from the store before the call."""

    def __init__(self, store, ws, owner, crash_point=None, record_intent=True, crash_op="c7"):
        self.store, self.ws, self.owner = store, ws, owner
        self.tools = report_tools(ws)
        self.tools.register("export_report", "Export the daily report file to a destination.", {
            "type": "object", "properties": {"day": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["day", "destination"], "additionalProperties": False}, self._export)
        self.approval_required = {"export_report"}
        self.crash_point, self.crash_op, self.record_intent = crash_point, crash_op, record_intent
        self.fence = None
        self.status_n = 0

    def _export(self, day, destination):
        raise RuntimeError("export_report is never executed in this chapter")

    def crash(self, point, op_id):
        if self.crash_point == point and op_id == self.crash_op:
            os._exit(70)          # the process dies here: no cleanup, no commit

    def holder(self):
        return (self.owner, self.fence)

    # ---- control --------------------------------------------------------------
    def create(self):
        s = self.store
        with s.transaction("control", RUN) as tx:
            tx.append("run.created", RUN, {"text": TASK, "source": "user"}, event_id=f"{RUN}:created")
        self.claim()
        with s.transaction("executor", RUN, self.holder()) as tx:
            for path in FILES:
                tx.append("artifact.versioned", f"{path}@1",
                          {"path": path, "version_no": 1, "content": self.ws.read(path), "produced_by": None})

    def claim(self):
        s = self.store
        with s.transaction("control", RUN) as tx:
            run = s.run(RUN)
            if run["owner"] not in (None, self.owner) and run["lease_expires"] > s.now():
                raise Refused(f"lease held by {run['owner']} until {run['lease_expires']}")
            fence = run["fence"] + 1
            tx.append("lease.claimed", RUN, {"owner": self.owner, "fence": fence,
                                             "expires": s.now() + LEASE_TICKS},
                      event_id=f"{RUN}:lease:{fence}")
        self.fence = fence
        return fence

    def set_status(self, status):
        self.status_n += 1
        with self.store.transaction("control", RUN, self.holder()) as tx:
            tx.append("run.status", RUN, {"status": status}, event_id=f"{RUN}:status:{self.status_n}:{status}")

    def revise_task(self, text):
        run = self.store.run(RUN)
        rev = run["task_revision"] + 1
        with self.store.transaction("control", RUN, self.holder()) as tx:
            tx.append("task.revised", RUN, {"revision": rev, "text": text, "source": "user"},
                      event_id=f"{RUN}:task:{rev}")
            # proposals made under the old task that have not started are withdrawn
            for (op_id,) in self.store.conn.execute(
                    "SELECT op_id FROM operations WHERE run_id=? AND task_revision<? AND state IN "
                    "('prepared','awaiting_approval') ORDER BY op_id", (RUN, rev)).fetchall():
                tx.append("op.cancelled", op_id, {"reason": f"task revised to revision {rev} before approval"})
        return rev

    def evidence_for_completion(self):
        """The verifier's rule: the latest passing test run must cover the current versions."""
        s = self.store
        current = {p: text_version(s.artifact_head(p)[1]) for p in ("report.py", "test_report.py")}
        row = s.conn.execute(
            "SELECT o.op_id, o.covers FROM operations o JOIN observations b ON o.op_id=b.op_id "
            "WHERE o.run_id=? AND o.tool='run_tests' AND b.status='passed' "
            "ORDER BY (SELECT MAX(seq) FROM events e WHERE e.entity=o.op_id) DESC LIMIT 1", (RUN,)).fetchone()
        if row is None:
            return False, "no passing test run recorded", current
        covers = json.loads(row[1])
        stale = [p for p in current if covers.get(p) != current[p]]
        if stale:
            return False, f"latest passing run {row[0]} covers {', '.join(f'{p} {covers[p]}' for p in stale)}; " \
                          f"current is {', '.join(f'{p} {current[p]}' for p in stale)}", current
        return True, f"latest passing run {row[0]} covers the current versions", current

    def complete(self):
        ok, reason, _ = self.evidence_for_completion()
        if not ok:
            self.store.record_refusal("control", "run.completed", RUN, "verification: " + reason, {})
            return False, reason
        with self.store.transaction("control", RUN, self.holder()) as tx:
            tx.append("run.completed", RUN, {"status": "completed", "evidence": reason},
                      event_id=f"{RUN}:completed")
        return True, reason

    # ---- the model adapter and the context builder ------------------------------
    def model_input(self, upto=None):
        return model_input_from_records(self.store.conn, RUN, upto)

    def model_turn(self, model, ordinal):
        messages = self.model_input()
        response = model.generate(messages, self.tools.definitions())
        run = self.store.run(RUN)
        call_id = f"m{ordinal}"
        with self.store.transaction("model_adapter", RUN, self.holder()) as tx:
            tx.append("model_call.recorded", call_id,
                      {"ordinal": ordinal, "task_revision": run["task_revision"], "input_sha": sha(messages),
                       "input_messages": len(messages), "response": response})
        with self.store.transaction("control", RUN, self.holder()) as tx:
            for c in response["tool_calls"]:
                tx.append("op.prepared", c["id"], {"model_call": call_id, "tool": c["name"],
                                                   "arguments": c["arguments"],
                                                   "task_revision": run["task_revision"]})
        return response

    # ---- the executor -----------------------------------------------------------
    def covers(self, c):
        if c["name"] == "run_tests":
            return {p: text_version(self.ws.read(p)) for p in ("report.py", "test_report.py")}
        if c["name"] in ("replace_text", "read_file"):
            return {c["arguments"]["path"]: text_version(self.ws.read(c["arguments"]["path"]))}
        return {}

    def execute(self, c):
        s, op_id = self.store, c["id"]
        if c["name"] in self.approval_required:
            with s.transaction("policy", RUN) as tx:
                tx.append("op.awaiting_approval", op_id, {"requires": "person", "tool": c["name"]})
            return "awaiting_approval"
        try:
            check_args(self.tools.tools[c["name"]]["definition"]["parameters"], c["arguments"])
        except InputError as exc:
            with s.transaction("executor", RUN, self.holder()) as tx:
                tx.append("op.failed_before_effect", op_id, {"reason": str(exc)})
                tx.append("observation.recorded", op_id, {"status": "input_error", "result": {
                    "status": "input_error", "message": str(exc)}})
            return "failed_before_effect"
        covers = self.covers(c)
        # T1: record the intent to dispatch, with what the operation covers
        if self.record_intent:
            with s.transaction("executor", RUN, self.holder()) as tx:
                tx.append("op.dispatched", op_id, {"covers": covers, "fence": self.fence})
                self.crash("during_t1", op_id)
        self.crash("after_t1", op_id)
        result = self.tools.execute_checked(c)                      # the effect
        self.crash("after_effect", op_id)
        # T2: the outcome, the observation and any new artifact version, together
        with s.transaction("executor", RUN, self.holder()) as tx:
            status = result.get("status", "ok")
            if not self.record_intent:        # the flawed design writes dispatch and outcome together
                tx.append("op.dispatched", op_id, {"covers": covers, "fence": self.fence})
            tx.append("observation.recorded", op_id, {"status": status, "result": result})
            self.crash("during_t2", op_id)
            if status == "input_error":
                tx.append("op.failed_before_effect", op_id, {"reason": result["message"]})
            else:
                tx.append("op.completed", op_id, {"status": status}, expect_version=2)
            for path in FILES:
                n, content = s.artifact_head(path)
                now = self.ws.read(path)
                if content != now:
                    tx.append("artifact.versioned", f"{path}@{n + 1}",
                              {"path": path, "version_no": n + 1, "content": now, "produced_by": op_id})
        self.crash("after_t2", op_id)
        return status

    def restart_classification(self):
        """What a restarted worker may conclude on its own: a dispatched operation with no
        outcome is uncertain. Establishing what actually happened is Chapter 13's reconciler."""
        s = self.store
        moved = []
        for (op_id,) in s.conn.execute("SELECT op_id FROM operations WHERE run_id=? AND state='dispatched' "
                                       "AND op_id NOT IN (SELECT op_id FROM observations)", (RUN,)).fetchall():
            with s.transaction("reconciler", RUN) as tx:
                tx.append("op.uncertain", op_id, {"reason": "dispatched with no recorded outcome at restart"})
            moved.append(op_id)
        return moved

    def drive(self, model, start_ordinal, stop_after=None):
        ordinal = start_ordinal
        while True:
            response = self.model_turn(model, ordinal)
            if not response["tool_calls"]:
                return ordinal
            for c in response["tool_calls"]:
                if self.execute(c) == "awaiting_approval":
                    self.set_status("waiting_for_person")
                    return ordinal
            if stop_after is not None and ordinal >= stop_after:
                return ordinal
            ordinal += 1


def model_input_from_records(conn, run_id, upto=None):
    """The context builder: the next model input as a projection of the records."""
    messages = [{"role": "system", "content": SYSTEM}]
    for e in read_events(conn, run_id, upto=upto):
        t, p = e["type"], e["payload"]
        if t == "run.created":
            messages.append({"role": "user", "content": p["text"]})
        elif t == "task.revised":
            messages.append({"role": "user", "content": f"Correction (task revision {p['revision']}): {p['text']}"})
        elif t == "model_call.recorded":
            messages.append(p["response"])
        elif t == "observation.recorded":
            messages.append({"role": "tool", "tool_call_id": e["entity"], "content": p["result"]})
        elif t == "op.cancelled":
            messages.append({"role": "tool", "tool_call_id": e["entity"],
                             "content": {"status": "cancelled", "reason": p["reason"]}})
    return messages


def ui_from_records(store, rt):
    ok, reason, current = rt.evidence_for_completion()
    run = store.run(RUN)
    last = store.conn.execute(
        "SELECT o.op_id, b.status, o.covers FROM operations o JOIN observations b ON o.op_id=b.op_id "
        "WHERE o.run_id=? AND o.tool='run_tests' AND b.status!='input_error' "
        "ORDER BY (SELECT MAX(seq) FROM events e WHERE e.entity=o.op_id) DESC LIMIT 1", (RUN,)).fetchone()
    return {"run_status": run["status"], "task_revision": run["task_revision"],
            "latest_test": {"op": last[0], "status": last[1], "covers": json.loads(last[2])} if last else None,
            "current_versions": current,
            "verification": "satisfied" if ok else "required", "because": reason}


def ui_from_stream(text):
    """A deterministic stand-in for an interface that sets status from streamed model text."""
    return {"status": "tests passed" if "tests passed" in text.lower() else "working", "source_text": text}


def dump_store(store, run_id, out, name):
    conn = store.conn
    tables = {}
    for t in ("runs", "task_revisions", "model_calls", "operations", "observations", "artifact_versions",
              "refusals", "snapshots"):
        cur = conn.execute(f"SELECT * FROM {t}")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            for k in ("response", "content", "state", "result", "payload"):
                if k in r and isinstance(r[k], str) and len(r[k]) > 160:
                    r[k] = r[k][:157] + "..."
        tables[t] = rows
    write_run(out, f"{name}_tables.json", tables)
    write_run(out, f"{name}_events.json", [
        {k: e[k] for k in ("seq", "event_id", "type", "entity", "writer")} for e in read_events(conn, run_id)])


# =============================================================================
# The main run, with experiments 4 and 5 at the point where they bite
# =============================================================================
def main_run(out):
    ws = DirWorkspace(os.path.join(out, "main_workspace"))
    store = RunStore(os.path.join(out, "main.db"), create=True)
    rt = Runtime(store, ws, "worker-1")
    rt.create()
    model = ScriptedModel(responses())
    last = rt.drive(model, 1)                     # calls 1 to 7; c9 waits for a person
    snapshot_seq = store.conn.execute(
        "SELECT MAX(seq) FROM events WHERE entity='c8'").fetchone()[0]
    after_c8 = {"run": store.run(RUN)["status"], "ops": store.op_states()}

    # the person answers with a correction instead of an approval
    rev = rt.revise_task(CORRECTION)
    rt.set_status("running")
    after_revision = {"task_revision": rev, "c9": store.op("c9")["state"]}

    last = rt.drive(model, last + 1, stop_after=last + 1)    # call 8: says "tests passed", edits report.py
    m8 = json.loads(store.conn.execute("SELECT response FROM model_calls WHERE call_id='m8'").fetchone()[0])

    # Experiment 4: two statuses for the same moment
    projection = {"ui_from_stream": ui_from_stream(m8["content"]), "ui_from_records": ui_from_records(store, rt)}

    # Experiment 5: writes by components that do not own the record
    attempts = []

    def attempt(label, writer, etype, entity, payload, holder=None):
        try:
            with store.transaction(writer, RUN, holder) as tx:
                tx.append(etype, entity, payload, event_id=f"attempt:{label}")
            attempts.append({"attempt": label, "writer": writer, "type": etype, "result": "accepted"})
        except Refused as r:
            attempts.append({"attempt": label, "writer": writer, "type": etype, "result": "refused",
                             "reason": r.reason})
    attempt("browser marks run complete", "human_interface", "run.completed", RUN, {"status": "completed"})
    attempt("model text becomes a test result", "model", "observation.recorded", "c10",
            {"status": "passed", "result": {"claim": "All tests passed."}})
    attempt("control writes an operation outcome", "control", "op.completed", "c10", {"status": "ok"},
            rt.holder())
    attempt("executor marks run complete", "executor", "run.completed", RUN, {"status": "completed"},
            rt.holder())
    ok, reason = rt.complete()
    attempts.append({"attempt": "control completes before verification", "writer": "control",
                     "type": "run.completed", "result": "accepted" if ok else "refused",
                     "reason": "verification: " + reason})

    last = rt.drive(model, last + 1)               # call 9 runs tests, call 10 answers
    ok, reason = rt.complete()
    attempts.append({"attempt": "control completes after verification", "writer": "control",
                     "type": "run.completed", "result": "accepted" if ok else "refused", "reason": reason})
    projection["ui_from_records_final"] = ui_from_records(store, rt)

    # every model input can be rebuilt from the records, from a fresh connection
    fresh = connect(store.path)
    rebuilt = []
    for cid, recorded_sha, n in fresh.execute("SELECT call_id, input_sha, input_messages FROM model_calls "
                                              "ORDER BY ordinal"):
        seq = fresh.execute("SELECT seq FROM events WHERE entity=? AND type='model_call.recorded'",
                            (cid,)).fetchone()[0]
        msgs = model_input_from_records(fresh, RUN, upto=seq - 1)
        rebuilt.append({"call": cid, "messages": n, "recorded_sha": recorded_sha, "rebuilt_sha": sha(msgs),
                        "task_revision_seen": fresh.execute(
                            "SELECT task_revision FROM model_calls WHERE call_id=?", (cid,)).fetchone()[0]})
    fresh.close()

    audit = [dict(zip(("writer", "events"), r)) for r in store.conn.execute(
        "SELECT writer, COUNT(*) FROM events WHERE run_id=? GROUP BY writer ORDER BY writer", (RUN,))]
    refusals = [dict(zip(("writer", "type", "reason"), r)) for r in store.conn.execute(
        "SELECT writer, type, reason FROM refusals ORDER BY id")]
    result = {"after_call_7": after_c8, "after_revision": after_revision, "projection": projection,
              "ownership_attempts": attempts, "model_inputs_rebuilt": rebuilt, "audit_by_writer": audit,
              "refusals": refusals, "counts": store.counts(), "op_states": store.op_states(),
              "executions": len(rt.tools.executions),
              "artifact_versions": [dict(zip(("path", "version_no", "version", "produced_by"),
                                             (r[0], r[1], text_version(r[2]), r[3]))) for r in
                                    store.conn.execute("SELECT path, version_no, content, produced_by "
                                                       "FROM artifact_versions ORDER BY path, version_no")],
              "snapshot_seq": snapshot_seq}
    return store, rt, ws, result


# =============================================================================
# Experiment 3: reconstruction from events
# =============================================================================
def reconstruction(store, rt, ws, snapshot_seq, out):
    conn = store.conn
    # a snapshot of state as of the outcome of c8, with the sequence it covers
    state_k = fold(read_events(conn, RUN, upto=snapshot_seq))
    conn.execute("INSERT INTO snapshots VALUES(?,?,?,?,?)",
                 (RUN, snapshot_seq, SCHEMA_VERSION, canon(state_k), sha(state_k)))
    last_seq = conn.execute("SELECT MAX(seq) FROM events").fetchone()[0]
    executions_before = len(rt.tools.executions)
    full = fold(read_events(conn, RUN))
    snap_state, snap_seq = conn.execute("SELECT state, seq FROM snapshots WHERE run_id=? ORDER BY seq DESC "
                                        "LIMIT 1", (RUN,)).fetchone()
    from_snapshot = fold(read_events(conn, RUN, after=snap_seq), state=json.loads(snap_state))
    tables = state_from_tables(conn, RUN)
    rebuild = {"events": last_seq, "snapshot_seq": snap_seq, "tail_events": last_seq - snap_seq,
               "fold_all_sha": sha(full), "snapshot_plus_tail_sha": sha(from_snapshot),
               "tables_sha": sha(tables), "all_equal": sha(full) == sha(from_snapshot) == sha(tables),
               "tool_invocations_during_rebuild": len(rt.tools.executions) - executions_before,
               "workspace_unchanged": {p: text_version(ws.read(p)) for p in ("report.py", "test_report.py")}}

    # duplicate delivery of an event already applied
    before = store.counts()["events"]
    with store.transaction("executor", RUN, rt.holder()) as tx:
        applied = tx.append("observation.recorded", "c8", {"status": "passed", "result": {}})
    dup = {"applied": applied, "events_before": before, "events_after": store.counts()["events"],
           "state_sha_after": sha(fold(read_events(conn, RUN)))}

    # the wrong kind of replay: re-execute the recorded operations against the workspace
    replay_ws = DirWorkspace(os.path.join(out, "replay_workspace"), fresh=False)
    shutil.rmtree(replay_ws.root, ignore_errors=True)
    shutil.copytree(ws.root, replay_ws.root)
    tools = report_tools(replay_ws)
    before_v = {p: text_version(replay_ws.read(p)) for p in ("report.py", "test_report.py")}
    outcomes = []
    for op_id, tool, args in conn.execute("SELECT op_id, tool, arguments FROM operations WHERE run_id=? AND "
                                          "state='completed' ORDER BY rowid", (RUN,)).fetchall():
        res = tools.execute_checked({"id": op_id, "name": tool, "arguments": json.loads(args)})
        outcomes.append({"op": op_id, "tool": tool, "status": res.get("status"),
                         **({"message": res["message"]} if "message" in res else {}),
                         **({"tests_run": res["tests_run"]} if "tests_run" in res else {})})
    after_v = {p: text_version(replay_ws.read(p)) for p in ("report.py", "test_report.py")}
    reexec = {"tool_invocations": len(tools.executions), "outcomes": outcomes, "versions_before": before_v,
              "versions_after": after_v,
              "near_midnight_tests_in_file": replay_ws.read("test_report.py").count(
                  "def test_order_near_midnight")}

    # events written by an older build, and one written by a newer build
    legacy = [
        {"event_id": "L1", "v": 1, "type": "run.created", "entity": "run-old", "payload": {"text": TASK}},
        {"event_id": "L2", "v": 1, "type": "model_call.recorded", "entity": "m1",
         "payload": {"ordinal": 1, "task_revision": 1, "input_sha": "0" * 12}},
        {"event_id": "L3", "v": 1, "type": "op.prepared", "entity": "c1",
         "payload": {"tool": "read_file", "task_revision": 1}},
        {"event_id": "L4", "v": 1, "type": "op.dispatched", "entity": "c1", "payload": {"covers": {}}},
        {"event_id": "L5", "v": 1, "type": "op.finished", "entity": "c1", "payload": {}}]
    old = fold(legacy)
    try:
        fold(legacy + [{"event_id": "N1", "v": 3, "type": "op.compensated", "entity": "c1", "payload": {}}])
        newer = "accepted"
    except Refused as r:
        newer = "refused: " + r.reason
    versions = {"legacy_v1_events": len(legacy), "c1_state_after_upcast": old["operations"]["c1"]["state"],
                "newer_event": newer}
    return {"rebuild": rebuild, "duplicate_event": dup, "reexecution_replay": reexec,
            "schema_versions": versions}


# =============================================================================
# Experiment 1: transaction boundaries and crashes
# =============================================================================
POINTS = ("during_t1", "after_t1", "after_effect", "during_t2", "after_t2")


def child(point, d, record_intent):
    ws = DirWorkspace(os.path.join(d, "workspace"))
    store = RunStore(os.path.join(d, "store.db"), create=True)
    rt = Runtime(store, ws, "worker-1", crash_point=point, record_intent=record_intent)
    rt.create()
    rt.drive(ScriptedModel(responses()), 1)
    sys.exit(0)


def inspect_after_crash(d):
    """A new process opens what the dead one left behind."""
    ws = DirWorkspace(os.path.join(d, "workspace"), fresh=False)
    journal_left = os.path.exists(os.path.join(d, "store.db-journal"))
    store = RunStore(os.path.join(d, "store.db"))
    op = store.op("c7")
    obs = store.conn.execute("SELECT status FROM observations WHERE op_id='c7'").fetchone()
    found = {"rollback_journal_left": journal_left, "c7_state": op["state"] if op else None, "c7_observation": obs[0] if obs else None,
             "report_py_versions_recorded": store.conn.execute(
                 "SELECT COUNT(*) FROM artifact_versions WHERE path='report.py'").fetchone()[0],
             "events": store.counts()["events"],
             "report_py_on_disk": text_version(ws.read("report.py"))}
    return store, ws, found


def crash_experiment(out):
    rows = []
    for point in POINTS:
        d = os.path.join(out, "crash", point)
        os.makedirs(d, exist_ok=True)
        code = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", point, "--dir", d],
                              capture_output=True, text=True).returncode
        store, ws, found = inspect_after_crash(d)
        rt = Runtime(store, ws, "worker-2")
        store.advance(LEASE_TICKS + 1)            # the dead worker's lease runs out
        if store.run(RUN):
            rt.claim()
        moved = rt.restart_classification() if store.run(RUN) else []
        found.update({"point": point, "exit_code": code, "restart_marked_uncertain": moved,
                      "c7_state_after_restart": store.op("c7")["state"] if store.op("c7") else None})
        rows.append(found)
        write_run(d, "report.json", found)
    # the flawed design: no intent record before the effect
    d = os.path.join(out, "crash", "no_intent_after_effect")
    os.makedirs(d, exist_ok=True)
    code = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "after_effect", "--dir", d,
                           "--no-intent"], capture_output=True, text=True).returncode
    store, ws, found = inspect_after_crash(d)
    rt = Runtime(store, ws, "worker-2", record_intent=False)
    store.advance(LEASE_TICKS + 1)
    rt.claim()
    moved = rt.restart_classification()
    # the restarted worker sees a prepared operation and dispatches it again
    c7 = store.op("c7")
    status = rt.execute({"id": "c7", "name": c7["tool"], "arguments": json.loads(c7["arguments"])})
    obs = json.loads(store.conn.execute("SELECT result FROM observations WHERE op_id='c7'").fetchone()[0])
    found.update({"point": "after_effect (no intent record)", "exit_code": code,
                  "restart_marked_uncertain": moved, "redispatch_status": status,
                  "c7_state_after_redispatch": store.op("c7")["state"], "redispatch_message": obs.get("message"),
                  "report_py_on_disk_after": text_version(ws.read("report.py"))})
    write_run(d, "report.json", found)
    return rows, found


# =============================================================================
# Experiment 2: competing updates
# =============================================================================
def lost_update(out):
    path = os.path.join(out, "lost_update.db")
    results = {}
    for mode in ("unconditional", "conditional"):
        store = RunStore(path, create=True)
        rt = Runtime(store, DirWorkspace(os.path.join(out, "lost_update_ws")), "worker-1")
        rt.create()
        worker = connect(path)      # the worker process
        api = connect(path)         # the API process handling the person's correction
        cols = "run_id, task_revision, status, step, version"
        w = dict(zip(cols.split(", "), worker.execute(f"SELECT {cols} FROM runs").fetchone()))
        a = dict(zip(cols.split(", "), api.execute(f"SELECT {cols} FROM runs").fetchone()))
        trace = [f"worker read  step={w['step']} task_revision={w['task_revision']} version={w['version']}",
                 f"api    read  step={a['step']} task_revision={a['task_revision']} version={a['version']}"]
        api.execute("INSERT INTO task_revisions VALUES(?,?,?,?)", (RUN, 2, CORRECTION, "user"))
        if mode == "unconditional":
            n = api.execute("UPDATE runs SET task_revision=?, status=?, step=?, version=? WHERE run_id=?",
                            (2, a["status"], a["step"], a["version"] + 1, RUN)).rowcount
            trace.append(f"api    write task_revision=2 rows={n}")
            n = worker.execute("UPDATE runs SET task_revision=?, status=?, step=?, version=? WHERE run_id=?",
                               (w["task_revision"], w["status"], w["step"] + 1, w["version"] + 1, RUN)).rowcount
            trace.append(f"worker write step={w['step'] + 1} (from its copy) rows={n}")
        else:
            sql = ("UPDATE runs SET task_revision=?, status=?, step=?, version=version+1 "
                   "WHERE run_id=? AND version=?")
            n = api.execute(sql, (2, a["status"], a["step"], RUN, a["version"])).rowcount
            trace.append(f"api    write task_revision=2 if version={a['version']} rows={n}")
            n = worker.execute(sql, (w["task_revision"], w["status"], w["step"] + 1, RUN, w["version"])).rowcount
            trace.append(f"worker write step={w['step'] + 1} if version={w['version']} rows={n}")
            if n == 0:
                w = dict(zip(cols.split(", "), worker.execute(f"SELECT {cols} FROM runs").fetchone()))
                trace.append(f"worker reread step={w['step']} task_revision={w['task_revision']} "
                             f"version={w['version']}")
                n = worker.execute(sql, (w["task_revision"], w["status"], w["step"] + 1, RUN,
                                         w["version"])).rowcount
                trace.append(f"worker write step={w['step'] + 1} if version={w['version']} rows={n}")
        final = dict(zip(cols.split(", "), connect(path).execute(f"SELECT {cols} FROM runs").fetchone()))
        revisions = connect(path).execute("SELECT COUNT(*) FROM task_revisions").fetchone()[0]
        results[mode] = {"trace": trace, "final": final, "task_revision_rows": revisions}
        worker.close()
        api.close()
        store.conn.close()
    return results


def stale_worker(out):
    results = {}
    for policy in ("naive", "version", "fenced"):
        path = os.path.join(out, f"race_{policy}.db")
        ws = DirWorkspace(os.path.join(out, f"race_{policy}_ws"))
        store = RunStore(path, create=True, policy=policy)
        a = Runtime(store, ws, "worker-A")
        a.create()
        a_holder = a.holder()
        model_a = ScriptedModel([reply("", call("c11", "run_tests", tests=["test_report.py"]))])
        a.model_turn(model_a, 1)
        with store.transaction("executor", RUN, a_holder) as tx:
            tx.append("op.dispatched", "c11", {"covers": a.covers({"name": "run_tests"}), "fence": a.fence})
        a_read_version = store.op("c11")["version"]
        store.advance(LEASE_TICKS + 1)                 # worker A stalls past its lease
        b = Runtime(RunStore(path, policy=policy), ws, "worker-B")
        b.claim()
        model_b = ScriptedModel([reply("", call("c12", "run_tests", tests=["test_report.py"]))])
        b.model_turn(model_b, 2)
        b.execute({"id": "c12", "name": "run_tests", "arguments": {"tests": ["test_report.py"]}})
        ui_before = ui_latest_test(store)
        # worker A wakes up and records the outcome of its own attempt (an injected timeout)
        try:
            with store.transaction("executor", RUN, a_holder) as tx:
                tx.append("observation.recorded", "c11", {"status": "failed", "result": {
                    "status": "failed", "message": "injected: test process timed out after 60s"}})
                tx.append("op.completed", "c11", {"status": "failed"}, expect_version=a_read_version)
            a_write = "accepted"
        except Refused as r:
            a_write = "refused: " + r.reason
        run = store.run(RUN)
        results[policy] = {"lease": {"owner": run["owner"], "fence": run["fence"]},
                           "worker_A_holder": list(a_holder), "worker_A_expected_version": a_read_version,
                           "worker_A_late_write": a_write, "c11_state": store.op("c11")["state"],
                           "ui_latest_test_before": ui_before, "ui_latest_test_after": ui_latest_test(store),
                           "refusals": store.counts()["refusals"]}
    return results


def ui_latest_test(store):
    row = store.conn.execute(
        "SELECT o.op_id, b.status FROM operations o JOIN observations b ON o.op_id=b.op_id "
        "WHERE o.tool='run_tests' ORDER BY (SELECT MAX(seq) FROM events e WHERE e.entity=o.op_id) DESC "
        "LIMIT 1").fetchone()
    return f"{row[0]} {row[1]}" if row else None


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--child")
    ap.add_argument("--dir")
    ap.add_argument("--no-intent", action="store_true")
    args = ap.parse_args()
    if args.child:
        child(args.child, args.dir, record_intent=not args.no_intent)
        return
    out = args.output
    if not out:
        sys.exit("--output is required")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    store, rt, ws, main_result = main_run(out)
    recon = reconstruction(store, rt, ws, main_result["snapshot_seq"], out)
    dump_store(store, RUN, out, "main")
    crash_rows, flawed = crash_experiment(out)
    lost = lost_update(out)
    stale = stale_worker(out)

    summary = {"main_run": main_result, "reconstruction": recon, "crash": crash_rows,
               "crash_without_intent": flawed, "lost_update": lost, "stale_worker": stale}
    write_run(out, "summary.json", summary)

    p = print
    p("== main run ==")
    p("counts:", main_result["counts"])
    p("operation states:", main_result["op_states"])
    p("tool executions:", main_result["executions"])
    p("operation c8 row:", store.conn.execute(
        "SELECT op_id, tool, state, task_revision, covers, fence, version FROM operations "
        "WHERE op_id='c8'").fetchone())
    p("after call 7:", main_result["after_call_7"])
    p("after correction:", main_result["after_revision"])
    for a in main_result["artifact_versions"]:
        p("artifact:", a)
    p("\n== experiment 1: crash around op c7 (replace_text report.py) ==")
    p(f"{'point':<14}{'exit':>5} {'journal':>8}  {'c7 stored':<11}{'obs':<5}{'versions':>9}{'events':>7}  "
      f"{'report.py file':<15}{'after restart'}")
    for r in crash_rows:
        p(f"{r['point']:<14}{r['exit_code']:>5} {str(r['rollback_journal_left']):>8}  {str(r['c7_state']):<11}"
          f"{str(r['c7_observation']):<5}{r['report_py_versions_recorded']:>9}{r['events']:>7}  "
          f"{r['report_py_on_disk']:<15}{r['c7_state_after_restart']}")
    p("without intent record:", {k: flawed[k] for k in ("c7_state", "report_py_on_disk", "redispatch_status",
                                                         "c7_state_after_redispatch", "redispatch_message")})
    p("\n== experiment 2a: lost update ==")
    for mode, r in lost.items():
        p(mode)
        for line in r["trace"]:
            p("  ", line)
        p("   final:", r["final"], "task_revision rows:", r["task_revision_rows"])
    p("\n== experiment 2b: stale worker ==")
    for policy, r in stale.items():
        p(policy, {k: r[k] for k in ("lease", "worker_A_holder", "worker_A_expected_version",
                                     "worker_A_late_write", "c11_state", "ui_latest_test_before",
                                     "ui_latest_test_after")})
    p("\n== experiment 3: reconstruction ==")
    p("rebuild:", recon["rebuild"])
    p("duplicate event:", recon["duplicate_event"])
    p("re-execution replay:", recon["reexecution_replay"])
    p("schema versions:", recon["schema_versions"])
    p("\n== experiment 4: interface ==")
    p("from stream:", main_result["projection"]["ui_from_stream"])
    p("from records:", main_result["projection"]["ui_from_records"])
    p("from records, end:", main_result["projection"]["ui_from_records_final"])
    p("model inputs rebuilt:", [(r["call"], r["messages"], r["recorded_sha"] == r["rebuilt_sha"],
                                  r["task_revision_seen"]) for r in main_result["model_inputs_rebuilt"]])
    p("\n== experiment 5: ownership ==")
    for a in main_result["ownership_attempts"]:
        p(a)
    p("audit by writer:", main_result["audit_by_writer"])


if __name__ == "__main__":
    main()
