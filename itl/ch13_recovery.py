"""Chapter 13 helpers: recovery when the outcome is unknown.

Four external services, each with its own SQLite database, and a runtime
State store (runtime.db). Every service call opens and closes its own
connection, so nothing survives an injected interruption except what was
committed to a database file.

    ReadService          safe to repeat (reads a workspace file revision)
    ExportService        idempotent by contract: operation key + payload digest
    JobService           queryable by the caller's operation id (status endpoint)
    NotificationService  no idempotency, no query: a repeated send is a second message

Time is simulated (SimClock) so backoff, deadlines, key retention and job
durations are reproducible to the tenth of a second.
"""
import hashlib
import json
import os
import random
import sqlite3


class SimClock:
    def __init__(self, start=0.0):
        self.now = float(start)

    def advance(self, seconds):
        self.now = round(self.now + seconds, 3)
        return self.now


class Interrupted(Exception):
    """Injected loss of the worker process at a named point."""


class ServiceError(Exception):
    def __init__(self, status, message):
        super().__init__(f"{status} {message}")
        self.status = status
        self.message = message


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]


def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# External services
# ---------------------------------------------------------------------------

class ExportService:
    """Creates report files. Honors an operation key bound to a payload digest.

    Keys are retained for `retention_s` seconds of simulated time; after that
    the service forgets them and a repeated key creates a new export.
    """

    def __init__(self, path, clock, retention_s=86400):
        self.path, self.clock, self.retention_s = path, clock, retention_s
        with connect(path) as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS exports (
                export_id TEXT PRIMARY KEY, op_key TEXT, payload_digest TEXT,
                report_date TEXT, source_revision TEXT, destination TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                op_key TEXT PRIMARY KEY, payload_digest TEXT, export_id TEXT, first_seen REAL);
            """)
        con.close()

    def create_export(self, op_key, payload):
        d = digest(payload)
        con = connect(self.path)
        try:
            with con:
                con.execute("DELETE FROM idempotency_keys WHERE first_seen <= ?",
                            (self.clock.now - self.retention_s,))
                row = con.execute("SELECT * FROM idempotency_keys WHERE op_key = ?", (op_key,)).fetchone()
                if row is not None:
                    if row["payload_digest"] != d:
                        raise ServiceError(422, "operation key reused with a different payload")
                    exp = con.execute("SELECT * FROM exports WHERE export_id = ?", (row["export_id"],)).fetchone()
                    return {"export_id": exp["export_id"], "op_key": op_key, "payload_digest": d,
                            "created_at": exp["created_at"], "replayed": True}
                n = con.execute("SELECT COUNT(*) FROM exports").fetchone()[0] + 1
                export_id = f"export-file-{n:03d}"
                con.execute("INSERT INTO exports VALUES (?,?,?,?,?,?,?)",
                            (export_id, op_key, d, payload["report_date"], payload["source_revision"],
                             payload["destination"], self.clock.now))
                con.execute("INSERT INTO idempotency_keys VALUES (?,?,?,?)",
                            (op_key, d, export_id, self.clock.now))
                return {"export_id": export_id, "op_key": op_key, "payload_digest": d,
                        "created_at": self.clock.now, "replayed": False}
        finally:
            con.close()


class JobService:
    """Runs dashboard refresh jobs. Submission is not idempotent, but every
    job carries the caller's operation id and can be queried by it."""

    def __init__(self, path, clock, duration_s=40.0, no_return_after_s=5.0):
        self.path, self.clock = path, clock
        self.duration_s, self.no_return_after_s = duration_s, no_return_after_s
        self.unavailable_until = None
        with connect(path) as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, client_op_id TEXT, payload_json TEXT, status TEXT,
                submitted_at REAL, finish_at REAL, no_return_at REAL,
                cancel_requested_at REAL, finished_at REAL, result_json TEXT);
            """)
        con.close()

    def _refresh(self, con):
        con.execute("""UPDATE jobs SET status='completed', finished_at=finish_at,
                       result_json=json_object('dashboard','daily-orders','refreshed_at',finish_at)
                       WHERE status='running' AND finish_at <= ?""", (self.clock.now,))

    def _available(self):
        if self.unavailable_until is not None and self.clock.now < self.unavailable_until:
            raise ServiceError(503, "status endpoint unavailable")

    def submit(self, client_op_id, payload):
        con = connect(self.path)
        try:
            with con:
                n = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] + 1
                job_id = f"job-{n:03d}"
                t = self.clock.now
                con.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (job_id, client_op_id, canonical(payload), "running", t,
                             t + self.duration_s, t + self.no_return_after_s, None, None, None))
            return {"job_id": job_id, "status": "running", "poll_after_s": 10.0}
        finally:
            con.close()

    def _row(self, row):
        return {"job_id": row["job_id"], "client_op_id": row["client_op_id"], "status": row["status"],
                "cancel_requested": row["cancel_requested_at"] is not None,
                "result": json.loads(row["result_json"]) if row["result_json"] else None}

    def query_by_client_op(self, client_op_id):
        self._available()
        con = connect(self.path)
        try:
            with con:
                self._refresh(con)
                rows = con.execute("SELECT * FROM jobs WHERE client_op_id=? ORDER BY job_id",
                                   (client_op_id,)).fetchall()
            return [self._row(r) for r in rows]
        finally:
            con.close()

    def get(self, job_id):
        self._available()
        con = connect(self.path)
        try:
            with con:
                self._refresh(con)
                return self._row(con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())
        finally:
            con.close()

    def cancel(self, job_id):
        """Cooperative: acknowledged always, honored only before the job starts
        writing to the dashboard (its point of no return)."""
        con = connect(self.path)
        try:
            with con:
                self._refresh(con)
                row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                con.execute("UPDATE jobs SET cancel_requested_at=? WHERE job_id=?", (self.clock.now, job_id))
                if row["status"] == "running" and self.clock.now < row["no_return_at"]:
                    con.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE job_id=?",
                                (self.clock.now, job_id))
                row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return {"job_id": job_id, "cancel_requested": True, "status": row["status"]}
        finally:
            con.close()


class NotificationService:
    """Sends a message. No idempotency key, no way for the caller to ask
    whether a message was sent."""

    def __init__(self, path, clock):
        self.path, self.clock = path, clock
        with connect(path) as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY, recipient TEXT, subject TEXT, sent_at REAL);
            """)
        con.close()

    def send(self, recipient, subject, body):
        con = connect(self.path)
        try:
            with con:
                n = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0] + 1
                mid = f"msg-{n:03d}"
                con.execute("INSERT INTO messages VALUES (?,?,?,?)", (mid, recipient, subject, self.clock.now))
            return {"message_id": mid}
        finally:
            con.close()


def oracle_messages(path):
    """Test oracle: reads the notification provider's own table. The runtime
    and its reconciler have no such access; a person can see the provider's
    delivery log, which this stands in for."""
    con = connect(path)
    try:
        return [dict(r) for r in con.execute("SELECT * FROM messages ORDER BY message_id")]
    finally:
        con.close()


def oracle_rows(path, table):
    con = connect(path)
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
    finally:
        con.close()


class ReadService:
    def __init__(self, workspace):
        self.ws = workspace

    def read_revision(self, path):
        text = self.ws.read(path)
        return {"path": path, "version": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]}


# ---------------------------------------------------------------------------
# Tool catalog: recovery class is declared per tool, with its owner
# ---------------------------------------------------------------------------

CATALOG = {
    "version": "catalog-2026-09-25.1",
    "tools": {
        "read_revision": {"recovery_class": "safe_repeat", "owner": "runtime-team"},
        "create_export": {"recovery_class": "idempotent_key", "owner": "reporting-platform",
                          "key_retention_s": 86400, "reuse_margin_s": 3600},
        "refresh_dashboard": {"recovery_class": "queryable", "owner": "analytics-infra",
                              "status_query": "jobs.query_by_client_op"},
        "notify_oncall": {"recovery_class": "non_repeatable", "owner": "reporting-platform",
                          "resolver_role": "reports-oncall-lead"},
    },
}


# ---------------------------------------------------------------------------
# Runtime State store
# ---------------------------------------------------------------------------

class StateStore:
    def __init__(self, path, clock):
        self.path, self.clock = path, clock
        with connect(path) as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, task_revision INTEGER);
            CREATE TABLE IF NOT EXISTS operations (
                op_id TEXT PRIMARY KEY, run_id TEXT, tool TEXT, recovery_class TEXT,
                catalog_version TEXT, op_key TEXT, payload_json TEXT, payload_digest TEXT,
                basis_task_revision INTEGER, status TEXT, attempts INTEGER,
                first_dispatched_at REAL, receipt_json TEXT, cancel_requested_at REAL, note TEXT);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, op_id TEXT, kind TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS repair_queue (
                item_id TEXT PRIMARY KEY, op_id TEXT, opened_at REAL, reason TEXT, evidence TEXT,
                status TEXT, resolved_by TEXT, resolution TEXT, resolved_at REAL);
            """)
        con.close()

    def _write(self, fn):
        con = connect(self.path)
        try:
            with con:
                return fn(con)
        finally:
            con.close()

    def event(self, con, op_id, kind, detail=None):
        con.execute("INSERT INTO events (t, op_id, kind, detail) VALUES (?,?,?,?)",
                    (self.clock.now, op_id, kind, canonical(detail or {})))

    def start_run(self, run_id, task_revision):
        self._write(lambda con: con.execute("INSERT OR REPLACE INTO runs VALUES (?,?)", (run_id, task_revision)))

    def revise_task(self, run_id, task_revision):
        def fn(con):
            con.execute("UPDATE runs SET task_revision=? WHERE run_id=?", (task_revision, run_id))
            self.event(con, None, "task_revised", {"run_id": run_id, "task_revision": task_revision})
        self._write(fn)

    def task_revision(self, run_id):
        con = connect(self.path)
        try:
            return con.execute("SELECT task_revision FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
        finally:
            con.close()

    def prepare(self, op_id, run_id, tool, payload, op_key=None):
        entry = CATALOG["tools"][tool]
        rev = self.task_revision(run_id)

        def fn(con):
            con.execute("INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (op_id, run_id, tool, entry["recovery_class"], CATALOG["version"], op_key,
                         canonical(payload), digest(payload), rev, "prepared", 0, None, None, None, None))
            self.event(con, op_id, "prepared", {"tool": tool, "op_key": op_key, "payload_digest": digest(payload),
                                                "recovery_class": entry["recovery_class"]})
        self._write(fn)

    def mark_dispatched(self, op_id):
        def fn(con):
            con.execute("""UPDATE operations SET status='dispatched', attempts=attempts+1,
                           first_dispatched_at=COALESCE(first_dispatched_at, ?) WHERE op_id=?""",
                        (self.clock.now, op_id))
            self.event(con, op_id, "dispatched", {})
        self._write(fn)

    def set_status(self, op_id, status, receipt=None, note=None, kind=None, detail=None):
        def fn(con):
            if receipt is not None:
                con.execute("UPDATE operations SET status=?, receipt_json=? WHERE op_id=?",
                            (status, canonical(receipt), op_id))
            else:
                con.execute("UPDATE operations SET status=? WHERE op_id=?", (status, op_id))
            if note is not None:
                con.execute("UPDATE operations SET note=? WHERE op_id=?", (note, op_id))
            self.event(con, op_id, kind or status, detail if detail is not None else (receipt or {}))
        self._write(fn)

    def request_cancel(self, op_id):
        def fn(con):
            con.execute("UPDATE operations SET cancel_requested_at=? WHERE op_id=?", (self.clock.now, op_id))
            self.event(con, op_id, "cancel_requested", {})
        self._write(fn)

    def op(self, op_id):
        con = connect(self.path)
        try:
            r = con.execute("SELECT * FROM operations WHERE op_id=?", (op_id,)).fetchone()
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            d["receipt"] = json.loads(d.pop("receipt_json")) if d["receipt_json"] else None
            return d
        finally:
            con.close()

    def ops(self):
        con = connect(self.path)
        try:
            ids = [r[0] for r in con.execute("SELECT op_id FROM operations ORDER BY op_id")]
        finally:
            con.close()
        return [self.op(i) for i in ids]

    def events(self):
        con = connect(self.path)
        try:
            return [{**dict(r), "detail": json.loads(r["detail"])} for r in
                    con.execute("SELECT * FROM events ORDER BY seq")]
        finally:
            con.close()

    def open_repair_item(self, op_id, reason, evidence):
        def fn(con):
            n = con.execute("SELECT COUNT(*) FROM repair_queue").fetchone()[0] + 1
            item = f"R-{n}"
            con.execute("INSERT INTO repair_queue VALUES (?,?,?,?,?,?,?,?,?)",
                        (item, op_id, self.clock.now, reason, canonical(evidence), "open", None, None, None))
            con.execute("UPDATE operations SET status='uncertain', note=? WHERE op_id=?", (reason, op_id))
            self.event(con, op_id, "uncertain", {"repair_item": item, "reason": reason})
            return item
        return self._write(fn)

    def repair_items(self):
        con = connect(self.path)
        try:
            return [{**dict(r), "evidence": json.loads(r["evidence"]),
                     "resolution": json.loads(r["resolution"]) if r["resolution"] else None}
                    for r in con.execute("SELECT * FROM repair_queue ORDER BY item_id")]
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Executor: prepare -> dispatched -> call -> receipt, with injectable interruption
# ---------------------------------------------------------------------------

POINTS = ["before_dispatch", "in_transit", "after_commit", "after_receipt"]


class Executor:
    def __init__(self, store, services):
        self.store, self.s = store, services

    def call_service(self, op):
        p, tool = op["payload"], op["tool"]
        if tool == "read_revision":
            return self.s["read"].read_revision(p["path"])
        if tool == "create_export":
            return self.s["export"].create_export(op["op_key"], p)
        if tool == "refresh_dashboard":
            return self.s["jobs"].submit(op["op_id"], p)
        if tool == "notify_oncall":
            return self.s["notify"].send(p["to"], p["subject"], p["body"])
        raise ValueError(tool)

    def dispatch(self, op_id, interrupt=None):
        if interrupt == "before_dispatch":
            raise Interrupted(interrupt)
        self.store.mark_dispatched(op_id)
        if interrupt == "in_transit":
            raise Interrupted(interrupt)       # the request never reaches the service
        op = self.store.op(op_id)
        receipt = self.call_service(op)
        if interrupt == "after_commit":
            raise Interrupted(interrupt)       # the service committed; the response is lost
        self.record_receipt(op, receipt)
        if interrupt == "after_receipt":
            raise Interrupted(interrupt)
        return receipt

    def record_receipt(self, op, receipt):
        status = "accepted" if op["tool"] == "refresh_dashboard" else "completed"
        self.store.set_status(op["op_id"], status, receipt=receipt)


# ---------------------------------------------------------------------------
# Backoff with full jitter, against a deadline, in simulated time
# ---------------------------------------------------------------------------

class Backoff:
    def __init__(self, base_s=1.0, cap_s=30.0, max_attempts=8, seed=13):
        self.base_s, self.cap_s, self.max_attempts = base_s, cap_s, max_attempts
        self.rng = random.Random(seed)

    def bound(self, attempt):
        return min(self.cap_s, self.base_s * 2 ** (attempt - 1))

    def delay(self, attempt):
        return round(self.rng.uniform(0, self.bound(attempt)), 3)


# ---------------------------------------------------------------------------
# Reconciler: one decision per (recorded status, recovery class)
# ---------------------------------------------------------------------------

class Reconciler:
    def __init__(self, store, executor, services, clock, backoff=None, deadline_s=None, log=None):
        self.store, self.ex, self.s, self.clock = store, executor, services, clock
        self.backoff = backoff or Backoff()
        self.deadline_s = deadline_s
        self.log = log if log is not None else []

    def reconcile_all(self):
        decisions = []
        for op in self.store.ops():
            if op["status"] in ("completed", "withdrawn", "cancelled", "failed"):
                if op["status"] == "completed":
                    decisions.append((op["op_id"], "continue from receipt"))
                continue
            decisions.append((op["op_id"], self.reconcile(op)))
        return decisions

    def reconcile(self, op):
        cls, status, op_id = op["recovery_class"], op["status"], op["op_id"]
        if status == "prepared":
            current = self.store.task_revision(op["run_id"])
            if current != op["basis_task_revision"]:
                self.store.set_status(op_id, "withdrawn", note=f"prepared under task revision "
                                      f"{op['basis_task_revision']}, current is {current}",
                                      kind="withdrawn", detail={"basis": op["basis_task_revision"],
                                                                "current": current})
                return "withdraw stale intent"
            self.ex.dispatch(op_id)
            if op["tool"] == "refresh_dashboard":
                self.poll_job(self.store.op(op_id))
            return "revalidate and dispatch"
        if status == "accepted":
            self.poll_job(op)
            return "poll job by receipt"
        if status == "uncertain" and cls == "queryable":
            return self.query_job(op, again=True)
        if status != "dispatched":
            return f"leave {status}"
        if cls == "safe_repeat":
            receipt = self.ex.call_service(op)
            self.ex.record_receipt(op, receipt)
            return "repeat read"
        if cls == "idempotent_key":
            entry = CATALOG["tools"][op["tool"]]
            age = self.clock.now - op["first_dispatched_at"]
            if age >= entry["key_retention_s"] - entry["reuse_margin_s"]:
                self.store.open_repair_item(op_id, "operation key may have expired at the service", {
                    "op_key": op["op_key"], "payload_digest": op["payload_digest"],
                    "first_dispatched_at": op["first_dispatched_at"], "age_s": age,
                    "service_retention_s": entry["key_retention_s"]})
                return "escalate (key past retention)"
            receipt = self.s["export"].create_export(op["op_key"], op["payload"])
            self.store.set_status(op_id, "completed", receipt=receipt, kind="recovered_by_key")
            return "reuse key"
        if cls == "queryable":
            return self.query_job(op)
        if cls == "non_repeatable":
            self.store.open_repair_item(op_id, "dispatched; no receipt; service offers no idempotency and no query", {
                "tool": op["tool"], "payload_digest": op["payload_digest"],
                "recipient": op["payload"]["to"], "subject": op["payload"]["subject"],
                "first_dispatched_at": op["first_dispatched_at"], "attempts": op["attempts"]})
            return "escalate to a person"
        raise ValueError(cls)

    def _with_backoff(self, fn, label):
        attempt = 0
        while True:
            try:
                return fn()
            except ServiceError as exc:
                if exc.status != 503:
                    raise
                attempt += 1
                if attempt >= self.backoff.max_attempts:
                    self.log.append({"t": self.clock.now, "event": "attempts exhausted", "what": label})
                    return None
                bound = self.backoff.bound(attempt)
                d = self.backoff.delay(attempt)
                if self.deadline_s is not None and self.clock.now + d > self.deadline_s:
                    self.log.append({"t": self.clock.now, "event": "deadline", "what": label,
                                     "attempt": attempt, "would_wait_s": d, "deadline": self.deadline_s})
                    return None
                self.log.append({"t": self.clock.now, "event": "retry", "what": label, "attempt": attempt,
                                 "status": exc.status, "bound_s": bound, "delay_s": d})
                self.clock.advance(d)

    def query_job(self, op, again=False):
        found = self._with_backoff(lambda: self.s["jobs"].query_by_client_op(op["op_id"]), "status query")
        if found is None:
            self.store.open_repair_item(op["op_id"], "status query unavailable before deadline; re-query later",
                                        {"op_id": op["op_id"], "deadline": self.deadline_s})
            return "query status: unavailable, mark uncertain"
        if not found:
            self.store.set_status(op["op_id"], "prepared", kind="query_absent", detail={"found": 0})
            self.ex.dispatch(op["op_id"])
            self.poll_job(self.store.op(op["op_id"]))
            return "query status: absent, dispatch"
        job = found[0]
        self.store.set_status(op["op_id"], "accepted", receipt={"job_id": job["job_id"], "status": job["status"],
                                                                "via": "query"}, kind="recovered_by_query")
        job = self.poll_job(self.store.op(op["op_id"])) or job
        if again:
            def close(con):
                con.execute("""UPDATE repair_queue SET status='resolved', resolved_by='reconciler',
                               resolution=?, resolved_at=? WHERE op_id=? AND status='open'""",
                            (canonical({"finding": "job found by status query", "job": job}),
                             self.clock.now, op["op_id"]))
                self.store.event(con, op["op_id"], "repair_closed_by_query", {"job_id": job["job_id"]})
            self.store._write(close)
        return "query status" + (" again" if again else "")

    def poll_job(self, op):
        job_id = op["receipt"]["job_id"]
        while True:
            job = self._with_backoff(lambda: self.s["jobs"].get(job_id), "job status")
            if job is None:
                return None
            if job["status"] in ("completed", "cancelled", "failed"):
                self.store.set_status(op["op_id"], job["status"], receipt=job, kind="job_" + job["status"])
                return job
            self.clock.advance(10.0)


def naive_retry_all(store, executor):
    """The flawed policy: anything without a receipt is sent again."""
    out = []
    for op in store.ops():
        if op["status"] in ("completed",):
            continue
        if op["status"] == "accepted":
            out.append((op["op_id"], "leave accepted"))
            continue
        store.mark_dispatched(op["op_id"])
        receipt = executor.call_service(op)
        executor.record_receipt(op, receipt)
        out.append((op["op_id"], "resend"))
    return out


def resolve_repair_item(store, services, item_id, principal, roles, finding, evidence):
    """A person closes an uncertain operation. Only the resolver role named in
    the catalog may do it; every attempt, refused or accepted, is recorded."""
    items = {i["item_id"]: i for i in store.repair_items()}
    item = items[item_id]
    op = store.op(item["op_id"])
    required = CATALOG["tools"][op["tool"]].get("resolver_role")
    if required not in roles:
        def refuse(con):
            store.event(con, op["op_id"], "resolution_refused",
                        {"item": item_id, "principal": principal, "required_role": required})
        store._write(refuse)
        return {"accepted": False, "reason": f"{principal} lacks role {required}"}
    resolution = {"finding": finding, "evidence": evidence}
    receipt = None
    if finding == "delivered":
        receipt = {"source": "manual", "message_id": evidence["message_id"], "resolved_by": principal}
        new_status = "completed"
    elif finding == "not_delivered_resend":
        sent = services["notify"].send(op["payload"]["to"], op["payload"]["subject"], op["payload"]["body"])
        receipt = {"source": "resend authorized by " + principal, **sent}
        new_status = "completed"
    else:
        new_status = "abandoned"

    def fn(con):
        con.execute("""UPDATE repair_queue SET status='resolved', resolved_by=?, resolution=?, resolved_at=?
                       WHERE item_id=?""", (principal, canonical(resolution), store.clock.now, item_id))
        con.execute("UPDATE operations SET status=?, receipt_json=? WHERE op_id=?",
                    (new_status, canonical(receipt) if receipt else None, op["op_id"]))
        store.event(con, op["op_id"], "manual_resolution",
                    {"item": item_id, "principal": principal, "finding": finding, "evidence": evidence})
    store._write(fn)
    return {"accepted": True, "status": new_status, "receipt": receipt}


def fresh_stores(outdir, clock, workspace, retention_s=86400):
    os.makedirs(outdir, exist_ok=True)
    for name in ("runtime.db", "export.db", "jobs.db", "notify.db"):
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            os.remove(p)
    store = StateStore(os.path.join(outdir, "runtime.db"), clock)
    services = {
        "read": ReadService(workspace),
        "export": ExportService(os.path.join(outdir, "export.db"), clock, retention_s=retention_s),
        "jobs": JobService(os.path.join(outdir, "jobs.db"), clock),
        "notify": NotificationService(os.path.join(outdir, "notify.db"), clock),
    }
    return store, services


def restart(outdir, clock, workspace, retention_s=86400, jobs_unavailable_until=None):
    """A new process: new objects, new connections, only the database files persist."""
    store = StateStore(os.path.join(outdir, "runtime.db"), clock)
    services = {
        "read": ReadService(workspace),
        "export": ExportService(os.path.join(outdir, "export.db"), clock, retention_s=retention_s),
        "jobs": JobService(os.path.join(outdir, "jobs.db"), clock),
        "notify": NotificationService(os.path.join(outdir, "notify.db"), clock),
    }
    services["jobs"].unavailable_until = jobs_unavailable_until
    return store, services
