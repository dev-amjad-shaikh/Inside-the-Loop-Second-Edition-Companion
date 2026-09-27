"""Chapter 14 helpers: working with a person during a run.

TaskState        task revisions recorded as events (a correction is an event)
EventLog         one ordered account of model calls, corrections and dispatches
RevisionLoop     the Chapter 3 loop plus a task-revision check at dispatch
EvidenceBook     test and signature evidence tied to artifact version and task revision
ApprovalStore    sqlite-backed approval requests, decisions and bound approvals
ExportService    writes report exports to destination directories (the real effect)

Standard library only. Model behavior comes from scripted responses.
"""
import copy
import difflib
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

from .core import ScriptedModel, content_version

EPOCH = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:16]


class Clock:
    """Logical clock in seconds after EPOCH, so every timestamp is reproducible."""

    def __init__(self):
        self.s = 0

    def tick(self, n=1):
        self.s += n
        return self.s

    def iso(self, s=None):
        return (EPOCH + timedelta(seconds=self.s if s is None else s)).strftime("%H:%M:%S")


class EventLog:
    def __init__(self, clock):
        self.clock = clock
        self.events = []

    def add(self, kind, advance=1, **data):
        self.clock.tick(advance)
        ev = {"seq": len(self.events) + 1, "t": self.clock.iso(), "kind": kind}
        ev.update(data)
        self.events.append(ev)
        return ev


# --------------------------------------------------------------------------
# Task revisions
# --------------------------------------------------------------------------

class TaskState:
    """The task and its revisions. Control owns it; the Human interface appends to it."""

    def __init__(self, text, log):
        self.log = log
        self.revisions = []
        self.revise(text, checks=[], source="initial request")

    @property
    def current(self):
        return self.revisions[-1]["revision"]

    def active_checks(self):
        out = []
        for r in self.revisions:
            out.extend(r["checks"])
        return out

    def revise(self, text, checks, source="user message"):
        rev = len(self.revisions) + 1
        ev = self.log.add("task_revised", advance=0 if not self.revisions else 1,
                          task_revision=rev, text=text, checks=checks, source=source)
        self.revisions.append({"revision": rev, "text": text, "checks": checks, "seq": ev["seq"]})
        return rev


# --------------------------------------------------------------------------
# Signature check: a routine constraint turned into an executable check
# --------------------------------------------------------------------------

SIG_PROBE = ("import inspect, report; "
             "print(inspect.signature(report.belongs_to_report))")


def read_signature(root):
    proc = subprocess.run([sys.executable, "-c", SIG_PROBE], cwd=root,
                          capture_output=True, text=True, timeout=30)
    return proc.stdout.strip()


def add_signature_tool(reg, ws, baseline):
    def check_signature():
        sig = read_signature(ws.root)
        return {"status": "ok", "function": "belongs_to_report", "signature": sig,
                "baseline": baseline, "matches_baseline": sig == baseline,
                "version": content_version(ws.read("report.py"))}

    reg.register("check_signature",
                 "Compare the public signature of belongs_to_report with the run's recorded baseline.",
                 {"type": "object", "properties": {}, "additionalProperties": False},
                 check_signature)


# --------------------------------------------------------------------------
# Evidence tied to artifact version and task revision
# --------------------------------------------------------------------------

class EvidenceBook:
    def __init__(self, log):
        self.log = log
        self.items = []

    def record(self, kind, observation, version, task_revision, call_id):
        eid = f"E{len(self.items) + 1}"
        item = {"id": eid, "kind": kind, "call_id": call_id, "artifact": "report.py",
                "version": version, "task_revision": task_revision, "state": "current"}
        if kind == "tests":
            item["result"] = observation["status"]
            item["tests_run"] = observation["tests_run"]
        else:
            item["result"] = "match" if observation["matches_baseline"] else "mismatch"
            item["signature"] = observation["signature"]
        self.items.append(item)
        self.log.add("evidence_recorded", advance=0, evidence=eid, evidence_kind=kind,
                     result=item["result"], version=version, task_revision=task_revision)
        return item

    def artifact_changed(self, new_version):
        for item in self.items:
            if item["state"] == "current" and item["version"] != new_version:
                item["state"] = "historical"
                item["superseded_by_version"] = new_version
                self.log.add("evidence_historical", advance=0, evidence=item["id"],
                             was_version=item["version"], now_version=new_version)

    def current(self, kind, version):
        found = [i for i in self.items if i["kind"] == kind and i["version"] == version
                 and i["state"] == "current"]
        return found[-1] if found else None


def flawed_verifier(messages):
    """Chapter 3's verifier: a pass recorded after the latest edit means complete."""
    last_edit = last_pass = -1
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and isinstance(m.get("content"), dict):
            c = m["content"]
            if "after_version" in c:
                last_edit = i
            if c.get("status") == "passed":
                last_pass = i
    ok = last_pass > last_edit >= 0
    return {"complete": ok, "reason": "latest test result passed after latest edit" if ok
            else "no pass after latest edit"}


def revision_verifier(task, evidence, ws):
    """Complete only if every check required by the current task revision holds on the
    current artifact version."""
    version = content_version(ws.read("report.py"))
    missing = []
    tests = evidence.current("tests", version)
    if not tests or tests["result"] != "passed":
        missing.append(f"passing tests on report.py {version}")
    if "signature:belongs_to_report" in task.active_checks():
        sig = evidence.current("signature", version)
        if not sig or sig["result"] != "match":
            missing.append(f"signature match on report.py {version}")
    return {"complete": not missing, "task_revision": task.current, "version": version,
            "missing": missing,
            "supporting": [i["id"] for i in evidence.items
                           if i["state"] == "current" and i["version"] == version]}


# --------------------------------------------------------------------------
# The loop with a revision check at dispatch
# --------------------------------------------------------------------------

EFFECTFUL = {"replace_text", "write_file", "export_report", "deploy"}


class CorrectionModel(ScriptedModel):
    """Scripted model whose generation can be overlapped by outside events.

    during[k] runs after the k-th request is captured and before its response
    is returned: the moment a person types while the model is generating.
    """

    def __init__(self, responses, during=None):
        super().__init__(responses)
        self.during = during or {}

    def generate(self, messages, definitions):
        reply = super().generate(messages, definitions)
        hook = self.during.get(len(self.requests))
        if hook:
            hook()
        return reply


def run_with_revisions(model, tools, ws, task, log, evidence, system, limit=16,
                       check_revision=True):
    definitions = tools.definitions()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task.revisions[0]["text"]}]
    delivered = 1
    verdicts = []
    for call_no in range(1, limit + 1):
        # Context builder: every task revision not yet delivered becomes model input.
        for r in task.revisions[delivered:]:
            messages.append({"role": "user",
                             "content": f"Task revision {r['revision']}: {r['text']}"})
        delivered = len(task.revisions)
        based_on = task.current
        log.add("model_call_started", call=call_no, task_revision=based_on)
        reply = model.generate(messages, definitions)
        reply = dict(reply)
        reply["task_revision"] = based_on
        calls = reply.get("tool_calls") or []
        log.add("model_call_finished", advance=6, call=call_no, task_revision=based_on,
                proposal=[f"{c['id']} {c['name']}" for c in calls] or ["final answer"])
        messages.append(reply)
        if not calls:
            flawed = flawed_verifier(messages)
            proper = revision_verifier(task, evidence, ws)
            stale = check_revision and based_on != task.current
            verdicts.append({"call": call_no, "proposal_task_revision": based_on,
                             "current_task_revision": task.current,
                             "flawed_verifier": flawed, "revision_verifier": proper,
                             "accepted": (not stale) and proper["complete"]})
            if stale:
                log.add("completion_refused", call=call_no, reason="stale_task_revision",
                        proposal_task_revision=based_on, current_task_revision=task.current)
                messages.append({"role": "system", "content":
                                 f"Runtime: completion proposed under task revision {based_on} "
                                 f"was not accepted; the current revision is {task.current}."})
                continue
            if not proper["complete"]:
                log.add("completion_refused", call=call_no, reason="verification",
                        missing=proper["missing"])
                messages.append({"role": "system", "content":
                                 "Runtime: not complete. Missing: " + "; ".join(proper["missing"])})
                continue
            log.add("run_completed", call=call_no, task_revision=task.current,
                    version=proper["version"], supporting=proper["supporting"])
            return {"answer": reply["content"], "messages": messages, "verdicts": verdicts}
        for c in calls:
            if check_revision and c["name"] in EFFECTFUL and based_on != task.current:
                obs = {"status": "not_dispatched", "reason": "stale_task_revision",
                       "proposal_task_revision": based_on,
                       "current_task_revision": task.current,
                       "message": "Proposed under an earlier task revision; regenerate "
                                  "against the current task."}
                log.add("dispatch_refused", call_id=c["id"], tool=c["name"],
                        reason="stale_task_revision", proposal_task_revision=based_on,
                        current_task_revision=task.current)
            else:
                log.add("dispatched", call_id=c["id"], tool=c["name"], task_revision=based_on)
                obs = tools.execute_checked(c)
                ev = {"call_id": c["id"], "status": obs.get("status")}
                if "after_version" in obs:
                    ev.update(path=obs["path"], before_version=obs["before_version"],
                              after_version=obs["after_version"])
                log.add("observed", advance=2, **ev)
                if obs.get("path") == "report.py" and "after_version" in obs:
                    evidence.artifact_changed(obs["after_version"])
                version = content_version(ws.read("report.py"))
                if c["name"] == "run_tests" and obs.get("status") in ("passed", "failed"):
                    evidence.record("tests", obs, version, task.current, c["id"])
                if c["name"] == "check_signature":
                    evidence.record("signature", obs, version, task.current, c["id"])
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
    raise RuntimeError("Model-call limit reached")


# --------------------------------------------------------------------------
# Export service and approvals
# --------------------------------------------------------------------------

class ArtifactStore:
    """Immutable report revisions, identified by name and revision label."""

    def __init__(self, fixture_dir):
        self.revisions = {}
        for name in sorted(os.listdir(fixture_dir)):
            if name.endswith(".csv"):
                report, rev, _ = name.rsplit(".", 2)
                with open(os.path.join(fixture_dir, name), encoding="utf-8") as f:
                    text = f.read()
                self.revisions[(report, rev)] = text
        self.latest = {}

    def publish(self, report, rev):
        self.latest[report] = rev

    def text(self, report, rev):
        return self.revisions[(report, rev)]

    def version(self, report, rev):
        return content_version(self.text(report, rev))

    def diff(self, report, a, b):
        return "".join(difflib.unified_diff(
            self.text(report, a).splitlines(True), self.text(report, b).splitlines(True),
            fromfile=f"{report} {a}", tofile=f"{report} {b}", n=0))


class ExportService:
    """Writes one file per export into outbox/<destination>/ and returns a receipt."""

    def __init__(self, store, root):
        self.store = store
        self.root = root
        self.receipts = []

    def export(self, report, revision, destination):
        text = self.store.text(report, revision)
        rid = f"X{len(self.receipts) + 1}"
        d = os.path.join(self.root, destination)
        os.makedirs(d, exist_ok=True)
        fname = f"{report}.{revision}.{rid}.csv"
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            f.write(text)
        receipt = {"receipt": rid, "report": report, "revision": revision,
                   "destination": destination, "version": content_version(text), "file": fname}
        self.receipts.append(receipt)
        return receipt

    def listing(self):
        out = {}
        for dest in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, dest)):
                continue
            out[dest] = sorted(os.listdir(os.path.join(self.root, dest)))
        return out


ROLES = {"maria": "report_owner", "dev": "engineer", "sam": "analyst"}
MAY_APPROVE = {"export_report": {"report_owner"}}

SCHEMA = """
CREATE TABLE requests (
  request_id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, run_id TEXT,
  operation TEXT, args_json TEXT, args_digest TEXT, artifact TEXT,
  artifact_version TEXT, presented_json TEXT, status TEXT, created_t TEXT);
CREATE TABLE decisions (
  decision_id TEXT PRIMARY KEY, request_id TEXT, submission_key TEXT UNIQUE NOT NULL,
  actor TEXT, role TEXT, delegated_by TEXT, choice TEXT, decided_t TEXT);
CREATE TABLE approvals (
  approval_id TEXT PRIMARY KEY, request_id TEXT, actor TEXT, role TEXT,
  delegated_by TEXT, operation TEXT, args_digest TEXT, artifact TEXT,
  artifact_version TEXT, scope TEXT, expires_s INTEGER, status TEXT, consumed_by TEXT);
CREATE TABLE delegations (
  delegate TEXT, delegator TEXT, operation TEXT, destination TEXT, expires_s INTEGER);
CREATE TABLE audit (seq INTEGER PRIMARY KEY, t TEXT, kind TEXT, detail TEXT);
"""


class ApprovalStore:
    def __init__(self, path, clock, run_id="run-14"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.clock = clock
        self.run_id = run_id

    def audit(self, kind, **detail):
        self.clock.tick()
        self.db.execute("INSERT INTO audit (t, kind, detail) VALUES (?, ?, ?)",
                        (self.clock.iso(), kind, canonical(detail)))
        self.db.commit()

    def binding(self, operation, args, artifact_version):
        return {"operation": operation, "args_digest": digest({"op": operation, "args": args}),
                "artifact": f"{args['report']}", "artifact_version": artifact_version}

    def request(self, operation, args, artifact_version, presented):
        """Create or return the pending request for exactly this operation."""
        b = self.binding(operation, args, artifact_version)
        key = digest({"run": self.run_id, **b})
        row = self.db.execute("SELECT request_id, status FROM requests WHERE request_key=?",
                              (key,)).fetchone()
        if row:
            self.audit("request_reused", request_id=row[0], request_key=key)
            return row[0], False
        n = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] + 1
        rid = f"RQ{n}"
        self.db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, key, self.run_id, operation, canonical(args), b["args_digest"],
                         b["artifact"], artifact_version, canonical(presented), "pending",
                         self.clock.iso()))
        self.db.commit()
        self.audit("request_created", request_id=rid, **b)
        return rid, True

    def delegate(self, delegate, delegator, operation, destination, ttl):
        self.db.execute("INSERT INTO delegations VALUES (?,?,?,?,?)",
                        (delegate, delegator, operation, destination, self.clock.s + ttl))
        self.db.commit()
        self.audit("delegation_granted", delegate=delegate, delegator=delegator,
                   operation=operation, destination=destination, expires=self.clock.iso(self.clock.s + ttl))

    def decide(self, request_id, actor, choice, submission_key, ttl=900):
        """Record a person's decision. Idempotent on submission_key."""
        prior = self.db.execute("SELECT decision_id, choice FROM decisions WHERE submission_key=?",
                                (submission_key,)).fetchone()
        if prior:
            self.audit("decision_resubmitted", submission_key=submission_key, decision_id=prior[0])
            return {"status": "duplicate_submission", "decision_id": prior[0]}
        req = self.db.execute("SELECT operation, args_json, args_digest, artifact, artifact_version, "
                              "status FROM requests WHERE request_id=?", (request_id,)).fetchone()
        if req is None:
            return {"status": "unknown_request"}
        operation, args_json, args_digest, artifact, version, status = req
        if status != "pending":
            self.audit("decision_refused", request_id=request_id, actor=actor,
                       reason=f"request is {status}")
            return {"status": "refused", "reason": f"request is {status}"}
        args = json.loads(args_json)
        role = ROLES.get(actor, "unknown")
        delegated_by = None
        if role not in MAY_APPROVE.get(operation, set()):
            d = self.db.execute("SELECT delegator, destination, expires_s FROM delegations "
                                "WHERE delegate=? AND operation=?", (actor, operation)).fetchone()
            if d and d[1] == args.get("destination") and d[2] > self.clock.s:
                delegated_by = d[0]
            else:
                why = (f"role {role} may not approve {operation}" if not d else
                       f"delegation from {d[0]} covers destination {d[1]}, not {args.get('destination')}")
                self.audit("decision_refused", request_id=request_id, actor=actor, reason=why)
                return {"status": "refused", "reason": why}
        n = self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] + 1
        did = f"D{n}"
        self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                        (did, request_id, submission_key, actor, role, delegated_by, choice,
                         self.clock.iso()))
        self.db.execute("UPDATE requests SET status=? WHERE request_id=?",
                        ("decided", request_id))
        out = {"status": "recorded", "decision_id": did, "choice": choice}
        if choice == "approve":
            m = self.db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] + 1
            aid = f"AP{m}"
            scope = f"{self.run_id}; destination {args.get('destination')}; one use"
            self.db.execute("INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (aid, request_id, actor, role, delegated_by, operation, args_digest,
                             artifact, version, scope, self.clock.s + ttl, "active", None))
            out["approval_id"] = aid
        self.db.commit()
        self.audit("decision_recorded", request_id=request_id, actor=actor, role=role,
                   delegated_by=delegated_by, choice=choice, submission_key=submission_key,
                   approval_id=out.get("approval_id"))
        return out

    def approval_row(self, aid):
        cols = ["approval_id", "request_id", "actor", "role", "delegated_by", "operation",
                "args_digest", "artifact", "artifact_version", "scope", "expires_s", "status",
                "consumed_by"]
        row = self.db.execute("SELECT * FROM approvals WHERE approval_id=?", (aid,)).fetchone()
        rec = dict(zip(cols, row))
        rec["expires"] = self.clock.iso(rec.pop("expires_s"))
        return rec

    def check_at_dispatch(self, call_id, operation, args, current_version):
        """Find an active approval for exactly this operation on this artifact version."""
        want = self.binding(operation, args, current_version)
        rows = self.db.execute("SELECT approval_id, args_digest, artifact_version, expires_s, status "
                               "FROM approvals WHERE operation=? AND artifact=? AND status='active'",
                               (operation, want["artifact"])).fetchall()
        if not rows:
            self.audit("dispatch_refused", call_id=call_id, operation=operation,
                       args_digest=want["args_digest"], mismatches=[], reason="no active approval")
            return {"allowed": False, "reason": "no active approval"}
        mismatches = []
        for aid, adig, aver, exp, status in rows:
            problems = []
            if adig != want["args_digest"]:
                problems.append("arguments differ")
            if aver != current_version:
                problems.append(f"artifact version {aver} approved, {current_version} proposed")
            if exp <= self.clock.s:
                problems.append("approval expired")
            if not problems:
                self.db.execute("UPDATE approvals SET status='consumed', consumed_by=? "
                                "WHERE approval_id=?", (call_id, aid))
                self.db.commit()
                self.audit("approval_consumed", approval_id=aid, call_id=call_id)
                return {"allowed": True, "approval_id": aid}
            mismatches.append({"approval_id": aid, "problems": problems})
            new_status = "expired" if problems == ["approval expired"] else "invalidated"
            self.db.execute("UPDATE approvals SET status=? WHERE approval_id=?", (new_status, aid))
        self.db.commit()
        self.audit("dispatch_refused", call_id=call_id, operation=operation,
                   args_digest=want["args_digest"], mismatches=mismatches)
        return {"allowed": False, "reason": "approval does not cover this operation",
                "mismatches": mismatches}

    def table(self, name):
        cur = self.db.execute(f"SELECT * FROM {name}")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


class LooseApprovals:
    """The flawed design: an approval names a tool and destination, nothing else."""

    def __init__(self):
        self.grants = []

    def approve(self, actor, operation, destination):
        self.grants.append({"actor": actor, "operation": operation, "destination": destination,
                            "uses": 0})

    def check(self, operation, args):
        for g in self.grants:
            if g["operation"] == operation and g["destination"] == args["destination"]:
                g["uses"] += 1
                return {"allowed": True, "grant": g}
        return {"allowed": False}


def tmpdir(prefix):
    return tempfile.mkdtemp(prefix=prefix)


def rmtree(path):
    shutil.rmtree(path, ignore_errors=True)


def jcopy(x):
    return copy.deepcopy(x)
