"""Chapter 24: the integration layer for the capstone run.

The capstone composes the Part II helper modules into one runtime. Where two
modules already agree on an interface they are called directly. Where they do
not, this module holds the thin adapter, and each adapter says what it changed:

  authenticate / CapstoneEnv   trusted identity and the environment shim that
                               lets Chapter 16's Policy and evaluate() govern a
                               reporting workspace instead of a support one
  ContextRecords               Chapter 6's RecordStore plus retrieved passages
                               and memory records as context items with provenance
  render_stream                turns an authored reply into a block-format SSE
                               stream so that every main-agent call passes
                               through Chapter 5's assemble()
  StoreBridge                  Chapter 13's StateStore interface implemented over
                               Chapter 12's RunStore, so the Chapter 13 executor
                               and reconciler work on the one authoritative store
  DeliveringExportService      Chapter 13's keyed export service, delivering the
                               Chapter 14 report revision as a file
  Capstone                     the integrated runtime: intake, context, adapter,
                               policy, skills, executor, control, delegation,
                               approvals, recovery, verification and the record

Model behavior is scripted (the scenario supplies authored replies); the person's
answers are scripted; reviewer workers are Chapter 15's scripted stand-ins. File
reads, edits, test runs, boundary cases, signature inspection, the probe script,
SQLite writes, the export and the delivered file are real.
"""
import ast
import copy
import json
import math
import os
import shutil
import subprocess
import sys

from .core import FIXTURES, check_args, InputError, content_version, report_tools
from . import ch05_adapter as adapter
from .ch05_fixtures import cbd, cbe, cbs, mdelta, msg_start, mstop, t as at_ms
from .ch06_context import RecordStore, StructuredBuilder, approx_tokens
from .ch07_retrieval import Index, assemble as assemble_evidence, ingest, open_index
from .ch08_memory import MemoryStore, retrieve as memory_retrieve
from .ch10_skills import (LexicalActivator, Registry, RunPolicy, SkillExecutor, discover,
                          materialize, render_catalog, skill_tool_definitions)
from .ch11_control import REPAIR_ALLOWANCE, Evidence as PlanEvidence, Plan, baseline_tests, workspace_semantic
from .ch12_state import DirWorkspace, Refused, RunStore, fold, read_events, state_from_tables
from .ch13_recovery import (CATALOG as RECOVERY_CATALOG, ExportService as KeyedExportService,
                            Executor as EffectExecutor, Interrupted, Reconciler, SimClock,
                            digest as op_digest, naive_retry_all)
from .ch14_human import (ApprovalStore, ArtifactStore, Clock, EventLog, EvidenceBook,
                         LooseApprovals, TaskState, flawed_verifier, revision_verifier)
from .ch15_delegation import (BOUNDARY_CASES, EvidenceLog, RevisionStore, integrate, make_contract,
                              run_worker, script_iface, script_tz)
from .ch16_policy import Policy, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
FIX24 = os.path.join(FIXTURES, "ch24_capstone")
EXPORT_FIXTURE = os.path.join(FIXTURES, "ch14_export")
SKILL_VERSIONS = os.path.join(FIXTURES, "ch10_skills", "versions")
STREAM_META = os.path.join(FIXTURES, "ch05_streams", "meta.json")

RUN_ID = "run-24"
REPORT = "daily-2026-09-11"
REQUESTED_MODEL = "coder-large"
REPORTED_MODEL = "coder-large-2026-08-14"

# Trusted context. The authentication layer maps a session token to a principal;
# nothing the client or the model sends can change these records.
SESSIONS = {
    "sess-acme-dev": {"user": "dev", "tenant": "acme", "roles": ["engineer"]},
    "sess-acme-maria": {"user": "maria", "tenant": "acme", "roles": ["report_owner"]},
}

# Tools that belong to the runtime rather than to any skill. Chapter 10's
# SkillExecutor hard-codes activate_skill as its only runtime tool; the
# integration layer names the full set here, in one place.
RUNTIME_TOOLS = {"activate_skill", "request_input"}
EFFECTFUL = {"replace_text", "export_report"}

SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: settings.json, report.py, test_report.py.")


def authenticate(session_token):
    rec = SESSIONS.get(session_token)
    if rec is None:
        raise PermissionError("unknown session")
    return dict(rec, session=session_token)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CapstoneEnv:
    """What Chapter 16's Policy.grant and evaluate() need from an environment."""

    def __init__(self, root, workspace_root):
        self.root = os.path.abspath(root)
        self._ws = os.path.abspath(workspace_root)
        self.orders = {}

    def workspace(self, tenant):
        return self._ws

    def rel(self, path):
        return os.path.relpath(path, self.root)


# ---------------------------------------------------------------------------
# Context: Chapter 6's record store with context items from Retrieval and Memory
# ---------------------------------------------------------------------------

class ContextRecords(RecordStore):
    def add_context_item(self, kind, source, revision, trust, text, **extra):
        return self._add(kind=kind, trust=trust, source=source, revision=revision,
                         message={"role": "user", "content": text}, **extra)


# ---------------------------------------------------------------------------
# Adapter: an authored reply rendered as a block-format stream (Chapter 5)
# ---------------------------------------------------------------------------

def _fragments(s, n=3):
    step = max(1, math.ceil(len(s) / n))
    return [s[i:i + step] for i in range(0, len(s), step)]


def render_stream(reply, message_id, input_tokens, cache_read):
    """Block-format SSE for one authored reply. Times are authored, not measured."""
    p = [at_ms(0), at_ms(140), msg_start(message_id, REPORTED_MODEL, input_tokens, cache_read)]
    ms, i = 160, 0
    if reply.get("content"):
        p += [at_ms(ms), cbs(i, {"type": "text", "text": ""})]
        words = reply["content"].split(" ")
        for k in range(0, len(words), 9):
            ms += 40
            chunk = " ".join(words[k:k + 9]) + (" " if k + 9 < len(words) else "")
            p += [at_ms(ms), cbd(i, {"type": "text_delta", "text": chunk})]
        p += [cbe(i)]
        i += 1
    for call in reply.get("tool_calls") or []:
        ms += 30
        p += [at_ms(ms), cbs(i, {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}})]
        for frag in _fragments(json.dumps(call["arguments"])):
            ms += 25
            p += [at_ms(ms), cbd(i, {"type": "input_json_delta", "partial_json": frag})]
        p += [cbe(i)]
        i += 1
    ms += 20
    stop = "tool_use" if reply.get("tool_calls") else "end_turn"
    p += [at_ms(ms), mdelta(stop, approx_tokens(reply)), mstop()]
    return "".join(p)


def stream_meta(fixture, request_id):
    with open(STREAM_META, encoding="utf-8") as f:
        base = json.load(f)
    return {"fixtures": {fixture: {"format": "block", "transport_end": "closed",
                                   "request_id": request_id}},
            "adapters": {"block": base["adapters"]["block"]}}


# ---------------------------------------------------------------------------
# Recovery: Chapter 13's StateStore interface over Chapter 12's RunStore
# ---------------------------------------------------------------------------

C12_TO_C13 = {"prepared": "prepared", "awaiting_approval": "prepared", "dispatched": "dispatched",
              "completed": "completed", "uncertain": "uncertain", "failed_before_effect": "failed",
              "cancelled": "withdrawn", "skipped": "withdrawn"}


class StoreBridge:
    """The methods Chapter 13's Executor and Reconciler call, backed by the run store.

    Ownership follows Chapter 12: the executor writes dispatch, observation and
    completion (with the current fence); the reconciler writes only op.uncertain.
    When the reconciler recovers a receipt it calls set_status, and the bridge
    records that completion under the executor's name and fence, with the
    reconciler's decision in the observation. With swallow_refusals=True the
    bridge behaves like careless glue: it ignores a refusal from the store.
    """

    def __init__(self, runstore, clock, holder, swallow_refusals=False):
        self.rs, self.clock, self.holder = runstore, clock, holder
        self.swallow = swallow_refusals
        self.swallowed = []
        self.attempts = {}
        self.first_dispatch = {}
        self.decisions = []

    def _payload(self, op_id, etype):
        row = self.rs.conn.execute("SELECT payload FROM events WHERE entity=? AND type=? ORDER BY seq",
                                   (op_id, etype)).fetchall()
        return [json.loads(r[0]) for r in row]

    def _tx(self, writer, fenced=True):
        return self.rs.transaction(writer, RUN_ID, self.holder() if fenced else None)

    def task_revision(self, run_id):
        return self.rs.run(run_id)["task_revision"]

    def mark_dispatched(self, op_id):
        n = self.attempts.get(op_id, 0) + 1
        try:
            with self._tx("executor") as tx:
                prepared = self._payload(op_id, "op.prepared")[0]
                tx.append("op.dispatched", op_id, {"covers": prepared.get("covers", {}),
                                                    "fence": self.holder()[1], "at": self.clock.now,
                                                    "attempt": n},
                          event_id=f"{op_id}:dispatched:{n}")
        except Refused as exc:
            if not self.swallow:
                raise
            self.swallowed.append({"op": op_id, "write": "op.dispatched", "reason": exc.reason})
        self.attempts[op_id] = n
        self.first_dispatch.setdefault(op_id, self.clock.now)

    def op(self, op_id):
        row = self.rs.op(op_id)
        prepared = self._payload(op_id, "op.prepared")[0]
        dispatched = self._payload(op_id, "op.dispatched")
        obs = self.rs.conn.execute("SELECT status, result FROM observations WHERE op_id=?",
                                   (op_id,)).fetchone()
        key = prepared["op_key"]
        return {"op_id": op_id, "run_id": row["run_id"], "tool": prepared["service_operation"],
                "recovery_class": prepared["recovery_class"], "op_key": key,
                "payload": prepared["payload"], "payload_digest": prepared["payload_digest"],
                "basis_task_revision": row["task_revision"], "status": C12_TO_C13[row["state"]],
                "attempts": self.attempts.get(op_id, len(dispatched)),
                "first_dispatched_at": dispatched[0]["at"] if dispatched else None,
                "receipt": json.loads(obs[1]) if obs else None}

    def ops(self):
        ids = [r[0] for r in self.rs.conn.execute(
            "SELECT entity FROM events WHERE type='op.prepared' ORDER BY seq")]
        return [self.op(i) for i in ids if "op_key" in self._payload(i, "op.prepared")[0]]

    def set_status(self, op_id, status, receipt=None, note=None, kind=None, detail=None):
        via = kind or "receipt"
        self.decisions.append({"op": op_id, "status": status, "via": via, "t": self.clock.now})
        if status == "completed":
            try:
                with self._tx("executor") as tx:
                    tx.append("observation.recorded", op_id,
                              {"status": "completed", "result": dict(receipt or {}, recorded_via=via)},
                              event_id=f"{op_id}:observation")
                    tx.append("op.completed", op_id, {"via": via}, event_id=f"{op_id}:completed")
            except Refused as exc:
                if not self.swallow:
                    raise
                self.swallowed.append({"op": op_id, "write": "op.completed", "reason": exc.reason})
        elif status == "uncertain":
            with self._tx("reconciler", fenced=False) as tx:
                tx.append("op.uncertain", op_id, {"note": note}, event_id=f"{op_id}:uncertain")
        else:
            raise ValueError(f"status {status} is not written through the bridge")

    def open_repair_item(self, op_id, reason, evidence):
        self.set_status(op_id, "uncertain", note=reason)
        return "R-1"

    def event(self, con, op_id, kind, detail=None):
        self.decisions.append({"op": op_id, "event": kind, "detail": detail})


class DeliveringExportService(KeyedExportService):
    """Chapter 13's export service (operation key bound to a payload digest),
    delivering the Chapter 14 report revision as a file in outbox/<destination>/."""

    def __init__(self, path, clock, artifacts, outbox, retention_s=86400):
        super().__init__(path, clock, retention_s=retention_s)
        self.artifacts, self.outbox = artifacts, outbox

    def create_export(self, op_key, payload):
        receipt = super().create_export(op_key, payload)
        text = self.artifacts.text(payload["report"], payload["revision"])
        name = f"{payload['report']}.{payload['revision']}.{receipt['export_id']}.csv"
        d = os.path.join(self.outbox, payload["destination"])
        os.makedirs(d, exist_ok=True)
        if not receipt["replayed"]:
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
        return dict(receipt, file=f"{payload['destination']}/{name}", file_version=content_version(text))


def export_rows(path):
    import sqlite3
    con = sqlite3.connect(path)
    try:
        cur = con.execute("SELECT export_id, op_key, payload_digest, destination, source_revision "
                          "FROM exports ORDER BY export_id")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def outbox_listing(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for dest in sorted(os.listdir(root)):
        files = sorted(os.listdir(os.path.join(root, dest)))
        out[dest] = [{"file": f, "version": content_version(open(os.path.join(root, dest, f),
                                                                 encoding="utf-8").read())}
                     for f in files]
    return out


# ---------------------------------------------------------------------------
# Report rows checked against the source revision (the export's content evidence)
# ---------------------------------------------------------------------------

ROW_PROBE = r'''
import csv, json, sys
from datetime import date, datetime
sys.path.insert(0, ".")
from report import belongs_to_report
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["order_id"] != "total"]
out = [{"order_id": r["order_id"], "belongs": bool(belongs_to_report(
    datetime.fromisoformat(r["created_at_utc"]), date.fromisoformat(sys.argv[2]), sys.argv[3]))} for r in rows]
print(json.dumps(out))
'''


def rows_against_source(source_dir, csv_path, day, tz):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", ROW_PROBE, csv_path, day, tz], cwd=source_dir,
                          capture_output=True, text=True, timeout=60, env=env)
    return json.loads(proc.stdout)


def signature_of(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "belongs_to_report":
            return "belongs_to_report(" + ", ".join(a.arg for a in node.args.args) + ")"
    return None


# ---------------------------------------------------------------------------
# The integrated runtime
# ---------------------------------------------------------------------------

EXPORT_DEF = {"name": "export_report",
              "description": "Export one published revision of a report to a registered destination. "
                             "Changes external state: recipients can read the file immediately. "
                             "The tenant comes from the signed-in session.",
              "parameters": {"type": "object", "additionalProperties": False,
                             "required": ["report", "revision", "destination"],
                             "properties": {"report": {"type": "string"},
                                            "revision": {"type": "string"},
                                            "destination": {"type": "string"}}}}
REQUEST_INPUT_DEF = {"name": "request_input",
                     "description": "Ask the user for information the task needs and the workspace lacks. "
                                    "The run waits for the answer.",
                     "parameters": {"type": "object", "additionalProperties": False,
                                    "required": ["question", "choices"],
                                    "properties": {"question": {"type": "string"},
                                                   "choices": {"type": "array", "items": {"type": "string"}}}}}


class Capstone:
    """One run of the reporting task as a service, through every Part II component."""

    def __init__(self, outdir, scenario, wiring="revised"):
        self.out = os.path.abspath(outdir)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)
        os.makedirs(self.out)
        self.sc = scenario
        self.wiring = wiring
        self.clock = Clock()
        self.log = EventLog(self.clock)
        self.sim = SimClock()
        self.ws = DirWorkspace(os.path.join(self.out, "workspace"), fixture=scenario.get("fixture", "report"))
        self.before = self.ws.snapshot()
        self.env = CapstoneEnv(self.out, self.ws.root)
        self.policy = Policy(os.path.join(FIX24, "reporting_policy.json"))
        self.artifacts = ArtifactStore(EXPORT_FIXTURE)
        self.outbox = os.path.join(self.out, "outbox")
        self.model_calls = []
        self.ops = []            # integration view of every proposal and its fate
        self.completion = []
        self.workers = []
        self.pending_workers = []
        self.human = []
        self.restarts = []
        self.outcome = None
        self.repairs_used = 0
        self.skill_scope = []
        self.stage_marks = {}
        self.review = None
        self.probe = None

    # ---------------------------------------------------------------- intake
    def intake(self, session_token, request):
        principal = authenticate(session_token)
        ignored = sorted(k for k in request if k in self.policy.doc["identity_arguments"])
        grant = self.policy.grant(principal, request["profile"], self.env)
        self.principal, self.grant = principal, grant
        self.log.add("intake", advance=0, user=principal["user"], tenant=principal["tenant"],
                     session=session_token, profile=grant["profile"],
                     identity_fields_ignored={k: request[k] for k in ignored})
        # State store: the run and the first worker's lease (Chapter 12)
        self.rs_path = os.path.join(self.out, "run.sqlite")
        self.rs = RunStore(self.rs_path, create=True)
        with self.rs.transaction("control", RUN_ID) as tx:
            tx.append("run.created", RUN_ID, {"text": request["task"], "source": f"user:{principal['user']}"},
                      event_id=f"{RUN_ID}:created")
        self.worker = "worker-1"
        self.claim_lease()
        self.task = TaskState(request["task"], self.log)
        self.evidence = EvidenceBook(self.log)
        self.plan_ev = PlanEvidence(semantic=True)
        self.plan = Plan(deploy=False, baseline_tests=baseline_tests(self.ws))
        self.initial_sem = workspace_semantic(self.ws)
        self.approvals = ApprovalStore(os.path.join(self.out, "approvals.sqlite"), self.clock, run_id=RUN_ID)
        self.loose = LooseApprovals()
        self.artifacts.publish(REPORT, "r5")
        self.publications = [{"report": REPORT, "revision": "r5", "by": "reporting-pipeline",
                              "version": self.artifacts.version(REPORT, "r5"),
                              "source_revision": "339d3a9f2656", "t": self.clock.iso()}]
        # Services, executor and reconciler for external effects (Chapter 13)
        self.bridge = StoreBridge(self.rs, self.sim, self.holder,
                                  swallow_refusals=self.wiring == "naive")
        self.services = {"export": DeliveringExportService(os.path.join(self.out, "export.sqlite"), self.sim,
                                                           self.artifacts, self.outbox)}
        self.effects = EffectExecutor(self.bridge, self.services)
        self.setup_knowledge(principal)
        self.setup_skills(grant)
        self.setup_context(request)

    def holder(self):
        return (self.worker, self.fence)

    def claim_lease(self):
        run = self.rs.run(RUN_ID)
        fence = run["fence"] + 1
        with self.rs.transaction("control", RUN_ID) as tx:
            tx.append("lease.claimed", RUN_ID, {"owner": self.worker, "fence": fence,
                                                "expires": self.rs.now() + 30},
                      event_id=f"{RUN_ID}:lease:{fence}")
        self.fence = fence

    def set_run_status(self, status):
        n = self.rs.conn.execute("SELECT COUNT(*) FROM events WHERE type='run.status'").fetchone()[0] + 1
        with self.rs.transaction("control", RUN_ID, self.holder()) as tx:
            tx.append("run.status", RUN_ID, {"status": status}, event_id=f"{RUN_ID}:status:{n}")

    def setup_knowledge(self, principal):
        # Retrieval (Chapter 7): filters before ranking, lineage on every passage
        db = open_index(os.path.join(self.out, "retrieval.sqlite"))
        ingest(db, os.path.join(FIX24, "docs"), "structured")
        who = {"id": f"{principal['user']}@{principal['tenant']}", "audiences": ["engineering", "all"]}
        res = Index(db).search("daily report day timezone midnight orders UTC local date",
                               who, "2026-09-25", k=2)
        self.retrieval = {"principal": who, "stages": res["stages"],
                          "top": [{k: c[k] for k in ("chunk_id", "doc_id", "version", "section", "start", "end",
                                                     "content_hash")} for c in res["top"]],
                          "evidence": assemble_evidence(res)}
        # Memory (Chapter 8): scoped, expiring records with provenance
        mem = MemoryStore(os.path.join(self.out, "memory.sqlite"))
        seeds = [
            {"subject": "dev", "kind": "response_preference", "key": "final_summary",
             "value": "list each changed file with its content version", "scope_type": "user",
             "scope_id": "dev", "source_event": "msg-2026-09-02-dev-14", "source_kind": "user_message",
             "assertion": "stated", "at": "2026-09-02T09:14:00Z"},
            {"subject": "dev", "kind": "display_timezone", "key": "display_timezone",
             "value": "America/Los_Angeles", "scope_type": "user", "scope_id": "dev",
             "source_event": "msg-2026-08-20-dev-03", "source_kind": "user_message",
             "assertion": "stated", "at": "2026-08-20T16:03:00Z"},
            {"subject": "dev", "kind": "response_preference", "key": "tests",
             "value": "skip the slow integration tests this session", "scope_type": "session",
             "scope_id": "s-19", "session": "s-19", "source_event": "msg-2026-09-18-dev-07",
             "source_kind": "user_message", "assertion": "stated", "at": "2026-09-18T11:07:00Z",
             "expires_at": "2026-09-19T00:00:00Z"},
            {"subject": "dev", "kind": "report_timezone", "key": "report_timezone",
             "value": "America/Los_Angeles", "scope_type": "user", "scope_id": "dev",
             "source_event": "run-23:call-4", "source_kind": "model_inference",
             "assertion": "inferred", "at": "2026-09-23T10:00:00Z"},
        ]
        self.memory_writes = [dict(mem.propose(dict(s)), key=s["key"], kind=s["kind"]) for s in seeds]
        self.memory = memory_retrieve(mem, "dev", {"user": "dev", "session": "s-24"}, "2026-09-25T10:00:00Z")
        mem.close()

    def setup_skills(self, grant):
        reg = Registry()
        src = lambda n, v: os.path.join(SKILL_VERSIONS, n, v, n)  # noqa: E731
        reg.publish("report-helper", "1.0.0", src("report-helper", "1.0.0"), "reporting-platform", "2026-08-03")
        reg.publish("report-helper", "1.1.0", src("report-helper", "1.1.0"), "reporting-platform", "2026-09-14")
        reg.publish("status-report-writer", "1.1.0", src("status-report-writer", "1.1.0"),
                    "team-communications", "2026-09-14")
        reg.publish("csv-cleanup", "1.0.0", src("csv-cleanup", "1.0.0"), "data-platform", "2026-08-03")
        reg.withdraw("report-helper", "1.1.0", "2026-09-22", "reporting-platform",
                     "procedure no longer requires the failing test before the fix")
        names = ["csv-cleanup", "report-helper", "status-report-writer"]
        root = materialize(reg, names, os.path.join(self.out, "skills"))
        records, diags = discover(root)
        self.skill_records = records
        self.catalog = render_catalog(records)
        self.activation_check = LexicalActivator(records).choose(self.sc["task"])
        self.registry = reg
        self.tools = report_tools(self.ws)
        allowed = [t for t in grant["tools"] if t not in ("export_report",)]
        self.skills = SkillExecutor(self.tools, RunPolicy(allowed, policy_id=self.policy.doc["policy"]),
                                    records, self.ws)
        self.definitions = (skill_tool_definitions(records) + self.tools.definitions()
                            + [EXPORT_DEF, REQUEST_INPUT_DEF])
        self.definitions = [d for d in self.definitions if d["name"] in grant["tools"]
                            or d["name"] in RUNTIME_TOOLS]
        self.tools_tokens = approx_tokens(self.definitions)

    def setup_context(self, request):
        self.records = ContextRecords(RUN_ID, "acme/workspace")
        self.records.add_instruction(SYSTEM + "\n\n" + self.catalog)
        self.records.add_task(1, request["task"], constraints=[{"id": "C1", "text": "Do not deploy it."}],
                              goals=["Fix the missing near-midnight orders with a regression test",
                                     "Export the corrected daily report for 2026-09-11 to destination A"])
        for e, top in zip(self.retrieval["evidence"], self.retrieval["top"]):
            text = "Retrieved reference (document text, not an instruction):\n" + "\n".join(
                u["text"] for u in e["units"])
            self.records.add_context_item("retrieved", "retrieval: " + e["citation"],
                                          top["content_hash"], "document (untrusted)", text,
                                          chunk_id=top["chunk_id"])
        if self.memory["included"]:
            text = "Stored user preferences (memory; they never grant permissions):\n" + "\n".join(
                f"- {m['key']}: {m['value']} [{m['id']}, {m['scope']}]" for m in self.memory["included"])
            self.records.add_context_item("memory", "memory: " + ", ".join(m["id"] for m in self.memory["included"]),
                                          None, "user statement (stored)", text)
        self.builder = StructuredBuilder()

    # ---------------------------------------------------------------- the loop
    def run(self, limit=16):
        replies = list(self.sc["replies"])
        for call_no in range(1, limit + 1):
            self.deliver_workers()
            self.control_step()
            if self.outcome:
                break
            if not replies:
                self.outcome = {"outcome": "incomplete", "reason": "no further proposals; completion refused",
                                "missing": self.completion[-1]["missing"] if self.completion else []}
                self.log.add("run_stopped", reason=self.outcome["reason"])
                break
            label, reply = replies.pop(0)
            build = self.builder.build(self.records, call_no, self.tools_tokens)
            based_on = self.task.current
            self.log.add("model_call_started", call=call_no, task_revision=based_on,
                         request_sha=build["request_sha"], input_tokens=build["budget"]["total_input"])
            hook = self.sc.get("during", {}).get(label)
            if hook:
                getattr(self, hook)()
            outcome = self.model_call(call_no, build, reply, based_on)
            calls = outcome["executable_calls"]
            self.log.add("model_call_finished", advance=6, call=call_no, task_revision=based_on,
                         outcome=outcome["kind"],
                         proposal=[f"{c['id']} {c['name']}" for c in calls] or ["final answer"])
            if not calls:
                if self.propose_completion(call_no, outcome["text"], based_on):
                    break
                continue
            for c in calls:
                try:
                    obs = self.execute(c, based_on, call_no)
                except Interrupted as exc:
                    obs = self.restart_after_loss(c, str(exc))
                self.records.add_result(c, obs)
                if self.outcome:
                    break
            self.control_step()
        return self.outcome

    def model_call(self, call_no, build, reply, based_on):
        mid = f"msg_{RUN_ID}_{call_no:02d}"
        fixture = f"{RUN_ID}-call-{call_no:02d}"
        raw = render_stream(reply, mid, build["budget"]["total_input"], 0)
        outcome, asm = adapter.assemble(raw, fixture, stream_meta(fixture, f"req_{RUN_ID}_{call_no:02d}"))
        with open(os.path.join(self.streams_dir(), f"call{call_no:02d}.sse"), "w", encoding="utf-8") as f:
            f.write(raw)
        assistant = {"role": "assistant", "content": outcome["text"],
                     "tool_calls": copy.deepcopy(outcome["executable_calls"])}
        self.records.add_assistant(assistant, call_no)
        entry = {"call": call_no, "task_revision": based_on, "request_sha": build["request_sha"],
                 "messages": len(build["messages"]), "input_tokens": build["budget"]["total_input"],
                 "policy": build["policy"]["id"], "policy_hash": build["policy"]["hash"],
                 "provenance": build["provenance"], "omissions": build["omissions"],
                 "adapter": {k: outcome[k] for k in ("adapter_version", "requested_model", "reported_model",
                                                     "kind", "raw_stop_reason", "normalized_stop", "usage",
                                                     "sse_events", "fixture_sha")},
                 "calls": [c["id"] for c in outcome["executable_calls"]], "text": outcome["text"],
                 "messages_sent": build["messages"]}
        self.model_calls.append(entry)
        with self.rs.transaction("model_adapter", RUN_ID, self.holder()) as tx:
            tx.append("model_call.recorded", f"m{call_no:02d}",
                      {"ordinal": call_no, "task_revision": based_on, "input_sha": build["request_sha"],
                       "input_messages": len(build["messages"]),
                       "response": {"kind": outcome["kind"], "raw_stop_reason": outcome["raw_stop_reason"],
                                    "reported_model": outcome["reported_model"],
                                    "calls": outcome["executable_calls"], "text": outcome["text"],
                                    "usage": outcome["usage"]}},
                      event_id=f"{RUN_ID}:m{call_no:02d}")
        return outcome

    def streams_dir(self):
        d = os.path.join(self.out, "streams")
        os.makedirs(d, exist_ok=True)
        return d

    # ---------------------------------------------------------------- execution
    def prepare(self, call, based_on, call_no, extra=None):
        payload = {"model_call": f"m{call_no:02d}", "tool": call["name"], "arguments": call["arguments"],
                   "task_revision": based_on, "covers": {p: content_version(t) for p, t in self.ws.snapshot().items()}}
        payload.update(extra or {})
        with self.rs.transaction("control", RUN_ID, self.holder()) as tx:
            tx.append("op.prepared", call["id"], payload, event_id=f"{call['id']}:prepared")

    def write(self, writer, etype, entity, payload, n=""):
        fenced = writer != "policy"
        with self.rs.transaction(writer, RUN_ID, self.holder() if fenced else None) as tx:
            tx.append(etype, entity, payload, event_id=f"{entity}:{etype}{n}")

    def record_op(self, call, based_on, decision, rules, outcome, **extra):
        rec = {"call_id": call["id"], "tool": call["name"], "arguments": call["arguments"],
               "task_revision": based_on, "decision": decision,
               "rules": [r for r in rules if r["outcome"] != "pass"],
               "rules_evaluated": len(rules), "principal": self.principal["user"],
               "session": self.principal["session"], "policy_version": self.policy.version,
               "skill_scope": sorted(self.skills.active), "outcome": outcome, "t": self.clock.iso()}
        rec.update(extra)
        self.ops.append(rec)
        return rec

    def evaluate(self, call):
        decision, rules = evaluate(self.policy, self.principal, self.grant, self.env, call)
        if call["name"] == "export_report" and isinstance(call["arguments"], dict):
            rev = call["arguments"].get("revision")
            latest = self.artifacts.latest.get(call["arguments"].get("report"))
            ok = rev == latest
            rules.append({"rule": "freshness.latest_revision", "outcome": "pass" if ok else "deny",
                          "detail": f"{rev} is the latest published revision" if ok else
                                    f"{rev} is superseded; latest published revision is {latest}"})
            if not ok:
                decision = "deny"
        return decision, rules

    def execute(self, call, based_on, call_no):
        name = call["name"]
        extra = None
        if name == "export_report" and isinstance(call["arguments"], dict) and \
                (call["arguments"].get("report"), call["arguments"].get("revision")) in self.artifacts.revisions:
            extra = self.export_identity(call)
        self.prepare(call, based_on, call_no, extra)
        decision, rules = self.evaluate(call)
        if decision == "deny":
            self.write("executor", "op.failed_before_effect", call["id"], {"stage": "policy"})
            obs = {"status": "refused", "stage": "policy",
                   "rules": [r["rule"] for r in rules if r["outcome"] == "deny"],
                   "message": "; ".join(r["detail"] for r in rules if r["outcome"] == "deny"),
                   "retryable": False}
            self.record_op(call, based_on, "deny", rules, "refused by policy")
            self.log.add("dispatch_refused", call_id=call["id"], tool=name, stage="policy",
                         rules=obs["rules"])
            return obs
        # Chapter 10: while a skill is active, its declarations narrow the run
        declared = set()
        for rec in self.skills.active.values():
            declared |= set(rec["allowed_tools"])
        if declared and name not in declared | RUNTIME_TOOLS:
            self.write("executor", "op.failed_before_effect", call["id"], {"stage": "skill declarations"})
            self.record_op(call, based_on, "deny", rules, "outside active skill declarations")
            return {"status": "refused", "stage": "skill declarations",
                    "message": f"{name} is outside the active skills' declared tools"}
        # Shape (Chapter 3 / 9)
        definition = next((d for d in self.definitions if d["name"] == name), None)
        try:
            check_args(definition["parameters"], call["arguments"])
        except InputError as exc:
            self.repairs_used += 1
            self.write("executor", "op.failed_before_effect", call["id"], {"stage": "shape"})
            self.record_op(call, based_on, decision, rules, "input_error",
                           repair_allowance=f"{self.repairs_used} of {REPAIR_ALLOWANCE} used")
            self.log.add("input_error", call_id=call["id"], message=str(exc))
            return {"status": "input_error", "stage": "shape", "message": str(exc), "retryable": True}
        # Chapter 14: effects proposed under an earlier task revision are not dispatched
        if name in EFFECTFUL and based_on != self.task.current:
            self.write("control", "op.skipped", call["id"], {"reason": "stale_task_revision"})
            self.record_op(call, based_on, decision, rules, "not dispatched: stale task revision",
                           current_task_revision=self.task.current)
            self.log.add("dispatch_refused", call_id=call["id"], tool=name, reason="stale_task_revision",
                         proposal_task_revision=based_on, current_task_revision=self.task.current)
            return {"status": "not_dispatched", "reason": "stale_task_revision",
                    "proposal_task_revision": based_on, "current_task_revision": self.task.current,
                    "message": "Proposed under an earlier task revision; regenerate against the current task."}
        if name == "export_report":
            return self.execute_export(call, based_on, decision, rules)
        if name == "request_input":
            return self.execute_request_input(call, based_on, decision, rules)
        # Workspace and skill operations through Chapter 10's SkillExecutor
        self.write("executor", "op.dispatched", call["id"], {"covers": {}, "fence": self.fence})
        self.log.add("dispatched", call_id=call["id"], tool=name, task_revision=based_on)
        obs = self.skills.execute(call)
        status = obs.get("status")
        self.write("executor", "observation.recorded", call["id"], {"status": status, "result": {
            k: v for k, v in obs.items() if k not in ("text", "content")}})
        self.write("executor", "op.completed", call["id"], {})
        if name == "activate_skill" and status == "ok":
            self.skill_scope.append({"event": "opened", "skill": obs["skill"], "version": obs["version"],
                                     "call_id": call["id"], "t": self.clock.iso()})
        if name == "replace_text" and status == "ok":
            n = self.rs.conn.execute("SELECT COUNT(*) FROM artifact_versions WHERE path=?",
                                     (obs["path"],)).fetchone()[0] + 1
            self.write("executor", "artifact.versioned", call["id"],
                       {"path": obs["path"], "version_no": n, "content": self.ws.read(obs["path"]),
                        "produced_by": call["id"]}, n=f":{n}")
            if obs["path"] == "report.py":
                self.evidence.artifact_changed(obs["after_version"])
        self.log.add("observed", advance=2, call_id=call["id"], status=status,
                     **({"path": obs["path"], "before_version": obs["before_version"],
                         "after_version": obs["after_version"]} if "after_version" in obs else {}))
        self.plan_evidence(call, obs)
        self.record_op(call, based_on, decision, rules, status)
        return obs

    def plan_evidence(self, call, obs):
        name, step = call["name"], len(self.model_calls)
        version = content_version(self.ws.read("report.py"))
        if name == "read_file" and obs.get("status") == "ok":
            p = call["arguments"]["path"]
            v = self.plan_ev.version(p, obs["text"])
            self.plan_ev.add("read", step, space="file", path=p, version=v, key=[p, v])
        elif name == "replace_text" and obs.get("status") == "ok":
            p = call["arguments"]["path"]
            v_after = self.plan_ev.version(p, self.ws.read(p))
            self.plan_ev.add("edit", step, space="file", path=p, version=v_after, source=p != "test_report.py",
                             semantic_changed=v_after != self.initial_sem.get(p), raw_version=obs["after_version"],
                             key=[p, v_after])
        elif name == "run_tests" and obs.get("status") in ("passed", "failed"):
            sem = workspace_semantic(self.ws)
            self.plan_ev.add("tests", step, status=obs["status"], failed=obs["failed_tests"],
                             tests_run=obs["tests_run"], workspace=sem,
                             key=[obs["status"], sorted(obs["failed_tests"]), sorted(sem.items())])
            self.evidence.record("tests", obs, version, self.task.current, call["id"])
        elif name == "run_skill_script":
            self.probe = {"call_id": call["id"], "status": obs.get("status"), "version": version,
                          "stdout": obs.get("stdout", "")}

    def export_identity(self, call):
        a = call["arguments"]
        payload = {"report": a["report"], "revision": a["revision"], "destination": a["destination"],
                   "report_date": "2026-09-11", "source_revision": content_version(self.ws.read("report.py")),
                   "artifact_version": self.artifacts.version(a["report"], a["revision"])}
        return {"op_key": f"{RUN_ID}:export:{a['report']}:{a['revision']}:{a['destination']}",
                "payload": payload, "payload_digest": op_digest(payload),
                "recovery_class": RECOVERY_CATALOG["tools"]["create_export"]["recovery_class"],
                "service_operation": "create_export", "catalog_version": RECOVERY_CATALOG["version"],
                "covers": {"report": a["report"], "revision": a["revision"],
                           "artifact_version": payload["artifact_version"]}}

    # ---------------------------------------------------------------- approvals and export
    def present(self, args, version, diff=None):
        text = self.artifacts.text(args["report"], args["revision"])
        rows = [l for l in text.splitlines()[1:] if not l.startswith("total")]
        total = [l for l in text.splitlines() if l.startswith("total")][0].split(",")[3]
        ev = [f"{i['id']}: {i['kind']} {i['result']} on report.py {i['version']}" for i in self.evidence.items
              if i["state"] == "current"]
        p = {"decision": f"Export {args['report']} {args['revision']} to destination {args['destination']}?",
             "operation": "export_report", "arguments": args, "artifact_version": version,
             "what_changes": f"Writes one CSV ({len(rows)} orders, total {total} USD) to destination "
                             f"{args['destination']}; recipients can read it immediately.",
             "evidence": ev,
             "choices": [{"id": "approve", "consequence": "Export runs once, for this revision and destination only."},
                         {"id": "reject", "consequence": "Nothing is exported; the run reports the refusal."}],
             "expires_in_s": 900}
        if diff:
            p["diff"] = diff
        return p

    def approval_round(self, call, args, version, diff=None):
        rq, created = self.approvals.request("export_report", args, version, self.present(args, version, diff))
        self.write("policy", "op.awaiting_approval", call["id"], {"request_id": rq}, n=f":{rq}")
        self.set_run_status("waiting_for_person")
        self.log.add("approval_requested", request_id=rq, call_id=call["id"], revision=args["revision"],
                     artifact_version=version)
        for actor, choice in self.sc["decisions"].get(rq, []):
            principal = authenticate(f"sess-acme-{actor}")
            res = self.approvals.decide(rq, principal["user"], choice, submission_key=f"{rq}:{actor}:1")
            self.human.append({"request_id": rq, "actor": actor, "session": principal["session"],
                               "choice": choice, "result": res})
            self.log.add("decision", request_id=rq, actor=actor, choice=choice, status=res["status"],
                         approval_id=res.get("approval_id"), reason=res.get("reason"))
            if res.get("approval_id"):
                self.loose.approve(actor, "export_report", args["destination"])
        self.set_run_status("running")
        hook = self.sc.get("after_decision", {}).get(rq)
        if hook:
            getattr(self, hook)()
        return rq

    def execute_export(self, call, based_on, decision, rules):
        args = call["arguments"]
        version = self.artifacts.version(args["report"], args["revision"])
        approval = None
        if self.wiring == "revised":
            chk = self.approvals.check_at_dispatch(call["id"], "export_report", args, version)
            if not chk["allowed"]:
                self.log.add("approval_check", call_id=call["id"], allowed=False,
                             mismatches=chk.get("mismatches", []), reason=chk["reason"])
                diff = None
                prior = [m for m in chk.get("mismatches", [])]
                if prior:
                    prev = self.approvals.approval_row(prior[0]["approval_id"])
                    old_rev = [r for (rep, r) in self.artifacts.revisions if rep == args["report"]
                               and self.artifacts.version(rep, r) == prev["artifact_version"]][0]
                    diff = self.artifacts.diff(args["report"], old_rev, args["revision"])
                self.approval_round(call, args, version, diff)
                decision, rules = self.evaluate(call)       # authority is re-checked at dispatch
                if decision == "deny":
                    self.write("control", "op.cancelled", call["id"], {"reason": "refused by policy at dispatch"})
                    obs = {"status": "refused", "stage": "policy at dispatch",
                           "rules": [r["rule"] for r in rules if r["outcome"] == "deny"],
                           "message": "; ".join(r["detail"] for r in rules if r["outcome"] == "deny"),
                           "retryable": False}
                    self.record_op(call, based_on, "deny", rules, "refused at dispatch")
                    self.log.add("dispatch_refused", call_id=call["id"], tool="export_report",
                                 stage="policy at dispatch", rules=obs["rules"])
                    return obs
                chk = self.approvals.check_at_dispatch(call["id"], "export_report", args, version)
                if not chk["allowed"]:
                    self.write("control", "op.cancelled", call["id"], {"reason": "no approval covers it"})
                    self.record_op(call, based_on, "needs_approval", rules, "no approval")
                    return {"status": "refused", "stage": "approval", "message": chk["reason"]}
            approval = self.approvals.approval_row(chk["approval_id"])
            self.log.add("approval_consumed", call_id=call["id"], approval_id=approval["approval_id"],
                         actor=approval["actor"], artifact_version=approval["artifact_version"])
        else:
            chk = self.loose.check("export_report", args)
            if not chk["allowed"]:
                self.approval_round(call, args, version)
                decision, rules = self.evaluate(call)
                if decision == "deny":
                    self.write("control", "op.cancelled", call["id"], {"reason": "refused by policy at dispatch"})
                    self.record_op(call, based_on, "deny", rules, "refused at dispatch")
                    return {"status": "refused", "stage": "policy at dispatch",
                            "rules": [r["rule"] for r in rules if r["outcome"] == "deny"],
                            "message": "; ".join(r["detail"] for r in rules if r["outcome"] == "deny")}
                chk = self.loose.check("export_report", args)
            approval = {"approval_id": "loose", "actor": chk["grant"]["actor"],
                        "artifact_version": None, "scope": "tool and destination"}
            self.log.add("approval_consumed", call_id=call["id"], approval_id="loose",
                         actor=chk["grant"]["actor"], artifact_version=None)
        self.current_export = {"call_id": call["id"], "approval": approval, "version": version,
                               "based_on": based_on, "rules": rules, "decision": decision}
        interrupt = "after_commit" if self.sc.get("interrupt_export") and not self.restarts else None
        self.sim.advance(1.0)
        receipt = self.effects.dispatch(call["id"], interrupt=interrupt)
        return self.finish_export(call, receipt)

    def finish_export(self, call, receipt):
        cur = self.current_export
        self.log.add("observed", advance=2, call_id=call["id"], status="completed",
                     export_id=receipt["export_id"], replayed=receipt["replayed"])
        self.record_op(call, cur["based_on"], cur["decision"], cur["rules"], "completed",
                       approval=cur["approval"], receipt={k: receipt[k] for k in
                                                          ("export_id", "replayed", "file", "file_version")})
        return {"status": "completed", "export_id": receipt["export_id"], "file": receipt["file"],
                "file_version": receipt["file_version"], "replayed": receipt["replayed"]}

    def execute_request_input(self, call, based_on, decision, rules):
        self.write("executor", "op.dispatched", call["id"], {"covers": {}, "fence": self.fence})
        rq = {"request_id": "IN1", "question": call["arguments"]["question"],
              "choices": call["arguments"]["choices"], "asked_by": "model", "t": self.clock.iso()}
        self.input_requests = [rq]
        self.write("executor", "observation.recorded", call["id"], {"status": "waiting", "result": rq})
        self.write("executor", "op.completed", call["id"], {})
        self.set_run_status("waiting_for_person")
        self.record_op(call, based_on, decision, rules, "waiting for person")
        self.log.add("input_requested", call_id=call["id"], question=rq["question"])
        self.outcome = {"outcome": "waiting_for_person", "reason": "input needed: " + rq["question"],
                        "effects": 0}
        return {"status": "waiting", "request_id": "IN1"}

    # ---------------------------------------------------------------- interruption and recovery
    def restart_after_loss(self, call, point):
        self.log.add("worker_lost", call_id=call["id"], point=point, worker=self.worker, fence=self.fence)
        before = {"run_store_op": self.rs.op(call["id"])["state"],
                  "observation": self.rs.conn.execute("SELECT COUNT(*) FROM observations WHERE op_id=?",
                                                      (call["id"],)).fetchone()[0],
                  "export_rows": len(export_rows(os.path.join(self.out, "export.sqlite")))}
        # A new process: only the database files and the workspace survive.
        self.rs.conn.close()
        self.sim.advance(4.5)
        self.rs = RunStore(self.rs_path)
        self.rs.advance(31)
        self.worker = "worker-2"
        self.claim_lease()
        self.bridge = StoreBridge(self.rs, self.sim, self.holder,
                                  swallow_refusals=self.wiring == "naive")
        self.services = {"export": DeliveringExportService(os.path.join(self.out, "export.sqlite"), self.sim,
                                                           self.artifacts, self.outbox)}
        self.effects = EffectExecutor(self.bridge, self.services)
        self.log.add("worker_restarted", advance=5, worker=self.worker, fence=self.fence)
        rebuilt = self.rebuild_check()
        if self.wiring == "revised":
            rec = Reconciler(self.bridge, self.effects, self.services, self.sim)
            decisions = rec.reconcile_all()
        else:
            decisions = naive_retry_all(self.bridge, self.effects)
        op = self.bridge.op(call["id"])
        receipt = op["receipt"]
        self.restarts.append({"point": point, "before": before, "decisions": decisions,
                              "bridge": self.bridge.decisions, "swallowed_refusals": self.bridge.swallowed,
                              "receipt": receipt, "rebuilt_conversation_matches": rebuilt,
                              "new_holder": list(self.holder())})
        self.log.add("reconciled", call_id=call["id"], decisions=decisions,
                     export_id=receipt.get("export_id") if receipt else None,
                     replayed=receipt.get("replayed") if receipt else None)
        return self.finish_export(call, receipt)

    def rebuild_check(self):
        """Every interface is computed from committed records: rebuild the model-visible
        conversation from run-store events and compare it with the context records."""
        events = read_events(self.rs.conn, RUN_ID)
        from_events = [e["payload"]["response"]["calls"] for e in events if e["type"] == "model_call.recorded"]
        from_records = [r["message"]["tool_calls"] for r in self.records.records if r["kind"] == "assistant"]
        folded = fold(events)
        return {"assistant_turns_equal": from_events == from_records,
                "fold_equals_tables": canon(folded) == canon(state_from_tables(self.rs.conn, RUN_ID))}

    # ---------------------------------------------------------------- human events (scripted)
    def correction(self):
        text = "Keep the public function signature unchanged."
        self.clock.tick(2)
        run = self.rs.run(RUN_ID)
        rev = run["task_revision"] + 1
        with self.rs.transaction("control", RUN_ID, self.holder()) as tx:
            tx.append("task.revised", RUN_ID, {"revision": rev, "text": text, "source": "user:dev"},
                      event_id=f"{RUN_ID}:task:{rev}")
        self.task.revise(text, checks=["signature:belongs_to_report"])
        self.records.add_task(rev, text, constraints=[{"id": "C2", "text": text,
                                                       "contract": "signature:belongs_to_report"}], goals=[])
        self.human.append({"event": "correction", "actor": "dev", "task_revision": rev, "text": text,
                           "t": self.clock.iso()})

    def publish_r6(self):
        self.artifacts.publish(REPORT, "r6")
        self.publications.append({"report": REPORT, "revision": "r6", "by": "finance-review",
                                  "version": self.artifacts.version(REPORT, "r6"),
                                  "source_revision": "339d3a9f2656",
                                  "change": "removed refunded order A-1003", "t": self.clock.iso()})
        self.log.add("artifact_published", report=REPORT, revision="r6", by="finance-review",
                     version=self.artifacts.version(REPORT, "r6"))

    # ---------------------------------------------------------------- control
    def control_step(self):
        sem = workspace_semantic(self.ws)
        self.plan.update(self.plan_ev, sem, [])
        stage = self.plan.stage()
        self.stage_marks.setdefault(stage, self.clock.iso())
        label = self.revision_label()
        # Delegation (Chapter 15): characterize the defect at r4, review the fix at r5
        if self.sc.get("characterize_at_r4") and label == "r4" and not any(w["contract_id"] == "review-tz-r4" for w in
                                                                 self.workers + self.pending_workers) \
                and self.plan.items[1]["status"] == "satisfied":
            self.dispatch_worker("review-tz-r4", "tz-boundary-reviewer", "r4")
        if self.sc.get("delegate") and label == "r5" and not any(w["contract_id"].endswith("-r5") for w in
                                                                 self.workers + self.pending_workers):
            self.dispatch_worker("review-tz-r5", "tz-boundary-reviewer", "r5")
            if "signature:belongs_to_report" in self.task.active_checks():
                self.dispatch_worker("review-iface-r5", "interface-reviewer", "r5")
        # Once the fix is verified and the reviews for this revision are back, close the
        # skill's scope (its declarations governed the fix stages) and integrate the reviews.
        if self.plan.items[3]["status"] == "satisfied" and self.reviews_settled() \
                and getattr(self, "integrated_at", None) != label:
            if not self.sc.get("keep_skill_scope"):
                for name, rec in list(self.skills.active.items()):
                    self.skills.active.pop(name)
                    self.skill_scope.append({"event": "closed", "skill": name, "version": rec["version"],
                                             "reason": "fix and verify stages satisfied", "t": self.clock.iso()})
                    self.log.add("skill_scope_closed", skill=name, version=rec["version"])
            self.integrated_at = label
            self.integrate_reviews()

    def revision_label(self):
        v = {p: content_version(t) for p, t in self.ws.snapshot().items()}
        for label in ("r3", "r4", "r5"):
            if self.revstore.revs[label]["versions"] == v:
                return label
        return "unlabeled:" + content_version(json.dumps(v, sort_keys=True))

    @property
    def revstore(self):
        if not hasattr(self, "_revstore"):
            self._revstore = RevisionStore()
            self.evlog = EvidenceLog()
        return self._revstore

    def dispatch_worker(self, cid, worker, revision):
        if worker == "tz-boundary-reviewer":
            contract = make_contract(cid, worker, "Run every boundary case against belongs_to_report and report "
                                     "each with its evidence id.", revision, self.revstore,
                                     ["read_file", "run_cases"], BOUNDARY_CASES, max_tokens=4000)
            script = script_tz(contract, BOUNDARY_CASES)
        else:
            contract = make_contract(cid, worker, "Compare the public signature of belongs_to_report with r3.",
                                     revision, self.revstore, ["read_file", "inspect_interface"],
                                     ["public_interface"], max_tokens=4000, reference_revision="r3")
            script = script_iface(contract)
        res = run_worker(contract, script, self.revstore, self.evlog)
        due = self.clock.s + math.ceil(res["ledger"]["sim_seconds"])
        entry = {"contract_id": cid, "worker": worker, "revision": revision, "dispatched": self.clock.iso(),
                 "due_s": due, "result": res, "contract": contract, "task_revision": self.task.current}
        e = self.log.add("worker_dispatched", contract_id=cid, worker=worker, input_revision=revision,
                         input_digest=contract["input_digest"])
        entry["dispatched"] = e["t"]
        entry["due_s"] = self.clock.s + math.ceil(res["ledger"]["sim_seconds"])
        self.pending_workers.append(entry)

    def deliver_workers(self, force=False):
        for w in list(self.pending_workers):
            if force or self.clock.s >= w["due_s"]:
                self.pending_workers.remove(w)
                if w["due_s"] > self.clock.s:
                    self.clock.tick(w["due_s"] - self.clock.s)
                w["returned"] = self.clock.iso(w["due_s"])
                self.workers.append(w)
                r = w["result"]
                self.log.add("worker_returned", advance=0, contract_id=w["contract_id"], finished=w["returned"],
                             status=r["status"],
                             verdict=(r["result"] or {}).get("verdict"), input_revision=w["revision"],
                             current_revision=self.revision_label())

    def reviews_settled(self):
        if not self.sc.get("delegate"):
            return True
        return not any(w["revision"] == self.revision_label() for w in self.pending_workers) and \
            any(w["revision"] == self.revision_label() for w in self.workers + self.pending_workers)

    def integrate_reviews(self):
        if not self.sc.get("delegate"):
            self.review = None
            return
        self.deliver_workers(force=True)
        outcomes = [w["result"] for w in self.workers]
        required = list(BOUNDARY_CASES)
        if "signature:belongs_to_report" in self.task.active_checks():
            required.append("public_interface")
        label = self.revision_label()
        self.review = integrate(outcomes, label, required)
        self.log.add("reviews_integrated", current_revision=label, verdict=self.review["verdict"],
                     stale=[s["worker"] + "@" + s["reviewed"] for s in self.review.get("stale", [])])
        version = content_version(self.ws.read("report.py"))
        for w in self.workers:
            r = w["result"]
            if w["revision"] == label and r["status"] == "accepted" and w["worker"] == "interface-reviewer":
                ev = [self.evlog.items[i] for i in r["result"]["coverage"]["public_interface"]][0]
                obs = {"matches_baseline": ev["payload"]["unchanged"], "signature": ev["payload"]["signature"]}
                self.evidence.record("signature", obs, version, self.task.current, w["contract_id"])

    # ---------------------------------------------------------------- verification
    def naive_verdict(self):
        """The glue's own completion check: a pass after the latest edit, and the export tool said ok."""
        msgs = [r["message"] for r in self.records.records]
        tests = flawed_verifier(msgs)
        exported = any(o["tool"] == "export_report" and o["outcome"] == "completed" for o in self.ops)
        return {"complete": tests["complete"] and exported, "reason": tests["reason"] +
                ("; export tool returned completed" if exported else "; no export")}

    def propose_completion(self, call_no, text, based_on):
        verdict = self.verify(text, based_on)
        if self.wiring == "naive":
            naive = self.naive_verdict()
            verdict = dict(verdict, revised_verifier_complete=verdict["complete"],
                           revised_verifier_missing=verdict["missing"], complete=naive["complete"],
                           missing=[] if naive["complete"] else [naive["reason"]], naive_verifier=naive)
        self.completion.append(dict(verdict, call=call_no))
        with open(os.path.join(self.out, "answer.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        if verdict["complete"]:
            with self.rs.transaction("control", RUN_ID, self.holder()) as tx:
                tx.append("run.completed", RUN_ID, {"status": "completed"}, event_id=f"{RUN_ID}:completed")
            self.log.add("run_completed", call=call_no, task_revision=self.task.current)
            self.outcome = {"outcome": "completed", "call": call_no,
                            "verified_by": "completion predicate" if self.wiring == "revised" else
                            "glue check: tests passed after the last edit and the export tool returned completed"}
            return True
        self.log.add("completion_refused", call=call_no, missing=verdict["missing"])
        self.records.add_context_item("runtime_note", "verifier", None, "runtime",
                                      "Runtime: not complete. Missing: " + "; ".join(verdict["missing"]))
        return False

    def verify(self, text, based_on):
        sem = workspace_semantic(self.ws)
        items = self.plan.update(self.plan_ev, sem, [])
        missing = [f"{i['id']} {i['claim']}" for i in items if i["status"] != "satisfied"]
        rv = revision_verifier(self.task, self.evidence, self.ws)
        missing += rv["missing"]
        checks = {"plan": [{k: i[k] for k in ("id", "claim", "status", "evidence")} for i in items],
                  "revision_verifier": rv}
        if based_on != self.task.current:
            missing.append(f"completion proposed under task revision {based_on}; current is {self.task.current}")
        if self.sc.get("delegate"):
            label = self.revision_label()
            ok = self.review and self.review.get("verdict") == "verified" and \
                self.review.get("current_revision") == label
            checks["P6"] = {"claim": "Independent review verified at the current revision",
                            "status": "satisfied" if ok else "pending",
                            "evidence": self.review}
            if not ok:
                missing.append("P6 independent review verified at the current revision")
        if self.sc.get("export"):
            p7 = self.check_export()
            checks["P7"] = p7
            if p7["status"] != "satisfied":
                missing += ["P7 " + m for m in p7["problems"]]
        claims = self.check_claims(text)
        checks["claims"] = claims
        if claims["unsupported"]:
            missing.append("answer names identifiers with no record: " + ", ".join(claims["unsupported"]))
        return {"complete": not missing, "missing": missing, "checks": checks,
                "task_revision": self.task.current}

    def check_export(self):
        rows = export_rows(os.path.join(self.out, "export.sqlite"))
        box = outbox_listing(self.outbox)
        problems = []
        exp = [o for o in self.ops if o["tool"] == "export_report" and o["outcome"] == "completed"]
        if not exp:
            return {"claim": "Export delivered as approved", "status": "pending", "problems": ["no export completed"]}
        last = exp[-1]
        appr = last["approval"]
        files = box.get("A", [])
        by_key = {}
        for r in rows:
            by_key.setdefault(r["op_key"], []).append(r["export_id"])
        dup = {k: v for k, v in by_key.items() if len(v) > 1}
        if dup:
            problems.append(f"one operation produced {sum(len(v) for v in dup.values())} exports: {dup}")
        if len(files) != 1:
            problems.append(f"destination A holds {len(files)} files")
        delivered = files[-1]["version"] if files else None
        if appr.get("artifact_version") != delivered:
            problems.append(f"approval {appr['approval_id']} by {appr['actor']} covers "
                            f"{appr.get('artifact_version')}, delivered file is {delivered}")
        content = None
        if files:
            path = os.path.join(self.outbox, "A", files[-1]["file"])
            now = rows_against_source(self.ws.root, path, "2026-09-11", "America/Los_Angeles")
            src = os.path.join(self.out, "source-r3")
            if not os.path.exists(src):
                os.makedirs(src)
                with open(os.path.join(src, "report.py"), "w", encoding="utf-8") as f:
                    f.write(self.before["report.py"])
            old = rows_against_source(src, path, "2026-09-11", "America/Los_Angeles")
            content = {"file": files[-1]["file"], "rows": len(now),
                       "belong_under_current_source": sum(r["belongs"] for r in now),
                       "belong_under_original_source": sum(r["belongs"] for r in old),
                       "current_source": content_version(self.ws.read("report.py")),
                       "original_source": content_version(self.before["report.py"])}
            if content["belong_under_current_source"] != content["rows"]:
                problems.append("delivered rows disagree with the current source revision")
        return {"claim": "Export delivered exactly once, as approved, consistent with the source",
                "status": "pending" if problems else "satisfied", "problems": problems,
                "export_rows": rows, "outbox": box, "approval": appr, "content": content}

    def check_claims(self, text):
        import re
        known = set()
        for rec in self.rs.conn.execute("SELECT payload FROM events"):
            known.update(re.findall(r"[0-9a-f]{12}|export-file-\d{3}|\br\d\b", rec[0]))
        for p in self.publications:
            known.update([p["revision"], p["version"]])
        named = sorted(set(re.findall(r"\b[0-9a-f]{12}\b|export-file-\d{3}|\br\d\b", text)))
        return {"named": named, "unsupported": [n for n in named if n not in known]}

    # ---------------------------------------------------------------- the record
    def five_questions(self):
        ev = self.evidence.items
        state = fold(read_events(self.rs.conn, RUN_ID))
        after = self.ws.snapshot()
        def trust_counts(m):
            out = {}
            for p in m["provenance"]:
                out[p["trust"]] = out.get(p["trust"], 0) + 1
            return out
        q1 = {"model_calls": [{
            "call": m["call"], "task_revision": m["task_revision"], "request_sha": m["request_sha"],
            "messages": m["messages"], "input_tokens_approx": m["input_tokens"],
            "context_policy": f"{m['policy']} {m['policy_hash']}", "items_by_trust": trust_counts(m),
            "omissions": len(m["omissions"]), "adapter_outcome": m["adapter"]["kind"],
            "raw_stop_reason": m["adapter"]["raw_stop_reason"], "reported_model": m["adapter"]["reported_model"]}
            for m in self.model_calls],
            "last_call_provenance": [{k: p.get(k) for k in ("source", "transform", "revision", "trust", "tokens")}
                                     for p in self.model_calls[-1]["provenance"]],
            "excluded_before_the_model": {
                "retrieval_filters": self.retrieval["stages"][:4],
                "memory_excluded": [{"id": x["id"], "reason": x["reason"]} for x in self.memory["excluded"]],
                "memory_writes_rejected": [{"key": w["key"], "reasons": w.get("reasons")} for w in self.memory_writes
                                           if w["status"] == "rejected"]}}
        q2 = [{"call_id": o["call_id"], "tool": o["tool"], "task_revision": o["task_revision"],
               "principal": f"{o['principal']} via {o['session']}", "decision": o["decision"],
               "rules_not_passed": [r["rule"] + ":" + r["outcome"] for r in o["rules"]],
               "approval": ({k: o["approval"].get(k) for k in ("approval_id", "actor", "role", "artifact_version",
                                                              "scope")} if o.get("approval") else None),
               "outcome": o["outcome"]} for o in self.ops]
        changed = {p: {"before": content_version(self.before[p]), "after": content_version(after[p])}
                   for p in sorted(after) if after[p] != self.before.get(p)}
        q3 = {"workspace": changed, "exports": export_rows(os.path.join(self.out, "export.sqlite")),
              "outbox": outbox_listing(self.outbox),
              "memory_writes_by_run": 0, "deployments": sum(1 for o in self.ops if o["tool"] == "deploy"),
              "run_store_operations": {k: v["state"] for k, v in state["operations"].items()}}
        q4 = {"evidence": [{k: i.get(k) for k in ("id", "kind", "result", "version", "task_revision", "state",
                                                  "call_id")} for i in ev],
              "review": ({k: self.review.get(k) for k in ("current_revision", "verdict", "admitted", "stale",
                                                          "missing_coverage")} if getattr(self, "review", None)
                         else None),
              "probe": getattr(self, "probe", None),
              "verdict": self.completion[-1] if self.completion else None}
        uncertain = []
        for k, v in q3["run_store_operations"].items():
            if v in ("dispatched", "uncertain"):
                uncertain.append(f"operation {k} is {v} without a recorded outcome")
        for s in (self.review or {}).get("stale", []) if getattr(self, "review", None) else []:
            uncertain.append(f"{s['worker']} result for {s['reviewed']} ({s['verdict']}) kept as evidence about "
                             f"{s['reviewed']} only")
        for p in self.publications:
            if p["by"] not in ("reporting-pipeline",):
                uncertain.append(f"{p['report']} {p['revision']} was published by {p['by']} ({p.get('change')}); "
                                 f"this run did not verify that change")
        if getattr(self, "review", None) and self.review.get("coverage"):
            uncertain.append("boundary evidence covers " + str(len([c for c in self.review["coverage"]
                                                                  if c != "public_interface"]))
                             + " named cases in America/Los_Angeles only")
        for r in self.restarts:
            uncertain.append(f"the export receipt was recovered after a lost worker by {r['decisions']}; "
                             f"delivery to readers of destination A is not observed by this run")
        if self.outcome and self.outcome["outcome"] != "completed":
            uncertain.append("run not complete: " + self.outcome["reason"])
        uncertain.append("model replies are scripted stand-ins; the record shows what the runtime did "
                         "with them, and says nothing about how often a live model proposes them")
        return {"run": RUN_ID, "outcome": self.outcome,
                "1_what_did_the_model_receive": q1,
                "2_which_operation_was_authorized_and_by_whom": q2,
                "3_what_actually_changed": q3,
                "4_which_evidence_supports_the_result": q4,
                "5_what_remains_uncertain": uncertain}

    def manifest(self):
        """Which version of every component ran (governance)."""
        skill = self.registry.resolve("report-helper")
        return {"model_adapter": adapter.ADAPTER_VERSION, "requested_model": REQUESTED_MODEL,
                "reported_model": REPORTED_MODEL,
                "context_policy": {"id": self.builder.policy["id"], "hash": self.builder.policy["hash"]},
                "retrieval": {"chunker": "structured", "embedder": "hashvec-4gram-1024/v1"},
                "memory_write_policy": "wp-3",
                "skill": {"name": "report-helper", "version": skill["version"], "hash": skill["hash"][:12],
                          "withdrawn_versions_excluded": ["1.1.0"]},
                "policy": {"name": self.policy.doc["policy"], "version": self.policy.version},
                "recovery_catalog": RECOVERY_CATALOG["version"],
                "review_contract": "review-contract/1.1", "run_store_schema": 2,
                "controller": "ch11 Plan P1-P5 + capstone P6-P7"}
