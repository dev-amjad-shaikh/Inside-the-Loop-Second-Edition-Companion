"""Chapter 15 helpers: a coordinator, typed worker contracts and read-only reviewers.

Everything here is deterministic. Model decisions are scripted; the final
response of each reviewer is composed by a deterministic function from the
observations that reviewer actually received (a stand-in for the model's
summary step). File reads, boundary-case execution and interface inspection
are real operations against materialized artifact revisions.

Token counts are approximate: characters of the JSON payload divided by four,
rounded up. Latency is simulated from invented constants (SIM below) so that
schedules are reproducible; no measured model latency appears anywhere.
"""
import ast
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

from .core import FIXTURES, ScriptedModel, ToolRegistry, content_version

# ---------------------------------------------------------------------------
# Simulation constants (invented, stated in the chapter).
# ---------------------------------------------------------------------------
SIM = {
    "model_call_overhead_s": 0.8,
    "input_s_per_token": 0.00025,
    "output_s_per_token": 0.02,
    "tool_s": {"read_file": 0.05, "run_cases": 0.9, "inspect_interface": 0.3},
}


def approx_tokens(obj):
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return math.ceil(len(text) / 4)


def model_seconds(tin, tout):
    return round(SIM["model_call_overhead_s"] + tin * SIM["input_s_per_token"]
                 + tout * SIM["output_s_per_token"], 3)


# ---------------------------------------------------------------------------
# Artifact revisions of the reporting workspace.
# ---------------------------------------------------------------------------
NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''
BUGGY_LINE = "    return created_at.date() == requested_day"
FIX = ("    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
       "    return local_time.date() == requested_day")
ALT_FIX = ("    local_time = created_at - timedelta(hours=7)\n"
           "    return local_time.date() == requested_day")


def _fixture():
    out = {}
    root = os.path.join(FIXTURES, "report")
    for name in sorted(os.listdir(root)):
        with open(os.path.join(root, name), encoding="utf-8") as f:
            out[name] = f.read()
    return out


def build_revisions():
    """r3 is the task's starting workspace; r4 adds the regression test; r5 applies
    the Chapter 3 fix. r5-alt is a counterfactual patch with a hard-coded offset."""
    r3 = _fixture()
    r4 = dict(r3)
    r4["test_report.py"] = r3["test_report.py"].replace('\n\nif __name__ == "__main__":', "\n" + NEW_TEST, 1)
    r5 = dict(r4)
    r5["report.py"] = r4["report.py"].replace(BUGGY_LINE, FIX, 1)
    alt = dict(r4)
    alt["report.py"] = (r4["report.py"].replace("from zoneinfo import ZoneInfo",
                                                "from datetime import timedelta\nfrom zoneinfo import ZoneInfo", 1)
                        .replace(BUGGY_LINE, ALT_FIX, 1))
    revs = {"r3": r3, "r4": r4, "r5": r5, "r5-alt": alt}
    return {k: {"files": v, "versions": {n: content_version(t) for n, t in v.items()},
                "digest": content_version(json.dumps({n: content_version(t) for n, t in sorted(v.items())}))}
            for k, v in revs.items()}


class RevisionStore:
    def __init__(self):
        self.revs = build_revisions()
        self.dirs = {}

    def path(self, rev):
        if rev not in self.dirs:
            d = tempfile.mkdtemp(prefix=f"itl15-{rev}-")
            for name, text in self.revs[rev]["files"].items():
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(text)
            self.dirs[rev] = d
        return self.dirs[rev]

    def read(self, rev, name):
        return self.revs[rev]["files"][name]

    def close(self):
        for d in self.dirs.values():
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Boundary-case catalog. Expected values follow America/Los_Angeles.
# ---------------------------------------------------------------------------
CASES = {
    "pdt_midday": ("2026-09-11T19:00:00+00:00", "2026-09-11", True, "pdt_midday"),
    "pdt_evening": ("2026-09-12T00:30:00+00:00", "2026-09-11", True, "pdt_evening"),
    "pdt_after_local_midnight": ("2026-09-12T07:30:00+00:00", "2026-09-11", False, "pdt_after_local_midnight"),
    "pst_evening": ("2026-01-12T07:30:00+00:00", "2026-01-11", True, "pst_evening"),
    "dst_fall_back_evening": ("2026-11-02T07:30:00+00:00", "2026-11-01", True, "dst_fall_back_evening"),
    "dst_spring_forward_evening": ("2026-03-08T07:30:00+00:00", "2026-03-07", True, "dst_spring_forward_evening"),
}
BRIEF_CASES = ["pdt_midday", "pdt_evening"]  # the two cases in test_report.py at r4
BOUNDARY_CASES = list(CASES)

CASE_RUNNER = r'''
import json, sys
from datetime import date, datetime
sys.path.insert(0, ".")
from report import belongs_to_report
cases = json.loads(sys.argv[1])
out = []
for c in cases:
    got = belongs_to_report(datetime.fromisoformat(c["utc"]), date.fromisoformat(c["day"]), "America/Los_Angeles")
    out.append({"case": c["case"], "actual": bool(got)})
print(json.dumps(out))
'''


# ---------------------------------------------------------------------------
# Worker runtime: read-only tools over one revision, with an evidence log the
# runtime (not the worker) writes.
# ---------------------------------------------------------------------------
class EvidenceLog:
    def __init__(self):
        self.items = {}

    def record(self, worker, revision, operation, payload, coverage=None):
        body = {"worker": worker, "revision": revision, "operation": operation, "payload": payload}
        eid = "ev-" + content_version(json.dumps(body, sort_keys=True))
        self.items[eid] = dict(body, coverage=coverage, evidence_id=eid)
        return eid


def worker_tools(store, revision, worker, allowed, log, reference_revision=None):
    reg = ToolRegistry()

    def read_file(path):
        text = store.read(revision, path)
        eid = log.record(worker, revision, "read_file", {"path": path, "version": content_version(text)})
        return {"status": "ok", "revision": revision, "path": path, "version": content_version(text),
                "text": text, "evidence_id": eid}

    def run_cases(cases):
        unknown = [c for c in cases if c not in CASES]
        if unknown:
            return {"status": "input_error", "message": f"unknown case {unknown[0]}"}
        payload = [{"case": c, "utc": CASES[c][0], "day": CASES[c][1]} for c in cases]
        env = dict(os.environ, TZ="UTC", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-c", CASE_RUNNER, json.dumps(payload)],
                              cwd=store.path(revision), capture_output=True, text=True, timeout=60, env=env)
        if proc.returncode != 0:
            return {"status": "failed", "stderr": proc.stderr.strip().splitlines()[-1:]}
        results = []
        for row in json.loads(proc.stdout):
            utc, day, expected, cov = CASES[row["case"]]
            rec = {"case": row["case"], "utc": utc, "requested_day": day, "expected": expected,
                   "actual": row["actual"], "ok": row["actual"] == expected}
            rec["evidence_id"] = log.record(worker, revision, "run_cases", {k: rec[k] for k in
                                            ("case", "utc", "requested_day", "expected", "actual", "ok")},
                                            coverage=cov)
            results.append(rec)
        return {"status": "ok", "revision": revision, "results": results}

    def inspect_interface(function):
        def sig(rev):
            tree = ast.parse(store.read(rev, "report.py"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == function:
                    a = node.args
                    return f"{function}({', '.join(x.arg for x in a.args)})"
            return None
        now, ref = sig(revision), sig(reference_revision)
        payload = {"function": function, "revision": revision, "signature": now,
                   "reference_revision": reference_revision, "reference_signature": ref,
                   "unchanged": now == ref}
        eid = log.record(worker, revision, "inspect_interface", payload, coverage="public_interface")
        return dict(payload, status="ok", evidence_id=eid)

    impl = {
        "read_file": ("Read a file of the assigned revision.",
                      {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}, read_file),
        "run_cases": ("Execute named boundary cases against belongs_to_report at the assigned revision.",
                      {"type": "object", "properties": {"cases": {"type": "array", "items": {"type": "string"}}},
                       "required": ["cases"], "additionalProperties": False}, run_cases),
        "inspect_interface": ("Compare a public function signature with the reference revision.",
                              {"type": "object", "properties": {"function": {"type": "string"}},
                               "required": ["function"], "additionalProperties": False}, inspect_interface),
    }
    for name in allowed:
        desc, schema, fn = impl[name]
        reg.register(name, desc, schema, fn)
    return reg


class StandInModel(ScriptedModel):
    """ScriptedModel whose entries may be callables of the messages received."""

    def generate(self, messages, definitions):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": [d["name"] for d in definitions]})
        if not self.responses:
            raise RuntimeError("scripted model has no response left")
        item = self.responses.pop(0)
        return copy.deepcopy(item(messages) if callable(item) else item)


# ---------------------------------------------------------------------------
# Contracts and validation.
# ---------------------------------------------------------------------------
RESULT_SCHEMA = {
    "type": "object",
    "required": ["contract_id", "input_revision", "verdict", "findings", "coverage"],
    "properties": {
        "contract_id": {"type": "string"},
        "input_revision": {"type": "string"},
        "verdict": {"enum": ["pass", "fail", "incomplete"]},
        "findings": {"type": "array", "items": {"type": "object",
                                                "required": ["claim", "evidence"]}},
        "coverage": {"type": "object"},
    },
}


def make_contract(cid, worker, objective, revision, store, allowed, coverage, max_tokens,
                  deadline_s=30.0, reference_revision=None, worker_version="1.0.0"):
    return {
        "contract_id": cid,
        "contract_version": "review-contract/1.1",
        "worker": worker,
        "worker_version": worker_version,
        "objective": objective,
        "input_revision": revision,
        "input_digest": store.revs[revision]["digest"],
        "reference_revision": reference_revision,
        "allowed_operations": list(allowed),
        "required_coverage": list(coverage),
        "budget": {"max_tokens": max_tokens, "max_model_calls": 20, "repair_attempts": 1},
        "deadline_s": deadline_s,
        "result_schema": RESULT_SCHEMA,
    }


def validate_result(contract, raw, log, executions):
    """Return a list of violations. An empty list means the result is admissible."""
    errs = []
    if isinstance(raw, str):
        try:
            res = json.loads(raw)
        except json.JSONDecodeError:
            return ["result is not a JSON object"]
    else:
        res = raw
    if not isinstance(res, dict):
        return ["result is not a JSON object"]
    for field in RESULT_SCHEMA["required"]:
        if field not in res:
            errs.append(f"missing field {field}")
    if errs:
        return errs
    if res["contract_id"] != contract["contract_id"]:
        errs.append("contract_id does not match the contract")
    if res["input_revision"] != contract["input_revision"]:
        errs.append(f"result reviews {res['input_revision']}, contract assigned {contract['input_revision']}")
    if res["verdict"] not in ("pass", "fail", "incomplete"):
        errs.append("verdict must be pass, fail or incomplete")
    used = sorted({e["name"] for e in executions})
    extra = [u for u in used if u not in contract["allowed_operations"]]
    if extra:
        errs.append(f"operation {extra[0]} not allowed by contract")
    cited = []
    for f in res["findings"]:
        if not f.get("evidence"):
            errs.append(f"finding without evidence: {f.get('claim', '')[:40]}")
        cited += f.get("evidence", [])
    for cls, ids in res["coverage"].items():
        cited += ids
    for eid in cited:
        item = log.items.get(eid)
        if item is None:
            errs.append(f"evidence {eid} was not recorded by the runtime")
        elif item["worker"] != contract["worker"] or item["revision"] != contract["input_revision"]:
            errs.append(f"evidence {eid} belongs to {item['worker']} at {item['revision']}")
    for cls, ids in res["coverage"].items():
        for eid in ids:
            item = log.items.get(eid)
            if item is not None and item["coverage"] != cls:
                errs.append(f"evidence {eid} does not cover {cls}")
    if res["verdict"] == "pass":
        missing = [c for c in contract["required_coverage"] if not res["coverage"].get(c)]
        if missing:
            errs.append("pass without evidence for " + ", ".join(missing))
    if res["verdict"] == "fail" and not res["findings"]:
        errs.append("fail verdict without findings")
    return sorted(set(errs), key=errs.index)


# ---------------------------------------------------------------------------
# Scripted reviewers. Each composes its final JSON from its own observations.
# ---------------------------------------------------------------------------
def _call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def _observations(messages):
    return [m["content"] for m in messages if m.get("role") == "tool"]


def compose_result(contract):
    """Deterministic stand-in for the model's summary: read observations, write the result."""
    def fn(messages):
        findings, coverage = [], {}
        for obs in _observations(messages):
            for r in obs.get("results", []):
                cov = CASES[r["case"]][3]
                coverage.setdefault(cov, []).append(r["evidence_id"])
                if not r["ok"]:
                    findings.append({"claim": f"{r['case']}: order at {r['utc']} expected "
                                              f"{r['expected']} for {r['requested_day']}, got {r['actual']}",
                                     "severity": "defect", "evidence": [r["evidence_id"]]})
            if obs.get("function"):
                coverage.setdefault("public_interface", []).append(obs["evidence_id"])
                if not obs["unchanged"]:
                    findings.append({"claim": f"signature changed: {obs['reference_signature']} -> "
                                              f"{obs['signature']}", "severity": "defect",
                                     "evidence": [obs["evidence_id"]]})
                else:
                    findings.append({"claim": f"signature unchanged: {obs['signature']}",
                                     "severity": "info", "evidence": [obs["evidence_id"]]})
        defects = [f for f in findings if f["severity"] == "defect"]
        result = {"contract_id": contract["contract_id"], "input_revision": contract["input_revision"],
                  "verdict": "fail" if defects else "pass", "findings": findings, "coverage": coverage}
        return {"role": "assistant", "content": json.dumps(result), "tool_calls": []}
    return fn


def prose_result(text):
    return lambda messages: {"role": "assistant", "content": text, "tool_calls": []}


def fabricated_result(contract):
    def fn(messages):
        res = json.loads(compose_result(contract)(messages)["content"])
        res["coverage"] = {c: ["ev-allcasespass"] for c in contract["required_coverage"]}
        return {"role": "assistant", "content": json.dumps(res), "tool_calls": []}
    return fn


def script_tz(contract, cases, final=None):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            _call("t1", "read_file", path="report.py"), _call("t2", "read_file", path="settings.json")]},
        {"role": "assistant", "content": "", "tool_calls": [_call("t3", "run_cases", cases=cases)]},
        final or compose_result(contract),
    ]


def script_iface(contract, final=None):
    return [
        {"role": "assistant", "content": "", "tool_calls": [_call("i1", "read_file", path="report.py")]},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("i2", "inspect_interface", function="belongs_to_report")]},
        final or compose_result(contract),
    ]


def script_combined(contract, final=None):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            _call("s1", "read_file", path="report.py"), _call("s2", "read_file", path="settings.json")]},
        {"role": "assistant", "content": "", "tool_calls": [_call("s3", "run_cases", cases=BOUNDARY_CASES)]},
        {"role": "assistant", "content": "", "tool_calls": [
            _call("s4", "inspect_interface", function="belongs_to_report")]},
        final or compose_result(contract),
    ]


def script_runaway(contract, n=24):
    r = []
    for i in range(n):
        r.append({"role": "assistant", "content": "", "tool_calls": [
            _call(f"x{i}a", "read_file", path="report.py"), _call(f"x{i}b", "read_file", path="test_report.py")]})
    r.append(compose_result(contract))
    return r


WORKER_SYSTEM = ("You are a read-only reviewer working under a contract. Use only the allowed operations "
                 "on the assigned revision. Return exactly one JSON object that matches result_schema, "
                 "citing the evidence_id of every observation you rely on.")


def run_worker(contract, script, store, log, repair_script=None, extra_repair=None):
    """Run one worker loop with metering, validation and a bounded repair allowance."""
    tools = worker_tools(store, contract["input_revision"], contract["worker"], contract["allowed_operations"],
                         log, contract.get("reference_revision"))
    model = StandInModel(script + (repair_script or []))
    messages = [{"role": "system", "content": WORKER_SYSTEM},
                {"role": "user", "content": json.dumps({"contract": contract})}]
    ledger = {"model_calls": 0, "tokens_in": 0, "tokens_out": 0, "sim_seconds": 0.0, "steps": []}
    attempts = []
    repairs_left = contract["budget"]["repair_attempts"]
    status = None
    raw = None
    while True:
        if ledger["model_calls"] >= contract["budget"]["max_model_calls"]:
            status = "stopped: model-call limit"
            break
        if not model.responses:
            status = "stopped: no response"
            break
        tin = approx_tokens(messages)
        if ledger["tokens_in"] + ledger["tokens_out"] + tin > contract["budget"]["max_tokens"]:
            status = "stopped: token budget"
            ledger["steps"].append({"kind": "refused_model_call", "tokens_in": tin})
            break
        reply = model.generate(messages, tools.definitions())
        tout = approx_tokens(reply)
        ledger["model_calls"] += 1
        ledger["tokens_in"] += tin
        ledger["tokens_out"] += tout
        secs = model_seconds(tin, tout)
        ledger["sim_seconds"] = round(ledger["sim_seconds"] + secs, 3)
        ledger["steps"].append({"kind": "model", "tokens_in": tin, "tokens_out": tout, "seconds": secs})
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if calls:
            for c in calls:
                obs = tools.execute_checked(c)
                t = SIM["tool_s"].get(c["name"], 0.0)
                ledger["sim_seconds"] = round(ledger["sim_seconds"] + t, 3)
                ledger["steps"].append({"kind": "tool", "name": c["name"], "seconds": t})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
            continue
        raw = reply["content"]
        errs = validate_result(contract, raw, log, tools.executions)
        attempts.append({"attempt": len(attempts) + 1, "raw_excerpt": raw[:160], "violations": errs})
        if not errs:
            status = "accepted"
            break
        if repairs_left <= 0:
            status = "rejected"
            break
        repairs_left -= 1
        messages.append({"role": "user", "content": "Result rejected by the coordinator: " + "; ".join(errs)
                         + ". Return one JSON object matching result_schema."})
    total = ledger["tokens_in"] + ledger["tokens_out"]
    ledger["tokens_total"] = total
    result = None
    if status == "accepted":
        result = raw if isinstance(raw, dict) else json.loads(raw)
    over_deadline = ledger["sim_seconds"] > contract["deadline_s"]
    if over_deadline and status == "accepted":
        status = "late"
    return {"contract_id": contract["contract_id"], "worker": contract["worker"],
            "worker_version": contract["worker_version"], "contract_version": contract["contract_version"],
            "input_revision": contract["input_revision"], "status": status, "attempts": attempts,
            "result": result, "ledger": ledger, "executions": tools.executions,
            "trace": model.requests}


# ---------------------------------------------------------------------------
# Integration.
# ---------------------------------------------------------------------------
def integrate(outcomes, current_revision, required_coverage):
    """Admit accepted results for the current revision; classify everything else."""
    admitted, stale, rejected = [], [], []
    coverage = {}
    for o in outcomes:
        if o["status"] not in ("accepted",):
            rejected.append({"worker": o["worker"], "status": o["status"]})
            continue
        if o["input_revision"] != current_revision:
            stale.append({"worker": o["worker"], "reviewed": o["input_revision"], "current": current_revision,
                          "verdict": o["result"]["verdict"],
                          "reclassified_as": f"evidence about {o['input_revision']}"})
            continue
        admitted.append(o)
        for cls, ids in o["result"]["coverage"].items():
            coverage.setdefault(cls, set()).update(ids)
    missing = [c for c in required_coverage if not coverage.get(c)]
    defects = [f for o in admitted for f in o["result"]["findings"] if f.get("severity") == "defect"]
    if defects:
        verdict = "defects found"
    elif missing:
        verdict = "not verified: missing " + ", ".join(missing)
    else:
        verdict = "verified"
    return {"current_revision": current_revision, "verdict": verdict,
            "admitted": [o["worker"] for o in admitted], "stale": stale, "rejected": rejected,
            "missing_coverage": missing, "defects": [d["claim"] for d in defects],
            "coverage": {k: sorted(v) for k, v in coverage.items()}}


def naive_integrate(outcomes):
    """Flawed coordinator: counts approvals and ignores revisions and coverage."""
    votes = {"pass": 0, "fail": 0}
    for o in outcomes:
        if o["result"]:
            votes[o["result"]["verdict"]] = votes.get(o["result"]["verdict"], 0) + 1
    return {"votes": votes, "verdict": "approved" if votes["pass"] > votes["fail"] else "rejected"}


# ---------------------------------------------------------------------------
# Shared state policies.
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self, policy):
        self.policy = policy
        self.values = {}
        self.step_writes = {}
        self.log = []

    def begin_step(self):
        self.step_writes = {}

    def write(self, writer, key, value):
        if self.policy == "namespaced":
            key = f"results/{writer}"
        if self.policy == "reject_second_write" and key in self.step_writes:
            self.log.append({"writer": writer, "key": key, "outcome": "rejected",
                             "reason": f"already written by {self.step_writes[key]} in this step"})
            return False
        prev = self.values.get(key)
        self.values[key] = value
        self.step_writes[key] = writer
        self.log.append({"writer": writer, "key": key,
                         "outcome": "overwrote " + prev["worker"] if prev else "written"})
        return True


# ---------------------------------------------------------------------------
# Schedules.
# ---------------------------------------------------------------------------
def schedule(tasks):
    """tasks: {name: (duration, [deps])}. Unlimited capacity earliest-start schedule."""
    start, finish = {}, {}
    pending = dict(tasks)
    while pending:
        for name, (dur, deps) in list(pending.items()):
            if all(d in finish for d in deps):
                start[name] = max([finish[d] for d in deps], default=0.0)
                finish[name] = round(start[name] + dur, 3)
                del pending[name]
    sequential = round(sum(d for d, _ in tasks.values()), 3)
    return {"sequential_s": sequential, "parallel_s": max(finish.values()),
            "tasks": {n: {"start": start[n], "finish": finish[n]} for n in tasks}}
