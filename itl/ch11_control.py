"""Chapter 11 helpers: three controllers over the reporting task family.

StandInModel     a deterministic, rule-based stand-in for a model. It chooses
                 the next action only from the messages it receives. Each
                 scenario sets a behavior ("competent", "hasty", "repeat",
                 "malformed", "wander"). It shows how controllers respond to
                 a given behavior; it says nothing about how often a live
                 model behaves that way.
run_workflow     fixed workflow: the program chooses every operation and asks
                 the model only for a regression test and a patch.
run_loop         the Chapter 3 loop (itl.core.run_agent), unchanged.
run_hybrid       required stages inspect -> reproduce -> fix -> verify with
                 model-chosen actions inside them, an evidence-based progress
                 detector, separate budgets, a repair allowance, an evidence
                 plan and distinct termination outcomes.
"""
import ast
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from itl.core import Workspace, report_tools, run_agent, content_version

# ---------------------------------------------------------------- task text

SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: {files}.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")
TASK_DEPLOY = ("The daily report is missing some orders placed near midnight. Find the cause, "
               "prepare a fix and run the tests. Deploy it to production once they pass.")

NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''
TEST_ANCHOR = '\n\nif __name__ == "__main__":'

DEFECT_LINE = "    return created_at.date() == requested_day"
GOOD_FIX = ("    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
            "    return local_time.date() == requested_day")
WRONG_FIX = '    return created_at.astimezone(ZoneInfo("UTC")).date() == requested_day'
HELPER_DEFECT = "    return moment.astimezone(timezone.utc).date()"
HELPER_FIX = "    return moment.date()"
COMMENT_EDIT = "    # Compare calendar dates.\n" + DEFECT_LINE

ANSWER = ("The filter used the UTC calendar date. I changed it to convert each timestamp to the "
          "configured report timezone before comparing dates. The new near-midnight test failed "
          "before the fix; both tests now pass.")
HASTY_ANSWER = ("Fixed: the comparison now converts the timestamp before taking its date. "
                "Both tests pass.")

# ------------------------------------------------------------ normalization


def norm_path(p):
    return os.path.normpath(p).replace(os.sep, "/")


def op_identity(call):
    """Normalized operation identity: tool name plus canonical arguments."""
    args = call.get("arguments")
    if not isinstance(args, dict):
        return f"{call['name']}({args!r})"
    canon = {}
    for k, v in sorted(args.items()):
        if k == "path" and isinstance(v, str):
            v = norm_path(v)
        elif k == "tests" and isinstance(v, list):
            v = sorted(norm_path(t) for t in v if isinstance(t, str))
        elif isinstance(v, str):
            v = " ".join(v.split())
        canon[k] = v
    digest = hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:8]
    shown = canon.get("path") or canon.get("tests") or canon.get("environment") or ""
    return f"{call['name']}:{shown}:{digest}"


def semantic_version(path, text):
    """Revision that ignores comments and layout: AST for Python, parsed JSON for JSON."""
    try:
        if path.endswith(".py"):
            return content_version(ast.dump(ast.parse(text)))
        if path.endswith(".json"):
            return content_version(json.dumps(json.loads(text), sort_keys=True))
    except (SyntaxError, ValueError):
        pass
    return content_version(text)


def estimate_tokens(obj):
    return len(json.dumps(obj, ensure_ascii=False)) // 4


# ------------------------------------------------------------- stand-in model

FILE_BLOCK = re.compile(r"=== (\S+) ===\n(.*?)\n=== end ===", re.S)


class StandInModel:
    """Chooses the next action from its input only. Records every request."""

    def __init__(self, behavior, clock=None):
        self.behavior = behavior
        self.requests = []
        self.clock = clock

    # -- reading the input
    def _parse(self, messages):
        calls, history, seen, controller = {}, [], {}, []
        files = []
        for m in messages:
            if m["role"] == "system":
                mm = re.search(r"Workspace files: (.*)\.$", m["content"])
                if mm:
                    files = [f.strip() for f in mm.group(1).split(",")]
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    calls[c["id"]] = c
            if m["role"] == "tool":
                c = calls.get(m["tool_call_id"])
                r = m["content"]
                history.append((c, r))
                if c and c["name"] == "read_file" and r.get("status") == "ok":
                    seen[norm_path(r["path"])] = r["text"]
            if m["role"] == "user" and isinstance(m["content"], str):
                for path, text in FILE_BLOCK.findall(m["content"]):
                    seen[norm_path(path)] = text
                if m["content"].startswith(("Controller:", "Stage:")):
                    controller.append(m["content"])
        task = next(m["content"] for m in messages if m["role"] == "user")
        return {"calls": calls, "history": history, "seen": seen, "files": files,
                "controller": controller, "task": task}

    def generate(self, messages, definitions):
        self.requests.append({"messages": copy.deepcopy(messages),
                              "tools": [d["name"] for d in definitions]})
        st = self._parse(messages)
        last = messages[-1]
        if last["role"] == "user" and last["content"].startswith("Controller: stopping"):
            return self._summary(st)
        if last["role"] == "user" and last["content"].startswith("Stage:"):
            return self._stage_reply(st, last["content"])
        return self._next_action(st)

    # -- facts derived from the input
    @staticmethod
    def _edits(st):
        return [(c, r) for c, r in st["history"]
                if c and c["name"] == "replace_text" and r.get("status") == "ok"]

    @staticmethod
    def _runs(st):
        return [i for i, (c, r) in enumerate(st["history"]) if "tests_run" in r]

    def _test_added(self, st):
        return any(norm_path(c["arguments"]["path"]) == "test_report.py" for c, _ in self._edits(st))

    def _need_run(self, st):
        edits = [i for i, (c, r) in enumerate(st["history"])
                 if c and c["name"] == "replace_text" and r.get("status") == "ok"]
        runs = self._runs(st)
        return bool(edits) and (not runs or edits[-1] > runs[-1])

    def _last_run(self, st):
        runs = self._runs(st)
        return st["history"][runs[-1]][1] if runs else None

    def _applied(self, st, old):
        return any(c["arguments"]["old"] == old for c, _ in self._edits(st))

    def _choose_fix(self, st):
        """Pick a patch from visible evidence. None when the defect is not visible."""
        report = st["seen"].get("report.py", "")
        helper = st["seen"].get("timeutil.py", "")
        if DEFECT_LINE in report:
            if self.behavior == "repeat":
                return ("report.py", DEFECT_LINE, COMMENT_EDIT)
            if self.behavior == "hasty" and not self._applied(st, DEFECT_LINE):
                return ("report.py", DEFECT_LINE, WRONG_FIX)
            if not self._applied(st, DEFECT_LINE):
                return ("report.py", DEFECT_LINE, GOOD_FIX)
            if self._applied(st, DEFECT_LINE) and not self._applied(st, WRONG_FIX):
                wrong = any(c["arguments"]["new"] == WRONG_FIX for c, _ in self._edits(st))
                if wrong:
                    return ("report.py", WRONG_FIX, GOOD_FIX)
        if HELPER_DEFECT in helper and not self._applied(st, HELPER_DEFECT):
            return ("timeutil.py", HELPER_DEFECT, HELPER_FIX)
        return None

    def _unread_imports(self, st):
        out = []
        for path, text in st["seen"].items():
            if not path.endswith(".py"):
                continue
            for mod in re.findall(r"^from (\w+) import", text, re.M):
                cand = mod + ".py"
                if cand in st["files"] and cand not in st["seen"]:
                    out.append(cand)
        return sorted(set(out))

    # -- replies
    def _reply(self, text="", calls=()):
        n = len(self.requests)
        tool_calls = [{"id": f"m{n}.{i + 1}", "name": name, "arguments": args}
                      for i, (name, args) in enumerate(calls)]
        return {"role": "assistant", "content": text, "tool_calls": tool_calls}

    def _summary(self, st):
        """Summarize from the compact stop request: plan statuses and evidence lines."""
        text = st["controller"][-1]
        plan = json.loads(re.search(r"Plan: (\{.*\})", text).group(1))
        tests = re.findall(r"e\d+ tests\s+(\w+)", text)
        done = [k for k, v in plan.items() if v == "satisfied"]
        todo = [k for k, v in plan.items() if v != "satisfied"]
        return self._reply(f"Stopped before completion. Established: {', '.join(done) or 'nothing'}. "
                           f"Open: {', '.join(todo)}. Last recorded test result: "
                           f"{tests[-1] if tests else 'none'}.")

    def _stage_reply(self, st, prompt):
        stage = prompt.split()[1].rstrip(".")
        if stage == "reproduce":
            return self._reply(calls=[("replace_text", {"path": "test_report.py", "old": TEST_ANCHOR,
                                                         "new": "\n" + NEW_TEST})])
        if stage == "fix":
            # the workflow puts earlier patches in the prompt as "Applied patch" lines
            applied = re.findall(r"Applied patch: replaced (.*)$", prompt, re.M)
            fix = self._choose_fix_given(st, applied)
            if fix is None:
                return self._reply("The defect is not visible in the files provided. report.py already "
                                   "converts to the report timezone; calendar_day in timeutil.py needs "
                                   "inspection.")
            path, old, new = fix
            return self._reply(calls=[("replace_text", {"path": path, "old": old, "new": new})])
        return self._reply("No action for this stage.")

    def _choose_fix_given(self, st, applied):
        report = st["seen"].get("report.py", "")
        if DEFECT_LINE in report and self.behavior == "repeat":
            return ("report.py", DEFECT_LINE, COMMENT_EDIT)
        if WRONG_FIX in report:
            return ("report.py", WRONG_FIX, GOOD_FIX)
        if DEFECT_LINE in report:
            if self.behavior == "hasty" and not applied:
                return ("report.py", DEFECT_LINE, WRONG_FIX)
            return ("report.py", DEFECT_LINE, GOOD_FIX)
        helper = st["seen"].get("timeutil.py", "")
        if HELPER_DEFECT in helper:
            return ("timeutil.py", HELPER_DEFECT, HELPER_FIX)
        return None

    def _next_action(self, st):
        b = self.behavior
        told = st["controller"]
        compliant = any("refused" in t for t in told)
        told_verify = any("verification" in t for t in told)
        # 1. inspect
        if b == "wander":
            unread = [f for f in st["files"] if f not in st["seen"]]
            if unread:
                return self._reply(calls=[("read_file", {"path": unread[0]})])
        if "report.py" not in st["seen"]:
            return self._reply(calls=[("read_file", {"path": p})
                                      for p in ("settings.json", "report.py", "test_report.py")])
        unread = self._unread_imports(st)
        if unread:
            return self._reply(calls=[("read_file", {"path": p}) for p in unread])
        # 2. reproduce (a hasty model skips this until the controller refuses an edit)
        if not self._test_added(st) and (b != "hasty" or compliant):
            return self._reply(calls=[("replace_text", {"path": "test_report.py", "old": TEST_ANCHOR,
                                                         "new": "\n" + NEW_TEST})])
        if b == "malformed" and self._test_added(st):
            return self._reply(calls=[("run_tests", {"tests": "test_report.py"})])
        if self._need_run(st):
            last_edit = self._edits(st)[-1][0]
            source_edit = norm_path(last_edit["arguments"]["path"]) != "test_report.py"
            if b == "hasty" and source_edit and not told_verify:
                return self._reply(HASTY_ANSWER)
            return self._reply(calls=[("run_tests", {"tests": ["test_report.py"]})])
        run = self._last_run(st)
        last_tool = st["history"][-1][0]["name"] if st["history"] and st["history"][-1][0] else ""
        if b == "repeat" and run and run["status"] == "failed" and last_tool == "run_tests":
            return self._reply(calls=[("read_file", {"path": "./report.py"})])
        # 3. fix
        if run and run["status"] == "failed":
            fix = self._choose_fix(st)
            if fix:
                path, old, new = fix
                return self._reply(calls=[("replace_text", {"path": path, "old": old, "new": new})])
        if b == "hasty" and not compliant and not self._edits(st):
            path, old, new = self._choose_fix(st)
            return self._reply(calls=[("replace_text", {"path": path, "old": old, "new": new})])
        # 4. after a passing run
        if run and run["status"] == "passed":
            if "Deploy it" in st["task"] and not any(c and c["name"] == "deploy" for c, _ in st["history"]):
                return self._reply(calls=[("deploy", {"environment": "production"})])
            return self._reply(ANSWER)
        return self._reply(HASTY_ANSWER if b == "hasty" else ANSWER)


# -------------------------------------------------------------- simulation


class SimClock:
    """Simulated wall clock. Model latency grows with input size; tools have fixed costs."""
    TOOL_SECONDS = {"read_file": 0.2, "replace_text": 0.3, "run_tests": 6.0, "deploy": 5.0}

    def __init__(self):
        self.t = 0.0

    def model(self, in_tokens):
        self.t = round(self.t + 1.5 + in_tokens / 1500, 3)

    def tool(self, name):
        self.t = round(self.t + self.TOOL_SECONDS.get(name, 0.5), 3)


def deploy_tool(reg, ws):
    def deploy(environment):
        ws.write("DEPLOYED.txt", f"deployed to {environment}\n")
        return {"status": "ok", "environment": environment}
    reg.register("deploy", "Deploy the current workspace revision.", {
        "type": "object", "properties": {"environment": {"type": "string"}},
        "required": ["environment"], "additionalProperties": False}, deploy)


def make_env(fixture, scenario):
    ws = Workspace(fixture)
    tools = report_tools(ws)
    if scenario["deploy"]:
        deploy_tool(tools, ws)
    files = sorted(n for n in os.listdir(ws.root) if n.endswith((".py", ".json")))
    messages = [{"role": "system", "content": SYSTEM.format(files=", ".join(files))},
                {"role": "user", "content": TASK_DEPLOY if scenario["deploy"] else TASK}]
    return ws, tools, messages


def workspace_semantic(ws):
    snap = ws.snapshot()
    return {p: semantic_version(p, t) for p, t in snap.items()}


def audit(ws):
    """Independent check after the run: an acceptance test the agent never saw."""
    tmp = tempfile.mkdtemp(prefix="itl-audit-")
    try:
        for n in os.listdir(ws.root):
            if n.endswith(".py") or n.endswith(".json"):
                shutil.copy(os.path.join(ws.root, n), tmp)
        with open(os.path.join(tmp, "test_acceptance.py"), "w") as f:
            f.write(ACCEPTANCE)
        proc = subprocess.run([sys.executable, "-m", "unittest", "test_acceptance"], cwd=tmp,
                              capture_output=True, text=True, timeout=60)
        return {"acceptance_passed": proc.returncode == 0,
                "deployed": os.path.exists(os.path.join(ws.root, "DEPLOYED.txt"))}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


ACCEPTANCE = '''import unittest
from datetime import date, datetime
from report import belongs_to_report


class Acceptance(unittest.TestCase):
    def test_boundaries(self):
        tz = "America/Los_Angeles"
        cases = [("2026-09-12T06:59:00+00:00", date(2026, 9, 11), True),
                 ("2026-09-12T07:01:00+00:00", date(2026, 9, 11), False),
                 ("2026-09-11T19:00:00+00:00", date(2026, 9, 11), True)]
        for ts, day, expected in cases:
            self.assertEqual(belongs_to_report(datetime.fromisoformat(ts), day, tz), expected, ts)
'''


class Meter:
    """Wraps a model: advances the simulated clock and counts tokens."""

    def __init__(self, model, clock):
        self.model, self.clock = model, clock
        self.calls = 0
        self.tokens = 0

    @property
    def requests(self):
        return self.model.requests

    def generate(self, messages, definitions):
        n = estimate_tokens(messages) + estimate_tokens(definitions)
        self.calls += 1
        self.tokens += n
        self.clock.model(n)
        return self.model.generate(messages, definitions)


# ---------------------------------------------------------- 1. fixed workflow

WORKFLOW_FILES = ["settings.json", "report.py", "test_report.py"]


def run_workflow(scenario, fixture):
    ws, tools, messages = make_env(fixture, scenario)
    clock = SimClock()
    model = Meter(StandInModel(scenario["behavior"]), clock)
    events = []
    edit_defs = [d for d in tools.definitions() if d["name"] == "replace_text"]
    cancel_at = scenario.get("cancel_at")

    def ex(name, **args):
        call = {"id": f"w{len(tools.executions) + 1}", "name": name, "arguments": args}
        out = tools.execute_checked(call)
        clock.tool(name)
        events.append({"t": clock.t, "decided_by": "program", "op": op_identity(call),
                       "status": out.get("status")})
        return out

    def cancelled():
        return cancel_at is not None and clock.t >= cancel_at

    def finish(outcome, reason):
        return ws, {"controller": "workflow", "outcome": outcome, "reason": reason,
                    "model_calls": model.calls, "tokens": model.tokens, "sim_seconds": clock.t,
                    "executions": len(tools.executions)}, model, events, messages

    # stage 1: inspect (program)
    files = {p: ex("read_file", path=p)["text"] for p in WORKFLOW_FILES}
    block = "\n".join(f"=== {p} ===\n{t}\n=== end ===" for p, t in files.items())
    if cancelled():
        return finish("cancelled", "cancellation observed before stage reproduce")
    # stage 2: reproduce (model writes the test, program applies and runs it)
    messages.append({"role": "user", "content": "Stage: reproduce. Propose one replace_text call that "
                     "adds a regression test to test_report.py.\n" + block})
    reply = model.generate(messages, edit_defs)
    messages.append(reply)
    events.append({"t": clock.t, "decided_by": "model", "op": "propose regression test"})
    for c in reply["tool_calls"][:1]:
        ex("replace_text", **c["arguments"])
    first = ex("run_tests", tests=["test_report.py"])
    if first["status"] != "failed":
        return finish("stages-exhausted", "regression test did not fail")
    if cancelled():
        return finish("cancelled", "cancellation observed before stage fix")
    # stage 3+4: fix then verify, one retry with the failure output
    applied = []
    for attempt in (1, 2):
        text = {p: ws.read(p) for p in WORKFLOW_FILES}
        block = "\n".join(f"=== {p} ===\n{t}\n=== end ===" for p, t in text.items())
        lines = "".join(f"Applied patch: replaced {a}\n" for a in applied)
        run = first if attempt == 1 else last
        messages.append({"role": "user", "content": f"Stage: fix. Attempt {attempt}. Failing tests: "
                         f"{run['failed_tests']}.\n{lines}{block}"})
        reply = model.generate(messages, edit_defs)
        messages.append(reply)
        events.append({"t": clock.t, "decided_by": "model", "op": f"propose patch (attempt {attempt})"})
        if not reply["tool_calls"]:
            return finish("stages-exhausted", "no patch proposed: " + reply["content"][:60])
        args = reply["tool_calls"][0]["arguments"]
        ex("replace_text", **args)
        applied.append(args["old"].strip()[:40])
        if cancelled():
            return finish("cancelled", "cancellation observed before stage verify")
        last = ex("run_tests", tests=["test_report.py"])
        if last["status"] == "passed":
            return finish("completed-verified", f"tests passed after attempt {attempt}")
        if cancelled():
            return finish("cancelled", "cancellation observed after verify")
    return finish("stages-exhausted", "verification failed after 2 patch attempts")


# ------------------------------------------------------ 2. model-directed loop


def run_loop(scenario, fixture, limit=12):
    ws, tools, messages = make_env(fixture, scenario)
    clock = SimClock()
    model = Meter(StandInModel(scenario["behavior"]), clock)
    events = []
    original = tools.execute_checked

    def timed(call):
        out = original(call)
        clock.tool(call["name"])
        events.append({"t": clock.t, "decided_by": "model", "op": op_identity(call),
                       "status": out.get("status")})
        return out
    tools.execute_checked = timed
    try:
        answer = run_agent(model, tools, messages, limit=limit)
        outcome, reason = "model-stopped", answer[:70]
    except RuntimeError as exc:
        outcome, reason = "limit-reached", str(exc)
    return ws, {"controller": "loop", "outcome": outcome, "reason": reason, "model_calls": model.calls,
                "tokens": model.tokens, "sim_seconds": clock.t,
                "executions": len(tools.executions)}, model, events, messages


# ---------------------------------------------------------------- 3. hybrid

LIMITS = {"model_calls": 16, "tokens": 30000, "sim_seconds": 240.0,
          "tool:run_tests": 4, "tool:replace_text": 8, "tool:read_file": 20, "tool:deploy": 1}
RESERVE_MODEL_CALLS = 1
RESERVE_TOKENS = 1000
STALL_WINDOW = 3
REPAIR_ALLOWANCE = 2
APPROVAL_REQUIRED = {"deploy"}


class Evidence:
    """Append-only evidence log. Each record has a key; a key seen before is not new."""

    def __init__(self, semantic=True):
        self.records = []
        self.keys = set()
        self.semantic = semantic

    def version(self, path, text):
        return semantic_version(path, text) if self.semantic else content_version(text)

    def add(self, kind, step, space=None, **data):
        key = json.dumps([space or kind, data.get("key")], sort_keys=True)
        new = key not in self.keys
        self.keys.add(key)
        rec = {"id": f"e{len(self.records) + 1}", "kind": kind, "step": step, "new": new, **data}
        rec.pop("key", None)
        self.records.append(rec)
        return rec


class Plan:
    """Plan items carry evidence requirements. Status is computed from evidence only."""

    def __init__(self, deploy, baseline_tests):
        self.baseline = baseline_tests
        self.items = [
            {"id": "P1", "stage": "inspect", "claim": "Reporting timezone established",
             "requires": "read of settings.json"},
            {"id": "P2", "stage": "reproduce", "claim": "Failure reproduced",
             "requires": "test run in which a test outside the baseline fails"},
            {"id": "P3", "stage": "fix", "claim": "Source changed after reproduction",
             "requires": "edit that changes the semantic revision of a non-test file, after P2"},
            {"id": "P4", "stage": "verify", "claim": "Fix verified",
             "requires": "passing run on the current semantic revision, including the P2 test"},
            ({"id": "P5", "stage": "release", "claim": "Deployment approved",
              "requires": "approval record bound to the verified revision"} if deploy else
             {"id": "P5", "stage": "constraint", "claim": "No deployment executed",
              "requires": "no deploy execution in the record"}),
        ]
        self.deploy = deploy
        for it in self.items:
            it["status"], it["evidence"] = "pending", None

    def update(self, ev, current_sem, approvals):
        recs = ev.records
        st = {it["id"]: it for it in self.items}

        def set_(i, ok, ref):
            st[i]["status"], st[i]["evidence"] = ("satisfied", ref) if ok else ("pending", None)
        r1 = next((r for r in recs if r["kind"] == "read" and r["path"] == "settings.json"), None)
        set_("P1", r1, r1 and r1["id"])
        r2 = next((r for r in recs if r["kind"] == "tests" and
                   any(t.split(".")[-1] not in self.baseline for t in r["failed"])), None)
        set_("P2", r2, r2 and r2["id"])
        r3 = None
        if r2:
            r3 = next((r for r in reversed(recs) if r["kind"] == "edit" and r["source"] and
                       r["semantic_changed"] and recs.index(r) > recs.index(r2)), None)
        set_("P3", r3, r3 and r3["id"])
        r4 = None
        if r2 and r3:
            r4 = next((r for r in reversed(recs) if r["kind"] == "tests" and r["status"] == "passed"
                       and r["workspace"] == current_sem and recs.index(r) > recs.index(r3)), None)
        set_("P4", r4, r4 and r4["id"])
        if self.deploy:
            a = next((x for x in approvals if r4 and x["bound_revision"] == current_sem), None)
            set_("P5", a, a and a["id"])
        else:
            d = next((r for r in recs if r["kind"] == "deploy"), None)
            set_("P5", not d, None if d else "absence checked")
        return self.items

    def complete(self):
        return all(it["status"] == "satisfied" for it in self.items)

    def stage(self):
        return next((it["stage"] for it in self.items if it["status"] != "satisfied"), "done")

    def pending(self):
        return [f"{it['id']} ({it['claim']}): requires {it['requires']}"
                for it in self.items if it["status"] != "satisfied"]


def controller_definition(deploy):
    """Everything that shapes the hybrid's decisions, hashed so each run names the version it ran."""
    spec = {"limits": LIMITS, "reserve_model_calls": RESERVE_MODEL_CALLS, "reserve_tokens": RESERVE_TOKENS,
            "stall_window": STALL_WINDOW, "repair_allowance": REPAIR_ALLOWANCE,
            "approval_required": sorted(APPROVAL_REQUIRED),
            "plan": [(it["id"], it["stage"], it["requires"]) for it in Plan(deploy, set()).items]}
    return {"digest": content_version(json.dumps(spec, sort_keys=True)), "spec": spec}


def baseline_tests(ws):
    tree = ast.parse(ws.read("test_report.py"))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")}


def run_hybrid(scenario, fixture, semantic=True, limits=None):
    limits = dict(LIMITS, **(limits or {}))
    ws, tools, messages = make_env(fixture, scenario)
    clock = SimClock()
    model = Meter(StandInModel(scenario["behavior"]), clock)
    ev = Evidence(semantic=semantic)
    plan = Plan(scenario["deploy"], baseline_tests(ws))
    initial_sem = {p: ev.version(p, t) for p, t in ws.snapshot().items()}
    used = {k: 0 for k in limits}
    repairs, stall, events, approvals, pending_approval = {}, 0, [], [], None
    cancel_at = scenario.get("cancel_at")
    defs = tools.definitions()

    def current_sem():
        return {p: ev.version(p, t) for p, t in ws.snapshot().items() if p != "DEPLOYED.txt"}

    def budgets():
        return {k: {"used": used[k], "limit": limits[k]} for k in limits}

    def stop(outcome, reason, summarize=True):
        summary = None
        plan.update(ev, current_sem(), approvals)
        if summarize and used["model_calls"] < limits["model_calls"]:
            # The reserved call gets a compact input built from records, not the transcript.
            facts = [f"{r['id']} {r['kind']} {r.get('path', '')} {r.get('status', '')}".strip()
                     for r in ev.records]
            compact = [messages[0], messages[1],
                       {"role": "user", "content": f"Controller: stopping ({outcome}: {reason}). "
                        "Summarize what is established and what remains, without tools.\n"
                        "Plan: " + json.dumps({it["id"]: it["status"] for it in plan.items}) +
                        "\nEvidence: " + "; ".join(facts)}]
            n = estimate_tokens(compact)
            reply = model.generate(compact, [])
            used["model_calls"] += 1
            used["tokens"] += n
            summary = {"input_tokens": n, "text": reply["content"]}
            messages.append({"role": "user", "content": compact[-1]["content"]})
            messages.append(reply)
        rec = {"controller": "hybrid" if semantic else "hybrid-raw-revisions",
               "controller_definition": controller_definition(scenario["deploy"])["digest"],
               "outcome": outcome,
               "reason": reason, "model_calls": model.calls, "tokens": model.tokens,
               "sim_seconds": clock.t, "executions": len(tools.executions), "budgets": budgets(),
               "stage_at_end": plan.stage(), "plan": copy.deepcopy(plan.items),
               "pending_approval": pending_approval, "summary_from_model": summary}
        return ws, rec, model, events, messages

    step = 0
    while True:
        step += 1
        plan.update(ev, current_sem(), approvals)
        # ---- boundary checks, in a fixed order
        if cancel_at is not None and clock.t >= cancel_at:
            return stop("cancelled", f"cancellation observed at step boundary, t={clock.t}s", False)
        nxt = estimate_tokens(messages) + estimate_tokens(defs)
        if used["model_calls"] + 1 + RESERVE_MODEL_CALLS > limits["model_calls"]:
            return stop("budget-exhausted", "model_calls (one call held in reserve)")
        if used["tokens"] + nxt + RESERVE_TOKENS > limits["tokens"]:
            return stop("budget-exhausted", f"tokens (next call needs {nxt}, "
                        f"{limits['tokens'] - used['tokens']} left, {RESERVE_TOKENS} reserved)")
        if clock.t > limits["sim_seconds"]:
            return stop("budget-exhausted", "sim_seconds")
        # ---- model proposes
        stage_before = plan.stage()
        reply = model.generate(messages, defs)
        used["model_calls"] += 1
        used["tokens"] += nxt
        used["sim_seconds"] = clock.t
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        n_before = len(ev.records)
        step_ops, malformed_only = [], bool(calls)
        if not calls:
            plan.update(ev, current_sem(), approvals)
            if plan.complete():
                events.append({"step": step, "t": clock.t, "stage": stage_before, "ops": ["final answer"],
                               "new_evidence": [], "stall": stall, "decision": "completion predicate true"})
                return stop("completed-verified", "all plan items satisfied by evidence", False)
            msg = ("Controller: completion refused; verification evidence is missing. Pending: "
                   + "; ".join(plan.pending()))
            messages.append({"role": "user", "content": msg})
            step_ops.append("final answer (refused)")
            malformed_only = False
        for call in calls:
            ident = op_identity(call)
            tool = call["name"]
            if f"tool:{tool}" in limits and used[f"tool:{tool}"] >= limits[f"tool:{tool}"]:
                return stop("budget-exhausted", f"tool:{tool}")
            if tool in APPROVAL_REQUIRED:
                pending_approval = {"operation": call, "bound_revision": current_sem(),
                                    "evidence": plan.items[3]["evidence"]}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": {"status": "awaiting_approval"}})
                events.append({"step": step, "t": clock.t, "stage": stage_before, "ops": [ident],
                               "new_evidence": [], "stall": stall, "decision": "approval required"})
                return stop("blocked-awaiting-approval", f"{tool} requires approval", False)
            source = tool == "replace_text" and norm_path(call["arguments"].get("path", "")) != "test_report.py"
            if source and plan.items[1]["status"] != "satisfied":
                out = {"status": "stage_refused",
                       "message": "Controller refused: reproduce the failure with a test before editing source."}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
                messages.append({"role": "user", "content": "Controller: edit refused; stage is reproduce."})
                step_ops.append(ident + " (refused)")
                malformed_only = False
                continue
            before = ws.snapshot()
            out = tools.execute_checked(call)
            clock.tool(tool)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
            step_ops.append(ident)
            if out.get("status") == "input_error":
                repairs[ident] = repairs.get(ident, 0) + 1
                if repairs[ident] > REPAIR_ALLOWANCE:
                    events.append({"step": step, "t": clock.t, "stage": stage_before, "ops": step_ops,
                                   "new_evidence": [], "stall": stall, "repairs": dict(repairs),
                                   "decision": "repair allowance exceeded"})
                    return stop("budget-exhausted", f"repair allowance for {ident}")
                continue
            malformed_only = False
            used[f"tool:{tool}"] = used.get(f"tool:{tool}", 0) + 1
            p = norm_path(call["arguments"].get("path", "")) if isinstance(call["arguments"], dict) else ""
            if tool == "read_file" and out.get("status") == "ok":
                v = ev.version(p, out["text"])
                ev.add("read", step, space="file", path=p, version=v, key=[p, v])
            elif tool == "replace_text" and out.get("status") == "ok":
                after = ws.read(p)
                v_before, v_after = ev.version(p, before[p]), ev.version(p, after)
                ev.add("edit", step, space="file", path=p, version=v_after, source=p != "test_report.py",
                       semantic_changed=v_after != initial_sem.get(p) and v_after != v_before,
                       raw_version=out["after_version"], key=[p, v_after])
            elif tool == "run_tests":
                sem = current_sem()
                ev.add("tests", step, status=out["status"], failed=out["failed_tests"],
                       tests_run=out["tests_run"], workspace=sem,
                       key=[out["status"], sorted(out["failed_tests"]), out["tests_run"],
                            sorted(sem.items())])
        new = [r["id"] + ":" + r["kind"] for r in ev.records[n_before:] if r["new"]]
        if malformed_only:
            decision = "repair allowance charged"
        elif new:
            stall, decision = 0, "progress"
        else:
            stall += 1
            decision = f"no new evidence ({stall}/{STALL_WINDOW})"
        plan.update(ev, current_sem(), approvals)
        events.append({"step": step, "t": clock.t, "stage": stage_before, "ops": step_ops,
                       "new_evidence": new, "stall": stall, "repairs": dict(repairs),
                       "decision": decision, "plan": {it["id"]: it["status"] for it in plan.items},
                       "budgets": {k: used[k] for k in ("model_calls", "tokens", "tool:run_tests")},
                       "sim_seconds": clock.t})
        if stall >= STALL_WINDOW:
            return stop("no-progress-escalated",
                        f"{STALL_WINDOW} consecutive steps without new evidence")
