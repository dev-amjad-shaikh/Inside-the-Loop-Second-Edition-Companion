"""Chapter 14: working with a person during a run.

    python3 ch14_human.py --output runs/ch14

Seven experiments, each writing its own files under the output directory:

  a_correction_before_dispatch   correction arrives while call 4 is generating;
                                 the stale proposal is refused at dispatch
  b_correction_after_edit        correction arrives after a signature-changing
                                 edit and a passing test; adjustment and re-verification
  c_approval_binding             approval bound to r5 is invalidated when the
                                 model re-proposes the export for r6 (bound vs loose)
  d_decision_requests            vague vs concrete decision request and what each authorizes
  e_prompt_policy                prompts per policy on one scripted session
  f_cancellation                 stop scheduling, settle in-flight work, report outcomes
  g_idempotent_submissions       double click and reconnect with and without keys

File reads, edits, test runs, signature probes and exports are executed for real
in temporary directories. Model responses and the person's answers are scripted.
"""
import argparse
import json
import os
import signal
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import Workspace, content_version, report_tools, write_run  # noqa: E402
from itl.ch14_human import (  # noqa: E402
    ApprovalStore, ArtifactStore, Clock, CorrectionModel, EventLog, EvidenceBook,
    ExportService, LooseApprovals, TaskState, add_signature_tool, canonical, digest,
    read_signature, rmtree, run_with_revisions, tmpdir)

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_FIXTURE = os.path.join(HERE, "fixtures", "ch14_export")

SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: settings.json, report.py, test_report.py.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")
CORRECTION = "Keep the public function signature unchanged."

NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''

OLD_FUNC = ('def belongs_to_report(created_at, requested_day, timezone):\n'
            '    """created_at is an aware UTC datetime; requested_day is a local date."""\n'
            '    return created_at.date() == requested_day')
SIG_CHANGING_FUNC = (
    'def belongs_to_report(created_at, requested_day, timezone="America/Los_Angeles"):\n'
    '    """created_at is an aware UTC datetime; requested_day is a local date."""\n'
    '    local_time = created_at.astimezone(ZoneInfo(timezone))\n'
    '    return local_time.date() == requested_day')
CHANGED_DEF = 'def belongs_to_report(created_at, requested_day, timezone="America/Los_Angeles"):'
ORIGINAL_DEF = 'def belongs_to_report(created_at, requested_day, timezone):'


def call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def say(cid_calls):
    return {"role": "assistant", "content": "", "tool_calls": cid_calls}


def opening():
    return [
        say([call("c1", "read_file", path="settings.json"),
             call("c2", "read_file", path="report.py"),
             call("c3", "read_file", path="test_report.py")]),
        say([call("c4", "replace_text", path="test_report.py",
                  old='\n\nif __name__ == "__main__":', new="\n" + NEW_TEST)]),
        say([call("c5", "run_tests", tests=["test_report.py"])]),
    ]


ANSWER_R1 = ("The filter used the UTC calendar date. It now converts each timestamp to the "
             "report timezone, and timezone defaults to the configured zone. The new "
             "near-midnight test failed before the fix; both tests pass. Nothing was deployed.")
ANSWER_R2 = ("The filter used the UTC calendar date. It now converts each timestamp to the "
             "report timezone before comparing dates. The public signature of "
             "belongs_to_report is unchanged. The near-midnight test failed before the fix; "
             "both tests pass on the final revision. Nothing was deployed.")


def run_correction(outdir, scenario):
    clock = Clock()
    log = EventLog(clock)
    ws = Workspace("report")
    before = ws.snapshot()
    baseline = read_signature(ws.root)
    task = TaskState(TASK, log)
    tools = report_tools(ws)
    add_signature_tool(tools, ws, baseline)
    evidence = EvidenceBook(log)

    def correct():
        # The person types the correction; the Human interface records it as a task revision.
        clock.tick(2)
        task.revise(CORRECTION, checks=["signature:belongs_to_report"])

    if scenario == "a":
        responses = opening() + [
            # call 4: generated under revision 1 while the correction arrives
            say([call("c6", "replace_text", path="report.py", old=OLD_FUNC, new=SIG_CHANGING_FUNC)]),
            # call 5: regenerated under revision 2
            say([call("c7", "replace_text", path="report.py",
                      old="    return created_at.date() == requested_day",
                      new="    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
                          "    return local_time.date() == requested_day")]),
            say([call("c8", "run_tests", tests=["test_report.py"])]),
            say([call("c9", "check_signature")]),
            {"role": "assistant", "content": ANSWER_R2, "tool_calls": []},
        ]
        model = CorrectionModel(responses, during={4: correct})
    else:
        responses = opening() + [
            say([call("c6", "replace_text", path="report.py", old=OLD_FUNC, new=SIG_CHANGING_FUNC)]),
            say([call("c7", "run_tests", tests=["test_report.py"])]),
            # call 6: final answer generated under revision 1 while the correction arrives
            {"role": "assistant", "content": ANSWER_R1, "tool_calls": []},
            say([call("c8", "check_signature")]),
            say([call("c9", "replace_text", path="report.py", old=CHANGED_DEF, new=ORIGINAL_DEF)]),
            say([call("c10", "run_tests", tests=["test_report.py"])]),
            say([call("c11", "check_signature")]),
            {"role": "assistant", "content": ANSWER_R2, "tool_calls": []},
        ]
        model = CorrectionModel(responses, during={6: correct})

    result = run_with_revisions(model, tools, ws, task, log, evidence, SYSTEM)
    after = ws.snapshot()
    write_run(outdir, "events.json", log.events)
    write_run(outdir, "task_revisions.json", task.revisions)
    write_run(outdir, "evidence.json", evidence.items)
    write_run(outdir, "completion_verdicts.json", result["verdicts"])
    write_run(outdir, "trace.json", model.requests)
    write_run(outdir, "messages.json", result["messages"])
    write_run(outdir, "executions.json", tools.executions)
    write_run(outdir, "answer.txt", result["answer"] + "\n")
    for label, snap in (("before", before), ("after", after)):
        for name, text in snap.items():
            write_run(os.path.join(outdir, label), name, text)
    final_sig = read_signature(ws.root)
    ws.close()
    return {"baseline_signature": baseline, "final_signature": final_sig,
            "final_report_version": content_version(after["report.py"]),
            "model_calls": len(model.requests),
            "tool_calls_proposed": sum(len(m.get("tool_calls") or []) for m in result["messages"]
                                       if m.get("role") == "assistant"),
            "executions": len(tools.executions),
            "refusals": [e for e in log.events if e["kind"] in ("dispatch_refused", "completion_refused")],
            "evidence": [{k: i[k] for k in ("id", "kind", "result", "version", "task_revision", "state")}
                         for i in evidence.items],
            "verdicts": result["verdicts"],
            "events": log.events}


# --------------------------------------------------------------------------
# c: approval bound to artifact version
# --------------------------------------------------------------------------

REPORT = "daily-2026-09-11"


def present(store, args, version, diff=None, evidence=None):
    """The decision request the Human interface renders for an export."""
    text = store.text(args["report"], args["revision"])
    total = [l for l in text.splitlines() if l.startswith("total")][0].split(",")[3]
    rows = len(text.splitlines()) - 2
    req = {
        "decision": f"Export {args['report']} {args['revision']} to destination {args['destination']}?",
        "operation": "export_report", "arguments": args,
        "artifact_version": version,
        "what_changes": f"Writes one CSV ({rows} orders, total {total} USD) to destination "
                        f"{args['destination']}; recipients can read it immediately.",
        "choices": [
            {"id": "approve", "consequence": "Export runs once, for this revision and destination only."},
            {"id": "hold", "consequence": "Nothing is exported; the run waits and keeps its work."},
            {"id": "reject", "consequence": "Export is dropped; the run continues without it."},
        ],
        "expires_in_s": 900,
    }
    if evidence:
        req["evidence"] = evidence
    if diff is not None:
        req["changed_since_last_approval"] = diff
    return req


def run_approvals(outdir, ev):
    clock = Clock()
    store = ArtifactStore(EXPORT_FIXTURE)
    out = {}

    # ---- bound approvals (the improved design) ----
    root = tmpdir("ch14-outbox-")
    svc = ExportService(store, root)
    appr = ApprovalStore(os.path.join(outdir, "approvals.sqlite"), clock)
    steps = []

    p1 = {"report": REPORT, "revision": "r5", "destination": "A"}
    v5 = store.version(REPORT, "r5")
    rq1, _ = appr.request("export_report", p1, v5, present(store, p1, v5, evidence=ev))
    steps.append({"step": "model proposes P1", "proposal": p1, "request": rq1})
    d1 = appr.decide(rq1, "maria", "approve", submission_key="ui-maria-0001")
    steps.append({"step": "maria approves", **d1, "approval": appr.approval_row(d1["approval_id"])})

    # a reviewer's edit produces r6 before the export is dispatched
    clock.tick(40)
    appr.audit("artifact_revised", report=REPORT, previous="r5", current="r6",
               previous_version=v5, current_version=store.version(REPORT, "r6"),
               reason="refunded orders excluded")
    p2 = {"report": REPORT, "revision": "r6", "destination": "A"}
    v6 = store.version(REPORT, "r6")
    chk = appr.check_at_dispatch("x2", "export_report", p2, v6)
    steps.append({"step": "model re-proposes P2 for r6; dispatch check", "proposal": p2, **chk})
    diff = store.diff(REPORT, "r5", "r6")
    rq2, _ = appr.request("export_report", p2, v6, present(store, p2, v6, diff=diff, evidence=ev))
    steps.append({"step": "re-request with diff", "request": rq2, "diff": diff})
    d2 = appr.decide(rq2, "maria", "approve", submission_key="ui-maria-0002")
    steps.append({"step": "maria approves r6", **d2, "approval": appr.approval_row(d2["approval_id"])})
    chk2 = appr.check_at_dispatch("x3", "export_report", p2, v6)
    receipt = svc.export(**p2) if chk2["allowed"] else None
    steps.append({"step": "dispatch P2", **chk2, "receipt": receipt})
    chk3 = appr.check_at_dispatch("x4", "export_report", p2, v6)
    steps.append({"step": "model proposes P2 again", **chk3})
    out["bound"] = {"steps": steps, "outbox": svc.listing(),
                    "approvals": [appr.approval_row(a["approval_id"]) for a in appr.table("approvals")]}
    write_run(outdir, "decision_request_r6.json",
              json.loads(appr.table("requests")[1]["presented_json"]))
    write_run(outdir, "diff_r5_r6.txt", diff)
    rmtree(root)

    # ---- governance: roles, delegation, expiry ----
    gov = []
    p3 = {"report": REPORT, "revision": "r6", "destination": "B"}
    rq3, _ = appr.request("export_report", p3, v6, present(store, p3, v6, evidence=ev))
    gov.append({"attempt": "sam (analyst) approves export to B",
                **appr.decide(rq3, "sam", "approve", submission_key="ui-sam-0001")})
    appr.delegate("dev", "maria", "export_report", "A", ttl=3600)
    gov.append({"attempt": "dev (delegated for A) approves export to B",
                **appr.decide(rq3, "dev", "approve", submission_key="ui-dev-0001")})
    p4 = {"report": REPORT, "revision": "r6", "destination": "A", "note": "resend"}
    rq4, _ = appr.request("export_report", p4, v6, present(store, p4, v6, evidence=ev))
    d4 = appr.decide(rq4, "dev", "approve", submission_key="ui-dev-0002")
    gov.append({"attempt": "dev (delegated for A) approves export to A", **d4,
                "approval": appr.approval_row(d4["approval_id"])})
    clock.tick(1000)
    gov.append({"attempt": "dispatch 1000 s later",
                **appr.check_at_dispatch("x5", "export_report", p4, v6)})
    out["governance"] = gov
    out["audit"] = appr.table("audit")
    write_run(outdir, "audit.json", [dict(r, detail=json.loads(r["detail"])) for r in out["audit"]])

    # ---- loose approvals (the flawed design) ----
    root = tmpdir("ch14-outbox-")
    svc = ExportService(store, root)
    loose = LooseApprovals()
    loose.approve("maria", "export_report", "A")   # shown r5 when she clicked
    res = loose.check("export_report", p2)          # the model now proposes r6
    receipt = svc.export(**p2) if res["allowed"] else None
    out["loose"] = {"person_saw": "r5", "exported": receipt, "outbox": svc.listing(),
                    "grant": res.get("grant")}
    rmtree(root)
    write_run(outdir, "approvals.json", out)
    appr.db.close()
    return out


# --------------------------------------------------------------------------
# d: vague vs concrete decision requests
# --------------------------------------------------------------------------

QUESTIONS = [
    ("names the operation", lambda r: "operation" in r),
    ("names the artifact and version", lambda r: "artifact_version" in r),
    ("states what changes in the world", lambda r: "what_changes" in r),
    ("shows evidence with its version", lambda r: bool(r.get("evidence"))),
    ("offers alternatives with consequences",
     lambda r: all(isinstance(c, dict) and c.get("consequence") for c in r.get("choices", []))),
    ("answer binds to an exact operation", lambda r: "arguments" in r and "artifact_version" in r),
]


def run_decisions(outdir, ev):
    clock = Clock()
    store = ArtifactStore(EXPORT_FIXTURE)
    v6 = store.version(REPORT, "r6")
    p = {"report": REPORT, "revision": "r6", "destination": "A"}
    vague = {"decision": "Should I proceed?", "choices": ["yes", "no"]}
    concrete = present(store, p, v6, evidence=ev)
    scores = {}
    for name, req in (("vague", vague), ("concrete", concrete)):
        scores[name] = {q: bool(fn(req)) for q, fn in QUESTIONS}

    # The model's next proposal after the answer names a different destination.
    nxt = {"report": REPORT, "revision": "r6", "destination": "B"}
    outcomes = {}

    # vague: the answer "yes" is stored as approval of whatever comes next
    root = tmpdir("ch14-outbox-")
    svc = ExportService(store, root)
    loose = LooseApprovals()
    loose.approve("maria", "export_report", "B")  # "yes" had no operation; runtime grants the next one
    r = loose.check("export_report", nxt)
    outcomes["vague"] = {"answer": "yes", "next_proposal": nxt,
                         "dispatch": "executed" if r["allowed"] else "refused",
                         "receipt": svc.export(**nxt) if r["allowed"] else None,
                         "outbox": None}
    outcomes["vague"]["outbox"] = svc.listing()
    rmtree(root)

    # concrete: the answer is a choice bound to (operation, arguments, artifact version)
    root = tmpdir("ch14-outbox-")
    svc = ExportService(store, root)
    appr = ApprovalStore(os.path.join(root, "d.sqlite"), clock)
    rq, _ = appr.request("export_report", p, v6, concrete)
    appr.decide(rq, "maria", "approve", submission_key="ui-maria-0101")
    chk = appr.check_at_dispatch("y1", "export_report", nxt, v6)
    outcomes["concrete"] = {"answer": "approve", "next_proposal": nxt,
                            "dispatch": "executed" if chk["allowed"] else "refused",
                            "check": chk}
    appr.db.close()
    os.remove(os.path.join(root, "d.sqlite"))
    outcomes["concrete"]["outbox"] = svc.listing()
    rmtree(root)
    data = {"vague_request": vague, "concrete_request": concrete, "scores": scores,
            "outcomes": outcomes}
    write_run(outdir, "decision_requests.json", data)
    return data


# --------------------------------------------------------------------------
# e: prompt policy
# --------------------------------------------------------------------------

SESSION = [
    ("read_file", {"path": "settings.json"}),
    ("read_file", {"path": "report.py"}),
    ("read_file", {"path": "test_report.py"}),
    ("replace_text", {"path": "test_report.py"}),
    ("run_tests", {}),
    ("replace_text", {"path": "report.py"}),
    ("run_tests", {}),
    ("write_file", {"path": "../shared/settings.json"}),
    ("replace_text", {"path": "report.py"}),
    ("run_tests", {}),
    ("check_signature", {}),
    ("deploy", {"service": "report-service"}),
    ("write_file", {"path": "NOTES.md"}),
    ("export_report", {"revision": "r6", "destination": "A"}),
]
READS = {"read_file", "check_signature", "run_tests"}


def ask_every_write(op, args):
    return "read" if op in READS else "ask"


def enforce_then_ask(op, args):
    """Routine constraints are enforced; only consequential choices reach a person."""
    if op in READS:
        return "allow"
    if op in ("replace_text", "write_file"):
        return "refuse:outside workspace" if args["path"].startswith("..") else "allow"
    if op == "deploy":
        return "refuse:task forbids deployment"
    if op == "export_report":
        return "ask"
    return "refuse:unknown operation"


def run_prompts(outdir):
    results = {}
    for pname, policy in (("ask_every_write", ask_every_write), ("enforce_then_ask", enforce_then_ask)):
        for person in ("approves_all", "careful"):
            ws_root = tmpdir("ch14-ws-")
            shared = os.path.join(os.path.dirname(ws_root), os.path.basename(ws_root) + "-shared")
            os.makedirs(shared, exist_ok=True)
            deploy_dir = tmpdir("ch14-deploy-")
            rows, prompts, executed_forbidden = [], 0, []
            for i, (op, args) in enumerate(SESSION, 1):
                decision = policy(op, args)
                if decision == "ask":
                    prompts += 1
                    forbidden = op == "deploy" or args.get("path", "").startswith("..")
                    answer = "approve" if person == "approves_all" or not forbidden else "reject"
                    decision = "allow (person approved)" if answer == "approve" else "refuse (person rejected)"
                if decision.startswith("allow") or decision == "read":
                    # perform the effect so its presence can be checked afterwards
                    if op == "write_file" and args["path"].startswith(".."):
                        with open(os.path.join(shared, "settings.json"), "w") as f:
                            f.write('{"report_timezone": "UTC"}\n')
                        executed_forbidden.append(f"{i} write outside workspace")
                    if op == "deploy":
                        with open(os.path.join(deploy_dir, "DEPLOYED"), "w") as f:
                            f.write("report-service\n")
                        executed_forbidden.append(f"{i} deploy")
                rows.append({"n": i, "operation": op, "arguments": args, "decision": decision})
            results[f"{pname}/{person}"] = {
                "prompts": prompts, "operations": len(SESSION),
                "forbidden_effects_executed": executed_forbidden,
                "shared_settings_written": os.path.exists(os.path.join(shared, "settings.json")),
                "deployed": os.path.exists(os.path.join(deploy_dir, "DEPLOYED")),
                "rows": rows}
            rmtree(ws_root)
            rmtree(shared)
            rmtree(deploy_dir)
    write_run(outdir, "prompt_policy.json", results)
    return results


# --------------------------------------------------------------------------
# f: cancellation
# --------------------------------------------------------------------------

class RemoteJobs:
    """Deterministic stand-in for a remote export service with a cancel endpoint.

    A job commits at its commit tick. A cancel received before that tick stops it and is
    acknowledged; after it, the job completes. Job 'regen' never acknowledges a cancel.
    """

    def __init__(self, store, root):
        self.store, self.root = store, root
        self.jobs = {}

    def start(self, jid, spec, tick):
        self.jobs[jid] = dict(spec, started=tick, state="running", cancel_at=None)

    def cancel(self, jid, tick):
        self.jobs[jid]["cancel_at"] = tick

    def poll(self, jid, tick):
        j = self.jobs[jid]
        if j["state"] != "running":
            return j["state"]
        if j.get("silent"):
            return "running"
        commit = j["started"] + j["commit_after"]
        if j["cancel_at"] is not None and j["cancel_at"] < commit:
            j["state"] = "cancelled"
        elif tick >= j["started"] + j["duration"]:
            d = os.path.join(self.root, j["destination"])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{j['report']}.r6.{jid}.csv"), "w") as f:
                f.write(self.store.text(j["report"], "r6"))
            j["state"] = "completed"
        return j["state"]


def run_cancellation(outdir):
    store = ArtifactStore(EXPORT_FIXTURE)
    root = tmpdir("ch14-outbox-")
    remote = RemoteJobs(store, root)
    plan = [
        {"op": "o1", "kind": "local", "what": "run_tests (subprocess)", "duration": 5},
        {"op": "o2", "kind": "remote", "what": "export r6 to A", "destination": "A",
         "report": REPORT, "duration": 4, "commit_after": 1},
        {"op": "o3", "kind": "remote", "what": "regenerate report 2026-09-12", "destination": "A",
         "report": REPORT, "duration": 6, "commit_after": 5, "silent": True},
        {"op": "o4", "kind": "remote", "what": "export r6 to B", "destination": "B",
         "report": REPORT, "duration": 4, "commit_after": 1},
        {"op": "o5", "kind": "local", "what": "write NOTES.md", "duration": 1},
        {"op": "o6", "kind": "remote", "what": "notify report channel", "destination": "C",
         "report": REPORT, "duration": 2, "commit_after": 1},
    ]
    concurrency, cancel_tick, settle_deadline = 3, 2, 8
    queue = list(plan)
    running, outcome, timeline = {}, {}, []
    procs = {}
    cancelled = False
    for tick in range(0, 20):
        if tick == cancel_tick:
            cancelled = True
            timeline.append({"tick": tick, "event": "cancel requested by person"})
            for op in list(queue):
                outcome[op["op"]] = {"what": op["what"], "outcome": "not started",
                                     "evidence": "never scheduled"}
                timeline.append({"tick": tick, "event": f"{op['op']} dropped from queue"})
            queue.clear()
            for oid, op in running.items():
                if op["kind"] == "local":
                    os.killpg(procs[oid].pid, signal.SIGTERM)
                    rc = procs[oid].wait(timeout=10)
                    outcome[oid] = {"what": op["what"], "outcome": "cancelled",
                                    "evidence": f"process group terminated, exit code {rc}"}
                    timeline.append({"tick": tick, "event": f"{oid} process terminated (exit {rc})"})
                else:
                    remote.cancel(oid, tick)
                    timeline.append({"tick": tick, "event": f"{oid} cancel sent to service"})
            for oid in [o for o in running if o in outcome]:
                running.pop(oid)
        # schedule new work only while not cancelled
        while not cancelled and queue and len(running) < concurrency:
            op = queue.pop(0)
            running[op["op"]] = op
            if op["kind"] == "local":
                procs[op["op"]] = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
            else:
                remote.start(op["op"], op, tick)
            timeline.append({"tick": tick, "event": f"{op['op']} started ({op['what']})"})
        # settle remote work
        for oid, op in list(running.items()):
            if op["kind"] == "remote":
                state = remote.poll(oid, tick)
                if state in ("completed", "cancelled"):
                    was_cancel = remote.jobs[oid]["cancel_at"] is not None
                    outcome[oid] = {"what": op["what"],
                                    "outcome": ("completed after cancel" if state == "completed" and was_cancel
                                                else state),
                                    "evidence": ("service receipt; file present" if state == "completed"
                                                 else "service acknowledged cancel before commit")}
                    timeline.append({"tick": tick, "event": f"{oid} settled: {outcome[oid]['outcome']}"})
                    running.pop(oid)
                elif cancelled and tick >= settle_deadline:
                    outcome[oid] = {"what": op["what"], "outcome": "unknown",
                                    "evidence": "cancel sent, no acknowledgment by settle deadline; "
                                                "handed to reconciler"}
                    timeline.append({"tick": tick, "event": f"{oid} unknown at deadline"})
                    running.pop(oid)
        if cancelled and not running:
            timeline.append({"tick": tick, "event": "run cancelled; all operations accounted for"})
            break
    report = [dict(op=o["op"], **outcome[o["op"]]) for o in plan]
    files = {d: sorted(os.listdir(os.path.join(root, d))) for d in sorted(os.listdir(root))}
    flawed_status = {o["op"]: "cancelled" for o in plan}
    data = {"plan": plan, "cancel_tick": cancel_tick, "settle_deadline": settle_deadline,
            "timeline": timeline, "report": report, "files_in_outbox": files,
            "flawed_status_report": flawed_status}
    write_run(outdir, "cancellation.json", data)
    rmtree(root)
    return data


# --------------------------------------------------------------------------
# g: idempotent submissions
# --------------------------------------------------------------------------

def run_idempotency(outdir):
    store = ArtifactStore(EXPORT_FIXTURE)
    v6 = store.version(REPORT, "r6")
    p = {"report": REPORT, "revision": "r6", "destination": "A"}
    res = {}

    # flawed handler: every POST creates a request / decision and resumes dispatch
    root = tmpdir("ch14-outbox-")
    svc = ExportService(store, root)
    requests_shown, approvals = [], []
    for attempt in ("initial page load", "reconnect after network drop"):
        requests_shown.append({"request": f"RQ{len(requests_shown) + 1}", "cause": attempt})
    for click in ("click", "second click 180 ms later"):
        approvals.append({"approval": f"AP{len(approvals) + 1}", "cause": click})
        svc.export(**p)   # each approval resumes the waiting run, which dispatches
    res["flawed"] = {"requests_shown": requests_shown, "approvals": approvals,
                     "exports": svc.receipts, "outbox": svc.listing()}
    rmtree(root)

    # keyed handler
    clock = Clock()
    root = tmpdir("ch14-outbox-")
    svc = ExportService(store, root)
    appr = ApprovalStore(os.path.join(root, "g.sqlite"), clock)
    trail = []
    for attempt in ("initial page load", "reconnect after network drop"):
        rq, created = appr.request("export_report", p, v6, present(store, p, v6))
        trail.append({"attempt": attempt, "request": rq, "created": created})
    for click in ("click", "second click 180 ms later"):
        d = appr.decide(rq, "maria", "approve", submission_key="ui-maria-7f3a")
        trail.append({"attempt": click, **d})
        if d["status"] == "recorded":
            chk = appr.check_at_dispatch("z1", "export_report", p, v6)
            if chk["allowed"]:
                svc.export(**p)
    res["keyed"] = {"trail": trail, "requests": len(appr.table("requests")),
                    "decisions": len(appr.table("decisions")),
                    "approvals": len(appr.table("approvals")), "exports": svc.receipts,
                    "outbox": {k: v for k, v in svc.listing().items()}}
    appr.db.close()
    os.remove(os.path.join(root, "g.sqlite"))
    rmtree(root)
    write_run(outdir, "idempotency.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    if os.path.exists(out):
        sys.exit(f"{out} exists; choose a new directory")
    os.makedirs(out)

    a = run_correction(os.path.join(out, "a_correction_before_dispatch"), "a")
    b = run_correction(os.path.join(out, "b_correction_after_edit"), "b")
    # Evidence shown at approval time comes from run a's evidence book.
    ev = [f"{i['id']}: tests {i['result']} on report.py {i['version']} (task revision {i['task_revision']})"
          for i in a["evidence"] if i["kind"] == "tests" and i["state"] == "current"]
    c = run_approvals(os.path.join(out, "c_approval_binding"), ev)
    d = run_decisions(os.path.join(out, "d_decision_requests"), ev)
    e = run_prompts(os.path.join(out, "e_prompt_policy"))
    f = run_cancellation(os.path.join(out, "f_cancellation"))
    g = run_idempotency(os.path.join(out, "g_idempotent_submissions"))

    print("== a: correction arrives while call 4 is generating")
    for ev in a["events"]:
        if ev["kind"] in ("model_call_started", "model_call_finished", "task_revised",
                          "dispatch_refused", "dispatched", "run_completed", "evidence_historical"):
            extra = {k: v for k, v in ev.items() if k not in ("seq", "t", "kind")}
            print(f"  {ev['seq']:>3} {ev['t']} {ev['kind']:<20} {canonical(extra)}")
    print("  model calls:", a["model_calls"], " proposed tool calls:", a["tool_calls_proposed"],
          " executions:", a["executions"])
    print("  final report.py version:", a["final_report_version"], " signature:", a["final_signature"])

    print("== b: correction arrives after the signature-changing edit")
    for ev in b["events"]:
        if ev["kind"] in ("task_revised", "completion_refused", "evidence_historical", "run_completed",
                          "dispatched"):
            extra = {k: v for k, v in ev.items() if k not in ("seq", "t", "kind")}
            print(f"  {ev['seq']:>3} {ev['t']} {ev['kind']:<20} {canonical(extra)}")
    for v in b["verdicts"]:
        print(f"  completion at call {v['call']}: proposal rev {v['proposal_task_revision']}, "
              f"current rev {v['current_task_revision']}; flawed verifier complete="
              f"{v['flawed_verifier']['complete']}; revision verifier complete="
              f"{v['revision_verifier']['complete']} missing={v['revision_verifier']['missing']}")
    for item in b["evidence"]:
        print("  evidence", canonical(item))
    print("  final report.py version:", b["final_report_version"], " signature:", b["final_signature"])

    print("== c: approval bound to artifact version")
    for s in c["bound"]["steps"]:
        keep = {k: s[k] for k in ("allowed", "reason", "mismatches", "approval_id", "request", "status")
                if k in s}
        print(f"  {s['step']}: {canonical(keep)}")
    print("  outbox (bound):", c["bound"]["outbox"])
    print("  outbox (loose): ", c["loose"]["outbox"], " person saw", c["loose"]["person_saw"])
    for gv in c["governance"]:
        keep = {k: gv[k] for k in ("status", "reason", "allowed", "mismatches", "approval_id") if k in gv}
        print(f"  {gv['attempt']}: {canonical(keep)}")

    print("== d: vague vs concrete decision request")
    for name in ("vague", "concrete"):
        print(f"  {name}: {sum(d['scores'][name].values())}/{len(QUESTIONS)} questions answered;"
              f" next proposal to B -> {d['outcomes'][name]['dispatch']};"
              f" outbox {d['outcomes'][name]['outbox']}")

    print("== e: prompts per policy (14 proposed operations)")
    for k, v in e.items():
        print(f"  {k:<32} prompts={v['prompts']:>2} forbidden effects executed={v['forbidden_effects_executed']}")

    print("== f: cancellation")
    for t in f["timeline"]:
        print(f"  tick {t['tick']}: {t['event']}")
    for r in f["report"]:
        print(f"  {r['op']} {r['what']:<30} {r['outcome']:<24} {r['evidence']}")
    print("  files in outbox:", f["files_in_outbox"])

    print("== g: idempotent submissions")
    print("  flawed: requests shown", len(g["flawed"]["requests_shown"]), " approvals",
          len(g["flawed"]["approvals"]), " exports", len(g["flawed"]["exports"]))
    print("  keyed:  requests", g["keyed"]["requests"], " decisions", g["keyed"]["decisions"],
          " approvals", g["keyed"]["approvals"], " exports", len(g["keyed"]["exports"]))
    for t in g["keyed"]["trail"]:
        print("   ", canonical(t))

    write_run(out, "summary.json", {
        "a": {k: a[k] for k in ("baseline_signature", "final_signature", "final_report_version",
                                "model_calls", "tool_calls_proposed", "executions", "refusals")},
        "b": {k: b[k] for k in ("baseline_signature", "final_signature", "final_report_version",
                                "model_calls", "tool_calls_proposed", "executions", "refusals",
                                "evidence")},
        "c_outbox_bound": c["bound"]["outbox"], "c_outbox_loose": c["loose"]["outbox"],
        "d_scores": {k: sum(v.values()) for k, v in d["scores"].items()},
        "e_prompts": {k: v["prompts"] for k, v in e.items()},
        "f_report": f["report"],
        "g": {"flawed_exports": len(g["flawed"]["exports"]), "keyed_exports": len(g["keyed"]["exports"])},
    })


if __name__ == "__main__":
    main()
