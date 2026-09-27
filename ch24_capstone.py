"""Chapter 24: the capstone run, the five boundary cases and a naive wiring.

    python3 ch24_capstone.py --output runs/ch24

Every scenario runs the same integrated runtime (itl/ch24_capstone.py), which
composes the Part II helper modules. Scenarios differ only in the authored
model replies, the person's scripted answers and the injected events:

  capstone        the whole task as a service: intake, skill activation,
                  context with provenance, streamed replies through the adapter,
                  tools through policy and the executor, a malformed request,
                  a correction during generation, delegated review (a stale r4
                  result and a verified r5 review), an approval for r5
                  invalidated by r6 and re-approved, a worker lost after the
                  export dispatch, reconciliation, verification, the record
  normal          the boundary case with no disturbance
  missing_input   settings.json has no report_timezone
  malformed       one request with a string where a list is required
  interruption    the worker is lost after the export service commits
  correction      the user corrects the task while a patch is being generated
  capstone_naive  the capstone with three common glue shortcuts: approvals bound to
                  tool and destination, a restart that resends every operation
                  without a receipt and ignores the store's refusals, and a
                  completion check that trusts the loop
  skill_scope_open  the normal run with the skill's declarations left in force
                  after its stages are done

Standard library only. Model replies and the person's answers are scripted.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import write_run  # noqa: E402
from itl.ch15_delegation import BUGGY_LINE, FIX, NEW_TEST  # noqa: E402
from itl.ch24_capstone import REPORT, Capstone  # noqa: E402

TASK = ("The daily report is missing some orders placed near midnight. Find the cause, prepare a fix "
        "and run the tests. Do not deploy it. When the fix is verified, export the corrected daily "
        "report for 2026-09-11 to destination A.")

OLD_FUNC = ('def belongs_to_report(created_at, requested_day, timezone):\n'
            '    """created_at is an aware UTC datetime; requested_day is a local date."""\n'
            '    return created_at.date() == requested_day')
SIG_CHANGING_FUNC = (
    'def belongs_to_report(created_at, requested_day, timezone="America/Los_Angeles"):\n'
    '    """created_at is an aware UTC datetime; requested_day is a local date."""\n'
    '    local_time = created_at.astimezone(ZoneInfo(timezone))\n'
    '    return local_time.date() == requested_day')
PROBE_ARGS = ["report.py", "America/Los_Angeles", "2026-09-11", "2026-11-01"]

ANSWER_R5 = ("The filter used the UTC calendar date. report.py now converts each timestamp to the "
             "configured report timezone before comparing dates. Changed files: report.py 339d3a9f2656, "
             "test_report.py 319f11ce22b8. The near-midnight test failed before the fix and both tests "
             "pass on 339d3a9f2656. Report daily-2026-09-11 r5 was exported to destination A as "
             "export-file-001 after approval. Nothing was deployed.")
ANSWER_R5_SIG = ANSWER_R5.replace("Changed files:", "The public signature of belongs_to_report is "
                                                    "unchanged. Changed files:")
ANSWER_R6 = ("The filter used the UTC calendar date. report.py now converts each timestamp to the "
             "configured report timezone before comparing dates, and the public signature of "
             "belongs_to_report is unchanged. Changed files: report.py 339d3a9f2656, test_report.py "
             "319f11ce22b8. The near-midnight test failed before the fix and both tests pass on "
             "339d3a9f2656. Report daily-2026-09-11 r6, published by finance after removing a refunded "
             "order, was exported to destination A as export-file-001 after approval. Nothing was deployed.")


def call(cid, tool, **arguments):
    return {"id": cid, "name": tool, "arguments": arguments}


def say(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def answer(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


R = {
    "activate": say(call("k1", "activate_skill", name="report-helper")),
    "read": say(call("c1", "read_file", path="settings.json"), call("c2", "read_file", path="report.py"),
                call("c3", "read_file", path="test_report.py"),
                call("r1", "read_skill_resource", skill="report-helper", path="references/regression-test.md")),
    "add_test": say(call("c4", "replace_text", path="test_report.py",
                         old='\n\nif __name__ == "__main__":', new="\n" + NEW_TEST)),
    "malformed": say(call("c5", "run_tests", tests="test_report.py")),
    "reproduce": say(call("c6", "run_tests", tests=["test_report.py"])),
    "sig_change": say(call("c7", "replace_text", path="report.py", old=OLD_FUNC, new=SIG_CHANGING_FUNC)),
    "fix": say(call("c8", "replace_text", path="report.py", old=BUGGY_LINE, new=FIX)),
    "probe": say(call("s1", "run_skill_script", skill="report-helper", script="scripts/probe_boundary.py",
                      args=PROBE_ARGS)),
    "verify": say(call("c9", "run_tests", tests=["test_report.py"])),
    "export_r5": say(call("x1", "export_report", report=REPORT, revision="r5", destination="A")),
    "export_r6": say(call("x2", "export_report", report=REPORT, revision="r6", destination="A")),
    "ask": say(call("q1", "request_input",
                    question="settings.json has no report_timezone. Which timezone defines the report day "
                             "for acme?",
                    choices=["America/Los_Angeles", "UTC", "Another zone"])),
}


def replies(*labels, final=None):
    out = [(l, R[l]) for l in labels]
    if final:
        out.append(("answer", answer(final)))
    return out


BASE = ["activate", "read", "add_test", "reproduce", "fix", "probe", "verify", "export_r5"]


def scenarios():
    capstone = {
        "task": TASK, "delegate": True, "characterize_at_r4": True, "export": True, "interrupt_export": True,
        "replies": replies("activate", "read", "add_test", "malformed", "reproduce", "sig_change", "fix",
                           "probe", "verify", "export_r5", "export_r6", final=ANSWER_R6),
        "during": {"sig_change": "correction"},
        "decisions": {"RQ1": [("dev", "approve"), ("maria", "approve")], "RQ2": [("maria", "approve")]},
        "after_decision": {"RQ1": "publish_r6"},
    }
    normal = {"task": TASK, "delegate": True, "export": True,
              "replies": replies(*BASE, final=ANSWER_R5), "decisions": {"RQ1": [("maria", "approve")]}}
    missing = {"task": TASK, "fixture": "ch24_capstone/report_missing_tz", "export": False,
               "replies": replies("activate", "read", "ask")}
    malformed = dict(normal, replies=replies("activate", "read", "add_test", "malformed", "reproduce", "fix",
                                             "probe", "verify", "export_r5", final=ANSWER_R5))
    interruption = dict(normal, interrupt_export=True)
    correction = dict(normal, replies=replies("activate", "read", "add_test", "reproduce", "sig_change", "fix",
                                              "probe", "verify", "export_r5", final=ANSWER_R5_SIG),
                      during={"sig_change": "correction"})
    scope_open = dict(normal, keep_skill_scope=True)
    return [("capstone", capstone, "revised"), ("normal", normal, "revised"),
            ("missing_input", missing, "revised"), ("malformed", malformed, "revised"),
            ("interruption", interruption, "revised"), ("correction", correction, "revised"),
            ("capstone_naive", capstone, "naive"), ("skill_scope_open", scope_open, "revised")]


REQUEST = {"task": TASK, "profile": "report.fix_and_export", "tenant": "globex"}


def run_one(outdir, sc, wiring):
    cap = Capstone(outdir, sc, wiring=wiring)
    cap.intake("sess-acme-dev", dict(REQUEST, task=sc["task"]))
    try:
        cap.run()
        record = cap.five_questions()
        write_outputs(cap, outdir, record)
    finally:
        if hasattr(cap, "_revstore"):
            cap._revstore.close()
        for root, dirs, _ in os.walk(outdir):
            for d in list(dirs):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        shutil.rmtree(os.path.join(outdir, "source-r3"), ignore_errors=True)
    return cap, record


def write_outputs(cap, outdir, record):
    write_run(outdir, "five_questions.json", record)
    write_run(outdir, "manifest.json", cap.manifest())
    write_run(outdir, "timeline.json", cap.log.events)
    write_run(outdir, "operations.json", cap.ops)
    write_run(outdir, "completion.json", cap.completion)
    write_run(outdir, "retrieval.json", cap.retrieval)
    write_run(outdir, "memory.json", {"writes": cap.memory_writes, "retrieved": cap.memory})
    write_run(outdir, "skills.json", {"activation_check": cap.activation_check, "scope": cap.skill_scope,
                                      "decisions": cap.skills.decisions, "scripts": cap.skills.scripts,
                                      "registry": cap.registry.snapshot()})
    write_run(outdir, "human.json", {"events": cap.human, "publications": cap.publications,
                                     "requests": cap.approvals.table("requests"),
                                     "decisions": cap.approvals.table("decisions"),
                                     "approvals": cap.approvals.table("approvals"),
                                     "audit": cap.approvals.table("audit")})
    write_run(outdir, "workers.json", [{"contract_id": w["contract_id"], "worker": w["worker"],
                                        "input_revision": w["revision"], "dispatched": w["dispatched"],
                                        "returned": w.get("returned"), "status": w["result"]["status"],
                                        "verdict": (w["result"]["result"] or {}).get("verdict"),
                                        "findings": (w["result"]["result"] or {}).get("findings"),
                                        "contract": w["contract"], "ledger": w["result"]["ledger"]}
                                       for w in cap.workers])
    write_run(outdir, "review_integration.json", cap.review)
    write_run(outdir, "restarts.json", cap.restarts)
    calls = []
    for m in cap.model_calls:
        slim = {k: v for k, v in m.items() if k != "messages_sent"}
        calls.append(slim)
        write_run(os.path.join(outdir, "requests"), f"call{m['call']:02d}.json",
                  {"call": m["call"], "messages": m["messages_sent"], "provenance": m["provenance"],
                   "omissions": m["omissions"]})
    write_run(outdir, "model_calls.json", calls)
    write_run(outdir, "five_questions.txt", render_record(record))


def render_record(r):
    lines = [f"Run {r['run']}: {r['outcome']}", ""]
    q1 = r["1_what_did_the_model_receive"]
    lines.append("1. What did the model receive?")
    for m in q1["model_calls"]:
        lines.append(f"   call {m['call']:>2}  task r{m['task_revision']}  request {m['request_sha']}  "
                     f"{m['messages']:>2} messages  ~{m['input_tokens_approx']} tokens  "
                     f"{m['adapter_outcome']} ({m['raw_stop_reason']})  omissions {m['omissions']}")
    ex = q1["excluded_before_the_model"]
    lines.append("   excluded before the model: " + "; ".join(f"{a} {b}" for a, b in ex["retrieval_filters"]))
    for x in ex["memory_excluded"]:
        lines.append(f"   memory excluded: {x['id']} ({x['reason']})")
    for x in ex["memory_writes_rejected"]:
        lines.append(f"   memory write rejected: {x['key']} ({'; '.join(x['reasons'] or [])})")
    lines.append("")
    lines.append("2. Which operation was authorized, and by whom?")
    for o in r["2_which_operation_was_authorized_and_by_whom"]:
        a = o["approval"]
        extra = f"  approval {a['approval_id']} by {a['actor']} for {a['artifact_version']}" if a else ""
        rules = ("  rules: " + ", ".join(o["rules_not_passed"])) if o["rules_not_passed"] else ""
        lines.append(f"   {o['call_id']:<3} {o['tool']:<19} r{o['task_revision']}  {o['decision']:<14} "
                     f"{o['outcome']}{extra}{rules}")
    lines.append("")
    q3 = r["3_what_actually_changed"]
    lines.append("3. What actually changed?")
    for p, v in q3["workspace"].items():
        lines.append(f"   {p}: {v['before']} -> {v['after']}")
    for e in q3["exports"]:
        lines.append(f"   export {e['export_id']} key {e['op_key']} destination {e['destination']}")
    for d, files in q3["outbox"].items():
        for f in files:
            lines.append(f"   outbox/{d}/{f['file']} version {f['version']}")
    lines.append(f"   deployments {q3['deployments']}, memory writes by the run {q3['memory_writes_by_run']}")
    lines.append("")
    q4 = r["4_which_evidence_supports_the_result"]
    lines.append("4. Which evidence supports the result?")
    for e in q4["evidence"]:
        lines.append(f"   {e['id']} {e['kind']} {e['result']} on report.py {e['version']} "
                     f"(task r{e['task_revision']}, {e['state']}, from {e['call_id']})")
    if q4["review"]:
        lines.append(f"   review at {q4['review']['current_revision']}: {q4['review']['verdict']}; "
                     f"admitted {q4['review']['admitted']}; stale {[s['worker'] + '@' + s['reviewed'] for s in (q4['review']['stale'] or [])]}")
    if q4["verdict"]:
        lines.append(f"   verifier: complete={q4['verdict']['complete']} missing={q4['verdict']['missing']}")
    lines.append("")
    lines.append("5. What remains uncertain?")
    for u in r["5_what_remains_uncertain"]:
        lines.append("   - " + u)
    return "\n".join(lines) + "\n"


class Tee:
    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.output)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    say_ = Tee()
    summary = {}
    for name, sc, wiring in scenarios():
        cap, rec = run_one(os.path.join(out, name), sc, wiring)
        q3 = rec["3_what_actually_changed"]
        refused = [o["call_id"] + ":" + o["outcome"] for o in cap.ops
                   if o["outcome"] not in ("ok", "completed", "passed", "failed")]
        summary[name] = {"outcome": cap.outcome, "model_calls": len(cap.model_calls),
                         "operations": len(cap.ops), "not_executed": refused,
                         "exports": [e["export_id"] for e in q3["exports"]],
                         "outbox": {d: [f["file"] for f in fs] for d, fs in q3["outbox"].items()},
                         "report_py": q3["workspace"].get("report.py"),
                         "completion_verdicts": [{"call": c["call"], "complete": c["complete"],
                                                  "missing": c["missing"]} for c in cap.completion],
                         "executions": [e["call_id"] + " " + e["name"] for e in cap.tools.executions],
                         "restarts": [{"decisions": r["decisions"],
                                       "swallowed": r["swallowed_refusals"]} for r in cap.restarts]}
        say_(f"[{name}] wiring={wiring} outcome={cap.outcome}")
        say_(f"  model calls {len(cap.model_calls)}, operations {len(cap.ops)}, not executed {refused}")
        say_(f"  executed by the tool registry: {summary[name]['executions']}")
        say_(f"  report.py {q3['workspace'].get('report.py')}; exports {summary[name]['exports']}; "
             f"outbox {summary[name]['outbox']}")
        for c in cap.completion:
            say_(f"  completion at call {c['call']}: complete={c['complete']} missing={c['missing']}")
            if "revised_verifier_missing" in c:
                say_(f"    revised verifier on the same records: complete={c['revised_verifier_complete']} "
                     f"missing={c['revised_verifier_missing']}")
        for r in cap.restarts:
            say_(f"  restart after {r['point']}: before={r['before']} decisions={r['decisions']} "
                 f"new holder={r['new_holder']} rebuilt={r['rebuilt_conversation_matches']}")
            if r["swallowed_refusals"]:
                say_(f"    refusals swallowed by the glue: {r['swallowed_refusals']}")
        if name == "capstone":
            say_("")
            say_(render_record(rec).rstrip())
            say_("")
            for e in cap.log.events:
                keep = {k: v for k, v in e.items() if k not in ("seq",)}
                say_("  " + json.dumps(keep, sort_keys=False))
            say_("")
            say_("  activation check (lexical stand-in):", cap.activation_check["chosen"],
                 [(s["skill"], s["score"]) for s in cap.activation_check["ranking"]])
            say_("  retrieval stages:", cap.retrieval["stages"])
            say_("  retrieval top:", [t["chunk_id"] + " " + t["section"] + " " + t["content_hash"]
                                      for t in cap.retrieval["top"]])
            say_("  memory included:", [(m["id"], m["key"], m["scope"]) for m in cap.memory["included"]])
            say_("  memory excluded:", [(m["id"], m["reason"]) for m in cap.memory["excluded"]])
            say_("  memory writes:", [(w["key"], w["status"], w.get("reasons")) for w in cap.memory_writes])
            say_("  skill scope:", cap.skill_scope)
            say_("  manifest:", json.dumps(cap.manifest()))
            for w in cap.workers:
                say_(f"  worker {w['contract_id']} at {w['revision']}: {w['result']['status']} "
                     f"{(w['result']['result'] or {}).get('verdict')} dispatched {w['dispatched']} "
                     f"returned {w.get('returned')} tokens {w['result']['ledger']['tokens_total']} "
                     f"sim_s {w['result']['ledger']['sim_seconds']}")
            say_("  review integration:", json.dumps({k: cap.review[k] for k in
                                                        ("current_revision", "verdict", "admitted", "stale",
                                                         "missing_coverage")}))
            for a in cap.approvals.table("approvals"):
                say_("  approval:", json.dumps({k: a[k] for k in ("approval_id", "request_id", "actor", "role",
                                                                 "args_digest", "artifact_version", "status",
                                                                 "consumed_by")}))
            for h in cap.human:
                say_("  human:", json.dumps(h))
            last = cap.model_calls[-1]
            say_(f"  last call provenance ({last['request_sha']}):")
            for p in last["provenance"]:
                say_(f"    {p['transform']:<18} {str(p['source'])[:70]:<70} rev={p.get('revision')} "
                     f"trust={p['trust']} ~{p['tokens']}")
        say_("")
    write_run(out, "summary.json", summary)
    with open(os.path.join(out, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(say_.lines) + "\n")


if __name__ == "__main__":
    main()
