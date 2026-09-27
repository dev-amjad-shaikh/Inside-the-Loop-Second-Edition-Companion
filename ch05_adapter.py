"""Chapter 5: the model adapter.

Replays hand-authored SSE streams (fixtures/ch05_streams) through a
provider-neutral adapter and through a flawed one, and executes whatever each
adapter releases against a real temporary copy of the reporting workspace.

    python3 ch05_adapter.py --output runs/ch05

Files written to the output directory:
  outcomes.json            one explicit outcome per recorded stream
  deltas.json              the fix call, delta by delta: parse status and executability
  workspace_experiment.json  flawed vs improved adapter on the interrupted fix, with retries
  report_py_flawed.txt, report_py_improved.txt   report.py after each experiment
  truncation_and_unknown.json  flawed vs improved on max_tokens and an unknown block
  followups.json           next-call history with and without reasoning blocks
  barrier.json             response barrier vs block barrier on a cut two-tool stream
  translation.json         runtime request -> provider requests, with dropped/rejected fields
  contract_tests.json      expected outcome per recorded stream, both adapters
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import Workspace, report_tools, write_run, content_version  # noqa: E402
from itl.ch05_adapter import (assemble, FlawedAdapter, strict_object, lenient_parse,  # noqa: E402
                              translate_request, build_followup, check_followup, sha, REQUIRED)

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "ch05_streams")

SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: settings.json, report.py, test_report.py.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")
NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''


def load(name):
    with open(os.path.join(FX, name + ".sse"), encoding="utf-8") as f:
        return f.read()


def prepared_workspace():
    """The workspace as it stood before Chapter 3's call 5: regression test added, fix not yet made."""
    ws = Workspace("report")
    tools = report_tools(ws)
    tools.execute_checked({"id": "c4", "name": "replace_text", "arguments": {
        "path": "test_report.py", "old": '\n\nif __name__ == "__main__":', "new": "\n" + NEW_TEST}})
    tools.executions.clear()
    return ws, tools


def short_result(r):
    keep = ("status", "message", "replacements", "before_version", "after_version",
            "tests_run", "failures", "errors", "failed_tests", "exit_code")
    return {k: r[k] for k in keep if k in r}


def import_check(ws):
    """Import report.py in a fresh interpreter and keep the last line of any error."""
    proc = subprocess.run([sys.executable, "-c", "import report"], cwd=ws.root,
                          capture_output=True, text=True, timeout=30)
    lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
    return {"exit_code": proc.returncode, "error": lines[-1] if lines else None}


def run_tests(tools):
    return short_result(tools.execute_checked({"id": "verify", "name": "run_tests",
                                               "arguments": {"tests": ["test_report.py"]}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(FX, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = list(meta["fixtures"])

    # 1. every recorded stream becomes one explicit outcome ------------------
    outcomes = {}
    for n in names:
        o, _ = assemble(load(n), n, meta)
        outcomes[n] = o
    write_run(out, "outcomes.json", outcomes)
    print("== Outcomes (response barrier) ==")
    print(f"{'fixture':19} {'raw stop':32} {'outcome':22} exec  model reported")
    for n, o in outcomes.items():
        print(f"{n:19} {str(o['raw_stop_reason']):32} {o['kind']:22} {len(o['executable_calls']):>4}  "
              f"{o['reported_model']}")

    # 2. the fix call, delta by delta ----------------------------------------
    deltas = {}
    for n in ("b_fix_done", "b_fix_cut"):
        trace = []
        o, asm = assemble(load(n), n, meta, trace=trace)
        rows, acc, flawed_fired = [], "", False
        for e in trace:
            if e["kind"] == "args_delta":
                acc += e["fragment"]
                lp = lenient_parse(acc)
                has_req = lp is not None and all(k in lp for k in REQUIRED["replace_text"])
                fires = has_req and not flawed_fired
                flawed_fired = flawed_fired or has_req
                rows.append({"t": e["t"], "event": "args_delta", "args_bytes": len(acc.encode("utf-8")),
                             "strict_parse": strict_object(acc) is not None,
                             "lenient_keys": sorted(lp.keys()) if lp else [],
                             "lenient_new": (lp or {}).get("new"),
                             "flawed_dispatches_here": fires, "executable": False})
            elif e["kind"] in ("block_stop", "stop", "terminal"):
                rows.append({"t": e["t"], "event": e["kind"] + (f" index {e['index']}" if "index" in e else ""),
                             "executable": e["kind"] == "terminal" and bool(o["executable_calls"])})
        rows.append({"t": o["timing_ms"]["last_event"], "event": "end of input",
                     "transport_end": o["transport_end"], "discarded_bytes": o["discarded_bytes"],
                     "outcome": o["kind"]})
        deltas[n] = rows
    write_run(out, "deltas.json", deltas)
    print("\n== Fix call, argument deltas (b_fix_cut) ==")
    for r in deltas["b_fix_cut"]:
        if r["event"] == "args_delta":
            print(f"t={r['t']:>5}  bytes={r['args_bytes']:>3}  strict={str(r['strict_parse']):5}  "
                  f"lenient_keys={r['lenient_keys']}  flawed_dispatch={r['flawed_dispatches_here']}")
        else:
            print(f"t={r['t']:>5}  {r['event']}  " + json.dumps({k: v for k, v in r.items()
                                                                 if k not in ('t', 'event')}))
    first = [r for r in deltas["b_fix_done"] if r.get("flawed_dispatches_here")][0]
    print("flawed adapter would dispatch at t=%d with new=%r" % (first["t"], first["lenient_new"]))
    ex = [r for r in deltas["b_fix_done"] if r.get("executable")]
    print("improved adapter: b_fix_done executable at t=%d" % ex[0]["t"])

    # 3. workspace experiment: interrupted fix, flawed vs improved, with retries
    exp = {}
    ws, tools = prepared_workspace()
    exp["start_versions"] = {"report.py": content_version(ws.read("report.py")),
                             "test_report.py": content_version(ws.read("test_report.py"))}
    exp["start_tests"] = run_tests(tools)
    tools.executions.clear()
    results = []
    flawed = FlawedAdapter(lambda call: results.append({"call": call,
                                                         "result": short_result(tools.execute_checked(call))}))
    r1 = flawed.run(load("b_fix_cut"), "item")
    fl = {"attempt_1_stream": "b_fix_cut", "adapter_status": r1["status"], "dispatch_log": r1["dispatched"],
          "executions": list(results), "report_py_version": content_version(ws.read("report.py")),
          "tests_after": run_tests(tools)}
    results.clear()
    r2 = flawed.run(load("b_fix_done"), "item")  # operator reruns the step
    fl["import_check"] = import_check(ws)
    fl["rerun_stream"] = "b_fix_done"
    fl["rerun_dispatch_log"] = r2["dispatched"]
    fl["rerun_executions"] = list(results)
    fl["report_py_version_final"] = content_version(ws.read("report.py"))
    fl["tests_final"] = run_tests(tools)
    fl["tool_executions_total"] = len([e for e in tools.executions if e["name"] == "replace_text"])
    write_run(out, "report_py_flawed.txt", ws.read("report.py"))
    exp["flawed"] = fl
    ws.close()

    ws, tools = prepared_workspace()
    request_body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}],
                    "tools": [d["name"] for d in tools.definitions()], "note": "same request for both attempts"}
    attempts, executed = [], []
    for attempt, stream in enumerate(["b_fix_cut", "b_fix_done"], start=1):
        o, asm = assemble(load(stream), stream, meta)
        dispatched_from_this = 0
        rec = {"attempt": attempt, "stream": stream, "request_sha": sha(request_body),
               "response_id": o["response_id"], "outcome": o["kind"], "cause": o["cause"],
               "usage": o["usage"], "executable_calls": len(o["executable_calls"])}
        if o["kind"] == "completed_tool_calls":
            for call in o["executable_calls"]:
                executed.append({"call": call["id"], "result": short_result(tools.execute_checked(call))})
                dispatched_from_this += 1
            rec["dispatched"] = dispatched_from_this
            attempts.append(rec)
            break
        retry_ok = o["kind"] == "interrupted" and dispatched_from_this == 0 and not o["released_early"]
        rec["adapter_retry_allowed"] = retry_ok
        rec["retry_reason"] = ("no tool call from this response was released and no final text was shown"
                               if retry_ok else "not retryable at the adapter")
        attempts.append(rec)
        if not retry_ok:
            break
    im = {"attempts": attempts, "executions": executed,
          "report_py_version": content_version(ws.read("report.py")), "tests_after": run_tests(tools),
          "tool_executions_total": len([e for e in tools.executions if e["name"] == "replace_text"])}
    write_run(out, "report_py_improved.txt", ws.read("report.py"))
    # carry the improved workspace forward for the follow-up experiment
    tests_obs = tools.execute_checked({"id": "toolu_tests_2", "name": "run_tests",
                                       "arguments": {"tests": ["test_report.py"]}})
    exp["improved"] = im
    ws.close()
    write_run(out, "workspace_experiment.json", exp)
    print("\n== Interrupted fix against the workspace ==")
    print("start: report.py", exp["start_versions"]["report.py"], "tests", exp["start_tests"]["status"],
          f"({exp['start_tests']['tests_run']} run, {exp['start_tests']['failures']} failure)")
    d = fl["dispatch_log"][0]
    print(f"flawed: status={fl['adapter_status']} dispatched {d['dispatched']['id']} at t={d['t']} "
          f"after {d['args_bytes_seen']} bytes; new={d['dispatched']['arguments']['new']!r}")
    print("flawed: execution ->", fl["executions"][0]["result"])
    print("flawed: tests ->", fl["tests_after"])
    print("flawed: import report ->", fl["import_check"])
    print("flawed rerun on completed stream: execution ->", fl["rerun_executions"][0]["result"])
    print("flawed: final report.py", fl["report_py_version_final"], "tests", fl["tests_final"]["status"])
    for a in im["attempts"]:
        print(f"improved attempt {a['attempt']}: {a['stream']} -> {a['outcome']}"
              f" (cause={a['cause']}) request_sha={a['request_sha']} response_id={a['response_id']}"
              + (f" adapter_retry_allowed={a['adapter_retry_allowed']}" if "adapter_retry_allowed" in a else "")
              + f" usage={a['usage']}")
    print("improved: execution ->", im["executions"][0]["result"])
    print("improved: tests ->", im["tests_after"], "| replace_text executions:", im["tool_executions_total"])

    # 4 and 5. truncation and unknown blocks --------------------------------
    tu = {}
    for n in ("a_answer_length", "b_length_mid_args", "a_unknown_block", "b_filtered", "a_refusal"):
        fmt = meta["fixtures"][n]["format"]
        seen = []
        f = FlawedAdapter(lambda call: seen.append(call)).run(load(n), fmt)
        o = outcomes[n]
        tu[n] = {"flawed": {"status": f["status"], "text": f["text"], "dispatched": [x["dispatched"] for x in f["dispatched"]],
                            "kept_block_types": f["kept_block_types"]},
                 "improved": {k: o[k] for k in ("kind", "cause", "raw_stop_reason", "stop_detail", "text",
                                                "not_executable", "unknown_blocks", "unknown_events",
                                                "ignored_events", "usage")}}
    write_run(out, "truncation_and_unknown.json", tu)
    print("\n== Truncation, refusal and unknown content ==")
    for n, v in tu.items():
        print(f"{n}: flawed status={v['flawed']['status']} dispatched={len(v['flawed']['dispatched'])} "
              f"| improved kind={v['improved']['kind']} cause={v['improved']['cause']}")
    print("a_answer_length text ends:", repr(tu["a_answer_length"]["improved"]["text"][-40:]))
    print("a_unknown_block unknown blocks:", [b["type"] for b in tu["a_unknown_block"]["improved"]["unknown_blocks"]],
          "unknown events:", tu["a_unknown_block"]["improved"]["unknown_events"])

    # 6. follow-up history: reasoning blocks travel back unchanged ----------
    fu = {}
    for n, results_for in (("a_tests_done", [("toolu_tests_2", short_result(tests_obs))]),
                           ("b_fix_done", [("call_fix_1", im["executions"][0]["result"])])):
        o, asm = assemble(load(n), n, meta)
        faithful = build_followup(o, asm, results_for, faithful=True)
        lossy = build_followup(o, asm, results_for, faithful=False)
        fu[n] = {"faithful": {"history": faithful, "check": check_followup(o, asm, faithful)},
                 "flawed": {"history": lossy, "check": check_followup(o, asm, lossy)},
                 "original_signature_sha": [r["signature_sha"] for r in o["reasoning"]]}
    write_run(out, "followups.json", fu)
    print("\n== Follow-up history (stand-in contract check) ==")
    for n, v in fu.items():
        print(f"{n}: faithful -> {v['faithful']['check']['verdict']} "
              f"({v['faithful']['check']['reasoning_blocks_sent_back']}/{v['faithful']['check']['reasoning_blocks_in_response']}); "
              f"flawed -> {v['flawed']['check']['verdict']} "
              f"({v['flawed']['check']['reasoning_blocks_sent_back']}/{v['flawed']['check']['reasoning_blocks_in_response']})")

    # 7. response barrier vs block barrier ------------------------------------
    br = {}
    for barrier in ("response", "block"):
        ws, tools = prepared_workspace()
        journal = []

        def release(call, t, tools=tools, journal=journal):
            journal.append({"t": t, "step": "dispatched", "call_id": call["id"]})
            r = short_result(tools.execute_checked(call))
            journal.append({"t": t, "step": "observed", "call_id": call["id"], "status": r["status"]})

        o, _ = assemble(load("a_two_tools_cut"), "a_two_tools_cut", meta, barrier=barrier,
                        on_release=release, journal=journal)
        br[barrier] = {"outcome": o["kind"], "cause": o["cause"], "released_early": o["released_early"],
                       "executions": [e["name"] + ":" + e["call_id"] for e in tools.executions],
                       "journal": journal, "not_executable": [{k: x[k] for k in ("id", "name", "complete_block")}
                                                              for x in o["not_executable"]]}
        ws.close()
    write_run(out, "barrier.json", br)
    print("\n== Barrier comparison on a_two_tools_cut ==")
    for b, v in br.items():
        print(f"{b:8} outcome={v['outcome']} executions={v['executions']} journal={[j['step'] for j in v['journal']]}")

    # 8. request translation ---------------------------------------------------
    ws, tools = prepared_workspace()
    req = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}],
           "tools": tools.definitions(),
           "settings": {"max_output_tokens": 1024, "temperature": 0.2, "parallel_tool_calls": False}}
    ws.close()
    tr = {m: translate_request(req, m, meta) for m in ("coder-large", "coder-pro", "coder-mini")}
    write_run(out, "translation.json", tr)
    print("\n== Request translation ==")
    for m, v in tr.items():
        print(f"{m}: sent={v['sent']} carried={v['carried_settings']} dropped={[x['field'] for x in v['dropped']]} "
              f"rejected={[x['field'] for x in v['rejected']]} schemas {v['schemas_sha_in']}->{v['schemas_sha_out']}")

    # 9. contract tests over the recorded streams ----------------------------
    ct = {"improved": {}, "flawed": {}}
    for n in names:
        want = meta["expected"][n]
        o = outcomes[n]
        got = {"kind": o["kind"], "executable": len(o["executable_calls"])}
        if "arguments_sha" in want:
            got["arguments_sha"] = [sha(c["arguments"]) for c in o["executable_calls"]]
        ct["improved"][n] = {"expected": want, "got": got, "pass": got == want}
        seen = []
        f = FlawedAdapter(lambda call: seen.append(call)).run(load(n), meta["fixtures"][n]["format"])
        fgot = {"kind": "completed_tool_calls" if f["dispatched"] else "completed_text",
                "executable": len(f["dispatched"])}
        if "arguments_sha" in want:
            fgot["arguments_sha"] = [sha(x["dispatched"]["arguments"]) for x in f["dispatched"]]
        ct["flawed"][n] = {"expected": want, "got": fgot, "pass": fgot == want}
    for k in ("improved", "flawed"):
        ct[k + "_passed"] = sum(1 for v in ct[k].values() if v["pass"])
    write_run(out, "contract_tests.json", ct)
    print("\n== Contract tests over recorded streams ==")
    print(f"improved {ct['improved_passed']}/{len(names)}  flawed {ct['flawed_passed']}/{len(names)}")
    print("flawed failures:", [n for n, v in ct["flawed"].items() if not v["pass"]])


if __name__ == "__main__":
    main()
