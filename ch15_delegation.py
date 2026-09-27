"""Chapter 15: delegation and multiple agents.

A coordinator issues typed contracts to read-only reviewers of the reporting
fix, validates what comes back, detects stale results, and integrates. The
same reviews are then run by one worker for comparison.

    python3 ch15_delegation.py --output runs/ch15

Model decisions are scripted; each reviewer's final JSON is composed by a
deterministic function from the observations it received. Files, boundary
cases and interface inspection are executed for real against materialized
revisions. Token counts are approximate (characters / 4). Latency is
simulated from the constants in itl/ch15_delegation.py (SIM).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import write_run  # noqa: E402
from itl import ch15_delegation as d  # noqa: E402

TZ_OBJECTIVE = ("Check revision {rev} of report.py for timezone boundary errors in belongs_to_report "
                "and return concrete failing cases.")
IFACE_OBJECTIVE = ("Check that revision {rev} keeps the public signature of belongs_to_report unchanged "
                   "relative to {ref}.")
TZ_COVERAGE = ["pdt_evening", "pdt_after_local_midnight", "pst_evening",
               "dst_fall_back_evening", "dst_spring_forward_evening"]
IFACE_COVERAGE = ["public_interface"]
REQUIRED = TZ_COVERAGE + IFACE_COVERAGE
COORD_SYSTEM = ("You coordinate reviews of the reporting fix. Integrate only admitted results. "
                "Report missing evidence as missing.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, prepare a fix and "
        "run the tests. Do not deploy it. Keep the public function signature unchanged.")


def tz_contract(store, rev, budget, cid=None, coverage=TZ_COVERAGE, worker="tz-boundary-reviewer"):
    return d.make_contract(cid or f"review-tz-{rev}", worker, TZ_OBJECTIVE.format(rev=rev), rev, store,
                           ["read_file", "run_cases"], coverage, budget, worker_version="1.2.0")


def iface_contract(store, rev, budget, ref="r3"):
    return d.make_contract(f"review-iface-{rev}", "interface-reviewer", IFACE_OBJECTIVE.format(rev=rev, ref=ref),
                           rev, store, ["read_file", "inspect_interface"], IFACE_COVERAGE, budget,
                           reference_revision=ref, worker_version="1.0.3")


def coordinator_call(results_payload, verdict):
    """One integration model call; token counts come from the actual payload."""
    messages = [{"role": "system", "content": COORD_SYSTEM}, {"role": "user", "content": TASK},
                {"role": "user", "content": json.dumps({"admitted_results": results_payload})}]
    reply = {"role": "assistant", "content": f"Integration verdict: {verdict}.", "tool_calls": []}
    tin, tout = d.approx_tokens(messages), d.approx_tokens(reply)
    return {"tokens_in": tin, "tokens_out": tout, "tokens_total": tin + tout,
            "sim_seconds": d.model_seconds(tin, tout)}


def summary_of(o):
    return {"worker": o["worker"], "worker_version": o["worker_version"],
            "contract_version": o["contract_version"], "input_revision": o["input_revision"],
            "status": o["status"], "verdict": o["result"]["verdict"] if o["result"] else None,
            "attempts": o["attempts"], "tokens": o["ledger"]["tokens_total"],
            "model_calls": o["ledger"]["model_calls"], "sim_seconds": o["ledger"]["sim_seconds"],
            "operations": [e["name"] for e in o["executions"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    os.makedirs(out, exist_ok=True)
    store = d.RevisionStore()
    revs = {k: {"versions": v["versions"], "digest": v["digest"]} for k, v in store.revs.items()}
    write_run(out, "revisions.json", revs)
    print("revisions:")
    for k in ("r3", "r4", "r5", "r5-alt"):
        print(f"  {k:7} report.py {revs[k]['versions']['report.py']}  "
              f"test_report.py {revs[k]['versions']['test_report.py']}  digest {revs[k]['digest']}")

    BUDGET = 16000
    RESERVE = 0.25

    # ------------------------------------------------------------------
    # Experiment 1: contracts and validation at r5.
    # ------------------------------------------------------------------
    log = d.EvidenceLog()
    c_tz = tz_contract(store, "r5", 6000)
    c_if = iface_contract(store, "r5", 3000)
    write_run(out, "contracts.json", [c_tz, c_if])
    o_tz = d.run_worker(c_tz, d.script_tz(c_tz, d.BOUNDARY_CASES, final=d.fabricated_result(c_tz)), store, log,
                        repair_script=[d.compose_result(c_tz)])
    o_if = d.run_worker(c_if, d.script_iface(c_if, final=d.prose_result(
        "Looks fine to me. The signature of belongs_to_report is unchanged and the timezone logic reads "
        "correctly.")), store, log, repair_script=[d.compose_result(c_if)])
    integ1 = d.integrate([o_tz, o_if], "r5", REQUIRED)
    coord1 = coordinator_call([o_tz["result"], o_if["result"]], integ1["verdict"])
    exp1 = {"workers": [summary_of(o_tz), summary_of(o_if)], "integration": integ1, "coordinator": coord1,
            "findings": {o["worker"]: o["result"]["findings"] for o in (o_tz, o_if)},
            "results": {o["worker"]: o["result"] for o in (o_tz, o_if)}}
    write_run(out, "exp1_contracts.json", exp1)
    write_run(os.path.join(out, "traces"), "exp1_tz.json", o_tz["trace"])
    write_run(os.path.join(out, "traces"), "exp1_iface.json", o_if["trace"])
    write_run(out, "evidence_log.json", log.items)
    print("\n[1] contract validation at r5")
    for o in (o_tz, o_if):
        for a in o["attempts"]:
            print(f"  {o['worker']} attempt {a['attempt']}: "
                  f"{'accepted' if not a['violations'] else 'rejected: ' + '; '.join(a['violations'])}")
    print("  integration:", integ1["verdict"], "| admitted:", integ1["admitted"])

    # ------------------------------------------------------------------
    # Experiment 2: a stale result. Reviewer dispatched at r4; the coordinator
    # applies the fix (r5) while the review runs.
    # ------------------------------------------------------------------
    log2 = d.EvidenceLog()
    c_old = tz_contract(store, "r4", 6000, cid="review-tz-r4")
    o_old = d.run_worker(c_old, d.script_tz(c_old, d.BOUNDARY_CASES), store, log2)
    fix_at = 3.0
    timeline = [
        {"t": 0.0, "event": "coordinator at r4 dispatches tz-boundary-reviewer with input_revision r4"},
        {"t": fix_at, "event": "coordinator applies fix: report.py " + revs["r4"]["versions"]["report.py"]
         + " -> " + revs["r5"]["versions"]["report.py"] + "; current revision r5"},
        {"t": o_old["ledger"]["sim_seconds"], "event": "tz-boundary-reviewer returns result for r4: "
         + o_old["result"]["verdict"]},
    ]
    stale_integ = d.integrate([o_old, o_if], "r5", REQUIRED)
    naive = d.naive_integrate([o_old, o_if])
    naive_claim = ("r5 has defects: " + "; ".join(f["claim"] for f in o_old["result"]["findings"]
                                                  if f["severity"] == "defect"))
    c_new = tz_contract(store, "r5", 6000, cid="review-tz-r5-redispatch")
    o_new = d.run_worker(c_new, d.script_tz(c_new, d.BOUNDARY_CASES), store, log2)
    t_new = round(o_old["ledger"]["sim_seconds"] + o_new["ledger"]["sim_seconds"], 3)
    timeline.append({"t": o_old["ledger"]["sim_seconds"], "event": "coordinator marks r4 result stale; "
                     "re-dispatches tz-boundary-reviewer at r5"})
    timeline.append({"t": t_new, "event": "tz-boundary-reviewer returns result for r5: " + o_new["result"]["verdict"]})
    final2 = d.integrate([o_old, o_new, o_if], "r5", REQUIRED)
    exp2 = {"timeline": timeline, "stale_result": summary_of(o_old),
            "stale_findings": [f["claim"] for f in o_old["result"]["findings"]],
            "integration_with_stale_only": stale_integ, "naive_coordinator": dict(naive, claim=naive_claim),
            "redispatch": summary_of(o_new), "final_integration": final2,
            "extra_tokens_for_redispatch": o_new["ledger"]["tokens_total"]}
    write_run(out, "exp2_stale.json", exp2)
    print("\n[2] stale result")
    for ev in timeline:
        print(f"  t={ev['t']:6.2f}s  {ev['event']}")
    print("  with stale result only:", stale_integ["verdict"], "| stale:", stale_integ["stale"])
    print("  naive coordinator:", naive["verdict"], naive["votes"])
    print("  after re-dispatch:", final2["verdict"])

    # ------------------------------------------------------------------
    # Experiment 3: shared state. Both reviewers write their result in the
    # same step, in simulated finish order.
    # ------------------------------------------------------------------
    finish_order = sorted([o_new, o_if], key=lambda o: o["ledger"]["sim_seconds"])
    exp3 = {"finish_order": [(o["worker"], o["ledger"]["sim_seconds"]) for o in finish_order]}
    print("\n[3] shared state (finish order:", ", ".join(f"{w} {s}s" for w, s in exp3["finish_order"]) + ")")
    for policy in ("last_write_wins", "reject_second_write", "namespaced"):
        st = d.SharedState(policy)
        st.begin_step()
        for o in finish_order:
            st.write(o["worker"], "review_result",
                     {"worker": o["worker"], "verdict": o["result"]["verdict"],
                      "input_revision": o["input_revision"]})
        visible = {k: v["worker"] for k, v in st.values.items()}
        exp3[policy] = {"log": st.log, "keys_after_step": visible}
        print(f"  {policy:20} keys={visible}  log={[x['outcome'] for x in st.log]}")
    write_run(out, "exp3_shared_state.json", exp3)

    # ------------------------------------------------------------------
    # Experiment 4: correlated agreement over a flawed patch (r5-alt).
    # ------------------------------------------------------------------
    log5 = d.EvidenceLog()
    brief_cov = ["pdt_midday", "pdt_evening"]
    cA = tz_contract(store, "r5-alt", 6000, cid="brief-A", coverage=brief_cov, worker="brief-reviewer-A")
    cB = tz_contract(store, "r5-alt", 6000, cid="brief-B", coverage=brief_cov, worker="brief-reviewer-B")
    cI = iface_contract(store, "r5-alt", 3000)
    oA = d.run_worker(cA, d.script_tz(cA, d.BRIEF_CASES), store, log5)
    oB = d.run_worker(cB, d.script_tz(cB, d.BRIEF_CASES), store, log5)
    oI = d.run_worker(cI, d.script_iface(cI), store, log5)
    flawed_votes = d.naive_integrate([oA, oB, oI])
    evidence_sets = {}
    for o in (oA, oB):
        key = tuple(sorted((r["payload"]["case"], r["payload"]["actual"]) for r in log5.items.values()
                           if r["worker"] == o["worker"] and r["operation"] == "run_cases"))
        evidence_sets.setdefault(key, []).append(o["worker"])
    revised_flawed_team = d.integrate([oA, oB, oI], "r5-alt", REQUIRED)
    cT = tz_contract(store, "r5-alt", 6000)
    oT = d.run_worker(cT, d.script_tz(cT, d.BOUNDARY_CASES), store, log5)
    revised = d.integrate([oT, oI], "r5-alt", REQUIRED)
    classes = ["pdt_midday"] + REQUIRED
    matrix = {}
    for o in (oA, oB, oI, oT):
        row = {c: "absent" for c in classes}
        for item in log5.items.values():
            if item["worker"] == o["worker"] and item["coverage"] in row:
                ok = item["payload"].get("ok", item["payload"].get("unchanged"))
                row[item["coverage"]] = "ok" if ok else "fail"
        matrix[o["worker"]] = {"cells": row, "verdict": o["result"]["verdict"]}
    exp4 = {"patch": revs["r5-alt"]["versions"]["report.py"], "flawed_team_votes": flawed_votes,
            "coverage_matrix": {"classes": classes, "rows": matrix},
            "distinct_case_evidence_sets": [{"cases": list(k), "workers": v} for k, v in evidence_sets.items()],
            "coverage_check_on_flawed_team": revised_flawed_team,
            "boundary_reviewer": summary_of(oT), "revised_integration": revised}
    write_run(out, "exp4_correlated.json", exp4)
    print("\n[4] correlated agreement on r5-alt (hard-coded offset)")
    print("  flawed team votes:", flawed_votes)
    print("  distinct case-evidence sets among brief reviewers:", len(evidence_sets),
          [v for v in evidence_sets.values()])
    print("  coverage check on the same results:", revised_flawed_team["verdict"])
    print("  boundary reviewer:", oT["result"]["verdict"], "| defects:")
    for c in revised["defects"]:
        print("    -", c)

    # ------------------------------------------------------------------
    # Experiment 5: one worker vs two workers at r5, clean and with one
    # malformed first answer.
    # ------------------------------------------------------------------
    comp = {}
    for fault in (False, True):
        log6 = d.EvidenceLog()
        ct = tz_contract(store, "r5", 6000)
        ci = iface_contract(store, "r5", 3000)
        cs = d.make_contract("review-both-r5", "combined-reviewer",
                             TZ_OBJECTIVE.format(rev="r5") + " " + IFACE_OBJECTIVE.format(rev="r5", ref="r3"),
                             "r5", store, ["read_file", "run_cases", "inspect_interface"], REQUIRED, 9000,
                             reference_revision="r3", worker_version="1.0.0")
        prose = d.prose_result("Both reviews look fine.")
        ot = d.run_worker(ct, d.script_tz(ct, d.BOUNDARY_CASES), store, log6)
        oi = d.run_worker(ci, d.script_iface(ci, final=prose if fault else None), store, log6,
                          repair_script=[d.compose_result(ci)] if fault else None)
        os_ = d.run_worker(cs, d.script_combined(cs, final=prose if fault else None), store, log6,
                           repair_script=[d.compose_result(cs)] if fault else None)
        two_int = d.integrate([ot, oi], "r5", REQUIRED)
        one_int = d.integrate([os_], "r5", REQUIRED)
        two_coord = coordinator_call([ot["result"], oi["result"]], two_int["verdict"])
        one_coord = coordinator_call([os_["result"]], one_int["verdict"])
        two_sched = d.schedule({"tz-boundary-reviewer": (ot["ledger"]["sim_seconds"], []),
                                "interface-reviewer": (oi["ledger"]["sim_seconds"], []),
                                "integrate": (two_coord["sim_seconds"], ["tz-boundary-reviewer",
                                                                         "interface-reviewer"])})
        one_sched = d.schedule({"combined-reviewer": (os_["ledger"]["sim_seconds"], []),
                                "integrate": (one_coord["sim_seconds"], ["combined-reviewer"])})
        key = "with_malformed_first_answer" if fault else "clean"
        comp[key] = {
            "two_workers": {"latency_s": two_sched["parallel_s"], "schedule": two_sched,
                            "tokens": ot["ledger"]["tokens_total"] + oi["ledger"]["tokens_total"]
                            + two_coord["tokens_total"],
                            "worker_tokens": {"tz-boundary-reviewer": ot["ledger"]["tokens_total"],
                                              "interface-reviewer": oi["ledger"]["tokens_total"]},
                            "coordinator_tokens": two_coord["tokens_total"],
                            "worker_steps": {o["worker"]: o["ledger"]["steps"] for o in (ot, oi)},
                            "model_calls": ot["ledger"]["model_calls"] + oi["ledger"]["model_calls"] + 1,
                            "file_reads": sum(1 for o in (ot, oi) for e in o["executions"]
                                              if e["name"] == "read_file"),
                            "verdict": two_int["verdict"], "defects": two_int["defects"],
                            "repair_attempts": sum(len(o["attempts"]) - 1 for o in (ot, oi))},
            "one_worker": {"latency_s": one_sched["parallel_s"], "schedule": one_sched,
                           "tokens": os_["ledger"]["tokens_total"] + one_coord["tokens_total"],
                           "worker_tokens": {"combined-reviewer": os_["ledger"]["tokens_total"]},
                           "coordinator_tokens": one_coord["tokens_total"],
                           "worker_steps": {os_["worker"]: os_["ledger"]["steps"]},
                           "model_calls": os_["ledger"]["model_calls"] + 1,
                           "file_reads": sum(1 for e in os_["executions"] if e["name"] == "read_file"),
                           "verdict": one_int["verdict"], "defects": one_int["defects"],
                           "repair_attempts": len(os_["attempts"]) - 1},
        }
        if fault:
            comp[key]["repair_cost"] = {
                "two_workers_interface_repair_call": oi["ledger"]["steps"][-1],
                "one_worker_repair_call": os_["ledger"]["steps"][-1]}
    comp["second_edition_schedule"] = d.schedule({
        "read settings.json": (2, []), "read report.py": (3, []), "read test_report.py": (2, []),
        "propose patch": (4, ["read settings.json", "read report.py", "read test_report.py"]),
        "apply patch": (1, ["propose patch"]), "run tests": (5, ["apply patch"])})
    comp["sim_constants"] = d.SIM
    write_run(out, "exp5_comparison.json", comp)
    print("\n[5] one worker vs two workers at r5 (simulated seconds, approximate tokens)")
    for key in ("clean", "with_malformed_first_answer"):
        for arr in ("one_worker", "two_workers"):
            r = comp[key][arr]
            print(f"  {key:28} {arr:11} latency {r['latency_s']:6.2f}s  tokens {r['tokens']:6}  "
                  f"calls {r['model_calls']}  reads {r['file_reads']}  repairs {r['repair_attempts']}  "
                  f"verdict: {r['verdict']}")
    rc = comp["with_malformed_first_answer"]["repair_cost"]
    print("  repair call, interface worker:", rc["two_workers_interface_repair_call"])
    print("  repair call, combined worker: ", rc["one_worker_repair_call"])
    s = comp["second_edition_schedule"]
    print(f"  second-edition schedule: sequential {s['sequential_s']}s, dependency path {s['parallel_s']}s")

    # ------------------------------------------------------------------
    # Experiment 6: budget allocation with a reserve for integration.
    # ------------------------------------------------------------------
    reserve = int(BUDGET * RESERVE)
    pool = BUDGET - reserve
    clean = comp["clean"]["two_workers"]["worker_tokens"]
    estimates = {"tz-boundary-reviewer": clean["tz-boundary-reviewer"],
                 "interface-reviewer": clean["interface-reviewer"],
                 "extra-reviewer": clean["tz-boundary-reviewer"]}
    scale = pool / sum(estimates.values())
    alloc = {k: int(v * scale) for k, v in estimates.items()}
    runs = {}
    for mode in ("naive", "allocated"):
        log4 = d.EvidenceLog()
        caps = {k: BUDGET for k in alloc} if mode == "naive" else alloc
        ct = tz_contract(store, "r5", caps["tz-boundary-reviewer"])
        ci = iface_contract(store, "r5", caps["interface-reviewer"])
        cx = tz_contract(store, "r5", caps["extra-reviewer"], cid="review-extra-r5", worker="extra-reviewer")
        ot = d.run_worker(ct, d.script_tz(ct, d.BOUNDARY_CASES), store, log4)
        oi = d.run_worker(ci, d.script_iface(ci), store, log4)
        ox = d.run_worker(cx, d.script_runaway(cx), store, log4)
        spent = sum(o["ledger"]["tokens_total"] for o in (ot, oi, ox))
        integ = d.integrate([ot, oi, ox], "r5", REQUIRED)
        coord = coordinator_call([o["result"] for o in (ot, oi, ox) if o["result"]], integ["verdict"])
        runs[mode] = {"caps": caps, "per_worker": {o["worker"]: {"tokens": o["ledger"]["tokens_total"],
                                                                 "model_calls": o["ledger"]["model_calls"],
                                                                 "status": o["status"]} for o in (ot, oi, ox)},
                      "workers_total": spent, "left_for_integration": BUDGET - spent,
                      "integration_needs": coord["tokens_total"],
                      "integration_fits": BUDGET - spent >= coord["tokens_total"],
                      "integration_verdict": integ["verdict"]}
    exp6 = {"parent_budget": BUDGET, "reserve": reserve, "pool": pool, "estimates": estimates,
            "allocation": alloc, "runs": runs}
    write_run(out, "exp6_budget.json", exp6)
    print(f"\n[6] budget: parent {BUDGET}, reserve {reserve}, pool {pool}, allocation {alloc}")
    for mode, r in runs.items():
        print(f"  {mode:9} per worker {{" + ", ".join(f"{w}: {v['tokens']} ({v['status']})"
                                                     for w, v in r["per_worker"].items())
              + f"}} total {r['workers_total']}; left {r['left_for_integration']}; "
                f"integration needs {r['integration_needs']}; fits={r['integration_fits']}")

    # Per-worker cost ledger for attribution.
    ledger = [{"worker": o["worker"], "worker_version": o["worker_version"],
               "contract_id": o["contract_id"], "contract_version": o["contract_version"],
               "input_revision": o["input_revision"], "status": o["status"],
               "tokens": o["ledger"]["tokens_total"], "model_calls": o["ledger"]["model_calls"],
               "sim_seconds": o["ledger"]["sim_seconds"]}
              for o in (o_tz, o_if, o_old, o_new, oA, oB, oI, oT)]
    write_run(out, "cost_ledger.json", ledger)
    store.close()


if __name__ == "__main__":
    main()
