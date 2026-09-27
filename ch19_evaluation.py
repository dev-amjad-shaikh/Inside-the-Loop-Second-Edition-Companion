"""Chapter 19: evaluation.

An evaluation harness over 60 scripted-agent task instances from the
reporting and export domains. The agents are seeded stochastic scripted
policies standing in for live models (see itl/ch19_evaluation.py); the loop,
tools, tests and export service are real.

    python3 ch19_evaluation.py --output runs/ch19

Writes JSON results to the output directory and prints the key numbers.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import write_run  # noqa: E402
from itl.ch19_evaluation import (  # noqa: E402
    BASELINE, CANDIDATE, CATEGORIES, make_release_tasks, bootstrap_diff, bootstrap_mean, cohen_kappa, first_errors,
    judge_v1, judge_v1_pairwise, judge_v2, judge_v2_pairwise, make_tasks, mcnemar_exact,
    pass_at_k, pass_hat_k, run_one, seed_for, wilson)

N_TRIALS = 16
KS = list(range(1, 17))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def say(line=""):
    OUT.append(line)
    print(line)


def r3(x):
    return None if x is None else round(x, 3)


def run_set(tasks, params, trials, builder="flawed", keep_trace=False, memo=None):
    return [run_one(t, params, k, builder_mode=builder, keep_trace=keep_trace, memo=memo)
            for t in tasks for k in range(trials)]


def successes(rows, key="end_state_pass"):
    c = defaultdict(int)
    n = defaultdict(int)
    for r in rows:
        c[r["task"]] += 1 if r[key] else 0
        n[r["task"]] += 1
    return c, n


def curves(rows, tasks):
    c, n = successes(rows)
    ids = [t["id"] for t in tasks]
    return {
        "pass@k": [r3(sum(pass_at_k(n[i], c[i], k) for i in ids) / len(ids)) for k in KS],
        "pass^k": [r3(sum(pass_hat_k(n[i], c[i], k) for i in ids) / len(ids)) for k in KS],
    }


def per_task_rate(rows, tasks):
    c, n = successes(rows)
    return [c[t["id"]] / n[t["id"]] for t in tasks]


def per_task_hat(rows, tasks, k):
    c, n = successes(rows)
    return [pass_hat_k(n[t["id"]], c[t["id"]], k) for t in tasks]


def paired_table(rows_a, rows_b, tasks, trial=0):
    a = {r["task"]: r["end_state_pass"] for r in rows_a if r["trial"] == trial}
    b = {r["task"]: r["end_state_pass"] for r in rows_b if r["trial"] == trial}
    cells = Counter()
    regress, improve = [], []
    for t in tasks:
        x, y = a[t["id"]], b[t["id"]]
        cells[(x, y)] += 1
        if x and not y:
            regress.append(t["id"])
        if y and not x:
            improve.append(t["id"])
    return {"both succeed": cells[(True, True)], "candidate only": cells[(False, True)],
            "baseline only": cells[(True, False)], "both fail": cells[(False, False)],
            "regressions": regress, "improvements": improve,
            "mcnemar_p": r3(mcnemar_exact(cells[(True, False)], cells[(False, True)]))}


def component_evidence():
    """Read the evidence earlier chapters' companions produced (read-only)."""
    runs = os.path.join(HERE, "runs")
    out = []

    def load(rel):
        p = os.path.join(runs, rel)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    ctx = load("ch06/context_tests.json")
    if ctx:
        for builder in ("truncate", "structured"):
            checks = [v for v in ctx[builder].values() if isinstance(v, bool)]
            out.append({"chapter": 6, "component": "Context builder", "layer": "runtime contract",
                        "evidence": f"context tests, {builder} builder", "k": sum(checks), "n": len(checks)})
    ev = load("ch07/evaluation.json")
    if ev:
        for idx in ("fixed", "structured"):
            k, n = map(int, ev[idx]["recall@3"].split("/"))
            out.append({"chapter": 7, "component": "Retrieval", "layer": "component (stand-in ranker)",
                        "evidence": f"recall@3, {idx} chunking", "k": k, "n": n})
    sel = load("ch09/selection_summary.json")
    if sel:
        for part in ("development", "held_out"):
            d = sel["revised"][part]
            out.append({"chapter": 9, "component": "Tools and executor", "layer": "decision (lexical stand-in)",
                        "evidence": f"tool selection, revised contracts, {part}",
                        "k": d["correct"], "n": d["correct"] + d["correct_by_tie_break"] + d["wrong"]})
    act = load("ch10/activation.json")
    if act:
        for state in ("A_flawed", "C_both_revised"):
            rows = act["states"][state]["rows"]
            out.append({"chapter": 10, "component": "Skills", "layer": "decision (lexical stand-in)",
                        "evidence": f"activation suite, {state}", "k": sum(r["pass"] for r in rows), "n": len(rows)})
    ctl = load("ch11/summary.json")
    if ctl:
        for controller in ("loop", "hybrid"):
            rows = [r for r in ctl if r["controller"] == controller]
            out.append({"chapter": 11, "component": "Control", "layer": "runtime contract (scripted)",
                        "evidence": f"acceptance passed, {controller} controller",
                        "k": sum(r["acceptance_passed"] for r in rows), "n": len(rows)})
    pol = load("ch16/boundary_suite.json")
    if pol:
        out.append({"chapter": 16, "component": "Policy and authority", "layer": "runtime contract",
                    "evidence": "boundary suite, governed policy",
                    "k": sum(r["governed_pass"] for r in pol), "n": len(pol)})
    for row in out:
        lo, hi = wilson(row["k"], row["n"])
        row["wilson95"] = [r3(lo), r3(hi)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    os.makedirs(out, exist_ok=True)

    tasks = make_tasks()
    by_id = {t["id"]: t for t in tasks}
    dev = [t for t in tasks if t["split"] == "dev"]
    held = [t for t in tasks if t["split"] == "heldout"]
    write_run(out, "tasks.json", [{k: v for k, v in t.items() if k != "hidden_cases"} for t in tasks])
    t2 = by_id["R02"]
    write_run(out, "hidden_reference_R02.json", {
        "task": "R02", "zone": t2["zone"], "correction": t2["correction"], "hidden_cases": t2["hidden_cases"],
        "checks": (["signature_unchanged"] if t2["correction"] else []) +
                  ["boundary_cases", "settings_unchanged", "existing_test_kept"]})
    say("== Task family")
    say(f"tasks {len(tasks)} (report {sum(t['domain'] == 'report' for t in tasks)}, "
        f"export {sum(t['domain'] == 'export' for t in tasks)}); dev {len(dev)}, held-out {len(held)}")
    say("slices (held-out): " + ", ".join(f"{s} {c}" for s, c in sorted(Counter(t['slice'] for t in held).items())))

    # ------------------------------------------------------------------ E1
    say("\n== E1 pass@k and pass^k on held-out tasks (16 trials each, flawed context builder)")
    base = run_set(held, BASELINE, N_TRIALS, keep_trace=True)
    cand = run_set(held, CANDIDATE, N_TRIALS, keep_trace=True)
    with open(os.path.join(out, "runs_heldout.jsonl"), "w", encoding="utf-8") as f:
        for r in base + cand:
            f.write(json.dumps({k: v for k, v in r.items() if k != "trace"}) + "\n")
    cv = {"baseline": curves(base, held), "candidate": curves(cand, held), "k": KS}
    slices = sorted({t["slice"] for t in held})
    cv["by_slice"] = {}
    for s in slices:
        ts = [t for t in held if t["slice"] == s]
        cv["by_slice"][s] = {"baseline": curves([r for r in base if r["slice"] == s], ts),
                             "candidate": curves([r for r in cand if r["slice"] == s], ts)}
    write_run(out, "passk.json", cv)
    say(f"{'k':>3} {'base pass@k':>11} {'cand pass@k':>11} {'base pass^k':>11} {'cand pass^k':>11}")
    for k in (1, 2, 4, 8, 16):
        i = k - 1
        say(f"{k:>3} {cv['baseline']['pass@k'][i]:>11.3f} {cv['candidate']['pass@k'][i]:>11.3f} "
            f"{cv['baseline']['pass^k'][i]:>11.3f} {cv['candidate']['pass^k'][i]:>11.3f}")
    say("pass@1 by slice (baseline / candidate):")
    for s in slices:
        say(f"  {s:<20} {cv['by_slice'][s]['baseline']['pass@k'][0]:.3f} / "
            f"{cv['by_slice'][s]['candidate']['pass@k'][0]:.3f}   pass^4 "
            f"{cv['by_slice'][s]['baseline']['pass^k'][3]:.3f} / {cv['by_slice'][s]['candidate']['pass^k'][3]:.3f}")

    # ------------------------------------------------------------------ E2
    say("\n== E2 paired comparison")
    illus = {"both succeed": 5, "candidate only": 2, "baseline only": 1, "both fail": 2}
    ia = [1] * 5 + [0] * 2 + [1] * 1 + [0] * 2
    ib = [1] * 5 + [1] * 2 + [0] * 1 + [0] * 2
    ci_illus = bootstrap_diff(ia, ib, seed=seed_for("illus") % 10 ** 6)
    illus.update({"baseline rate": 0.6, "candidate rate": 0.7, "mcnemar_p": r3(mcnemar_exact(1, 2)),
                  "bootstrap95_diff": [r3(ci_illus[0]), r3(ci_illus[1])],
                  "wilson95_baseline": [r3(x) for x in wilson(6, 10)],
                  "wilson95_candidate": [r3(x) for x in wilson(7, 10)]})
    say(f"second-edition illustrative table 5/2/1/2: McNemar p={illus['mcnemar_p']}, "
        f"bootstrap 95% CI of difference {illus['bootstrap95_diff']}, "
        f"Wilson 95% baseline {illus['wilson95_baseline']} candidate {illus['wilson95_candidate']}")
    pt = paired_table(base, cand, held, trial=0)
    say(f"held-out, trial 0 only: both {pt['both succeed']}, candidate only {pt['candidate only']}, "
        f"baseline only {pt['baseline only']}, both fail {pt['both fail']}; McNemar p={pt['mcnemar_p']}")
    say(f"  regressions (baseline passed, candidate failed): {', '.join(pt['regressions'])}")
    t0a = [1 if r["end_state_pass"] else 0 for r in base if r["trial"] == 0]
    t0b = [1 if r["end_state_pass"] else 0 for r in cand if r["trial"] == 0]
    d0 = bootstrap_diff(t0a, t0b)
    pt["bootstrap95_diff"] = [r3(d0[0]), r3(d0[1])]
    say(f"  trial-0 difference bootstrap 95% CI {pt['bootstrap95_diff']}")
    ra, rb = per_task_rate(base, held), per_task_rate(cand, held)
    d1 = bootstrap_diff(ra, rb)
    ha, hb = per_task_hat(base, held, 4), per_task_hat(cand, held, 4)
    d4 = bootstrap_diff(ha, hb)
    pa, pb = [pass_at_k(16, round(x * 16), 4) for x in ra], [pass_at_k(16, round(x * 16), 4) for x in rb]
    d4at = bootstrap_diff(pa, pb)
    consistent = [t["id"] for t, x, y in zip(held, ra, rb) if x - y >= 0.5]
    better = [t["id"] for t, x, y in zip(held, ra, rb) if y - x >= 0.5]
    paired = {"illustrative_second_edition": illus, "trial0": pt,
              "pass@1": {"baseline": r3(sum(ra) / len(ra)), "candidate": r3(sum(rb) / len(rb)),
                         "diff": r3((sum(rb) - sum(ra)) / len(ra)), "bootstrap95": [r3(d1[0]), r3(d1[1])],
                         "baseline95": [r3(x) for x in bootstrap_mean(ra)],
                         "candidate95": [r3(x) for x in bootstrap_mean(rb)]},
              "pass^4": {"diff": r3((sum(hb) - sum(ha)) / len(ha)), "bootstrap95": [r3(d4[0]), r3(d4[1])]},
              "pass@4": {"diff": r3((sum(pb) - sum(pa)) / len(pa)), "bootstrap95": [r3(d4at[0]), r3(d4at[1])]},
              "tasks_candidate_worse_by_0.5": consistent, "tasks_candidate_better_by_0.5": better,
              "per_task": [{"task": t["id"], "slice": t["slice"], "baseline": r3(x), "candidate": r3(y)}
                           for t, x, y in zip(held, ra, rb)]}
    write_run(out, "paired.json", paired)
    say(f"pass@1 over 16 trials: baseline {paired['pass@1']['baseline']:.3f} {paired['pass@1']['baseline95']}, "
        f"candidate {paired['pass@1']['candidate']:.3f} {paired['pass@1']['candidate95']}")
    say(f"  difference {paired['pass@1']['diff']:+.3f}, paired bootstrap 95% CI {paired['pass@1']['bootstrap95']}")
    say(f"pass^4 difference {paired['pass^4']['diff']:+.3f}, CI {paired['pass^4']['bootstrap95']}; "
        f"pass@4 difference {paired['pass@4']['diff']:+.3f}, CI {paired['pass@4']['bootstrap95']}")
    say(f"tasks where candidate is worse by >= 0.5: {', '.join(consistent)}")
    say(f"tasks where candidate is better by >= 0.5: {', '.join(better)}")

    # ------------------------------------------------------------------ E3
    say("\n== E3 how sample size and trials change the conclusion (pass@1 difference)")
    rng = random.Random(seed_for("ch19-sample-size") % 10 ** 9)
    res_a = defaultdict(list)
    res_b = defaultdict(list)
    for r in base:
        res_a[r["task"]].append(r["end_state_pass"])
    for r in cand:
        res_b[r["task"]].append(r["end_state_pass"])
    ids = [t["id"] for t in held]
    full = paired["pass@1"]["diff"]
    designs = []
    for m in (10, 20, 40):
        for trials in (1, 4, 16):
            diffs, widths = [], []
            for _ in range(1000):
                pick = rng.sample(ids, m)
                ds = []
                for i in pick:
                    ta = rng.sample(range(N_TRIALS), trials)
                    tb = rng.sample(range(N_TRIALS), trials)
                    ds.append(sum(res_b[i][j] for j in tb) / trials - sum(res_a[i][j] for j in ta) / trials)
                mean = sum(ds) / m
                sd = statistics.pstdev(ds) if m > 1 else 0.0
                diffs.append(mean)
                widths.append(2 * 1.96 * sd / math.sqrt(m))
            designs.append({"tasks": m, "trials": trials,
                            "share_candidate_ahead_by_5pts": r3(sum(d >= 0.05 for d in diffs) / 1000),
                            "share_candidate_behind_by_5pts": r3(sum(d <= -0.05 for d in diffs) / 1000),
                            "median_ci_width": r3(statistics.median(widths))})
    write_run(out, "sample_size.json", {"full_diff": full, "draws_per_design": 1000, "designs": designs,
                                        "ci_method": "normal approximation on per-task paired differences"})
    say(f"full held-out difference {full:+.3f}; 1000 random evaluations per design")
    say(f"{'tasks':>5} {'trials':>6} {'cand +5pts':>10} {'cand -5pts':>10} {'CI width':>9}")
    for d in designs:
        say(f"{d['tasks']:>5} {d['trials']:>6} {d['share_candidate_ahead_by_5pts']:>10.3f} "
            f"{d['share_candidate_behind_by_5pts']:>10.3f} {d['median_ci_width']:>9.3f}")

    # ------------------------------------------------------------------ E4
    say("\n== E4 trajectory checker: right end state, prohibited path")
    traj = {}
    example = None
    for name, rows in (("baseline", base), ("candidate", cand)):
        right_but_bad = [r for r in rows if r["end_state_pass"] and r["violations"]]
        by_rule = Counter((v["rule"], v["executed"]) for r in right_but_bad for v in r["violations"])
        traj[name] = {"runs": len(rows), "end_state_pass": sum(r["end_state_pass"] for r in rows),
                      "end_state_pass_with_violation": len(right_but_bad),
                      "by_rule": {f"{k[0]} ({'executed' if k[1] else 'refused'})": v for k, v in sorted(by_rule.items())}}
        say(f"{name}: {traj[name]['end_state_pass']} of {len(rows)} runs pass the end-state checks; "
            f"{len(right_but_bad)} of those violate a path rule")
        for k, v in traj[name]["by_rule"].items():
            say(f"    {k:<40} {v}")
        if example is None:
            for r in right_but_bad:
                if any(v["rule"] == "no_unrequested_change" for v in r["violations"]):
                    example = r
                    break
    write_run(out, "trajectory.json", traj)
    if example:
        write_run(out, "example_right_end_state_bad_path.json",
                  {"task": example["task"], "config": example["config"], "trial": example["trial"],
                   "trajectory": example["trace"]["trajectory"], "hidden": example["trace"]["hidden"],
                   "violations": example["violations"], "answer": example["answer"]})
        say(f"example: {example['config']} {example['task']} trial {example['trial']}: "
            + " -> ".join(f"{s['name']}[{s['status']}]" for s in example["trace"]["trajectory"]))

    # ------------------------------------------------------------------ E5
    say("\n== E5 judge calibration against hand labels")
    with open(os.path.join(HERE, "fixtures", "ch19_eval", "judge_calibration.json"), encoding="utf-8") as f:
        cal = json.load(f)
    items = cal["items"]
    judge = {"items": [], "summary": {}}
    for it in items:
        v2, why = judge_v2(it["answer"], it["record"])
        judge["items"].append({"id": it["id"], "part": it["part"], "human": it["human_label"],
                               "v1": judge_v1(it["answer"]), "v2": v2, "v2_reasons": why,
                               "chars": len(it["answer"])})
    for part in ("rubric-development", "held-out", "all"):
        rows = [x for x in judge["items"] if part == "all" or x["part"] == part]
        h = [x["human"] for x in rows]
        s = {}
        for j in ("v1", "v2"):
            p = [x[j] for x in rows]
            s[j] = {"agreement": r3(sum(a == b for a, b in zip(h, p)) / len(rows)),
                    "kappa": r3(cohen_kappa(h, p)),
                    "confusion": {f"human {a} / judge {b}": sum(1 for x, y in zip(h, p) if x == a and y == b)
                                  for a in ("pass", "fail") for b in ("pass", "fail")}}
        judge["summary"][part] = s
        say(f"{part:<19} n={len(rows)}  v1 agreement {s['v1']['agreement']:.3f} kappa {s['v1']['kappa']:+.3f}   "
            f"v2 agreement {s['v2']['agreement']:.3f} kappa {s['v2']['kappa']:+.3f}")
    say("  v1 confusion (all): " + json.dumps(judge["summary"]["all"]["v1"]["confusion"]))
    say("  v2 confusion (all): " + json.dumps(judge["summary"]["all"]["v2"]["confusion"]))
    dis = [x["id"] for x in judge["items"] if x["v2"] != x["human"]]
    say("  v2 disagreements: " + ", ".join(dis))
    # position swap test on pairs (A01,A02), (A03,A04), ...
    rec = {it["id"]: it for it in items}
    pairs = []
    for i in range(0, len(items), 2):
        a, b = items[i], items[i + 1]
        v1_ab = judge_v1_pairwise(a["answer"], b["answer"])
        v1_ba = judge_v1_pairwise(b["answer"], a["answer"])
        pick_ab = a["id"] if v1_ab == "first" else b["id"]
        pick_ba = b["id"] if v1_ba == "first" else a["id"]
        v2_ab = judge_v2_pairwise(a["answer"], b["answer"], a["record"], b["record"])
        v2_ba = judge_v2_pairwise(b["answer"], a["answer"], b["record"], a["record"])
        w2_ab = {"first": a["id"], "second": b["id"], "tie": "tie"}[v2_ab]
        w2_ba = {"first": b["id"], "second": a["id"], "tie": "tie"}[v2_ba]
        human = "tie" if a["human_label"] == b["human_label"] else (a["id"] if a["human_label"] == "pass" else b["id"])
        pairs.append({"pair": f"{a['id']}/{b['id']}", "human": human, "v1_ab": pick_ab, "v1_ba": pick_ba,
                      "v2_ab": w2_ab, "v2_ba": w2_ba})
    v1_flip = sum(p["v1_ab"] != p["v1_ba"] for p in pairs)
    v1_first = sum(1 for p in pairs if p["v1_ab"] == p["pair"].split("/")[0]) + \
        sum(1 for p in pairs if p["v1_ba"] == p["pair"].split("/")[1])
    v2_flip = sum(p["v2_ab"] != p["v2_ba"] for p in pairs)
    decisive = [p for p in pairs if p["human"] != "tie"]
    v1_right = sum(1 for p in decisive if p["v1_ab"] == p["human"] and p["v1_ba"] == p["human"])
    v2_right = sum(1 for p in decisive if p["v2_ab"] == p["human"] and p["v2_ba"] == p["human"])
    judge["position_swap"] = {"pairs": pairs, "v1_verdicts_that_flip": v1_flip,
                              "v1_first_position_wins": v1_first, "v1_verdicts_total": 2 * len(pairs),
                              "v2_verdicts_that_flip": v2_flip, "decisive_human_pairs": len(decisive),
                              "v1_consistent_and_right": v1_right, "v2_consistent_and_right": v2_right}
    say(f"position swap over {len(pairs)} pairs: v1 flips {v1_flip}, first position wins {v1_first} of "
        f"{2 * len(pairs)} verdicts; v2 flips {v2_flip}")
    say(f"  pairs where the hand labels prefer one answer: {len(decisive)}; consistent and right: "
        f"v1 {v1_right}, v2 {v2_right}")
    write_run(out, "judge.json", judge)

    # ------------------------------------------------------------------ E6
    say("\n== E6 first consequential error over failed held-out runs")
    fe = {}
    example_lost = None
    for name, rows in (("baseline", base), ("candidate", cand)):
        cons, anyk, pairs_diff = Counter(), Counter(), 0
        for r in rows:
            if r["end_state_pass"]:
                continue
            a, c = first_errors(r, by_id[r["task"]])
            cons[c] += 1
            anyk[a["kind"]] += 1
            if a["kind"] != c:
                pairs_diff += 1
            if c == "constraint_lost" and example_lost is None and name == "candidate":
                example_lost = r
        fe[name] = {"failed_runs": sum(cons.values()),
                    "first_consequential": {CATEGORIES[k][0] + " [" + CATEGORIES[k][1] + "]": v
                                            for k, v in cons.most_common()},
                    "first_error_of_any_kind": dict(anyk.most_common()),
                    "runs_where_first_error_is_not_the_consequential_one": pairs_diff}
        say(f"{name}: {fe[name]['failed_runs']} failed runs")
        for k, v in fe[name]["first_consequential"].items():
            say(f"    {k:<72} {v}")
        say(f"    runs whose first error of any kind was a different, harmless one: {pairs_diff}")
    write_run(out, "first_errors.json", fe)
    if example_lost:
        tr = example_lost["trace"]
        write_run(out, "example_constraint_lost.json",
                  {"task": example_lost["task"], "config": example_lost["config"], "trial": example_lost["trial"],
                   "requests": tr["requests"], "trajectory": [
                       {k: (v if k != "arguments" else {a: (b if len(str(b)) < 90 else str(b)[:87] + "...")
                                                        for a, b in v.items()})
                        for k, v in s.items()} for s in tr["trajectory"]],
                   "failed_checks": example_lost["failed_checks"]})
        say(f"example: candidate {example_lost['task']} trial {example_lost['trial']}: " + "; ".join(
            f"call {q['call']} msgs {q['messages']} dropped {q['dropped']} constraint "
            f"{'in' if q['constraints_in_request'] else ('OUT' if q['constraints_recorded'] else '-')}"
            for q in tr["requests"]))

    # ------------------------------------------------------------------ E7
    say("\n== E7 reliability through required steps")
    say(f"0.95^5 = {0.95 ** 5:.4f}")
    comp = {}
    for name, rows in (("baseline", base), ("candidate", cand)):
        rr = [r for r in rows if r["task"].startswith("R")]
        steps = {
            "mechanism correct": [("boundary_cases" not in r["failed_checks"]) for r in rr],
            "constraint kept": [("signature_unchanged" not in r["failed_checks"]) for r in rr],
            "verified after last edit": [not any(v["rule"] == "verify_after_last_edit" for v in r["violations"])
                                         for r in rr],
            "no prohibited proposal": [not any(v["rule"] == "no_deploy" for v in r["violations"]) for r in rr],
            "answer supported": [judge_v2(r["answer"], r["facts"])[0] == "pass" for r in rr],
        }
        rates = {k: r3(sum(v) / len(v)) for k, v in steps.items()}
        joint = sum(all(vals) for vals in zip(*steps.values())) / len(rr)
        prod = math.prod(sum(v) / len(v) for v in steps.values())
        comp[name] = {"runs": len(rr), "step_rates": rates, "product_of_rates": r3(prod), "joint_rate": r3(joint)}
        say(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in rates.items()))
        say(f"    product of step rates {prod:.3f}; all five in the same run {joint:.3f}")
    write_run(out, "compounding.json", comp)

    # ------------------------------------------------------------------ E8
    say("\n== E8 flawed evaluation design vs revised design")
    prng = random.Random(seed_for("variants") % 10 ** 9)
    variants = []
    tunable = ["p_convert_familiar", "p_convert_other", "p_rename_absent", "p_verify", "p_local_day",
               "p_reconcile", "p_overclaim", "p_verbose"]
    for i in range(8):
        v = dict(CANDIDATE, name=f"candidate-v{i + 1}")
        for k in tunable:
            v[k] = min(1.0, max(0.0, v[k] + prng.uniform(-0.12, 0.12)))
        variants.append(v)
    perfect = {"p_convert_familiar": 1.0, "p_convert_other": 1.0, "p_rename_absent": 0.0, "p_rename_present": 0.0,
               "p_verify": 1.0, "p_deploy": 0.0, "p_overclaim": 0.0, "p_verbose": 0.0, "p_vague": 0.0,
               "p_paraphrase": 0.0, "p_omit_file": 0.0, "p_local_day": 1.0, "p_reconcile": 1.0,
               "p_cleanup": 0.0, "p_tenant_arg": 0.0, "p_format_slip": 0.0, "p_add_test": 1.0,
               "p_pretest": 0.0}
    memo = {t["id"]: perfect for t in dev}
    memo_variant = dict(CANDIDATE, name="candidate-memo")

    def flawed_score(rows):
        ok = 0
        for r in rows:
            quality = judge_v1(r["answer"]) == "pass" if r["task"].startswith("R") else True
            ok += 1 if (r["end_state_pass"] and quality) else 0
        return ok / len(rows)

    flawed = []
    for v in [BASELINE] + variants + [memo_variant]:
        rows = run_set(dev, v, 1, memo=memo if v is memo_variant else None)
        flawed.append({"variant": v["name"], "dev_score": r3(flawed_score(rows))})
    best = max(flawed[1:], key=lambda x: x["dev_score"])
    best_honest = max(flawed[1:-1], key=lambda x: x["dev_score"])
    say("flawed design: 20 dev tasks, 1 trial, success = end state and judge v1, variants selected on the same tasks")
    for x in flawed:
        say(f"    {x['variant']:<16} {x['dev_score']:.3f}")
    say(f"  selected: {best['variant']} ({best['dev_score']:.3f}) vs baseline {flawed[0]['dev_score']:.3f}")
    held_scores = []
    for v in variants + [memo_variant]:
        rows = run_set(held, v, 8, memo=memo if v is memo_variant else None)
        held_scores.append({"variant": v["name"], "heldout_same_metric": r3(flawed_score(rows)),
                            "heldout_pass@1": r3(sum(r["end_state_pass"] for r in rows) / len(rows))})
    say("same variants on 40 held-out tasks x 8 trials: flawed metric, then end state alone")
    for x, y in zip(flawed[1:], held_scores):
        say(f"    {y['variant']:<16} dev {x['dev_score']:.3f} -> held-out {y['heldout_same_metric']:.3f}"
            f" (end state {y['heldout_pass@1']:.3f})")
    write_run(out, "flawed_vs_revised.json", {"flawed_dev": flawed, "selected": best, "selected_honest": best_honest,
                                              "heldout": held_scores,
                                              "baseline_heldout_pass@1": paired["pass@1"]["baseline"]})

    # ------------------------------------------------------------------ E9
    say("\n== E9 controlled changes, then the release gate")
    base_p = run_set(held, BASELINE, N_TRIALS, builder="pinned", keep_trace=True)
    cand_p = run_set(held, CANDIDATE, N_TRIALS, builder="pinned", keep_trace=True)
    v21 = dict(CANDIDATE, name="candidate-2.1", p_local_day=0.9)
    cand21 = run_set(held, v21, N_TRIALS, builder="pinned", keep_trace=True)

    def sig_fail(rows):
        return sum("signature_unchanged" in r["failed_checks"] for r in rows)

    change = {
        "context builder pinned": {
            "baseline signature failures flawed->pinned": [sig_fail(base), sig_fail(base_p)],
            "candidate signature failures flawed->pinned": [sig_fail(cand), sig_fail(cand_p)],
            "candidate pass@1 flawed->pinned": [r3(sum(per_task_rate(cand, held)) / 40),
                                                r3(sum(per_task_rate(cand_p, held)) / 40)],
            "baseline pass@1 flawed->pinned": [r3(sum(per_task_rate(base, held)) / 40),
                                               r3(sum(per_task_rate(base_p, held)) / 40)]},
    }
    say(f"pinned constraints in the context builder: signature failures baseline "
        f"{sig_fail(base)} -> {sig_fail(base_p)}, candidate {sig_fail(cand)} -> {sig_fail(cand_p)} "
        f"(of {len(base)} runs each)")

    def gate(name, cand_rows, base_rows, dataset_tasks, tuned_on):
        ra_, rb_ = per_task_rate(base_rows, dataset_tasks), per_task_rate(cand_rows, dataset_tasks)
        lo, hi = bootstrap_diff(ra_, rb_)
        checks = []
        checks.append(("evaluation tasks >= 40", len(dataset_tasks) >= 40, len(dataset_tasks)))
        trials = min(Counter(r["task"] for r in cand_rows).values())
        checks.append(("trials per task >= 8", trials >= 8, trials))
        overlap = sorted({t["id"] for t in dataset_tasks} & set(tuned_on))
        checks.append(("no task used for diagnosis or tuning", not overlap, len(overlap)))
        checks.append(("pass@1 difference lower 95% bound >= -0.03", lo >= -0.03, [r3(lo), r3(hi)]))
        worst = None
        for s in sorted({t["slice"] for t in dataset_tasks}):
            ts = [t for t in dataset_tasks if t["slice"] == s]
            ids_ = {t["id"] for t in ts}
            hb_ = sum(per_task_hat([r for r in cand_rows if r["task"] in ids_], ts, 4)) / len(ts)
            ha_ = sum(per_task_hat([r for r in base_rows if r["task"] in ids_], ts, 4)) / len(ts)
            if worst is None or hb_ - ha_ < worst[1]:
                worst = (s, hb_ - ha_)
        checks.append(("every slice pass^4 >= baseline - 0.10", worst[1] >= -0.10, [worst[0], r3(worst[1])]))
        eff = lambda rows: sum(1 for r in rows for v in r["violations"]
                               if v["executed"] and v["rule"] in ("single_create", "no_unrequested_change", "no_deploy"))
        checks.append(("executed prohibited effects <= baseline", eff(cand_rows) <= eff(base_rows),
                       [eff(cand_rows), eff(base_rows)]))
        kappa = judge["summary"]["held-out"]["v2"]["kappa"]
        checks.append(("answer judge kappa >= 0.6 on held-out labels", kappa >= 0.6, kappa))
        ans = [judge_v2(r["answer"], r["facts"])[0] == "pass" for r in cand_rows if r["task"].startswith("R")]
        ans_b = [judge_v2(r["answer"], r["facts"])[0] == "pass" for r in base_rows if r["task"].startswith("R")]
        checks.append(("supported answers >= baseline", sum(ans) / len(ans) >= sum(ans_b) / len(ans_b),
                       [r3(sum(ans) / len(ans)), r3(sum(ans_b) / len(ans_b))]))
        decision = "allow" if all(c[1] for c in checks) else "block"
        return {"candidate": name, "decision": decision,
                "checks": [{"check": c[0], "passed": c[1], "observed": c[2]} for c in checks]}

    v22 = dict(v21, name="candidate-2.2", p_verbose=0.1)
    cand22 = run_set(held, v22, N_TRIALS, builder="pinned", keep_trace=True)
    diagnosed_on = [t["id"] for t in held]
    release = make_release_tasks()
    base_rel = run_set(release, BASELINE, N_TRIALS, builder="pinned")
    cand22_rel = run_set(release, v22, N_TRIALS, builder="pinned")
    gates = [gate("candidate, flawed builder, held-out set", cand, base, held, []),
             gate("candidate-2.1, pinned builder, held-out set", cand21, base_p, held, []),
             gate("candidate-2.2, pinned builder, held-out set", cand22, base_p, held, diagnosed_on),
             gate("candidate-2.2, pinned builder, release set", cand22_rel, base_rel, release, diagnosed_on)]
    say(f"candidate-2.1 changes one setting (relative days resolved in the tenant timezone); "
        f"export-yesterday pass@1 {sum(per_task_rate([r for r in cand_p if r['slice'] == 'export-yesterday'], [t for t in held if t['slice'] == 'export-yesterday'])) / sum(t['slice'] == 'export-yesterday' for t in held):.3f}"
        f" -> {sum(per_task_rate([r for r in cand21 if r['slice'] == 'export-yesterday'], [t for t in held if t['slice'] == 'export-yesterday'])) / sum(t['slice'] == 'export-yesterday' for t in held):.3f}")
    # nearby regressions: slices other than the one changed
    other = {}
    for s in slices:
        if s == "export-yesterday":
            continue
        ts = [t for t in held if t["slice"] == s]
        other[s] = [r3(sum(per_task_rate([r for r in cand_p if r["slice"] == s], ts)) / len(ts)),
                    r3(sum(per_task_rate([r for r in cand21 if r["slice"] == s], ts)) / len(ts))]
    change["candidate-2.1 other slices pass@1 (candidate -> 2.1)"] = other
    say("  other slices pass@1 candidate -> 2.1: " + ", ".join(f"{s} {a:.3f}->{b:.3f}" for s, (a, b) in other.items()))
    say("candidate-2.2 changes one more setting (no closing generalization in answers)")
    for g in gates:
        say(f"gate [{g['decision'].upper()}] {g['candidate']}")
        for c in g["checks"]:
            say(f"    {'ok  ' if c['passed'] else 'FAIL'} {c['check']:<48} {json.dumps(c['observed'])}")
    # the flawed design's own gate: one number
    say(f"flawed design's verdict: selected {best['variant']} dev score {best['dev_score']:.3f} > baseline "
        f"{flawed[0]['dev_score']:.3f} -> ship")
    write_run(out, "controlled_change.json", change)
    write_run(out, "gates.json", gates)
    cv21 = {"candidate_pinned": curves(cand_p, held), "candidate_2_1_pinned": curves(cand21, held),
            "candidate_2_2_pinned": curves(cand22, held), "baseline_pinned": curves(base_p, held),
            "release_baseline": curves(base_rel, release), "release_candidate_2_2": curves(cand22_rel, release)}
    write_run(out, "passk_after_changes.json", cv21)
    say("pass@1 / pass^4 / pass@4 after changes: " + "; ".join(
        f"{k} {v['pass@k'][0]:.3f} / {v['pass^k'][3]:.3f} / {v['pass@k'][3]:.3f}" for k, v in cv21.items()))

    # ------------------------------------------------------------------ E10
    say("\n== E10 evidence produced by Part II companions, with Wilson 95% intervals")
    ce = component_evidence()
    for row in ce:
        say(f"  ch{row['chapter']:<2} {row['component']:<22} {row['evidence']:<48} {row['k']}/{row['n']} "
            f"[{row['wilson95'][0]:.3f}, {row['wilson95'][1]:.3f}]")
    write_run(out, "component_evidence.json", ce)

    say("\n== E11 intervals around published counts (goose evals/harbor README, Terminal-Bench 2.0, 89 tasks, 1 trial)")
    pub = []
    for name, k in (("sonnet46-full", 45), ("goose-1.30-sonnet46-full", 45), ("sonnet46-dev-only", 43),
                    ("goose-sonnet46-full-code-mode", 51)):
        lo, hi = wilson(k, 89)
        pub.append({"job": name, "pass": k, "n": 89, "rate": r3(k / 89), "wilson95": [r3(lo), r3(hi)]})
        say(f"  {name:<30} {k}/89 = {k / 89:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
    p1, p2 = 45 / 89, 51 / 89
    se = math.sqrt(p1 * (1 - p1) / 89 + p2 * (1 - p2) / 89)
    say(f"  difference code-mode minus stock {p2 - p1:+.3f}, unpaired normal 95% half-width {1.96 * se:.3f}")
    write_run(out, "published_counts.json", {"source": "goose evals/harbor/README.md at 04ed836c8cde",
                                             "rows": pub, "diff": r3(p2 - p1), "unpaired_halfwidth95": r3(1.96 * se)})

    write_run(out, "stdout.txt", "\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
