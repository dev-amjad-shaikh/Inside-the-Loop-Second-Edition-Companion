"""Chapter 25: open problems and the next runtime.

Three parts, all deterministic and standard library only:

  A. An evidence ledger. For each open problem, the line in a Part II run
     output that demonstrated the limitation, copied verbatim with its file
     hash. Run the Part II companions first; missing files are reported.
  B. Compounding arithmetic: run success as a product of per-step success,
     under an independence assumption the chapter questions.
  C. Executor-level learning, simulated: a timeout policy distilled from
     recorded runs, evaluated against the static floor on identical world
     tapes, pinned per run, checked for drift after a latency shift, and
     replayed exactly from its journal. Time is simulated.

    python3 ch25_open.py --output runs/ch25
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.ch25_open import (  # noqa: E402
    build_ledger, compounding_table, pass_hat_k, static_v0, policy_version, validate, resolve,
    draw_tape, run_tape, summarize, distill_timeouts, drift_check, journal_hash, replay_divergence)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch25")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, args.output) if not os.path.isabs(args.output) else args.output
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "executor"))
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # A. ledger -------------------------------------------------------------
    ledger = build_ledger(os.path.join(here, "runs"))
    dump(os.path.join(out, "ledger.json"), ledger)
    say("== A. evidence ledger (Part II run outputs)")
    for r in ledger:
        if r["found"]:
            say(f"  [{r['problem']}] ch{r['chapter']} {r['file']}:{r['line']} ({r['file_sha']})")
            say(f"      {r['text'][:160]}")
        else:
            say(f"  [{r['problem']}] ch{r['chapter']} {r['file']}: {r['reason']}")
    say(f"  found {sum(r['found'] for r in ledger)} of {len(ledger)}")

    # B. compounding --------------------------------------------------------
    comp = {"independent_steps": compounding_table(), "all_k_trials_succeed": pass_hat_k(0.9)}
    dump(os.path.join(out, "compounding.json"), comp)
    say()
    say("== B. compounding under independence (arithmetic, not measurement)")
    for r in comp["independent_steps"]:
        say(f"  per-step {r['per_step']:<6} steps {r['steps']:>3}  run success {r['run_success']:.4f}")
    for r in comp["all_k_trials_succeed"]:
        say(f"  p=0.9, all {r['k']} trials succeed: {r['all_succeed']:.4f}")

    # C. executor learning --------------------------------------------------
    ex = os.path.join(out, "executor")
    say()
    say("== C. executor-level learning of timeouts (simulated time)")
    floor = static_v0()
    say(f"  floor {policy_version(floor)}: timeout = lease {resolve(floor, 'export_status')[0]:.0f} s, "
        f"attempts {resolve(floor, 'export_status')[1]}")

    # C1. record under the floor and distill
    train = draw_tape(2501, 800)
    train_out, train_journal = run_tape(train, floor, "R-train")
    cand, notes = distill_timeouts(train_journal)
    cand["name"] = "candidate"
    v1 = policy_version(cand)
    dump(os.path.join(ex, "train_journal.json"), train_journal)
    say(f"  recorded {len(train)} ops under the floor, {len(train_journal)} decisions journaled")
    for callee, note in notes.items():
        say(f"  distill {callee}: {note}")
    say(f"  candidate {v1}: {cand['timeout_s']}")

    bad = {"name": "bad", "timeout_s": {"export_status": 0.5}, "max_attempts": 5}
    say(f"  out-of-envelope candidate: {validate(bad)}; decision point resolves "
        f"export_status to {resolve(bad, 'export_status')} (floor)")
    wide = {"name": "wide", "timeout_s": {}, "max_attempts": 5}
    say(f"  candidate asking for 5 attempts resolves to {resolve(wide, 'model_endpoint')[1]} "
        f"(task budget {3}; learned budgets only narrow)")

    # C2. twin-style evaluation on a held-out tape
    held = draw_tape(2502, 800)
    f_out, _ = run_tape(held, floor, "E-floor")
    c_out, _ = run_tape(held, cand, "E-cand")
    evaluation = {}
    for callee in ("export_status", "model_endpoint"):
        evaluation[callee] = {"floor": summarize(f_out, callee), "candidate": summarize(c_out, callee)}
    dump(os.path.join(ex, "evaluation_heldout.json"), evaluation)
    say("  held-out tape, identical world for both arms:")
    for callee, arms in evaluation.items():
        for arm in ("floor", "candidate"):
            s = arms[arm]
            say(f"    {callee:<15} {arm:<9} completed {s['completed']}/{s['ops']}  "
                f"mean {s['mean_wall_s']:>6} s  p95 {s['p95_wall_s']:>6} s  total {s['total_wall_s']} s")

    # Promotion gate: completion may not fall by more than MARGIN on any callee, and
    # total wall time must fall. The margin is a choice, and the chapter discusses it.
    MARGIN = 0.005
    gate = {}
    for callee, arms in evaluation.items():
        drop = round(arms["floor"]["completion"] - arms["candidate"]["completion"], 4)
        gain = round(arms["floor"]["total_wall_s"] - arms["candidate"]["total_wall_s"], 1)
        gate[callee] = {"completion_drop": drop, "wall_saved_s": gain,
                        "pass": drop <= MARGIN and gain > 0,
                        "pass_if_margin_zero": drop <= 0 and gain > 0}
    admitted = all(g["pass"] for g in gate.values())
    dump(os.path.join(ex, "gate.json"), {"margin": MARGIN, "per_callee": gate, "admitted": admitted})
    say(f"  gate (completion drop <= {MARGIN}, wall time must fall): {gate}; admitted={admitted}")

    # C3. promote, pin, run in production
    activation = [{"event": "promoted", "version": v1, "approved_by": "user:owner-ops",
                   "evidence": "evaluation_heldout.json, gate.json"}]
    prod = draw_tape(2503, 400)
    p_out, p_journal = run_tape(prod, cand, "R-25a")
    dump(os.path.join(ex, "R-25a_journal.json"), p_journal)
    say(f"  run R-25a pinned to {v1}: {len(p_journal)} decisions, journal {journal_hash(p_journal)}, "
        f"completed {sum(o['completed'] for o in p_out)}/{len(p_out)}")

    # C4. the world shifts: the model endpoint slows threefold
    shifted = draw_tape(2504, 800, shift={"model_endpoint": 3.0})
    s_cand, _ = run_tape(shifted, cand, "R-shift-cand")
    s_floor, s_floor_journal = run_tape(shifted, floor, "R-shift-floor")
    shift_eval = {c: {"floor": summarize(s_floor, c), "candidate": summarize(s_cand, c)}
                  for c in ("export_status", "model_endpoint")}
    drift = drift_check(evaluation["model_endpoint"]["candidate"], shift_eval["model_endpoint"]["candidate"])
    dump(os.path.join(ex, "evaluation_shift.json"), {"eval": shift_eval, "drift": drift})
    say("  after a 3x slowdown of model_endpoint (same candidate, new tape):")
    for callee, arms in shift_eval.items():
        for arm in ("floor", "candidate"):
            s = arms[arm]
            say(f"    {callee:<15} {arm:<9} completed {s['completed']}/{s['ops']}  "
                f"mean {s['mean_wall_s']:>6} s  p95 {s['p95_wall_s']:>6} s")
    say(f"  drift check on model_endpoint: {drift}")
    active = floor if drift["drift"] else cand
    activation.append({"event": "reverted" if drift["drift"] else "kept", "version": policy_version(active),
                       "reason": "completion drop beyond threshold" if drift["drift"] else "no drift"})
    cand2, notes2 = distill_timeouts(s_floor_journal)
    cand2["name"] = "candidate-2"
    v2 = policy_version(cand2)
    activation.append({"event": "candidate distilled from shifted floor traffic", "version": v2,
                       "timeout_s": cand2["timeout_s"], "status": "awaiting evaluation"})
    dump(os.path.join(ex, "activations.json"), activation)
    for callee, note in notes2.items():
        say(f"  distill from shifted floor traffic, {callee}: {note}")
    say(f"  active policy now {policy_version(active)}; new candidate {v2}: {cand2['timeout_s']}")

    # C5. replay R-25a two ways
    served, served_journal = run_tape(prod, cand, "R-25a", journal_decisions=p_journal)
    rec_floor, rec_floor_journal = run_tape(prod, active, "R-25a")
    rec_c2, rec_c2_journal = run_tape(prod, cand2, "R-25a")
    replay = {
        "recorded": {"journal": journal_hash(p_journal), "completed": sum(o["completed"] for o in p_out)},
        "served_from_journal": {"journal": journal_hash(served_journal),
                                "completed": sum(o["completed"] for o in served),
                                "identical": journal_hash(served_journal) == journal_hash(p_journal)},
        "recomputed_under_active": {"version": policy_version(active),
                                    "journal": journal_hash(rec_floor_journal),
                                    "completed": sum(o["completed"] for o in rec_floor),
                                    **replay_divergence(p_journal, rec_floor_journal)},
        "recomputed_under_candidate_2": {"version": v2, "journal": journal_hash(rec_c2_journal),
                                         "completed": sum(o["completed"] for o in rec_c2),
                                         **replay_divergence(p_journal, rec_c2_journal)},
    }
    dump(os.path.join(ex, "replay.json"), replay)
    say("  replay of R-25a:")
    for k, v in replay.items():
        say(f"    {k:<30} {v}")

    # The Rust test log was produced once with cargo (see the chapter); it is copied, not re-run.
    rust_log = os.path.join(here, "fixtures", "ch25", "rusty_tests.txt")
    if os.path.exists(rust_log):
        shutil.copy(rust_log, os.path.join(out, "rusty_tests.txt"))
    with open(os.path.join(out, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
