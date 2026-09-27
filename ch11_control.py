"""Chapter 11: planning and control.

Runs three controllers over the same reporting task family:

  workflow  fixed stages chosen by the program; the model writes a test and a patch
  loop      the Chapter 3 loop: the model chooses every step until it stops or hits 12 calls
  hybrid    required stages inspect -> reproduce -> fix -> verify, model-chosen actions,
            evidence-based progress detection, separate budgets, repair allowance,
            evidence plan and distinct termination outcomes

The model is a deterministic stand-in (itl/ch11_control.StandInModel) that picks its next
action from its input only. File reads, edits and test runs are real.

    python3 ch11_control.py --output runs/ch11
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import write_run  # noqa: E402
from itl.ch11_control import audit, run_hybrid, run_loop, run_workflow, LIMITS  # noqa: E402

SCENARIOS = [
    # name, fixture, behavior, deploy task, cancellation time (simulated seconds)
    {"name": "normal", "fixture": "report", "behavior": "competent", "deploy": False},
    {"name": "helper", "fixture": "report_helper", "behavior": "competent", "deploy": False},
    {"name": "repeat", "fixture": "report", "behavior": "repeat", "deploy": False},
    {"name": "early-stop", "fixture": "report", "behavior": "hasty", "deploy": False},
    {"name": "malformed", "fixture": "report", "behavior": "malformed", "deploy": False},
    {"name": "wander", "fixture": "report_wide", "behavior": "wander", "deploy": False},
    {"name": "deploy", "fixture": "report", "behavior": "competent", "deploy": True},
    {"name": "cancel", "fixture": "report", "behavior": "competent", "deploy": False, "cancel_at": 10.0},
]

CONTROLLERS = [("workflow", run_workflow), ("loop", run_loop), ("hybrid", run_hybrid)]


def save(out, ctrl, scen, ws, rec, model, events, messages):
    d = os.path.join(out, ctrl, scen)
    rec["audit"] = audit(ws)
    write_run(d, "outcome.json", rec)
    write_run(d, "events.json", events)
    write_run(d, "messages.json", messages)
    write_run(d, "trace.json", model.requests)
    for name, text in ws.snapshot().items():
        write_run(os.path.join(d, "after"), name, text)
    ws.close()
    return rec


def cell(rec):
    return f"{rec['outcome']} ({rec['model_calls']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = args.output
    os.makedirs(out, exist_ok=True)

    summary = []
    for scen in SCENARIOS:
        for ctrl, fn in CONTROLLERS:
            ws, rec, model, events, messages = fn(scen, scen["fixture"])
            rec = save(out, ctrl, scen["name"], ws, rec, model, events, messages)
            summary.append({"scenario": scen["name"], "controller": ctrl, "outcome": rec["outcome"],
                            "reason": rec["reason"], "model_calls": rec["model_calls"],
                            "tokens": rec["tokens"], "sim_seconds": rec["sim_seconds"],
                            "executions": rec["executions"],
                            "acceptance_passed": rec["audit"]["acceptance_passed"],
                            "deployed": rec["audit"]["deployed"]})

    # ablation: the progress detector with raw byte revisions instead of semantic ones
    scen = next(s for s in SCENARIOS if s["name"] == "repeat")
    ws, rec, model, events, messages = run_hybrid(scen, scen["fixture"], semantic=False)
    rec = save(out, "hybrid-raw-revisions", "repeat", ws, rec, model, events, messages)
    summary.append({"scenario": "repeat", "controller": "hybrid-raw-revisions", "outcome": rec["outcome"],
                    "reason": rec["reason"], "model_calls": rec["model_calls"], "tokens": rec["tokens"],
                    "sim_seconds": rec["sim_seconds"], "executions": rec["executions"],
                    "acceptance_passed": rec["audit"]["acceptance_passed"],
                    "deployed": rec["audit"]["deployed"]})

    write_run(out, "summary.json", summary)
    write_run(out, "limits.json", LIMITS)

    by = {(r["scenario"], r["controller"]): r for r in summary}
    lines = ["outcome (model calls) per controller; * = acceptance test fails on final workspace",
             f"{'scenario':<11} {'workflow':<30} {'loop':<26} {'hybrid':<34}"]
    for s in SCENARIOS:
        row = [s["name"]]
        for ctrl, _ in CONTROLLERS:
            r = by[(s["name"], ctrl)]
            row.append(cell(r) + ("" if r["acceptance_passed"] else " *") + (" D" if r["deployed"] else ""))
        lines.append(f"{row[0]:<11} {row[1]:<30} {row[2]:<26} {row[3]:<34}")
    r = by[("repeat", "hybrid-raw-revisions")]
    lines.append(f"ablation   repeat, hybrid with raw byte revisions: {cell(r)} reason: {r['reason']}")
    lines.append("D = deployment executed")
    table = "\n".join(lines)
    write_run(out, "summary.txt", table + "\n")
    print(table)
    print()
    for r in summary:
        if r["controller"].startswith("hybrid"):
            print(f"hybrid {r['scenario']:<10} {r['controller']:<21} reason: {r['reason']}")


if __name__ == "__main__":
    main()
