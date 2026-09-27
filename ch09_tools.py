"""Chapter 9: tools and the executor.

Two toolsets over the same sqlite export service. No model is called:

- Tool choice comes from a lexical selector, a labeled stand-in for model
  selection (keyword overlap between task phrasing and tool name plus
  description). It shows how contract text separates or confuses operations;
  it cannot show how a trained model uses context, synonyms or intent.
- Arguments are authored, the way Chapter 3's responses are authored.
- Every call really executes against the service database, and the database
  state is recorded after each call.

    python3 ch09_tools.py --output runs/ch09
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import write_run  # noqa: E402
from itl import ch09_tools as T  # noqa: E402

# Development phrasings (descriptions were revised once after seeing the
# stand-in's ties on these) and held-out phrasings (written before that
# revision and never used to edit descriptions).
DEV_TASKS = [
    ("Export the daily orders report for 2026-09-11 as CSV to the finance SFTP server.",
     "exports_create", "create_export"),
    ("Send the daily orders file for September 11 to finance.", "exports_create", "create_export"),
    ("Deliver the weekly revenue report to the analytics bucket as parquet.",
     "exports_create", "create_export"),
    ("Is export exp_0413 finished yet?", "exports_get_status", "get_export"),
    ("What is the status of the export I started this morning?", "exports_list", "get_export"),
    ("Why did the export for September 10 fail? Show the end of its job log.",
     "exports_read_log", "get_logs"),
    ("Show the job log for export exp_0412.", "exports_read_log", "get_logs"),
    ("Which report lists refunds by region?", "reports_search", "get_data"),
    ("Find the daily orders report in the catalog and its current version.",
     "reports_search", "get_data"),
    ("List all failed exports from this week.", "exports_list", "get_export"),
    ("Cancel the parquet export that is still running.", "exports_cancel", "process_data"),
    ("Re-run yesterday's daily orders export; it failed on upload.",
     "exports_create", "create_export"),
]
HELD_OUT = [
    ("Push the refunds by region report for 2026-09-08 to the analytics bucket in parquet.",
     "exports_create", "create_export"),
    ("Did exp_0412 get delivered?", "exports_get_status", "get_export"),
    ("Give me the last lines of the log for exp_0412.", "exports_read_log", "get_logs"),
    ("Stop export exp_0415 before it uploads.", "exports_cancel", "process_data"),
    ("Which exports are still queued?", "exports_list", "get_export"),
    ("Is there a report about salaries?", "reports_search", "get_data"),
]

SESSION = T.Session("acme", "maria@acme", ["exports.create", "exports.read", "reports.read"])


def call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def classify(sel, expected):
    if sel["selected"] != expected:
        return "wrong"
    return "correct" if sel["margin"] > 0 else "correct_by_tie_break"


def selection_experiment(out):
    traces, summary = [], {}
    for set_name, defs, col in (("flawed", T.flawed_definitions(), 2),
                                ("revised", T.revised_definitions(), 1)):
        counts = {}
        for split, tasks in (("development", DEV_TASKS), ("held_out", HELD_OUT)):
            c = {"correct": 0, "correct_by_tie_break": 0, "wrong": 0}
            for task in tasks:
                sel = T.lexical_select(task[0], defs)
                verdict = classify(sel, task[col])
                c[verdict] += 1
                traces.append(dict(sel, toolset=set_name, split=split, expected=task[col],
                                   verdict=verdict))
            counts[split] = c
        summary[set_name] = counts
    write_run(out, "selection_traces.json", traces)
    write_run(out, "selection_summary.json", summary)
    return traces, summary


def state(con):
    return [r for r in T.export_rows(con) if r["call_id"] != "seed"]


def flawed_run(out, trace_t1):
    path = os.path.join(out, "export_service_flawed.db")
    if os.path.exists(path):
        os.remove(path)
    con = T.open_service(path)
    tools = T.FlawedTools(con, SESSION)
    ex = T.NaiveExecutor(tools, T.flawed_definitions())
    chosen = trace_t1["selected"]
    authored = {
        "process_data": {"data": "daily orders", "action": "export 2026-09-11 as csv to finance sftp"},
        "export_report": {"report": "daily orders", "options": "csv, finance sftp, 2026-09-11"},
    }
    calls = [
        call("f1", chosen, **authored[chosen]),
        None,  # filled from f1's result
        call("f3", "get_data", query="daily orders"),
        call("f4", "create_export", report_id="daily_orders", tenant="acme", day="Sept 11",
             format="csv", destination="finance_sftp"),
        call("f5", "create_export", report_id="daily_orders", tenant="globex", day="2026-09-11",
             format="csv", destination="finance_sftp"),
    ]
    steps = []
    for i, c in enumerate(calls):
        if c is None:
            prior = json.loads(steps[0]["observation"]["content"])
            returned = prior.get("result") or prior.get("id")
            c = call("f2", "get_export", id=returned)
        obs = ex.execute(c)
        steps.append({"call": c, "observation": obs, "exports_after": state(con)})
    intent = {"tenant": "acme", "report": "daily_orders", "day": "2026-09-11", "format": "csv",
              "destination": "finance_sftp"}
    intent_check = [{"export_id": r["export_id"], "created_by_call": r["call_id"],
                     "differs_from_task": {k: {"task": v, "found": r[k]}
                                           for k, v in intent.items() if r[k] != v}}
                    for r in state(con)]
    write_run(out, "flawed_run.json", {"selected_by_stand_in": chosen,
                                       "tie": trace_t1["tied_at_top"], "steps": steps,
                                       "task_intent": intent, "intent_check": intent_check})
    con.close()
    return steps, intent_check


def revised_run(out):
    path = os.path.join(out, "export_service_revised.db")
    if os.path.exists(path):
        os.remove(path)
    con = T.open_service(path, daily_orders=(4, "78721bd28acc"))
    tools = T.RevisedTools(con)
    ex = T.CheckedExecutor(tools, T.revised_definitions(), SESSION)
    steps = []

    def run(c, note):
        obs = ex.execute(c)
        steps.append({"note": note, "call": c, "observation": obs, "exports_after": state(con)})

    run(call("r1", "reports_search", query="daily orders"), "catalog read before the fix ships")
    T.publish_report_version(con, "acme", "daily_orders", 5, "339d3a9f2656")
    steps.append({"note": "external event: reporting team publishes daily_orders v5 "
                          "(source 339d3a9f2656, the timezone fix)"})
    run(call("r2", "exports_create", report="daily_orders", day="Sept 11", format="CSV",
             destination="finance sftp", expected_report_version=4, tenant="acme"), "shape")
    run(call("r3", "exports_create", report="daily_orders", day="2026-09-11", format="parquet",
             destination="finance_sftp", expected_report_version=4), "cross-field")
    run(call("r4", "exports_create", report="daily_orders", day="2026-09-12", format="csv",
             destination="finance_sftp", expected_report_version=4), "range")
    run(call("r5", "exports_create", report="daily_orders", day="2026-09-11", format="csv",
             destination="finance_sftp", expected_report_version=4), "precondition")
    run(call("r6", "reports_search", query="daily orders"), "re-read after precondition failure")
    run(call("r7", "exports_create", report="daily_orders", day="2026-09-11", format="csv",
             destination="finance_sftp", expected_report_version=5), "correct call")
    run(call("r8", "exports_create", report="payroll_summary", day="2026-09-11", format="csv",
             destination="finance_sftp", expected_report_version=7), "authorization")
    run(call("r9", "exports_read_log", export_id="exp_0412", lines=5000), "range on a read")
    write_run(out, "revised_run.json", {"session": SESSION.as_dict(), "steps": steps,
                                        "executor_records": ex.records})
    con.close()
    return steps


def result_use(out):
    path = os.path.join(out, "export_service_logs.db")
    if os.path.exists(path):
        os.remove(path)
    con = T.open_service(path)
    total_lines = con.execute("SELECT COUNT(*) FROM export_log WHERE export_id='exp_0412'").fetchone()[0]
    flawed = T.NaiveExecutor(T.FlawedTools(con, SESSION), T.flawed_definitions())
    naive = flawed.execute(call("g1", "get_logs", id="exp_0412"))
    full = json.dumps(T.FlawedTools(con, SESSION).get_logs("exp_0412", call_id="g1"))
    shown = naive["content"]
    shown_lines = shown.count("\\n") + 1
    marked = T.bounded(full, 4000)
    revised = T.CheckedExecutor(T.RevisedTools(con), T.revised_definitions(), SESSION)
    tail = revised.execute(call("g2", "exports_read_log", export_id="exp_0412"))
    head = revised.execute(call("g3", "exports_read_log", export_id="exp_0412", from_end=False,
                                lines=40))
    status = revised.execute(call("g4", "exports_get_status", export_id="exp_0412"))
    cases = [
        {"case": "flawed tool, head-only truncation without marker", "chars_available": len(full),
         "chars_shown": len(shown), "lines_shown_of_total": f"{shown_lines} of {total_lines}",
         "reader_concludes": T.read_outcome(shown), "observation_end": shown[-160:]},
        {"case": "flawed tool, runtime keeps head and tail with a marker",
         "chars_shown": len(marked), "reader_concludes": T.read_outcome(marked),
         "marker": next(l for l in marked.split("\n") if "omitted from the middle" in l),
         "observation_end": marked[-230:]},
        {"case": "revised tool, default read from the end", "observation": tail,
         "reader_concludes": T.read_outcome(json.dumps(tail))},
        {"case": "revised tool, explicit read from the start", "omitted": head["result"]["omitted"],
         "returned_lines": head["result"]["returned_lines"],
         "reader_concludes": T.read_outcome(json.dumps(head))},
        {"case": "revised status tool", "observation": status,
         "reader_concludes": T.read_outcome(json.dumps(status))},
    ]
    write_run(out, "result_use.json", {"total_log_lines": total_lines, "cases": cases})
    con.close()
    return total_lines, cases


def response_sizes(out):
    path = os.path.join(out, "export_service_sizes.db")
    if os.path.exists(path):
        os.remove(path)
    con = T.open_service(path)
    f = T.FlawedTools(con, SESSION)
    r = T.RevisedTools(con)
    samples = {
        "flawed get_export(exp_0412)": f.get_export("exp_0412", call_id="s1"),
        "revised exports_get_status concise": r.exports_get_status(SESSION, "exp_0412"),
        "revised exports_get_status detailed": r.exports_get_status(SESSION, "exp_0412", "detailed"),
        "flawed get_export(all)": f.get_export("all", call_id="s2"),
        "revised exports_list limit 10": r.exports_list(SESSION),
    }
    rows = []
    for name, value in samples.items():
        text = json.dumps(value)
        rows.append({"response": name, "chars": len(text), "approx_tokens": round(len(text) / 4),
                     "text": text})
    defs = {"flawed": T.flawed_definitions(), "revised": T.model_view(T.revised_definitions())}
    contract_sizes = {k: len(json.dumps(v)) for k, v in defs.items()}
    write_run(out, "response_sizes.json", {"responses": rows, "definition_chars": contract_sizes,
                                           "note": "approx_tokens = chars / 4, an approximation"})
    con.close()
    return rows, contract_sizes


CALLERS = [
    ("nightly finance workflow", "daily_orders", "2026-09-11", "csv", "finance_sftp", 5),
    ("saved skill: month-end pack", "daily_orders", "2026-08-31", "csv", "finance_sftp", 5),
    ("replay of agent run r-2291", "daily_orders", "2026-09-10", "csv", "finance_sftp", 5),
    ("scheduled task: weekly revenue", "weekly_revenue", "2026-09-07", "parquet",
     "analytics_bucket", 3),
    ("analytics agent", "refunds_by_region", "2026-09-11", "parquet", "analytics_bucket", 2),
]


def versioning(out):
    v1 = T.revised_definitions(1)
    v2 = T.revised_definitions(2)
    v1_create = next(d for d in v1 if d["name"] == "exports_create")
    v2_create = next(d for d in v2 if d["name"] == "exports_create")
    lax = json.loads(json.dumps(v2_create))
    lax["parameters"]["additionalProperties"] = True
    lax["parameters"]["required"] = [n for n in lax["parameters"]["required"] if n != "report_date"]
    compat = {"strict_v2": T.schema_changes(v1_create["parameters"], v2_create["parameters"]),
              "lax_v2": T.schema_changes(v1_create["parameters"], lax["parameters"])}
    results = {}
    variants = ["v1", "v2 unversioned, strict schema", "v2 unversioned, lax schema",
                "v2 versioned, v1 pinned by caller"]
    for variant in variants:
        path = os.path.join(out, "export_service_versioning.db")
        if os.path.exists(path):
            os.remove(path)
        con = T.open_service(path)
        tools = T.RevisedTools(con)
        rows = []
        for i, (who, report, day, fmt, dest, ver) in enumerate(CALLERS, 1):
            args = {"report": report, "day": day, "format": fmt, "destination": dest,
                    "expected_report_version": ver}
            c = call(f"v{i}", "exports_create", **args)
            note = None
            if variant == "v1":
                defs = v1
            elif variant == "v2 unversioned, strict schema":
                defs = v2
            elif variant == "v2 unversioned, lax schema":
                defs = [lax if d["name"] == "exports_create" else d for d in v2]
                # The old caller still sends day; see lax_execute.
            else:
                defs = v2
                # The catalog binding records that this caller was written against
                # exports_create@1; the executor adapts the call and says so.
                adapted = {("report_date" if k == "day" else k): v for k, v in args.items()}
                c = call(f"v{i}", "exports_create", **adapted)
                note = ("exports_create@1 is deprecated: rename day to report_date. "
                        "Version 1 calls are adapted until 2027-03-31.")
            ex = T.CheckedExecutor(tools, defs, SESSION)
            if variant == "v2 unversioned, lax schema":
                obs = lax_execute(ex, tools, c)
            else:
                obs = ex.execute(c)
            if note and obs["status"] == "ok":
                obs["deprecation"] = note
            created = [r for r in T.export_rows(con) if r["call_id"] == c["id"]]
            rows.append({"caller": who, "requested_day": day, "status": obs["status"],
                         "message": obs.get("message"), "deprecation": obs.get("deprecation"),
                         "exported_day": created[0]["day"] if created else None,
                         "silently_wrong": bool(created) and created[0]["day"] != day})
        results[variant] = rows
        con.close()
    os.remove(os.path.join(out, "export_service_versioning.db"))
    write_run(out, "versioning.json", {"compatibility": compat, "callers": results})
    return compat, results


def lax_execute(ex, tools, c):
    """The lax v2 implementation: unknown arguments are ignored and a missing
    report_date means yesterday. Validation passes; the day is lost."""
    args = dict(c["arguments"])
    args.pop("day", None)
    args.setdefault("report_date", (T.SERVICE_TODAY - T.dt.timedelta(days=1)).isoformat())
    return ex.execute({"id": c["id"], "name": c["name"], "arguments": args})


def batch_plan(out):
    effects = {d["name"]: d["effect"] for d in T.revised_definitions()}
    batch = [call("b1", "exports_get_status", export_id="exp_0411"),
             call("b2", "exports_get_status", export_id="exp_0412"),
             call("b3", "exports_read_log", export_id="exp_0412"),
             call("b4", "exports_create", report="daily_orders", day="2026-09-10", format="csv",
                  destination="finance_sftp", expected_report_version=5),
             call("b5", "exports_list", status="failed")]
    groups, current = [], []
    for c in batch:
        if effects[c["name"]] == "read_only":
            current.append(c["id"])
        else:
            if current:
                groups.append({"mode": "parallel", "calls": current})
                current = []
            groups.append({"mode": "exclusive", "calls": [c["id"]]})
    if current:
        groups.append({"mode": "parallel", "calls": current})
    plan = {"batch": [(c["id"], c["name"], effects[c["name"]]) for c in batch], "groups": groups}
    write_run(out, "batch_plan.json", plan)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch09")
    out = ap.parse_args().output
    os.makedirs(out, exist_ok=True)

    write_run(out, "contracts_flawed.json", T.flawed_definitions())
    write_run(out, "contracts_revised_model_view.json", T.model_view(T.revised_definitions()))
    write_run(out, "contracts_revised_catalog.json", [
        {k: d[k] for k in ("name", "version", "owner", "effect")} for d in T.revised_definitions()])

    print("== Selection (lexical stand-in, not a model) ==")
    traces, summary = selection_experiment(out)
    for set_name, splits in summary.items():
        for split, c in splits.items():
            print(f"{set_name:8} {split:12} correct={c['correct']} "
                  f"correct_by_tie_break={c['correct_by_tie_break']} wrong={c['wrong']}")
    t1 = {s: next(t for t in traces if t["toolset"] == s and t["task"] == DEV_TASKS[0][0])
          for s in ("flawed", "revised")}
    for s, t in t1.items():
        top = ", ".join(f"{c['tool']}={c['score']}" for c in t["candidates"][:4])
        print(f"T1 {s}: selected {t['selected']} (tie: {t['tied_at_top'] or 'none'}); {top}")

    print("\n== Flawed toolset: calls and external state ==")
    fsteps, intent_check = flawed_run(out, t1["flawed"])
    for st in fsteps:
        obs = st["observation"]
        shown = obs.get("content", obs.get("message"))
        print(f"{st['call']['id']} {st['call']['name']} {json.dumps(st['call']['arguments'])}")
        print(f"   -> {obs['status']}: {shown[:150]}")
        print(f"   exports created so far: {[(r['export_id'], r['tenant'], r['report'], r['day'], r['format'], r['destination']) for r in st['exports_after']]}")

    for row in intent_check:
        print(f"intent check {row['export_id']} (call {row['created_by_call']}): "
              f"{json.dumps(row['differs_from_task'])}")

    print("\n== Revised toolset: checked executor and external state ==")
    for st in revised_run(out):
        if "call" not in st:
            print(f"-- {st['note']}")
            continue
        obs = st["observation"]
        detail = obs.get("message") or json.dumps(obs.get("result"))[:150]
        print(f"{st['call']['id']} {st['call']['name']} [{st['note']}] -> {obs['status']}"
              f" at {obs.get('stage', 'done')}: {detail}")
        if obs.get("verification"):
            print(f"   verification: {json.dumps(obs['verification'])}")
        print(f"   exports created so far: {[(r['export_id'], r['report'], r['report_version'], r['day'], r['format'], r['destination']) for r in st['exports_after']]}")

    print("\n== Using a result: the failure at the end of a long log ==")
    total, cases = result_use(out)
    print(f"exp_0412 log has {total} lines")
    for c in cases:
        extra = {k: c[k] for k in ("chars_available", "chars_shown", "lines_shown_of_total",
                                   "omitted", "returned_lines") if k in c}
        print(f"{c['case']}: reader concludes {c['reader_concludes']!r} {json.dumps(extra)}")

    print("\n== Response sizes ==")
    rows, dsize = response_sizes(out)
    for r in rows:
        print(f"{r['response']:40} {r['chars']:6} chars  ~{r['approx_tokens']} tokens")
    print(f"model-visible definitions: {dsize}")

    print("\n== Changing a tool: rename day -> report_date ==")
    compat, results = versioning(out)
    for k, v in compat.items():
        print(f"compatibility check {k}: breaking={v['breaking']}; "
              + "; ".join(f"{c['change']} ({'breaking' if c['breaking'] else 'compatible'})"
                          for c in v["changes"]))
    for variant, rows in results.items():
        ok = sum(r["status"] == "ok" for r in rows)
        wrong = sum(r["silently_wrong"] for r in rows)
        print(f"{variant:36} ok={ok}/5 silently_wrong_day={wrong} first_message="
              f"{rows[0]['message'] or rows[0]['deprecation']}")

    print("\n== Scheduling a batch by declared effect ==")
    print(json.dumps(batch_plan(out)["groups"]))


if __name__ == "__main__":
    main()
