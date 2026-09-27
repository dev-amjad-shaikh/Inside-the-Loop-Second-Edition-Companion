"""Chapter 13: recovery when the outcome is unknown.

Runs six experiments against real SQLite files (a runtime State store and one
database per external service) with injected interruptions and simulated time.

    python3 ch13_recovery.py --output runs/ch13

E1  lost receipt: export recovered through its operation key; changed payload rejected;
    a fresh key per retry creates a duplicate
E2  matrix: interruption point x recovery class -> reconciler decision and final
    external state, beside a naive retry-all policy
E3  notification: naive retry sends twice; reconciler escalates; a person resolves
E4  backoff with full jitter against a deadline; later sweep settles the job
E5  cancellation: the remote job continues and finishes
E6  key retention: a key reused after the service forgot it
"""
import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import Workspace, write_run  # noqa: E402
from itl.ch13_recovery import (  # noqa: E402
    CATALOG, POINTS, Backoff, Executor, Interrupted, Reconciler, ServiceError, SimClock,
    digest, fresh_stores, naive_retry_all, oracle_messages, oracle_rows, resolve_repair_item, restart)

RUN = "run-13"
FIX_OLD = "    return created_at.date() == requested_day"
FIX_NEW = ("    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
           "    return local_time.date() == requested_day")


def export_payload(revision, day="2026-09-11"):
    return {"report": "daily-orders", "report_date": day, "source_revision": revision,
            "destination": f"reports/daily-orders-{day}.csv", "format": "csv"}


NOTIFY = {"to": "reports-oncall@example.com", "subject": "Daily orders export for 2026-09-11 is ready",
          "body": "export-file-001 was created from revision 339d3a9f2656."}
JOB = {"dashboard": "daily-orders", "export_id": "export-file-001"}


def make_workspace():
    ws = Workspace("report")
    text = ws.read("report.py")
    ws.write("report.py", text.replace(FIX_OLD, FIX_NEW))
    return ws


def op_spec(tool, revision):
    if tool == "read_revision":
        return {"path": "report.py"}, None
    if tool == "create_export":
        return export_payload(revision), f"{RUN}:export:daily-orders:2026-09-11"
    if tool == "refresh_dashboard":
        return JOB, None
    if tool == "notify_oncall":
        return NOTIFY, None
    raise ValueError(tool)


def runtime_row(store, op_id):
    op = store.op(op_id)
    return {k: op[k] for k in ("op_id", "tool", "recovery_class", "op_key", "payload_digest",
                               "status", "attempts", "first_dispatched_at", "receipt")}


def external_state(outdir, tool, store):
    if tool == "read_revision":
        return "no external change"
    if tool == "create_export":
        rows = oracle_rows(os.path.join(outdir, "export.db"), "exports")
        return f"{len(rows)} export" + ("s" if len(rows) != 1 else "") + \
            (": " + ", ".join(r["export_id"] for r in rows) if rows else "")
    if tool == "refresh_dashboard":
        rows = oracle_rows(os.path.join(outdir, "jobs.db"), "jobs")
        return f"{len(rows)} job" + ("s" if len(rows) != 1 else "") + \
            (": " + ", ".join(f"{r['job_id']} {r['status']}" for r in rows) if rows else "")
    rows = oracle_messages(os.path.join(outdir, "notify.db"))
    open_items = [i["item_id"] for i in store.repair_items() if i["status"] == "open"]
    s = f"{len(rows)} message" + ("s" if len(rows) != 1 else "")
    if open_items:
        s += "; " + ", ".join(open_items) + " open"
    return s


# ---------------------------------------------------------------------------

def e1_lost_receipt(root, ws):
    out = os.path.join(root, "e1_lost_receipt")
    clock = SimClock()
    store, svc = fresh_stores(out, clock, ws)
    ex = Executor(store, svc)
    store.start_run(RUN, 3)
    store.prepare("op-0001", RUN, "read_revision", {"path": "report.py"})
    clock.advance(0.4)
    rev = ex.dispatch("op-0001")["version"]
    clock.advance(1.1)
    payload, key = op_spec("create_export", rev)
    store.prepare("op-0002", RUN, "create_export", payload, op_key=key)
    clock.advance(0.2)
    try:
        ex.dispatch("op-0002", interrupt="after_commit")
    except Interrupted:
        pass
    clock.advance(6.0)
    store, svc = restart(out, clock, ws)
    after = runtime_row(store, "op-0002")
    exports_after = oracle_rows(os.path.join(out, "export.db"), "exports")
    rec = Reconciler(store, Executor(store, svc), svc, clock)
    decisions = rec.reconcile_all()
    recovered = store.op("op-0002")["receipt"]
    clock.advance(0.5)
    changed = export_payload(rev, day="2026-09-12")
    try:
        svc["export"].create_export(key, changed)
        changed_result = "accepted"
    except ServiceError as exc:
        changed_result = f"rejected ({exc.status}: {exc.message})"
    exports_final = oracle_rows(os.path.join(out, "export.db"), "exports")
    lines = [
        "Runtime after interruption: " + after["status"] + (", no receipt" if after["receipt"] is None else ""),
        "Service after interruption: " + ", ".join(r["export_id"] for r in exports_after) + " exists",
        "Repeated key and same payload: " + ("existing receipt returned" if recovered["replayed"] else "new export"),
        "Repeated key and different payload: " + changed_result.split(" (")[0],
        f"Final number of export records: {len(exports_final)}",
    ]

    # Flawed variant: a fresh key generated for the retry.
    out2 = os.path.join(root, "e1_fresh_key")
    clock2 = SimClock()
    store2, svc2 = fresh_stores(out2, clock2, ws)
    store2.start_run(RUN, 3)
    store2.prepare("op-0002", RUN, "create_export", payload, op_key=key)
    try:
        Executor(store2, svc2).dispatch("op-0002", interrupt="after_commit")
    except Interrupted:
        pass
    clock2.advance(6.0)
    store2, svc2 = restart(out2, clock2, ws)
    fresh = svc2["export"].create_export(key + ":retry-1", payload)
    fresh_exports = oracle_rows(os.path.join(out2, "export.db"), "exports")

    result = {"source_revision": rev, "op_key": key, "payload_digest": digest(payload),
              "changed_payload_digest": digest(changed),
              "runtime_after_interruption": after, "service_after_interruption": exports_after,
              "decisions": decisions, "recovered_receipt": recovered, "changed_payload": changed_result,
              "service_final": exports_final, "events": store.events(), "observed": lines,
              "fresh_key_variant": {"retry_key": key + ":retry-1", "receipt": fresh,
                                    "exports": fresh_exports}}
    write_run(out, "result.json", result)
    return result


def run_cell(root, ws, tool, point, policy, revise_task=False):
    out = os.path.join(root, "matrix", f"{policy}_{tool}_{point}" + ("_revised" if revise_task else ""))
    clock = SimClock()
    store, svc = fresh_stores(out, clock, ws)
    store.start_run(RUN, 3)
    payload, key = op_spec(tool, "339d3a9f2656")
    store.prepare("op-0001", RUN, tool, payload, op_key=key)
    clock.advance(0.5)
    try:
        Executor(store, svc).dispatch("op-0001", interrupt=point)
    except Interrupted:
        pass
    clock.advance(5.0)
    store, svc = restart(out, clock, ws)
    if revise_task:
        store.revise_task(RUN, 4)
    status_after = store.op("op-0001")["status"]
    ex = Executor(store, svc)
    if policy == "reconciler":
        decision = Reconciler(store, ex, svc, clock).reconcile(store.op("op-0001")) \
            if status_after != "completed" else "continue from receipt"
    else:
        d = naive_retry_all(store, ex)
        decision = d[0][1] if d else "continue from receipt"
    if tool == "refresh_dashboard":
        clock.now = max(clock.now, 60.0)  # observe the job service at the same horizon for both policies
        svc["jobs"].query_by_client_op("op-0001")
    final = store.op("op-0001")
    return {"tool": tool, "class": CATALOG["tools"][tool]["recovery_class"], "point": point,
            "policy": policy, "status_after_interruption": status_after, "decision": decision,
            "final_status": final["status"], "external": external_state(out, tool, store)}


def e2_matrix(root, ws):
    rows = []
    for tool in ["read_revision", "create_export", "refresh_dashboard", "notify_oncall"]:
        for point in POINTS:
            r = run_cell(root, ws, tool, point, "reconciler")
            n = run_cell(root, ws, tool, point, "naive")
            r["naive_decision"], r["naive_external"] = n["decision"], n["external"]
            rows.append(r)
    stale = run_cell(root, ws, "notify_oncall", "before_dispatch", "reconciler", revise_task=True)
    write_run(os.path.join(root, "matrix"), "matrix.json", {"rows": rows, "stale_intent": stale})
    return rows, stale


def e3_notification(root, ws):
    results = {}
    for policy in ("naive", "reconciler"):
        for point in ("after_commit", "in_transit"):
            out = os.path.join(root, f"e3_{policy}_{point}")
            clock = SimClock()
            store, svc = fresh_stores(out, clock, ws)
            ex = Executor(store, svc)
            store.start_run(RUN, 3)
            store.prepare("op-0001", RUN, "read_revision", {"path": "report.py"})
            clock.advance(0.4)
            rev = ex.dispatch("op-0001")["version"]
            payload, key = op_spec("create_export", rev)
            clock.advance(1.1)
            store.prepare("op-0002", RUN, "create_export", payload, op_key=key)
            ex.dispatch("op-0002")
            clock.advance(0.8)
            store.prepare("op-0003", RUN, "notify_oncall", NOTIFY)
            clock.advance(0.2)
            try:
                ex.dispatch("op-0003", interrupt=point)
            except Interrupted:
                pass
            clock.advance(6.0)
            store, svc = restart(out, clock, ws)
            row_after = runtime_row(store, "op-0003")
            ex = Executor(store, svc)
            if policy == "naive":
                decisions = naive_retry_all(store, ex)
            else:
                decisions = Reconciler(store, ex, svc, clock).reconcile_all()
            msgs = oracle_messages(os.path.join(out, "notify.db"))
            entry = {"runtime_row_after_interruption": row_after, "decisions": decisions,
                     "messages": msgs, "message_count": len(msgs),
                     "repair_items": store.repair_items(), "op_final": runtime_row(store, "op-0003")}
            if policy == "reconciler":
                clock.advance(240.0)
                refused = resolve_repair_item(store, svc, "R-1", "agent:run-13", [],
                                              "delivered", {"message_id": "msg-001"})
                clock.advance(60.0)
                if point == "after_commit":
                    accepted = resolve_repair_item(
                        store, svc, "R-1", "user:dana", ["reports-oncall-lead"], "delivered",
                        {"message_id": "msg-001", "checked": "provider delivery log",
                         "sent_at": msgs[0]["sent_at"]})
                else:
                    accepted = resolve_repair_item(
                        store, svc, "R-1", "user:dana", ["reports-oncall-lead"], "not_delivered_resend",
                        {"checked": "provider delivery log", "matching_messages": 0})
                entry["resolution_refused"] = refused
                entry["resolution_accepted"] = accepted
                entry["repair_items_after"] = store.repair_items()
                entry["message_count_after_resolution"] = len(oracle_messages(os.path.join(out, "notify.db")))
                entry["op_after_resolution"] = runtime_row(store, "op-0003")
                entry["audit"] = [e for e in store.events() if e["kind"] in
                                  ("uncertain", "resolution_refused", "manual_resolution")]
            write_run(out, "result.json", entry)
            results[f"{policy}_{point}"] = entry
    return results


def e4_backoff(root, ws):
    out = os.path.join(root, "e4_backoff")
    clock = SimClock()
    store, svc = fresh_stores(out, clock, ws)
    store.start_run(RUN, 3)
    store.prepare("op-0001", RUN, "refresh_dashboard", JOB)
    clock.advance(0.5)
    try:
        Executor(store, svc).dispatch("op-0001", interrupt="after_commit")
    except Interrupted:
        pass
    clock.advance(5.0)
    restart_t = clock.now
    deadline = restart_t + 120.0
    outage_until = restart_t + 200.0
    store, svc = restart(out, clock, ws, jobs_unavailable_until=outage_until)
    log = []
    rec = Reconciler(store, Executor(store, svc), svc, clock,
                     backoff=Backoff(base_s=1.0, cap_s=30.0, max_attempts=20, seed=13),
                     deadline_s=deadline, log=log)
    first = rec.reconcile(store.op("op-0001"))
    after_first = runtime_row(store, "op-0001")
    t_first_end = clock.now
    clock.now = restart_t + 300.0
    store, svc = restart(out, clock, ws, jobs_unavailable_until=outage_until)
    rec2 = Reconciler(store, Executor(store, svc), svc, clock, deadline_s=None, log=log)
    second = rec2.reconcile(store.op("op-0001"))
    final = runtime_row(store, "op-0001")

    # Synchronized retries: 50 reconcilers that all saw the same outage at t=0.
    def herd(jitter, attempts=6, clients=50):
        rng = random.Random(7)
        times = []
        for _ in range(clients):
            t = 0.0
            for a in range(1, attempts + 1):
                bound = min(30.0, 1.0 * 2 ** (a - 1))
                t += rng.uniform(0, bound) if jitter else bound
                times.append(t)
        buckets = {}
        for t in times:
            b = int(t * 10)  # 100 ms windows
            buckets[b] = buckets.get(b, 0) + 1
        return {"requests": len(times), "max_per_window": max(buckets.values()),
                "busiest_window_start_s": max(buckets, key=buckets.get) / 10, "windows_used": len(buckets),
                "last_request_s": round(max(times), 3)}

    result = {"restart_t": restart_t, "deadline": deadline, "outage_until": outage_until,
              "first_pass": first, "first_pass_end_t": t_first_end, "after_first_pass": after_first,
              "second_pass_t": restart_t + 300.0, "second_pass": second, "final": final, "log": log,
              "repair_items": store.repair_items(),
              "herd_no_jitter": herd(False), "herd_full_jitter": herd(True),
              "jobs": oracle_rows(os.path.join(out, "jobs.db"), "jobs")}
    write_run(out, "result.json", result)
    return result


def e5_cancel(root, ws):
    res = {}
    for variant, cancel_at in (("flawed_late", 12.0), ("revised_late", 12.0), ("revised_early", 3.0)):
        out = os.path.join(root, f"e5_{variant}")
        clock = SimClock()
        store, svc = fresh_stores(out, clock, ws)
        ex = Executor(store, svc)
        store.start_run(RUN, 3)
        store.prepare("op-0001", RUN, "refresh_dashboard", JOB)
        t0 = clock.now
        receipt = ex.dispatch("op-0001")
        clock.now = t0 + cancel_at
        timeline = [{"t": t0, "what": f"submitted {receipt['job_id']}"}]
        if variant.startswith("flawed"):
            store.set_status("op-0001", "cancelled", kind="cancelled_locally",
                             detail={"note": "worker stopped waiting"})
            timeline.append({"t": clock.now, "what": "runtime marks op cancelled, stops polling"})
            clock.now = t0 + 60.0
        else:
            store.request_cancel("op-0001")
            ack = svc["jobs"].cancel(receipt["job_id"])
            timeline.append({"t": clock.now, "what": "cancel requested", "ack": ack})
            Reconciler(store, ex, svc, clock).poll_job(store.op("op-0001"))
        job = oracle_rows(os.path.join(out, "jobs.db"), "jobs")[0]
        svc["jobs"].get(receipt["job_id"])  # let the service settle its own state at the final time
        job = oracle_rows(os.path.join(out, "jobs.db"), "jobs")[0]
        op = store.op("op-0001")
        timeline.append({"t": job["finished_at"], "what": f"service: {job['job_id']} {job['status']}"})
        res[variant] = {"cancel_at": cancel_at, "runtime_status": op["status"],
                        "cancel_requested_at": op["cancel_requested_at"], "service_status": job["status"],
                        "service_finished_at": job["finished_at"], "no_return_at": job["no_return_at"],
                        "timeline": timeline, "agree": op["status"] == job["status"]}
        write_run(out, "result.json", res[variant])
    return res


def e6_retention(root, ws):
    res = {}
    for policy in ("naive_key_reuse", "reconciler"):
        out = os.path.join(root, f"e6_{policy}")
        clock = SimClock()
        store, svc = fresh_stores(out, clock, ws)
        store.start_run(RUN, 3)
        payload, key = op_spec("create_export", "339d3a9f2656")
        store.prepare("op-0001", RUN, "create_export", payload, op_key=key)
        try:
            Executor(store, svc).dispatch("op-0001", interrupt="after_commit")
        except Interrupted:
            pass
        clock.advance(30 * 3600.0)
        store, svc = restart(out, clock, ws)
        if policy == "naive_key_reuse":
            receipt = svc["export"].create_export(key, payload)
            store.set_status("op-0001", "completed", receipt=receipt)
            decision = "reuse key"
        else:
            decision = Reconciler(store, Executor(store, svc), svc, clock).reconcile(store.op("op-0001"))
        res[policy] = {"age_s": clock.now, "decision": decision,
                       "exports": [r["export_id"] for r in oracle_rows(os.path.join(out, "export.db"), "exports")],
                       "op_status": store.op("op-0001")["status"], "repair_items": store.repair_items()}
        write_run(out, "result.json", res[policy])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch13")
    args = ap.parse_args()
    root = args.output
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    ws = make_workspace()
    try:
        write_run(root, "catalog.json", CATALOG)
        e1 = e1_lost_receipt(root, ws)
        rows, stale = e2_matrix(root, ws)
        e3 = e3_notification(root, ws)
        e4 = e4_backoff(root, ws)
        e5 = e5_cancel(root, ws)
        e6 = e6_retention(root, ws)
    finally:
        ws.close()

    p = print
    p("== E1 lost receipt (runtime.db and export.db) ==")
    p(f"source revision read: {e1['source_revision']}")
    p(f"op_key: {e1['op_key']}  payload_digest: {e1['payload_digest']}")
    p("runtime row after interruption:", json.dumps(e1["runtime_after_interruption"]))
    for line in e1["observed"]:
        p("  " + line)
    p("recovered receipt:", json.dumps(e1["recovered_receipt"]))
    p("changed payload:", e1["changed_payload"], "digest", e1["changed_payload_digest"])
    fk = e1["fresh_key_variant"]
    p(f"fresh key per retry: {len(fk['exports'])} export records "
      f"({', '.join(r['export_id'] for r in fk['exports'])})")

    p("\n== E2 interruption point x recovery class ==")
    p(f"{'point':16} {'class':15} {'after':11} {'reconciler decision':26} {'final external state':34} naive")
    for r in rows:
        p(f"{r['point']:16} {r['class']:15} {r['status_after_interruption']:11} {r['decision']:26} "
          f"{r['external']:34} {r['naive_decision']}: {r['naive_external']}")
    p(f"stale intent (task revised 3 -> 4 before recovery): {stale['decision']}; "
      f"final status {stale['final_status']}; {stale['external']}")

    p("\n== E3 notification: naive retry-all vs reconciler ==")
    for k, v in e3.items():
        p(f"{k:24} messages={v['message_count']} op_status={v['op_final']['status']} "
          f"open_items={[i['item_id'] for i in v['repair_items'] if i['status'] == 'open']}")
    a, b = e3["reconciler_after_commit"], e3["reconciler_in_transit"]
    same = {k: (a['runtime_row_after_interruption'][k], b['runtime_row_after_interruption'][k])
            for k in a['runtime_row_after_interruption']}
    p("runtime row after interruption, after_commit vs in_transit:")
    for k, (x, y) in same.items():
        p(f"  {k:20} {json.dumps(x):28} {json.dumps(y):28} {'same' if x == y else 'DIFFERENT'}")
    p("repair item:", json.dumps({k: a["repair_items"][0][k] for k in ("item_id", "op_id", "reason", "evidence")}))
    p("resolution by agent:run-13:", json.dumps(a["resolution_refused"]))
    p("resolution by user:dana (after_commit):", json.dumps(a["resolution_accepted"]))
    p("resolution by user:dana (in_transit):", json.dumps(b["resolution_accepted"]))
    p(f"messages after resolution: after_commit={a['message_count_after_resolution']} "
      f"in_transit={b['message_count_after_resolution']}")

    p("\n== E4 backoff with full jitter, deadline (simulated seconds) ==")
    p(f"restart t={e4['restart_t']}  deadline t={e4['deadline']}  status endpoint down until t={e4['outage_until']}")
    for e in e4["log"]:
        if e["event"] == "retry":
            p(f"  t={e['t']:7.3f} attempt {e['attempt']} got {e['status']}; bound {e['bound_s']:5.1f}s; "
              f"wait {e['delay_s']:6.3f}s")
        else:
            p(f"  t={e['t']:7.3f} {e['event']}: " + json.dumps({k: v for k, v in e.items() if k not in ('t', 'event')}))
    p(f"first pass: {e4['first_pass']} -> status {e4['after_first_pass']['status']} (ended t={e4['first_pass_end_t']})")
    p(f"later sweep at t={e4['second_pass_t']}: {e4['second_pass']} -> status {e4['final']['status']}; "
      f"jobs at service: {len(e4['jobs'])}")
    for k in ("herd_no_jitter", "herd_full_jitter"):
        h = e4[k]
        p(f"50 reconcilers x 6 attempts, {k[5:]}: {h['requests']} requests; busiest 100 ms window "
          f"{h['max_per_window']} requests at t={h['busiest_window_start_s']}; {h['windows_used']} windows used; "
          f"last request t={h['last_request_s']}")

    p("\n== E5 cancellation ==")
    for k, v in e5.items():
        p(f"{k:14} cancel at +{v['cancel_at']}s (no return at +{v['no_return_at']}s): runtime={v['runtime_status']} "
          f"service={v['service_status']} at t={v['service_finished_at']} agree={v['agree']}")
        for step in v["timeline"]:
            if "ack" in step:
                p(f"    ack: {json.dumps(step['ack'])}")

    p("\n== E6 key retention (restart after 30 h; service keeps keys 24 h) ==")
    for k, v in e6.items():
        p(f"{k:16} decision={v['decision']}; exports={v['exports']}; op status={v['op_status']}")

    write_run(root, "summary.json", {"e1": e1["observed"], "matrix": rows, "stale": stale,
                                     "e3_counts": {k: v["message_count"] for k, v in e3.items()},
                                     "e5": e5, "e6": e6})


if __name__ == "__main__":
    main()
