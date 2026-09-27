"""Chapter 21: cost, latency and capacity.

    python3 ch21_cost.py --output runs/ch21 [--measure-commits]

Reads recorded runs from earlier chapters (runs/ch03, runs/ch11, runs/ch15),
prices them with the illustrative parameters in itl/ch21_cost.py, builds a
timeline of Chapter 3's run with its critical path, and runs a seeded
discrete-event simulation of an agent service. All times are simulated
seconds and all prices are invented cost units; nothing here measures a
provider. --measure-commits additionally times real SQLite commits on this
machine (machine-dependent, written to commit_latency.json).
"""
import argparse
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import write_run  # noqa: E402
from itl import ch21_cost as c  # noqa: E402

OUT = []


def say(s=""):
    OUT.append(s)


def need(path, chapter):
    if not os.path.exists(path):
        sys.exit(f"missing {path}: run the Chapter {chapter} companion first")


def fmt(x, n=2):
    return f"{x:.{n}f}"


# ------------------------------------------------------------ service model

def hybrid_templates(runs):
    return {s: r for (ctl, s), r in runs.items() if ctl == "hybrid"}


def service_sampler(templates, seed):
    rng = random.Random(seed)

    def sample():
        scen = c.pick(rng, c.MIX)
        run = templates[scen]
        steps = c.sample_steps(rng, run)
        return scen, steps, sum(d for _, d, _ in steps)
    return sample


def make_jobs(rate_fn, horizon, sampler, seed, tenant="t", thin_max=None):
    """Poisson arrivals; rate_fn(t) may vary (thinning with thin_max)."""
    rng = random.Random(seed)
    jobs = []
    t = 0.0
    lam_max = thin_max if thin_max else rate_fn(0.0)
    i = 0
    while True:
        t += rng.expovariate(lam_max)
        if t >= horizon:
            break
        if thin_max and rng.random() > rate_fn(t) / lam_max:
            continue
        scen, steps, svc = sampler()
        j = c.Job(f"{tenant}-{i}", tenant, t, svc, steps=steps,
                  tokens=0)
        j.scenario = scen
        jobs.append(j)
        i += 1
    return jobs


def latency_stats(jobs, deadline=None):
    done = [j.end - j.arrival for j in jobs if j.status == "completed"]
    n = len(jobs)
    out = {"arrived": n, "completed": len(done),
           "rejected": sum(1 for j in jobs if j.status == "rejected"),
           "expired": sum(1 for j in jobs if j.status == "expired"),
           "failed": sum(1 for j in jobs if j.status == "failed"),
           "p50": round(c.pctl(done, 0.50), 1), "p95": round(c.pctl(done, 0.95), 1),
           "p99": round(c.pctl(done, 0.99), 1)}
    if deadline is not None:
        out["within_deadline"] = sum(1 for x in done if x <= deadline)
    return out


# ------------------------------------------------------------ experiments

def exp_ledger_and_timeline(outdir, queue_wait_loaded, commit_s=None):
    need(os.path.join(c.RUNS, "ch03", "trace.json"), 3)
    trace = json.load(open(os.path.join(c.RUNS, "ch03", "trace.json")))
    messages = json.load(open(os.path.join(c.RUNS, "ch03", "messages.json")))
    led = c.ledger_from_trace(trace, messages)
    say("== 1. Chapter 3's run as a usage ledger (tokens approximate: JSON characters / 4)")
    say("  call  input  reusable  output  tools")
    for r in led:
        tools = ", ".join(x["name"] + (" (rejected)" if x["rejected"] else "") for x in r["tools"]) or "-"
        say(f"  {r['call']:>4}  {r['input']:>5}  {r['reusable']:>8}  {r['output']:>6}  {tools}")
    tin = sum(r["input"] for r in led)
    reu = sum(r["reusable"] for r in led)
    tout = sum(r["output"] for r in led)
    say(f"  totals: input {tin}, reusable {reu} ({100 * reu / tin:.1f}%), output {tout}")
    say()
    variants = {
        "idle, sequential tools, cache on": c.build_timeline(led, 0.0, False, True, commit_s),
        "idle, parallel reads, cache on": c.build_timeline(led, 0.0, True, True, commit_s),
        "idle, sequential tools, cache off": c.build_timeline(led, 0.0, False, False, commit_s),
        f"queued {queue_wait_loaded:.1f} s (p50 wait at 90% load), sequential, cache on":
            c.build_timeline(led, queue_wait_loaded, False, True, commit_s),
    }
    say("== 2. Where the time goes: critical path of Chapter 3's run (simulated seconds)")
    kinds = ["queue", "sandbox", "ttft", "generation", "tool", "commit", "verification"]
    heads = ["queue", "sandbox", "ttft", "gen", "tools", "commits", "verify"]
    say("  variant" + " " * 58 + " ".join(f"{k:>7}" for k in heads) + "   total   model$  sandbox$")
    for name, tl in variants.items():
        cp = tl["critical_path_by_kind"]
        say(f"  {name:<64} " + " ".join(f"{cp.get(k, 0.0):>7.2f}" for k in kinds)
            + f" {tl['total_seconds']:>7.2f} {tl['model_cost_units']:>8.3f} {tl['sandbox_cost_units']:>8.3f}")
    base = variants["idle, sequential tools, cache on"]
    tools = sum(s["seconds"] for s in base["segments"] if s["kind"] in ("tool", "verification"))
    model = sum(s["seconds"] for s in base["segments"] if s["kind"] in ("ttft", "generation"))
    say(f"  idle run: model time {model:.2f} s, tool and verification time {tools:.2f} s, "
        f"share of run in tools {100 * tools / base['total_seconds']:.1f}%")
    cin = sum(s["cost_units"] for s in base["segments"] if s["kind"] == "ttft")
    cout = sum(s["cost_units"] for s in base["segments"] if s["kind"] == "generation")
    say(f"  idle run, cache on: input {cin:.3f} units, output {cout:.3f} units "
        f"({100 * cout / (cin + cout):.1f}% of model cost from {100 * tout / (tin + tout):.1f}% of tokens)")
    say()
    write_run(outdir, "ledger_ch03.json", led)
    write_run(outdir, "timeline_ch03.json", variants)
    return led, variants


def exp_cost(outdir, runs):
    say("== 3. Cost per run over Chapter 11's recorded runs (price table "
        f"{c.PRICES['table_id']}, digest {c.price_table_digest()})")
    say("  units: 1 unit = 1,000 uncached input tokens on the large model; cache on, full hit rate")
    say("  controller scenario     outcome                    calls  tokens  cost(machine)  person  verified")
    rows = []
    for ctl in ("loop", "hybrid", "workflow"):
        for scen, _ in c.MIX:
            r = runs[(ctl, scen)]
            total, parts = c.run_cost(r, human=False)
            rows.append({"controller": ctl, "scenario": scen, "outcome": r["outcome"],
                         "model_calls": r["model_calls"], "tokens": r["tokens"],
                         "machine_units": total, "parts": parts, "person": c.needs_person(r),
                         "verified_success": c.verified_success(r), "claimed": c.claimed_completion(r)})
            if ctl != "workflow":
                say(f"  {ctl:<10} {scen:<11} {r['outcome']:<26} {r['model_calls']:>5} {r['tokens']:>7}"
                    f" {total:>14.3f} {c.needs_person(r):>7}  {c.verified_success(r)}")
    write_run(outdir, "run_costs.json", rows)
    say()

    # expected values under the scenario mix
    say("== 4. Four ways to report cost, under the scenario mix "
        + ", ".join(f"{s} {w:.2f}" for s, w in c.MIX))
    summary = {}
    for ctl in ("loop", "hybrid", "workflow"):
        e_cost_m = e_cost_h = e_cost_d = e_calls = e_succ = e_claim = 0.0
        for scen, w in c.MIX:
            r = runs[(ctl, scen)]
            m, _ = c.run_cost(r, human=False)
            h, _ = c.run_cost(r, human=True)
            e_cost_m += w * m
            e_cost_h += w * h
            e_cost_d += w * c.run_cost(r, human=True, discovery=True)[0]
            e_calls += w * r["model_calls"]
            e_succ += w * c.verified_success(r)
            e_claim += w * c.claimed_completion(r)
        summary[ctl] = {
            "cost_per_attempt_machine": round(e_cost_m, 4),
            "cost_per_call_machine": round(e_cost_m / e_calls, 4),
            "claimed_completion_rate": round(e_claim, 3),
            "cost_per_claimed_completion_machine": round(e_cost_m / e_claim, 4) if e_claim else None,
            "verified_success_rate": round(e_succ, 3),
            "cost_per_verified_success_machine": round(e_cost_m / e_succ, 4),
            "cost_per_verified_success_with_people": round(e_cost_h / e_succ, 4),
            "cost_per_verified_success_with_people_and_discovery": round(e_cost_d / e_succ, 4),
        }
    say("  metric                                   loop      hybrid    workflow")
    for k, label in [("cost_per_call_machine", "cost per model call"),
                     ("cost_per_attempt_machine", "cost per attempt"),
                     ("claimed_completion_rate", "claimed completion rate"),
                     ("cost_per_claimed_completion_machine", "cost per claimed completion"),
                     ("verified_success_rate", "verified success rate"),
                     ("cost_per_verified_success_machine", "cost per verified success"),
                     ("cost_per_verified_success_with_people", "  ... with people (100 per decision)"),
                     ("cost_per_verified_success_with_people_and_discovery",
                      "  ... and silent failures found later")]:
        vals = [summary[x][k] for x in ("loop", "hybrid", "workflow")]
        say(f"  {label:<38}" + "".join(f"{v:>10.3f}" if v is not None else f"{'-':>10}" for v in vals))
    say()

    # distribution: batches of 40 tasks
    rng = random.Random(2121)
    batches = {"machine": {"loop": [], "hybrid": []}, "people": {"loop": [], "hybrid": []}}
    per_attempt = {"loop": [], "hybrid": []}
    B, N = 1000, 40
    for _ in range(B):
        scen_list = [c.pick(rng, c.MIX) for _ in range(N)]
        for ctl in ("loop", "hybrid"):
            cm = cp = succ = 0
            for s in scen_list:
                r = runs[(ctl, s)]
                cm += c.run_cost(r, human=False)[0]
                cp += c.run_cost(r, human=True, discovery=True)[0]
                succ += c.verified_success(r)
            batches["machine"][ctl].append(round(cm / succ, 3) if succ else None)
            batches["people"][ctl].append(round(cp / succ, 3) if succ else None)
    for ctl in ("loop", "hybrid"):
        for s, w in c.MIX:
            r = runs[(ctl, s)]
            per_attempt[ctl].append({"scenario": s, "weight": w, "machine": c.run_cost(r, human=False)[0],
                                     "with_people": c.run_cost(r, human=True, discovery=True)[0],
                                     "success": c.verified_success(r)})
    say(f"== 5. Cost per verified success in {B} seeded batches of {N} tasks")
    dist = {}
    for kind, label in (("machine", "machine cost"), ("people", "with people and later discovery")):
        dist[kind] = {}
        for ctl in ("loop", "hybrid"):
            xs = [x for x in batches[kind][ctl] if x is not None]
            d = {"p10": round(c.pctl(xs, 0.10), 2), "p50": round(c.pctl(xs, 0.50), 2),
                 "p90": round(c.pctl(xs, 0.90), 2), "min": round(min(xs), 2), "max": round(max(xs), 2)}
            dist[kind][ctl] = d
            say(f"  {label:<32} {ctl:<7} p10 {d['p10']:>7.2f}  p50 {d['p50']:>7.2f}  p90 {d['p90']:>7.2f}"
                f"  range {d['min']:.2f} to {d['max']:.2f}")
        worse = sum(1 for a, b in zip(batches[kind]["loop"], batches[kind]["hybrid"])
                    if a is not None and b is not None and b > a)
        say(f"  {label}: batches in which hybrid costs more per verified success than loop: {worse} of {B}")
    say()
    write_run(outdir, "cost_summary.json", {"summary": summary, "batch_distribution": dist,
                                            "batches": batches, "per_attempt": per_attempt,
                                            "mix": c.MIX, "prices": c.PRICES})
    return summary


def exp_cache(outdir, runs):
    say("== 6. Sensitivity to cache hit rate (machine cost per verified success)")
    hs = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = {}
    say("  controller   cache off  " + "  ".join(f"hit {h:.2f}" for h in hs))
    for ctl in ("loop", "hybrid"):
        row = {}
        succ = sum(w * c.verified_success(runs[(ctl, s)]) for s, w in c.MIX)
        off = sum(w * c.run_cost(runs[(ctl, s)], caching=False, human=False)[0] for s, w in c.MIX) / succ
        row["off"] = round(off, 3)
        for h in hs:
            row[str(h)] = round(sum(w * c.run_cost(runs[(ctl, s)], caching=True, hit_rate=h, human=False)[0]
                                    for s, w in c.MIX) / succ, 3)
        res[ctl] = row
        say(f"  {ctl:<10} {row['off']:>10.3f}  " + "  ".join(f"{row[str(h)]:>8.3f}" for h in hs))
    # break-even hit rate on input only
    for ctl in ("loop", "hybrid"):
        I = sum(w * sum(x["input"] for x in runs[(ctl, s)]["ledger"]) for s, w in c.MIX)
        R = sum(w * sum(x["reusable"] for x in runs[(ctl, s)]["ledger"]) for s, w in c.MIX)
        p = c.PRICES["models"]["large"]
        be = (p["cache_write"] - p["input"]) * I / ((p["cache_write"] - p["cache_read"]) * R)
        res[ctl]["reusable_share"] = round(R / I, 3)
        res[ctl]["break_even_hit_rate"] = round(be, 3)
        say(f"  {ctl}: reusable share of input {R / I:.3f}; caching pays for itself above hit rate {be:.3f}")
    # reasoning tokens
    for ctl in ("loop", "hybrid"):
        succ = sum(w * c.verified_success(runs[(ctl, s)]) for s, w in c.MIX)
        with_r = sum(w * c.run_cost(runs[(ctl, s)], reasoning=300, human=False)[0] for s, w in c.MIX) / succ
        res[ctl]["with_300_reasoning_per_call"] = round(with_r, 3)
        say(f"  {ctl}: adding 300 reasoning tokens per call raises cost per verified success to {with_r:.3f}")
    say()
    write_run(outdir, "cache_sensitivity.json", res)
    return res


def exp_routing(outdir, runs):
    say("== 7. Routing on the hybrid: small model first, large model when the verifier reports failure")
    say("  20,000 sampled tasks per row; q = chance the small model succeeds where the large one did")
    say("  (q is not observed in any recording; it is swept)")
    out = {}
    say("  policy           q     success   machine/success   with people/success   mean seconds to outcome")
    for policy in ("large only", "small only", "cascade"):
        qs = [1.0] if policy == "large only" else [0.9, 0.7, 0.5]
        for q in qs:
            rng = random.Random(777)
            cost = succ = mcost = secs = 0.0
            n = 20000
            for _ in range(n):
                s = c.pick(rng, c.MIX)
                r = runs[("hybrid", s)]
                large_ok = c.verified_success(r)
                lc, _ = c.run_cost(r, model="large", human=False)
                sc, _ = c.run_cost(r, model="small", human=False)
                small_ok = large_ok and rng.random() < q
                person = c.PRICES["human_review"]
                if policy == "large only":
                    cost += lc + person * c.needs_person(r)
                    mcost += lc
                    secs += r["sim_seconds"]
                    succ += large_ok
                elif policy == "small only":
                    cost += sc
                    mcost += sc
                    secs += r["sim_seconds"]
                    if small_ok:
                        succ += 1
                        cost += person * c.needs_person(r)
                    elif s != "cancel":
                        cost += person          # the verifier's failure goes to a person
                else:
                    cost += sc
                    mcost += sc
                    secs += r["sim_seconds"]
                    if small_ok:
                        succ += 1
                        cost += person * c.needs_person(r)
                    elif s == "cancel":
                        pass                    # the person stopped it; nothing to escalate
                    else:
                        cost += lc + person * c.needs_person(r)
                        mcost += lc
                        secs += r["sim_seconds"]
                        succ += large_ok
            key = f"{policy}/{q}"
            out[key] = {"success_rate": round(succ / n, 4), "machine_per_success": round(mcost / succ, 3),
                        "with_people_per_success": round(cost / succ, 3), "mean_seconds": round(secs / n, 2)}
            say(f"  {policy:<14} {q:>5.2f}  {succ / n:>8.3f}  {mcost / succ:>15.3f}  {cost / succ:>20.3f}"
                f"  {secs / n:>22.2f}")
    say()
    write_run(outdir, "routing.json", out)
    return out


def exp_budgets(outdir, runs):
    say("== 8. Budgets checked before dispatch versus after the fact")
    r = runs[("loop", "wander")]
    budget, reserve = 5.0, 0.5
    p = c.PRICES["models"]["large"]
    max_out = 400
    spent = 0.0
    post = pre = None
    trail = []
    for row in r["ledger"]:
        cc = c.call_cost(row)
        cost = cc["input_units"] + cc["output_units"]
        trail.append(round(cost, 4))
    # post-hoc: check after each call
    s = 0.0
    for i, x in enumerate(trail):
        s += x
        if s > budget:
            post = {"stopped_after_call": i + 1, "spent": round(s, 3), "over_by": round(s - budget, 3)}
            break
    # pre-dispatch: the runtime knows the request it is about to send, so it prices the
    # prefix it shares with the previous request at the cache-read rate, the rest at the
    # write rate, and the output at the call's max_tokens; it keeps a reserve.
    s = 0.0
    for i, row in enumerate(r["ledger"]):
        projection = (row["reusable"] * p["cache_read"] + (row["input"] - row["reusable"]) * p["cache_write"]
                      + max_out * p["output"]) / 1000
        if s + projection + reserve > budget:
            pre = {"refused_call": i + 1, "spent": round(s, 3), "projection": round(projection, 3),
                   "left_for_reserve": round(budget - s, 3)}
            break
        s += trail[i]
    say(f"  loop/wander, budget {budget} units, reserve {reserve}, max_tokens {max_out}")
    say(f"  cost of each call: {trail}")
    say(f"  after-the-fact check : stopped after call {post['stopped_after_call']}, spent {post['spent']}, "
        f"over budget by {post['over_by']}")
    say(f"  before-dispatch check: refused call {pre['refused_call']} (projected {pre['projection']}), "
        f"spent {pre['spent']}, {pre['left_for_reserve']} left for the reserve")

    # fleet: concurrent runs against one tenant budget
    rng = random.Random(4242)
    tenant_budget, run_cap = 60.0, 4.0
    runs_list = [runs[("hybrid", c.pick(rng, c.MIX))] for _ in range(30)]
    costs = [[c.call_cost(x)["input_units"] + c.call_cost(x)["output_units"] for x in rr["ledger"]]
             for rr in runs_list]

    def lockstep(admit_with_reservation):
        billed = 0.0          # what the tenant's meter shows; updated when calls finish
        reserved = 0.0
        active = []
        refused = 0
        for i, cs in enumerate(costs):
            if admit_with_reservation:
                if billed + reserved + run_cap > tenant_budget:
                    refused += 1
                    continue
                reserved += run_cap
            active.append({"i": i, "k": 0, "spent": 0.0})
        stopped = 0
        rounds = 0
        while any(a["k"] < len(costs[a["i"]]) for a in active):
            rounds += 1
            this_round = 0.0
            for a in active:
                cs = costs[a["i"]]
                if a["k"] >= len(cs):
                    continue
                x = cs[a["k"]]
                if admit_with_reservation:
                    if a["spent"] + x > run_cap:
                        a["k"] = len(cs)
                        stopped += 1
                        continue
                elif billed >= tenant_budget:
                    a["k"] = len(cs)
                    stopped += 1
                    continue
                a["spent"] += x
                a["k"] += 1
                this_round += x
            billed += this_round      # the meter catches up after the round
        if admit_with_reservation:
            pass
        return {"admitted": len(active), "refused_at_admission": refused, "runs_stopped": stopped,
                "runs_completed": len(active) - stopped,
                "billed": round(billed, 3), "over_budget_by": round(max(0.0, billed - tenant_budget), 3)}

    shared = lockstep(False)
    reserv = lockstep(True)
    say(f"  30 concurrent runs, tenant budget {tenant_budget}: check shared meter before each call -> "
        f"billed {shared['billed']}, over by {shared['over_budget_by']}, {shared['runs_stopped']} runs cut off, "
        f"{shared['runs_completed']} completed")
    say(f"  reserve {run_cap} per run at admission -> admitted {reserv['admitted']}, refused "
        f"{reserv['refused_at_admission']}, billed {reserv['billed']}, over by {reserv['over_budget_by']}, "
        f"{reserv['runs_stopped']} stopped at its own cap, {reserv['runs_completed']} completed")
    say()
    write_run(outdir, "budgets.json", {"per_run": {"budget": budget, "reserve": reserve, "calls": trail,
                                                   "after_the_fact": post, "before_dispatch": pre},
                                       "fleet": {"shared_meter": shared, "reservation": reserv,
                                                 "tenant_budget": tenant_budget, "run_cap": run_cap}})


def exp_sweep(outdir, runs, W=16, horizon=3 * 3600):
    templates = hybrid_templates(runs)
    est = service_sampler(templates, 99)
    svc = [est()[2] for _ in range(20000)]
    ES = statistics.fmean(svc)
    S90 = c.pctl(svc, 0.90)
    say(f"== 9. Agent service simulation: {W} workers, runs drawn from the hybrid recordings by the mix")
    say(f"  service time: mean {ES:.1f} s, p50 {c.pctl(svc, 0.5):.1f}, p90 {S90:.1f}, p99 {c.pctl(svc, 0.99):.1f}"
        f"; capacity {W / ES * 60:.2f} runs/min")
    rhos = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    deadline = 150.0
    res = {"unbounded": [], "admission": [], "W": W, "mean_service": ES, "deadline": deadline,
           "service_p90": S90}
    say(f"  admission control: queue bound {2 * W}, deadline {deadline:.0f} s end to end, "
        f"start refused when wait + p90 service ({S90:.1f} s) would pass it")
    say("  load   policy      arrived  completed  refused  p50      p95      p99      in-deadline  util")
    little = None
    for rho in rhos:
        lam = rho * W / ES
        for pol in ("unbounded", "admission"):
            sampler = service_sampler(templates, 1000 + int(rho * 100))
            jobs = make_jobs(lambda t: lam, horizon, sampler, seed=int(rho * 1000))
            policy = {} if pol == "unbounded" else {"queue_cap": 2 * W, "deadline_s": deadline,
                                                     "service_estimate_s": S90}
            info = c.simulate_queue(jobs, W, policy)
            st = latency_stats(jobs, deadline)
            busy = sum(j.service for j in jobs if j.status == "completed")
            util = busy / (W * info["t_end"])
            st.update({"rho": rho, "utilization": round(util, 3), "t_end": round(info["t_end"], 1)})
            refused = st["rejected"] + st["expired"]
            say(f"  {rho:>4.2f}  {pol:<10} {st['arrived']:>8} {st['completed']:>10} {refused:>8}  "
                f"{st['p50']:>7.1f}  {st['p95']:>7.1f}  {st['p99']:>7.1f}  {st['within_deadline']:>11}  {util:>5.3f}")
            if pol == "unbounded":
                done = [j for j in jobs if j.status == "completed"]
                L = info["area"] / info["t_end"]
                lam_eff = len(done) / info["t_end"]
                Wbar = statistics.fmean(j.end - j.arrival for j in done)
                st["little"] = {"L": round(L, 4), "lambda": round(lam_eff, 5), "W": round(Wbar, 3),
                                "lambda_W": round(lam_eff * Wbar, 4)}
                if rho == 0.9:
                    little = st["little"]
                    waits = [j.start - j.arrival for j in done]
                    st["wait_p50"] = round(c.pctl(waits, 0.5), 2)
                    res["wait_p50_at_0.9"] = st["wait_p50"]
            res[pol].append(st)
    say(f"  Little's law at load 0.90 (unbounded, the path starts and ends empty): L = {little['L']}, "
        f"lambda x W = {little['lambda']} x {little['W']} = {little['lambda_W']}")
    say()
    write_run(outdir, "sweep.json", res)
    return res, templates, ES, S90


def exp_noisy(outdir, templates, ES, S90, W=16):
    say("== 10. A noisy tenant: three quiet tenants at 0.18 of capacity each; tenant N at 0.10, "
        "bursting to 1.00 from minute 20 to 50")
    horizon = 90 * 60
    cap_rate = W / ES
    burst = (20 * 60, 50 * 60)
    deadline = 150.0

    def jobs_for_all():
        js = []
        for k, t in enumerate(["A", "B", "C"]):
            js += make_jobs(lambda _t: 0.18 * cap_rate, horizon, service_sampler(templates, 500 + k),
                            seed=600 + k, tenant=t)
        js += make_jobs(lambda _t: (1.0 if burst[0] <= _t < burst[1] else 0.10) * cap_rate, horizon,
                        service_sampler(templates, 509), seed=609, tenant="N", thin_max=1.0 * cap_rate)
        js.sort(key=lambda j: j.arrival)
        return js

    policies = {
        "one FIFO, unbounded": {},
        "global bound + deadline": {"queue_cap": 2 * W, "deadline_s": deadline, "service_estimate_s": S90},
        "per-tenant limits + fair": {"queue_cap": 2 * W, "deadline_s": deadline, "service_estimate_s": S90,
                                     "tenant_running_cap": 6, "tenant_queue_cap": 8, "fair": True},
    }
    res = {}
    say("  policy                    group   arrived  completed  refused  p50     p95     in-deadline")
    for name, pol in policies.items():
        jobs = jobs_for_all()
        c.simulate_queue(jobs, W, pol)
        res[name] = {}
        for group, members in (("quiet", {"A", "B", "C"}), ("N", {"N"})):
            g = [j for j in jobs if j.tenant in members]
            st = latency_stats(g, deadline)
            res[name][group] = st
            say(f"  {name:<25} {group:<6} {st['arrived']:>8} {st['completed']:>10} "
                f"{st['rejected'] + st['expired']:>8}  {st['p50']:>6.1f}  {st['p95']:>6.1f}  {st['within_deadline']:>11}")
        # time series for the figure: quiet tenants by arrival minute bucket (5 min)
        series = []
        for b0 in range(0, horizon, 300):
            g = [j for j in jobs if j.tenant != "N" and b0 <= j.arrival < b0 + 300]
            lat = [j.end - j.arrival for j in g if j.status == "completed"]
            nserved = sum(1 for j in g if j.status != "completed")
            series.append({"minute": b0 / 60, "quiet_p95": round(c.pctl(lat, 0.95), 1) if lat else None,
                           "quiet_p50": round(c.pctl(lat, 0.5), 1) if lat else None,
                           "quiet_not_served": nserved, "quiet_arrived": len(g)})
        res[name]["series"] = series
    say()
    write_run(outdir, "noisy_tenant.json", res)
    return res


def exp_rate_limit(outdir, templates, ES, W=16):
    say("== 11. Provider token limit: blind retry after 429 versus a runtime-side limiter")
    horizon = 3600
    rho = 0.75
    lam = rho * W / ES
    probe = service_sampler(templates, 71)
    toks = [sum(n for _, _, n in probe()[1]) for _ in range(5000)]
    per_run = statistics.fmean(toks)
    demand = lam * per_run * 60
    tpm = demand / 0.9
    say(f"  load {rho} of workers; mean tokens per run {per_run:.0f}; demand {demand:.0f} tokens/min; "
        f"limit {tpm:.0f} tokens/min (demand at 90% of the limit), burst 5 s")
    say(f"  capacity: workers {W / ES * 60:.2f} runs/min, provider {tpm / per_run:.2f} runs/min")
    res = {}
    say("  mode                completed  failed  429s   p50     p95     p99     wasted tokens  held s/run  queue s/run")
    say("  (latency percentiles over completed runs; held = seconds a run occupies a worker and its sandbox)")
    for mode, workers in (("blind", W), ("limiter", W)):
        jobs = make_jobs(lambda t: lam, horizon, service_sampler(templates, 72), seed=73)
        stats = c.simulate_steps(jobs, workers, tpm, 5.0, mode, random.Random(74))
        st = latency_stats(jobs)
        st.update(stats)
        outcome = [j.end - j.arrival for j in jobs if j.status in ("completed", "failed")]
        st["p95_all_outcomes"] = round(c.pctl(outcome, 0.95), 1)
        st["retries"] = sum(j.retries for j in jobs)
        held = [j.end - j.start for j in jobs if j.start is not None and j.end is not None]
        st["mean_worker_seconds_per_run"] = round(statistics.fmean(held), 2)
        waits = [j.start - j.arrival for j in jobs if j.start is not None]
        st["mean_admission_wait"] = round(statistics.fmean(waits), 2)
        name = f"{mode}, {workers} workers"
        res[name] = st
        say(f"  {name:<20} {st['completed']:>9} {st['failed_runs']:>7} {st['429']:>5}  {st['p50']:>6.1f}  "
            f"{st['p95']:>6.1f}  {st['p99']:>6.1f}  {st['wasted_tokens']:>12}  {st['mean_worker_seconds_per_run']:>10.2f}"
            f"  {st['mean_admission_wait']:>10.2f}")
    say()
    write_run(outdir, "rate_limit.json", {"tpm": tpm, "tokens_per_run": per_run, "results": res})
    return res


def exp_workers(outdir, W=16):
    path = os.path.join(c.RUNS, "ch15", "exp5_comparison.json")
    need(path, 15)
    d = json.load(open(path))
    say("== 12. Chapter 15's one worker versus two workers, as capacity")
    res = {}
    for case in ("clean", "with_malformed_first_answer"):
        for arr in ("one_worker", "two_workers"):
            x = d[case][arr]
            slot_s = sum(t["finish"] - t["start"] for t in x["schedule"]["tasks"].values())
            per_min = W * 60 / slot_s
            res[f"{case}/{arr}"] = {"latency_s": x["latency_s"], "slot_seconds": round(slot_s, 3),
                                    "tokens": x["tokens"], "reviews_per_min_at_16_slots": round(per_min, 2)}
            say(f"  {case:<28} {arr:<11} latency {x['latency_s']:>6.2f} s  slot-seconds {slot_s:>6.2f}  "
                f"tokens {x['tokens']:>5}  saturates at {per_min:.1f} reviews/min on {W} slots")
    say()
    write_run(outdir, "workers_capacity.json", res)


def measure_commits(outdir, n=300):
    d = tempfile.mkdtemp(prefix="ch21-")
    db = sqlite3.connect(os.path.join(d, "t.db"))
    db.execute("create table ops(id integer primary key, state text, payload text)")
    db.commit()
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        db.execute("insert into ops(state, payload) values (?, ?)", ("dispatched", "x" * 200))
        db.commit()
        ts.append((time.perf_counter() - t0) * 1000)
    db.close()
    conn = sqlite3.connect(os.path.join(d, "t.db"))
    mode = conn.execute("pragma journal_mode").fetchone()[0]
    sync = conn.execute("pragma synchronous").fetchone()[0]
    conn.close()
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    out = {"commits": n, "median_ms": round(statistics.median(ts), 3), "p95_ms": round(c.pctl(ts, 0.95), 3),
           "p99_ms": round(c.pctl(ts, 0.99), 3), "journal_mode": mode, "synchronous": sync, "machine": os.uname().machine,
           "note": "machine- and filesystem-dependent; not part of the deterministic output"}
    say(f"== 13. Measured SQLite commit latency on this machine ({n} commits, journal_mode={mode}): "
        f"median {out['median_ms']} ms, p95 {out['p95_ms']} ms, p99 {out['p99_ms']} ms")
    write_run(outdir, "commit_latency.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch21")
    ap.add_argument("--measure-commits", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    need(os.path.join(c.RUNS, "ch11", "summary.json"), 11)
    runs = c.load_ch11()
    say(f"Chapter 11 recordings loaded: {len(runs)} runs; token totals match summary.json for every run")
    say()
    # The sweep runs first because the timeline uses its queue wait; output order follows the chapter.
    saved = list(OUT)
    sweep, templates, ES, S90 = exp_sweep(a.output, runs)
    sweep_text = OUT[len(saved):]
    del OUT[len(saved):]
    exp_ledger_and_timeline(a.output, sweep["wait_p50_at_0.9"])
    exp_cost(a.output, runs)
    exp_cache(a.output, runs)
    exp_routing(a.output, runs)
    exp_budgets(a.output, runs)
    OUT.extend(sweep_text)
    exp_noisy(a.output, templates, ES, S90)
    exp_rate_limit(a.output, templates, ES)
    exp_workers(a.output)
    with open(os.path.join(a.output, "stdout.txt"), "w") as f:
        f.write("\n".join(OUT) + "\n")
    print("\n".join(OUT))
    if a.measure_commits:
        n = len(OUT)
        measure_commits(a.output)
        print("\n".join(OUT[n:]))


if __name__ == "__main__":
    main()
