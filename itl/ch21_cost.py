"""Chapter 21 helpers: usage ledgers, a price model, a run timeline and a
discrete-event simulation of an agent service.

Everything here is deterministic given its seed. Prices and latencies are
invented parameters, stated in PRICES and TIMING; they are not vendor prices
or measurements of any provider. Token counts use Chapter 11's estimator
(characters of the JSON / 4), which reproduces Chapter 11's recorded totals.
"""
import hashlib
import heapq
import json
import math
import os
import random
from collections import defaultdict, deque

from itl.core import Workspace, report_tools
from itl.ch11_control import deploy_tool, estimate_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")

# ----------------------------------------------------------------- parameters

# One cost unit is the price of 1,000 uncached input tokens on the "large"
# model. Every number below is an illustrative parameter.
PRICES = {
    "table_id": "pt-illustrative-2026-09",
    "models": {
        "large": {"input": 1.00, "cache_write": 1.25, "cache_read": 0.10, "output": 5.00},
        "small": {"input": 0.20, "cache_write": 0.25, "cache_read": 0.02, "output": 1.00},
    },
    "sandbox_per_second": 0.002,
    "tool_fee": {"read_file": 0.0, "replace_text": 0.0, "run_tests": 0.0, "deploy": 0.0},
    "human_review": 100.0,
}

TIMING = {
    "sandbox_start_s": 2.0,
    "commit_s": 0.004,          # one State store commit; two per tool operation (Chapter 12)
    "ttft_base_s": 0.5,
    "prefill_uncached_s_per_token": 0.00025,
    "prefill_cached_s_per_token": 0.000025,
    "decode_s_per_token": 0.020,
    # Chapter 11's simulated tool durations (itl/ch11_control.py SimClock).
    "tool_s": {"read_file": 0.2, "replace_text": 0.3, "run_tests": 6.0, "deploy": 5.0},
    "verifier_check_s": 0.1,
}


def price_table_digest():
    return hashlib.sha256(json.dumps(PRICES, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------- the ledgers

_DEFS = None


def definitions():
    global _DEFS
    if _DEFS is None:
        ws = Workspace()
        reg = report_tools(ws)
        deploy_tool(reg, ws)
        _DEFS = {d["name"]: d for d in reg.definitions()}
        ws.close()
    return _DEFS


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
            break
        n += 1
    return n


def ledger_from_trace(trace, messages):
    """Per-call usage for a recorded run.

    input:    Chapter 11's estimate of messages + tool definitions
    reusable: the part of the input identical to the previous request, compared
              message by message after identical tool definitions (a prefix a
              provider cache could serve)
    output:   estimate of the assistant message the call returned
    tools:    names of the tool calls in that message, with rejected calls marked
    """
    defs = definitions()
    rows = []
    prev = None
    for i, req in enumerate(trace):
        msgs = req["messages"]
        tdefs = [defs[n] for n in req["tools"]]
        tin = estimate_tokens(msgs) + estimate_tokens(tdefs)
        reusable = 0
        if prev is not None and prev["tools"] == req["tools"]:
            k = _common_prefix(prev["messages"], msgs)
            reusable = estimate_tokens(tdefs) + (estimate_tokens(msgs[:k]) if k else 0)
            reusable = min(reusable, tin)
        # the reply is the first message after this request's messages, if recorded
        if i + 1 < len(trace):
            nxt = trace[i + 1]["messages"]
            reply = nxt[len(msgs)] if len(nxt) > len(msgs) else None
            following = nxt[len(msgs) + 1:]
        else:
            reply = messages[len(msgs)] if len(messages) > len(msgs) else None
            following = messages[len(msgs) + 1:]
        if reply is None or reply.get("role") != "assistant":
            reply = {"role": "assistant", "content": ""}
            following = []
        tout = estimate_tokens(reply)
        tools = []
        results = {m.get("tool_call_id"): m for m in following if m.get("role") == "tool"}
        for c in reply.get("tool_calls") or []:
            res = results.get(c.get("id"), {})
            content = res.get("content")
            status = content.get("status") if isinstance(content, dict) else None
            tools.append({"name": c["name"], "rejected": status in ("input_error", "refused", "denied",
                                                                   "blocked")})
        rows.append({"call": i + 1, "input": tin, "reusable": reusable, "output": tout, "tools": tools})
        prev = req
    return rows


def load_ch11():
    base = os.path.join(RUNS, "ch11")
    summary = json.load(open(os.path.join(base, "summary.json")))
    out = {}
    for r in summary:
        c, s = r["controller"], r["scenario"]
        d = os.path.join(base, c, s)
        if not os.path.isdir(d):
            continue
        trace = json.load(open(os.path.join(d, "trace.json")))
        messages = json.load(open(os.path.join(d, "messages.json")))
        led = ledger_from_trace(trace, messages)
        assert sum(x["input"] for x in led) == r["tokens"], (c, s)
        out[(c, s)] = dict(r, ledger=led)
    return out


# ----------------------------------------------------------------- pricing

def call_cost(row, model="large", caching=True, hit_rate=1.0, reasoning=0):
    p = PRICES["models"][model]
    tin = row["input"]
    if caching:
        read = int(round(hit_rate * row["reusable"]))
        write = tin - read
        cin = (read * p["cache_read"] + write * p["cache_write"]) / 1000
    else:
        read, write = 0, 0
        cin = tin * p["input"] / 1000
    cout = (row["output"] + reasoning) * p["output"] / 1000
    return {"input_units": cin, "output_units": cout, "cache_read_tokens": read, "cache_write_tokens": write}


def run_cost(run, model="large", caching=True, hit_rate=1.0, reasoning=0, human=True, discovery=False):
    model_units = 0.0
    parts = defaultdict(float)
    for row in run["ledger"]:
        c = call_cost(row, model, caching, hit_rate, reasoning)
        parts["input"] += c["input_units"]
        parts["output"] += c["output_units"]
        for t in row["tools"]:
            if not t["rejected"]:
                parts["tool_fees"] += PRICES["tool_fee"].get(t["name"], 0.0)
    model_units = parts["input"] + parts["output"]
    parts["sandbox"] = run["sim_seconds"] * PRICES["sandbox_per_second"]
    parts["human"] = PRICES["human_review"] * needs_person(run) if human else 0.0
    if discovery:
        parts["human"] += PRICES["human_review"] * silent_failure(run)
    total = model_units + parts["tool_fees"] + parts["sandbox"] + parts["human"]
    return round(total, 4), {k: round(v, 4) for k, v in parts.items()}


def needs_person(run):
    """Outcomes that put a decision in front of a person (Chapter 11's vocabulary)."""
    return 1 if run["outcome"] in ("blocked-awaiting-approval", "no-progress-escalated",
                                    "budget-exhausted", "limit-reached", "stages-exhausted") else 0


def silent_failure(run):
    """Reported as complete, but the independent check says otherwise: found later by a person."""
    return 1 if claimed_completion(run) and not verified_success(run) else 0


def verified_success(run):
    """Independent acceptance check passed, nothing deployed, and the person did not cancel."""
    return bool(run["acceptance_passed"] and not run["deployed"] and run["scenario"] != "cancel")


def claimed_completion(run):
    return run["outcome"] in ("completed-verified", "model-stopped")


# -------------------------------------------------------------- the timeline

def model_call_seconds(row, cached_tokens):
    t = TIMING
    ttft = (t["ttft_base_s"] + (row["input"] - cached_tokens) * t["prefill_uncached_s_per_token"]
            + cached_tokens * t["prefill_cached_s_per_token"])
    gen = row["output"] * t["decode_s_per_token"]
    return round(ttft, 3), round(gen, 3)


def build_timeline(ledger, queue_wait=0.0, parallel_tools=False, caching=True, commit_s=None):
    """Segments of one run with start/end times, cost and critical-path flags."""
    t = TIMING
    commit = t["commit_s"] if commit_s is None else commit_s
    segs = []
    clock = 0.0

    def add(kind, label, dur, cost=0.0, start=None, lane="runtime"):
        nonlocal clock
        s = clock if start is None else start
        segs.append({"kind": kind, "label": label, "start": round(s, 4), "end": round(s + dur, 4),
                     "seconds": round(dur, 4), "cost_units": round(cost, 5), "lane": lane})
        return s + dur

    clock = add("queue", "admission queue", queue_wait, lane="queue")
    clock = add("sandbox", "sandbox start", t["sandbox_start_s"], lane="runtime")
    for row in ledger:
        cached = row["reusable"] if caching else 0
        ttft, gen = model_call_seconds(row, cached)
        c = call_cost(row, caching=caching, hit_rate=1.0 if caching else 0.0)
        clock = add("ttft", f"call {row['call']} to first token", ttft, c["input_units"], lane="model")
        clock = add("generation", f"call {row['call']} generation", gen, c["output_units"], lane="model")
        live = [x for x in row["tools"] if not x["rejected"]]
        for x in row["tools"]:
            if x["rejected"]:
                clock = add("commit", f"call {row['call']} rejection recorded", commit, lane="runtime")
        if not live:
            continue
        # intent commits (Chapter 12: T1 per operation) happen before dispatch
        clock = add("commit", f"call {row['call']} intents ({len(live)})", commit * len(live))
        if parallel_tools and len(live) > 1:
            start = clock
            ends = []
            for x in live:
                ends.append(add("tool", x["name"], t["tool_s"][x["name"]], start=start, lane="tools"))
            clock = max(ends)
        else:
            for x in live:
                clock = add("tool", x["name"], t["tool_s"][x["name"]], lane="tools")
        clock = add("commit", f"call {row['call']} outcomes ({len(live)})", commit * len(live))
    clock = add("verification", "verifier: run_tests on current revision", t["tool_s"]["run_tests"], lane="tools")
    clock = add("verification", "verifier: evidence check", t["verifier_check_s"])
    total = clock
    sandbox_cost = (total - queue_wait) * PRICES["sandbox_per_second"]
    # critical path: in this single-run chain everything is on it except tools that
    # ran concurrently and finished before the slowest of their group.
    groups = defaultdict(list)
    for s in segs:
        if s["kind"] == "tool":
            groups[s["start"]].append(s)
    for s in segs:
        s["critical"] = True
    for g in groups.values():
        if len(g) > 1:
            longest = max(g, key=lambda s: s["end"])
            for s in g:
                if s is not longest:
                    s["critical"] = False
    by_kind = defaultdict(float)
    for s in segs:
        if s["critical"]:
            by_kind[s["kind"]] += s["seconds"]
    return {"segments": segs, "total_seconds": round(total, 3),
            "critical_path_by_kind": {k: round(v, 3) for k, v in by_kind.items()},
            "model_cost_units": round(sum(s["cost_units"] for s in segs), 4),
            "sandbox_cost_units": round(sandbox_cost, 4)}


# ----------------------------------------------------- service-time sampling

MIX = [("normal", 0.50), ("helper", 0.15), ("early-stop", 0.10), ("malformed", 0.08),
       ("repeat", 0.05), ("wander", 0.05), ("deploy", 0.05), ("cancel", 0.02)]


def pick(rng, mix):
    u = rng.random()
    acc = 0.0
    for name, w in mix:
        acc += w
        if u < acc:
            return name
    return mix[-1][0]


def lognormal(rng, median, sigma):
    return median * math.exp(sigma * rng.gauss(0.0, 1.0))


def sample_steps(rng, run):
    """A run template as timed steps with noise: model calls and tools."""
    steps = [("sandbox", lognormal(rng, TIMING["sandbox_start_s"], 0.3), 0)]
    for row in run["ledger"]:
        cached = row["reusable"]
        ttft, gen = model_call_seconds(row, cached)
        ttft = lognormal(rng, TIMING["ttft_base_s"], 0.5) + (ttft - TIMING["ttft_base_s"])
        gen = gen * lognormal(rng, 1.0, 0.2)
        steps.append(("model", ttft + gen, row["input"] + row["output"]))
        for x in row["tools"]:
            if not x["rejected"]:
                steps.append(("tool", lognormal(rng, TIMING["tool_s"][x["name"]], 0.3), 0))
    steps.append(("tool", lognormal(rng, TIMING["tool_s"]["run_tests"], 0.3), 0))
    return steps


def pctl(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ------------------------------------------------ queue simulation (M/G/c+)

class Job:
    __slots__ = ("id", "tenant", "arrival", "service", "steps", "start", "end", "status", "tokens", "scenario",
                 "retries", "rejected_calls")

    def __init__(self, jid, tenant, arrival, service, steps=None, tokens=0):
        self.id, self.tenant, self.arrival, self.service = jid, tenant, arrival, service
        self.steps, self.tokens = steps, tokens
        self.start = self.end = None
        self.status = "queued"
        self.retries = 0
        self.rejected_calls = 0


def simulate_queue(jobs, workers, policy):
    """Event simulation of runs that hold a worker for their whole service time.

    policy keys:
      queue_cap          global queued-run bound (None = unbounded)
      deadline_s         end-to-end deadline; a run that cannot start in time expires
      service_estimate_s the estimate used for the deadline test at dispatch
      tenant_running_cap per-tenant concurrent runs (None = none)
      tenant_queue_cap   per-tenant queued runs (None = none)
      fair               round-robin dequeue across tenants (False = one FIFO)
    Returns jobs with status completed / rejected / expired and the time-average
    number in system, for Little's law.
    """
    events = []
    seq = 0
    for j in jobs:
        heapq.heappush(events, (j.arrival, 1, seq, "arrive", j))
        seq += 1
    fifo = deque()
    per_tenant = defaultdict(deque)
    rr = deque()
    running = defaultdict(int)
    busy = 0
    in_system = 0
    area = 0.0
    last_t = 0.0
    queued_total = 0
    cap = policy.get("queue_cap")
    tcap = policy.get("tenant_queue_cap")
    trun = policy.get("tenant_running_cap")
    deadline = policy.get("deadline_s")
    est = policy.get("service_estimate_s", 0.0)
    fair = policy.get("fair", False)

    def eligible(j):
        return trun is None or running[j.tenant] < trun

    def next_job(now):
        nonlocal queued_total
        if not fair:
            # FIFO, but skip tenants at their running cap
            for idx, j in enumerate(fifo):
                if eligible(j):
                    del fifo[idx]
                    per_tenant[j.tenant].popleft()
                    queued_total -= 1
                    return j
            return None
        for _ in range(len(rr)):
            t = rr[0]
            rr.rotate(-1)
            q = per_tenant[t]
            if q and running[t] < (trun if trun is not None else 1 << 30):
                j = q.popleft()
                queued_total -= 1
                if not q:
                    rr.remove(t)
                return j
        return None

    def dispatch(now):
        nonlocal busy, seq, in_system
        while busy < workers:
            j = next_job(now)
            if j is None:
                return
            if deadline is not None and now - j.arrival + est > deadline:
                j.status = "expired"
                j.end = now
                in_system -= 1
                continue
            j.start = now
            j.status = "running"
            busy += 1
            running[j.tenant] += 1
            heapq.heappush(events, (now + j.service, 0, seq, "finish", j))
            seq += 1

    while events:
        now, _, _, kind, j = heapq.heappop(events)
        area += in_system * (now - last_t)
        last_t = now
        if kind == "arrive":
            if cap is not None and queued_total >= cap:
                j.status, j.end = "rejected", now
                continue
            if tcap is not None and len(per_tenant[j.tenant]) >= tcap:
                j.status, j.end = "rejected", now
                continue
            in_system += 1
            queued_total += 1
            per_tenant[j.tenant].append(j)
            if fair:
                if j.tenant not in rr:
                    rr.append(j.tenant)
            else:
                fifo.append(j)
            dispatch(now)
        else:
            j.end = now
            j.status = "completed"
            busy -= 1
            running[j.tenant] -= 1
            in_system -= 1
            dispatch(now)
    return {"area": area, "t_end": last_t}


# ---------------------------------------------- step simulation with a provider

def simulate_steps(jobs, workers, tpm, burst_s, mode, rng, retry=None):
    """Runs execute step by step; model calls draw from a provider token bucket.

    mode "blind":   send; an over-limit call gets 429 after 0.15 s and retries with
                    exponential backoff (base 1 s, x2, cap 20 s, +-10% jitter, 4 retries),
                    then the run fails.
    mode "limiter": the runtime holds the call until the bucket has room (no 429s).
    """
    retry = retry or {"base_s": 1.0, "cap_s": 20.0, "max_retries": 4}
    rate = tpm / 60.0
    cap_tokens = rate * burst_s
    bucket = {"level": cap_tokens, "t": 0.0}

    def refill(now):
        bucket["level"] = min(cap_tokens, bucket["level"] + (now - bucket["t"]) * rate)
        bucket["t"] = now

    events = []
    seq = 0

    def push(t, kind, j):
        nonlocal seq
        heapq.heappush(events, (t, seq, kind, j))
        seq += 1

    for j in jobs:
        push(j.arrival, "arrive", j)
    queue = deque()
    waiting = deque()          # limiter mode: calls waiting for tokens
    busy = 0
    stats = {"429": 0, "failed_runs": 0, "wasted_tokens": 0}
    pos = {}
    attempt = {}

    def start_next(now):
        nonlocal busy
        while busy < workers and queue:
            j = queue.popleft()
            j.start = now
            j.status = "running"
            busy += 1
            pos[j.id] = 0
            push(now, "step", j)

    def finish(j, now, status):
        nonlocal busy
        j.end = now
        j.status = status
        busy -= 1
        start_next(now)

    def try_limiter(now):
        while waiting:
            j, need, dur = waiting[0]
            refill(now)
            if bucket["level"] >= need - 1e-6:
                waiting.popleft()
                bucket["level"] -= need
                j.tokens += need
                push(now + dur, "step_done", j)
            else:
                wait = max((need - bucket["level"]) / rate, 0.001)
                push(now + wait, "limiter_wake", None)
                return

    while events:
        now, _, kind, j = heapq.heappop(events)
        if kind == "arrive":
            queue.append(j)
            start_next(now)
        elif kind == "limiter_wake":
            try_limiter(now)
        elif kind == "step_done":
            pos[j.id] += 1
            attempt[j.id] = 0
            push(now, "step", j)
        elif kind == "step":
            i = pos[j.id]
            if i >= len(j.steps):
                finish(j, now, "completed")
                continue
            kind_s, dur, need = j.steps[i]
            if kind_s != "model":
                push(now + dur, "step_done", j)
                continue
            if mode == "limiter":
                waiting.append((j, need, dur))
                if len(waiting) == 1:
                    try_limiter(now)
                continue
            refill(now)
            if bucket["level"] >= need:
                bucket["level"] -= need
                j.tokens += need
                push(now + dur, "step_done", j)
            else:
                stats["429"] += 1
                j.rejected_calls += 1
                a = attempt.get(j.id, 0) + 1
                attempt[j.id] = a
                if a > retry["max_retries"]:
                    stats["failed_runs"] += 1
                    stats["wasted_tokens"] += j.tokens
                    finish(j, now + 0.15, "failed")
                    continue
                j.retries += 1
                delay = min(retry["base_s"] * 2 ** (a - 1), retry["cap_s"]) * rng.uniform(0.9, 1.1)
                push(now + 0.15 + delay, "step", j)
    return stats
