"""Chapter 22: scaling inference and scaling the runtime.

Two groups of experiments, all arithmetic or seeded simulation in simulated time:

  Inference
    1. memory budget: weights + KV cache + reserve, and concurrent sequences by context length
    2. allocation: contiguous reservation vs paged blocks vs paged with a shared prefix,
       admitting a seeded mix of agent-call lengths until memory is full
    3. prefill/decode contention: a long prompt arriving among decoding sequences,
       with and without chunked prefill, under a stated roofline cost model

  Runtime
    4. at-least-once queue: worker crashes, freezes and lease expiry under three write
       guards (lease only, owner check, fence), with stable and fresh worker ids
    5. admission: unbounded, globally capped and per-tenant capped queues under a noisy tenant
    6. sandbox starts: cold starts vs warm pools of several sizes, steady load and a burst
    7. storage: the Chapter 3 run's model inputs stored in full, content-addressed, compressed

    python3 ch22_scaling.py --output runs/ch22

Nothing here measures a GPU, a serving engine, a queue product or a sandbox provider.
Every assumption is a named parameter in itl/ch22_scaling.py.
"""
import argparse
import hashlib
import io
import json
import os
import random
import sys
import zlib
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from itl.core import ScriptedModel, Workspace, report_tools, run_agent, write_run  # noqa: E402
from itl.ch22_scaling import (GiB, ModelShape, Hardware, capacity_table, kv_budget,  # noqa: E402
                              agent_lengths, allocate_mix, prefill_decode, plan_fates, QueueSim,
                              admission, sandbox_pool)
import ch03_chapter_lab as ch03  # noqa: E402

# Illustrative shapes: grouped-query attention with 8 KV heads of dimension 128, as in Chapter 4.
M8 = ModelShape("M-8B bf16", 8.0e9, 32, 8, 128, 2.0)
M8Q = ModelShape("M-8B 4.5-bit", 8.0e9, 32, 8, 128, 0.5625)
M70 = ModelShape("M-70B bf16", 70.0e9, 80, 8, 128, 2.0)
M70Q = ModelShape("M-70B 4.5-bit", 70.0e9, 80, 8, 128, 0.5625)

WS24 = Hardware("1 x 24 GiB workstation card", 1, 24)
DC80 = Hardware("1 x 80 GiB data-center card", 1, 80)
DC160 = Hardware("2 x 80 GiB", 2, 80)
DC640 = Hardware("8 x 80 GiB node", 8, 80)

CONTEXTS = [4_096, 16_384, 32_768, 131_072]


def gib(x):
    return round(x / GiB, 2)


def inference(out):
    print("== 1. Memory budget and concurrent sequences at full context")
    print("  utilization 0.92, reserve 1.5 GiB per GPU; KV bytes per token = 2 x layers x kv_heads x head_dim x 2")
    rows = []
    for m, hw in [(M8, WS24), (M8Q, WS24), (M8, DC80), (M70, DC80), (M70Q, DC80), (M70, DC160),
                  (M70, DC640)]:
        r = capacity_table(m, hw, CONTEXTS)
        rows.append(r)
        seqs = " ".join(f"{c // 1024}k:{n}" for c, n in r["max_sequences_at_full_context"].items())
        print(f"  {m.name:<14} on {hw.name:<28} weights {gib(r['weights_bytes']):7.2f} GiB  "
              f"KV budget {gib(r['kv_bytes']):7.2f} GiB  KV/token {r['kv_per_token'] // 1024} KiB  "
              f"tokens {r['kv_tokens']:>9,}  seqs {seqs}")
    write_run(out, "capacity.json", rows)

    print("\n== 2. Allocation of a seeded mix of agent-call lengths (median 12k, cap 32k, +1k output)")
    rng = random.Random(22)
    lengths = agent_lengths(rng, 4_000)
    tenants = [f"tenant{rng.randrange(20)}" for _ in lengths]
    print(f"  mix: n={len(lengths)} mean={sum(lengths) // len(lengths):,} tokens "
          f"min={min(lengths):,} max={max(lengths):,}")
    alloc = []
    for m, hw in [(M8, DC80), (M70, DC640), (M70, DC160)]:
        budget = kv_budget(m, hw)["kv_bytes"] // m.kv_per_token
        res = {"model": m.name, "hardware": hw.name, "budget_tokens": int(budget), "results": []}
        for scheme, kw in [("contiguous", {}), ("paged", {}),
                           ("paged_prefix", {"shared_prefix": 6_144}),
                           ("paged_prefix", {"shared_prefix": 6_144, "tenants": tenants})]:
            r = allocate_mix(budget, lengths, scheme, **kw)
            r["label"] = scheme + (" (20 salted tenants)" if "tenants" in kw else
                                   " (one cache domain)" if scheme == "paged_prefix" else "")
            res["results"].append(r)
            print(f"  {m.name:<11} {hw.name:<28} {r['label']:<36} admitted {r['admitted']:>4}  "
                  f"allocated {r['allocated_tokens']:>9,}  waste {r['waste_pct']:>5}%  "
                  f"prefix copies {r['prefix_copies']}")
        alloc.append(res)
    write_run(out, "allocation.json", {"lengths_summary": {"n": len(lengths), "mean": sum(lengths) // len(lengths),
                                                            "min": min(lengths), "max": max(lengths)},
                                       "shared_prefix_tokens": 6_144, "block_tokens": 16, "configs": alloc})

    print("\n== 3. Prefill/decode contention: 24 decoders at 8k context, a 32k-token prompt arrives at 0.5 s")
    print("  cost model: step = max((weights + KV read) / 2.0 TB/s, 2 x params x tokens / 400 TFLOP/s)")
    pd = []
    for chunk in (None, 8_192, 2_048, 512):
        r = prefill_decode(M8, chunk=chunk)
        pd.append(r)
        label = "one step" if chunk is None else f"chunk {chunk}"
        print(f"  M-8B {label:<11} steps {r['steps']:>4}  decode gap p50 {r['decode_gap_p50_ms']:>6} ms  "
              f"p99 {r['decode_gap_p99_ms']:>7} ms  max {r['decode_gap_max_ms']:>7} ms  "
              f"long-prompt TTFT {r['long_prompt_ttft_ms']} ms")
    write_run(out, "prefill_decode.json", pd)


def runtime(out):
    print("\n== 4. At-least-once queue: 3,000 tasks, 24 worker slots, lease 30 s, heartbeat 10 s")
    print("  fates per attempt: 1% crash before effect, 1% crash after effect, 2% freeze mid-work,")
    print("  1% freeze between effect and write; freezes of 40-240 s; replacement after 45 s")
    fates = plan_fates(22, 3_000)
    rows = []
    for ids in ("fresh", "stable"):
        for policy in ("lease_only", "owner_check", "fenced"):
            db = os.path.join(out, f"queue_{policy}_{ids}.db")
            sim = QueueSim(policy, ids, fates, 3_000, db_path=db)
            r = sim.run()
            bad = [t for (t,) in sim.db.execute(
                "SELECT task_id FROM tasks WHERE status='done' AND result_fence < fence ORDER BY task_id")]
            r["superseded_examples"] = {t: sim.timeline(t) for t in bad[:1]}
            rows.append(r)
            print(f"  ids={ids:<6} {policy:<11} done {r['done']}  claims {r['claims']}  "
                  f"effects {r['effect_invocations']} (dup {r['duplicate_effects_without_key']})  "
                  f"abandoned {r['abandoned_after_lost_lease']}  late writes {r['late_writes_attempted']} "
                  f"accepted {r['late_writes_accepted']}  written twice {r['tasks_written_twice']}  "
                  f"final from superseded {r['final_result_from_superseded_attempt']}  "
                  f"refused {r['refused_writes']}")
    for r in rows:
        if r["ids"] == "stable" and r["policy"] in ("lease_only", "owner_check"):
            for t, tl in r["superseded_examples"].items():
                print(f"  timeline of {t} under {r['policy']} with stable ids (time, record, fence, worker, detail):")
                for ev in tl:
                    print(f"    {ev}")
    write_run(out, "queue_fencing.json", rows)

    print("\n== 5. Admission: 16 slots, 9 quiet tenants at 0.18/s, one tenant adds 0.30/s for 30 min; deadline 15 min")
    adm = []
    for policy in ("unbounded", "global_cap", "tenant_cap"):
        r = admission(policy)
        adm.append(r)
        for g in ("quiet", "noisy"):
            s = r[g]
            print(f"  {policy:<10} {g:<5} arrivals {r['arrivals'][g]:>5}  served {s['served']:>5}  "
                  f"rejected {s['rejected_at_submit']:>4}  missed deadline {s['missed_deadline']:>4}  "
                  f"wait p50 {s['wait_p50_s']:>7} s  p95 {s['wait_p95_s']:>7} s")
    write_run(out, "admission.json", adm)

    print("\n== 6. Sandbox starts: 0.25 runs/s, a burst of 40 at 30 min; cold median 6 s, warm claim 0.15 s")
    sb = []
    for pool in (0, 4, 8, 16, 48):
        r = sandbox_pool(pool)
        sb.append(r)
        print(f"  pool {pool:>2}  steady p50 {r['steady_p50_s']:>5} s  p95 {r['steady_p95_s']:>5} s  "
              f"p99 {r['steady_p99_s']:>5} s  burst p50 {r['burst_p50_s']:>5} s  max {r['burst_max_s']:>5} s  "
              f"idle warm sandbox-hours {r['warm_idle_sandbox_hours']}")
    write_run(out, "sandbox_pool.json", sb)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def storage(out):
    print("\n== 7. Storage: the Chapter 3 run's seven model inputs, with tool definitions")
    ws = Workspace()
    tools = report_tools(ws)
    definitions = tools.definitions()
    model = ScriptedModel(ch03.responses())
    messages = [{"role": "system", "content": ch03.SYSTEM}, {"role": "user", "content": ch03.TASK}]
    run_agent(model, tools, messages)
    ws.close()
    reqs = [{"messages": r["messages"], "tools": definitions} for r in model.requests]
    full = [len(canon(r).encode()) for r in reqs]
    per_call_zlib = [len(zlib.compress(canon(r).encode(), 6)) for r in reqs]
    seen, blobs, manifests, prev = set(), 0, [], 0
    for r in reqs:
        items = [{"tools": r["tools"]}] + r["messages"]
        new = []
        for m in items:
            c = canon(m).encode()
            h = hashlib.sha256(c).hexdigest()[:16]
            if h not in seen:
                seen.add(h)
                blobs += len(c)
            new.append(h)
        # a manifest names its parent manifest and appends only the hashes the parent lacks
        manifests.append(len(canon({"parent": prev, "append": new[prev:]}).encode()))
        prev = len(new)
    whole_zlib = len(zlib.compress(b"\n".join(canon(r).encode() for r in reqs), 6))
    res = {"calls": len(reqs), "full_bytes_per_call": full, "full_total": sum(full),
           "per_call_zlib6": per_call_zlib, "per_call_zlib6_total": sum(per_call_zlib),
           "whole_run_zlib6_total": whole_zlib,
           "content_addressed_blobs": blobs, "content_addressed_manifests": manifests,
           "content_addressed_total": blobs + sum(manifests)}
    print(f"  full input per call (bytes): {full}")
    print(f"  A. every input stored in full           {sum(full):>8,} bytes")
    print(f"  B. every input compressed separately    {sum(per_call_zlib):>8,} bytes")
    print(f"  C. content-addressed items + manifests  {blobs + sum(manifests):>8,} bytes "
          f"({blobs:,} item bytes + {sum(manifests):,} manifest bytes)")
    print(f"  D. whole run in one compressed file     {whole_zlib:>8,} bytes")
    slope = (full[-1] - full[0]) / (len(full) - 1)
    man = sum(manifests[1:]) / (len(manifests) - 1)
    proj = {}
    print(f"  projection (arithmetic): each call adds {slope:.0f} bytes of new items, manifests {man:.0f} bytes")
    for n in (7, 50, 200):
        a = sum(full[0] + slope * k for k in range(n))
        c = full[0] + slope * (n - 1) + man * n
        proj[n] = {"full_bytes": int(a), "content_addressed_bytes": int(c)}
        print(f"    n={n:>3} calls: A {int(a):>12,} bytes   C {int(c):>10,} bytes")
    res["projection"] = {"slope_bytes_per_call": round(slope, 1), "manifest_bytes_per_call": round(man, 1),
                         "by_calls": proj}
    write_run(out, "storage.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch22")
    out = ap.parse_args().output
    os.makedirs(out, exist_ok=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        inference(out)
        runtime(out)
        storage(out)
    write_run(out, "stdout.txt", buf.getvalue())


if __name__ == "__main__":
    main()
