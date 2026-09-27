"""Chapter 22 helpers: inference capacity arithmetic and runtime-scaling simulations.

Everything here is arithmetic from stated shapes or a seeded discrete-event
simulation in simulated time. Nothing measures a GPU, a serving engine, a
queue product or a sandbox provider. Each function states its inputs; change
them to match your own hardware and workload.

  ModelShape, Hardware, kv_budget, capacity_table     memory arithmetic
  allocate_mix                                         contiguous vs paged vs shared prefix
  prefill_decode                                       step scheduler with and without chunked prefill
  QueueSim                                             at-least-once queue, leases, fences, crashes, pauses
  admission                                            unbounded vs bounded admission under a noisy tenant
  sandbox_pool                                         cold starts vs warm pools
"""
import heapq
import math
import os
import random
import sqlite3
from dataclasses import dataclass

GiB = 1024 ** 3


# ---------------------------------------------------------------------------
# 1. Inference memory arithmetic
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelShape:
    name: str
    params: float          # parameter count
    layers: int
    kv_heads: int
    head_dim: int
    weight_bytes: float    # bytes per parameter as stored (2.0 for bf16, 0.5625 for 4.5-bit)
    kv_bytes: int = 2      # bytes per stored K or V element

    @property
    def weights(self):
        return self.params * self.weight_bytes

    @property
    def kv_per_token(self):
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_bytes


@dataclass(frozen=True)
class Hardware:
    name: str
    gpus: int
    gib_per_gpu: float


def kv_budget(model, hw, utilization=0.92, reserve_gib_per_gpu=1.5):
    """Bytes left for the KV cache after weights and a per-GPU reserve.

    utilization: the fraction of device memory the engine may use (vLLM's default is 0.92).
    reserve: activations, CUDA graphs, communication buffers and fragmentation, as a flat
    per-GPU allowance. Both are assumptions; profile your own engine to replace them.
    """
    usable = hw.gpus * hw.gib_per_gpu * GiB * utilization
    reserve = hw.gpus * reserve_gib_per_gpu * GiB
    return {
        "device_bytes": int(hw.gpus * hw.gib_per_gpu * GiB),
        "usable_bytes": int(usable),
        "weights_bytes": int(model.weights),
        "reserve_bytes": int(reserve),
        "kv_bytes": int(usable - model.weights - reserve),
    }


def capacity_table(model, hw, contexts, **kw):
    b = kv_budget(model, hw, **kw)
    tokens = max(0, b["kv_bytes"]) // model.kv_per_token
    row = {"model": model.name, "hardware": hw.name, **b,
           "kv_per_token": model.kv_per_token, "kv_tokens": int(tokens),
           "fits_by_weights": model.weights < hw.gpus * hw.gib_per_gpu * GiB,
           "max_sequences_at_full_context": {}}
    for c in contexts:
        row["max_sequences_at_full_context"][c] = int(tokens // c)
    return row


# ---------------------------------------------------------------------------
# 2. Allocation waste and prefix sharing, from a seeded length distribution
# ---------------------------------------------------------------------------
def agent_lengths(rng, n, median=12_000, sigma=0.8, lo=1_000, hi=32_768, output_reserve=1_024):
    """Context lengths (prompt + reserved output) of simultaneously active agent calls."""
    out = []
    for _ in range(n):
        x = int(math.exp(math.log(median) + sigma * rng.gauss(0, 1)))
        out.append(min(hi, max(lo, x)) + output_reserve)
    return out


def allocate_mix(budget_tokens, lengths, scheme, max_ctx=32_768 + 1_024, block=16,
                 shared_prefix=0, tenants=None):
    """Admit sequences in arrival order until the next one does not fit.

    scheme:
      contiguous   reserve max_ctx tokens per sequence up front
      paged        allocate ceil(len / block) blocks; waste is the last partial block
      paged_prefix paged, and the first shared_prefix tokens (a multiple of block) are
                   stored once per cache domain; tenants[i] names sequence i's domain
                   (salted caches share only within a domain)
    """
    used = allocated = admitted = 0
    domains = set()
    for i, n in enumerate(lengths):
        if scheme == "contiguous":
            need = max_ctx
            new_domain = None
        elif scheme == "paged":
            need = -(-n // block) * block
            new_domain = None
        elif scheme == "paged_prefix":
            dom = tenants[i] if tenants else "all"
            private = n - shared_prefix
            need = -(-private // block) * block
            new_domain = None if dom in domains else dom
            if new_domain is not None:
                need += shared_prefix
        else:
            raise ValueError(scheme)
        if allocated + need > budget_tokens:
            break
        allocated += need
        admitted += 1
        used += n if scheme != "paged_prefix" else (n - shared_prefix) + (shared_prefix if new_domain else 0)
        if new_domain is not None:
            domains.add(new_domain)
    logical = sum(lengths[:admitted])
    return {"scheme": scheme, "admitted": admitted, "allocated_tokens": allocated,
            "logical_tokens": logical,
            "waste_pct": round(100 * (allocated - used) / allocated, 1) if allocated else 0.0,
            "prefix_copies": len(domains)}


# ---------------------------------------------------------------------------
# 3. Prefill/decode contention: a step scheduler with a stated cost model
# ---------------------------------------------------------------------------
def step_ms(model, prefill_tokens, decode_seqs, decode_ctx_tokens, bandwidth=2.0e12, flops=4.0e14):
    """Time of one engine step under a simple roofline cost model (an assumption, not a measurement).

    Every step reads the weights once and the KV cache of every decoding sequence; prefill
    tokens and decode tokens each cost 2 * params floating-point operations. The step takes
    the larger of the memory time and the compute time.
    """
    mem = (model.weights + decode_ctx_tokens * model.kv_per_token) / bandwidth
    comp = 2 * model.params * (prefill_tokens + decode_seqs) / flops
    return 1000 * max(mem, comp)


def prefill_decode(model, decoders=24, decode_ctx=8_192, new_prompt=32_768, arrive_ms=500.0,
                   horizon_ms=6_000.0, chunk=None):
    """Decoders emit one token per step. A long prompt arrives; with chunk=None it is prefilled in
    one step, otherwise at most `chunk` prompt tokens are scheduled per step (chunked prefill)."""
    t = 0.0
    last_token_at = [0.0] * decoders
    gaps = []
    remaining = None
    first_token_ms = None
    ctx = [decode_ctx] * decoders
    steps = 0
    while t < horizon_ms:
        if remaining is None and t >= arrive_ms:
            remaining = new_prompt
        pf = 0
        if remaining:
            pf = remaining if chunk is None else min(chunk, remaining)
        dt = step_ms(model, pf, decoders, sum(ctx))
        t += dt
        steps += 1
        for i in range(decoders):
            gaps.append(t - last_token_at[i])
            last_token_at[i] = t
            ctx[i] += 1
        if remaining:
            remaining -= pf
            if remaining == 0:
                first_token_ms = t - arrive_ms
                remaining = 0
    gaps.sort()
    return {"chunk": chunk, "steps": steps,
            "decode_gap_p50_ms": round(gaps[len(gaps) // 2], 1),
            "decode_gap_p99_ms": round(gaps[int(len(gaps) * 0.99)], 1),
            "decode_gap_max_ms": round(gaps[-1], 1),
            "long_prompt_ttft_ms": round(first_token_ms, 1) if first_token_ms is not None else None}


# ---------------------------------------------------------------------------
# 4. Queue with at-least-once delivery, leases and fences
# ---------------------------------------------------------------------------
QUEUE_SCHEMA = """
CREATE TABLE tasks(
  task_id TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL,
  owner TEXT, lease_expires REAL, fence INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
  result_fence INTEGER, result_owner TEXT, writes INTEGER NOT NULL DEFAULT 0);
CREATE TABLE effects(seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
  fence INTEGER NOT NULL, worker TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE refusals(seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
  fence INTEGER NOT NULL, worker TEXT NOT NULL, at REAL NOT NULL, reason TEXT NOT NULL);
CREATE TABLE claims(seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
  fence INTEGER NOT NULL, worker TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE writes(seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
  fence INTEGER NOT NULL, worker TEXT NOT NULL, at REAL NOT NULL, superseded INTEGER NOT NULL);
"""

COMPLETE_SQL = {
    # The completion write each policy uses. The guard lives in the WHERE clause, so the check
    # and the write are one statement inside one transaction.
    "lease_only": ("UPDATE tasks SET status='done', result_fence=?, result_owner=?, writes=writes+1 "
                   "WHERE task_id=?"),
    "owner_check": ("UPDATE tasks SET status='done', result_fence=?, result_owner=?, writes=writes+1, "
                    "owner=NULL WHERE task_id=? AND status='leased' AND owner=?"),
    "fenced": ("UPDATE tasks SET status='done', result_fence=?, result_owner=?, writes=writes+1, "
               "owner=NULL WHERE task_id=? AND status='leased' AND fence=?"),
}


def plan_fates(seed, tasks, max_attempts=8, p_crash=0.02, p_pause=0.02, p_pause_write=0.01):
    """Pre-draw every attempt's fate so each policy replays the same world.

    fate per (task, attempt): work seconds, and one of
      ok                    works, performs the effect, writes the result
      crash_before_effect   process dies mid-work
      crash_after_effect    effect performed, process dies before the write (lost receipt)
      pause                 process freezes mid-work for pause_s, then resumes, checks its
                            lease with a heartbeat, and finishes only if the heartbeat is accepted
      pause_before_write    effect performed, then the process freezes for pause_s and
                            resumes straight into its result write (no check in between)
    """
    rng = random.Random(seed)
    fates = {}
    for t in range(tasks):
        for a in range(1, max_attempts + 1):
            work = round(math.exp(math.log(45) + 0.5 * rng.gauss(0, 1)), 2)
            r = rng.random()
            if r < p_crash / 2:
                kind = "crash_before_effect"
            elif r < p_crash:
                kind = "crash_after_effect"
            elif r < p_crash + p_pause:
                kind = "pause"
            elif r < p_crash + p_pause + p_pause_write:
                kind = "pause_before_write"
            else:
                kind = "ok"
            fates[(t, a)] = {"work": work, "kind": kind,
                             "pause_s": round(rng.uniform(40, 240), 1),
                             "at_frac": round(rng.uniform(0.1, 0.9), 3)}
    return fates


HEARTBEAT_SQL = {
    # lease_only renews whatever lease the row holds; the other two renew only their own.
    "lease_only": "UPDATE tasks SET lease_expires=? WHERE task_id=? AND status='leased'",
    "owner_check": "UPDATE tasks SET lease_expires=? WHERE task_id=? AND status='leased' AND owner=?",
    "fenced": "UPDATE tasks SET lease_expires=? WHERE task_id=? AND status='leased' AND fence=?",
}


class QueueSim:
    """Worker slots claim tasks from a SQLite queue in simulated time.

    Lease 30 s, heartbeat every 10 s. A frozen process stops heartbeating. When a freeze
    outlasts `liveness_s`, the supervisor starts a replacement process in the same slot:
    with ids="stable" the replacement reuses the slot's worker id (a configured id, or a
    pod name that survives restarts); with ids="fresh" every process gets a new id. The
    frozen process later resumes, finishes its attempt as its fate says, and exits.
    Delivery is at-least-once: an expired lease makes the task claimable again.
    """

    def __init__(self, policy, ids, fates, tasks, slots=24, lease=30.0, heartbeat=10.0,
                 liveness_s=45.0, restart_s=20.0, db_path=":memory:"):
        self.policy, self.ids, self.fates = policy, ids, fates
        self.n_tasks, self.slots = tasks, slots
        self.lease, self.hb, self.liveness, self.restart = lease, heartbeat, liveness_s, restart_s
        self.db_path = db_path                         # the run happens in memory; saved at the end
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.executescript(QUEUE_SCHEMA)
        self.db.executemany("INSERT INTO tasks(task_id, created, status) VALUES(?,?, 'queued')",
                            [(f"t{t:04d}", t * 0.5) for t in range(tasks)])
        self.events, self.seq = [], 0
        self.gen = {s: 0 for s in range(slots)}
        self.jobs = {}                                # (task_id, fence) -> job dict
        self.stats = {"claims": 0, "late_writes": 0, "late_writes_accepted": 0, "refused": 0,
                      "effects": 0, "abandoned_after_lost_lease": 0}

    def push(self, t, kind, key=None, **p):
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, key, p))

    def worker_id(self, slot):
        return f"w{slot:02d}" if self.ids == "stable" else f"w{slot:02d}-{self.gen[slot]}"

    # -- queue operations: the guard is in the WHERE clause of one statement ----
    def claim(self, now, worker):
        self.db.execute("BEGIN IMMEDIATE")
        row = self.db.execute(
            "SELECT task_id FROM tasks WHERE created <= ? AND (status='queued' OR "
            "(status='leased' AND lease_expires <= ?)) ORDER BY created LIMIT 1", (now, now)).fetchone()
        if row is None:
            self.db.execute("COMMIT")
            return None
        self.db.execute("UPDATE tasks SET status='leased', owner=?, lease_expires=?, fence=fence+1, "
                        "attempts=attempts+1 WHERE task_id=?", (worker, now + self.lease, row[0]))
        fence, attempt = self.db.execute("SELECT fence, attempts FROM tasks WHERE task_id=?",
                                         (row[0],)).fetchone()
        self.db.execute("INSERT INTO claims(task_id, fence, worker, at) VALUES(?,?,?,?)",
                        (row[0], fence, worker, now))
        self.db.execute("COMMIT")
        self.stats["claims"] += 1
        return row[0], fence, attempt

    def heartbeat(self, now, job):
        sql = HEARTBEAT_SQL[self.policy]
        args = [now + self.lease, job["task_id"]]
        if self.policy == "owner_check":
            args.append(job["worker"])
        elif self.policy == "fenced":
            args.append(job["fence"])
        return self.db.execute(sql, args).rowcount == 1

    def complete(self, now, job):
        task_id, worker, fence = job["task_id"], job["worker"], job["fence"]
        self.db.execute("BEGIN IMMEDIATE")
        cur_fence = self.db.execute("SELECT fence FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
        superseded = fence < cur_fence
        if self.policy == "lease_only":
            cur = self.db.execute(COMPLETE_SQL["lease_only"], (fence, worker, task_id))
        elif self.policy == "owner_check":
            cur = self.db.execute(COMPLETE_SQL["owner_check"], (fence, worker, task_id, worker))
        else:
            cur = self.db.execute(COMPLETE_SQL["fenced"], (fence, worker, task_id, fence))
        ok = cur.rowcount == 1
        if ok:
            self.db.execute("INSERT INTO writes(task_id, fence, worker, at, superseded) VALUES(?,?,?,?,?)",
                            (task_id, fence, worker, now, int(superseded)))
        else:
            st = self.db.execute("SELECT status, owner, fence FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            self.db.execute("INSERT INTO refusals(task_id, fence, worker, at, reason) VALUES(?,?,?,?,?)",
                            (task_id, fence, worker, now, f"row status={st[0]} owner={st[1]} fence={st[2]}"))
            self.stats["refused"] += 1
        self.db.execute("COMMIT")
        if superseded:
            self.stats["late_writes"] += 1
            self.stats["late_writes_accepted"] += int(ok)
        return ok

    def effect(self, now, job):
        self.db.execute("INSERT INTO effects(task_id, fence, worker, at) VALUES(?,?,?,?)",
                        (job["task_id"], job["fence"], job["worker"], now))
        self.stats["effects"] += 1

    # -- simulation -----------------------------------------------------------------
    def run(self, horizon=40_000.0):
        for s in range(self.slots):
            self.push(0.0, "idle", slot=s, gen=0)
        while self.events:
            now, _, kind, key, p = heapq.heappop(self.events)
            if now > horizon:
                break
            if key is None:
                getattr(self, "on_" + kind)(now, **p)
            else:
                getattr(self, "on_" + kind)(now, self.jobs[key])
        if self.db_path != ":memory:":
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            disk = sqlite3.connect(self.db_path)
            self.db.backup(disk)
            disk.close()
        return self.summary()

    def timeline(self, task_id):
        """Every claim, effect, accepted write and refused write for one task, in time order."""
        rows = []
        for table, cols in (("claims", "fence, worker, at, ''"), ("effects", "fence, worker, at, ''"),
                            ("writes", "fence, worker, at, superseded"), ("refusals", "fence, worker, at, reason")):
            for f, w, at, extra in self.db.execute(f"SELECT {cols} FROM {table} WHERE task_id=?", (task_id,)):
                rows.append((round(at, 1), table[:-1] if table != "refusals" else "refusal", f, w, extra))
        return sorted(rows)

    def next_idle(self, now, job):
        """After an attempt ends, the process takes new work only if it is still the slot's process."""
        if job["gen"] == self.gen[job["slot"]]:
            self.push(now, "idle", slot=job["slot"], gen=job["gen"])

    def on_idle(self, now, slot, gen):
        if gen != self.gen[slot]:
            return
        worker = self.worker_id(slot)
        got = self.claim(now, worker)
        if got is None:
            if self.db.execute("SELECT COUNT(*) FROM tasks WHERE status!='done'").fetchone()[0]:
                self.push(now + 1.0, "idle", slot=slot, gen=gen)
            return
        task_id, fence, attempt = got
        fate = self.fates[(int(task_id[1:]), min(attempt, 8))]
        job = dict(slot=slot, gen=gen, worker=worker, task_id=task_id, fence=fence, running=True, **fate)
        key = (task_id, fence)
        self.jobs[key] = job
        self.push(now + self.hb, "beat", key)
        k = fate["kind"]
        if k == "ok":
            self.push(now + fate["work"], "finish", key)
        elif k.startswith("crash"):
            self.push(now + fate["work"] * fate["at_frac"], "crash", key)
        elif k == "pause":
            self.push(now + fate["work"] * fate["at_frac"], "freeze", key)
        else:                                          # pause_before_write
            self.push(now + fate["work"], "freeze", key)

    def on_beat(self, now, job):
        if not job["running"]:
            return
        if self.heartbeat(now, job):
            self.push(now + self.hb, "beat", (job["task_id"], job["fence"]))
        # A refused heartbeat is noticed at the worker's next check (resume or finish).

    def on_finish(self, now, job):
        job["running"] = False
        if job["kind"] == "pause" and not self.heartbeat(now, job):
            self.stats["abandoned_after_lost_lease"] += 1   # worker-side check: abandon, no effect
            self.next_idle(now, job)
            return
        if job["kind"] != "pause_before_write":
            self.effect(now, job)
        self.complete(now, job)
        self.next_idle(now, job)

    def on_crash(self, now, job):
        job["running"] = False
        if job["kind"] == "crash_after_effect":
            self.effect(now, job)
        self.gen[job["slot"]] += 1
        self.push(now + self.restart, "idle", slot=job["slot"], gen=self.gen[job["slot"]])

    def on_freeze(self, now, job):
        job["running"] = False                          # a frozen process sends nothing
        if job["kind"] == "pause_before_write":
            self.effect(now, job)                       # the effect happened just before the freeze
        if job["pause_s"] > self.liveness:
            self.gen[job["slot"]] += 1                  # supervisor starts a replacement process
            self.push(now + self.liveness + self.restart, "idle", slot=job["slot"], gen=self.gen[job["slot"]])
        remaining = 0.0 if job["kind"] == "pause_before_write" else job["work"] * (1 - job["at_frac"])
        self.push(now + job["pause_s"] + remaining, "finish", (job["task_id"], job["fence"]))

    def summary(self):
        q = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        tasks_with_effect = q("SELECT COUNT(DISTINCT task_id) FROM effects")
        return {
            "policy": self.policy, "ids": self.ids, "tasks": self.n_tasks,
            "done": q("SELECT COUNT(*) FROM tasks WHERE status='done'"),
            "claims": self.stats["claims"], "redeliveries": self.stats["claims"] - self.n_tasks,
            "effect_invocations": self.stats["effects"],
            "duplicate_effects_without_key": self.stats["effects"] - tasks_with_effect,
            "abandoned_after_lost_lease": self.stats["abandoned_after_lost_lease"],
            "late_writes_attempted": self.stats["late_writes"],
            "late_writes_accepted": self.stats["late_writes_accepted"],
            "tasks_written_twice": q("SELECT COUNT(*) FROM tasks WHERE writes > 1"),
            "final_result_from_superseded_attempt": q(
                "SELECT COUNT(*) FROM tasks WHERE status='done' AND result_fence < fence"),
            "refused_writes": self.stats["refused"],
        }


# ---------------------------------------------------------------------------
# 5. Admission: unbounded vs bounded, with a noisy tenant
# ---------------------------------------------------------------------------
def admission(policy, seed=22, slots=16, horizon=7_200.0, deadline=900.0, global_cap=160, tenant_cap=24):
    rng = random.Random(seed)
    arrivals = []
    t = 0.0
    while t < horizon:                               # nine quiet tenants, 0.18 tasks/s in total
        t += rng.expovariate(0.18)
        arrivals.append((t, f"q{rng.randrange(9)}"))
    t = 1_800.0
    while t < 3_600.0:                               # one tenant adds 0.30 tasks/s for 30 minutes
        t += rng.expovariate(0.30)
        arrivals.append((t, "noisy"))
    arrivals.sort()
    service = [math.exp(math.log(60) + 0.4 * rng.gauss(0, 1)) for _ in arrivals]

    free_at = [0.0] * slots
    queue = []                                       # (arrival, idx)
    waits = {"quiet": [], "noisy": []}
    missed = {"quiet": 0, "noisy": 0}
    rejected = {"quiet": 0, "noisy": 0}
    served = {"quiet": 0, "noisy": 0}
    per_tenant = {}
    events = []                                      # (time, kind, idx)
    for i, (at, tenant) in enumerate(arrivals):
        heapq.heappush(events, (at, 1, i))
    running = []                                     # heap of (end, tenant)
    idle = slots

    def group(tn):
        return "noisy" if tn == "noisy" else "quiet"

    def dispatch(now):
        nonlocal idle
        while idle and queue:
            at, i = queue.pop(0)
            tn = arrivals[i][1]
            if policy != "unbounded" and now - at > deadline:
                missed[group(tn)] += 1               # expired while waiting: refuse instead of running late
                per_tenant[tn] -= 1
                continue
            w = now - at
            waits[group(tn)].append(w)
            if w > deadline:
                missed[group(tn)] += 1
            served[group(tn)] += 1
            idle -= 1
            heapq.heappush(events, (now + service[i], 0, i))

    while events:
        now, kind, i = heapq.heappop(events)
        tn = arrivals[i][1]
        if kind == 0:                                # completion
            idle += 1
            per_tenant[tn] -= 1
        else:                                        # arrival
            n_q = len(queue)
            if policy == "global_cap" and n_q >= global_cap:
                rejected[group(tn)] += 1
            elif policy == "tenant_cap" and per_tenant.get(tn, 0) >= tenant_cap:
                rejected[group(tn)] += 1
            else:
                per_tenant[tn] = per_tenant.get(tn, 0) + 1
                queue.append((now, i))
        dispatch(now)

    def pct(xs, p):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(len(xs) * p))], 1) if xs else None

    out = {"policy": policy, "arrivals": {"quiet": sum(1 for a in arrivals if a[1] != "noisy"),
                                          "noisy": sum(1 for a in arrivals if a[1] == "noisy")}}
    for g in ("quiet", "noisy"):
        out[g] = {"served": served[g], "rejected_at_submit": rejected[g], "missed_deadline": missed[g],
                  "wait_p50_s": pct(waits[g], 0.5), "wait_p95_s": pct(waits[g], 0.95),
                  "wait_max_s": pct(waits[g], 1.0)}
    return out


# ---------------------------------------------------------------------------
# 6. Sandbox pool: cold starts vs warm pools
# ---------------------------------------------------------------------------
def sandbox_pool(pool, seed=22, rate=0.25, horizon=3_600.0, burst_at=1_800.0, burst=40,
                 cold_median=6.0, warm_claim=0.15, hold_median=120.0):
    """Runs ask for a sandbox; a warm one is claimed in `warm_claim` s, otherwise a cold start is
    paid. Each claim of a warm sandbox triggers a background cold start to refill the pool.
    Timings are stated assumptions for illustration, not measurements of any provider."""
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    while t < horizon:
        t += rng.expovariate(rate)
        reqs.append(t)
    reqs += [burst_at + 0.05 * k for k in range(burst)]
    reqs.sort()
    ready = pool                                      # warm sandboxes available now
    refills = []                                      # times at which refills become ready
    idle_seconds = 0.0
    last = 0.0
    lat, lat_burst = [], []
    for at in reqs:
        while refills and refills[0] <= at:
            r = heapq.heappop(refills)
            idle_seconds += ready * (r - last)
            last = r
            ready += 1
        idle_seconds += ready * (at - last)
        last = at
        if ready > 0:
            ready -= 1
            latency = warm_claim
            heapq.heappush(refills, at + math.exp(math.log(cold_median) + 0.5 * rng.gauss(0, 1)))
        else:
            latency = math.exp(math.log(cold_median) + 0.5 * rng.gauss(0, 1))
        (lat_burst if burst_at <= at < burst_at + 2 else lat).append(latency)

    def pct(xs, p):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2)

    return {"pool": pool, "requests": len(reqs),
            "steady_p50_s": pct(lat, 0.5), "steady_p95_s": pct(lat, 0.95), "steady_p99_s": pct(lat, 0.99),
            "burst_p50_s": pct(lat_burst, 0.5), "burst_max_s": pct(lat_burst, 1.0),
            "warm_idle_sandbox_hours": round(idle_seconds / 3600, 2)}
