"""Chapter 25 helpers: an evidence ledger over Part II run outputs, and a
small, simulated test of executor-level learning with policy pinning.

Everything here is standard library and deterministic. Latencies are drawn
from seeded distributions and time is simulated; nothing is measured on a
real network or a real model endpoint.
"""
import hashlib
import json
import math
import os
import random
import re

# ---------------------------------------------------------------------------
# Part A: the evidence ledger
# ---------------------------------------------------------------------------

# Each entry names the open problem it supports, the chapter that demonstrated
# the limitation, the run file that holds the evidence and a pattern whose
# first match is copied into the ledger verbatim.
LEDGER_ITEMS = [
    ("verification", 15, "ch15/stdout.txt", r"flawed team votes: .*"),
    ("verification", 15, "ch15/stdout.txt", r"coverage check on the same results: .*"),
    ("verification", 16, "ch16/stdout.txt", r"answer check: \{'answer_claims_export': True, 'exports_delivered': 0.*"),
    ("verification", 12, "ch12/stdout.txt", r"from stream: .*"),
    ("long-horizon", 11, "ch11/loop/wander/outcome.json", r"\"tokens\": \d+"),
    ("long-horizon", 11, "ch11/summary.txt", r"wander .*"),
    ("long-horizon", 11, "ch11/limits.json", r"\"model_calls\": \d+"),
    ("long-horizon", 11, "ch11/hybrid/wander/outcome.json", r"\"tokens\": \d+"),
    ("context-memory", 6, "ch06/truncate/result.json", r"\"result\": \"VIOLATED\""),
    ("context-memory", 6, "ch06/structured/result.json", r"\"path_taken\": \"compliant\""),
    ("context-memory", 8, "ch08/stdout.txt", r"T4 stale: .*"),
    ("context-memory", 7, "ch07/stdout.txt", r"withdrawn doc chunks=.*"),
    ("recovery-protocols", 13, "ch13/stdout.txt", r"fresh key per retry: .*"),
    ("recovery-protocols", 13, "ch13/stdout.txt", r"reconciler +decision=escalate.*"),
    ("injection", 16, "ch16/stdout.txt", r"outbound: messages=1 .*"),
    ("injection", 16, "ch16/stdout.txt", r"governed passes \d+/\d+"),
    ("injection", 16, "ch16/stdout.txt", r"classifier signal: .*'score': 1.*"),
    ("authority", 14, "ch14/stdout.txt", r"model re-proposes P2 for r6; dispatch check: .*"),
    ("human-role", 14, "ch14/stdout.txt", r"ask_every_write/approves_all .*"),
    ("human-role", 14, "ch14/stdout.txt", r"enforce_then_ask/approves_all .*"),
    ("human-role", 14, "ch14/stdout.txt", r"vague: .*"),
    ("evaluation", 9, "ch09/stdout.txt", r"revised  held_out .*"),
    ("evaluation", 12, "ch12/stdout.txt", r"model inputs rebuilt: .*"),
    ("cost", 15, "ch15/stdout.txt", r"naive     per worker .*"),
    ("cost", 6, "ch06/summary.txt", r"structured_task_last +requests .*"),
    ("cost", 6, "ch06/summary.txt", r"structured_task_first_status +requests .*"),
]


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def build_ledger(runs_root):
    rows = []
    for problem, chapter, rel, pattern in LEDGER_ITEMS:
        path = os.path.join(runs_root, rel)
        row = {"problem": problem, "chapter": chapter, "file": "runs/" + rel}
        if not os.path.exists(path):
            row.update(found=False, reason="file missing; run that chapter's companion first")
            rows.append(row)
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        for i, line in enumerate(lines, 1):
            m = re.search(pattern, line)
            if m:
                row.update(found=True, line=i, text=m.group(0).strip(), file_sha=sha12(path))
                break
        else:
            row.update(found=False, reason="pattern not found")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Part B: compounding arithmetic (closed form, independence assumed)
# ---------------------------------------------------------------------------

def compounding_table(step_success=(0.99, 0.995, 0.999), steps=(10, 50, 100, 300)):
    return [{"per_step": s, "steps": n, "run_success": round(s ** n, 4)}
            for s in step_success for n in steps]


def pass_hat_k(p, ks=(1, 2, 4, 8)):
    """Probability that k independent trials all succeed at per-trial rate p."""
    return [{"p": p, "k": k, "all_succeed": round(p ** k, 4)} for k in ks]


# ---------------------------------------------------------------------------
# Part C: executor-level learning of a timeout bound, with policy pinning
# ---------------------------------------------------------------------------

LADDER_S = [1, 2, 5, 10, 20, 30, 60, 120, 300]   # closed action set; 300 = the floor's stance
MIN_RUNG_S = 1
LEASE_S = 300.0          # floor: a hang is discovered only when the lease expires
TASK_ATTEMPTS = 3        # the task's declared attempt budget
BACKOFF_S = [1.0, 2.0]   # fixed backoff before attempts 2 and 3 (jitter omitted for exact replay)
HANG = None              # an attempt that never returns
ATTEMPT_NOISE = 0.15     # lognormal sigma of attempt-to-attempt variation for one request

CALLEES = {
    # name: (median seconds, lognormal sigma, hang probability)
    "export_status": (0.4, 0.6, 0.03),
    "model_endpoint": (8.0, 0.9, 0.01),
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def policy_version(policy):
    if policy.get("name") == "static-v0":
        return "static-v0"
    body = {k: v for k, v in policy.items() if k != "name"}
    return "p-" + hashlib.sha256(canonical(body).encode()).hexdigest()[:12]


def static_v0():
    return {"name": "static-v0", "timeout_s": {}, "max_attempts": 3}


def validate(policy):
    """Envelope check at the promotion gate. Returns a list of problems."""
    problems = []
    for callee, t in policy.get("timeout_s", {}).items():
        if t not in LADDER_S:
            problems.append(f"{callee}: {t} s is not a ladder rung")
        elif t < MIN_RUNG_S:
            problems.append(f"{callee}: {t} s is below the minimum rung")
    if policy.get("max_attempts", 3) < 1:
        problems.append("max_attempts below 1")
    return problems


def resolve(policy, callee):
    """Decision-point read path. Invalid policies steer nothing (fail closed to the floor).
    A learned attempt budget may only narrow the task's declared budget."""
    if validate(policy):
        policy = static_v0()
    timeout = policy["timeout_s"].get(callee, LEASE_S)
    attempts = min(policy.get("max_attempts", 3), TASK_ATTEMPTS)
    return float(timeout), attempts


def draw_tape(seed, n_ops, shift=None):
    """A world tape: for every operation, the latency each of three attempts would see.
    Floor and candidate are evaluated against the same tape (the twin's rule)."""
    rng = random.Random(seed)
    ops = []
    for i in range(n_ops):
        callee = "export_status" if i % 2 == 0 else "model_endpoint"
        median, sigma, p_hang = CALLEES[callee]
        if shift and callee in shift:
            median = median * shift[callee]
        # A request's size is shared by all its attempts: a slow request stays slow when
        # retried. Each attempt adds a little independent noise and may hang on its own.
        base = math.exp(rng.gauss(math.log(median), sigma))
        attempts = []
        for _ in range(TASK_ATTEMPTS):
            if rng.random() < p_hang:
                attempts.append(HANG)
            else:
                attempts.append(round(base * math.exp(rng.gauss(0.0, ATTEMPT_NOISE)), 3))
        ops.append({"op": i, "callee": callee, "attempts": attempts})
    return ops


def run_tape(tape, policy, run_id, journal_decisions=None):
    """Execute a tape under one policy. Returns per-op outcomes and a decision journal.
    If journal_decisions is given, decisions are served from it (exact replay) instead of
    being computed from the policy."""
    version = policy_version(policy)
    served = {(d["op"], d["attempt"]): d for d in journal_decisions} if journal_decisions else None
    journal, outcomes = [], []
    for op in tape:
        wall, done = 0.0, False
        used = 0
        for k, latency in enumerate(op["attempts"], 1):
            if served is not None:
                d = served.get((op["op"], k))
                if d is None:
                    break
                timeout, allowed = d["timeout_s"], d["attempts_allowed"]
                ver = d["policy_version"]
            else:
                timeout, allowed = resolve(policy, op["callee"])
                ver = version
                if k > allowed:
                    break
            if k > 1:
                wall += BACKOFF_S[k - 2]
            used = k
            ok = latency is not HANG and latency <= timeout
            wall += latency if ok else timeout
            journal.append({"run": run_id, "op": op["op"], "attempt": k, "callee": op["callee"],
                            "decision": "SetTimeout", "timeout_s": timeout, "attempts_allowed": allowed,
                            "policy_version": ver,
                            "observed": "completed" if ok else ("hung" if latency is HANG else "timed_out"),
                            "latency_s": latency if ok else None})
            if ok:
                done = True
                break
        outcomes.append({"op": op["op"], "callee": op["callee"], "completed": done,
                         "attempts": used, "wall_s": round(wall, 3)})
    return outcomes, journal


def journal_hash(journal):
    return hashlib.sha256("\n".join(canonical(e) for e in journal).encode()).hexdigest()[:12]


def summarize(outcomes, callee=None):
    rows = [o for o in outcomes if callee is None or o["callee"] == callee]
    walls = sorted(o["wall_s"] for o in rows)
    n = len(rows)
    p95 = walls[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    return {"ops": n,
            "completed": sum(o["completed"] for o in rows),
            "completion": round(sum(o["completed"] for o in rows) / n, 4),
            "mean_wall_s": round(sum(walls) / n, 2),
            "p95_wall_s": round(p95, 2),
            "total_wall_s": round(sum(walls), 1)}


def distill_timeouts(journal, tolerance=0.01, min_samples=50):
    """Offline learner. For each callee, the smallest ladder rung whose premature-abort
    fraction on recorded successful latencies is within tolerance. Abstains when the
    evidence is thin or no rung below the top fits. Never emits the top rung."""
    by_callee = {}
    for e in journal:
        if e["observed"] == "completed":
            by_callee.setdefault(e["callee"], []).append(e["latency_s"])
    fitted, notes = {}, {}
    for callee in sorted(CALLEES):
        lat = sorted(by_callee.get(callee, []))
        if len(lat) < min_samples:
            notes[callee] = f"abstain: {len(lat)} samples < {min_samples}"
            continue
        choice = None
        for rung in LADDER_S[:-1]:
            cut = sum(1 for x in lat if x > rung) / len(lat)
            if cut <= tolerance:
                choice = (rung, cut)
                break
        if choice is None:
            notes[callee] = "abstain: no rung below the top meets the tolerance"
        else:
            fitted[callee] = choice[0]
            notes[callee] = (f"rung {choice[0]} s cuts {choice[1]:.4f} of {len(lat)} recorded "
                             f"successes (tolerance {tolerance})")
    return {"timeout_s": fitted, "max_attempts": 3}, notes


def drift_check(baseline, current, max_completion_drop=0.01):
    drop = round(baseline["completion"] - current["completion"], 4)
    return {"baseline_completion": baseline["completion"], "current_completion": current["completion"],
            "drop": drop, "drift": drop > max_completion_drop}


def replay_divergence(recorded, recomputed):
    a = {(e["op"], e["attempt"]): e for e in recorded}
    b = {(e["op"], e["attempt"]): e for e in recomputed}
    keys = sorted(set(a) | set(b))
    differ = [k for k in keys if a.get(k, {}).get("timeout_s") != b.get(k, {}).get("timeout_s")
              or (k in a) != (k in b)]
    return {"decisions_recorded": len(a), "decisions_recomputed": len(b), "decisions_differ": len(differ),
            "first_difference": (None if not differ else
                                 {"op": differ[0][0], "attempt": differ[0][1],
                                  "recorded": a.get(differ[0], {}).get("timeout_s"),
                                  "recomputed": b.get(differ[0], {}).get("timeout_s")})}
