"""Minimal agent-runtime loop for Chapter 23 (Python 3, asyncio, standard library only).

Each session holds a realistic transcript, waits on a simulated model call,
parses the model's tool-call reply, dispatches the tool through a registry,
appends the call and its result, and serializes a checkpoint. The Rust, Go and
Node programs next to this one do the same work with the same fixture.

Usage (the driver in ../ch23_language.py builds these command lines):
  python3 runtime_py.py --mode startup --fixtures DIR
  python3 runtime_py.py --mode sessions --fixtures DIR --n 1000 --steps 5 \
      --median-ms 2000 --sigma 0.6 --ckpt full --out result.json
  python3 runtime_py.py --mode checkpoint --fixtures DIR --out result.json
"""
import argparse
import asyncio
import gc
import json
import math
import os
import random
import sys
import time

# Compact separators so every runtime writes the same shape. ensure_ascii stays at its
# default (True): non-ASCII characters become \u escapes, the fast path in CPython.
DUMP = json.JSONEncoder(separators=(",", ":")).encode
DUMP_UTF8 = json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).encode

perf = time.perf_counter


def rss_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def hwm_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1


class Fixture:
    def __init__(self, d):
        with open(os.path.join(d, "transcript.json"), encoding="utf-8") as f:
            self.transcript_text = f.read()
        with open(os.path.join(d, "model_reply.json"), encoding="utf-8") as f:
            self.reply_text = f.read()
        with open(os.path.join(d, "tool_result.txt"), encoding="utf-8") as f:
            self.tool_text = f.read()
        self.state_1mb_path = os.path.join(d, "state_1mb.json")


# --- tools and registry -------------------------------------------------------

def make_registry(tool_text):
    def read_file(sid, step, args):
        return {"path": args["path"], "step": step, "session": sid, "content": tool_text}

    def run_tests(sid, step, args):
        return {"path": args["path"], "step": step, "session": sid, "passed": 14, "failed": 0,
                "content": tool_text}

    def other(sid, step, args):
        return {"path": args["path"], "step": step, "session": sid, "content": ""}

    reg = {"read_file": read_file, "run_tests": run_tests}
    for name in ("replace_text", "list_files", "search_code", "git_diff", "export_report", "ask_user"):
        reg[name] = other
    return reg


def check_args(args):
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    p = args.get("path")
    if not isinstance(p, str) or not p:
        raise ValueError("path must be a non-empty string")


# --- one step -----------------------------------------------------------------

def do_step(state, step, fx, registry, ckpt_mode, phases):
    t0 = perf()
    reply = json.loads(fx.reply_text)                      # model reply arrives as JSON text
    t1 = perf()
    call = reply["tool_calls"][0]
    handler = registry[call["name"]]
    check_args(call["arguments"])
    result = handler(state["session_id"], step, call["arguments"])
    tool_msg = {"role": "tool", "tool_call_id": call["id"], "content": DUMP(result)}
    t2 = perf()
    tr = state["transcript"]
    tr.append(reply)
    tr.append(tool_msg)
    state["step"] = step
    t3 = perf()
    if ckpt_mode == "full":
        blob = DUMP(state).encode("utf-8")
    else:
        blob = (DUMP({"session_id": state["session_id"], "step": step,
                            "append": [reply, tool_msg]}) + "\n").encode("utf-8")
    t4 = perf()
    phases[0] += t1 - t0
    phases[1] += t2 - t1
    phases[2] += t3 - t2
    phases[3] += t4 - t3
    return len(blob)


# --- sessions experiment ------------------------------------------------------

async def run_sessions(fx, a):
    registry = make_registry(fx.tool_text)
    mu = math.log(a.median_ms / 1000.0) if a.median_ms > 0 else 0.0
    gc_pauses = []
    gc_start = [0.0]

    def gc_cb(phase, info):
        if phase == "start":
            gc_start[0] = perf()
        else:
            gc_pauses.append(perf() - gc_start[0])

    gc.collect()
    rss_base = rss_kib()
    t_create0 = perf()
    states = []
    for i in range(a.n):
        st = {"session_id": "s%05d" % i, "step": 0, "transcript": json.loads(fx.transcript_text)}
        states.append(st)
    t_create = perf() - t_create0

    gate = asyncio.Event()
    sched = []
    work = []
    phases = [0.0, 0.0, 0.0, 0.0]
    bytes_out = [0]
    parked = [0]

    async def session(i, st):
        rng = random.Random(a.seed * 1000003 + i)
        parked[0] += 1
        await gate.wait()
        if a.stagger_ms > 0:                                    # desynchronize session start
            await asyncio.sleep(round(rng.uniform(0, a.stagger_ms / 1000.0), 3))
        for step in range(1, a.steps + 1):
            if a.median_ms > 0:
                d = round(rng.lognormvariate(mu, a.sigma), 3)   # whole milliseconds
            else:
                d = 0.0
            ts = perf()
            await asyncio.sleep(d)
            tw = perf()
            bytes_out[0] += do_step(st, step, fx, registry, a.ckpt, phases)
            td = perf()
            sched.append(tw - ts - d)
            work.append(td - tw)

    tasks = [asyncio.ensure_future(session(i, st)) for i, st in enumerate(states)]
    while parked[0] < a.n:
        await asyncio.sleep(0)
    del states
    if a.gc == "freeze":
        gc.collect()
        gc.freeze()            # move every existing object to the permanent generation
    rss_parked = rss_kib()
    gc.callbacks.append(gc_cb)
    t0 = perf()
    gate.set()
    await asyncio.gather(*tasks)
    elapsed = perf() - t0
    gc.callbacks.remove(gc_cb)
    return {
        "lang": "python", "n": a.n, "steps": a.steps, "median_ms": a.median_ms, "sigma": a.sigma,
        "ckpt": a.ckpt, "gc_mode": a.gc, "stagger_ms": a.stagger_ms, "rss_base_kib": rss_base, "rss_parked_kib": rss_parked,
        "rss_hwm_kib": hwm_kib(), "create_s": t_create, "elapsed_s": elapsed,
        "total_steps": len(work), "bytes_out": bytes_out[0],
        "phase_s": {"parse": phases[0], "dispatch": phases[1], "append": phases[2], "checkpoint": phases[3]},
        "sched_us": [round(x * 1e6, 1) for x in sched],
        "work_us": [round(x * 1e6, 1) for x in work],
        "gc": {"count": len(gc_pauses), "total_ms": sum(gc_pauses) * 1e3,
               "max_ms": max(gc_pauses) * 1e3 if gc_pauses else 0.0},
        "version": sys.version.split()[0],
    }


# --- checkpoint experiment ----------------------------------------------------

def run_checkpoint(fx, a):
    with open(fx.state_1mb_path, encoding="utf-8") as f:
        text = f.read()
    state = json.loads(text)
    ser, de, wr, ser8 = [], [], [], []
    size = 0
    for i in range(a.reps + 5):
        t0 = perf()
        blob = DUMP(state).encode("utf-8")
        t1 = perf()
        back = json.loads(blob)
        t2 = perf()
        DUMP_UTF8(state).encode("utf-8")
        t3 = perf()
        if i >= 5:
            ser.append(t1 - t0)
            de.append(t2 - t1)
            ser8.append(t3 - t2)
        size = len(blob)
        assert len(back["transcript"]) == len(state["transcript"])
    tmpdir = a.tmpdir
    os.makedirs(tmpdir, exist_ok=True)
    blob = DUMP(state).encode("utf-8")
    for i in range(a.write_reps):
        path = os.path.join(tmpdir, "ckpt.json")
        tmp = path + ".tmp"
        t0 = perf()
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(tmpdir, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
        wr.append(perf() - t0)
    return {"lang": "python", "bytes": size, "messages": len(state["transcript"]),
            "serialize_us": [x * 1e6 for x in ser], "deserialize_us": [x * 1e6 for x in de],
            "write_fsync_us": [x * 1e6 for x in wr],
            "serialize_utf8_us": [x * 1e6 for x in ser8], "version": sys.version.split()[0]}


def main():
    t_start = perf()
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["startup", "sessions", "checkpoint"])
    p.add_argument("--fixtures", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--median-ms", type=float, default=2000.0)
    p.add_argument("--sigma", type=float, default=0.6)
    p.add_argument("--ckpt", choices=["full", "delta"], default="full")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--stagger-ms", type=float, default=0.0)
    p.add_argument("--gc", choices=["default", "freeze"], default="default")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--write-reps", type=int, default=30)
    p.add_argument("--tmpdir", default="/tmp/ch23_ckpt_py")
    p.add_argument("--out")
    a = p.parse_args()
    fx = Fixture(a.fixtures)
    if a.mode == "startup":
        registry = make_registry(fx.tool_text)
        json.loads(fx.reply_text)
        print(json.dumps({"ready": True, "tools": len(registry), "init_ms": (perf() - t_start) * 1e3}))
        return
    if a.mode == "sessions":
        res = asyncio.run(run_sessions(fx, a))
    else:
        res = run_checkpoint(fx, a)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
