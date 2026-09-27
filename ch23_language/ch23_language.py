"""Chapter 23 companion: does the runtime language change measurable properties?

Two modes.

  python3 ch23_language/ch23_language.py --measure full   # builds and runs every experiment (about 75 min)
  python3 ch23_language/ch23_language.py --measure quick  # smaller matrix (about 6 min)
  python3 ch23_language/ch23_language.py --output runs/ch23

--measure runs the four equivalent runtimes (Python asyncio, Rust tokio + serde_json,
Go goroutines + encoding/json, Node.js async/await) and stores every raw result
under runs/ch23/raw/. Without --measure the script only summarizes what is in
raw/: percentiles, spread across repetitions, the context calculations and the
tables quoted in the chapter. The summary is deterministic for a given raw/.

Run it from the companion directory. Measurements are pinned to one CPU with
taskset unless a configuration says otherwise.
"""
import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COMPANION = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FIXTURES = os.path.join(COMPANION, "fixtures", "ch23_language")

PY = shutil.which("python3.11") or sys.executable
PY313 = shutil.which("python3.13")
PY314 = os.path.expanduser("~/.local/bin/python3.14")
RUST = os.path.join(HERE, "rust", "target", "release", "runtime_rs")
RUST_MI = os.path.join(HERE, "rust", "target-mimalloc", "release", "runtime_rs")
GO = os.path.join(HERE, "go", "runtime_go")
NODE_SCRIPT = os.path.join(HERE, "node", "runtime_node.mjs")
PY_SCRIPT = os.path.join(HERE, "py", "runtime_py.py")

# Configurations: name -> (argv prefix, extra args, pin to one CPU)
CONFIGS = {
    "python": ([PY, PY_SCRIPT], [], True),
    "python-freeze": ([PY, PY_SCRIPT], ["--gc", "freeze"], True),
    "python3.13": ([PY313 or PY, PY_SCRIPT], [], True),
    "python3.14rc2": ([PY314, PY_SCRIPT], [], True),
    "rust-typed": ([RUST], ["--repr", "typed"], True),
    "rust-value": ([RUST], ["--repr", "value"], True),
    "go": ([GO], [], True),
    "node": (["node", NODE_SCRIPT], [], True),
    "rust-typed-2t": ([RUST], ["--repr", "typed", "--threads", "2"], False),
    "rust-mimalloc": ([RUST_MI], ["--repr", "typed"], True),
    "rust-mimalloc-2t": ([RUST_MI], ["--repr", "typed", "--threads", "2"], False),
    "go-2p": ([GO], [], False),
}
MAIN = ["python", "rust-typed", "rust-value", "go", "node"]

# The simulated model call: lognormal, median 4 s, sigma 0.5 (p99 about 12.8 s).
LATENCY = ["--median-ms", "4000", "--sigma", "0.5", "--stagger-ms", "4000", "--steps", "4"]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build():
    r = sh(["cargo", "build", "--release"], cwd=os.path.join(HERE, "rust"))
    if r.returncode:
        sys.exit(r.stderr)
    r = sh(["cargo", "build", "--release", "--features", "mimalloc"], cwd=os.path.join(HERE, "rust"),
           env=dict(os.environ, CARGO_TARGET_DIR="target-mimalloc"))
    if r.returncode:
        sys.exit(r.stderr)
    r = sh(["go", "build", "-ldflags", "-s -w", "-o", "runtime_go", "."], cwd=os.path.join(HERE, "go"))
    if r.returncode:
        sys.exit(r.stderr)


def env_info():
    def first(cmd):
        try:
            return sh(cmd).stdout.strip().splitlines()[0]
        except Exception:
            return "unavailable"
    cpu = "unknown"
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    mem_kib = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                mem_kib = int(line.split()[1])
    lock = open(os.path.join(HERE, "rust", "Cargo.lock")).read()

    def crate(name):
        i = lock.find(f'name = "{name}"\nversion = "')
        return lock[i:].split("\n")[1].split('"')[1] if i >= 0 else "?"
    return {
        "cpu": cpu, "logical_cpus": os.cpu_count(), "mem_gib": round(mem_kib / 1048576, 1),
        "kernel": platform.release(),
        "python": first([PY, "-V"]), "python3.13": first([PY313, "-V"]) if PY313 else None,
        "python3.14": first([PY314, "-V"]) if os.path.exists(PY314) else None,
        "rustc": first(["rustc", "--version"]), "cargo": first(["cargo", "--version"]),
        "go": first(["go", "version"]), "node": first(["node", "--version"]),
        "crates": {c: crate(c) for c in ("tokio", "serde", "serde_json", "rand", "rand_distr")},
    }


def run_config(name, mode, extra, out):
    prefix, cfg_extra, pin = CONFIGS[name]
    cmd = (["taskset", "-c", "0"] if pin else []) + prefix + [
        "--mode", mode, "--fixtures", FIXTURES, "--out", out] + cfg_extra + extra
    t0 = time.perf_counter()
    r = sh(cmd)
    if r.returncode:
        sys.exit(f"{name} failed: {r.stderr[-2000:]}")
    return time.perf_counter() - t0


def cold_start(name, reps):
    prefix, cfg_extra, _ = CONFIGS[name]
    cmd = ["taskset", "-c", "0"] + prefix + ["--mode", "startup", "--fixtures", FIXTURES] + cfg_extra
    for _ in range(3):
        sh(cmd)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = sh(cmd)
        out.append((time.perf_counter() - t0) * 1e3)
        assert '"ready"' in r.stdout.replace(" ", "") or "ready" in r.stdout
    return out


def du_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def sizes():
    py_real = os.path.realpath(PY)
    ver = "python" + ".".join(platform.python_version_tuple()[:2]) if PY == sys.executable else os.path.basename(py_real)
    stdlib = sh([PY, "-c", "import sysconfig;print(sysconfig.get_paths()['stdlib'])"]).stdout.strip()
    node_bin = os.path.realpath(shutil.which("node"))
    return {
        "rust_binary": os.path.getsize(RUST),
        "go_binary": os.path.getsize(GO),
        "python_interpreter": os.path.getsize(py_real),
        "python_stdlib_dir": du_bytes(stdlib),
        "python_stdlib_path": stdlib,
        "python_script": os.path.getsize(PY_SCRIPT),
        "node_binary": os.path.getsize(node_bin),
        "node_script": os.path.getsize(NODE_SCRIPT),
        "note": f"{ver}; sizes of files on this machine, not container images",
    }


def measure_boundary(out):
    """Build the PyO3 extension (needs crates.io) and time four ways to cross a boundary."""
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    bdir = os.path.join(HERE, "binding")
    r = sh(["cargo", "build", "--release"], cwd=bdir,
           env=dict(os.environ, PYO3_PYTHON=PY))
    args = [PY, os.path.join(HERE, "boundary.py"), "--out", os.path.join(raw, "boundary.json")]
    if r.returncode == 0:
        so = os.path.join(bdir, "target", "release", "libch23_binding.so")
        shutil.copy(so, os.path.join(bdir, "ch23_binding.so"))
        args += ["--binding-dir", bdir]
    else:
        print("PyO3 build failed; measuring without the binding case\n" + r.stderr[-1500:])
    r = sh(["taskset", "-c", "0"] + args)
    if r.returncode:
        sys.exit(r.stderr)


def measure(level, out):
    import fixtures_gen
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    build()
    fx = fixtures_gen.write_fixtures(FIXTURES)
    reps = 5 if level == "full" else 1
    lreps = 3 if level == "full" else 1
    plan = []
    for name in MAIN:
        for ck in ("full", "delta"):
            for r in range(reps):
                plan.append(("tput", name, ck, 1000, r))
    for name in ("rust-typed-2t", "go-2p", "rust-mimalloc", "rust-mimalloc-2t"):
        for r in range(3 if level == "full" else 1):
            plan.append(("tput", name, "full", 1000, r))
    ns = [1000, 10000] if level == "full" else [1000]
    for n in ns:
        for name in MAIN:
            for r in range(lreps):
                plan.append(("lat", name, "full", n, r))
    if level == "full":
        for name in MAIN + ["python-freeze"]:
            for r in range(lreps):
                plan.append(("lat", name, "delta", 10000, r))
        for name in ("python-freeze", "python3.13", "python3.14rc2", "rust-mimalloc"):
            for r in range(lreps):
                plan.append(("lat", name, "full", 10000, r))
    log = []
    for i, (kind, name, ck, n, r) in enumerate(plan):
        fn = os.path.join(raw, f"{kind}_{name}_{ck}_n{n}_r{r}.json")
        extra = ["--n", str(n), "--ckpt", ck, "--seed", str(23 + r)]
        if kind == "tput":
            extra += ["--median-ms", "0", "--steps", "5"]
        else:
            extra += LATENCY
        secs = run_config(name, "sessions", extra, fn)
        log.append({"file": os.path.basename(fn), "wall_s": round(secs, 2)})
        print(f"[{i + 1}/{len(plan)}] {os.path.basename(fn)} {secs:.1f}s", flush=True)
    for name in MAIN:
        for r in range(3 if level == "full" else 1):
            fn = os.path.join(raw, f"ckpt_{name}_r{r}.json")
            run_config(name, "checkpoint", ["--reps", "100", "--write-reps", "30",
                                            "--tmpdir", os.path.join(raw, "_tmp_" + name)], fn)
            shutil.rmtree(os.path.join(raw, "_tmp_" + name), ignore_errors=True)
    starts = {}
    for name in ("python", "rust-typed", "go", "node"):
        starts[name] = cold_start(name, 30 if level == "full" else 10)
    meta = {"level": level, "env": env_info(), "fixtures": fx, "sizes": sizes(), "cold_start_ms": starts,
            "latency_args": LATENCY, "log": log, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(raw, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


# --- summary ---------------------------------------------------------------------

def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def spread(vals, nd=1):
    """median [min, max] across repetitions."""
    return {"median": round(statistics.median(vals), nd), "min": round(min(vals), nd), "max": round(max(vals), nd),
            "reps": len(vals)}


def summarize(out):
    raw = os.path.join(out, "raw")
    meta = json.load(open(os.path.join(raw, "meta.json")))
    files = sorted(f for f in os.listdir(raw) if f.endswith(".json") and f not in ("meta.json", "boundary.json", "imports.json"))
    groups = {}
    for f in files:
        if f.startswith("ckpt_"):
            key = ("ckpt", f.split("_")[1])
        else:
            kind, rest = f.split("_", 1)
            name, ck, n, _ = rest.rsplit("_", 3)
            key = (kind, name, ck, int(n[1:]))
        groups.setdefault(key, []).append(json.load(open(os.path.join(raw, f))))
    S = {"env": meta["env"], "fixtures": meta["fixtures"], "sizes": meta["sizes"], "level": meta["level"],
         "throughput": {}, "latency": {}, "checkpoint": {}, "cold_start": {}}

    for key, runs in sorted(groups.items()):
        if key[0] == "tput":
            _, name, ck, n = key
            sps = [r["total_steps"] / r["elapsed_s"] for r in runs]
            per = [r["elapsed_s"] / r["total_steps"] * 1e6 for r in runs]
            ph = {k: statistics.median([r["phase_s"][k] / r["total_steps"] * 1e6 for r in runs])
                  for k in ("parse", "dispatch", "append", "checkpoint")}
            S["throughput"][f"{name}|{ck}"] = {
                "steps_per_s": spread(sps, 0), "us_per_step": spread(per, 1),
                "phase_us": {k: round(v, 2) for k, v in ph.items()},
                "ckpt_bytes_per_step": runs[0]["bytes_out"] // runs[0]["total_steps"]}
        elif key[0] == "lat":
            _, name, ck, n = key
            p50, p99, p999, mx, w50, w99, kib, hwm, util, gcmax, gccount, base = ([] for _ in range(12))
            for r in runs:
                o = [a + b for a, b in zip(r["sched_us"], r["work_us"])]
                p50.append(pct(o, .5) / 1e3)
                p99.append(pct(o, .99) / 1e3)
                p999.append(pct(o, .999) / 1e3)
                mx.append(max(o) / 1e3)
                w50.append(pct(r["work_us"], .5))
                w99.append(pct(r["work_us"], .99))
                kib.append((r["rss_parked_kib"] - r["rss_base_kib"]) / r["n"])
                hwm.append(r["rss_hwm_kib"] / r["n"])
                base.append(r["rss_base_kib"] / 1024)
                util.append(sum(r["work_us"]) / 1e6 / r["elapsed_s"])
                gcmax.append(r["gc"]["max_ms"])
                gccount.append(r["gc"]["count"])
            S["latency"][f"{name}|{ck}|{n}"] = {
                "overhead_ms_p50": spread(p50, 2), "overhead_ms_p99": spread(p99, 2),
                "overhead_ms_p999": spread(p999, 2), "overhead_ms_max": spread(mx, 1),
                "work_us_p50": spread(w50, 1), "work_us_p99": spread(w99, 1),
                "kib_per_session_parked": spread(kib, 1), "kib_per_session_peak": spread(hwm, 1),
                "base_rss_mib": spread(base, 1), "cpu_utilization": spread(util, 3),
                "gc_max_pause_ms": spread(gcmax, 2), "gc_count": spread(gccount, 0),
                "samples_per_rep": runs[0]["total_steps"]}
        elif key[0] == "ckpt":
            name = key[1]
            S["checkpoint"][name] = {
                "bytes": runs[0]["bytes"], "messages": runs[0]["messages"],
                "serialize_ms_median": spread([statistics.median(r["serialize_us"]) / 1e3 for r in runs], 2),
                "serialize_ms_p99": spread([pct(r["serialize_us"], .99) / 1e3 for r in runs], 2),
                "deserialize_ms_median": spread([statistics.median(r["deserialize_us"]) / 1e3 for r in runs], 2),
                "write_fsync_ms_median": spread([statistics.median(r["write_fsync_us"]) / 1e3 for r in runs], 2),
            }
            if "serialize_utf8_us" in runs[0]:
                S["checkpoint"][name]["serialize_utf8_ms_median"] = spread(
                    [statistics.median(r["serialize_utf8_us"]) / 1e3 for r in runs], 2)
    for name, v in meta["cold_start_ms"].items():
        S["cold_start"][name] = {"median_ms": round(statistics.median(v), 1), "p10_ms": round(pct(v, .1), 1),
                                 "p90_ms": round(pct(v, .9), 1), "runs": len(v)}
    bpath = os.path.join(raw, "boundary.json")
    if os.path.exists(bpath):
        S["boundary"] = json.load(open(bpath))
    ipath = os.path.join(raw, "imports.json")
    if os.path.exists(ipath):
        S["imports"] = json.load(open(ipath))
    S["context"] = context(S)
    return S


# Explicit assumptions for the context calculations (not measurements).
ASSUME = {
    "run_model_calls": 7,               # Chapter 3's run makes seven model calls
    "model_call_median_s": 4.0,         # the simulated distribution's median
    "vcpu_usd_per_hour": 0.04,          # order of magnitude for on-demand cloud vCPU
    "gib_usd_per_hour": 0.005,          # order of magnitude for cloud memory
    "model_call_usd": 0.03,             # about 10k input tokens at a few dollars per million
    "engineer_year_usd": 250_000,       # cost of a rewrite or a second-language team, per year
    "steps_per_session_per_s": 1 / 4.5, # one step per mean model latency (4.5 s)
}


def context(S):
    C = {"assumptions": ASSUME, "per_language": {}}
    base = None
    for name in MAIN:
        t = S["throughput"].get(f"{name}|full")
        lat = S["latency"].get(f"{name}|full|10000") or S["latency"].get(f"{name}|full|1000")
        if not t or not lat:
            continue
        cpu_ms_per_step = t["us_per_step"]["median"] / 1e3
        run_overhead_ms = ASSUME["run_model_calls"] * lat["overhead_ms_p50"]["median"]
        run_model_ms = ASSUME["run_model_calls"] * ASSUME["model_call_median_s"] * 1e3
        cores_per_10k = 1e4 * ASSUME["steps_per_session_per_s"] * cpu_ms_per_step / 1e3
        gib_per_10k = 1e4 * lat["kib_per_session_peak"]["median"] / 1048576
        model_usd_per_hour_10k = 1e4 * ASSUME["steps_per_session_per_s"] * 3600 * ASSUME["model_call_usd"]
        cpu_usd_per_hour_10k = cores_per_10k * ASSUME["vcpu_usd_per_hour"]
        mem_usd_per_hour_10k = gib_per_10k * ASSUME["gib_usd_per_hour"]
        C["per_language"][name] = {
            "cpu_ms_per_step_full_ckpt": round(cpu_ms_per_step, 3),
            "overhead_share_of_model_call_p50": lat["overhead_ms_p50"]["median"] / (ASSUME["model_call_median_s"] * 1e3),
            "overhead_share_of_model_call_p99": lat["overhead_ms_p99"]["median"] / (ASSUME["model_call_median_s"] * 1e3),
            "run_overhead_ms_p50": round(run_overhead_ms, 2), "run_model_ms": run_model_ms,
            "cores_per_10k_sessions": round(cores_per_10k, 2), "gib_per_10k_sessions": round(gib_per_10k, 2),
            "cpu_usd_per_hour_10k": round(cpu_usd_per_hour_10k, 4), "mem_usd_per_hour_10k": round(mem_usd_per_hour_10k, 4),
            "model_usd_per_hour_10k": round(model_usd_per_hour_10k, 0),
            "runtime_share_of_model_spend": (cpu_usd_per_hour_10k + mem_usd_per_hour_10k) / model_usd_per_hour_10k,
        }
    # Break-even: concurrent sessions at which Rust's lower CPU and memory would pay one engineer-year per year.
    pl = C["per_language"]
    if "python" in pl and "rust-typed" in pl:
        for other in ("python", "go", "node"):
            if other not in pl:
                continue
            d_usd_h = ((pl[other]["cpu_usd_per_hour_10k"] + pl[other]["mem_usd_per_hour_10k"]) -
                       (pl["rust-typed"]["cpu_usd_per_hour_10k"] + pl["rust-typed"]["mem_usd_per_hour_10k"])) / 1e4
            if d_usd_h > 0:
                C.setdefault("break_even_sessions", {})[other] = round(ASSUME["engineer_year_usd"] / (d_usd_h * 24 * 365))
    return C


def fmt_spread(s, unit=""):
    return f"{s['median']}{unit} [{s['min']}-{s['max']}]"


def report(S):
    L = []
    e = S["env"]
    L.append(f"Machine: {e['cpu']}, {e['logical_cpus']} logical CPUs, {e['mem_gib']} GiB, Linux {e['kernel']}")
    L.append(f"Toolchains: {e['python']}; {e['rustc']}; {e['go']}; node {e['node']}; crates {e['crates']}")
    f = S["fixtures"]
    L.append(f"Fixture: transcript {f['transcript_bytes']} bytes, {f['transcript_messages']} messages; "
             f"checkpoint state {f['state_1mb_bytes']} bytes, {f['state_1mb_messages']} messages")
    L.append("")
    L.append("Throughput, 1,000 sessions x 5 steps, no model wait, one CPU (median [min-max] over reps)")
    for k, v in S["throughput"].items():
        L.append(f"  {k:22s} steps/s {fmt_spread(v['steps_per_s'])}  us/step {fmt_spread(v['us_per_step'])}  "
                 f"phases {v['phase_us']}  ckpt bytes/step {v['ckpt_bytes_per_step']}")
    L.append("")
    L.append("Concurrent sessions, model wait lognormal median 4 s, one CPU (median [min-max] over reps)")
    for k, v in S["latency"].items():
        L.append(f"  {k:26s} KiB/session parked {fmt_spread(v['kib_per_session_parked'])} peak "
                 f"{fmt_spread(v['kib_per_session_peak'])} | overhead p50 {fmt_spread(v['overhead_ms_p50'], ' ms')} "
                 f"p99 {fmt_spread(v['overhead_ms_p99'], ' ms')} max {fmt_spread(v['overhead_ms_max'], ' ms')} | "
                 f"work p50 {v['work_us_p50']['median']} us | util {v['cpu_utilization']['median']} | "
                 f"gc max {v['gc_max_pause_ms']['median']} ms | base {v['base_rss_mib']['median']} MiB")
    L.append("")
    L.append("Checkpoint of a 1 MB state (median of 100 runs per rep; [min-max] across reps)")
    for k, v in S["checkpoint"].items():
        extra = f" | ensure_ascii=False {fmt_spread(v['serialize_utf8_ms_median'], ' ms')}" if "serialize_utf8_ms_median" in v else ""
        L.append(f"  {k:12s} {v['bytes']} bytes serialize {fmt_spread(v['serialize_ms_median'], ' ms')} "
                 f"deserialize {fmt_spread(v['deserialize_ms_median'], ' ms')} write+fsync "
                 f"{fmt_spread(v['write_fsync_ms_median'], ' ms')}{extra}")
    L.append("")
    L.append("Cold start to 'ready' (fixture loaded, registry built), wall clock")
    for k, v in S["cold_start"].items():
        L.append(f"  {k:12s} median {v['median_ms']} ms (p10 {v['p10_ms']}, p90 {v['p90_ms']}, {v['runs']} runs)")
    s = S["sizes"]
    L.append("")
    L.append(f"Deployable size: rust binary {s['rust_binary']} B; go binary {s['go_binary']} B; "
             f"python interpreter {s['python_interpreter']} B + stdlib {s['python_stdlib_dir']} B + script {s['python_script']} B; "
             f"node binary {s['node_binary']} B + script {s['node_script']} B")
    L.append("")
    L.append("Context (assumptions in summary.json -> context.assumptions)")
    for k, v in S["context"]["per_language"].items():
        L.append(f"  {k:12s} cpu/step {v['cpu_ms_per_step_full_ckpt']} ms | p50 overhead / model call "
                 f"{v['overhead_share_of_model_call_p50']:.2e} | p99 {v['overhead_share_of_model_call_p99']:.2e} | "
                 f"7-call run overhead {v['run_overhead_ms_p50']} ms of {v['run_model_ms']:.0f} ms | "
                 f"10k active sessions: {v['cores_per_10k_sessions']} cores, {v['gib_per_10k_sessions']} GiB, "
                 f"${v['cpu_usd_per_hour_10k']}+${v['mem_usd_per_hour_10k']}/h vs model ${v['model_usd_per_hour_10k']:.0f}/h "
                 f"(runtime share {v['runtime_share_of_model_spend']:.2e})")
    if "boundary" in S:
        L.append("")
        L.append(f"Boundary round trips from Python {S['boundary']['python']} (median / p99, microseconds)")
        for k, v in S["boundary"]["cases"].items():
            L.append(f"  {k:18s} {v['median_us']} / {v['p99_us']}  (round medians {v['median_us_min']}-{v['median_us_max']}, "
                     f"{v['rounds']} x {v['reps']} calls)")
    if "imports" in S:
        L.append("")
        L.append(f"Cold import in a fresh {S['imports']['python']} process (median [min-max] of 20; versions {S['imports']['versions']})")
        for k, v in S["imports"]["cases"].items():
            L.append(f"  {k:28s} {v['median_ms']} ms [{v['min_ms']}-{v['max_ms']}]")
    if "break_even_sessions" in S["context"]:
        L.append(f"  concurrent sessions at which switching to rust-typed saves one engineer-year per year: "
                 f"{S['context']['break_even_sessions']}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=os.path.join(COMPANION, "runs", "ch23"))
    p.add_argument("--measure", choices=["full", "quick"])
    p.add_argument("--boundary", action="store_true", help="also measure boundary crossings (PyO3, stdio, HTTP)")
    a = p.parse_args()
    out = os.path.abspath(a.output)
    if a.measure:
        measure(a.measure, out)
    if a.boundary:
        measure_boundary(out)
    if not os.path.exists(os.path.join(out, "raw", "meta.json")):
        sys.exit("No measurements in raw/. Run with --measure quick (or full) first.")
    S = summarize(out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, sort_keys=True)
    text = report(S)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
