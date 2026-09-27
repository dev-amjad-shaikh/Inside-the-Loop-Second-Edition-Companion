# Chapter 23 companion: choosing the runtime language

Four equivalent minimal agent runtimes and a driver that measures them.

| Program | Language and libraries |
|---|---|
| `py/runtime_py.py` | Python 3 (asyncio, json, standard library only) |
| `rust/src/main.rs` | Rust (tokio, serde, serde_json, rand, rand_distr; optional mimalloc) |
| `go/main.go` | Go (goroutines, encoding/json, standard library only) |
| `node/runtime_node.mjs` | Node.js (async/await, built-in JSON, no packages) |

Each program holds N sessions. A session starts with the same 50 KB transcript
(`../fixtures/ch23_language/transcript.json`, 74 messages), waits on a simulated
model call (lognormal latency), parses the model's tool-call reply, validates the
arguments, dispatches through a registry of eight tools, appends the call and its
result to the transcript and serializes a checkpoint (the whole state, or only the
two new messages with `--ckpt delta`). Every program writes the same checkpoint
bytes for the same input.

## Requirements

- Linux (the programs read `/proc/self/status` for resident memory, and the driver uses `taskset`)
- Python 3.11 (`python3.11` on PATH; 3.13 and 3.14 are used for the GC comparison if present)
- Rust with cargo (tested with 1.95.0); crates.io reachable for the first build
- Go (tested with 1.24.7), Node.js (tested with 22.22.2)

## Exact commands

Run everything from the companion directory.

```
# 1. Build (the driver also does this) and generate the fixtures
(cd ch23_language/rust && cargo build --release)
(cd ch23_language/rust && CARGO_TARGET_DIR=target-mimalloc cargo build --release --features mimalloc)
(cd ch23_language/go && go build -ldflags "-s -w" -o runtime_go .)
python3 ch23_language/fixtures_gen.py fixtures/ch23_language

# 2. Measure. "full" is the matrix quoted in the chapter (about 65 minutes on 2 vCPUs);
#    "quick" runs one repetition of the smaller configurations (about 4 minutes).
python3 ch23_language/ch23_language.py --measure full --output runs/ch23

# 3. Boundary crossings: in-process, PyO3 binding (built with cargo), stdio, HTTP
python3 ch23_language/ch23_language.py --boundary --output runs/ch23

# 4. Optional: cold import time of Python frameworks (needs a virtual environment)
uv venv -p python3.11 /tmp/fwvenv && uv pip install -p /tmp/fwvenv/bin/python pydantic-ai-slim langgraph openai-agents
python3 ch23_language/imports.py /tmp/fwvenv/bin/python runs/ch23/raw/imports.json

# 5. Summarize (deterministic for a given raw/ directory; takes a second)
python3 ch23_language/ch23_language.py --output runs/ch23
```

One runtime by hand:

```
taskset -c 0 python3 ch23_language/py/runtime_py.py --mode sessions --fixtures fixtures/ch23_language \
    --n 10000 --steps 4 --median-ms 4000 --sigma 0.5 --stagger-ms 4000 --out /tmp/py.json
taskset -c 0 python3 ch23_language/py/runtime_py.py --mode sessions ... --gc freeze
ch23_language/rust/target/release/runtime_rs --mode sessions ... --repr typed --threads 1
ch23_language/go/runtime_go --mode sessions ...
node ch23_language/node/runtime_node.mjs --mode sessions ...
```

Other modes: `--mode startup` (load fixtures, build the registry, print `ready`) and
`--mode checkpoint` (serialize and parse a 1 MB state 100 times, then write, fsync and
rename it 30 times).

## What is measured

- **throughput**: 1,000 sessions x 5 steps with no model wait on one CPU; steps per second and CPU per step split into parse, dispatch, append and checkpoint
- **sessions**: 1,000 and 10,000 sessions with a simulated model wait (lognormal, median 4 s, sigma 0.5, start staggered over 4 s, 4 steps); resident memory per session while parked and at peak; per-step overhead = timer lateness plus queueing plus the step's own CPU, reported as p50, p99, p99.9 and max; garbage-collector pauses where the runtime exposes them
- **checkpoint**: serialize, parse and write+fsync+rename a 1 MB state
- **cold start**: wall time from exec to `ready`, 30 runs after 3 warm-ups
- **sizes**: the files each deployment needs on this machine
- **boundary**: one round trip per call from Python through each kind of boundary

Configurations pinned with `taskset -c 0` run on one CPU. `rust-typed-2t`,
`rust-mimalloc-2t` and `go-2p` use both CPUs.

## Output

`runs/ch23/raw/` holds one JSON file per run (with every per-step sample),
`meta.json` (machine, toolchains, crate versions, sizes, cold starts),
`boundary.json` and `imports.json`. `runs/ch23/summary.json` holds percentiles and
the spread across repetitions (median and min-max), and `runs/ch23/report.txt` is the
printed report. `runs/ch23/measure.log` lists every run with its wall time.

## What these programs leave out

No network calls, no real model, no real tools (the tool returns a fixed 1.5 KB
body), no database or durable store (checkpoints stay in memory except in the
checkpoint test), no authentication, no streaming to clients, no framework. Real
runtimes carry more per step; see the chapter's section on what the measurements do
not capture.
