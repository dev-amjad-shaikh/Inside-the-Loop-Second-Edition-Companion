# Inside the Loop, second edition: companion

The programs in this directory produce every number the book reports as observed. Each chapter from 3 to 25 has one program, named after its chapter, and the output the book quotes is kept under `runs/`.

## Requirements

- Python 3.10 or later (3.11 was used for the recorded runs) and the system IANA timezone database.
- Nothing else for Chapters 3 to 22, 24 and 25: the programs use only the Python standard library, need no network connection and no API key.
- Chapter 23 additionally builds small runtimes in Rust (cargo, with access to crates.io for the first build), Go and Node.js. See `ch23_language/README.md` for its exact commands. Its full measurement takes about an hour.

## How the experiments work

No program calls a live model. Model behavior comes from authored responses replayed in order, or from a deterministic stand-in (for example a lexical matcher standing in for tool or skill selection) that the chapter names and whose limits it states. Everything around the model is real: files are read and edited, tests run in fresh processes, and SQLite databases play the external services. This isolates the runtime, so a change you make shows up as a change in the output instead of as a model behaving differently on a second attempt. Chapter 19 explains how to evaluate live models once the runtime is sound. Clock time, token counts and prices are simulated where a chapter says so.

Each program writes readable files (JSON traces, SQLite databases, before and after states, a `stdout.txt` or summary) to the directory given with `--output`. Choose a new directory; several programs refuse to overwrite an existing one. Two runs of the same program produce the same files, apart from timing measurements that a chapter labels as measured on the machine.

## Layout

- `itl/core.py`: the teaching runtime shared by all chapters (scripted model, workspace with content versions, tool registry with checked execution, the agent loop of Chapters 2 and 3).
- `itl/chNN_*.py`: helpers for one chapter.
- `chNN_*.py`: the chapter's experiment.
- `fixtures/`: inputs, including the reporting workspace (`fixtures/report`: `settings.json`, `report.py`, `test_report.py`) used throughout the book.
- `runs/chNN/`: the recorded output the book quotes.
- `ch23_language/`: Chapter 23's cross-language runtimes and measurement driver.

## Running

From this directory:

| Chapter | Command |
|---|---|
| 3 One complete run | `python3 ch03_chapter_lab.py --output my-run` (add `--break drop-observations`, `--break swap-ids`, `--break stop-after-edit`, or `--verifier --stop-early`) |
| 4 Model computation | `python3 ch04_model.py --output runs-mine/ch04` |
| 5 Model adapter | `python3 ch05_adapter.py --output runs-mine/ch05` |
| 6 Context | `python3 ch06_context.py --output runs-mine/ch06` |
| 7 Retrieval | `python3 ch07_retrieval.py --output runs-mine/ch07` |
| 8 Memory | `python3 ch08_memory.py --output runs-mine/ch08` |
| 9 Tools | `python3 ch09_tools.py --output runs-mine/ch09` |
| 10 Skills | `python3 ch10_skills.py --output runs-mine/ch10` |
| 11 Planning and control | `python3 ch11_control.py --output runs-mine/ch11` |
| 12 Execution state | `python3 ch12_state.py --output runs-mine/ch12` |
| 13 Recovery | `python3 ch13_recovery.py --output runs-mine/ch13` |
| 14 Working with a person | `python3 ch14_human.py --output runs-mine/ch14` |
| 15 Delegation | `python3 ch15_delegation.py --output runs-mine/ch15` |
| 16 Execution policy | `python3 ch16_policy.py --output runs-mine/ch16` |
| 17 Lifecycle | `python3 ch17_lifecycle.py --output runs-mine/ch17` |
| 18 Observability | `python3 ch18_observability.py --output runs-mine/ch18` |
| 19 Evaluation | `python3 ch19_evaluation.py --output runs-mine/ch19` |
| 20 Security operations | `python3 ch20_secops.py --output runs-mine/ch20` |
| 21 Cost, latency and capacity | `python3 ch21_cost.py --output runs-mine/ch21` (add `--measure-commits` for the SQLite commit timing) |
| 22 Scaling | `python3 ch22_scaling.py --output runs-mine/ch22` |
| 23 Runtime language | see `ch23_language/README.md` |
| 24 Assembling the whole | `python3 ch24_capstone.py --output runs-mine/ch24` |
| 25 Open problems | `python3 ch25_open.py --output runs-mine/ch25` |

Some later chapters reuse recorded runs of earlier ones (Chapter 21 reads the runs of Chapters 3, 11 and 15, for example); a program that needs a missing run says which chapter to run first. Comparing your output directory with `runs/chNN` is the quickest check that your environment reproduces the book.

## Limits

These programs are a laboratory for runtime mechanisms. They run with ordinary local process permissions and are not a sandbox for untrusted code. The export, job, notification and tool services are SQLite stand-ins for real systems. Measurements in Chapters 21 and 23 describe the machine they ran on; the chapters state its configuration and how to repeat them.

## Third-party excerpts

The book quotes short excerpts from public projects at pinned revisions listed in its closing Sources section. Each project remains under its own license (for example, Apache License 2.0 for smolagents, MIT for Pydantic AI and LangGraph, and MIT or Apache-2.0 for Rusty). The companion does not redistribute their source.
