"""Cold import time of agent frameworks (Chapter 23 side measurement).

Not part of the standard-library companion: it needs a virtual environment with
the frameworks installed (versions are recorded in the output, not pinned).
Usage: python3 imports.py VENV_PYTHON OUT.json
"""
import json
import statistics
import subprocess
import sys
import time

py, out = sys.argv[1], sys.argv[2]
mods = {"baseline (python -c pass)": "pass", "asyncio + json": "import asyncio, json",
        "pydantic_ai": "import pydantic_ai", "langgraph": "import langgraph.graph",
        "agents (openai-agents)": "import agents"}
res = {"python": subprocess.run([py, "-V"], capture_output=True, text=True).stdout.strip(), "cases": {}}
res["versions"] = json.loads(subprocess.run([py, "-c", "import importlib.metadata as m, json; print(json.dumps("
    "{p: m.version(p) for p in ['pydantic-ai-slim', 'langgraph', 'openai-agents', 'pydantic', 'pydantic-core']}))"],
    capture_output=True, text=True).stdout)
for label, stmt in mods.items():
    cmd = ["taskset", "-c", "0", py, "-c", stmt]
    for _ in range(3):
        subprocess.run(cmd, check=True, capture_output=True)
    t = []
    for _ in range(20):
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True)
        t.append((time.perf_counter() - t0) * 1e3)
    t.sort()
    res["cases"][label] = {"median_ms": round(statistics.median(t), 1), "min_ms": round(t[0], 1),
                           "max_ms": round(t[-1], 1), "runs": 20}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
