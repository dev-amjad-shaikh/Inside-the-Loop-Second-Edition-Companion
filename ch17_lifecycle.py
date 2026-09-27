"""Chapter 17: the agent lifecycle, catalog, release and withdrawal.

    python3 ch17_lifecycle.py --output runs/ch17

What it does, in order:
  1. Gate evidence: runs Chapter 10's report-helper 1.0.0 and 1.1.0 scenarios for
     real (file edits, test runs) and keeps their behavior-test results.
  2. Catalog: registers every part of five agents (runtime build, model record,
     context policy, controller, prompts, tools, skills, policies, evaluators)
     with owner, version, content hash, permissions, data scope and dependents,
     and builds one immutable manifest per release.
  3. Release: gates report-helper 1.1.0 with the evaluator in force, then
     canaries it at 10% of new report-fixer runs in prod.
  4. Fleet simulation (simulated time, 14 days, five agents): the revised
     lifecycle (manifests, canary, withdrawal at step boundaries) and a flawed
     one (mutable "latest", cached per worker, no manifest) over identical
     arrivals.
  5. Withdrawal: in-flight, paused and completed runs of the withdrawn version,
     a real reconciliation check, notices.
  6. Tool versioning (reuses Chapter 9's experiment), duplicate catalog entries
     (reuses Chapter 10's activation cases) and resuming under a newer store
     schema (reuses Chapter 12's upcaster).
  7. Governance queries over the catalog and run records.
"""
import argparse
import heapq
import json
import os
import random
import shutil
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ch09_tools as ch09                      # noqa: E402  (Chapter 9 lab: callers, versioning)
import ch10_skills as ch10                     # noqa: E402  (Chapter 10 lab: scenarios, cases)
from itl import ch09_tools as T                # noqa: E402
from itl import ch10_skills as sk              # noqa: E402
from itl import ch12_state as st12             # noqa: E402
from itl import ch17_lifecycle as L            # noqa: E402
from itl.core import Workspace, report_tools, write_run, content_version  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "ch17_lifecycle")
DAYS = 14
HOURS = DAYS * 24
CANARY_AT = 7 * 24 + 9          # Sep 08 09:00
ALIAS_MOVE_AT = 10 * 24 + 10    # Sep 11 10:00, flawed design only: provider moves the model alias
MIN_SAMPLE = 10
MARGIN = 0.2
RESPONSE_HOURS = 1              # on-call response from alert to withdrawal or rollback
CACHE_TTL = 6                   # flawed design: each worker refreshes its "latest" cache every 6 h
WORKERS = 4

AGENTS = {
    # agent: (owner, runs per hour, short id)
    "report-fixer": ("reporting-platform", 2.0, "rf"),
    "status-writer": ("team-communications", 1.0, "sw"),
    "export-assistant": ("finance-ops", 1.5, "ex"),
    "release-notes-bot": ("developer-experience", 0.3, "rn"),
    "csv-cleaner": ("data-platform", 0.5, "cc"),
}
REQUESTERS = ["maria@acme", "dana@acme", "li@acme", "sam@acme", "priya@acme", "omar@acme"]


def say(*a):
    print(*a)


# ---------------------------------------------------------------------------
# 1. Gate evidence: real runs of report-helper 1.0.0 and 1.1.0 (Chapter 10)
# ---------------------------------------------------------------------------

def gate_runs(out):
    d = os.path.join(out, "gate_runs")
    c_dir = ch10.materialize_versions(ch10.STATES["C_both_revised"], os.path.join(d, "catalog_1_0"))
    d_dir = ch10.materialize_versions(ch10.STATES["D_helper_1_1"], os.path.join(d, "catalog_1_1"))
    r10 = ch10.run_scenario("report-helper@1.0.0", c_dir, ch10.responses_v10, d)
    r11 = ch10.run_scenario("report-helper@1.1.0", d_dir, ch10.responses_v11, d)
    res = {}
    for v, r in (("1.0.0", r10), ("1.1.0", r11)):
        res[v] = {t["check"].split()[0]: t["pass"] for t in r["tests"]}
        res[v + "_names"] = {t["check"].split()[0]: t["check"] for t in r["tests"]}
        res[v + "_after"] = {f: content_version(t) for f, t in r["after"].items()}
        res[v + "_after_text"] = r["after"]
    return res


def reconcile_check(gate, out):
    """For a completed run that used 1.1.0, re-establish the missing evidence:
    put the run's final test file beside the pre-fix report.py and run it.
    If the new test fails there, the regression test does detect the bug."""
    ws = Workspace("report")
    before = ws.snapshot()
    ws.write("test_report.py", gate["1.1.0_after_text"]["test_report.py"])
    obs = report_tools(ws).execute_checked({"id": "rc1", "name": "run_tests",
                                            "arguments": {"tests": ["test_report.py"]}})
    result = {"report.py": content_version(before["report.py"]),
              "test_report.py": content_version(gate["1.1.0_after_text"]["test_report.py"]),
              "status": obs["status"], "failed_tests": obs["failed_tests"], "tests_run": obs["tests_run"]}
    ws.close()
    write_run(out, "reconcile_check.json", result)
    return result


# ---------------------------------------------------------------------------
# 2. The catalog and its parts
# ---------------------------------------------------------------------------

def skill_dir(name, version):
    return ch10.src(name, version)


def skill_content(name, version):
    d = skill_dir(name, version)
    files = {rel: sk.sha256_bytes(data)[:12] for rel, data in sorted(sk.package_files(d).items())}
    meta, _ = sk.parse_frontmatter(open(os.path.join(d, "SKILL.md"), encoding="utf-8").read())
    return ({"files": files, "declared_tools": (meta.get("allowed-tools") or "").split()},
            sk.package_hash(d), meta)


def build_catalog(path, gate):
    if os.path.exists(path):
        os.remove(path)
    cat = L.Catalog(path)
    t0 = 0

    cat.register("runtime", "itl-runtime", "2026.9.1",
                 {"build": "itl-runtime 2026.9.1", "store_schema": st12.SCHEMA_VERSION},
                 "agent-platform", t0, {"reads_store_schema": f"<= {st12.SCHEMA_VERSION}"})
    cat.register("model", "model-a", "1.0.0",
                 {"provider": "example-provider", "revision": "model-a-2026-06-01",
                  "decoding": {"temperature": 0, "max_output_tokens": 4096}},
                 "ml-platform", t0, {}, {"egress": "request content goes to example-provider"})
    cat.register("context", "ctx-default", "2.0.0",
                 {"policy": "recent-first with pinned task constraints", "token_counter": "chars/4",
                  "budget_tokens": 24000}, "agent-platform", t0)
    cat.register("controller", "fix-loop", "1.0.0",
                 {"max_model_calls": 14, "completion": "tests pass after the last edit"}, "agent-platform", t0)
    cat.register("controller", "answer-loop", "1.0.0", {"max_model_calls": 6}, "agent-platform", t0)

    # tools: the report tools from Part I and the export tools from Chapter 9
    ws = Workspace("report")
    report_defs = report_tools(ws).definitions()
    ws.close()
    effects = {"read_file": "read_only", "replace_text": "writes_workspace", "run_tests": "executes_code"}
    for d in report_defs:
        cat.register("tool", d["name"], "1.0.0", d, "agent-platform", t0,
                     {"effect": effects[d["name"]]}, {"workspace": "the run's workspace only"})
    records, _ = sk.discover(ch10.materialize_versions(ch10.STATES["C_both_revised"],
                                                       os.path.join(os.path.dirname(path), "tmp_catalog")))
    for d in sk.skill_tool_definitions(records):
        cat.register("tool", d["name"], "1.0.0", {k: d[k] for k in ("name", "description", "parameters")},
                     "agent-platform", t0, {"effect": "read_only" if d["name"] != "run_skill_script"
                                            else "executes_pinned_script"},
                     {"skills": "packages pinned by the manifest"})
    shutil.rmtree(os.path.join(os.path.dirname(path), "tmp_catalog"))
    for d in T.revised_definitions(1):
        if d["name"] in ("reports_search", "exports_create", "exports_get_status"):
            cat.register("tool", d["name"], "1.0.0", d, d["owner"], t0, {"effect": d["effect"]},
                         {"tenant": "bound from the signed-in session"}, compatibility=schema_compat)

    # skills: Chapter 10's packages, hashed exactly as Chapter 10 hashes them
    for name, version, owner in (("report-helper", "1.0.0", "reporting-platform"),
                                 ("status-report-writer", "1.1.0", "team-communications"),
                                 ("release-notes", "1.0.0", "developer-experience"),
                                 ("csv-cleanup", "1.0.0", "data-platform")):
        content, h, meta = skill_content(name, version)
        cat.register("skill", name, version, content, owner, t0,
                     {"declared_tools": content["declared_tools"], "network": "none declared"},
                     {"reads": "workspace files named in the task"}, content_hash=h)

    prompts = {
        "report-fixer-system": ch10.SYSTEM,
        "status-writer-system": "Write status reports from the supplied metrics. Do not invent numbers.",
        "export-assistant-system": "Export reports for the signed-in tenant. Confirm the report version.",
        "release-notes-system": "Draft release notes from merged pull requests.",
        "csv-cleaner-system": "Repair CSV files in the workspace; keep the original.",
    }
    owners = {"report-fixer-system": "reporting-platform", "status-writer-system": "team-communications",
              "export-assistant-system": "finance-ops", "release-notes-system": "developer-experience",
              "csv-cleaner-system": "data-platform"}
    for n, text in prompts.items():
        cat.register("prompt", n, "1.0.0", {"text": text}, owners[n], t0)

    cat.register("policy", "report-fix-no-deploy", "1.0.0", ch10.RUN_POLICY.describe(), "reporting-platform",
                 t0, {"allowed_tools": ch10.RUN_POLICY.describe()["allowed_tools"], "network": False},
                 {"workspace": "run workspace", "deploy": "refused"})
    cat.register("policy", "export-finance", "1.0.0",
                 {"allowed_tools": ["reports_search", "exports_create", "exports_get_status"],
                  "destinations": ["finance_sftp"], "identity": "session"}, "finance-ops", t0,
                 {"allowed_tools": ["reports_search", "exports_create", "exports_get_status"]},
                 {"tenant": "session tenant", "destinations": ["finance_sftp"]})
    cat.register("policy", "docs-read-write", "1.0.0",
                 {"allowed_tools": ["read_file", "replace_text", "activate_skill", "read_skill_resource"],
                  "network": False}, "agent-platform", t0,
                 {"allowed_tools": ["read_file", "replace_text", "activate_skill", "read_skill_resource"]},
                 {"workspace": "run workspace"})

    checks_v1 = ["B1", "B2", "B4", "B6", "B7", "B8"]
    cat.register("evaluator", "report-fix-behavior", "1.0.0",
                 {"checks": checks_v1, "dataset": "report-fix-tasks@1", "oracle": "run records"},
                 "reporting-platform", t0)
    cat.register("evaluator", "answer-checks", "1.0.0", {"checks": ["answer present"]}, "agent-platform", t0)

    common = [("runtime", "itl-runtime", "2026.9.1"), ("model", "model-a", "1.0.0"),
              ("context", "ctx-default", "2.0.0")]
    releases = {
        "report-fixer": common + [("controller", "fix-loop", "1.0.0"), ("prompt", "report-fixer-system", "1.0.0"),
                                  ("tool", "read_file", "1.0.0"), ("tool", "replace_text", "1.0.0"),
                                  ("tool", "run_tests", "1.0.0"), ("tool", "activate_skill", "1.0.0"),
                                  ("tool", "read_skill_resource", "1.0.0"), ("tool", "run_skill_script", "1.0.0"),
                                  ("skill", "report-helper", "1.0.0"),
                                  ("policy", "report-fix-no-deploy", "1.0.0"),
                                  ("evaluator", "report-fix-behavior", "1.0.0")],
        "status-writer": common + [("controller", "answer-loop", "1.0.0"),
                                   ("prompt", "status-writer-system", "1.0.0"), ("tool", "read_file", "1.0.0"),
                                   ("tool", "activate_skill", "1.0.0"), ("skill", "status-report-writer", "1.1.0"),
                                   ("policy", "docs-read-write", "1.0.0"), ("evaluator", "answer-checks", "1.0.0")],
        "export-assistant": common + [("controller", "answer-loop", "1.0.0"),
                                      ("prompt", "export-assistant-system", "1.0.0"),
                                      ("tool", "reports_search", "1.0.0"), ("tool", "exports_create", "1.0.0"),
                                      ("tool", "exports_get_status", "1.0.0"), ("policy", "export-finance", "1.0.0"),
                                      ("evaluator", "answer-checks", "1.0.0")],
        "release-notes-bot": common + [("controller", "answer-loop", "1.0.0"),
                                       ("prompt", "release-notes-system", "1.0.0"), ("tool", "read_file", "1.0.0"),
                                       ("tool", "activate_skill", "1.0.0"), ("skill", "release-notes", "1.0.0"),
                                       ("policy", "docs-read-write", "1.0.0"), ("evaluator", "answer-checks", "1.0.0")],
        "csv-cleaner": common + [("controller", "answer-loop", "1.0.0"), ("prompt", "csv-cleaner-system", "1.0.0"),
                                 ("tool", "read_file", "1.0.0"), ("tool", "replace_text", "1.0.0"),
                                 ("tool", "activate_skill", "1.0.0"), ("skill", "csv-cleanup", "1.0.0"),
                                 ("policy", "docs-read-write", "1.0.0"), ("evaluator", "answer-checks", "1.0.0")],
    }
    manifests = {}
    for agent, refs in releases.items():
        owner = AGENTS[agent][0]
        for kind, name, version in refs:
            cat.depend(agent, owner, kind, name, version)
        mid = cat.build_manifest(agent, "initial", refs, t0, owner)
        ev = gate_decision(gate, "1.0.0", None, ["B1", "B2", "B4", "B6", "B7", "B8"]) \
            if agent == "report-fixer" else {"decision": "allow", "suite": "answer-checks@1.0.0",
                                             "note": "not modeled in this companion"}
        cat.add_evidence(mid, "gate", ev, t0, owner)
        cat.point("staging", agent, t0, owner, "initial release", active=mid)
        cat.point("prod", agent, t0, owner, "initial release", active=mid)
        manifests[agent] = mid
    cat.db.commit()
    return cat, releases, manifests


def schema_compat(old, new):
    return T.schema_changes(old["parameters"], new["parameters"])


def gate_decision(gate, candidate, baseline, checks):
    """Allow when every check in the suite passes for the candidate and no
    check that passed for the baseline fails for it."""
    res = {c: gate[candidate][c] for c in checks}
    failed = [c for c in checks if not res[c]]
    regressed = [c for c in checks if baseline and gate[baseline][c] and not res[c]]
    return {"decision": "allow" if not failed and not regressed else "refuse",
            "candidate": f"report-helper@{candidate}", "baseline": baseline and f"report-helper@{baseline}",
            "checks": res, "failed": failed, "regressed": regressed,
            "dataset": "report-fix-tasks@1 (1 task; scripted model, so repeats are identical)"}


# ---------------------------------------------------------------------------
# 4. Fleet simulation
# ---------------------------------------------------------------------------

def run_specs(seed=1729):
    """Identical arrivals and durations for both lifecycles."""
    rng = random.Random(seed)
    specs = []
    for agent, (owner, rate, short) in AGENTS.items():
        t, n = 0.0, 0
        while True:
            t += rng.expovariate(rate)
            if t >= HOURS - 6:
                break
            n += 1
            steps = [rng.uniform(0.15, 0.6) for _ in range(rng.randint(6, 9))]
            pause = rng.uniform(8, 40) if rng.random() < 0.06 else 0.0
            specs.append({"run_id": f"{short}-{n:04d}", "agent": agent, "start": t, "steps": steps,
                          "pause_after": 3 if pause else None, "pause": pause,
                          "requester": REQUESTERS[rng.randrange(len(REQUESTERS))]})
    specs.sort(key=lambda s: s["start"])
    return specs


def injected_specs(m2):
    """Two runs added on purpose to exercise the paths a random sample may miss:
    one paused across the flawed design's change, one paused on the canary
    across the withdrawal. Both are labeled 'injected' in every output."""
    k = 1
    while L.canary_draw("prod:report-fixer", m2, f"rf-inj{k}") >= 0.10:
        k += 1
    return [
        {"run_id": "rf-inj0", "agent": "report-fixer", "start": CANARY_AT - 3.0, "steps": [0.3] * 7,
         "pause_after": 3, "pause": 4.0, "requester": "maria@acme",
         "why": "paused across the release"},
        {"run_id": f"rf-inj{k}", "agent": "report-fixer", "start": 9 * 24 + 2.0, "steps": [0.3] * 7,
         "pause_after": 3, "pause": 30.0, "requester": "dana@acme",
         "why": f"id chosen so the canary draw admits it (draw < 0.10 at k={k})"},
    ]


def step_times(spec):
    """Times of step boundaries (start of each step) and the completion time."""
    t, times = spec["start"], []
    for i, d in enumerate(spec["steps"]):
        if spec["pause_after"] is not None and i == spec["pause_after"]:
            t += spec["pause"]
        times.append(t)
        t += d
    return times, t


def worker_of(run_id, salt=""):
    return int(L.sha(run_id + salt), 16) % WORKERS


def simulate_revised(cat, specs, gate, m1, m2, runs_db):
    """Manifests pinned at admission; canary by seeded draw; the executor checks
    the withdrawn set at every step boundary; paused runs check it at resume."""
    con = sqlite3.connect(runs_db)
    con.executescript("""
    CREATE TABLE runs (run_id TEXT PRIMARY KEY, agent TEXT, env TEXT, manifest_id TEXT, slot TEXT,
      requester TEXT, admitted REAL, ended REAL, status TEXT, replaces TEXT, flags TEXT);
    CREATE TABLE run_events (run_id TEXT, at REAL, event TEXT, detail TEXT);
    CREATE TABLE online_checks (run_id TEXT, manifest_id TEXT, completed REAL, results TEXT);""")
    surface = "prod:report-fixer"
    q = []                       # (time, seq, kind, payload)
    seq = 0

    def push(t, kind, payload):
        nonlocal seq
        seq += 1
        heapq.heappush(q, (t, seq, kind, payload))

    for s in specs:
        push(s["start"], "admit", s)
    push(CANARY_AT, "canary", None)
    for h in range(HOURS):
        push(h + 0.999, "monitor", h)
    state = {"canary_start": None, "alert": None, "withdrawn_at": None, "decisions": [],
             "halted": [], "refused_resume": [], "replacements": []}
    live = {}

    def admit(spec, t, replaces=None):
        ptr = cat.pointer("prod", spec["agent"])
        slot, mid = "active", ptr["active"]
        if ptr["canary"] and L.canary_draw(f"prod:{spec['agent']}", ptr["canary"], spec["run_id"]) < ptr["fraction"]:
            slot, mid = "canary", ptr["canary"]
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (spec["run_id"], spec["agent"], "prod", mid, slot, spec["requester"], t, None, "running",
                     replaces, "[]"))
        con.execute("INSERT INTO run_events VALUES (?,?,?,?)", (spec["run_id"], t, "manifest_resolved",
                                                                json.dumps({"manifest_id": mid, "slot": slot})))
        times, end = step_times(dict(spec, start=t))
        live[spec["run_id"]] = {"spec": spec, "manifest": mid, "slot": slot}
        for i, st in enumerate(times):
            resumed = spec["pause_after"] is not None and i == spec["pause_after"]
            push(st, "resume" if resumed else "step", (spec["run_id"], i))
        push(end, "complete", spec["run_id"])

    def replace(run_id, t, reason):
        old = live.pop(run_id)
        spec = dict(old["spec"], run_id=run_id + "-r", pause_after=None, pause=0.0)
        state["replacements"].append({"run": run_id, "replacement": spec["run_id"], "at": round(t, 2),
                                      "reason": reason})
        admit(spec, t, replaces=run_id)

    while q:
        t, _, kind, p = heapq.heappop(q)
        if kind == "admit":
            admit(p, t)
        elif kind == "canary":
            ptr = cat.pointer("prod", "report-fixer")
            cat.point("prod", "report-fixer", CANARY_AT, "reporting-platform", "canary report-helper 1.1.0",
                      canary=m2, fraction=0.10)
            state["canary_start"] = t
        elif kind in ("step", "resume"):
            run_id, i = p
            if run_id not in live:
                continue
            bad = cat.withdrawn_parts(live[run_id]["manifest"])
            if bad:
                ev = "resume_refused" if kind == "resume" else "halted_at_boundary"
                con.execute("UPDATE runs SET status=?, ended=? WHERE run_id=?",
                            ("restarted" if kind == "resume" else "halted", t, run_id))
                con.execute("INSERT INTO run_events VALUES (?,?,?,?)", (run_id, t, ev, json.dumps(
                    {"step": i, "withdrawn": [f"{b['kind']}:{b['name']}@{b['version']}" for b in bad]})))
                (state["refused_resume"] if kind == "resume" else state["halted"]).append(
                    {"run": run_id, "at": round(t, 2), "step": i})
                replace(run_id, t, ev)
        elif kind == "complete":
            if p not in live:
                continue
            r = live.pop(p)
            con.execute("UPDATE runs SET status='completed', ended=? WHERE run_id=?", (t, p))
            if r["spec"]["agent"] == "report-fixer":
                skill_v = next(x["version"] for x in cat.manifest(r["manifest"])["parts"] if x["kind"] == "skill")
                con.execute("INSERT INTO online_checks VALUES (?,?,?,?)",
                            (p, r["manifest"], t, json.dumps({c: gate[skill_v][c] for c in gate[skill_v]})))
        elif kind == "monitor":
            if state["canary_start"] is None or state["alert"] is not None:
                continue
            rows = con.execute("SELECT r.slot, o.results FROM online_checks o JOIN runs r USING (run_id) "
                               "WHERE o.completed >= ?", (state["canary_start"],)).fetchall()
            can = [json.loads(x[1]) for x in rows if x[0] == "canary"]
            act = [json.loads(x[1]) for x in rows if x[0] == "active"]
            if len(can) < MIN_SAMPLE or len(act) < MIN_SAMPLE:
                state["decisions"].append({"hour": p, "canary_n": len(can), "decision": "insufficient_evidence"})
                continue
            worst = None
            for c in can[0]:
                fc = sum(not x[c] for x in can) / len(can)
                fa = sum(not x[c] for x in act) / len(act)
                if fc - fa >= MARGIN and (worst is None or fc - fa > worst[3]):
                    worst = (c, fc, fa, fc - fa)
            if worst:
                state["alert"] = {"hour": p + 0.999, "check": worst[0], "canary_fail": round(worst[1], 3),
                                  "active_fail": round(worst[2], 3), "canary_n": len(can), "active_n": len(act)}
                state["decisions"].append({"hour": p, "canary_n": len(can), "decision": "regression",
                                           "check": worst[0]})
                push(p + 0.999 + RESPONSE_HOURS, "withdraw", worst[0])
            else:
                state["decisions"].append({"hour": p, "canary_n": len(can), "decision": "no_regression"})
        elif kind == "withdraw":
            reason = (f"canary regression on {gate['1.1.0_names'][p]}: "
                      f"{state['alert']['canary_fail']:.0%} of canary runs vs "
                      f"{state['alert']['active_fail']:.0%} of active runs")
            cat.set_status("skill", "report-helper", "1.1.0", "withdrawn", t, "reporting-platform", reason,
                           replacement="1.0.0")
            for mid in cat.manifests_with("skill", "report-helper", "1.1.0"):
                for env in ("staging", "prod"):
                    ptr = cat.pointer(env, "report-fixer")
                    if ptr and ptr["canary"] == mid:
                        cat.point(env, "report-fixer", t, "reporting-platform", "withdrawal clears canary",
                                  canary=None, fraction=None)
            state["withdrawn_at"] = t
    con.commit()
    return con, state


def simulate_flawed(specs, gate, publish_at):
    """Parts referenced as 'latest'; each worker caches the catalog for
    CACHE_TTL hours; a paused run re-resolves 'latest' when it resumes; runs
    record the alias only. The simulation keeps the ground truth separately."""
    latest_log = [(0.0, "1.0.0")]          # (time, version) for report-helper@latest
    alias_log = [(0.0, "model-a-2026-06-01"), (ALIAS_MOVE_AT, "model-a-2026-09-10")]
    state = {"alert": None, "rolled_back_at": None}

    def latest_at(t, log):
        v = None
        for when, ver in log:
            if when <= t:
                v = ver
        return v

    def cached(worker, t):
        """Version a worker serves at t: latest as of its last refresh."""
        offset = worker * CACHE_TTL / WORKERS
        if t < offset:
            refresh = 0.0
        else:
            refresh = offset + ((t - offset) // CACHE_TTL) * CACHE_TTL
        return latest_at(refresh, latest_log)

    latest_log.append((publish_at, "1.1.0"))
    events = []
    truth, records = {}, {}
    rf = [s for s in specs if s["agent"] == "report-fixer"]
    # process in time order so the monitor can react
    completions = []
    for s in rf:
        times, end = step_times(s)
        completions.append((end, s, times))
    completions.sort(key=lambda x: x[0])
    # first pass: detection time depends on completions after publish, which depend on
    # versions served; versions before rollback do not depend on the rollback time.
    done = []
    for end, s, times in completions:
        w0 = worker_of(s["run_id"])
        v_start = cached(w0, times[0])
        v_after = v_start
        if s["pause_after"] is not None:
            v_after = cached(worker_of(s["run_id"], "resume"), times[s["pause_after"]])
        truth[s["run_id"]] = {"start_version": v_start, "resume_version": v_after if s["pause_after"] is not None
                              else None, "end": end, "start": times[0], "times": times, "spec": s}
        if state["alert"] is None and end >= publish_at:
            v_eff = v_after
            done.append((end, gate[v_eff]))
            after = [r for e, r in done if e >= publish_at]
            before = [r for e, r in done if e < publish_at]
            if len(after) >= MIN_SAMPLE:
                for c in after[0]:
                    fa = sum(not x[c] for x in after) / len(after)
                    fb = sum(not x[c] for x in before) / len(before) if before else 0.0
                    if fa - fb >= MARGIN:
                        state["alert"] = {"hour": end, "check": c, "after_fail": round(fa, 3),
                                          "before_fail": round(fb, 3), "n_after": len(after)}
                        state["rolled_back_at"] = end + RESPONSE_HOURS
                        latest_log.append((state["rolled_back_at"], "1.0.0"))
                        break
        elif end < publish_at:
            done.append((end, gate[v_start]))
    # second pass: recompute every run's versions with the full latest log (rollback included)
    for rid, tr in truth.items():
        s, times = tr["spec"], tr["times"]
        tr["start_version"] = cached(worker_of(rid), times[0])
        tr["resume_version"] = (cached(worker_of(rid, "resume"), times[s["pause_after"]])
                                if s["pause_after"] is not None else None)
        tr["served_models"] = sorted({latest_at(t, alias_log) for t in times})
        records[rid] = {"run_id": rid, "agent": "report-fixer", "started": round(times[0], 2),
                        "skill": "report-helper@latest", "model": "model-a-latest"}
    return truth, records, latest_log, state


def reconstruct(records, latest_log):
    """What an investigator can do with the flawed records: assume each run used
    whatever 'latest' pointed to when it started."""
    out = {}
    for rid, r in records.items():
        v = None
        for when, ver in latest_log:
            if when <= r["started"]:
                v = ver
        out[rid] = v
    return out


def in_flight_by_hour(intervals):
    """intervals: list of (start, end, label). Returns {label: [count at each hour]}."""
    labels = sorted({x[2] for x in intervals})
    series = {lab: [0] * HOURS for lab in labels}
    for s, e, lab in intervals:
        for h in range(max(0, int(s)), min(HOURS, int(e) + 1)):
            if s <= h + 0.5 < e:
                series[lab][h] += 1
    return series


# ---------------------------------------------------------------------------
# 7. Tool versioning, duplicate entries, schema on resume
# ---------------------------------------------------------------------------

def exp_tool_versioning(cat, out):
    v1 = next(d for d in T.revised_definitions(1) if d["name"] == "exports_create")
    v2 = next(d for d in T.revised_definitions(2) if d["name"] == "exports_create")
    attempts = []
    for version in ("1.0.0", "1.1.0", "2.0.0"):
        try:
            h = cat.register("tool", "exports_create", version, v2, "reporting-platform", 200,
                             {"effect": v2["effect"]}, {"tenant": "bound from the signed-in session"},
                             compatibility=schema_compat)
            attempts.append({"version": version, "result": "released", "hash": h[:12]})
        except L.Refused as exc:
            attempts.append({"version": version, "result": "refused", "message": str(exc)})
    # the five callers from Chapter 9 become catalog dependents of exports_create@1.0.0
    owners = {"nightly finance workflow": "finance-ops", "saved skill: month-end pack": "finance-ops",
              "replay of agent run r-2291": "eval-platform", "scheduled task: weekly revenue": "analytics",
              "analytics agent": "analytics"}
    for who, *_ in ch09.CALLERS:
        cat.depend(who, owners[who], "tool", "exports_create", "1.0.0")
    deps_before = cat.dependents("tool", "exports_create", "1.0.0")
    cat.set_status("tool", "exports_create", "1.0.0", "deprecated", 200, "reporting-platform",
                   "renamed day to report_date", replacement="2.0.0", until="2027-03-31")
    owners_notified = sorted({d["consumer_owner"] for d in deps_before})
    for d in deps_before:
        cat.notify(200, d["consumer_owner"], f"{d['consumer']} pins tool:exports_create@1.0.0",
                   "exports_create@1 is deprecated: rename day to report_date. Version 1 calls are adapted "
                   "until 2027-03-31.")
    # Chapter 9's experiment rerun: what each deployment does to the five callers
    d9 = os.path.join(out, "ch09_versioning")
    os.makedirs(d9, exist_ok=True)
    compat, results = ch09.versioning(d9)
    summary = {}
    for variant, rows in results.items():
        summary[variant] = {"ok": sum(r["status"] == "ok" and not r["silently_wrong"] for r in rows),
                            "failed": sum(r["status"] != "ok" for r in rows),
                            "silently_wrong": sum(r["silently_wrong"] for r in rows),
                            "deprecation_notices": sum(bool(r["deprecation"]) for r in rows)}
    # migration within the window (authored schedule; the catalog query is what is observed)
    schedule = [("nightly finance workflow", "2026-10-06"), ("scheduled task: weekly revenue", "2026-10-20"),
                ("saved skill: month-end pack", "2026-11-02"), ("analytics agent", "2027-01-15")]
    remaining = {}
    checkpoints = ["2026-09-30", "2026-10-31", "2026-12-31", "2027-03-31"]
    done = set()
    for cp in checkpoints:
        for who, day in schedule:
            if day <= cp and who not in done:
                cat.repin(who, "tool", "exports_create", "2.0.0", 200, on=day)
                done.add(who)
        remaining[cp] = [d["consumer"] for d in cat.dependents("tool", "exports_create", "1.0.0")]
    res = {"attempts": attempts, "dependents_at_deprecation": deps_before, "owners_notified": owners_notified,
           "ch09_rerun": summary, "remaining_on_1_0_0": remaining, "migration_schedule_authored": schedule}
    write_run(out, "tool_versioning.json", res)
    return res


def exp_duplicates(cat, out):
    """A second team registers its own copy of the reporting fix procedure."""
    base = [(n, v) for n, v in ch10.STATES["C_both_revised"]]
    fork_dir = os.path.join(FIX, "report-timezone-fixer", "1.0.0", "report-timezone-fixer")
    results = {}

    def suite(dirs, label):
        tmp = os.path.join(out, "catalog_states", label)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        for d in dirs:
            shutil.copytree(d, os.path.join(tmp, os.path.basename(d)))
        records, _ = sk.discover(tmp)
        act = sk.LexicalActivator(records)
        rows = []
        for cid, req, exp in ch10.CASES:
            ch = act.choose(req)
            rows.append({"id": cid, "expected": exp, "chosen": ch["chosen"],
                         "contested_with": ch["contested_with"],
                         "pass": ch["chosen"] == exp and not ch["contested_with"]})
        return {"passed": sum(r["pass"] for r in rows), "total": len(rows),
                "failed": [r["id"] for r in rows if not r["pass"]],
                "rows": rows, "activator": act, "records": records}

    base_dirs = [ch10.src(n, v) for n, v in base]
    before = suite(base_dirs, "C_catalog")
    with_fork = suite(base_dirs + [fork_dir], "C_plus_fork")
    ov = L.overlap(with_fork["activator"].index, "report-timezone-fixer")
    # registration attempt 1: the fork under the existing name (as a copy would arrive)
    content, h, meta = skill_content_dir(fork_dir)
    name_attempt = None
    try:
        cat.register("skill", "report-helper", "1.0.1", content, "data-platform", 210, content_hash=h)
    except L.Refused as exc:
        name_attempt = str(exc)
    # registration attempt 2: a distinct name; the catalog's overlap check holds it for review
    threshold = 0.25
    held = ov[0]["jaccard"] >= threshold
    review = {"skill": "report-timezone-fixer", "owner": "data-platform", "held_for_review": held,
              "threshold": threshold, "closest": ov[0],
              "suite_without": f"{before['passed']}/{before['total']}",
              "suite_with": f"{with_fork['passed']}/{with_fork['total']}",
              "newly_failing": sorted(set(with_fork["failed"]) - set(before["failed"])),
              "notified": ["reporting-platform", "data-platform"]}
    if held:
        cat.notify(210, "reporting-platform", "overlap: report-timezone-fixer vs report-helper",
                   json.dumps(ov[0]))
        cat.notify(210, "data-platform", "overlap: report-timezone-fixer vs report-helper", json.dumps(ov[0]))
        cat.depend("csv-cleaner", "data-platform", "skill", "report-helper", "1.0.0")
    results = {"name_collision": name_attempt, "overlap_review": review,
               "rows_with_fork": [r for r in with_fork["rows"] if not r["pass"]],
               "resolution": "data-platform depends on report-helper@1.0.0; fork not released",
               "suite_after_resolution": f"{before['passed']}/{before['total']}"}
    write_run(out, "discovery.json", results)
    return results


def skill_content_dir(d):
    files = {rel: sk.sha256_bytes(data)[:12] for rel, data in sorted(sk.package_files(d).items())}
    meta, _ = sk.parse_frontmatter(open(os.path.join(d, "SKILL.md"), encoding="utf-8").read())
    return {"files": files, "declared_tools": (meta.get("allowed-tools") or "").split()}, sk.package_hash(d), meta


def exp_schema(out):
    """A paused run's events were written by an older build; another run's by a
    newer canary build. What the current build (store schema 2) does with each."""
    old = {"event_id": "E17", "v": 1, "type": "op.finished", "entity": "op-c10", "payload": {}}
    newer = {"event_id": "N1", "v": 3, "type": "op.superseded", "entity": "op-c11", "payload": {}}
    res = {"build_reads_up_to": st12.SCHEMA_VERSION, "old_event": old, "upcast": st12.upcast(old)}
    try:
        st12.upcast(newer)
        res["newer_event"] = "accepted"
    except st12.Refused as exc:
        res["newer_event"] = f"refused: {exc}"
    write_run(out, "schema_on_resume.json", res)
    return res


# ---------------------------------------------------------------------------
# 6. Governance queries
# ---------------------------------------------------------------------------

def governance_queries(catalog_db, runs_db, run_x, out):
    con = sqlite3.connect(catalog_db)
    con.row_factory = sqlite3.Row
    con.execute("ATTACH DATABASE ? AS rs", (runs_db,))
    Q = {}
    sql_owner = ("SELECT owner, version, status, status_reason FROM parts "
                 "WHERE kind='skill' AND name='report-helper' ORDER BY version")
    Q["who maintains report-helper"] = {"sql": sql_owner,
                                        "rows": [dict(r) for r in con.execute(sql_owner)]}
    sql_ran = ("SELECT r.run_id, r.manifest_id, r.slot, mp.kind, mp.name, mp.version, "
               "substr(mp.content_hash,1,12) AS hash FROM rs.runs r "
               "JOIN manifest_parts mp ON mp.manifest_id = r.manifest_id "
               "WHERE r.run_id = ? AND mp.kind IN ('skill','model','prompt','policy','runtime') "
               "ORDER BY mp.kind")
    Q[f"which versions ran in {run_x}"] = {"sql": sql_ran, "rows": [dict(r) for r in con.execute(sql_ran, (run_x,))]}
    sql_access = ("SELECT mp.kind, mp.name, p.permissions, p.data_scope FROM rs.runs r "
                  "JOIN manifest_parts mp ON mp.manifest_id = r.manifest_id "
                  "JOIN parts p ON p.kind=mp.kind AND p.name=mp.name AND p.version=mp.version "
                  "WHERE r.run_id = ? AND mp.kind IN ('policy','skill','model') ORDER BY mp.kind")
    rows = [dict(r) for r in con.execute(sql_access, (run_x,))]
    for r in rows:
        r["permissions"], r["data_scope"] = json.loads(r["permissions"]), json.loads(r["data_scope"])
    pol = next(r for r in rows if r["kind"] == "policy")["permissions"]["allowed_tools"]
    skl = next(r for r in rows if r["kind"] == "skill")["permissions"]["declared_tools"]
    tools = [r["name"] for r in con.execute(
        "SELECT mp.name FROM rs.runs r JOIN manifest_parts mp ON mp.manifest_id=r.manifest_id "
        "WHERE r.run_id=? AND mp.kind='tool' ORDER BY mp.name", (run_x,))]
    Q[f"what could {run_x} access"] = {"sql": sql_access, "rows": rows,
                                        "effective_tools": sorted(set(tools) & set(pol)),
                                        "skill_declared_not_in_manifest": sorted(set(skl) - set(tools))}
    sql_ev = ("SELECT e.kind, e.result, e.at, e.by FROM rs.runs r JOIN evidence e ON e.subject = r.manifest_id "
              "WHERE r.run_id = ?")
    ev = [dict(r) for r in con.execute(sql_ev, (run_x,))]
    for r in ev:
        r["result"] = json.loads(r["result"])
    Q[f"what evidence released {run_x}'s manifest"] = {"sql": sql_ev, "rows": ev}
    sql_wd = ("SELECT r.status, count(*) AS runs FROM rs.runs r JOIN manifest_parts mp "
              "ON mp.manifest_id = r.manifest_id JOIN parts p ON p.kind=mp.kind AND p.name=mp.name "
              "AND p.version=mp.version WHERE p.status='withdrawn' GROUP BY r.status ORDER BY r.status")
    Q["runs that used a withdrawn version"] = {"sql": sql_wd, "rows": [dict(r) for r in con.execute(sql_wd)]}
    sql_dep = ("SELECT consumer, consumer_owner, version FROM depends WHERE kind='tool' "
               "AND name='exports_create' ORDER BY consumer")
    Q["who depends on exports_create"] = {"sql": sql_dep, "rows": [dict(r) for r in con.execute(sql_dep)]}
    con.close()
    write_run(out, "queries.json", Q)
    return Q


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="runs/ch17")
    out = os.path.abspath(ap.parse_args().output)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    say("== 1 gate evidence: Chapter 10 scenarios run for real")
    gate = gate_runs(out)
    for v in ("1.0.0", "1.1.0"):
        failed = [gate[v + "_names"][c] for c, ok in gate[v].items() if not ok]
        say(f"  report-helper@{v}: {sum(gate[v].values())}/{len(gate[v])} behavior checks; failed={failed}; "
            f"final report.py {gate[v + '_after']['report.py']}, test_report.py {gate[v + '_after']['test_report.py']}")

    say("\n== 2 catalog and manifests")
    catalog_db = os.path.join(out, "catalog.db")
    cat, releases, m_initial = build_catalog(catalog_db, gate)
    n_parts = cat.db.execute("SELECT count(*) FROM parts").fetchone()[0]
    by_kind = dict(cat.db.execute("SELECT kind, count(*) FROM parts GROUP BY kind ORDER BY kind").fetchall())
    owners = [r[0] for r in cat.db.execute("SELECT DISTINCT owner FROM parts ORDER BY owner")]
    say(f"  parts: {n_parts} {by_kind}")
    say(f"  owners: {owners}")
    for agent, mid in m_initial.items():
        say(f"  {agent:18s} manifest {mid}  parts={len(releases[agent])}")
    m1 = m_initial["report-fixer"]
    shared = cat.dependents("tool", "read_file")
    say(f"  dependents of tool:read_file@1.0.0: {[d['consumer'] for d in shared]}")

    say("\n== 3 release of report-helper 1.1.0")
    content, h, meta = skill_content("report-helper", "1.1.0")
    cat.register("skill", "report-helper", "1.1.0", content, "reporting-platform", CANARY_AT - 2,
                 {"declared_tools": content["declared_tools"], "network": "none declared"},
                 {"reads": "workspace files named in the task"}, content_hash=h)
    refs2 = [r if r[:2] != ("skill", "report-helper") else ("skill", "report-helper", "1.1.0")
             for r in releases["report-fixer"]]
    m2 = cat.build_manifest("report-fixer", "report-helper 1.1.0", refs2, CANARY_AT - 1, "reporting-platform")
    g1 = gate_decision(gate, "1.1.0", "1.0.0", ["B1", "B2", "B4", "B6", "B7", "B8"])
    g1["suite"] = "report-fix-behavior@1.0.0"
    cat.add_evidence(m2, "gate", g1, CANARY_AT - 1, "reporting-platform")
    cat.point("staging", "report-fixer", CANARY_AT - 1, "reporting-platform", "gate allowed", active=m2)
    say(f"  manifest {m2} differs from {m1} in: "
        f"{[p for p in cat.manifest(m2)['parts'] if p not in cat.manifest(m1)['parts']]}")
    say(f"  gate with report-fix-behavior@1.0.0 (checks {list(g1['checks'])}): {g1['decision']}")
    for mid in (m1, m2):
        write_run(os.path.join(out, "manifests"), f"{mid}.json", cat.manifest(mid))
    cat.db.commit()

    say("\n== 4 fleet simulation (simulated time; identical arrivals for both lifecycles)")
    specs = run_specs()
    say(f"  runs: {len(specs)} over {DAYS} days; by agent "
        f"{ {a: sum(s['agent'] == a for s in specs) for a in AGENTS} }; paused: {sum(bool(s['pause']) for s in specs)}")
    inj = injected_specs(m2)
    specs = sorted(specs + inj, key=lambda s: s["start"])
    for s in inj:
        say(f"  injected: {s['run_id']} starts {L.hour_label(s['start'])}, pauses after step 3 for {s['pause']} h "
            f"({s['why']})")
    runs_db = os.path.join(out, "runs.db")
    if os.path.exists(runs_db):
        os.remove(runs_db)
    rcon, rs = simulate_revised(cat, specs, gate, m1, m2, runs_db)
    cat.db.commit()
    ins = [d for d in rs["decisions"] if d["decision"] == "insufficient_evidence"]
    say(f"  revised: canary at {L.hour_label(CANARY_AT)} (10%); monitor said insufficient_evidence for "
        f"{len(ins)} hours; alert {L.hour_label(rs['alert']['hour'])} on {rs['alert']['check']} "
        f"canary {rs['alert']['canary_fail']:.0%} (n={rs['alert']['canary_n']}) vs active "
        f"{rs['alert']['active_fail']:.0%} (n={rs['alert']['active_n']}); withdrawn {L.hour_label(rs['withdrawn_at'])}")
    rf_runs = rcon.execute("SELECT run_id, manifest_id, slot, status, admitted, ended FROM runs "
                           "WHERE agent='report-fixer'").fetchall()
    on_m2 = [r for r in rf_runs if r[1] == m2]
    status_m2 = {}
    for r in on_m2:
        status_m2[r[3]] = status_m2.get(r[3], 0) + 1
    after_w = [r for r in rf_runs if r[1] == m2 and r[4] >= rs["withdrawn_at"]]
    say(f"  revised: report-fixer runs {len(rf_runs)}; admitted on {m2}: {len(on_m2)} {status_m2}; "
        f"admitted on it after withdrawal: {len(after_w)}")
    say(f"  revised: halted at a step boundary: {len(rs['halted'])}; resume refused: {len(rs['refused_resume'])}; "
        f"replacement runs: {len(rs['replacements'])}")
    for x in rs["halted"][:3] + rs["refused_resume"][:2]:
        say(f"    {x}")
    mixed_revised = 0   # each run's manifest is fixed at admission; replacements are new runs
    say(f"  revised: runs with more than one skill version: {mixed_revised}")

    truth, records, latest_log, fs = simulate_flawed(specs, gate, CANARY_AT)
    used_11 = [rid for rid, t in truth.items() if "1.1.0" in (t["start_version"], t["resume_version"])]
    mixed = [rid for rid, t in truth.items() if t["resume_version"] and t["resume_version"] != t["start_version"]]
    stale = [rid for rid, t in truth.items() if t["start"] >= fs["rolled_back_at"] and t["start_version"] == "1.1.0"]
    stale_resume = [rid for rid, t in truth.items() if t["resume_version"] == "1.1.0"
                    and t["times"][t["spec"]["pause_after"]] >= fs["rolled_back_at"]]
    guess = reconstruct(records, latest_log)
    wrong = [rid for rid, t in truth.items()
             if t["resume_version"] is None and guess[rid] != t["start_version"]]
    wrong += [rid for rid in mixed]
    listed = [rid for rid, g in guess.items() if g == "1.1.0"]
    over = [rid for rid in listed if rid not in used_11]
    missed = [rid for rid in used_11 if rid not in listed]
    last_stale = max((truth[r]["start"] for r in stale), default=None)
    say(f"  flawed: published to latest {L.hour_label(CANARY_AT)}; alert {L.hour_label(fs['alert']['hour'])} "
        f"({fs['alert']['check']} {fs['alert']['after_fail']:.0%} after vs {fs['alert']['before_fail']:.0%} before, "
        f"n={fs['alert']['n_after']}); rolled back {L.hour_label(fs['rolled_back_at'])}")
    say(f"  flawed: runs that used 1.1.0: {len(used_11)}; started on 1.1.0 after rollback (stale worker cache): "
        f"{len(stale)}, last at {L.hour_label(last_stale) if last_stale else '-'}; resumed onto 1.1.0 after "
        f"rollback: {len(stale_resume)}")
    say(f"  flawed: runs with two skill versions (paused across a change): {len(mixed)} {mixed}")
    say(f"  flawed: record says {records[used_11[0]]['skill']!r} / {records[used_11[0]]['model']!r}; "
        f"version inferred from start time is wrong or incomplete for {len(wrong)} runs")
    say(f"  flawed: listing runs by 'latest at start time' names {len(listed)} runs as 1.1.0: "
        f"{len(listed) - len(over)} did use it, {len(over)} did not; it misses {len(missed)} that did")
    model_after = [rid for rid, t in truth.items() if "model-a-2026-09-10" in t["served_models"]]
    model_mixed = [rid for rid, t in truth.items() if len(t["served_models"]) > 1]
    say(f"  flawed: runs served by model-a-2026-09-10 after the alias moved: {len(model_after)}; "
        f"runs spanning both model revisions: {len(model_mixed)}")

    # hourly distribution for the figure
    spec_of = {s["run_id"]: s for s in specs}
    rev_int = []
    for r in rcon.execute("SELECT run_id, manifest_id, status, admitted, ended FROM runs WHERE agent='report-fixer'"):
        rid, mid, status, adm, end = r
        lab = "1.1.0" if mid == m2 else "1.0.0"
        sp = spec_of.get(rid)
        if sp is None or sp["pause_after"] is None:
            rev_int.append((adm, end, lab))
            continue
        times, _ = step_times(dict(sp, start=adm))
        pa = sp["pause_after"]
        pause_start = times[pa - 1] + sp["steps"][pa - 1]
        rev_int.append((adm, min(end, pause_start), lab))
        if status == "completed" and end > times[pa]:
            rev_int.append((times[pa], end, lab))
    fl_int = []
    for rid, t in truth.items():
        s = t["spec"]
        if s["pause_after"] is None:
            fl_int.append((t["start"], t["end"], t["start_version"]))
        else:
            pa = t["times"][s["pause_after"]]
            fl_int.append((t["start"], t["times"][s["pause_after"] - 1] + s["steps"][s["pause_after"] - 1],
                           t["start_version"]))
            fl_int.append((pa, t["end"], t["resume_version"]))
    timeline = {"hours": HOURS, "canary_at": CANARY_AT, "revised": in_flight_by_hour(rev_int),
                "revised_alert": rs["alert"]["hour"], "revised_withdrawn": rs["withdrawn_at"],
                "flawed": in_flight_by_hour(fl_int), "flawed_alert": fs["alert"]["hour"],
                "flawed_rolled_back": fs["rolled_back_at"], "flawed_last_stale_start": last_stale}
    write_run(out, "timeline.json", timeline)

    say("\n== 5 withdrawal: in-flight, paused and completed runs")
    rec = reconcile_check(gate, out)
    say(f"  reconcile check: 1.1.0's test_report.py {rec['test_report.py']} against pre-fix report.py "
        f"{rec['report.py']}: {rec['status']} {rec['failed_tests']}")
    completed_m2 = [r[0] for r in rf_runs if r[1] == m2 and r[3] == "completed"]
    req = {}
    for rid in completed_m2:
        rcon.execute("UPDATE runs SET flags=? WHERE run_id=?",
                     (json.dumps(["used withdrawn skill:report-helper@1.1.0", "evidence restored by reconcile_check"]),
                      rid))
        rq = rcon.execute("SELECT requester FROM runs WHERE run_id=?", (rid,)).fetchone()[0]
        req.setdefault(rq, []).append(rid)
    for rq, rids in sorted(req.items()):
        cat.notify(rs["withdrawn_at"], rq, "report-helper 1.1.0 withdrawn",
                   f"runs {', '.join(rids)} used a withdrawn procedure; the missing failing-test evidence was "
                   f"re-established ({rec['status']} on {rec['report.py']}). No action needed.")
    cat.notify(rs["withdrawn_at"], "reporting-platform", "report-helper 1.1.0 withdrawn",
               f"{len(completed_m2)} completed, {len(rs['halted'])} halted, {len(rs['refused_resume'])} paused runs")
    rcon.commit()
    cat.db.commit()
    n_not = cat.db.execute("SELECT count(*) FROM notices WHERE subject='report-helper 1.1.0 withdrawn'").fetchone()[0]
    say(f"  completed runs flagged: {len(completed_m2)}; requesters notified: {len(req)}; notices: {n_not}")
    ptr = cat.pointer("prod", "report-fixer")
    say(f"  prod pointer after withdrawal: active={ptr['active']} canary={ptr['canary']}")
    try:
        cat.build_manifest("report-fixer", "retry", refs2, rs["withdrawn_at"] + 1, "reporting-platform")
    except L.Refused as exc:
        say(f"  building a manifest with 1.1.0 again: refused: {exc}")
    cat.register("evaluator", "report-fix-behavior", "2.0.0",
                 {"checks": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"], "dataset": "report-fix-tasks@1",
                  "oracle": "run records"}, "reporting-platform", rs["withdrawn_at"] + 2)
    g2 = gate_decision(gate, "1.1.0", "1.0.0", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    g2["suite"] = "report-fix-behavior@2.0.0"
    cat.add_evidence("skill:report-helper@1.1.0", "gate", g2, rs["withdrawn_at"] + 2, "reporting-platform")
    say(f"  the same candidate under report-fix-behavior@2.0.0: {g2['decision']} failed={g2['failed']}")
    cat.db.commit()
    wd = {"revised": {"alert": rs["alert"], "withdrawn_at": rs["withdrawn_at"], "on_m2": status_m2,
                      "halted": rs["halted"], "refused_resume": rs["refused_resume"],
                      "replacements": rs["replacements"], "completed_flagged": completed_m2,
                      "requesters_notified": sorted(req), "gate_v1": g1, "gate_v2": g2,
                      "monitor_decisions": rs["decisions"]},
          "flawed": {"alert": fs["alert"], "rolled_back_at": fs["rolled_back_at"], "used_1_1_0": len(used_11),
                     "stale_starts_after_rollback": stale, "stale_resumes_after_rollback": stale_resume,
                     "mixed_runs": mixed, "misattributed_by_start_time": sorted(wrong),
                     "listed_by_start_time": sorted(listed), "listed_but_not_used": sorted(over),
                     "used_but_not_listed": sorted(missed),
                     "mixed_detail": {r: {k: truth[r][k] for k in ("start_version", "resume_version")} for r in mixed},
                     "model_after_alias_move": len(model_after), "model_mixed": len(model_mixed),
                     "latest_log": latest_log,
                     "sample_record": records[used_11[0]]}}
    write_run(out, "withdrawal.json", wd)
    write_run(out, "flawed_records.json", list(records.values())[:40])

    say("\n== 6a tool versioning: exports_create day -> report_date")
    tv = exp_tool_versioning(cat, out)
    for a in tv["attempts"]:
        say(f"  register exports_create@{a['version']} with the renamed field: {a['result']}"
            + (f": {a['message']}" if a["result"] == "refused" else f" {a['hash']}"))
    say(f"  dependents of exports_create@1.0.0 at deprecation: {len(tv['dependents_at_deprecation'])}; "
        f"owners notified: {tv['owners_notified']}")
    for k, v in tv["ch09_rerun"].items():
        say(f"  Chapter 9 rerun, {k}: {v}")
    for cp, rem in tv["remaining_on_1_0_0"].items():
        say(f"  still pinned to exports_create@1.0.0 on {cp}: {rem}")

    say("\n== 6b duplicate catalog entries")
    du = exp_duplicates(cat, out)
    say(f"  same name from another team: {du['name_collision']}")
    r = du["overlap_review"]
    say(f"  activation suite without the fork {r['suite_without']}, with it {r['suite_with']}; "
        f"newly failing {r['newly_failing']}")
    say(f"  closest entry {r['closest']['with']} jaccard={r['closest']['jaccard']} shared={r['closest']['shared']}; "
        f"held for review: {r['held_for_review']} (threshold {r['threshold']})")
    for row in du["rows_with_fork"]:
        if row["id"] in r["newly_failing"]:
            say(f"    {row['id']} expected={row['expected']} chosen={row['chosen']} contested={row['contested_with']}")
    say(f"  after resolution ({du['resolution']}): {du['suite_after_resolution']}")

    say("\n== 6c resuming under a different store schema (Chapter 12 upcaster)")
    sc = exp_schema(out)
    say(f"  build reads up to schema {sc['build_reads_up_to']}; old event {sc['old_event']['type']}@v1 -> "
        f"{sc['upcast']['type']}@v{sc['upcast']['v']}; newer event: {sc['newer_event']}")

    cat.db.commit()
    say("\n== 7 governance queries")
    run_x = sorted(r[0] for r in on_m2 if r[3] == "completed")[0]
    Q = governance_queries(catalog_db, runs_db, run_x, out)
    for title, q in Q.items():
        say(f"  -- {title}")
        for row in q["rows"]:
            say(f"     {row}")
        for k in ("effective_tools", "skill_declared_not_in_manifest"):
            if k in q:
                say(f"     {k}: {q[k]}")

    cat.db.commit()
    log = [dict(r) for r in cat.db.execute("SELECT * FROM log WHERE at > 0 ORDER BY at, rowid")]
    write_run(out, "catalog_log.json", log)
    say("\n== catalog log after hour 0")
    for e in log:
        say(f"  {L.hour_label(e['at'])} {e['event']:15s} {e['subject']:38s} {e['by']:20s} {e['detail']}")
    cat.db.close()
    rcon.close()


if __name__ == "__main__":
    main()
