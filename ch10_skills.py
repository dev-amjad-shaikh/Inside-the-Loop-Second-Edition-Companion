"""Chapter 10: skills.

A minimal skills client runs the reporting task from Chapter 3 with a skill
package. Everything that touches files, tests and scripts is real; the
model's responses are authored, and the choice of skill is made by a
deterministic lexical stand-in (itl.ch10_skills.LexicalActivator).

    python3 ch10_skills.py --output runs/ch10
    python3 ch10_skills.py --output runs/ch10-real --measure-catalog DIR [DIR ...]
    python3 ch10_skills.py --output runs/ch10-skillsref --skills-ref PATH/skills-ref/src

Experiments, each written to its own file under --output:
  1 packages.json      format validation and lint for every package version
  2 catalogs/          discovered catalogs and their size in the model input
  3 activation.json    16 activation cases against four catalog states
  4 run_flawed/        report-helper 0.1.0, explicitly activated
  5 run_v1_0/          report-helper 1.0.0, activated by the stand-in
  6 run_v1_1/          report-helper 1.1.0 (a changed procedure step)
  7 lifecycle.json     behavior tests, regression, withdrawal, rerun
  8 authority.json     pinned-script tampering and a strict-policy refusal
"""
import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from itl.core import ScriptedModel, Workspace, content_version, report_tools, write_run  # noqa: E402
from itl import ch10_skills as sk  # noqa: E402

VERSIONS = os.path.join(HERE, "fixtures", "ch10_skills", "versions")

SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: settings.json, report.py, test_report.py.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")

RUN_POLICY = sk.RunPolicy(
    allowed_tools=["read_file", "replace_text", "run_tests", "activate_skill",
                   "read_skill_resource", "run_skill_script"],
    network=False, require_pinned_scripts=False, policy_id="report-fix-no-deploy")

CASES = [
    # (id, request, expected skill or None)
    ("t1", TASK, "report-helper"),
    ("t2", "Orders from late Sunday evening show up in Monday's sales report. Can you fix the date filter?",
     "report-helper"),
    ("t3", "After the daylight-saving switch the weekly report counts some orders twice.", "report-helper"),
    ("t4", "belongs_to_report compares UTC dates; make it use the configured report timezone.",
     "report-helper"),
    ("t5", "Write the weekly status report for the platform team from these metrics.",
     "status-report-writer"),
    ("t6", "Draft a monthly summary of support tickets for leadership.", "status-report-writer"),
    ("t7", "Turn this sales data into a short daily report for the regional managers.",
     "status-report-writer"),
    ("t8", "Prepare release notes for version 2.4 from the merged pull requests.", "release-notes"),
    ("t9", "The exported orders CSV will not load because the header row is duplicated.", "csv-cleanup"),
    ("n1", "Report a bug to the payments team about the refund button.", None),
    ("n2", "What timezone is the Los Angeles office in?", None),
    ("n3", "Schedule the daily report job to run at 6am instead of 5am.", None),
    ("n4", "Explain how the report data pipeline is deployed.", None),
    ("n5", "Fix the failing login test in auth_test.py.", None),
    ("n6", "Deploy the report fix to production.", None),
    ("n7", "Summarize yesterday's incident report in three bullet points.", None),
]

STATES = {
    "A_flawed": [("report-helper", "0.1.0"), ("status-report-writer", "1.0.0"),
                 ("release-notes", "1.0.0"), ("csv-cleanup", "1.0.0")],
    "B_helper_revised": [("report-helper", "1.0.0"), ("status-report-writer", "1.0.0"),
                         ("release-notes", "1.0.0"), ("csv-cleanup", "1.0.0")],
    "C_both_revised": [("report-helper", "1.0.0"), ("status-report-writer", "1.1.0"),
                       ("release-notes", "1.0.0"), ("csv-cleanup", "1.0.0")],
    "D_helper_1_1": [("report-helper", "1.1.0"), ("status-report-writer", "1.1.0"),
                     ("release-notes", "1.0.0"), ("csv-cleanup", "1.0.0")],
}

NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''
OLD_LINE = "    return created_at.date() == requested_day"
NEW_LINES = ("    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
             "    return local_time.date() == requested_day")
PROBE_ARGS = ["report.py", "America/Los_Angeles", "2026-09-11", "2026-11-01"]


def src(name, version):
    return os.path.join(VERSIONS, name, version, name)


def materialize_versions(pairs, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    for name, version in pairs:
        shutil.copytree(src(name, version), os.path.join(dest, name))
    return dest


def call(cid, tool, **arguments):
    return {"id": cid, "name": tool, "arguments": arguments}


def add_test(cid):
    return call(cid, "replace_text", path="test_report.py", old='\n\nif __name__ == "__main__":',
                new="\n" + NEW_TEST)


def fix(cid):
    return call(cid, "replace_text", path="report.py", old=OLD_LINE, new=NEW_LINES)


def reads(start):
    return [call(f"c{start}", "read_file", path="settings.json"),
            call(f"c{start + 1}", "read_file", path="report.py"),
            call(f"c{start + 2}", "read_file", path="test_report.py")]


def turn(*calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


ANSWER_V10 = ("Cause: belongs_to_report took the date of the UTC timestamp, so an order at 00:30 UTC "
              "on September 12 (17:30 on September 11 in America/Los_Angeles) missed the September 11 "
              "report. I added test_order_near_midnight; it failed before the change. I converted the "
              "timestamp to the configured timezone before comparing dates, keeping the signature. The "
              "boundary probe passed for 2026-09-11 and 2026-11-01, and both tests pass on the edited "
              "report.py. Nothing was deployed.")
ANSWER_V11 = ("I converted the timestamp to the configured timezone before comparing dates and added "
              "test_order_near_midnight. Both tests pass and the boundary probe passed. Nothing was "
              "deployed.")
ANSWER_FLAWED = ("I fixed the report: timestamps are now converted to the report timezone before the "
                 "date comparison, and the tests pass. The fix script could not download its rules, "
                 "and the deployment step was refused.")


def responses_v10(skill):
    return [
        turn(call("c1", "activate_skill", name=skill)),
        turn(call("c2", "read_skill_resource", skill=skill, path="references/regression-test.md"),
             *reads(3)),
        turn(add_test("c6")),
        turn(call("c7", "run_tests", tests=["test_report.py"])),
        turn(fix("c8")),
        turn(call("c9", "run_skill_script", skill=skill, script="scripts/probe_boundary.py",
                  args=PROBE_ARGS)),
        turn(call("c10", "run_tests", tests=["test_report.py"])),
        {"role": "assistant", "content": ANSWER_V10, "tool_calls": []},
    ]


def responses_v11(skill):
    return [
        turn(call("c1", "activate_skill", name=skill)),
        turn(call("c2", "read_skill_resource", skill=skill, path="references/regression-test.md"),
             *reads(3)),
        turn(fix("c6")),
        turn(add_test("c7")),
        turn(call("c8", "run_tests", tests=["test_report.py"])),
        turn(call("c9", "run_skill_script", skill=skill, script="scripts/probe_boundary.py",
                  args=PROBE_ARGS)),
        {"role": "assistant", "content": ANSWER_V11, "tool_calls": []},
    ]


def responses_flawed(skill):
    return [
        turn(call("c1", "run_skill_script", skill=skill, script="scripts/fix_report.py", args=["report.py"])),
        turn(call("c2", "read_file", path="report.py"), call("c3", "read_file", path="settings.json")),
        turn(fix("c4")),
        turn(call("c5", "run_tests", tests=["test_report.py"])),
        turn(call("c6", "deploy_report", target="production")),
        {"role": "assistant", "content": ANSWER_FLAWED, "tool_calls": []},
    ]


# ---------------------------------------------------------------------------
# Behavior tests: what the reporting fix must show, read from the records
# ---------------------------------------------------------------------------

def behavior_tests(result):
    msgs, before, after = result["messages"], result["before"], result["after"]
    by_id = {}
    order = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                by_id[c["id"]] = c
        if m["role"] == "tool":
            order.append((i, by_id[m["tool_call_id"]], m["content"]))
    test_edit = [i for i, c, o in order if c["name"] == "replace_text" and c["arguments"]["path"] ==
                 "test_report.py" and o.get("status") == "ok"]
    src_edit = [i for i, c, o in order if c["name"] == "replace_text" and c["arguments"]["path"] ==
                "report.py" and o.get("status") == "ok"]
    runs = [(i, o) for i, c, o in order if c["name"] == "run_tests"]
    probes = [(i, o) for i, c, o in order if c["name"] == "run_skill_script" and
              c["arguments"]["script"].endswith("probe_boundary.py")]
    proposed = [c["name"] for c in by_id.values()]
    fail_first = bool(test_edit and src_edit and any(
        test_edit[0] < i < src_edit[0] and o.get("status") == "failed" for i, o in runs))
    last_run = runs[-1] if runs else None
    final_pass = bool(last_run and src_edit and last_run[0] > src_edit[-1] and
                      last_run[1].get("status") == "passed")
    probe_ok = bool(probes and src_edit and probes[-1][0] > src_edit[-1] and probes[-1][1].get("status") == "ok")
    sig = "def belongs_to_report(created_at, requested_day, timezone):"
    denied = [d for s in result["scripts"] for d in s.get("denied_effects", [])]
    checks = [
        ("B1 skill version recorded in run manifest",
         all(a.get("version") and a.get("hash") for a in result["manifest"]["activated"])
         and bool(result["manifest"]["activated"])),
        ("B2 regression test added", after["test_report.py"].count("def test_") >
         before["test_report.py"].count("def test_")),
        ("B3 new test failed before the fix", fail_first),
        ("B4 all tests passed after the last edit", final_pass),
        ("B5 boundary probe passed on the final revision", probe_ok),
        ("B6 public signature unchanged", sig in after["report.py"]),
        ("B7 no deployment proposed", "deploy_report" not in proposed),
        ("B8 no effect denied by the guard", not denied),
    ]
    return [{"check": n, "pass": bool(ok)} for n, ok in checks]


def run_scenario(name, catalog_dir, responses_fn, out, explicit=None, stand_in_request=TASK):
    records, diags = sk.discover(catalog_dir)
    catalog_text = sk.render_catalog(records)
    ws = Workspace("report")
    before = ws.snapshot()
    tools = report_tools(ws)
    ex = sk.SkillExecutor(tools, RUN_POLICY, records, ws)
    # the model sees only tools the run permits, plus the skill tools
    defs = [d for d in tools.definitions() if d["name"] in RUN_POLICY.allowed_tools] + \
        sk.skill_tool_definitions(records)
    messages = [{"role": "system", "content": SYSTEM + "\n\n" + catalog_text}]
    stand_in = sk.LexicalActivator(records).choose(stand_in_request)
    if explicit:
        messages.append({"role": "user", "content": f"/{explicit} {TASK}"})
        act = ex.activate(explicit, mode="user-explicit")
        messages.append({"role": "system", "content": act["content"]})
        skill = explicit
    else:
        messages.append({"role": "user", "content": TASK})
        skill = stand_in["chosen"]
        if skill is None:
            raise SystemExit(f"{name}: stand-in chose no skill")
    model = ScriptedModel(responses_fn(skill))
    answer = sk.run_skill_agent(model, ex, defs, messages)
    after = ws.snapshot()
    manifest = {
        "run_id": name, "task": TASK, "policy": RUN_POLICY.describe(),
        "catalog": [{"name": r["name"], "version": r["version"], "hash": r["hash"][:12]} for r in records],
        "catalog_tokens": sk.approx_tokens(catalog_text),
        "activated": [{"name": d["skill"], "version": d["version"], "hash": d["hash"], "mode": d["mode"]}
                      for d in ex.decisions if d.get("event") == "activate"],
        "resources_loaded": ex.loaded_resources,
        "scripts": ex.scripts,
        "stand_in_choice": {"chosen": stand_in["chosen"], "contested_with": stand_in["contested_with"]},
    }
    result = {"messages": messages, "before": before, "after": after, "scripts": ex.scripts,
              "manifest": manifest}
    tests = behavior_tests(result)
    d = os.path.join(out, name)
    write_run(d, "manifest.json", manifest)
    write_run(d, "trace.json", model.requests)
    write_run(d, "messages.json", messages)
    write_run(d, "decisions.json", ex.decisions)
    write_run(d, "input_sizes.json", sk.input_accounting(model))
    write_run(d, "behavior_tests.json", tests)
    write_run(d, "answer.txt", answer + "\n")
    for label, snap in (("before", before), ("after", after)):
        for fname, text in snap.items():
            write_run(os.path.join(d, label), fname, text)
    ws.close()
    return {"manifest": manifest, "tests": tests, "decisions": ex.decisions, "diagnostics": diags,
            "after": after,
            "input_sizes": sk.input_accounting(model), "answer": answer, "calls": len(model.requests)}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp_packages(out):
    rows = []
    for name in sorted(os.listdir(VERSIONS)):
        for version in sorted(os.listdir(os.path.join(VERSIONS, name))):
            d = src(name, version)
            lint = sk.lint_skill(d)
            meta, body = sk.parse_frontmatter(open(os.path.join(d, "SKILL.md")).read())
            rows.append({"skill": name, "version": version, "hash": sk.package_hash(d)[:12],
                         "format_errors": sk.validate_skill(d),
                         "lint_failed": [f["check"] for f in lint if not f["ok"]],
                         "lint": lint,
                         "description_chars": len(" ".join(meta["description"].split())),
                         "body_tokens": sk.approx_tokens(body),
                         "files": sorted(sk.package_files(d))})
    write_run(out, "packages.json", rows)
    return rows


def exp_activation(out):
    results = {"cases": [{"id": c[0], "request": c[1], "expected": c[2]} for c in CASES], "states": {}}
    catalogs = {}
    for state, pairs in STATES.items():
        cdir = materialize_versions(pairs, os.path.join(out, "catalogs", state))
        records, diags = sk.discover(cdir)
        text = sk.render_catalog(records)
        write_run(os.path.join(out, "catalogs"), f"{state}.catalog.txt", text + "\n")
        act = sk.LexicalActivator(records)
        rows = []
        for cid, req, expected in CASES:
            ch = act.choose(req)
            ok = ch["chosen"] == expected and not ch["contested_with"]
            rows.append({"id": cid, "expected": expected, "chosen": ch["chosen"],
                         "contested_with": ch["contested_with"], "pass": ok,
                         "scores": {s["skill"]: s["score"] for s in ch["ranking"]},
                         "matched": {s["skill"]: s["matched"] for s in ch["ranking"] if s["matched"]},
                         "against": {s["skill"]: s["against"] for s in ch["ranking"] if s["against"]}})
        # catalog-level overlap: Jaccard similarity of positive description terms
        overlap = []
        for i in range(len(act.index)):
            for j in range(i + 1, len(act.index)):
                a, b = act.index[i][1], act.index[j][1]
                jac = len(a & b) / len(a | b)
                overlap.append({"pair": [act.index[i][0], act.index[j][0]], "jaccard": round(jac, 3),
                                "shared": sorted(a & b)})
        overlap.sort(key=lambda x: -x["jaccard"])
        catalogs[state] = {"skills": len(records), "chars": len(text), "tokens": sk.approx_tokens(text),
                           "per_skill_tokens": [sk.approx_tokens(f"<skill><name>{r['name']}</name>"
                                                                 f"<description>{r['description']}</description></skill>")
                                                for r in records]}
        results["states"][state] = {
            "versions": {n: v for n, v in pairs}, "rows": rows,
            "passed": sum(r["pass"] for r in rows), "total": len(rows),
            "wrong_skill": [r["id"] for r in rows if r["chosen"] != r["expected"]],
            "contested": [r["id"] for r in rows if r["contested_with"]],
            "top_overlap": overlap[:3], "catalog": catalogs[state]}
    write_run(out, "activation.json", results)
    return results


def exp_lifecycle(out):
    reg = sk.Registry()
    reg.publish("report-helper", "0.1.0", src("report-helper", "0.1.0"), "reporting-platform", "2026-08-03")
    for n, v, m in (("status-report-writer", "1.0.0", "team-communications"),
                    ("release-notes", "1.0.0", "developer-experience"),
                    ("csv-cleanup", "1.0.0", "data-platform")):
        reg.publish(n, v, src(n, v), m, "2026-08-03")
    names = ["report-helper", "status-report-writer", "release-notes", "csv-cleanup"]

    # the flawed version, explicitly activated by the user
    flawed_dir = sk.materialize(reg, names, os.path.join(out, "skills_dirs", "flawed"))
    flawed = run_scenario("run_flawed", flawed_dir, responses_flawed, out, explicit="report-helper")

    # the revised version, released with its test evidence
    # after the overlap review, both owners revise their descriptions
    reg.publish("report-helper", "1.0.0", src("report-helper", "1.0.0"), "reporting-platform", "2026-09-14")
    reg.publish("status-report-writer", "1.1.0", src("status-report-writer", "1.1.0"), "team-communications",
                "2026-09-14")
    for n in ("report-helper", "status-report-writer"):
        reg.skills[n]["versions"][0]["status"] = "superseded"
        reg.log.append({"at": "2026-09-14", "event": "superseded", "skill": n,
                        "version": reg.skills[n]["versions"][0]["version"]})
    v10_dir = sk.materialize(reg, names, os.path.join(out, "skills_dirs", "v1_0"))
    v10 = run_scenario("run_v1_0", v10_dir, responses_v10, out)
    reg.skills["report-helper"]["versions"][-1]["evidence"] = {
        "behavior_tests": f"{sum(t['pass'] for t in v10['tests'])}/{len(v10['tests'])}", "run": "run_v1_0"}

    # 1.1.0 changes one procedure step; its release is checked by the same tests
    reg.publish("report-helper", "1.1.0", src("report-helper", "1.1.0"), "reporting-platform", "2026-09-21")
    v11_dir = sk.materialize(reg, names, os.path.join(out, "skills_dirs", "v1_1"))
    v11 = run_scenario("run_v1_1", v11_dir, responses_v11, out)
    failed = [t["check"] for t in v11["tests"] if not t["pass"]]
    reg.skills["report-helper"]["versions"][-1]["evidence"] = {
        "behavior_tests": f"{sum(t['pass'] for t in v11['tests'])}/{len(v11['tests'])}", "run": "run_v1_1",
        "failed": failed}
    if failed:
        reg.withdraw("report-helper", "1.1.0", "2026-09-22", "reporting-platform",
                     "procedure no longer requires the regression test to fail before the fix: " + failed[0])
    runs = {"run_flawed": flawed["manifest"], "run_v1_0": v10["manifest"], "run_v1_1": v11["manifest"]}
    affected = [rid for rid, m in runs.items() for a in m["activated"]
                if any(v["status"] == "withdrawn" and v["version"] == a["version"]
                       for v in reg.skills.get(a["name"], {}).get("versions", []))]
    resolved = reg.resolve("report-helper")
    after_dir = sk.materialize(reg, names, os.path.join(out, "skills_dirs", "after_withdrawal"))
    rerun = run_scenario("run_after_withdrawal", after_dir, responses_v10, out)
    lifecycle = {
        "registry": reg.snapshot(), "log": reg.log,
        "runs": {rid: {"activated": m["activated"],
                       "tests_passed": f"{sum(t['pass'] for t in r['tests'])}/{len(r['tests'])}",
                       "failed": [t["check"] for t in r["tests"] if not t["pass"]]}
                 for rid, m, r in (("run_flawed", flawed["manifest"], flawed),
                                   ("run_v1_0", v10["manifest"], v10),
                                   ("run_v1_1", v11["manifest"], v11),
                                   ("run_after_withdrawal", rerun["manifest"], rerun))},
        "runs_that_used_withdrawn_versions": affected,
        "final_files_v1_0_vs_v1_1": {f: [content_version(v10["after"][f]), content_version(v11["after"][f])]
                                     for f in sorted(v10["after"])},
        "resolved_after_withdrawal": {"version": resolved["version"], "hash": resolved["hash"][:12]},
    }
    write_run(out, "lifecycle.json", lifecycle)
    return lifecycle, {"run_flawed": flawed, "run_v1_0": v10, "run_v1_1": v11, "run_after_withdrawal": rerun}


def exp_authority(out):
    """Two refusals that need no model: a tampered pinned script, and an unpinned script under a strict policy."""
    results = {}
    # 1. tamper with the pinned probe script after release
    d = materialize_versions([("report-helper", "1.0.0")], os.path.join(out, "skills_dirs", "tampered"))
    with open(os.path.join(d, "report-helper", "scripts", "probe_boundary.py"), "a") as f:
        f.write("\nimport urllib.request  # added after release\n")
    records, _ = sk.discover(d)
    ws = Workspace("report")
    ex = sk.SkillExecutor(report_tools(ws), RUN_POLICY, records, ws)
    ex.activate("report-helper", mode="user-explicit")
    results["tampered_pinned_script"] = ex.execute(call("x1", "run_skill_script", skill="report-helper",
                                                        script="scripts/probe_boundary.py", args=PROBE_ARGS))
    results["tampered_record"] = ex.scripts[-1]
    ws.close()
    # 2. the flawed script under a policy that requires pinned scripts
    strict = sk.RunPolicy(RUN_POLICY.allowed_tools, require_pinned_scripts=True, policy_id="strict-pinned")
    d = materialize_versions([("report-helper", "0.1.0")], os.path.join(out, "skills_dirs", "strict"))
    records, _ = sk.discover(d)
    ws = Workspace("report")
    ex = sk.SkillExecutor(report_tools(ws), strict, records, ws)
    ex.activate("report-helper", mode="user-explicit")
    results["unpinned_under_strict_policy"] = ex.execute(call("x2", "run_skill_script", skill="report-helper",
                                                              script="scripts/fix_report.py",
                                                              args=["report.py"]))
    results["strict_record"] = ex.scripts[-1]
    ws.close()
    write_run(out, "authority.json", results)
    return results


def measure_catalogs(dirs, out):
    rows = []
    for root in dirs:
        records, diags = sk.discover(root)
        text = sk.render_catalog(records)
        entries = []
        for entry in sorted(os.listdir(root)):
            d = os.path.join(root, entry)
            if os.path.isfile(os.path.join(d, "SKILL.md")):
                entries.append({"skill": entry, "format_errors": sk.validate_skill(d)})
        act = sk.LexicalActivator(records)
        pairs = []
        for i in range(len(act.index)):
            for j in range(i + 1, len(act.index)):
                a, b = act.index[i][1], act.index[j][1]
                pairs.append({"pair": [act.index[i][0], act.index[j][0]],
                              "jaccard": round(len(a & b) / len(a | b), 3), "shared": sorted(a & b)})
        pairs.sort(key=lambda x: -x["jaccard"])
        desc_chars = [len(r["description"]) for r in records]
        rows.append({"root": root, "skill_dirs": len(entries), "parsed": len(records),
                     "diagnostics": diags, "catalog_chars": len(text),
                     "catalog_tokens": sk.approx_tokens(text),
                     "description_chars": {"min": min(desc_chars), "max": max(desc_chars),
                                           "mean": round(sum(desc_chars) / len(desc_chars))} if desc_chars else {},
                     "format": entries, "top_overlap": pairs[:5]})
        print(f"{root}: {len(records)}/{len(entries)} parsed, catalog ~{sk.approx_tokens(text)} tokens, "
              f"format errors in {sum(1 for e in entries if e['format_errors'])}")
        for e in entries:
            if e["format_errors"]:
                print(f"   {e['skill']}: {e['format_errors']}")
        for dg in diags:
            print(f"   diagnostic {dg}")
        for p in pairs[:3]:
            print(f"   overlap {p['pair']} jaccard={p['jaccard']} shared={p['shared']}")
    write_run(out, "catalog_measurements.json", rows)


def skills_ref_crosscheck(path, out, extra_roots=()):
    """Run the pinned skills-ref validator (needs strictyaml) beside the reimplementation."""
    sys.path.insert(0, path)
    from pathlib import Path
    from skills_ref.validator import validate  # requires strictyaml
    dirs = [src(n, v) for n in sorted(os.listdir(VERSIONS)) for v in sorted(os.listdir(os.path.join(VERSIONS, n)))]
    for root in extra_roots:
        dirs += [os.path.join(root, e) for e in sorted(os.listdir(root))
                 if os.path.isfile(os.path.join(root, e, "SKILL.md"))]
    rows, agree = [], 0
    for d in dirs:
        ref, mine = validate(Path(d)), sk.validate_skill(d)
        agree += bool(ref) == bool(mine)
        rows.append({"dir": d, "skills_ref": ref, "reimplementation": mine})
        print(f"{os.path.relpath(d, os.path.dirname(os.path.dirname(d)))}: skills-ref={ref} "
              f"reimplementation={mine}")
    print(f"pass/fail agreement: {agree}/{len(dirs)}")
    write_run(out, "skills_ref_crosscheck.json", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--measure-catalog", nargs="+")
    ap.add_argument("--skills-ref")
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit(f"{args.output} exists; choose a new directory")
    out = args.output
    os.makedirs(out)
    if args.skills_ref:
        skills_ref_crosscheck(args.skills_ref, out, args.measure_catalog or ())
        return
    if args.measure_catalog:
        measure_catalogs(args.measure_catalog, out)
        return

    pk = exp_packages(out)
    print("== 1 packages: format errors (skills-ref checks) and lint failures")
    for r in pk:
        print(f"  {r['skill']:21s} {r['version']}  {r['hash']}  format_errors={len(r['format_errors'])}  "
              f"lint_failed={len(r['lint_failed'])} {r['lint_failed']}")

    act = exp_activation(out)
    print("== 2 catalogs in the model input (approx tokens = chars/4)")
    for s, v in act["states"].items():
        c = v["catalog"]
        print(f"  {s:18s} skills={c['skills']} chars={c['chars']} tokens={c['tokens']} "
              f"per_skill={c['per_skill_tokens']}")
    print("== 3 activation (lexical stand-in), 16 cases")
    for s, v in act["states"].items():
        print(f"  {s:18s} passed {v['passed']}/{v['total']}  wrong={v['wrong_skill']}  "
              f"contested={v['contested']}")
        top = v["top_overlap"][0]
        print(f"  {'':18s} top overlap {top['pair']} jaccard={top['jaccard']} shared={top['shared']}")
    for s in ("A_flawed", "C_both_revised"):
        for r in act["states"][s]["rows"]:
            print(f"    {s[:1]} {r['id']:3s} expected={str(r['expected']):21s} chosen={str(r['chosen']):21s} "
                  f"contested={r['contested_with']} scores={r['scores']}")

    lc, runs = exp_lifecycle(out)
    print("== 4-7 runs, behavior tests and lifecycle")
    for rid, r in runs.items():
        print(f"  {rid:22s} calls={r['calls']} activated={r['manifest']['activated']}")
        print(f"  {'':22s} tests {sum(t['pass'] for t in r['tests'])}/{len(r['tests'])} "
              f"failed={[t['check'] for t in r['tests'] if not t['pass']]}")
    for rid in ("run_flawed", "run_v1_0"):
        for d in runs[rid]["decisions"]:
            if d.get("decision") in ("refused",) or d.get("event") == "activate":
                print(f"  {rid} decision: {d}")
        for s in runs[rid]["manifest"]["scripts"]:
            print(f"  {rid} script: {s}")
        print(f"  {rid} input sizes: {runs[rid]['input_sizes']}")
        print(f"  {rid} resources loaded: {runs[rid]['manifest']['resources_loaded']}")
    print("  registry log:")
    for e in lc["log"]:
        print(f"    {e}")
    print(f"  runs that used withdrawn versions: {lc['runs_that_used_withdrawn_versions']}")
    print(f"  resolved after withdrawal: {lc['resolved_after_withdrawal']}")
    print(f"  final file versions, v1.0 run vs v1.1 run: {lc['final_files_v1_0_vs_v1_1']}")

    au = exp_authority(out)
    print("== 8 authority checks without a model")
    print(f"  tampered pinned script: {au['tampered_pinned_script']} {au['tampered_record']}")
    print(f"  unpinned script, strict policy: {au['unpinned_under_strict_policy']} {au['strict_record']}")


if __name__ == "__main__":
    main()
