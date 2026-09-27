"""Chapter 6: building the next model input.

Replays the Chapter 3 run, then continues it: the user adds task revision 2
("Keep the public function signature unchanged ...") and a long test log arrives.
The same scripted session runs under four context policies:

    full        every record verbatim (Chapter 3's loop)
    truncate    drop the oldest message until the request fits
    freetext    protected head + first-sentence summary + last 4 messages
    structured  pinned task and constraints, derived summary with record references,
                whole request/result groups, digests for oversized results

File reads, edits and test runs are real. The model is a scripted stand-in: its
responses are fixed, except at the proposal after the log, where it returns the
signature-preserving edit only if the request contains the text of constraint C2.
That rule is a labeled stand-in for a model that follows constraints it can see.

    python3 ch06_context.py --output runs/ch06
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import ScriptedModel, Workspace, content_version, report_tools, write_run  # noqa: E402
from itl import ch06_context as cx  # noqa: E402
import ch03_chapter_lab as ch3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ID, SCOPE = "run-41", "workspace-42"

TASK1_GOAL = ("The daily report is missing some orders placed near midnight. Find the cause, "
              "prepare a fix and run the tests.")
TASK2_TEXT = ("Please keep the diff small. Keep the public function signature unchanged; the billing "
              "service calls belongs_to_report directly. Also make the new timezone matrix in "
              "test_report_matrix.py pass.")
C2_TEXT = "Keep the public function signature unchanged; the billing service calls belongs_to_report directly."
C2_PHRASE = "keep the public function signature unchanged"
SUITE = ["test_report.py", "test_report_matrix.py", "test_billing.py"]


def call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


# ----------------------------------------------------------------------------------
# Tools: Chapter 3's three, plus run_suite (verbose log) and read_record (just-in-time)
# ----------------------------------------------------------------------------------

def session_tools(ws, store):
    reg = report_tools(ws)

    def run_suite(tests):
        modules = [t[:-3] if t.endswith(".py") else t for t in tests]
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", *modules], cwd=ws.root,
                              capture_output=True, text=True, timeout=120)
        log = proc.stderr.replace(os.path.realpath(ws.root) + os.sep, "").replace(ws.root + os.sep, "")
        lines = []
        run = 0
        for line in log.splitlines():
            if line.startswith("Ran "):
                run = int(line.split()[1])
                line = "Ran %d tests" % run
            lines.append(line)
        failures = sum(1 for l in lines if l.startswith("FAIL: "))
        errors = sum(1 for l in lines if l.startswith("ERROR: "))
        failed = [l.split("(", 1)[1].rstrip(")") for l in lines if l.startswith(("FAIL: ", "ERROR: "))]
        return {"status": "passed" if proc.returncode == 0 else "failed", "tests_run": run,
                "failures": failures, "errors": errors, "failed_tests": failed,
                "exit_code": proc.returncode, "log": "\n".join(lines)}

    def read_record(record_id, start_line=1, end_line=40):
        rec = store.get(record_id)
        if rec is None or rec["scope"] != store.scope:
            return {"status": "refused", "message": "no record %s in this run's scope" % record_id}
        content = rec["message"]["content"]
        text = content.get("log") or content.get("text") if isinstance(content, dict) else str(content)
        lines = (text or "").splitlines()
        return {"status": "ok", "record_id": record_id, "lines": lines[start_line - 1:end_line],
                "start_line": start_line, "total_lines": len(lines)}

    reg.register("run_suite", "Run test modules verbosely and return counts and the full log.", {
        "type": "object", "properties": {"tests": {"type": "array", "items": {"type": "string"}}},
        "required": ["tests"], "additionalProperties": False}, run_suite)
    reg.register("read_record", "Read lines of a record in this run's execution record, such as a full "
                 "test log shown only as a digest.", {
                     "type": "object", "properties": {"record_id": {"type": "string"},
                                                      "start_line": {"type": "integer"},
                                                      "end_line": {"type": "integer"}},
                     "required": ["record_id"], "additionalProperties": False}, read_record)
    return reg


# ----------------------------------------------------------------------------------
# The scripted session (Chapter 3's seven responses, then the continuation)
# ----------------------------------------------------------------------------------

VIOLATING = [
    call("c12", "replace_text", path="report.py", old="from zoneinfo import ZoneInfo",
         new="import json\nfrom zoneinfo import ZoneInfo"),
    call("c13", "replace_text", path="report.py",
         old='def belongs_to_report(created_at, requested_day, timezone):\n'
             '    """created_at is an aware UTC datetime; requested_day is a local date."""\n',
         new='def belongs_to_report(created_at, requested_day, settings_path="settings.json", timezone=None):\n'
             '    """created_at is an aware UTC datetime; requested_day is a local date."""\n'
             '    if timezone is None:\n'
             '        with open(settings_path, encoding="utf-8") as f:\n'
             '            timezone = json.load(f)["report_timezone"]\n'),
]

COMPLIANT = [
    call("c12", "replace_text", path="report.py", old="from zoneinfo import ZoneInfo",
         new="import json\nfrom zoneinfo import ZoneInfo"),
    call("c13", "replace_text", path="report.py", old="    return local_time.date() == requested_day\n",
         new="    return local_time.date() == requested_day\n\n\n"
             'def belongs_to_configured_report(created_at, requested_day, settings_path="settings.json"):\n'
             '    """Like belongs_to_report, using the configured report timezone."""\n'
             '    with open(settings_path, encoding="utf-8") as f:\n'
             '        timezone = json.load(f)["report_timezone"]\n'
             '    return belongs_to_report(created_at, requested_day, timezone)\n'),
    call("c14", "replace_text", path="test_report_matrix.py", old="from report import belongs_to_report\n",
         new="from report import belongs_to_configured_report, belongs_to_report\n"),
    call("c15", "replace_text", path="test_report_matrix.py",
         old="        self.assertTrue(belongs_to_report(order_time, local_day))",
         new="        self.assertTrue(belongs_to_configured_report(order_time, local_day))"),
]

ANSWER_VIOLATING = ("belongs_to_report now takes an optional settings_path and reads the configured "
                    "timezone when no timezone is passed, so the matrix's default case works. "
                    "I ran the full suite. Nothing was deployed.")
ANSWER_COMPLIANT = ("I kept belongs_to_report(created_at, requested_day, timezone) unchanged and added "
                    "belongs_to_configured_report, which reads settings.json and delegates to it. "
                    "test_configured_default now calls the new function; that is the only test I changed. "
                    "The full suite passes. Nothing was deployed.")


class StandInModel(ScriptedModel):
    """ScriptedModel whose entries may be functions of the request. Only the proposal
    after the long log depends on the input: the C2 text present -> compliant edit."""

    def __init__(self, responses):
        super().__init__(responses)
        self.path = None

    def generate(self, messages, definitions):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": [d["name"] for d in definitions]})
        nxt = self.responses.pop(0)
        return copy.deepcopy(nxt(self, messages) if callable(nxt) else nxt)


def proposal(model, messages):
    seen = C2_PHRASE in json.dumps(messages, ensure_ascii=False).lower()
    model.path = "compliant" if seen else "violating"
    return {"role": "assistant", "content": "", "tool_calls": COMPLIANT if seen else VIOLATING}


def rerun_suite(model, messages):
    cid = "c16" if model.path == "compliant" else "c14"
    return {"role": "assistant", "content": "", "tool_calls": [call(cid, "run_suite", tests=SUITE)]}


def final(model, messages):
    return {"role": "assistant", "content": ANSWER_COMPLIANT if model.path == "compliant" else ANSWER_VIOLATING,
            "tool_calls": []}


def script():
    r = ch3.responses()
    r.append({"role": "assistant", "content": "", "tool_calls": [
        call("c9", "read_file", path="report.py"),
        call("c10", "read_file", path="test_billing.py"),
        call("c11", "run_suite", tests=SUITE)]})
    r += [proposal, rerun_suite, final]
    return r


# ----------------------------------------------------------------------------------
# One run under one context policy
# ----------------------------------------------------------------------------------

def run_branch(name, builder, db, seed, limit=16, snapshot_at=None):
    ws = Workspace("ch06_report")
    store = cx.RecordStore(RUN_ID, SCOPE)
    tools = session_tools(ws, store)
    defs = tools.definitions()
    tools_tokens = cx.approx_tokens(defs)
    store.add_instruction(ch3.SYSTEM)
    store.add_task(1, ch3.TASK, [{"id": "C1", "text": "Do not deploy it."}], [TASK1_GOAL])
    model = StandInModel(script())
    builds, checks, snapshot = [], [], None
    task2_added = False
    observed = False
    for call_no in range(1, limit + 1):
        if snapshot_at == call_no:
            snapshot = copy.deepcopy(store)
        b = builder.build(store, call_no, tools_tokens)
        builds.append(b)
        if not b["budget"]["fits"]:
            b["not_sent"] = "request exceeds the input allowance"
            break
        reply = model.generate(b["messages"], defs)
        store.add_assistant(reply, call_no)
        calls = reply.get("tool_calls") or []
        if not calls:
            if not task2_added:
                # The person reviews the answer and revises the task (task revision 2).
                contract = cx.signature_of(ws.read("report.py"))
                store.add_task(2, TASK2_TEXT,
                               [{"id": "C2", "text": C2_TEXT,
                                 "contract": "%s at report.py@%s" % (contract, content_version(ws.read("report.py")))},
                                {"id": "C3", "text": "Keep the diff small."}],
                               ["Make the new timezone matrix in test_report_matrix.py pass."])
                task2_added = True
                continue
            break
        if any(c["name"] == "replace_text" and c["arguments"].get("path") == "report.py" for c in calls) and task2_added:
            files = {p: ws.read(p) for p in ("report.py",)}
            after = cx.simulate_edits(files, calls)
            before_sig, after_sig = cx.signature_of(files["report.py"]), cx.signature_of(after["report.py"])
            checks.append({"call": call_no, "constraint": "C2", "check": "signature of belongs_to_report",
                           "expected": before_sig, "proposed": after_sig,
                           "result": "satisfied" if before_sig == after_sig else "VIOLATED",
                           "mode": "reported (not enforced)"})
        for c in calls:
            obs = tools.execute_checked(c)
            rec = store.add_result(c, obs)
            if c["name"] == "read_file" and c["arguments"].get("path") == "settings.json" and not observed:
                db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
                           ("O-r%s-%s" % (RUN_ID.split("-")[1], rec["id"]), RUN_ID, SCOPE, "settings.json",
                            rec["revision"], 2, "2026-09-11T20:00:00Z", cx.canonical(json.loads(obs["text"])),
                            "read_file in this run"))
                db.commit()
                observed = True
    after = ws.snapshot()
    final_sig = cx.signature_of(after["report.py"])
    result = {"branch": name, "policy": builder.policy, "calls_built": len(builds),
              "requests_sent": len(model.requests), "path_taken": model.path,
              "final_signature": final_sig, "checks": checks, "answer": store.records[-1]["message"].get("content"),
              "last_suite": next((r["message"]["content"] for r in reversed(store.records)
                                  if r.get("tool") == "run_suite"), None)}
    ws.close()
    return {"result": result, "builds": builds, "store": store, "files": after, "defs": defs,
            "tools_tokens": tools_tokens, "requests": model.requests, "snapshot": snapshot}


# ----------------------------------------------------------------------------------
# Context tests (no model involved)
# ----------------------------------------------------------------------------------

def request_text(b):
    return json.dumps(b["messages"], ensure_ascii=False)


def context_tests(b, store):
    text = request_text(b).lower()
    derived = [m for m, p in zip(b["messages"], b["provenance"]) if not p["transform"].startswith("verbatim")]
    cfg_lines = [l for m in derived for l in str(m.get("content", "")).splitlines() if "report_timezone" in l]
    t = {}
    t["constraint C2 survives"] = C2_PHRASE in text
    t["constraint C1 survives"] = "do not deploy it" in text
    t["call ids paired"] = not b["protocol_problems"]
    t["other-workspace data excluded"] = not any(s in text for s in ("europe/berlin", "workspace-17", "o-r77"))
    if not cfg_lines:
        t["newer config preferred"] = "n/a"
    else:
        ok = True
        for l in cfg_lines:
            ll = l.lower()
            if "america/los_angeles" in ll:
                continue
            if "superseded" not in ll:
                ok = False
        t["newer config preferred"] = ok and any("america/los_angeles" in l.lower() and "current" in l.lower()
                                                 for l in cfg_lines)
    t["within input allowance"] = b["budget"]["fits"]
    return t


class Tee:
    """Print to the terminal and keep a copy in <output>/summary.txt."""

    def __init__(self, path):
        self.term = sys.stdout
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        self.term.write(s)
        self.f.write(s)

    def flush(self):
        self.term.flush()
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = ap.parse_args()
    out = args.output
    if os.path.exists(out):
        if not args.force:
            sys.exit(f"{out} exists; choose a new directory or pass --force")
        shutil.rmtree(out)
    os.makedirs(out)
    tee = Tee(os.path.join(out, "summary.txt"))
    sys.stdout = tee

    seed = json.load(open(os.path.join(HERE, "fixtures", "ch06_records", "seed_observations.json")))

    def fresh_db(tag):
        return cx.open_observations(os.path.join(out, "observations-%s.sqlite" % tag), seed)

    # ---- Part A: the second edition's budget arithmetic, from its declared counts ----
    declared = {"window": 8192, "output_reserve": 1536}
    declared["input_allowance"] = declared["window"] - declared["output_reserve"]
    required = [("Instructions", 720), ("Tool definitions", 980), ("Current task", 180), ("Recent exchange", 1640)]
    evidence = [("settings.json", 140), ("report.py (relevant source)", 980), ("test_report.py", 420)]
    req_total = sum(v for _, v in required)
    left = declared["input_allowance"] - req_total
    ev_total = sum(v for _, v in evidence)
    used = req_total + ev_total
    after = declared["input_allowance"] - used
    old_log = 2600
    arithmetic = {"declared_counts_not_tokenized": True, **declared, "required": required, "required_total": req_total,
                  "left_after_required": left, "evidence": evidence, "evidence_total": ev_total,
                  "input_used": used, "left_after_evidence": after, "old_log": old_log,
                  "old_log_fits": old_log <= after,
                  "omissions": [{"item": "old test log", "tokens": old_log,
                                 "reason": "needs %d, %d left" % (old_log, after)}]}
    assert (declared["input_allowance"], req_total, left, used, after) == (6656, 3520, 3136, 5060, 1596)
    write_run(out, "budget_arithmetic.json", arithmetic)

    # ---- Part B: the same session under four policies ----
    branches = {}
    branches["full"] = run_branch("full", cx.FullTranscriptBuilder(), fresh_db("full"), seed, snapshot_at=9)
    branches["truncate"] = run_branch("truncate", cx.TruncateOldestBuilder(), fresh_db("truncate"), seed)
    db_ft = fresh_db("freetext")
    branches["freetext"] = run_branch("freetext", cx.FreeTextSummaryBuilder(keep_recent=4, db=db_ft), db_ft, seed)
    db_st = fresh_db("structured")
    branches["structured"] = run_branch("structured", cx.StructuredBuilder(db=db_st), db_st, seed)
    db_rs = fresh_db("structured-resummarize")
    branches["structured-resummarize"] = run_branch(
        "structured-resummarize", cx.StructuredBuilder(db=db_rs, freeze_summary=False), db_rs, seed)
    db_tf = fresh_db("structured-task-first")
    branches["structured-task-first"] = run_branch(
        "structured-task-first", cx.StructuredBuilder(db=db_tf, layout="task_first", status_line=True), db_tf, seed)

    for name, br in branches.items():
        d = os.path.join(out, name)
        write_run(d, "result.json", br["result"])
        write_run(d, "builds.json", br["builds"])
        write_run(d, "records.json", br["store"].dump())
        write_run(d, "compactions.json", br["store"].compactions)
        for fname, text in br["files"].items():
            write_run(os.path.join(d, "after"), fname, text)

    st = branches["structured"]
    full = branches["full"]
    # Chapter 3's call 7, as the structured builder assembled it (no pressure yet)
    b7 = st["builds"][6]
    write_run(out, "ch3_call7_request.json", {"messages": b7["messages"], "provenance": b7["provenance"]})
    write_run(out, "ch3_call7_omissions.json", b7["omissions"])
    write_run(out, "ch3_call7_budget.json", b7["budget"])

    # The request for call 9 (the first call after the long log) under each policy
    call9 = {n: br["builds"][8] for n, br in branches.items() if len(br["builds"]) >= 9}
    for n, b in call9.items():
        write_run(out, "call9_%s.json" % n, b)

    tests = {n: context_tests(b, branches[n]["store"]) for n, b in call9.items()}
    write_run(out, "context_tests.json", tests)

    # ---- Sweeps over the same records (the state just before call 9) ----
    prefix = full["snapshot"]
    tt = full["tools_tokens"]
    sweep = {"freetext_keep_recent": [], "truncate_allowance": [], "structured_recent_budget": []}
    for k in range(1, 9):
        b = cx.FreeTextSummaryBuilder(keep_recent=k).build(prefix, 9, tt)
        sweep["freetext_keep_recent"].append({"keep_recent": k, "valid": not b["protocol_problems"],
                                              "C2": C2_PHRASE in request_text(b).lower(),
                                              "tokens": b["budget"]["total_input"], "fits": b["budget"]["fits"]})
    for a in range(4000, 8001, 500):
        b = cx.TruncateOldestBuilder(allowance=a).build(prefix, 9, tt)
        sweep["truncate_allowance"].append({"allowance": a, "valid": not b["protocol_problems"],
                                            "C2": C2_PHRASE in request_text(b).lower(),
                                            "removed": len(b["omissions"]), "tokens": b["budget"]["total_input"]})
    for rb in (0, 250, 500, 1000, 1500, 2000):
        b = cx.StructuredBuilder(recent_budget=rb).build(copy.deepcopy(prefix), 9, tt)
        sweep["structured_recent_budget"].append({"recent_budget": rb, "valid": not b["protocol_problems"],
                                                  "C2": C2_PHRASE in request_text(b).lower(),
                                                  "tokens": b["budget"]["total_input"],
                                                  "summarized": sum(1 for o in b["omissions"] if o["action"] == "summarized")})
    write_run(out, "sweeps.json", sweep)

    # ---- Governance: replay the same records under a candidate policy ----
    v1 = cx.StructuredBuilder(db=db_st)
    v2 = cx.StructuredBuilder(db=db_st, pin_task=False)
    v2.policy["id"] = "ctx-structured-v2-candidate"
    v2.policy["hash"] = cx.sha12({k: v for k, v in v2.policy.items() if k != "hash"})
    b1 = v1.build(copy.deepcopy(prefix), 9, tt)
    b1_again = cx.StructuredBuilder(db=db_st).build(copy.deepcopy(prefix), 9, tt)
    b2 = v2.build(copy.deepcopy(prefix), 9, tt)
    s1 = {o["record"] for o in b1["omissions"] if o["action"] == "summarized"}
    s2 = {o["record"] for o in b2["omissions"] if o["action"] == "summarized"}
    replay = {"records_prefix_sha": cx.sha12(prefix.dump()),
              "v1": {"policy": v1.policy, "request_sha": b1["request_sha"], "tokens": b1["budget"]["total_input"],
                     "tests": context_tests(b1, prefix)},
              "v1_rebuilt_request_sha": b1_again["request_sha"],
              "v2": {"policy": v2.policy, "request_sha": b2["request_sha"], "tokens": b2["budget"]["total_input"],
                     "tests": context_tests(b2, prefix)},
              "newly_summarized_in_v2": sorted(s2 - s1), "no_longer_summarized_in_v2": sorted(s1 - s2),
              "v1_minus_v2_tokens": b1["budget"]["total_input"] - b2["budget"]["total_input"]}
    write_run(out, "policy_replay.json", replay)

    # ---- Just-in-time retrieval of the full log through read_record ----
    log_rec = next(r for r in st["store"].records if r.get("tool") == "run_suite")
    lines = log_rec["message"]["content"]["log"].splitlines()
    err_at = next(i for i, l in enumerate(lines) if l.startswith("ERROR: ")) + 1
    ws = Workspace("ch06_report")
    reg = session_tools(ws, st["store"])
    jit = {"digest_points_to": log_rec["id"], "log_lines": len(lines),
           "read": reg.execute_checked(call("x1", "read_record", record_id=log_rec["id"],
                                            start_line=err_at, end_line=err_at + 10)),
           "other_scope": reg.execute_checked(call("x2", "read_record", record_id="O-r77-5"))}
    ws.close()
    write_run(out, "jit_retrieval.json", jit)

    # ---- Tool definitions as a budget item ----
    catalog = json.load(open(os.path.join(HERE, "fixtures", "ch06_records", "tool_catalog.json")))
    tool_budget = {"session_tools": [d["name"] for d in st["defs"]], "session_tokens": st["tools_tokens"],
                   "catalog_tools": len(catalog), "catalog_tokens": cx.approx_tokens(catalog),
                   "input_allowance": cx.INPUT_ALLOWANCE}
    write_run(out, "tool_budget.json", tool_budget)

    # ---- Prompt-cache arithmetic over the recorded requests ----
    def cache_for(br, upto=None):
        reqs = [(br["defs"], r["messages"]) for r in br["requests"]][:upto]
        rows = cx.cache_costs(reqs)
        return {"rows": rows, "total_input": sum(r["input_tokens"] for r in rows),
                "total_cached": sum(r["cached_prefix"] for r in rows),
                "cost_units": round(sum(r["cost_units"] for r in rows), 1),
                "uncached_cost_units": sum(r["uncached_cost_units"] for r in rows)}
    cache = {"assumptions": {"write_multiplier": cx.CACHE_WRITE, "read_multiplier": cx.CACHE_READ,
                             "model": "every request written in full; next request reads its longest common "
                                      "prefix, compared unit by unit (tool definitions, then messages); "
                                      "minimum cacheable length and expiry ignored"},
             "full_calls_1_8": cache_for(full, 8),
             "structured_task_last": cache_for(st),
             "structured_resummarize": cache_for(branches["structured-resummarize"]),
             "structured_task_first_status": cache_for(branches["structured-task-first"])}
    write_run(out, "cache_economics.json", cache)

    # ---- Print the key results ----
    print("== Budget arithmetic (second edition, declared counts) ==")
    print("window %d - reserve %d = input %d; required %d; left %d; +evidence %d -> used %d, left %d; "
          "old log %d fits: %s" % (8192, 1536, 6656, req_total, left, ev_total, used, after, old_log,
                                    arithmetic["old_log_fits"]))
    print("counter:", cx.COUNTER_ID)
    print("tool definitions: session %d tools = %d tokens; catalog %d tools = %d tokens (allowance %d)" % (
        len(tool_budget["session_tools"]), tool_budget["session_tokens"], len(catalog),
        tool_budget["catalog_tokens"], cx.INPUT_ALLOWANCE))
    print()
    print("== Chapter 3 call 7 under the structured builder ==")
    print("items %d, total input %d, headroom %d, omissions %d" % (
        len(b7["messages"]), b7["budget"]["total_input"], b7["budget"]["headroom"], len(b7["omissions"])))
    print("full transcript at call 7: %d tokens" % full["builds"][6]["budget"]["total_input"])
    print()
    print("== Call 9 (first request after the long log) ==")
    for n, b in call9.items():
        om = {}
        for o in b["omissions"]:
            om[o["action"]] = om.get(o["action"], 0) + 1
        print("%-22s tokens %5d fits %-5s items %2d problems %d omissions %s%s" % (
            n, b["budget"]["total_input"], b["budget"]["fits"], len(b["messages"]),
            len(b["protocol_problems"]), om, "  NOT SENT" if b.get("not_sent") else ""))
    print()
    print("== Outcome per policy ==")
    for n, br in branches.items():
        r = br["result"]
        suite = r["last_suite"] or {}
        print("%-22s path %-9s signature %-70s suite %s run %s fail %s err %s" % (
            n, r["path_taken"], r["final_signature"], suite.get("status"), suite.get("tests_run"),
            suite.get("failures"), suite.get("errors")))
        for c in r["checks"]:
            print("   check call %d %s: %s -> %s" % (c["call"], c["result"], c["expected"], c["proposed"]))
    print()
    print("== Context tests at call 9 ==")
    for n, t in tests.items():
        print("%-22s %s" % (n, t))
    print()
    print("== Sweeps ==")
    for k, rows in sweep.items():
        print(k, rows)
    print()
    print("== Policy replay ==")
    print("v1 sha %s rebuilt %s; v2 sha %s; v1-v2 tokens %d" % (
        replay["v1"]["request_sha"], replay["v1_rebuilt_request_sha"], replay["v2"]["request_sha"],
        replay["v1_minus_v2_tokens"]))
    print("v1 tests", replay["v1"]["tests"])
    print("v2 tests", replay["v2"]["tests"])
    print()
    print("== Cache arithmetic ==")
    for k in ("full_calls_1_8", "structured_task_last", "structured_resummarize", "structured_task_first_status"):
        c = cache[k]
        print("%-30s requests %d input %d cached %d cost %.1f (uncached %d)" % (
            k, len(c["rows"]), c["total_input"], c["total_cached"], c["cost_units"], c["uncached_cost_units"]))


if __name__ == "__main__":
    main()
