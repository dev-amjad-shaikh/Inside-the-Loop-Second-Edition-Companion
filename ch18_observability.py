"""Chapter 18: observability.

Instruments runs from earlier chapters so they emit OTel-shaped JSON spans, then
uses the spans for the jobs telemetry is good at and shows where it must defer to
the run's authoritative records.

  1 the Chapter 12 reporting run, traced: span tree, five-question audit, the same
    run under a flawed design (one "agent" span and free-text logs)
  2 a slow run on a wide repository: critical path from spans, before and after
    replacing a scanning search with an index
  3 a failure with no error: the stale-verification variant, as logs and as a trace
  4 content capture: off, structure-preserving redaction, naive truncation; replay
    of model inputs from records versus from telemetry
  5 telemetry across a crash: the Chapter 13 lost receipt, with the reconciler's
    trace linked through the operation record
  6 delegation spans: the Chapter 15 stale reviewer
  7 a simulated fleet: head sampling, tail sampling on errors, and sampling on
    recorded outcome; dashboards; linking provider bill lines to runs

    python3 ch18_observability.py --output runs/ch18

Model responses are scripted. File reads, edits and test runs are real. All
durations are simulated from stated constants; token counts are approximate
(characters / 4); prices are invented. The output directory is replaced.
"""
import argparse
import copy
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from itl.core import ScriptedModel, Workspace, report_tools, write_run  # noqa: E402
from itl.ch12_state import RunStore, DirWorkspace, read_events, sha, text_version  # noqa: E402
from itl import ch15_delegation as d15  # noqa: E402
from itl import ch13_recovery as r13  # noqa: E402
from itl.ch18_observability import (  # noqa: E402
    BatchProcessor, Exporter, SimClock, Tracer, by_trace, canon, critical_path, has_error, head_keep,
    input_messages_attr, redact_message, root_of, rule_orphans, rule_stale_evidence, seconds, sha12,
    tail_error_keep, tail_outcome_keep, traceparent, tree_lines)
import ch12_state as c12  # noqa: E402  (the Chapter 12 durable runtime, reused unchanged)

MODEL = "scripted-report-model"
AGENT = "report-fixer"
AGENT_VERSION = "1.4.0"
POLICY_VERSION = "report-policy/3"
TOOL_S = {"read_file": 0.05, "replace_text": 0.05, "run_tests": 4.8}   # simulated seconds
CONTEXT_S = 0.02
RECORD_S = 0.004
HUMAN_WAIT_S = 240.0
CANARY = "sk-itl-canary-7f3a9c"          # a fake credential in the adapter's request headers


class FreeTextLog:
    """The flawed design's logs: one line per happening, prose, whatever seemed useful."""

    def __init__(self, clock):
        self.clock, self.lines = clock, []

    def info(self, msg, level="INFO"):
        t = (self.clock.now - c18_epoch()) / 1e9
        self.lines.append(f"{t:9.3f} {level} agent: {msg}")


def c18_epoch():
    return SimClock().now


# =============================================================================
# The Chapter 12 runtime, traced
# =============================================================================
class TracedRuntime(c12.Runtime):
    """Chapter 12's durable runtime with spans at each component boundary.

    Every span names the record it corresponds to (model-call id, operation id,
    event sequence), so the trace is an index into the records, never a copy of them."""

    def __init__(self, store, ws, owner, tracer, clock, flog, verify=True):
        super().__init__(store, ws, owner)
        self.tracer, self.clock, self.flog, self.verify = tracer, clock, flog, verify
        self.captured = {}          # call_id -> exact model input (kept for experiment 4 only)
        self.approval_spans = {}
        self.root = None
        inner = self.tools.execute_checked

        def timed(call):
            out = inner(call)
            self.clock.advance(TOOL_S.get(call["name"], 0.05))
            return out
        self.tools.execute_checked = timed

    # ---- run span -------------------------------------------------------------
    def begin(self):
        self.root = self.tracer.start(f"invoke_agent {AGENT}", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.provider.name": "itl.scripted",
            "gen_ai.agent.name": AGENT, "gen_ai.agent.version": AGENT_VERSION,
            "gen_ai.conversation.id": c12.RUN, "itl.run.id": c12.RUN, "itl.policy.version": POLICY_VERSION})
        self.flog.info("starting agent")
        self.create()

    def finish(self):
        run = self.store.run(c12.RUN)
        unsettled = self.store.conn.execute(
            "SELECT COUNT(*) FROM operations WHERE run_id=? AND state IN ('dispatched','uncertain')",
            (c12.RUN,)).fetchone()[0]
        self.root["attributes"].update({"itl.run.outcome": "verified" if run["status"] == "completed" and self.verify
                                        else ("completed_unverified" if run["status"] == "completed" else run["status"]),
                                        "itl.task.revision": run["task_revision"],
                                        "itl.ops.unsettled": unsettled})
        self.tracer.end(self.root)
        self.flog.info(f"agent finished status={run['status']}")

    # ---- the context builder and the model adapter ---------------------------------
    def model_turn(self, model, ordinal):
        tr, run = self.tracer, self.store.run(c12.RUN)
        call_id = f"m{ordinal}"
        b = tr.start("build_context", attributes={"itl.component": "Context builder",
                                                  "itl.task.revision": run["task_revision"]})
        messages = self.model_input()
        self.clock.advance(CONTEXT_S)
        n_events = self.store.conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (c12.RUN,)).fetchone()[0]
        b["attributes"].update({"itl.context.records_read": n_events, "itl.context.messages": len(messages),
                                "itl.input.sha256": sha(messages)})
        tr.end(b)
        tin = d15.approx_tokens(messages)
        tool_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        ch = tr.start(f"chat {MODEL}", kind="CLIENT", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": "itl.scripted",
            "gen_ai.request.model": MODEL, "gen_ai.conversation.id": c12.RUN,
            "itl.request.id": f"req-{c12.RUN}-{call_id}", "itl.record.model_call": call_id,
            "itl.input.sha256": sha(messages), "itl.input.messages": len(messages),
            "itl.input.tool_call_ids": tool_ids, "itl.task.revision": run["task_revision"]})
        self.flog.info(f"POST /v1/chat model={MODEL} headers={{'authorization': 'Bearer {CANARY}', "
                       f"'x-request-id': 'req-{c12.RUN}-{call_id}'}} messages={len(messages)}")
        response = model.generate(messages, self.tools.definitions())
        tout = d15.approx_tokens(response)
        self.clock.advance(d15.model_seconds(tin, tout))
        self.captured[call_id] = messages
        with self.store.transaction("model_adapter", c12.RUN, self.holder()) as tx:
            tx.append("model_call.recorded", call_id,
                      {"ordinal": ordinal, "task_revision": run["task_revision"], "input_sha": sha(messages),
                       "input_messages": len(messages), "response": response})
        with self.store.transaction("control", c12.RUN, self.holder()) as tx:
            for c in response["tool_calls"]:
                tx.append("op.prepared", c["id"], {"model_call": call_id, "tool": c["name"],
                                                   "arguments": c["arguments"],
                                                   "task_revision": run["task_revision"]})
        self.clock.advance(RECORD_S)
        seq = self.store.conn.execute("SELECT seq FROM events WHERE entity=? AND type='model_call.recorded'",
                                      (call_id,)).fetchone()[0]
        ch["attributes"].update({
            "gen_ai.response.id": f"resp-{c12.RUN}-{call_id}", "gen_ai.response.model": MODEL,
            "gen_ai.usage.input_tokens": tin, "gen_ai.usage.output_tokens": tout,
            "gen_ai.response.finish_reasons": ["tool_calls"] if response["tool_calls"] else ["stop"],
            "itl.proposed.tool_call_ids": [c["id"] for c in response["tool_calls"]],
            "itl.record.event_seq": seq})
        tr.end(ch)
        self.flog.info(f"model replied: {response['content'][:60] or '(tool calls)'}")
        return response

    # ---- the executor, policy and approvals -------------------------------------------
    def execute(self, c):
        tr, name, op_id = self.tracer, c["name"], c["id"]
        if name in self.approval_required:
            s = tr.start(f"request_approval {name}", attributes={
                "itl.component": "Policy and authority", "itl.op.id": op_id, "gen_ai.tool.call.id": op_id,
                "itl.policy.decision": "require_approval", "itl.policy.rule": "export_needs_person",
                "itl.policy.version": POLICY_VERSION, "itl.approval.args_digest": sha(c["arguments"])})
            status = super().execute(c)
            s["attributes"]["itl.op.state"] = self.store.op(op_id)["state"]
            tr.end(s)
            self.approval_spans[op_id] = s
            self.flog.info(f"{name} needs approval, waiting")
            return status
        s = tr.start(f"execute_tool {name}", attributes={
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name, "gen_ai.tool.call.id": op_id,
            "gen_ai.tool.type": "function", "itl.op.id": op_id, "itl.policy.decision": "allow",
            "itl.policy.version": POLICY_VERSION})
        before = {p: text_version(self.ws.read(p)) for p in c12.FILES}
        status = super().execute(c)
        op = self.store.op(op_id)
        obs = self.store.conn.execute("SELECT status, result FROM observations WHERE op_id=?", (op_id,)).fetchone()
        a = s["attributes"]
        a["itl.op.state"] = op["state"]
        if obs:
            a["itl.observation.status"] = obs[0]
        if op["covers"] and name == "run_tests":
            for p, v in json.loads(op["covers"]).items():
                a[f"itl.evidence.covers.{p}"] = v
        for p, n, content in self.store.conn.execute(
                "SELECT path, version_no, content FROM artifact_versions WHERE produced_by=?", (op_id,)):
            a[f"itl.artifact.version_before.{p}"] = before[p]
            a[f"itl.artifact.version_after.{p}"] = text_version(content)
        a["itl.record.event_seq"] = self.store.conn.execute(
            "SELECT MAX(seq) FROM events WHERE entity=?", (op_id,)).fetchone()[0]
        if status == "failed_before_effect":
            a.update({"error.type": "invalid_arguments", "itl.failure.stage": "validation"})
            tr.end(s, "ERROR", "invalid_arguments")
            self.flog.info(f"tool {name} failed: {json.loads(obs[1]).get('message')}")
        else:
            tr.end(s)
            self.flog.info(f"tool {name} {obs[0] if obs else status}")
        return status

    def revise_task(self, text):
        withdrawn = [r[0] for r in self.store.conn.execute(
            "SELECT op_id FROM operations WHERE run_id=? AND state IN ('prepared','awaiting_approval')",
            (c12.RUN,))]
        links = [{"traceId": self.approval_spans[o]["traceId"], "spanId": self.approval_spans[o]["spanId"],
                  "attributes": {"itl.link.reason": "approval withdrawn by task revision"}}
                 for o in withdrawn if o in self.approval_spans]
        s = self.tracer.start("revise_task", links=links, attributes={"itl.component": "Human interface"})
        rev = super().revise_task(text)
        s["attributes"].update({"itl.task.revision": rev, "itl.ops.withdrawn": withdrawn})
        self.clock.advance(RECORD_S)
        self.tracer.end(s)
        self.flog.info(f"user sent a correction; cancelled {len(withdrawn)} pending operation(s)")
        return rev

    # ---- the verifier -------------------------------------------------------------
    def complete(self):
        s = self.tracer.start(f"verify {AGENT}", attributes={"itl.component": "Verifier",
                                                           "itl.verify.performed": self.verify})
        ok, reason, current = self.evidence_for_completion()
        for p, v in current.items():
            s["attributes"][f"itl.artifact.head.{p}"] = v
        if self.verify:
            accepted, why = super().complete()
        else:
            with self.store.transaction("control", c12.RUN, self.holder()) as tx:
                tx.append("run.completed", c12.RUN, {"status": "completed", "evidence": "model reported completion"},
                          event_id=f"{c12.RUN}:completed")
            accepted, why = True, "model reported completion"
        s["attributes"].update({"itl.verify.result": "accepted" if accepted else "refused",
                                "itl.verify.reason": why, "itl.evidence.current": ok})
        self.clock.advance(RECORD_S)
        self.tracer.end(s)
        self.flog.info("run complete" if accepted else "completion deferred")
        return accepted, why


def new_tracing(key, batch_size=64):
    clock = SimClock()
    exp = Exporter()
    proc = BatchProcessor(exp, batch_size=batch_size, schedule_delay=5.0, clock=clock)
    tracer = Tracer(key, clock, proc, resource={"service.name": "itl-runtime", "service.version": AGENT_VERSION})
    return clock, exp, proc, tracer


def stale_responses():
    """Chapter 12's script with the retest after the docstring edit removed."""
    r = c12.responses()
    return r[:8] + [r[9]]


def drive_ch12_run(out, name, verify=True, script=None):
    clock, exp, proc, tracer = new_tracing(f"ch18-{name}")
    flog = FreeTextLog(clock)
    ws = DirWorkspace(os.path.join(out, f"{name}_workspace"))
    store = RunStore(os.path.join(out, f"{name}.db"), create=True)
    rt = TracedRuntime(store, ws, "worker-1", tracer, clock, flog, verify=verify)
    model = ScriptedModel(script or c12.responses())
    rt.begin()
    last = rt.drive(model, 1)                       # through the export that needs a person
    clock.advance(HUMAN_WAIT_S)                     # the person answers four minutes later
    rt.revise_task(c12.CORRECTION)
    rt.set_status("running")
    last = rt.drive(model, last + 1, stop_after=last + 1)
    verdicts = []
    if verify:
        verdicts.append(rt.complete())              # refused: the evidence covers an older report.py
        last = rt.drive(model, last + 1)
    else:
        last = rt.drive(model, last + 1)
    verdicts.append(rt.complete())
    rt.finish()
    proc.flush()
    return {"store": store, "rt": rt, "spans": exp.spans, "logs": flog.lines, "ws": ws, "verdicts": verdicts,
            "exporter": exp}


# =============================================================================
# The five questions, asked of each telemetry design
# =============================================================================
def five_questions(spans, store):
    """Answer each question from the trace, then confirm the answer against the records."""
    conn = store.conn
    chats = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "chat"]
    tools = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "execute_tool"]
    approvals = [s for s in spans if s["name"].startswith("request_approval")]
    verifies = [s for s in spans if s["name"].startswith("verify")]
    root = root_of(spans)
    out = {}
    # Q1: what did the model receive? The final call's input digest, rebuilt from records.
    if chats and "itl.input.sha256" in chats[-1]["attributes"]:
        last = chats[-1]["attributes"]
        seq = last["itl.record.event_seq"]
        rebuilt = sha(c12.model_input_from_records(conn, c12.RUN, upto=seq - 1))
        out["Q1 what the model received"] = {"span": chats[-1]["name"], "call": last["itl.record.model_call"],
                                             "trace_sha": last["itl.input.sha256"], "records_sha": rebuilt,
                                             "answered": rebuilt == last["itl.input.sha256"]}
    else:
        out["Q1 what the model received"] = {"answered": False}
    # Q2: which operation was authorized, and by whom?
    if tools and "itl.policy.decision" in tools[0]["attributes"]:
        decisions = {}
        for s in tools + approvals:
            a = s["attributes"]
            decisions[a["itl.op.id"]] = a["itl.policy.decision"]
        rec = dict(conn.execute("SELECT op_id, state FROM operations WHERE run_id=?", (c12.RUN,)).fetchall())
        out["Q2 what was authorized"] = {"policy_version": tools[0]["attributes"]["itl.policy.version"],
                                         "decisions": decisions, "record_states": rec,
                                         "answered": set(decisions) == set(rec)}
    else:
        out["Q2 what was authorized"] = {"answered": False}
    # Q3: what actually changed?
    changes = {}
    for s in tools:
        for k, v in s["attributes"].items():
            if k.startswith("itl.artifact.version_after."):
                changes.setdefault(k.split(".", 3)[3], []).append(v)
    if changes:
        rec = {}
        for p, content in conn.execute("SELECT path, content FROM artifact_versions WHERE produced_by IS NOT NULL "
                                       "ORDER BY path, version_no"):
            rec.setdefault(p, []).append(text_version(content))
        out["Q3 what changed"] = {"trace": changes, "records": rec, "answered": changes == rec}
    else:
        out["Q3 what changed"] = {"answered": False}
    # Q4: which evidence supports the result?
    if verifies and "itl.verify.result" in verifies[-1]["attributes"]:
        rule = rule_stale_evidence(spans)
        a = verifies[-1]["attributes"]
        out["Q4 which evidence"] = {"verify": a["itl.verify.result"], "evidence_call": rule.get("evidence_call"),
                                    "finding": rule["finding"], "records_reason": a["itl.verify.reason"],
                                    "answered": True}
    else:
        out["Q4 which evidence"] = {"answered": False}
    # Q5: what remains uncertain?
    if "itl.ops.unsettled" in root["attributes"]:
        rec = conn.execute("SELECT COUNT(*) FROM operations WHERE run_id=? AND state IN ('dispatched','uncertain')",
                           (c12.RUN,)).fetchone()[0]
        out["Q5 what is uncertain"] = {"trace": root["attributes"]["itl.ops.unsettled"], "records": rec,
                                       "answered": rec == root["attributes"]["itl.ops.unsettled"]}
    else:
        out["Q5 what is uncertain"] = {"answered": False}
    return out


def flawed_telemetry(spans, logs):
    """The flawed design: one span for the whole run, and the free-text log lines."""
    root = root_of(spans)
    one = {"traceId": root["traceId"], "spanId": root["spanId"], "parentSpanId": None, "name": "agent",
           "kind": "INTERNAL", "startTimeUnixNano": root["startTimeUnixNano"],
           "endTimeUnixNano": root["endTimeUnixNano"], "attributes": {"agent": AGENT, "status": "ok"},
           "events": [], "links": [], "status": {"code": "OK"}}
    return [one], logs


# =============================================================================
# Experiment 2: the slow run
# =============================================================================
NEW_TEST = c12.NEW_TEST
FIX_OLD = "    return created_at.date() == requested_day"
FIX_NEW = ("    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
           "    return local_time.date() == requested_day")
SCAN_S_PER_FILE = 2.0
INDEX_LOOKUP_S = 0.15


def call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def slow_script():
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            call("s1", "search_code", pattern="belongs_to_report"),
            call("s2", "read_file", path="settings.json")]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("s3", "read_file", path="report.py"), call("s4", "read_file", path="test_report.py")]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("s5", "replace_text", path="test_report.py", old='\n\nif __name__ == "__main__":',
                 new="\n" + NEW_TEST)]},
        {"role": "assistant", "content": "", "tool_calls": [call("s6", "run_tests", tests=["test_report.py"])]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("s7", "replace_text", path="report.py", old=FIX_OLD, new=FIX_NEW)]},
        {"role": "assistant", "content": "", "tool_calls": [call("s8", "run_tests", tests=["test_report.py"])]},
        {"role": "assistant", "content": "The report used the UTC date. It now converts to the report timezone. "
                                         "The new test failed before the fix and both tests pass after it.",
         "tool_calls": []},
    ]


def search_tool(ws, mode, index):
    def search_code(pattern):
        files = sorted(f for f in os.listdir(ws.root) if f.endswith(".py"))
        if mode == "indexed":
            hits = index.get(pattern, [])
            scanned = 0
        else:
            hits = []
            for f in files:
                for i, line in enumerate(ws.read(f).splitlines(), 1):
                    if pattern in line:
                        hits.append({"path": f, "line": i, "text": line.strip()})
            scanned = len(files)
        return {"status": "ok", "matches": hits, "files_scanned": scanned}
    return search_code


def build_index(ws):
    idx = {}
    for f in sorted(os.listdir(ws.root)):
        if not f.endswith(".py"):
            continue
        for i, line in enumerate(ws.read(f).splitlines(), 1):
            for tok in set(t.strip("(),:.\"'") for t in line.split()):
                if tok.isidentifier():
                    idx.setdefault(tok, []).append({"path": f, "line": i, "text": line.strip()})
    return idx


def slow_run(mode):
    clock, exp, proc, tracer = new_tracing(f"ch18-slow-{mode}")
    ws = Workspace("report_wide")
    index = build_index(ws) if mode == "indexed" else None
    tools = report_tools(ws)
    tools.register("search_code", "Search the repository for a text pattern.", {
        "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"],
        "additionalProperties": False}, search_tool(ws, mode, index))
    model = ScriptedModel(slow_script())
    messages = [{"role": "system", "content": c12.SYSTEM.replace("Workspace files: settings.json, report.py, "
                                                                 "test_report.py.", "The repository has 15 files.")},
                {"role": "user", "content": c12.TASK}]
    root = tracer.start(f"invoke_agent {AGENT}", attributes={"gen_ai.operation.name": "invoke_agent",
                                                             "gen_ai.agent.name": AGENT, "itl.search.mode": mode})
    n = 0
    while True:
        n += 1
        b = tracer.start("build_context", attributes={"itl.context.messages": len(messages)})
        clock.advance(CONTEXT_S)
        tracer.end(b)
        tin = d15.approx_tokens(messages)
        ch = tracer.start(f"chat {MODEL}", kind="CLIENT", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.request.model": MODEL,
            "itl.input.tool_call_ids": [m["tool_call_id"] for m in messages if m.get("role") == "tool"]})
        reply = model.generate(messages, tools.definitions())
        tout = d15.approx_tokens(reply)
        clock.advance(d15.model_seconds(tin, tout))
        ch["attributes"].update({"gen_ai.usage.input_tokens": tin, "gen_ai.usage.output_tokens": tout})
        tracer.end(ch)
        messages.append(reply)
        if not reply["tool_calls"]:
            break
        t0, longest = clock.now, 0.0
        for c in reply["tool_calls"]:          # calls from one response run concurrently
            clock.now = t0
            s = tracer.start(f"execute_tool {c['name']}", attributes={
                "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": c["name"],
                "gen_ai.tool.call.id": c["id"]}, parent=root)
            obs = tools.execute_checked(c)
            if c["name"] == "search_code":
                if mode == "indexed":
                    r = tracer.start("retrieval workspace-index", kind="CLIENT", parent=s, attributes={
                        "gen_ai.operation.name": "retrieval", "gen_ai.data_source.id": "workspace-index",
                        "gen_ai.retrieval.top_k": 20})
                    clock.advance(INDEX_LOOKUP_S)
                    tracer.end(r)
                    dur = INDEX_LOOKUP_S
                else:
                    dur = SCAN_S_PER_FILE * obs["files_scanned"]
                    clock.advance(dur)
                s["attributes"].update({"itl.search.files_scanned": obs["files_scanned"],
                                        "itl.search.matches": len(obs["matches"])})
            else:
                dur = TOOL_S[c["name"]]
                clock.advance(dur)
            s["attributes"]["itl.observation.status"] = obs.get("status")
            tracer.end(s)
            longest = max(longest, dur)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
        clock.now = t0 + int(round(longest * 1e9))
    tracer.end(root)
    proc.flush()
    ws.close()
    return exp.spans, reply["content"]


def summarize_path(spans):
    seg = critical_path(spans)
    total = seconds(root_of(spans))
    agg = {}
    for s in seg:
        k = s["name"]
        if k.startswith("chat "):
            k = "model calls (chat)"
        elif k == "build_context":
            k = "context building"
        elif k.endswith("(self)"):
            k = "own time of " + k[:-7]
        agg[k] = round(agg.get(k, 0) + s["seconds"], 3)
    return {"agent_latency_s": round(total, 3), "segments": [(s["name"], round(s["seconds"], 3), s["span"]) for s in seg],
            "by_kind": dict(sorted(agg.items(), key=lambda kv: -kv[1])),
            "span_seconds_sum": round(sum(seconds(s) for s in spans if s["parentSpanId"] == root_of(spans)["spanId"]), 3)}


# =============================================================================
# Experiment 4: content capture and replay
# =============================================================================
def capture_variants(result, limit=4096):
    rt, spans = result["rt"], result["spans"]
    conn = result["store"].conn
    variants = {}
    source_marker = "created_at.astimezone"
    for mode in ("off", "structure", "full_truncated", "full"):
        vs = copy.deepcopy(spans)
        for s in vs:
            cid = s["attributes"].get("itl.record.model_call")
            if s["attributes"].get("gen_ai.operation.name") == "chat" and cid:
                msgs = rt.captured[cid]
                if mode == "full_truncated":
                    v = input_messages_attr(msgs, "full", limit=limit)
                else:
                    v = input_messages_attr(msgs, mode)
                if v is not None:
                    s["attributes"]["gen_ai.input.messages"] = v
        variants[mode] = vs
    report = {}
    for mode, vs in variants.items():
        chats = [s for s in vs if s["attributes"].get("gen_ai.operation.name") == "chat"]
        parsed = exact = structure_ok = 0
        for s in chats:
            a = s["attributes"]
            cid = a["itl.record.model_call"]
            rebuilt = c12.model_input_from_records(conn, c12.RUN, upto=a["itl.record.event_seq"] - 1)
            want_structure = [redact_message(m) for m in rebuilt]
            v = a.get("gen_ai.input.messages")
            if v is None:
                continue
            try:
                got = json.loads(v)
                parsed += 1
            except json.JSONDecodeError:
                continue
            if mode in ("full", "full_truncated") and sha(got) == a["itl.input.sha256"]:
                exact += 1
            shape = [redact_message(m) for m in got] if mode != "structure" else got
            if [(m["role"], m.get("tool_call_id"), [c["id"] for c in m.get("tool_calls", [])]) for m in shape] == \
               [(m["role"], m.get("tool_call_id"), [c["id"] for c in m.get("tool_calls", [])]) for m in want_structure]:
                structure_ok += 1
        blob = canon(vs)
        report[mode] = {"chat_spans": len(chats), "content_bytes": sum(len(s["attributes"].get("gen_ai.input.messages", ""))
                                                                      for s in chats),
                        "source_text_in_telemetry": source_marker in blob,
                        "json_parsable": parsed, "exact_input_recovered": exact, "structure_recovered": structure_ok,
                        "sizes": [len(canon(rt.captured[s["attributes"]["itl.record.model_call"]])) for s in chats]}
    # replay from records
    rebuilt = []
    for cid, recorded, n in conn.execute("SELECT call_id, input_sha, input_messages FROM model_calls ORDER BY ordinal"):
        seq = conn.execute("SELECT seq FROM events WHERE entity=? AND type='model_call.recorded'", (cid,)).fetchone()[0]
        rebuilt.append(sha(c12.model_input_from_records(conn, c12.RUN, upto=seq - 1)) == recorded)
    report["records"] = {"model_calls": len(rebuilt), "exact_input_recovered": sum(rebuilt)}
    return variants, report


# =============================================================================
# Experiment 5: telemetry across a crash (Chapter 13's lost receipt)
# =============================================================================
def crash_experiment(out):
    root = os.path.join(out, "crash")
    ws = Workspace("report")
    text = ws.read("report.py")
    ws.write("report.py", text.replace(FIX_OLD, FIX_NEW))
    rclock = r13.SimClock()
    store, svc = r13.fresh_stores(root, rclock, ws)
    ex = r13.Executor(store, svc)
    RUN13 = "run-13"
    # process A
    clock, expA, procA, trA = new_tracing("ch18-crash-A")
    runspan = trA.start("invoke_agent report-exporter", attributes={"gen_ai.operation.name": "invoke_agent",
                                                                    "gen_ai.agent.name": "report-exporter"})
    store.start_run(RUN13, 3)
    s1 = trA.start("execute_tool read_revision", attributes={"gen_ai.operation.name": "execute_tool",
                                                             "gen_ai.tool.name": "read_revision",
                                                             "gen_ai.tool.call.id": "op-0001"})
    store.prepare("op-0001", RUN13, "read_revision", {"path": "report.py"})
    rclock.advance(0.4)
    clock.advance(0.4)
    rev = ex.dispatch("op-0001")["version"]
    trA.end(s1)
    rclock.advance(1.1)
    clock.advance(1.1)
    payload = {"report": "daily-orders", "report_date": "2026-09-11", "source_revision": rev,
               "destination": "reports/daily-orders-2026-09-11.csv", "format": "csv"}
    key = f"{RUN13}:export:daily-orders:2026-09-11"
    s2 = trA.start("execute_tool create_export", attributes={"gen_ai.operation.name": "execute_tool",
                                                             "gen_ai.tool.name": "create_export",
                                                             "gen_ai.tool.call.id": "op-0002",
                                                             "itl.op.key": key})
    store.prepare("op-0002", RUN13, "create_export", payload, op_key=key)
    # the intent record carries the trace context, so a later process can link to it
    store._write(lambda con: store.event(con, "op-0002", "trace_context", {"traceparent": traceparent(s2)}))
    rclock.advance(0.2)
    clock.advance(0.2)
    try:
        ex.dispatch("op-0002", interrupt="after_commit")
    except r13.Interrupted:
        pass
    lost = trA.crash()
    exported_A = [s["name"] for s in expA.spans]
    # process B: a restart, a new trace, linked through the record
    rclock.advance(6.0)
    store, svc = r13.restart(root, rclock, ws)
    clockB, expB, procB, trB = new_tracing("ch18-crash-B")
    clockB.now = clock.now + int(6.0e9)
    events = store.events()
    tp = [e["detail"]["traceparent"] for e in events if e["kind"] == "trace_context" and e["op_id"] == "op-0002"][0]
    _, tid, sid, _ = tp.split("-")
    rspan = trB.start("reconcile run-13", attributes={"itl.component": "reconciler", "itl.run.id": RUN13})
    before = store.op("op-0002")["status"]
    s3 = trB.start("reconcile create_export", links=[{"traceId": tid, "spanId": sid,
                                                       "attributes": {"itl.link.reason": "operation dispatched by an "
                                                                                         "earlier process"}}],
                   attributes={"itl.op.id": "op-0002", "itl.recovery.class": store.op("op-0002")["recovery_class"],
                               "itl.op.state_before": before})
    decisions = r13.Reconciler(store, r13.Executor(store, svc), svc, rclock).reconcile_all()
    clockB.advance(0.3)
    after = store.op("op-0002")
    s3["attributes"].update({"itl.recovery.decision": dict(decisions)["op-0002"], "itl.op.state_after": after["status"],
                             "itl.receipt.replayed": after["receipt"].get("replayed")})
    trB.end(s3)
    trB.end(rspan)
    procB.flush()
    exports = r13.oracle_rows(os.path.join(root, "export.db"), "exports")
    return {"process_A_exported_spans": exported_A, "process_A_lost": lost,
            "record_events": [(e["op_id"], e["kind"]) for e in events],
            "op_0002_state_at_restart": before, "decisions": decisions,
            "op_0002_state_after": after["status"], "exports_at_service": [r["export_id"] for r in exports],
            "process_B_spans": [s["name"] for s in expB.spans],
            "link_from_B": {"traceId": tid, "spanId": sid,
                            "target_exported": any(s["spanId"] == sid for s in expA.spans)},
            "spans_B": expB.spans}


# =============================================================================
# Experiment 6: delegation spans (Chapter 15's stale reviewer)
# =============================================================================
def delegation_experiment():
    store = d15.RevisionStore()
    log = d15.EvidenceLog()
    TZ_COV = ["pdt_evening", "pdt_after_local_midnight", "pst_evening", "dst_fall_back_evening",
              "dst_spring_forward_evening"]
    obj = ("Check revision {rev} of report.py for timezone boundary errors in belongs_to_report and return "
           "concrete failing cases.")

    def contract(rev, cid):
        return d15.make_contract(cid, "tz-boundary-reviewer", obj.format(rev=rev), rev, store,
                                 ["read_file", "run_cases"], TZ_COV, 6000, worker_version="1.2.0")
    c_old, c_new = contract("r4", "review-tz-r4"), contract("r5", "review-tz-r5-redispatch")
    o_old = d15.run_worker(c_old, d15.script_tz(c_old, d15.BOUNDARY_CASES), store, log)
    o_new = d15.run_worker(c_new, d15.script_tz(c_new, d15.BOUNDARY_CASES), store, log)
    store.close()
    clock, exp, proc, tr = new_tracing("ch18-delegation")
    t0 = clock.now
    root = tr.start("invoke_agent review-coordinator", attributes={"gen_ai.operation.name": "invoke_agent",
                                                                   "gen_ai.agent.name": "review-coordinator",
                                                                   "itl.revision.current": "r4"})

    def worker_span(o, start_s):
        clock.now = t0 + int(start_s * 1e9)
        w = tr.start("invoke_agent tz-boundary-reviewer", parent=root, attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": o["worker"],
            "gen_ai.agent.version": o["worker_version"], "itl.contract.id": o["contract_id"],
            "itl.contract.version": o["contract_version"], "itl.contract.input_revision": o["input_revision"]})
        for st in o["ledger"]["steps"]:
            if st["kind"] == "model":
                s = tr.start(f"chat {MODEL}", kind="CLIENT", parent=w, attributes={
                    "gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": st["tokens_in"],
                    "gen_ai.usage.output_tokens": st["tokens_out"]})
                clock.advance(st["seconds"])
            else:
                s = tr.start(f"execute_tool {st['name']}", parent=w, attributes={
                    "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": st["name"]})
                clock.advance(st["seconds"])
            tr.end(s)
        w["attributes"].update({"itl.result.verdict": o["result"]["verdict"], "itl.result.status": o["status"]})
        tr.end(w)
        return w
    w_old = worker_span(o_old, 0.0)
    clock.now = t0 + int(3.0e9)
    tr.event(root, "artifact.revised", {"itl.revision.from": "r4", "itl.revision.to": "r5"})
    root["attributes"]["itl.revision.current"] = "r5"
    w_new = worker_span(o_new, o_old["ledger"]["sim_seconds"])
    integ = d15.integrate([o_old, o_new], "r5", TZ_COV)
    naive = d15.naive_integrate([o_old, o_new])
    s = tr.start("integrate review-coordinator", parent=root, links=[
        {"traceId": w["traceId"], "spanId": w["spanId"], "attributes": {"itl.contract.input_revision":
                                                                         w["attributes"]["itl.contract.input_revision"]}}
        for w in (w_old, w_new)], attributes={
        "itl.component": "Control", "itl.revision.current": "r5", "itl.integrate.admitted": integ["admitted"],
        "itl.integrate.stale": [x["reviewed"] for x in integ["stale"]], "itl.integrate.verdict": integ["verdict"]})
    clock.advance(0.9)
    tr.end(s)
    tr.end(root)
    proc.flush()
    spans = exp.spans
    stale_workers = [w["attributes"]["itl.contract.id"] for w in spans if w["name"].startswith("invoke_agent tz")
                     and w["attributes"]["itl.contract.input_revision"] != "r5"]
    return {"spans": spans, "tree": tree_lines(spans, attrs=("itl.contract.input_revision", "itl.result.verdict",
                                                             "itl.integrate.verdict")),
            "stale_workers_by_attribute": stale_workers, "integration": integ["verdict"],
            "naive_integration": naive["verdict"], "worker_seconds": [o_old["ledger"]["sim_seconds"],
                                                                     o_new["ledger"]["sim_seconds"]]}


# =============================================================================
# Experiment 7: a simulated fleet
# =============================================================================
PRICE_IN, PRICE_OUT = 3.0 / 1e6, 15.0 / 1e6        # invented prices, dollars per token


def fleet_trace(rng, seed, i, version, category):
    clock, exp, proc, tr = new_tracing(f"fleet-{seed}-{i}", batch_size=10_000)
    run_id = f"run-{seed}-{i:05d}"
    root = tr.start(f"invoke_agent {AGENT}", attributes={"gen_ai.operation.name": "invoke_agent",
                                                         "gen_ai.agent.name": AGENT, "gen_ai.agent.version": version,
                                                         "itl.run.id": run_id})
    bill, req_ids = [], []
    v_report = "78721bd28acc"
    covered = None
    ncalls = rng.randint(4, 9)

    def chat(k, fail=None):
        tin, tout = rng.randint(900, 5200), rng.randint(40, 400)
        req = f"req-{run_id}-{k}"
        req_ids.append(req)
        s = tr.start(f"chat {MODEL}", kind="CLIENT", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.request.model": MODEL, "itl.request.id": req,
            "gen_ai.usage.input_tokens": tin})
        if fail:
            clock.advance(30.0)
            s["attributes"]["error.type"] = fail
            tr.end(s, "ERROR", fail)
            bill.append({"request_id": req, "response_id": None, "input_tokens": tin, "output_tokens": 0})
            return
        clock.advance(d15.model_seconds(tin, tout))
        s["attributes"].update({"gen_ai.usage.output_tokens": tout, "gen_ai.response.id": f"resp-{run_id}-{k}"})
        tr.end(s)
        bill.append({"request_id": req, "response_id": f"resp-{run_id}-{k}", "input_tokens": tin,
                     "output_tokens": tout})

    def tool(name, k, fail=None, after=None):
        s = tr.start(f"execute_tool {name}", attributes={"gen_ai.operation.name": "execute_tool",
                                                         "gen_ai.tool.name": name, "gen_ai.tool.call.id": f"c{k}"})
        clock.advance(TOOL_S.get(name, 0.3) * rng.uniform(0.7, 1.4))
        if name == "run_tests":
            s["attributes"]["itl.evidence.covers.report.py"] = v_report
            s["attributes"]["itl.observation.status"] = "passed"
        if after:
            s["attributes"]["itl.artifact.version_after.report.py"] = after
        if fail:
            s["attributes"]["error.type"] = fail
            tr.end(s, "ERROR", fail)
        else:
            tr.end(s)

    for k in range(1, ncalls + 1):
        if category == "recovered_error" and k == 2:
            chat(k * 10, fail="timeout")
        chat(k)
        if k == 1:
            tool("read_file", k)
        elif k == ncalls - 2:
            v_report = sha12(f"{run_id}-fix")
            tool("replace_text", k, after=v_report)
        elif k == ncalls - 1:
            if category == "tool_failure":
                tool("run_tests", k, fail="sandbox_unavailable")
                break
            tool("run_tests", k)
            covered = v_report
        elif k == ncalls and category == "stale_evidence":
            v_report = sha12(f"{run_id}-doc")
            tool("replace_text", k, after=v_report)
        elif k < ncalls:
            tool("read_file", k)
        if category in ("budget_exhausted", "waiting_for_person") and k == ncalls - 2:
            break
    if category == "waiting_for_person":
        clock.advance(rng.uniform(120, 3600))
    outcome = {"verified": "verified", "recovered_error": "verified", "tool_failure": "failed",
               "budget_exhausted": "budget_exhausted", "stale_evidence": "completed_unverified",
               "waiting_for_person": "waiting_for_person"}[category]
    if category in ("verified", "recovered_error", "stale_evidence"):
        v = tr.start(f"verify {AGENT}", attributes={"itl.verify.performed": version != "1.3.2",
                                                     "itl.verify.result": "accepted"})
        clock.advance(0.004)
        tr.end(v)
    root["attributes"]["itl.run.outcome"] = outcome
    tr.end(root, "ERROR" if category == "tool_failure" else None, "tool_failure" if category == "tool_failure" else None)
    proc.flush()
    tokens_in = sum(b["input_tokens"] for b in bill)
    tokens_out = sum(b["output_tokens"] for b in bill)
    record = {"run_id": run_id, "version": version, "category": category, "outcome": outcome,
              "request_ids": req_ids, "cost": tokens_in * PRICE_IN + tokens_out * PRICE_OUT,
              "evidence_current": covered == v_report if covered else None}
    return exp.spans, record, bill


FLEET_RATES = {"stale_given_old": 0.03, "old_version_share": 0.10, "tool_failure": 0.02,
               "budget_exhausted": 0.01, "recovered_error": 0.08, "waiting_for_person": 0.02}
RATIO = 0.10


def fleet(seed, n, keep_traces=False):
    rng = random.Random(seed)
    counts = {}
    kept = {p: {} for p in ("head_10pct", "tail_errors", "tail_outcome")}
    spans_kept = {p: 0 for p in kept}
    records, bills, traces_kept = [], [], {p: [] for p in kept}
    total_spans = 0
    lat = []
    for i in range(n):
        version = "1.3.2" if rng.random() < FLEET_RATES["old_version_share"] else AGENT_VERSION
        u = rng.random()
        if version == "1.3.2" and rng.random() < FLEET_RATES["stale_given_old"]:
            cat = "stale_evidence"
        elif u < FLEET_RATES["tool_failure"]:
            cat = "tool_failure"
        elif u < FLEET_RATES["tool_failure"] + FLEET_RATES["budget_exhausted"]:
            cat = "budget_exhausted"
        elif u < 0.03 + FLEET_RATES["recovered_error"]:
            cat = "recovered_error"
        elif u < 0.11 + FLEET_RATES["waiting_for_person"]:
            cat = "waiting_for_person"
        else:
            cat = "verified"
        spans, rec, bill = fleet_trace(rng, seed, i, version, cat)
        records.append(rec)
        bills.extend(bill)
        counts[cat] = counts.get(cat, 0) + 1
        total_spans += len(spans)
        root = root_of(spans)
        lat.append(seconds(root))
        rec["root_error"] = root["status"]["code"] == "ERROR"
        rec["any_error_span"] = has_error(spans)
        rec["latency_s"] = seconds(root)
        decisions = {"head_10pct": head_keep(root["traceId"], RATIO),
                     "tail_errors": tail_error_keep(spans, RATIO),
                     "tail_outcome": tail_outcome_keep(spans, RATIO)}
        for p, keep in decisions.items():
            if keep:
                kept[p][cat] = kept[p].get(cat, 0) + 1
                spans_kept[p] += len(spans)
                if keep_traces:
                    traces_kept[p].append(spans)
        rec["kept"] = decisions
    return {"counts": counts, "kept": kept, "spans_kept": spans_kept, "total_spans": total_spans,
            "records": records, "bills": bills, "latency": lat, "traces_kept": traces_kept}


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--fleet-size", type=int, default=20000)
    args = ap.parse_args()
    out = args.output
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    summary = {}

    # ---- 1 ---------------------------------------------------------------------
    main_run = drive_ch12_run(out, "main")
    spans = main_run["spans"]
    write_run(out, "main_trace.json", spans)
    write_run(out, "main_logs.txt", "\n".join(main_run["logs"]) + "\n")
    tree = tree_lines(spans, attrs=("gen_ai.tool.call.id", "itl.record.model_call", "itl.verify.result",
                                    "itl.observation.status"))
    write_run(out, "main_tree.txt", "\n".join(tree) + "\n")
    q = five_questions(spans, main_run["store"])
    one, logs = flawed_telemetry(spans, main_run["logs"])
    qf = five_questions(one, main_run["store"])
    qf_detail = {"spans": len(one), "log_lines": len(logs),
                 "log_lines_naming_a_version": sum(1 for l in logs if "339d3a9f2656" in l or "8e8137d1687e" in l),
                 "log_lines_with_canary": sum(1 for l in logs if CANARY in l),
                 "trace_spans_with_canary": sum(1 for s in spans if CANARY in canon(s))}
    write_run(out, "flawed_telemetry.json", {"spans": one, "logs": logs})
    kinds = {}
    for s in spans:
        k = s["attributes"].get("gen_ai.operation.name") or s["name"].split(" ")[0]
        kinds[k] = kinds.get(k, 0) + 1
    summary["main"] = {"spans": len(spans), "span_kinds": kinds, "traces": len(by_trace(spans)),
                       "root_seconds": round(seconds(root_of(spans)), 3), "five_questions": q,
                       "five_questions_answered": sum(v["answered"] for v in q.values()),
                       "flawed_answered": sum(v["answered"] for v in qf.values()), "flawed": qf_detail,
                       "verdicts": main_run["verdicts"], "orphans": rule_orphans(spans),
                       "critical_path": summarize_path(spans)["by_kind"]}
    print("== 1 the Chapter 12 run, traced ==")
    print(f"spans: {len(spans)} in {len(by_trace(spans))} trace; kinds: {kinds}")
    for line in tree:
        print("  " + line)
    print("five questions (trace -> records):")
    for k, v in q.items():
        print(f"  {k}: answered={v['answered']}  " + json.dumps({x: y for x, y in v.items() if x != 'answered'})[:230])
    print(f"flawed design: {qf_detail['spans']} span, {qf_detail['log_lines']} log lines, questions answered "
          f"{summary['main']['flawed_answered']}/5; log lines naming an artifact version: "
          f"{qf_detail['log_lines_naming_a_version']}; log lines with the credential canary: "
          f"{qf_detail['log_lines_with_canary']}; trace spans with it: {qf_detail['trace_spans_with_canary']}")
    print("sample log lines:")
    for l in logs[:4] + logs[-3:]:
        print("  " + l)
    print("critical path by kind:", summary["main"]["critical_path"])
    print("orphaned tool results:", summary["main"]["orphans"])

    # ---- 2 ---------------------------------------------------------------------
    print("\n== 2 the slow run ==")
    slow = {}
    for mode in ("scan", "indexed"):
        sp, answer = slow_run(mode)
        write_run(out, f"slow_{mode}_trace.json", sp)
        s = summarize_path(sp)
        s["tree"] = tree_lines(sp, attrs=("itl.search.files_scanned",))
        slow[mode] = s
        print(f"[{mode}] agent latency {s['agent_latency_s']} s; sum of the root's child spans "
              f"{s['span_seconds_sum']} s")
        for line in s["tree"]:
            print("  " + line)
        print("  critical path:")
        for name, secs, _ in s["segments"]:
            print(f"    {secs:7.3f} s  {name}")
        print("  by kind:", s["by_kind"])
    summary["slow"] = slow

    # ---- 3 ---------------------------------------------------------------------
    print("\n== 3 a failure with no error ==")
    stale = drive_ch12_run(out, "stale", verify=False, script=stale_responses())
    ssp = stale["spans"]
    write_run(out, "stale_trace.json", ssp)
    write_run(out, "stale_logs.txt", "\n".join(stale["logs"]) + "\n")
    rule = rule_stale_evidence(ssp)
    ok, reason, current = stale["rt"].evidence_for_completion()
    levels = {}
    for l in stale["logs"]:
        lv = l.split()[1]
        levels[lv] = levels.get(lv, 0) + 1
    summary["stale"] = {"log_levels": levels, "error_spans": sum(1 for s in ssp if s["status"]["code"] == "ERROR"),
                        "error_span_names": [s["name"] for s in ssp if s["status"]["code"] == "ERROR"],
                        "root_outcome": root_of(ssp)["attributes"]["itl.run.outcome"],
                        "trace_rule": rule, "records_check": {"ok": ok, "reason": reason},
                        "logs_mentioning_passed": [l for l in stale["logs"] if "passed" in l],
                        "final_answer": stale["store"].conn.execute(
                            "SELECT response FROM model_calls ORDER BY ordinal DESC LIMIT 1").fetchone()[0][:200]}
    print("log levels:", levels)
    print("error spans:", summary["stale"]["error_span_names"])
    print("root outcome attribute:", summary["stale"]["root_outcome"])
    print("trace rule:", json.dumps(rule))
    print("records check:", ok, reason)
    for l in stale["logs"][-6:]:
        print("  " + l)

    # ---- 4 ---------------------------------------------------------------------
    print("\n== 4 content capture and replay ==")
    variants, cap = capture_variants(main_run)
    for mode, vs in variants.items():
        write_run(os.path.join(out, "capture"), f"trace_{mode}.json", vs)
    for mode in ("off", "structure", "full_truncated", "full"):
        r = cap[mode]
        print(f"  {mode:15} content bytes {r['content_bytes']:6}  source text present {str(r['source_text_in_telemetry']):5}"
              f"  parsable {r['json_parsable']}/{r['chat_spans']}  exact input {r['exact_input_recovered']}/"
              f"{r['chat_spans']}  structure {r['structure_recovered']}/{r['chat_spans']}")
    print("  model input sizes (characters):", cap["full"]["sizes"])
    print("  records:", cap["records"])
    summary["capture"] = cap

    # ---- 5 ---------------------------------------------------------------------
    print("\n== 5 telemetry across a crash ==")
    cr = crash_experiment(out)
    write_run(out, "crash_trace_B.json", cr.pop("spans_B"))
    for k, v in cr.items():
        print(f"  {k}: {v}")
    summary["crash"] = cr

    # ---- 6 ---------------------------------------------------------------------
    print("\n== 6 delegation spans ==")
    dl = delegation_experiment()
    write_run(out, "delegation_trace.json", dl.pop("spans"))
    for line in dl["tree"]:
        print("  " + line)
    print("  stale workers found by attribute:", dl["stale_workers_by_attribute"])
    print("  integration:", dl["integration"], "| naive coordinator:", dl["naive_integration"])
    summary["delegation"] = dl

    # ---- 7 ---------------------------------------------------------------------
    print("\n== 7 a simulated fleet ==")
    n = args.fleet_size
    f1 = fleet(1, n, keep_traces=True)
    recs = f1["records"]
    cats = ["verified", "recovered_error", "waiting_for_person", "tool_failure", "budget_exhausted", "stale_evidence"]
    print(f"runs: {n}; spans: {f1['total_spans']}; categories: " + ", ".join(f"{c} {f1['counts'].get(c, 0)}" for c in cats))
    print(f"{'policy':14}" + "".join(f"{c[:14]:>16}" for c in cats) + f"{'traces':>9}{'spans':>9}")
    for p in ("head_10pct", "tail_errors", "tail_outcome"):
        k = f1["kept"][p]
        print(f"{p:14}" + "".join(f"{k.get(c, 0):>7}/{f1['counts'].get(c, 0):<8}" for c in cats)
              + f"{sum(k.values()):>9}{f1['spans_kept'][p]:>9}")
    stale_err = sum(1 for r in recs if r["category"] == "stale_evidence" and r["any_error_span"])
    budget_err = sum(1 for r in recs if r["category"] == "budget_exhausted" and r["any_error_span"])
    print(f"stale-evidence runs with any ERROR span: {stale_err}; budget-exhausted runs with any ERROR span: {budget_err}")
    seeds = {}
    for sd in (2, 3, 4, 5, 6):
        fx = fleet(sd, n)
        seeds[sd] = {p: (fx["kept"][p].get("stale_evidence", 0), fx["counts"].get("stale_evidence", 0))
                     for p in fx["kept"]}
        seeds[sd]["spans"] = {p: fx["spans_kept"][p] for p in fx["kept"]}
        seeds[sd]["total_spans"] = fx["total_spans"]
    for sd, v in seeds.items():
        print(f"  seed {sd}: stale kept head {v['head_10pct']}, tail_errors {v['tail_errors']}, "
              f"tail_outcome {v['tail_outcome']}; spans kept {v['spans']} of {v['total_spans']}")

    # dashboards
    lat = f1["latency"]
    status_ok = sum(1 for r in recs if not r["root_error"])
    verified = [r for r in recs if r["outcome"] == "verified"]
    by_outcome = {}
    for r in recs:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    total_cost = sum(r["cost"] for r in recs)
    completed_claims = [r for r in recs if r["outcome"] in ("verified", "completed_unverified")]
    dash_flawed = {"agent_latency_p50_s": pct(lat, 0.5), "agent_latency_p95_s": pct(lat, 0.95),
                   "success_rate_by_span_status": round(status_ok / n, 4),
                   "cost_per_completed_run": round(total_cost / len(completed_claims), 5)}
    by_version = {}
    for r in recs:
        v = by_version.setdefault(r["version"], {"runs": 0, "verified": 0, "completed_unverified": 0})
        v["runs"] += 1
        if r["outcome"] in v:
            v[r["outcome"]] += 1
    dash_outcome = {"verified_rate": round(len(verified) / n, 4), "outcomes": by_outcome,
                    "cost_per_verified_success": round(total_cost / len(verified), 5),
                    "by_version": by_version,
                    "latency_p50_verified_s": pct([r["latency_s"] for r in verified], 0.5),
                    "latency_p95_verified_s": pct([r["latency_s"] for r in verified], 0.95),
                    "latency_p95_waiting_s": pct([r["latency_s"] for r in recs if r["outcome"] == "waiting_for_person"], 0.95)}
    print("flawed dashboard:", dash_flawed)
    print("outcome dashboard:", json.dumps(dash_outcome))

    # bills to runs
    bills = f1["bills"]
    req_to_run = {rid: r["run_id"] for r in recs for rid in r["request_ids"]}
    resp_in_sampled = {}
    for spans_ in f1["traces_kept"]["head_10pct"]:
        for s in spans_:
            if s["attributes"].get("gen_ai.response.id"):
                resp_in_sampled[s["attributes"]["gen_ai.response.id"]] = s["traceId"]
    billed_cost = sum(b["input_tokens"] * PRICE_IN + b["output_tokens"] * PRICE_OUT for b in bills)
    by_req = sum(1 for b in bills if b["request_id"] in req_to_run)
    by_resp_all = sum(1 for b in bills if b["response_id"])
    by_resp_sampled = sum(1 for b in bills if b["response_id"] in resp_in_sampled)
    timeouts = [b for b in bills if b["response_id"] is None]
    cost_timeouts = sum(b["input_tokens"] * PRICE_IN for b in timeouts)
    bill_report = {"bill_lines": len(bills), "billed_cost": round(billed_cost, 2),
                   "joined_by_request_id_in_records": by_req,
                   "joinable_by_response_id_unsampled": by_resp_all,
                   "joined_by_response_id_head_sampled": by_resp_sampled,
                   "lines_without_response_id": len(timeouts), "cost_of_those_lines": round(cost_timeouts, 2),
                   "cost_on_stale_runs": round(sum(r["cost"] for r in recs if r["category"] == "stale_evidence"), 2)}
    print("bills to runs:", bill_report)
    summary["fleet"] = {"n": n, "counts": f1["counts"], "kept": f1["kept"], "spans_kept": f1["spans_kept"],
                        "total_spans": f1["total_spans"], "stale_with_error_span": stale_err,
                        "budget_with_error_span": budget_err, "seeds": seeds, "dashboard_flawed": dash_flawed,
                        "dashboard_outcome": dash_outcome, "bills": bill_report, "rates": FLEET_RATES,
                        "ratio": RATIO, "prices_per_token": [PRICE_IN, PRICE_OUT]}
    write_run(out, "fleet_records_sample.json", recs[:20])
    write_run(out, "summary.json", summary)
    main_run["ws"].close()
    stale["ws"].close()


if __name__ == "__main__":
    main()
