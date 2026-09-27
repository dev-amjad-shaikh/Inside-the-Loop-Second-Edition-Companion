"""Chapter 6 helpers: an execution record, context builders over it, and context tests.

Nothing here calls a model. Every builder is a pure function of the record store,
the call number and its policy, so the same records and policy give the same request.

Token counts are APPROXIMATE: approx_tokens() charges ceil(UTF-8 bytes / 4) of the
canonical JSON of each message or definition. That is the same bytes-per-token rule
several runtimes use as their fallback estimate. It is not any provider's tokenizer;
a real count comes from the provider's usage fields.
"""
import ast
import copy
import hashlib
import json
import math
import re
import sqlite3

from .core import content_version

# ----------------------------------------------------------------------------------
# Approximate token counting
# ----------------------------------------------------------------------------------

COUNTER_ID = "approx-bytes/4 (canonical JSON, UTF-8); approximate, not a tokenizer"


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def approx_tokens(obj):
    text = obj if isinstance(obj, str) else canonical(obj)
    return math.ceil(len(text.encode("utf-8")) / 4)


def sha12(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------------------
# The execution record
# ----------------------------------------------------------------------------------

class RecordStore:
    """Append-only records of one run. Each record keeps the verbatim message plus
    the metadata the context builder needs: source, revision, scope and trust."""

    def __init__(self, run_id, scope):
        self.run_id = run_id
        self.scope = scope
        self.records = []
        self.revisions = {}      # path -> current content version, as the runtime knows it
        self.compactions = []    # recorded compaction decisions

    def _add(self, **fields):
        rid = "R%02d" % (len(self.records) + 1)
        rec = {"id": rid, "run": self.run_id, "scope": self.scope, "seq": len(self.records) + 1}
        rec.update(fields)
        self.records.append(rec)
        return rec

    def add_instruction(self, text):
        return self._add(kind="instruction", trust="runtime", source="application",
                         message={"role": "system", "content": text})

    def add_task(self, revision, text, constraints, goals):
        return self._add(kind="task", trust="user", source="human interface",
                         task_revision=revision, constraints=constraints, goals=goals,
                         message={"role": "user", "content": text})

    def add_assistant(self, reply, call_no):
        return self._add(kind="assistant", trust="model output", source="model call %d" % call_no,
                         call_no=call_no, message=copy.deepcopy(reply))

    def add_result(self, call, observation):
        rec = self._add(kind="tool_result", trust="workspace data" if call["name"] != "read_record" else "execution record",
                        source="%s(%s)" % (call["name"], canonical(call["arguments"])[1:-1]),
                        tool=call["name"], call_id=call["id"], arguments=copy.deepcopy(call["arguments"]),
                        message={"role": "tool", "tool_call_id": call["id"], "content": observation})
        path = call["arguments"].get("path") if isinstance(call["arguments"], dict) else None
        if call["name"] == "read_file" and observation.get("status") == "ok":
            rec["path"] = path
            rec["revision"] = content_version(observation["text"])
            self.revisions[path] = rec["revision"]
        elif call["name"] == "replace_text" and observation.get("status") == "ok":
            rec["path"] = path
            rec["revision"] = observation["after_version"]
            rec["before_revision"] = observation["before_version"]
            self.revisions[path] = observation["after_version"]
        elif call["name"] in ("run_tests", "run_suite"):
            rec["tested_revisions"] = dict(self.revisions)
        return rec

    def get(self, rid):
        for r in self.records:
            if r["id"] == rid:
                return r
        return None

    def dump(self):
        return self.records


def groups_of(records):
    """Protocol groups: an assistant message that requests tools together with all of
    its results; every other message is a group of one. Compaction moves whole groups."""
    groups, open_group, waiting = [], None, set()
    for rec in records:
        if rec["kind"] == "instruction":
            continue
        msg = rec["message"]
        if rec["kind"] == "assistant" and msg.get("tool_calls"):
            open_group = [rec]
            waiting = {c["id"] for c in msg["tool_calls"]}
            groups.append(open_group)
        elif rec["kind"] == "tool_result" and open_group is not None and rec["call_id"] in waiting:
            open_group.append(rec)
            waiting.discard(rec["call_id"])
            if not waiting:
                open_group = None
        else:
            open_group = None
            groups.append([rec])
    return groups


# ----------------------------------------------------------------------------------
# Checks used by the context tests and by the runtime
# ----------------------------------------------------------------------------------

def protocol_problems(messages):
    """Every tool result must follow the assistant message that requested it, and every
    requested call must have a result before the next assistant message."""
    problems, pending = [], {}
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            for cid in pending:
                problems.append("call %s has no result" % cid)
            pending = {c["id"]: i for c in (m.get("tool_calls") or [])}
        elif m["role"] == "tool":
            cid = m.get("tool_call_id")
            if cid not in pending:
                problems.append("result for %s has no preceding request" % cid)
            else:
                pending.pop(cid)
        else:
            for cid in pending:
                problems.append("call %s has no result" % cid)
            pending = {}
    for cid in pending:
        problems.append("call %s has no result" % cid)
    return problems


def signature_of(source, name="belongs_to_report"):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            names = [a.arg for a in args.args]
            defaults = [None] * (len(names) - len(args.defaults)) + list(args.defaults)
            parts = [n if d is None else "%s=%s" % (n, ast.unparse(d)) for n, d in zip(names, defaults)]
            return "%s(%s)" % (name, ", ".join(parts))
    return None


def simulate_edits(files, calls):
    """Apply proposed replace_text calls to copies of file texts, without touching the workspace."""
    out = dict(files)
    for c in calls:
        if c["name"] != "replace_text":
            continue
        a = c["arguments"]
        text = out.get(a["path"], "")
        if text.count(a["old"]) == 1:
            out[a["path"]] = text.replace(a["old"], a["new"], 1)
    return out


# ----------------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------------

def first_sentence(text):
    text = " ".join(str(text).split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return m.group(1) if m else text


def prov(rec, transform, **extra):
    p = {"records": [rec["id"]], "transform": transform, "source": rec.get("source"),
         "revision": rec.get("revision"), "run": rec["run"], "scope": rec["scope"], "trust": rec["trust"]}
    p.update(extra)
    return p


def digest_suite_result(rec, max_kept=40):
    """Replace a long test log by its counts, every non-'ok' line verbatim, and a pointer."""
    obs = rec["message"]["content"]
    lines = obs.get("log", "").splitlines()
    ok = [l for l in lines if l.endswith(" ... ok")]
    kept = [l for l in lines if l.strip() and not l.endswith(" ... ok") and set(l.strip()) - set("=-")]
    body = {k: v for k, v in obs.items() if k != "log"}
    body["log_digest"] = {"lines_total": len(lines), "ok_lines_omitted": len(ok), "kept_lines": kept[:max_kept]}
    body["full_log"] = "read_record(record_id='%s')" % rec["id"]
    return {"role": "tool", "tool_call_id": rec["message"]["tool_call_id"], "content": body}


def truncate_result(rec, cap_tokens):
    obs = copy.deepcopy(rec["message"]["content"])
    key = "log" if "log" in obs else ("text" if "text" in obs else None)
    if key is None:
        return rec["message"], False
    limit = cap_tokens * 4
    if len(obs[key].encode("utf-8")) <= limit:
        return rec["message"], False
    obs[key] = obs[key].encode("utf-8")[:limit].decode("utf-8", "ignore") + " [truncated]"
    return {"role": "tool", "tool_call_id": rec["message"]["tool_call_id"], "content": obs}, True


# ----------------------------------------------------------------------------------
# Shared observation table (several runs and workspaces) for scope and freshness tests
# ----------------------------------------------------------------------------------

def open_observations(path, seed):
    db = sqlite3.connect(path)
    db.execute("DROP TABLE IF EXISTS observations")
    db.execute("""CREATE TABLE observations (
        obs_id TEXT PRIMARY KEY, run TEXT, scope TEXT, path TEXT, revision TEXT,
        workspace_seq INTEGER, recorded_at TEXT, value TEXT, origin TEXT)""")
    for row in seed:
        db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
                   (row["obs_id"], row["run"], row["scope"], row["path"], row["revision"],
                    row["workspace_seq"], row["recorded_at"], canonical(row["value"]), row["origin"]))
    db.commit()
    return db


def config_naive(db, path="settings.json"):
    """Common mistake: a cache keyed by path, newest observation first, any workspace."""
    rows = db.execute("SELECT obs_id, scope, revision, value FROM observations WHERE path=? "
                      "ORDER BY recorded_at DESC", (path,)).fetchall()
    return [{"obs_id": r[0], "scope": r[1], "revision": r[2], "value": json.loads(r[3]),
             "status": "current" if i == 0 else "listed"} for i, r in enumerate(rows)]


def config_scoped(db, scope, path="settings.json"):
    """Filter by the run's scope first, then order by the workspace's own revision sequence."""
    rows = db.execute("SELECT obs_id, scope, revision, value FROM observations WHERE path=? AND scope=? "
                      "ORDER BY workspace_seq DESC", (path, scope)).fetchall()
    return [{"obs_id": r[0], "scope": r[1], "revision": r[2], "value": json.loads(r[3]),
             "status": "current" if i == 0 else "superseded"} for i, r in enumerate(rows)]


def excluded_by_scope(db, scope, path="settings.json"):
    return [r[0] for r in db.execute("SELECT obs_id FROM observations WHERE path=? AND scope<>? ORDER BY obs_id",
                                     (path, scope)).fetchall()]


# ----------------------------------------------------------------------------------
# Builders. Each returns a Build: messages, provenance per message, omissions, budget.
# ----------------------------------------------------------------------------------

WINDOW = 8192
OUTPUT_RESERVE = 1536
INPUT_ALLOWANCE = WINDOW - OUTPUT_RESERVE


def budget_report(sections, tools_tokens, allowance=INPUT_ALLOWANCE, **extra):
    # the input allowance is the window minus the reserve for the model's output
    used = tools_tokens + sum(s["tokens"] for s in sections)
    rep = {"counter": COUNTER_ID, "window": WINDOW, "output_reserve": OUTPUT_RESERVE,
           "input_allowance": allowance, "tool_definitions": tools_tokens,
           "sections": sections, "total_input": used, "headroom": allowance - used,
           "fits": used <= allowance}
    rep.update(extra)
    return rep


def finish(policy, call_no, messages, provenance, omissions, sections, tools_tokens, **extra):
    for m, p in zip(messages, provenance):
        p["tokens"] = approx_tokens(m)
    return {"policy": policy, "call": call_no, "messages": messages, "provenance": provenance,
            "omissions": omissions, "budget": budget_report(sections, tools_tokens, **extra),
            "protocol_problems": protocol_problems(messages), "request_sha": sha12(messages)}


def verbatim(store):
    msgs, provs = [], []
    for rec in store.records:
        msgs.append(copy.deepcopy(rec["message"]))
        provs.append(prov(rec, "verbatim"))
    return msgs, provs


class FullTranscriptBuilder:
    policy = {"id": "ctx-full-v1", "rule": "send every record verbatim, in order"}

    def build(self, store, call_no, tools_tokens):
        msgs, provs = verbatim(store)
        sections = [{"name": "transcript", "items": len(msgs), "tokens": sum(approx_tokens(m) for m in msgs)}]
        return finish(self.policy, call_no, msgs, provs, [], sections, tools_tokens)


class TruncateOldestBuilder:
    def __init__(self, allowance=INPUT_ALLOWANCE):
        self.allowance = allowance
        self.policy = {"id": "ctx-truncate-v1", "allowance": allowance,
                       "rule": "keep the system message; drop the oldest message until the request fits"}

    def build(self, store, call_no, tools_tokens):
        msgs, provs = verbatim(store)
        omissions = []
        while tools_tokens + sum(approx_tokens(m) for m in msgs) > self.allowance and len(msgs) > 1:
            m, p = msgs.pop(1), provs.pop(1)
            omissions.append({"record": p["records"][0], "action": "removed",
                              "reason": "oldest message after the system prompt; request over allowance",
                              "tokens": approx_tokens(m), "carried_by": None})
        sections = [{"name": "system", "items": 1, "tokens": approx_tokens(msgs[0])},
                    {"name": "kept history", "items": len(msgs) - 1,
                     "tokens": sum(approx_tokens(m) for m in msgs[1:])}]
        return finish(self.policy, call_no, msgs, provs, omissions, sections, tools_tokens, allowance=self.allowance)


class FreeTextSummaryBuilder:
    """Protect the head (system, first user message), keep the last K messages, cap long
    tool results by truncation, and replace the middle with a first-sentence summary.
    Adds a 'known configuration' line from a path-keyed observation cache."""

    def __init__(self, keep_recent=4, result_cap=1200, db=None):
        self.keep_recent = keep_recent
        self.result_cap = result_cap
        self.db = db
        self.policy = {"id": "ctx-freetext-v1", "keep_recent_messages": keep_recent,
                       "result_cap_tokens": result_cap,
                       "rule": "head + first-sentence summary of the middle + last K messages"}

    def build(self, store, call_no, tools_tokens):
        msgs, provs = verbatim(store)
        full = tools_tokens + sum(approx_tokens(m) for m in msgs)
        if full <= INPUT_ALLOWANCE:
            sections = [{"name": "transcript", "items": len(msgs), "tokens": full - tools_tokens}]
            return finish(self.policy, call_no, msgs, provs, [], sections, tools_tokens, compacted=False)
        recs = store.records
        head = recs[:2]
        tail = recs[max(2, len(recs) - self.keep_recent):]
        middle = recs[2:len(recs) - len(tail)]
        lines, omissions = [], []
        for rec in middle:
            m = rec["message"]
            if m["role"] == "user":
                lines.append("user: " + first_sentence(m["content"]))
            elif m["role"] == "assistant":
                if m.get("content"):
                    lines.append("assistant: " + first_sentence(m["content"]))
                if m.get("tool_calls"):
                    lines.append("assistant called " + ", ".join(c["name"] for c in m["tool_calls"]))
            else:
                lines.append("tool result: " + str(m["content"].get("status")))
            omissions.append({"record": rec["id"], "action": "summarized",
                              "reason": "outside head and last %d messages" % self.keep_recent,
                              "tokens": approx_tokens(m), "carried_by": "summary (first sentence)"})
        if self.db is not None:
            cfg = config_naive(self.db)
            lines.append("known configuration (latest observation): report_timezone = %s"
                         % cfg[0]["value"]["report_timezone"])
            for c in cfg:
                lines.append("observed %s report_timezone=%s" % (c["obs_id"], c["value"]["report_timezone"]))
        summary = {"role": "user", "content": "Summary of earlier conversation:\n" + "\n".join(lines)}
        out_m = [copy.deepcopy(r["message"]) for r in head] + [summary]
        out_p = [prov(r, "verbatim (head)") for r in head] + [
            {"records": [r["id"] for r in middle], "transform": "free-text summary", "source": "builder",
             "revision": None, "run": store.run_id, "scope": store.scope, "trust": "derived"}]
        for rec in tail:
            if rec["kind"] == "tool_result":
                m, cut = truncate_result(rec, self.result_cap)
                if cut:
                    omissions.append({"record": rec["id"], "action": "truncated",
                                      "reason": "result over %d tokens; kept its beginning" % self.result_cap,
                                      "tokens": approx_tokens(rec["message"]) - approx_tokens(m),
                                      "carried_by": None})
                out_m.append(m)
                out_p.append(prov(rec, "truncated" if cut else "verbatim"))
            else:
                out_m.append(copy.deepcopy(rec["message"]))
                out_p.append(prov(rec, "verbatim"))
        sections = [{"name": "head", "items": 2, "tokens": sum(approx_tokens(m) for m in out_m[:2])},
                    {"name": "summary", "items": 1, "tokens": approx_tokens(summary)},
                    {"name": "tail", "items": len(tail), "tokens": sum(approx_tokens(m) for m in out_m[3:])}]
        return finish(self.policy, call_no, out_m, out_p, omissions, sections, tools_tokens, compacted=True)


def summary_present(parts):
    return any(p["transform"].startswith("structured summary") for _, p in parts)


class StructuredBuilder:
    """Pinned task and constraints, a derived summary with record references, whole
    request/result groups, and digests for oversized results. Compaction fires at a
    trigger and compacts down to a lean tail, then the layout stays append-only."""

    def __init__(self, trigger=0.75, recent_budget=500, result_cap=1200, pinned_cap=450,
                 summary_cap=900, layout="task_last", db=None, status_line=False, allowance=INPUT_ALLOWANCE,
                 pin_task=True, freeze_summary=True):
        self.allowance = allowance
        self.freeze_summary = freeze_summary
        self.frozen = None
        self.pin_task = pin_task
        self.trigger = trigger
        self.recent_budget = recent_budget
        self.result_cap = result_cap
        self.pinned_cap = pinned_cap
        self.summary_cap = summary_cap
        self.layout = layout
        self.db = db
        self.status_line = status_line
        self.policy = {"id": "ctx-structured-v1", "trigger_fraction": trigger,
                       "recent_budget_tokens": recent_budget, "result_cap_tokens": result_cap,
                       "pinned_cap_tokens": pinned_cap, "summary_cap_tokens": summary_cap,
                       "layout": layout, "status_line": status_line, "allowance": allowance,
                       "pin_task": pin_task, "freeze_summary": freeze_summary}
        self.policy["hash"] = sha12(self.policy)
        self.watermark = None    # index of first verbatim group; set by a compaction

    # -- pinned task ------------------------------------------------------------
    def pinned(self, store):
        tasks = [r for r in store.records if r["kind"] == "task"]
        cur = tasks[-1]
        lines = ["Runtime note from the context builder (derived from task records, not written by the user).",
                 "Current task revision %d." % cur["task_revision"]]
        for t in tasks:
            for g in t["goals"]:
                lines.append("Goal [%s, revision %d]: %s" % (t["id"], t["task_revision"], g))
        lines.append("Constraints in force:")
        for t in tasks:
            for c in t["constraints"]:
                extra = ""
                if c.get("contract"):
                    extra = " Contract: %s." % c["contract"]
                lines.append("- %s %s%s [%s, revision %d]" % (c["id"], c["text"], extra, t["id"], t["task_revision"]))
        msg = {"role": "user", "content": "\n".join(lines)}
        p = {"records": [t["id"] for t in tasks], "transform": "pinned (derived)", "source": "task records",
             "revision": "task r%d" % cur["task_revision"], "run": store.run_id, "scope": store.scope,
             "trust": "user (via human interface)"}
        if approx_tokens(msg) > self.pinned_cap:
            raise ValueError("pinned task exceeds its cap: a configuration error, never a truncation")
        return msg, p

    # -- derived summary ----------------------------------------------------------
    def summarize(self, store, groups, call_no):
        lines = ["Derived summary of records %s-%s, written by the context builder at call %d (policy %s). "
                 "Not model-written. Each line names its source record; read_record(record_id) returns the original."
                 % (groups[0][0]["id"], groups[-1][-1]["id"], call_no, self.policy["id"])]
        covered = []
        for g in groups:
            for rec in g:
                covered.append(rec["id"])
                m = rec["message"]
                if rec["kind"] == "task" and self.pin_task:
                    lines.append("- Task revision %d recorded [%s]; carried in the pinned task note."
                                 % (rec["task_revision"], rec["id"]))
                elif rec["kind"] == "task":
                    lines.append("- Task revision %d [%s]: %s" % (rec["task_revision"], rec["id"],
                                                                first_sentence(m["content"])))
                elif rec["kind"] == "assistant":
                    if m.get("content"):
                        lines.append("- Model claim (unverified text) [%s]: %s" % (rec["id"], first_sentence(m["content"])))
                elif rec["kind"] == "tool_result":
                    o = m["content"]
                    if rec["tool"] == "read_file" and o.get("status") == "ok":
                        cur = store.revisions.get(rec["path"])
                        state = "still current" if cur == rec["revision"] else "superseded, now %s" % cur
                        line = "- Read %s@%s (%s) [%s]" % (rec["path"], rec["revision"], state, rec["id"])
                        if rec["path"].endswith(".json") and len(o["text"]) <= 200:
                            line += ": %s" % o["text"].strip()
                        lines.append(line)
                    elif rec["tool"] == "replace_text" and o.get("status") == "ok":
                        lines.append("- Edited %s %s -> %s by %s [%s]" % (rec["path"], rec["before_revision"],
                                                                         rec["revision"], rec["call_id"], rec["id"]))
                    elif o.get("status") == "input_error":
                        lines.append("- Call %s rejected before execution: %s [%s]" % (rec["call_id"], o["message"], rec["id"]))
                    elif rec["tool"] in ("run_tests", "run_suite"):
                        tested = rec.get("tested_revisions", {}).get("report.py")
                        lines.append("- Tests %s: %s run, %s failures, %s errors %s against report.py@%s [%s]"
                                     % (o.get("status"), o.get("tests_run"), o.get("failures"), o.get("errors"),
                                        o.get("failed_tests", []), tested, rec["id"]))
                    elif o.get("status") == "input_error":
                        lines.append("- Call %s rejected before execution: %s [%s]" % (rec["call_id"], o["message"], rec["id"]))
                    else:
                        lines.append("- %s -> %s [%s]" % (rec["source"], o.get("status"), rec["id"]))
        if self.db is not None:
            for c in config_scoped(self.db, store.scope):
                lines.append("- Configuration %s: report_timezone=%s (%s, %s)"
                             % (c["status"], c["value"]["report_timezone"], c["obs_id"], c["revision"]))
        msg = {"role": "user", "content": "\n".join(lines)}
        if approx_tokens(msg) > self.summary_cap:
            raise ValueError("summary exceeds its cap: raise the cap or compact less")
        p = {"records": covered, "transform": "structured summary (derived)", "source": "context builder",
             "revision": None, "run": store.run_id, "scope": store.scope, "trust": "derived"}
        return msg, p

    # -- groups -------------------------------------------------------------------
    def render_group(self, g, compacting, omissions):
        out = []
        for rec in g:
            m = rec["message"]
            if compacting and rec["kind"] == "tool_result" and approx_tokens(m) > self.result_cap:
                d = digest_suite_result(rec) if "log" in m["content"] else truncate_result(rec, self.result_cap)[0]
                omissions.append({"record": rec["id"], "action": "digested",
                                  "reason": "result over %d tokens; counts and non-ok lines kept" % self.result_cap,
                                  "tokens": approx_tokens(m) - approx_tokens(d),
                                  "carried_by": "digest; full text via read_record('%s')" % rec["id"]})
                out.append((d, prov(rec, "digest")))
            else:
                out.append((copy.deepcopy(m), prov(rec, "verbatim")))
        return out

    def build(self, store, call_no, tools_tokens):
        system = store.records[0]
        groups = groups_of(store.records)
        pinned_msg, pinned_p = self.pinned(store) if self.pin_task else (None, None)
        omissions = []
        status = None
        if self.status_line:
            last = [r for r in store.records if r["kind"] == "tool_result"]
            status = {"role": "user", "content": "Run status: model call %d; %d records; last result %s."
                      % (call_no, len(store.records), last[-1]["message"]["content"].get("status") if last else "none")}

        def assemble(watermark, compacting):
            om = []
            parts = [(copy.deepcopy(system["message"]), prov(system, "verbatim"))]
            summary = None
            if watermark:
                if self.freeze_summary and self.frozen and self.frozen[0] == (watermark, self.compacted_at):
                    summary = self.frozen[1]
                else:
                    summary = self.summarize(store, groups[:watermark], self.compacted_at)
                    self.frozen = ((watermark, self.compacted_at), summary)
                for g in groups[:watermark]:
                    for rec in g:
                        om.append({"record": rec["id"], "action": "summarized",
                                   "reason": "older than the verbatim tail chosen at call %d" % self.compacted_at,
                                   "tokens": approx_tokens(rec["message"]), "carried_by": "summary line [%s]" % rec["id"]})
            head_extra = []
            if self.layout == "task_first" and pinned_msg:
                head_extra.append((pinned_msg, pinned_p))
                if status:
                    head_extra.append((status, {"records": [], "transform": "status (derived)", "source": "runtime",
                                                "revision": None, "run": store.run_id, "scope": store.scope,
                                                "trust": "runtime"}))
            parts += head_extra
            if summary:
                parts.append(summary)
            for g in groups[watermark or 0:]:
                parts += self.render_group(g, compacting, om)
            if self.layout == "task_last" and pinned_msg:
                parts.append((pinned_msg, pinned_p))
            return parts, om, summary

        parts, om, summary = assemble(self.watermark, self.watermark is not None)
        used = tools_tokens + sum(approx_tokens(m) for m, _ in parts)
        fired = False
        if used > self.trigger * self.allowance:
            # Compaction: keep the newest whole groups within the recent budget, digest
            # oversized results, summarize everything older. Recorded as its own decision.
            fired = True
            self.compacted_at = call_no
            kept, cost = 0, 0
            for g in reversed(groups):
                tmp = []
                g_cost = sum(approx_tokens(m) for m, _ in self.render_group(g, True, tmp))
                if cost + g_cost > self.recent_budget and kept > 0:
                    break
                cost += g_cost
                kept += 1
            self.watermark = len(groups) - kept
            parts, om, summary = assemble(self.watermark, True)
            store.compactions.append({"call": call_no, "policy": self.policy["id"], "policy_hash": self.policy["hash"],
                                      "watermark_group": self.watermark,
                                      "first_verbatim_record": groups[self.watermark][0]["id"],
                                      "summarized_records": [r["id"] for g in groups[:self.watermark] for r in g]})
        omissions = om
        if self.db is not None and summary_present(parts):
            for oid in excluded_by_scope(self.db, store.scope):
                omissions.append({"record": oid, "action": "excluded", "reason": "observation from another scope",
                                  "tokens": 0, "carried_by": None})
        msgs = [m for m, _ in parts]
        provs = [p for _, p in parts]
        fixed = approx_tokens(system["message"])
        sections = [{"name": "instructions", "items": 1, "tokens": approx_tokens(system["message"])}]
        if pinned_msg:
            fixed += approx_tokens(pinned_msg)
            sections.append({"name": "pinned task", "items": 1, "tokens": approx_tokens(pinned_msg),
                             "cap": self.pinned_cap})
        if status:
            sections.append({"name": "status line", "items": 1, "tokens": approx_tokens(status)})
            fixed += approx_tokens(status)
        if summary:
            sections.append({"name": "derived summary", "items": 1, "tokens": approx_tokens(summary[0]),
                             "cap": self.summary_cap})
            fixed += approx_tokens(summary[0])
        hist = sum(approx_tokens(m) for m in msgs) - fixed
        sections.append({"name": "history (whole groups)", "items": len(msgs) - len(sections), "tokens": hist})
        return finish(self.policy, call_no, msgs, provs, omissions, sections, tools_tokens,
                      compacted=self.watermark is not None, compaction_fired=fired,
                      trigger_tokens=int(self.trigger * self.allowance), allowance=self.allowance)


# ----------------------------------------------------------------------------------
# Prompt-cache arithmetic
# ----------------------------------------------------------------------------------

CACHE_WRITE = 1.25   # multiplier on base input price for a 5-minute cache write
CACHE_READ = 0.10    # multiplier for a cache read


def prefix_tokens(prev, cur):
    """Tokens of the longest common prefix, compared whole unit by unit (tools, then messages)."""
    n = 0
    for a, b in zip(prev, cur):
        if canonical(a) != canonical(b):
            break
        n += approx_tokens(a)
    return n


def cache_costs(requests):
    """requests: list of (tool_definitions, messages). Assumes each request is written to
    the cache in full and the next request reads its longest common prefix."""
    rows, prev = [], None
    for i, (defs, msgs) in enumerate(requests):
        units = [defs] + list(msgs)
        total = sum(approx_tokens(u) for u in units)
        hit = prefix_tokens(prev, units) if prev else 0
        cost = CACHE_READ * hit + CACHE_WRITE * (total - hit)
        rows.append({"request": i + 1, "input_tokens": total, "cached_prefix": hit,
                     "cost_units": round(cost, 1), "uncached_cost_units": total})
        prev = units
    return rows
