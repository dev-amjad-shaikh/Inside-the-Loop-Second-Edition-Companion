"""Chapter 5: a provider-neutral model adapter over recorded SSE streams.

Layers, from the wire inward:

  read_sse()            SSE framing (WHATWG rules: an event ends at a blank
                        line; an unfinished event at end of input is discarded)
  BlockDecoder,
  ItemDecoder           provider format -> neutral events
  Assembler             neutral events -> one explicit Outcome
  FlawedAdapter         the design the chapter argues against
  translate_request()   runtime request -> provider request, with a record of
                        every field that could not be carried
  build_followup()      outcome -> the provider-format history for the next call
  check_followup()      a stand-in check of the documented rule that reasoning
                        blocks travel back unchanged (written from the docs; it
                        is not a provider)

Standard library only.
"""
import copy
import hashlib
import json

ADAPTER_VERSION = "ch05-adapter/1.0.0"
SUPPORTED_API = {"block": ["2026-06-01"], "item": ["responses-2026-05"]}

# Stop-reason vocabularies this adapter version understands. Anything else is
# reported as unsupported rather than guessed.
BLOCK_STOP = {"end_turn": "end", "stop_sequence": "end", "tool_use": "tool_use",
              "max_tokens": "length", "refusal": "refusal"}
ITEM_INCOMPLETE = {"max_output_tokens": "length", "content_filter": "refusal"}

# Events that carry no content and may be skipped (recorded, not interpreted).
BLOCK_IGNORABLE = {"ping"}
ITEM_IGNORABLE = {"response.in_progress", "response.content_part.added", "response.content_part.done",
                  "response.output_text.done", "response.function_call_arguments.done"}

TOOL_EFFECT = {"read_file": "read", "run_tests": "compute", "replace_text": "write"}
REQUIRED = {"read_file": ["path"], "replace_text": ["path", "old", "new"], "run_tests": ["tests"]}


def sha(text):
    if isinstance(text, (dict, list)):
        text = json.dumps(text, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------- framing

class SSEEvent:
    def __init__(self, name, data, t_ms):
        self.name, self.data, self.t = name, data, t_ms


def read_sse(raw):
    """Yield SSEEvent objects. Returns (events, discarded_text).

    Comment lines ': t=<ms>' set the authored arrival time of what follows.
    """
    events, name, data_lines, t_ms = [], None, [], 0
    lines = raw.split("\n")
    complete_upto = raw.rfind("\n\n")  # everything after the last blank line is unfinished
    consumed = 0
    for line in lines:
        consumed += len(line) + 1
        if line.startswith(":"):
            body = line[1:].strip()
            if body.startswith("t="):
                t_ms = int(body[2:])
            continue
        if line == "":
            if data_lines:
                events.append(SSEEvent(name or "message", "\n".join(data_lines), t_ms))
            name, data_lines = None, []
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            name = value
        elif field == "data":
            data_lines.append(value)
    discarded = raw[complete_upto + 2:] if complete_upto >= 0 else raw
    discarded = "\n".join(l for l in discarded.split("\n") if l and not l.startswith(":"))
    return events, discarded


# -------------------------------------------------------------------- decoders
# A neutral event is a dict with "kind", "t" and kind-specific fields.

class BlockDecoder:
    """Block format (modeled on Anthropic Messages streaming)."""

    fmt = "block"

    def decode(self, ev):
        t = ev.t
        try:
            d = json.loads(ev.data)
        except json.JSONDecodeError as exc:
            return [{"kind": "malformed_event", "t": t, "event": ev.name, "error": str(exc)}]
        n = ev.name
        if n == "message_start":
            m = d["message"]
            u = m.get("usage", {})
            return [{"kind": "response_start", "t": t, "id": m.get("id"), "model": m.get("model"),
                     "usage": {"input_tokens": u.get("input_tokens"),
                               "cache_read_tokens": u.get("cache_read_input_tokens")}}]
        if n == "content_block_start":
            b, i = d["content_block"], d["index"]
            typ = b.get("type")
            if typ == "text":
                return [{"kind": "block_start", "t": t, "index": i, "type": "text", "raw": b}]
            if typ == "thinking":
                return [{"kind": "block_start", "t": t, "index": i, "type": "reasoning", "raw": b}]
            if typ == "tool_use":
                return [{"kind": "block_start", "t": t, "index": i, "type": "tool_call",
                         "id": b.get("id"), "name": b.get("name"), "raw": b}]
            return [{"kind": "block_start", "t": t, "index": i, "type": "unknown", "raw_type": typ, "raw": b}]
        if n == "content_block_delta":
            i, delta = d["index"], d["delta"]
            dt = delta.get("type")
            if dt == "text_delta":
                return [{"kind": "text_delta", "t": t, "index": i, "text": delta["text"]}]
            if dt == "input_json_delta":
                return [{"kind": "args_delta", "t": t, "index": i, "fragment": delta["partial_json"]}]
            if dt == "thinking_delta":
                return [{"kind": "reasoning_delta", "t": t, "index": i, "text": delta["thinking"]}]
            if dt == "signature_delta":
                return [{"kind": "signature", "t": t, "index": i, "value": delta["signature"]}]
            return [{"kind": "unknown_delta", "t": t, "index": i, "raw_type": dt}]
        if n == "content_block_stop":
            return [{"kind": "block_stop", "t": t, "index": d["index"]}]
        if n == "message_delta":
            delta = d.get("delta", {})
            out = [{"kind": "stop", "t": t, "raw_reason": delta.get("stop_reason"),
                    "detail": delta.get("stop_details"),
                    "usage": {"output_tokens": d.get("usage", {}).get("output_tokens")}}]
            known = {"stop_reason", "stop_sequence", "stop_details"}
            for k in sorted(set(delta) - known):
                out.append({"kind": "unmapped_field", "t": t, "path": f"message_delta.delta.{k}"})
            return out
        if n == "message_stop":
            return [{"kind": "terminal", "t": t}]
        if n == "error":
            err = d.get("error", {})
            return [{"kind": "provider_error", "t": t, "type": err.get("type"), "message": err.get("message")}]
        if n in BLOCK_IGNORABLE:
            return [{"kind": "ignored", "t": t, "event": n}]
        return [{"kind": "unknown_event", "t": t, "event": n}]


class ItemDecoder:
    """Item format (modeled on OpenAI Responses streaming)."""

    fmt = "item"
    KNOWN_RESPONSE_FIELDS = {"id", "object", "model", "status", "output", "usage", "incomplete_details", "error"}

    def decode(self, ev):
        t = ev.t
        try:
            d = json.loads(ev.data)
        except json.JSONDecodeError as exc:
            return [{"kind": "malformed_event", "t": t, "event": ev.name, "error": str(exc)}]
        n = d.get("type", ev.name)
        if n == "response.created":
            r = d["response"]
            return [{"kind": "response_start", "t": t, "id": r.get("id"), "model": r.get("model"), "usage": {}}]
        if n == "response.output_item.added":
            item, i = d["item"], d["output_index"]
            typ = item.get("type")
            if typ == "message":
                return [{"kind": "block_start", "t": t, "index": i, "type": "text", "raw": item}]
            if typ == "reasoning":
                return [{"kind": "block_start", "t": t, "index": i, "type": "reasoning", "raw": item}]
            if typ == "function_call":
                return [{"kind": "block_start", "t": t, "index": i, "type": "tool_call",
                         "id": item.get("call_id"), "name": item.get("name"), "raw": item}]
            return [{"kind": "block_start", "t": t, "index": i, "type": "unknown", "raw_type": typ, "raw": item}]
        if n == "response.output_text.delta":
            return [{"kind": "text_delta", "t": t, "index": d["output_index"], "text": d["delta"]}]
        if n == "response.function_call_arguments.delta":
            return [{"kind": "args_delta", "t": t, "index": d["output_index"], "fragment": d["delta"]}]
        if n == "response.output_item.done":
            item, i = d["item"], d["output_index"]
            out = []
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                out.append({"kind": "signature", "t": t, "index": i, "value": item["encrypted_content"]})
                summary = " ".join(s.get("text", "") for s in item.get("summary", []))
                if summary:
                    out.append({"kind": "reasoning_delta", "t": t, "index": i, "text": summary})
            out.append({"kind": "block_stop", "t": t, "index": i, "final": item})
            return out
        if n in ("response.completed", "response.incomplete", "response.failed"):
            r = d["response"]
            u = r.get("usage") or {}
            status = r.get("status")
            if status == "completed":
                raw_reason = "completed"
            elif status == "incomplete":
                raw_reason = "incomplete:" + str((r.get("incomplete_details") or {}).get("reason"))
            else:
                raw_reason = str(status)
            out = [{"kind": "stop", "t": t, "raw_reason": raw_reason, "detail": r.get("incomplete_details"),
                    "model": r.get("model"),
                    "usage": {"input_tokens": u.get("input_tokens"),
                              "cache_read_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens"),
                              "output_tokens": u.get("output_tokens"),
                              "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens")},
                    "final_output": r.get("output", [])}]
            for k in sorted(set(r) - self.KNOWN_RESPONSE_FIELDS):
                out.append({"kind": "unmapped_field", "t": t, "path": f"response.{k}", "value": r[k]})
            out.append({"kind": "terminal", "t": t})
            return out
        if n == "error":
            return [{"kind": "provider_error", "t": t, "type": d.get("code"), "message": d.get("message")}]
        if n in ITEM_IGNORABLE:
            return [{"kind": "ignored", "t": t, "event": n}]
        return [{"kind": "unknown_event", "t": t, "event": n}]


DECODERS = {"block": BlockDecoder, "item": ItemDecoder}


# ---------------------------------------------------------- partial JSON tools

def strict_object(text):
    try:
        v = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def lenient_parse(text):
    """Close whatever is open. This is what a UI uses to show partial values.

    It is included because the flawed adapter misuses it for dispatch.
    """
    v = strict_object(text)
    if v is not None:
        return v
    stack, in_str, esc, last_comma = [], False, False, None
    for pos, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == "," and len(stack) == 1:
            last_comma = pos
    closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
    body = text[:-1] if esc else text
    candidates = [body + ('"' if in_str else '') + closers]
    if last_comma is not None:
        candidates.append(text[:last_comma] + "}")
    for c in candidates:
        v = strict_object(c)
        if v is not None:
            return v
    return None


# ------------------------------------------------------------------ assembler

class Assembler:
    """Folds neutral events into one explicit outcome.

    barrier="response": tool calls become executable only when the response
    ends with a tool-use stop and every tool block is complete.
    barrier="block": additionally, a completed tool block whose tool is
    declared read-only may be released early, after it has been recorded.
    """

    def __init__(self, fmt, fixture, meta, barrier="response", on_release=None, journal=None):
        self.fmt, self.fixture, self.meta = fmt, fixture, meta
        self.barrier, self.on_release = barrier, on_release
        self.journal = journal if journal is not None else []
        self.blocks, self.order = {}, []
        self.response_id = self.model = None
        self.usage = {}
        self.stop = None
        self.terminal_t = None
        self.error = None
        self.unknown_events, self.ignored, self.unmapped, self.malformed = [], [], [], []
        self.timeline = []
        self.released = []
        self.first = {}

    def _block(self, i):
        return self.blocks[i]

    def feed(self, e):
        k, t = e["kind"], e["t"]
        self.timeline.append({"t": t, "kind": k, **{x: e[x] for x in ("index", "event", "raw_reason") if x in e}})
        if k == "response_start":
            self.response_id, self.model = e["id"], e["model"]
            self.usage.update({x: y for x, y in e["usage"].items() if y is not None})
        elif k == "block_start":
            i = e["index"]
            self.blocks[i] = {"index": i, "type": e["type"], "raw_type": e.get("raw_type"), "id": e.get("id"),
                              "name": e.get("name"), "text": "", "args": "", "signature": None,
                              "complete": False, "raw_start": e["raw"], "final": None,
                              "t_start": t, "t_stop": None}
            self.order.append(i)
        elif k == "text_delta":
            self.first.setdefault("text", t)
            self._block(e["index"])["text"] += e["text"]
        elif k == "reasoning_delta":
            self._block(e["index"])["text"] += e["text"]
        elif k == "args_delta":
            self.first.setdefault("args", t)
            self._block(e["index"])["args"] += e["fragment"]
        elif k == "signature":
            self._block(e["index"])["signature"] = e["value"]
        elif k == "block_stop":
            b = self._block(e["index"])
            b["complete"], b["t_stop"] = True, t
            if e.get("final") is not None:
                b["final"] = e["final"]
                if b["type"] == "tool_call" and e["final"].get("arguments"):
                    b["args"] = e["final"]["arguments"]
            if self.barrier == "block":
                self._maybe_release(b, t)
        elif k == "stop":
            self.stop = e
            if e.get("model"):
                self.model = self.model or e["model"]
            self.usage.update({x: y for x, y in e["usage"].items() if y is not None})
        elif k == "terminal":
            self.terminal_t = t
        elif k == "provider_error":
            self.error = e
        elif k == "unknown_event":
            self.unknown_events.append(e["event"])
        elif k == "ignored":
            self.ignored.append(e["event"])
        elif k == "unmapped_field":
            self.unmapped.append(e["path"])
        elif k == "malformed_event":
            self.malformed.append(e)
        elif k == "unknown_delta":
            self._block(e["index"])["type"] = "unknown"
            self._block(e["index"])["raw_type"] = "delta:" + str(e["raw_type"])

    def _maybe_release(self, b, t):
        if b["type"] != "tool_call" or TOOL_EFFECT.get(b["name"]) != "read":
            return
        args = strict_object(b["args"] or "{}")
        if args is None:
            return
        call = {"id": b["id"], "name": b["name"], "arguments": args}
        self.journal.append({"t": t, "step": "recorded", "call_id": b["id"], "response_id": self.response_id,
                             "note": "completed read-only block recorded before release"})
        self.released.append(b["id"])
        if self.on_release:
            self.on_release(call, t)

    def normalized_stop(self):
        if self.stop is None:
            return None
        raw = self.stop["raw_reason"]
        if self.fmt == "block":
            return BLOCK_STOP.get(raw)
        if raw == "completed":
            has_calls = any(self.blocks[i]["type"] == "tool_call" for i in self.order)
            return "tool_use" if has_calls else "end"
        if raw.startswith("incomplete:"):
            return ITEM_INCOMPLETE.get(raw.split(":", 1)[1])
        return None

    def finish(self, transport_end, discarded):
        fx = self.meta["fixtures"][self.fixture]
        ad = self.meta["adapters"][self.fmt]
        norm = self.normalized_stop()
        tool_blocks = [self.blocks[i] for i in self.order if self.blocks[i]["type"] == "tool_call"]
        unknown_blocks = [self.blocks[i] for i in self.order if self.blocks[i]["type"] == "unknown"]
        calls, partial, problems = [], [], []
        for b in tool_blocks:
            args = strict_object(b["args"] or "{}") if b["complete"] else None
            if b["complete"] and args is not None:
                calls.append({"id": b["id"], "name": b["name"], "arguments": args})
            else:
                partial.append({"id": b["id"], "name": b["name"], "complete_block": b["complete"],
                                "args_received": b["args"], "args_bytes": len(b["args"].encode("utf-8")),
                                "strict_parse": strict_object(b["args"]) is not None,
                                "lenient_keys": sorted((lenient_parse(b["args"]) or {}).keys())})
        cause = None
        if self.error:
            kind, cause = "interrupted", "provider_error:" + str(self.error["type"])
        elif self.terminal_t is None:
            kind = "interrupted"
            cause = "connection_reset" if transport_end == "reset" else "closed_without_terminal_event"
        elif unknown_blocks:
            kind, cause = "unsupported", "unknown block type: " + ", ".join(str(b["raw_type"]) for b in unknown_blocks)
        elif self.malformed:
            kind, cause = "unsupported", "malformed event payload"
        elif norm is None:
            kind, cause = "unsupported", f"unknown stop reason {self.stop['raw_reason'] if self.stop else None!r}"
        elif norm == "length":
            kind, cause = "truncated_max_tokens", self.stop["raw_reason"]
        elif norm == "refusal":
            kind, cause = "refused", self.stop["raw_reason"]
        elif norm == "tool_use":
            if partial or not calls:
                kind, cause = "unsupported", "tool-use stop with incomplete or unparseable tool blocks"
                problems = partial
            else:
                kind = "completed_tool_calls"
        else:  # end
            if tool_blocks:
                kind, cause = "unsupported", "tool blocks present without a tool-use stop"
            else:
                kind = "completed_text"
        executable = calls if kind == "completed_tool_calls" else []
        if kind != "completed_tool_calls":
            # complete blocks from a response that did not finish are not executable either
            partial = partial + [dict(c, complete_block=True, args_received=json.dumps(c["arguments"]),
                                      note="complete block in a response that did not complete")
                                 for c in calls]
        text = "".join(self.blocks[i]["text"] for i in self.order if self.blocks[i]["type"] == "text")
        reasoning = [{"index": b["index"], "chars": len(b["text"]),
                      "signature_sha": sha(b["signature"]) if b["signature"] else None,
                      "complete": b["complete"]}
                     for b in (self.blocks[i] for i in self.order) if b["type"] == "reasoning"]
        usage = dict(self.usage)
        usage["complete"] = self.terminal_t is not None and "output_tokens" in usage
        return {
            "fixture": self.fixture, "format": self.fmt, "modeled_on": ad["modeled_on"],
            "adapter_version": ADAPTER_VERSION, "api_version": ad["api_version"],
            "api_version_supported": ad["api_version"] in SUPPORTED_API[self.fmt],
            "request_id": fx["request_id"], "requested_model": ad["requested_model"],
            "reported_model": self.model, "response_id": self.response_id,
            "kind": kind, "cause": cause,
            "raw_stop_reason": self.stop["raw_reason"] if self.stop else None,
            "stop_detail": self.stop.get("detail") if self.stop else None,
            "normalized_stop": norm,
            "text": text, "executable_calls": executable, "not_executable": partial,
            "reasoning": reasoning,
            "unknown_blocks": [{"index": b["index"], "type": b["raw_type"], "raw": b["raw_start"]}
                               for b in unknown_blocks],
            "unknown_events": self.unknown_events, "ignored_events": self.ignored,
            "unmapped_fields": self.unmapped,
            "usage": usage,
            "timing_ms": {"first_text": self.first.get("text"), "first_args": self.first.get("args"),
                          "terminal": self.terminal_t,
                          "executable_at": self.terminal_t if executable else None,
                          "last_event": self.timeline[-1]["t"] if self.timeline else None},
            "released_early": self.released,
            "discarded_bytes": len(discarded.encode("utf-8")), "discarded_tail": discarded[-60:],
            "transport_end": transport_end,
        }


def assemble(raw, fixture, meta, barrier="response", on_release=None, journal=None, trace=None):
    fx = meta["fixtures"][fixture]
    fmt = fx["format"]
    events, discarded = read_sse(raw)
    dec = DECODERS[fmt]()
    asm = Assembler(fmt, fixture, meta, barrier=barrier, on_release=on_release, journal=journal)
    for ev in events:
        for ne in dec.decode(ev):
            asm.feed(ne)
            if trace is not None:
                trace.append(ne)
    out = asm.finish(fx["transport_end"], discarded)
    out["fixture_sha"] = sha(raw)
    out["sse_events"] = len(events)
    return out, asm


# ------------------------------------------------------------ flawed adapter

class FlawedAdapter:
    """Dispatches on the first parseable tool-call JSON, drops block types it
    does not know, and maps every ending to "done"."""

    def __init__(self, dispatch):
        self.dispatch = dispatch

    def run(self, raw, fmt):
        events, _ = read_sse(raw)
        dec = DECODERS[fmt]()
        blocks, dispatched, text, log = {}, [], "", []
        for ev in events:
            for e in dec.decode(ev):
                k = e["kind"]
                if k == "block_start" and e["type"] in ("text", "tool_call"):
                    blocks[e["index"]] = {"type": e["type"], "id": e.get("id"), "name": e.get("name"), "args": ""}
                elif k == "text_delta" and e["index"] in blocks:
                    text += e["text"]
                elif k == "args_delta" and e["index"] in blocks:
                    b = blocks[e["index"]]
                    b["args"] += e["fragment"]
                    if b["id"] in dispatched:
                        continue
                    parsed = lenient_parse(b["args"])
                    if parsed is not None and all(r in parsed for r in REQUIRED.get(b["name"], [])):
                        dispatched.append(b["id"])
                        call = {"id": b["id"], "name": b["name"], "arguments": parsed}
                        log.append({"t": e["t"], "dispatched": call,
                                    "args_bytes_seen": len(b["args"].encode("utf-8"))})
                        self.dispatch(call)
                # everything else (reasoning, unknown blocks, stop reasons, terminal events) is ignored
        return {"status": "done", "text": text, "dispatched": log,
                "kept_block_types": sorted({b["type"] for b in blocks.values()})}


# --------------------------------------------------- request translation

def translate_request(request, model, meta):
    """Runtime request -> provider request plus an explicit account of fields
    that were carried, renamed, dropped (tuning) or rejected (semantic)."""
    cap = meta["capabilities"][model]
    fmt = cap["format"]
    settings = request["settings"]
    carried, dropped, rejected = {}, [], []
    tuning = {"temperature", "top_p", "seed"}
    for key, value in settings.items():
        if key in tuning:
            if cap["reasoning"] and key in cap["unsupported_with_reasoning"]:
                dropped.append({"field": key, "value": value, "reason": "unsupported with reasoning enabled"})
            else:
                carried[key] = value
        elif key == "max_output_tokens":
            carried["max_tokens" if fmt == "block" else "max_output_tokens"] = value
        elif key == "parallel_tool_calls":
            target = cap["parallel_tool_calls"]
            if target is None:
                rejected.append({"field": key, "value": value,
                                 "reason": "semantic setting has no equivalent for this model"})
            elif target == "tool_choice.disable_parallel_tool_use":
                carried["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": not value}
            else:
                carried[target] = value
        else:
            rejected.append({"field": key, "value": value, "reason": "unknown setting"})
    tools = request["tools"]
    if fmt == "block":
        system = [m["content"] for m in request["messages"] if m["role"] == "system"]
        body = {"model": model, "system": "\n".join(system),
                "messages": [m for m in request["messages"] if m["role"] != "system"],
                "tools": [{"name": d["name"], "description": d["description"], "input_schema": d["parameters"]}
                          for d in tools], "stream": True}
    else:
        body = {"model": model, "input": request["messages"],
                "tools": [{"type": "function", "name": d["name"], "description": d["description"],
                           "parameters": d["parameters"], "strict": False} for d in tools], "stream": True}
    body.update(carried)
    schemas_in = [d["parameters"] for d in tools]
    schemas_out = [x.get("input_schema", x.get("parameters")) for x in body["tools"]]
    return {"model": model, "format": fmt, "sent": not rejected,
            "body_keys": sorted(body), "carried_settings": carried,
            "dropped": dropped, "rejected": rejected,
            "tool_order_in": [d["name"] for d in tools], "tool_order_out": [x["name"] for x in body["tools"]],
            "schemas_sha_in": sha(schemas_in), "schemas_sha_out": sha(schemas_out)}


# ------------------------------------------------------------- follow-up

def build_followup(outcome, asm, results, faithful=True):
    """The assistant turn and tool results, in provider format, for the next call."""
    fmt = outcome["format"]
    blocks = [asm.blocks[i] for i in asm.order]
    if not faithful:
        blocks = [b for b in blocks if b["type"] in ("text", "tool_call")]
    if fmt == "block":
        content = []
        for b in blocks:
            if b["type"] == "reasoning":
                content.append({"type": "thinking", "thinking": b["text"], "signature": b["signature"]})
            elif b["type"] == "text":
                content.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_call":
                content.append({"type": "tool_use", "id": b["id"], "name": b["name"],
                                "input": json.loads(b["args"])})
            else:
                content.append(copy.deepcopy(b["raw_start"]))
        return [{"role": "assistant", "content": content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid,
                                              "content": json.dumps(r)} for cid, r in results]}]
    items = []
    for b in blocks:
        if b["type"] == "reasoning":
            items.append(copy.deepcopy(b["final"]))
        elif b["type"] == "tool_call":
            items.append({"type": "function_call", "call_id": b["id"], "name": b["name"], "arguments": b["args"]})
        elif b["type"] == "text":
            items.append({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": b["text"]}]})
    items += [{"type": "function_call_output", "call_id": cid, "output": json.dumps(r)} for cid, r in results]
    return items


def check_followup(outcome, asm, followup):
    """Stand-in for the documented rule: reasoning blocks from the response
    must travel back complete, unmodified and in order. Written from the
    provider documentation; it is not the provider's validator."""
    fmt = outcome["format"]
    originals = [asm.blocks[i] for i in asm.order if asm.blocks[i]["type"] == "reasoning"]
    if fmt == "block":
        sent = [c for c in followup[0]["content"] if c["type"] in ("thinking", "redacted_thinking")]
        sent_sig = [sha(c.get("signature") or "") for c in sent]
    else:
        sent = [i for i in followup if i.get("type") == "reasoning"]
        sent_sig = [sha(i.get("encrypted_content") or "") for i in sent]
    want_sig = [sha(b["signature"] or "") for b in originals]
    ok = sent_sig == want_sig
    return {"reasoning_blocks_in_response": len(originals), "reasoning_blocks_sent_back": len(sent),
            "signatures_match": ok,
            "verdict": "accepted" if ok else "would be rejected: reasoning blocks missing or modified"}
