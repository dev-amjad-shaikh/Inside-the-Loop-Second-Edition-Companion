"""Chapter 18 helpers: OTel-shaped tracing for the teaching runtime.

SimClock       simulated time in nanoseconds, so every duration is reproducible.
Tracer         creates spans with W3C-sized ids, parent/child relations, links and
               events, and hands ended spans to a batch processor. Only ended spans
               are exported; a crash drops whatever is still buffered or open.
Span shape     a simplified OTLP/JSON span: traceId, spanId, parentSpanId, name, kind,
               start/end in unix nanoseconds, a flat attribute map, events, links, status.
critical_path  the chain of spans that determines a trace's end-to-end duration.
redact_*       content capture off, structure-preserving redaction, and a naive
               truncation for comparison.
checks         trace rules that find failures without an ERROR status.
samplers       head (trace-id ratio), tail on errors, and tail on recorded outcome.

Attribute names follow the OpenTelemetry GenAI semantic conventions where one
exists (gen_ai.*). Everything the conventions do not cover is named itl.*.
Standard library only.
"""
import hashlib
import json
from datetime import datetime, timezone

EPOCH = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)
EPOCH_NS = int(EPOCH.timestamp()) * 1_000_000_000


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha12(obj):
    text = obj if isinstance(obj, str) else canon(obj)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class SimClock:
    def __init__(self, start_ns=EPOCH_NS):
        self.now = start_ns

    def advance(self, seconds):
        self.now += int(round(seconds * 1_000_000_000))
        return self.now


class Exporter:
    """Where exported spans land. A list stands in for an OTLP endpoint."""

    def __init__(self):
        self.spans = []
        self.batches = 0

    def export(self, batch):
        self.spans.extend(batch)
        self.batches += 1


class BatchProcessor:
    """Buffers ended spans and exports them in batches, like an SDK batch processor:
    when the buffer reaches batch_size, or when schedule_delay seconds have passed
    since the last export. Spans are handed over only when they end."""

    def __init__(self, exporter, batch_size=16, schedule_delay=5.0, clock=None):
        self.exporter, self.batch_size, self.buffer = exporter, batch_size, []
        self.delay_ns = int(schedule_delay * 1e9)
        self.clock = clock
        self.last = clock.now if clock else 0

    def on_end(self, span):
        self.buffer.append(span)
        due = self.clock is not None and self.clock.now - self.last >= self.delay_ns
        if len(self.buffer) >= self.batch_size or due:
            self.flush()

    def flush(self):
        if self.buffer:
            self.exporter.export(self.buffer)
            self.buffer = []
        if self.clock is not None:
            self.last = self.clock.now

    def crash(self):
        lost = len(self.buffer)
        self.buffer = []
        return lost


class Tracer:
    def __init__(self, key, clock, processor, resource=None):
        self.key, self.clock, self.processor = key, clock, processor
        self.resource = resource or {}
        self.n = 0
        self.stack = []
        self.open = {}

    def _id(self, width):
        self.n += 1
        return hashlib.sha256(f"{self.key}:{self.n}".encode()).hexdigest()[:width]

    def current(self):
        return self.stack[-1] if self.stack else None

    def start(self, name, kind="INTERNAL", attributes=None, parent="current", links=None, trace_id=None):
        par = self.current() if parent == "current" else parent
        span = {
            "traceId": par["traceId"] if par else (trace_id or self._id(32)),
            "spanId": self._id(16),
            "parentSpanId": par["spanId"] if par else None,
            "name": name, "kind": kind,
            "startTimeUnixNano": self.clock.now, "endTimeUnixNano": None,
            "attributes": dict(attributes or {}), "events": [], "links": list(links or []),
            "status": {"code": "UNSET"}, "resource": self.resource,
        }
        self.open[span["spanId"]] = span
        self.stack.append(span)
        return span

    def event(self, span, name, attributes=None):
        span["events"].append({"timeUnixNano": self.clock.now, "name": name, "attributes": attributes or {}})

    def end(self, span, status=None, message=None):
        span["endTimeUnixNano"] = self.clock.now
        if status:
            span["status"] = {"code": status, **({"message": message} if message else {})}
        self.open.pop(span["spanId"], None)
        if self.stack and self.stack[-1] is span:
            self.stack.pop()
        elif span in self.stack:
            self.stack.remove(span)
        self.processor.on_end(span)

    def crash(self):
        """The process dies: open spans are never ended, buffered spans never leave."""
        lost_open = sorted(s["name"] for s in self.open.values())
        lost_buffered = [s["name"] for s in self.processor.buffer]
        self.processor.crash()
        self.open, self.stack = {}, []
        return {"open_spans_lost": lost_open, "buffered_spans_lost": lost_buffered}


def traceparent(span):
    return f"00-{span['traceId']}-{span['spanId']}-01"


def seconds(span):
    return (span["endTimeUnixNano"] - span["startTimeUnixNano"]) / 1e9


# ---------------------------------------------------------------------------
# Reading a trace
# ---------------------------------------------------------------------------
def by_trace(spans):
    out = {}
    for s in spans:
        out.setdefault(s["traceId"], []).append(s)
    return out


def children_of(spans):
    kids = {}
    for s in spans:
        kids.setdefault(s["parentSpanId"], []).append(s)
    for v in kids.values():
        v.sort(key=lambda s: (s["startTimeUnixNano"], s["endTimeUnixNano"]))
    return kids


def root_of(spans):
    ids = {s["spanId"] for s in spans}
    roots = [s for s in spans if s["parentSpanId"] is None or s["parentSpanId"] not in ids]
    return sorted(roots, key=lambda s: s["startTimeUnixNano"])[0]


def tree_lines(spans, attrs=(), t0=None):
    kids = children_of(spans)
    root = root_of(spans)
    t0 = t0 if t0 is not None else root["startTimeUnixNano"]
    lines = []

    def walk(s, depth):
        a = " ".join(f"{k}={s['attributes'][k]}" for k in attrs if k in s["attributes"])
        st = s["status"]["code"]
        lines.append(f"{(s['startTimeUnixNano'] - t0) / 1e9:7.2f}s {seconds(s):7.2f}s  "
                     f"{'  ' * depth}{s['name']}" + (f"  [{st}]" if st != "UNSET" else "") + (f"  {a}" if a else ""))
        for k in kids.get(s["spanId"], []):
            walk(k, depth + 1)
    walk(root, 0)
    return lines


def critical_path(spans):
    """Segments that determine the root's end-to-end duration.

    Walk back from the root's end: the child that ends last is on the path; before it,
    the latest-ending child that finished before it started; and so on. Time on the
    root not covered by any chosen child is the parent's own time (self time).
    Recurse into each chosen child."""
    kids = children_of(spans)
    segments = []

    def walk(s):
        chosen = []
        cursor = s["endTimeUnixNano"]
        cands = sorted(kids.get(s["spanId"], []), key=lambda c: c["endTimeUnixNano"], reverse=True)
        for c in cands:
            if c["endTimeUnixNano"] <= cursor:
                if not chosen or c["endTimeUnixNano"] <= chosen[-1]["startTimeUnixNano"]:
                    chosen.append(c)
                    cursor = c["startTimeUnixNano"]
        chosen.reverse()
        t = s["startTimeUnixNano"]
        for c in chosen:
            if c["startTimeUnixNano"] > t:
                segments.append({"name": f"{s['name']} (self)", "seconds": (c["startTimeUnixNano"] - t) / 1e9,
                                 "span": s["spanId"]})
            walk(c)
            t = c["endTimeUnixNano"]
        if not chosen:
            segments.append({"name": s["name"], "seconds": seconds(s), "span": s["spanId"]})
        elif s["endTimeUnixNano"] > t:
            segments.append({"name": f"{s['name']} (self)", "seconds": (s["endTimeUnixNano"] - t) / 1e9,
                             "span": s["spanId"]})
    walk(root_of(spans))
    return segments


# ---------------------------------------------------------------------------
# Content capture and redaction
# ---------------------------------------------------------------------------
def redact_value(v):
    """Replace content with a marker that keeps its type, size and digest."""
    if isinstance(v, str):
        return {"redacted": "text", "length": len(v), "sha256": sha12(v)}
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return {"redacted": type(v).__name__, "sha256": sha12(v)}


def redact_args(args):
    return {k: redact_value(v) for k, v in sorted(args.items())} if isinstance(args, dict) else redact_value(args)


def redact_message(m):
    """Structure-preserving: roles, tool-call ids, tool names and argument keys stay;
    every text, argument value and tool result becomes a length and digest."""
    out = {"role": m.get("role")}
    if "tool_call_id" in m:
        out["tool_call_id"] = m["tool_call_id"]
    if "content" in m:
        c = m["content"]
        if isinstance(c, dict):
            out["content"] = {"status": c.get("status"), "fields": sorted(c), "body": redact_value(canon(c))}
        else:
            out["content"] = redact_value(c)
    if m.get("tool_calls"):
        out["tool_calls"] = [{"id": c["id"], "name": c["name"], "arguments": redact_args(c["arguments"])}
                             for c in m["tool_calls"]]
    return out


def naive_truncate(text, limit):
    return text if len(text) <= limit else text[:limit]


def input_messages_attr(messages, mode, limit=None):
    """The value a GenAI span would carry in gen_ai.input.messages under each mode."""
    if mode == "off":
        return None
    if mode == "structure":
        return canon([redact_message(m) for m in messages])
    text = canon(messages)
    return naive_truncate(text, limit) if limit else text


# ---------------------------------------------------------------------------
# Trace rules: failures that carry no ERROR status
# ---------------------------------------------------------------------------
def rule_stale_evidence(spans):
    """The latest passing test run must cover the versions the run finished with.
    Uses only attributes the executor wrote on execute_tool spans."""
    tools = sorted([s for s in spans if s["attributes"].get("gen_ai.operation.name") == "execute_tool"],
                   key=lambda s: s["endTimeUnixNano"])
    head, last_pass = {}, None
    for s in tools:
        a = s["attributes"]
        for k, v in a.items():
            if k.startswith("itl.artifact.version_after."):
                head[k[len("itl.artifact.version_after."):]] = (v, s["attributes"].get("gen_ai.tool.call.id"))
        if a.get("gen_ai.tool.name") == "run_tests" and a.get("itl.observation.status") == "passed":
            last_pass = s
    if last_pass is None:
        return {"finding": "no passing test run"}
    covers = {k[len("itl.evidence.covers."):]: v for k, v in last_pass["attributes"].items()
              if k.startswith("itl.evidence.covers.")}
    stale = {p: {"covered": covers[p], "current": head[p][0], "changed_by": head[p][1]}
             for p in covers if p in head and head[p][0] != covers[p]}
    return {"finding": "stale evidence" if stale else "evidence current",
            "evidence_call": last_pass["attributes"].get("gen_ai.tool.call.id"), "stale": stale}


def rule_orphans(spans):
    """Every completed tool result must appear in a later model input."""
    tools = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "execute_tool"]
    chats = [s for s in spans if s["attributes"].get("gen_ai.operation.name") == "chat"]
    orphans = []
    for t in tools:
        cid = t["attributes"].get("gen_ai.tool.call.id")
        later = [c for c in chats if c["startTimeUnixNano"] >= t["endTimeUnixNano"]]
        if later and not any(cid in c["attributes"].get("itl.input.tool_call_ids", []) for c in later):
            orphans.append(cid)
    return orphans


def has_error(spans):
    return any(s["status"]["code"] == "ERROR" for s in spans)


# ---------------------------------------------------------------------------
# Samplers. Each returns True to keep the whole trace.
# ---------------------------------------------------------------------------
def head_keep(trace_id, ratio):
    """Trace-id ratio: decided at the root from the id alone, before anything happens."""
    return int(trace_id[-14:], 16) < ratio * (1 << 56)


def tail_error_keep(spans, ratio):
    root = root_of(spans)
    return has_error(spans) or head_keep(root["traceId"], ratio)


def tail_outcome_keep(spans, ratio):
    """Decided after the run's outcome is recorded on the root span."""
    root = root_of(spans)
    outcome = root["attributes"].get("itl.run.outcome")
    if outcome not in ("verified", "waiting_for_person"):
        return True
    if rule_stale_evidence(spans)["finding"] == "stale evidence":
        return True
    return head_keep(root["traceId"], ratio)
