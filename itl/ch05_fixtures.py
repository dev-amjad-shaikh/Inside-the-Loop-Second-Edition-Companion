"""Authoring script for the Chapter 5 stream fixtures.

The fixtures under fixtures/ch05_streams/ are hand-authored server-sent-event
streams. Their event names and payload shapes are modeled on two documented
provider formats:

  block format  modeled on Anthropic Messages streaming (message_start,
                content_block_start/delta/stop, message_delta, message_stop)
  item format   modeled on OpenAI Responses streaming (response.created,
                response.output_item.added/done, function_call_arguments.delta,
                response.completed / response.incomplete)

They are NOT byte-exact recordings of any provider. Identifiers, model names,
signatures, encrypted content and token counts are invented.

Convention: an SSE comment line ": t=<ms>" gives the arrival time, in
milliseconds after the request was sent, of the events that follow it. The
times are authored to make a readable timeline; they are not measurements.

Run this module directly to rewrite the fixture files:
    python3 -m itl.ch05_fixtures
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "fixtures", "ch05_streams")

ANSWER = ("The filter used the UTC calendar date. I changed it to convert each timestamp to the "
          "configured report timezone before comparing dates. The new near-midnight test failed "
          "before the fix; both tests now pass. I changed report.py and test_report.py. "
          "Nothing was deployed.")

OLD_LINE = "    return created_at.date() == requested_day"
NEW_LINES = ("    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
             "    return local_time.date() == requested_day")

SIGNATURE = ("EqQBCkYIBRgCKkB0c2lnLWZpeHR1cmUtb25seS1ub3QtYS1yZWFsLXNpZ25hdHVyZQ"
             "SDGR0ZXN0LXNpZ25hdHVyZS1ieXRlcy0wMDEyMzQ1Njc4OWFiY2RlZg==")
ENCRYPTED = ("gAAAAABo-ZmlrdHVyZS1lbmNyeXB0ZWQtcmVhc29uaW5nLWl0ZW0tMDEtbm90LXJlYWwt"
             "Y29udGVudC1vbmx5LXNoYXBlLTAxMjM0NTY3ODlhYmNkZWY=")


def ev(name, data):
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def t(ms):
    return f": t={ms}\n"


class B:
    """Item-format builder with sequence numbers."""

    def __init__(self):
        self.seq = 0
        self.parts = []

    def at(self, ms):
        self.parts.append(t(ms))

    def e(self, name, **data):
        body = {"type": name, "sequence_number": self.seq}
        body.update(data)
        self.seq += 1
        self.parts.append(ev(name, body))

    def raw(self, s):
        self.parts.append(s)

    def text(self):
        return "".join(self.parts)


def fix_args():
    return json.dumps({"path": "report.py", "old": OLD_LINE, "new": NEW_LINES}, separators=(", ", ": "))


def fix_arg_deltas():
    full = fix_args()
    cuts = [full.index('"old"'), full.index('"new"'),
            full.index("ZoneInfo(timezone))"), full.index("time.date()")]
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(full[prev:c])
        prev = c
    pieces.append(full[prev:])
    assert "".join(pieces) == full
    return pieces


def item_prelude(b, resp_id, model):
    b.at(180)
    b.e("response.created", response={"id": resp_id, "object": "response", "model": model,
                                      "status": "in_progress", "output": []})
    b.e("response.in_progress", response={"id": resp_id, "status": "in_progress"})


def b_fix(cut):
    b = B()
    rid = "resp_b7f2c1" if not cut else "resp_b7f2a0"
    item_prelude(b, rid, "coder-pro-2026-07-30")
    b.at(410)
    b.e("response.output_item.added", output_index=0,
        item={"type": "reasoning", "id": "rs_01", "summary": []})
    b.at(1320)
    reasoning = {"type": "reasoning", "id": "rs_01",
                 "summary": [{"type": "summary_text",
                              "text": "Convert the timestamp to the report timezone before taking the date."}],
                 "encrypted_content": ENCRYPTED if not cut else ENCRYPTED.replace("LTAx", "LTAw")}
    b.e("response.output_item.done", output_index=0, item=reasoning)
    b.at(1340)
    fc = {"type": "function_call", "id": "fc_01", "call_id": "call_fix_1", "name": "replace_text",
          "arguments": "", "status": "in_progress"}
    b.e("response.output_item.added", output_index=1, item=fc)
    pieces = fix_arg_deltas()
    times = [1390, 1450, 1520, 1580, 1640]
    for i, (ms, piece) in enumerate(zip(times, pieces)):
        if cut and i == 4:
            b.at(ms)
            partial = ev("response.function_call_arguments.delta",
                         {"type": "response.function_call_arguments.delta", "sequence_number": b.seq,
                          "item_id": "fc_01", "output_index": 1, "delta": piece})
            b.raw(partial[:partial.index("time.da") + len("time.da")])  # connection lost mid-line
            return b.text()
        b.at(ms)
        b.e("response.function_call_arguments.delta", item_id="fc_01", output_index=1, delta=piece)
    full = fix_args()
    b.at(1650)
    b.e("response.function_call_arguments.done", item_id="fc_01", output_index=1, arguments=full)
    done = dict(fc, arguments=full, status="completed")
    b.e("response.output_item.done", output_index=1, item=done)
    b.at(1702)
    b.e("response.completed", response={
        "id": rid, "object": "response", "model": "coder-pro-2026-07-30", "status": "completed",
        "output": [reasoning, done],
        "usage": {"input_tokens": 2412, "input_tokens_details": {"cached_tokens": 1920},
                  "output_tokens": 188, "output_tokens_details": {"reasoning_tokens": 96},
                  "total_tokens": 2600},
        "routing_hint": "pool-3"})
    return b.text()


def b_length_mid_args():
    b = B()
    item_prelude(b, "resp_b7f3d4", "coder-pro-2026-07-30")
    b.at(390)
    fc = {"type": "function_call", "id": "fc_02", "call_id": "call_test_1", "name": "replace_text",
          "arguments": "", "status": "in_progress"}
    b.e("response.output_item.added", output_index=0, item=fc)
    pieces = ['{"path": "test_report.py", ',
              '"old": "\\n\\nif __name__ == \\"__main__\\":", ',
              '"new": "\\n    def test_order_near_midnight(self):\\n        order_time = datetime.fromisoformat(',
              '\\"2026-09-12T00:30:00+00:00\\")\\n        self.assertTrue(belongs_to_report(\\n']
    for ms, piece in zip([440, 500, 560, 620], pieces):
        b.at(ms)
        b.e("response.function_call_arguments.delta", item_id="fc_02", output_index=0, delta=piece)
    b.at(633)
    b.e("response.incomplete", response={
        "id": "resp_b7f3d4", "object": "response", "model": "coder-pro-2026-07-30", "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [dict(fc, arguments="".join(pieces), status="incomplete")],
        "usage": {"input_tokens": 1987, "input_tokens_details": {"cached_tokens": 1536},
                  "output_tokens": 64, "output_tokens_details": {"reasoning_tokens": 0},
                  "total_tokens": 2051}})
    return b.text()


def b_filtered():
    b = B()
    item_prelude(b, "resp_b7f5e6", "coder-pro-2026-07-30")
    b.at(520)
    b.e("response.incomplete", response={
        "id": "resp_b7f5e6", "object": "response", "model": "coder-pro-2026-07-30", "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"}, "output": [],
        "usage": {"input_tokens": 2530, "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0},
                  "total_tokens": 2530}})
    return b.text()


def b_answer_done():
    b = B()
    item_prelude(b, "resp_b7f6a1", "coder-pro-2026-07-30")
    b.at(300)
    msg = {"type": "message", "id": "msg_b01", "role": "assistant", "status": "in_progress", "content": []}
    b.e("response.output_item.added", output_index=0, item=msg)
    b.e("response.content_part.added", item_id="msg_b01", output_index=0, content_index=0,
        part={"type": "output_text", "text": ""})
    words = ANSWER.split(" ")
    chunks = [" ".join(words[i:i + 9]) + (" " if i + 9 < len(words) else "") for i in range(0, len(words), 9)]
    for k, chunk in enumerate(chunks):
        b.at(330 + 45 * k)
        b.e("response.output_text.delta", item_id="msg_b01", output_index=0, content_index=0, delta=chunk)
    end = 330 + 45 * len(chunks)
    b.at(end)
    b.e("response.output_text.done", item_id="msg_b01", output_index=0, content_index=0, text=ANSWER)
    final = dict(msg, status="completed", content=[{"type": "output_text", "text": ANSWER, "annotations": []}])
    b.e("response.output_item.done", output_index=0, item=final)
    b.at(end + 20)
    b.e("response.completed", response={
        "id": "resp_b7f6a1", "object": "response", "model": "coder-pro-2026-07-30", "status": "completed",
        "output": [final],
        "usage": {"input_tokens": 3120, "input_tokens_details": {"cached_tokens": 2944},
                  "output_tokens": 61, "output_tokens_details": {"reasoning_tokens": 0},
                  "total_tokens": 3181}})
    return b.text()


# ---------------------------------------------------------------- block format

def msg_start(mid, model, input_tokens, cache_read):
    return ev("message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "cache_read_input_tokens": cache_read, "output_tokens": 1}}})


def cbs(i, block):
    return ev("content_block_start", {"type": "content_block_start", "index": i, "content_block": block})


def cbd(i, delta):
    return ev("content_block_delta", {"type": "content_block_delta", "index": i, "delta": delta})


def cbe(i):
    return ev("content_block_stop", {"type": "content_block_stop", "index": i})


def mdelta(reason, out_tokens, **extra):
    d = {"stop_reason": reason, "stop_sequence": None}
    d.update(extra)
    return ev("message_delta", {"type": "message_delta", "delta": d, "usage": {"output_tokens": out_tokens}})


def mstop():
    return ev("message_stop", {"type": "message_stop"})


def a_tests_done():
    p = [t(0), t(150), msg_start("msg_a41c", "coder-large-2026-08-14", 2398, 1800)]
    p += [t(160), cbs(0, {"type": "thinking", "thinking": "", "signature": ""})]
    p += [t(420), cbd(0, {"type": "thinking_delta",
                          "thinking": "The fix is in place at revision 339d3a9f2656. "}),
          t(610), cbd(0, {"type": "thinking_delta",
                          "thinking": "Run both tests again to confirm the near-midnight case now passes."}),
          t(640), cbd(0, {"type": "signature_delta", "signature": SIGNATURE}), cbe(0)]
    p += [t(650), ev("ping", {"type": "ping"})]
    p += [t(700), cbs(1, {"type": "text", "text": ""}),
          cbd(1, {"type": "text_delta", "text": "Running the tests against the changed file."}), cbe(1)]
    p += [t(760), cbs(2, {"type": "tool_use", "id": "toolu_tests_2", "name": "run_tests", "input": {}}),
          t(830), cbd(2, {"type": "input_json_delta", "partial_json": '{"tests": ["test_report.py"]}'}),
          t(835), cbe(2)]
    p += [t(850), mdelta("tool_use", 141), mstop()]
    return "".join(p)


def answer_chunks(text):
    words = text.split(" ")
    return [" ".join(words[i:i + 9]) + (" " if i + 9 < len(words) else "") for i in range(0, len(words), 9)]


def a_answer(length_cut):
    p = [t(0), t(140), msg_start("msg_a51d" if not length_cut else "msg_a51e", "coder-large-2026-08-14", 3104, 2944)]
    p += [t(150), cbs(0, {"type": "text", "text": ""})]
    text = ANSWER
    if length_cut:
        text = ANSWER[:ANSWER.index("now pass.")]
    chunks = answer_chunks(text)
    for k, c in enumerate(chunks):
        p += [t(180 + 45 * k), cbd(0, {"type": "text_delta", "text": c})]
    end = 180 + 45 * len(chunks)
    p += [t(end), cbe(0), mdelta("max_tokens" if length_cut else "end_turn", 48 if length_cut else 62), mstop()]
    return "".join(p)


def a_refusal():
    p = [t(0), t(210), msg_start("msg_a61f", "coder-large-2026-08-14", 2251, 1800)]
    p += [t(260), mdelta("refusal", 3, stop_details={
        "category": "policy", "explanation": "The request was declined by the provider's usage policy."}), mstop()]
    return "".join(p)


def a_unknown_block():
    p = [t(0), t(160), msg_start("msg_a71a", "coder-large-2026-08-14", 2402, 1800)]
    p += [t(200), cbs(0, {"type": "text", "text": ""}),
          cbd(0, {"type": "text_delta", "text": "Updating the plan before running the tests."}), cbe(0)]
    p += [t(260), cbs(1, {"type": "plan_update", "plan": {"steps": [
        {"id": 1, "text": "Run test_report.py", "status": "in_progress"},
        {"id": 2, "text": "Report the result without deploying", "status": "pending"}]}}), cbe(1)]
    p += [t(270), ev("rate_limit_hint", {"type": "rate_limit_hint", "remaining_requests": 41})]
    p += [t(300), cbs(2, {"type": "tool_use", "id": "toolu_tests_3", "name": "run_tests", "input": {}}),
          t(350), cbd(2, {"type": "input_json_delta", "partial_json": '{"tests": ["test_report.py"]}'}), cbe(2)]
    p += [t(360), mdelta("tool_use", 97), mstop()]
    return "".join(p)


def a_two_tools_cut():
    p = [t(0), t(150), msg_start("msg_a81b", "coder-large-2026-08-14", 1204, 0)]
    p += [t(180), cbs(0, {"type": "tool_use", "id": "toolu_read_1", "name": "read_file", "input": {}}),
          t(230), cbd(0, {"type": "input_json_delta", "partial_json": '{"path": "settings.json"}'}),
          t(235), cbe(0)]
    p += [t(260), cbs(1, {"type": "tool_use", "id": "toolu_edit_1", "name": "replace_text", "input": {}}),
          t(300), cbd(1, {"type": "input_json_delta", "partial_json": '{"path": "test_report.py", '}),
          t(360), cbd(1, {"type": "input_json_delta",
                          "partial_json": '"old": "\\n\\nif __name__ == \\"__main__\\":", '})]
    return "".join(p)  # the server closes the connection cleanly here, with no message_delta or message_stop


FIXTURES = {
    "b_fix_done": (lambda: b_fix(False)),
    "b_fix_cut": (lambda: b_fix(True)),
    "b_length_mid_args": b_length_mid_args,
    "b_filtered": b_filtered,
    "b_answer_done": b_answer_done,
    "a_tests_done": a_tests_done,
    "a_answer_done": (lambda: a_answer(False)),
    "a_answer_length": (lambda: a_answer(True)),
    "a_refusal": a_refusal,
    "a_unknown_block": a_unknown_block,
    "a_two_tools_cut": a_two_tools_cut,
}

META = {
    "about": ("Hand-authored SSE fixtures for Chapter 5. Shapes are modeled on documented provider "
              "formats; they are not byte-exact recordings. Times in ': t=' comments are authored."),
    "adapters": {
        "block": {"modeled_on": "Anthropic Messages streaming", "api_version": "2026-06-01",
                  "requested_model": "coder-large"},
        "item": {"modeled_on": "OpenAI Responses streaming", "api_version": "responses-2026-05",
                 "requested_model": "coder-pro"},
    },
    "fixtures": {
        "b_fix_done": {"format": "item", "transport_end": "closed", "request_id": "req_9d01",
                       "about": "Reasoning item plus the report.py fix as one function call; completed."},
        "b_fix_cut": {"format": "item", "transport_end": "reset", "request_id": "req_9cff",
                      "about": "Same call; connection reset inside the fifth argument delta."},
        "b_length_mid_args": {"format": "item", "transport_end": "closed", "request_id": "req_9d02",
                              "about": "Regression-test edit cut by the output token limit mid-argument."},
        "b_filtered": {"format": "item", "transport_end": "closed", "request_id": "req_9d03",
                       "about": "Response ended by a content filter before any output."},
        "b_answer_done": {"format": "item", "transport_end": "closed", "request_id": "req_9d04",
                          "about": "Final answer from Chapter 3, streamed as text; completed."},
        "a_tests_done": {"format": "block", "transport_end": "closed", "request_id": "req_7a11",
                         "about": "Signed thinking block, short text, run_tests tool call; completed."},
        "a_answer_done": {"format": "block", "transport_end": "closed", "request_id": "req_7a12",
                          "about": "Final answer from Chapter 3; end_turn."},
        "a_answer_length": {"format": "block", "transport_end": "closed", "request_id": "req_7a13",
                            "about": "Final answer cut by max_tokens."},
        "a_refusal": {"format": "block", "transport_end": "closed", "request_id": "req_7a14",
                      "about": "Provider refusal with stop details and no content."},
        "a_unknown_block": {"format": "block", "transport_end": "closed", "request_id": "req_7a15",
                            "about": ("Invented 'plan_update' block type and invented 'rate_limit_hint' "
                                      "event, standing for additions from a newer API version.")},
        "a_two_tools_cut": {"format": "block", "transport_end": "closed", "request_id": "req_7a16",
                            "about": ("Complete read_file block, then replace_text cut after two "
                                      "properties; server closes without a terminal event.")},
    },
    "expected": {
        "b_fix_done": {"kind": "completed_tool_calls", "executable": 1,
                       "arguments_sha": ["FIX_ARGS_SHA"]},
        "b_fix_cut": {"kind": "interrupted", "executable": 0},
        "b_length_mid_args": {"kind": "truncated_max_tokens", "executable": 0},
        "b_filtered": {"kind": "refused", "executable": 0},
        "b_answer_done": {"kind": "completed_text", "executable": 0},
        "a_tests_done": {"kind": "completed_tool_calls", "executable": 1,
                         "arguments_sha": ["TEST_ARGS_SHA"]},
        "a_answer_done": {"kind": "completed_text", "executable": 0},
        "a_answer_length": {"kind": "truncated_max_tokens", "executable": 0},
        "a_refusal": {"kind": "refused", "executable": 0},
        "a_unknown_block": {"kind": "unsupported", "executable": 0},
        "a_two_tools_cut": {"kind": "interrupted", "executable": 0},
    },
    "capabilities": {
        "coder-large": {"format": "block", "reasoning": True,
                        "unsupported_with_reasoning": ["temperature", "top_p"],
                        "parallel_tool_calls": "tool_choice.disable_parallel_tool_use"},
        "coder-pro": {"format": "item", "reasoning": True,
                      "unsupported_with_reasoning": ["temperature", "top_p"],
                      "parallel_tool_calls": "parallel_tool_calls"},
        "coder-mini": {"format": "item", "reasoning": False,
                       "unsupported_with_reasoning": [],
                       "parallel_tool_calls": None},
    },
}


def args_sha(obj):
    import hashlib
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main():
    os.makedirs(OUT, exist_ok=True)
    META["expected"]["b_fix_done"]["arguments_sha"] = [args_sha(json.loads(fix_args()))]
    META["expected"]["a_tests_done"]["arguments_sha"] = [args_sha({"tests": ["test_report.py"]})]
    for name, fn in FIXTURES.items():
        with open(os.path.join(OUT, name + ".sse"), "w", encoding="utf-8") as f:
            f.write(fn())
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(META, indent=2) + "\n")
    print("wrote", len(FIXTURES), "fixtures to", OUT)


if __name__ == "__main__":
    main()
