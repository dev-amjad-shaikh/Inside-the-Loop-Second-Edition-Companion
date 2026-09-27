"""Deterministic fixtures shared by all four runtimes (Chapter 23).

transcript.json   about 50 KB: a reporting-bug session (system prompt, task,
                  assistant tool calls, file contents, test logs)
model_reply.json  the assistant tool-call reply every step parses
tool_result.txt   the 2 KB file body a tool returns
state_1mb.json    a checkpoint state of about 1 MB for the serialization test
"""
import json
import os
import random

SYSTEM = (
    "You are a coding agent working in a repository that produces a daily sales report. "
    "Inspect files before editing. Work only in the supplied workspace. Run the tests after "
    "every change and report the revision the tests ran against. Do not deploy. "
) * 6

TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")

TOOLS = ["read_file", "run_tests", "replace_text", "list_files", "search_code", "git_diff"]
PATHS = ["report.py", "test_report.py", "settings.json", "orders/fixtures.csv", "README.md"]


def code_block(rng, lines):
    out = []
    for i in range(lines):
        k = rng.randrange(1000)
        out.append(f"def bucket_{k}(order, day, tz=\"America/Los_Angeles\"):\n"
                   f"    local = order.created_at.astimezone(ZoneInfo(tz))\n"
                   f"    return local.date() == day  # revision 339d3a9f2656 line {i}\n")
    return "".join(out)


def test_log(rng, lines):
    out = []
    for i in range(lines):
        pct = min(100, (i + 1) * 100 // lines)
        status = "PASSED" if rng.random() > 0.08 else "FAILED"
        out.append(f"test_report.py::test_midnight_boundary_{i} {status} [{pct:3d}%]\n")
    return "".join(out)


def assistant_text(rng, i):
    base = [
        "The cutoff compares UTC timestamps with a local calendar day.",
        "Reading the settings to confirm the configured timezone.",
        "The failing test covers an order placed at 23:58 local time.",
        "Patching the comparison so both sides use America/Los_Angeles.",
        "Running the selected tests against the edited revision.",
    ]
    s = base[i % len(base)]
    if i % 5 == 3:
        s += " The report’s public signature stays unchanged."   # typographic apostrophe, as models emit
    return s


def build_transcript(rng, target_bytes):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    i = 0
    while len(json.dumps(msgs)) < target_bytes:
        tool = TOOLS[i % len(TOOLS)]
        path = PATHS[i % len(PATHS)]
        cid = f"call_{i:04d}"
        msgs.append({"role": "assistant", "content": assistant_text(rng, i),
                     "tool_calls": [{"id": cid, "name": tool, "arguments": {"path": path}}]})
        if tool == "run_tests":
            body = test_log(rng, 14)
        else:
            body = code_block(rng, 6)
        msgs.append({"role": "tool", "tool_call_id": cid,
                     "content": json.dumps({"path": path, "step": i, "content": body})})
        i += 1
    return msgs


def write_fixtures(d):
    os.makedirs(d, exist_ok=True)
    rng = random.Random(23)
    transcript = build_transcript(rng, 50_000)
    with open(os.path.join(d, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump(transcript, f)
    reply = {"role": "assistant", "content": "Reading the report module again after the edit.",
             "tool_calls": [{"id": "call_step", "name": "read_file", "arguments": {"path": "report.py"}}]}
    with open(os.path.join(d, "model_reply.json"), "w", encoding="utf-8") as f:
        json.dump(reply, f)
    body = code_block(random.Random(7), 9)[:2048]
    with open(os.path.join(d, "tool_result.txt"), "w", encoding="utf-8") as f:
        f.write(body)
    big = {"session_id": "s-1mb", "step": 400, "transcript": build_transcript(random.Random(11), 1_000_000)}
    with open(os.path.join(d, "state_1mb.json"), "w", encoding="utf-8") as f:
        json.dump(big, f)
    return {
        "transcript_bytes": os.path.getsize(os.path.join(d, "transcript.json")),
        "transcript_messages": len(transcript),
        "reply_bytes": os.path.getsize(os.path.join(d, "model_reply.json")),
        "tool_result_bytes": len(body.encode("utf-8")),
        "state_1mb_bytes": os.path.getsize(os.path.join(d, "state_1mb.json")),
        "state_1mb_messages": len(big["transcript"]),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(write_fixtures(sys.argv[1]), indent=2))
