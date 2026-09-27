"""Chapter 3: one complete run.

Replays seven authored model responses against a real temporary workspace.
File reads, edits and test runs are executed for real.

    python3 ch03_chapter_lab.py --output my-run
    python3 ch03_chapter_lab.py --output drop-run --break drop-observations
    python3 ch03_chapter_lab.py --output swap-run --break swap-ids
    python3 ch03_chapter_lab.py --output stop-run --break stop-after-edit
    python3 ch03_chapter_lab.py --output verify-run --verifier --stop-early
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itl.core import ScriptedModel, Workspace, report_tools, run_agent, write_run  # noqa: E402

SYSTEM = ("Inspect files before editing. Work only in the supplied workspace. "
          "Workspace files: settings.json, report.py, test_report.py.")
TASK = ("The daily report is missing some orders placed near midnight. Find the cause, "
        "prepare a fix and run the tests. Do not deploy it.")

NEW_TEST = '''    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":'''

ANSWER = ("The filter used the UTC calendar date. I changed it to convert each timestamp to the "
          "configured report timezone before comparing dates. The new near-midnight test failed "
          "before the fix; both tests now pass. I changed report.py and test_report.py. "
          "Nothing was deployed.")


def call(cid, name, **arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def responses(stop_early=False):
    r = [
        {"role": "assistant", "content": "", "tool_calls": [
            call("c1", "read_file", path="settings.json"),
            call("c2", "read_file", path="report.py"),
            call("c3", "read_file", path="test_report.py")]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("c4", "replace_text", path="test_report.py",
                 old='\n\nif __name__ == "__main__":', new="\n" + NEW_TEST)]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("c5", "run_tests", tests="test_report.py")]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("c6", "run_tests", tests=["test_report.py"])]},
        {"role": "assistant", "content": "", "tool_calls": [
            call("c7", "replace_text", path="report.py",
                 old="    return created_at.date() == requested_day",
                 new="    local_time = created_at.astimezone(ZoneInfo(timezone))\n"
                     "    return local_time.date() == requested_day")]},
    ]
    if stop_early:
        r.append({"role": "assistant", "content": ANSWER, "tool_calls": []})
        r.append({"role": "assistant", "content": "", "tool_calls": [
            call("c8", "run_tests", tests=["test_report.py"])]})
    else:
        r.append({"role": "assistant", "content": "", "tool_calls": [
            call("c8", "run_tests", tests=["test_report.py"])]})
    r.append({"role": "assistant", "content": ANSWER, "tool_calls": []})
    return r


def verifier(messages):
    """Accept completion only if the latest test result passed after the latest edit."""
    last_edit = last_pass = -1
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            c = m["content"]
            if "after_version" in c:
                last_edit = i
            if c.get("status") == "passed":
                last_pass = i
    if last_pass > last_edit >= 0:
        return {"complete": True}
    return {"complete": False,
            "message": "Verification required: no passing test result recorded after the latest edit."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--break", dest="brk", choices=["drop-observations", "swap-ids", "stop-after-edit"])
    ap.add_argument("--verifier", action="store_true")
    ap.add_argument("--stop-early", action="store_true")
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit(f"{args.output} exists; choose a new directory")

    ws = Workspace("report")
    before = ws.snapshot()
    tools = report_tools(ws)
    model = ScriptedModel(responses(stop_early=args.stop_early))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]

    if args.brk == "swap-ids":
        original = tools.execute_checked
        pending = {}

        def swapped(c):
            out = original(c)
            if c["id"] in ("c1", "c2"):
                pending[c["id"]] = out
                if len(pending) == 2:
                    # attach each result to the other request
                    messages.append({"role": "tool", "tool_call_id": "c1", "content": pending["c2"]})
                    messages.append({"role": "tool", "tool_call_id": "c2", "content": pending["c1"]})
                return None
            return out
        tools.execute_checked = swapped
        answer = run_agent_filtered(model, tools, messages)
    elif args.brk == "stop-after-edit":
        original = tools.execute_checked

        def crashing(c):
            out = original(c)
            if c["id"] == "c7":
                raise SystemExit("process stopped after the edit executed, before its result was recorded")
            return out
        tools.execute_checked = crashing
        try:
            answer = run_agent(model, tools, messages)
        except SystemExit as exc:
            answer = f"[run interrupted] {exc}"
    else:
        answer = run_agent(model, tools, messages,
                           append_observations=(args.brk != "drop-observations"),
                           verifier=verifier if args.verifier else None)

    after = ws.snapshot()
    out = args.output
    write_run(out, "trace.json", model.requests)
    write_run(out, "messages.json", messages)
    write_run(out, "executions.json", tools.executions)
    write_run(out, "answer.txt", answer + "\n")
    for label, snap in (("before", before), ("after", after)):
        for name, text in snap.items():
            write_run(os.path.join(out, label), name, text)
    counts = [len(r["messages"]) for r in model.requests]
    print("model calls:", len(model.requests))
    print("input message counts:", counts)
    print("requested tool calls:", sum(len(m.get("tool_calls") or []) for m in messages if m.get("role") == "assistant"))
    print("executions:", len(tools.executions))
    print("answer:", answer)
    ws.close()


def run_agent_filtered(model, tools, messages, limit=12):
    """Loop variant for the swap experiment: None observations are appended elsewhere."""
    definitions = tools.definitions()
    for _ in range(limit):
        reply = model.generate(messages, definitions)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            return reply["content"]
        for c in calls:
            obs = tools.execute_checked(c)
            if obs is not None:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
    raise RuntimeError("Model-call limit reached")


if __name__ == "__main__":
    main()
