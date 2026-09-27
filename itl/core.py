"""Core pieces shared by the chapter experiments.

ScriptedModel  replays authored responses and records every request.
Workspace      a temporary copy of a fixture directory with content versions.
ToolRegistry   tool definitions plus implementations, with checked execution.
run_agent      the loop from Chapters 2 and 3.
"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "fixtures")


def content_version(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def dump(obj):
    return json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False)


class ScriptedModel:
    """Returns authored responses in order and keeps a copy of each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, messages, definitions):
        self.requests.append({
            "messages": copy.deepcopy(messages),
            "tools": [d["name"] for d in definitions],
        })
        if not self.responses:
            raise RuntimeError("scripted model has no response left")
        return copy.deepcopy(self.responses.pop(0))


class Workspace:
    def __init__(self, fixture="report"):
        self.root = tempfile.mkdtemp(prefix="itl-")
        shutil.copytree(os.path.join(FIXTURES, fixture), self.root, dirs_exist_ok=True)

    def path(self, rel):
        full = os.path.realpath(os.path.join(self.root, rel))
        if not full.startswith(os.path.realpath(self.root) + os.sep):
            raise PermissionError(f"{rel} is outside the workspace")
        return full

    def read(self, rel):
        with open(self.path(rel), encoding="utf-8") as f:
            return f.read()

    def write(self, rel, text):
        with open(self.path(rel), "w", encoding="utf-8") as f:
            f.write(text)

    def snapshot(self):
        out = {}
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if os.path.isfile(p) and name.endswith((".py", ".json")):
                out[name] = self.read(name)
        return out

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


class InputError(Exception):
    pass


def check_args(schema, args):
    if not isinstance(args, dict):
        raise InputError("arguments must be an object")
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            raise InputError(f"{name} is required")
    if schema.get("additionalProperties") is False:
        extra = sorted(set(args) - set(props))
        if extra:
            raise InputError(f"unexpected argument {extra[0]}")
    for name, value in args.items():
        spec = props.get(name, {})
        kind = spec.get("type")
        if kind == "string" and not isinstance(value, str):
            raise InputError(f"{name} must be a string")
        if kind == "array":
            item = spec.get("items", {}).get("type")
            if not isinstance(value, list) or (item == "string" and not all(isinstance(v, str) for v in value)):
                raise InputError(f"{name} must be a list of strings")


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.executions = []

    def register(self, name, description, parameters, fn):
        self.tools[name] = {"definition": {"name": name, "description": description,
                                           "parameters": parameters}, "fn": fn}

    def definitions(self):
        return [t["definition"] for t in self.tools.values()]

    def execute_checked(self, call):
        tool = self.tools.get(call["name"])
        if tool is None:
            return {"status": "input_error", "message": f"unknown tool {call['name']}"}
        try:
            check_args(tool["definition"]["parameters"], call["arguments"])
        except InputError as exc:
            return {"status": "input_error", "message": str(exc)}
        self.executions.append({"call_id": call["id"], "name": call["name"],
                                "arguments": copy.deepcopy(call["arguments"])})
        return tool["fn"](**call["arguments"])


def report_tools(ws):
    reg = ToolRegistry()

    def read_file(path):
        text = ws.read(path)
        return {"status": "ok", "path": path, "text": text}

    def replace_text(path, old, new):
        text = ws.read(path)
        count = text.count(old)
        if count != 1:
            return {"status": "input_error",
                    "message": f"old text occurs {count} times; expected exactly once"}
        before = content_version(text)
        updated = text.replace(old, new, 1)
        ws.write(path, updated)
        return {"status": "ok", "path": path, "replacements": 1,
                "before_version": before, "after_version": content_version(updated)}

    def run_tests(tests):
        modules = [t[:-3] if t.endswith(".py") else t for t in tests]
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", *modules],
                              cwd=ws.root, capture_output=True, text=True, timeout=60)
        out = proc.stderr
        failed, fmsgs, errs = [], [], []
        run = 0
        for line in out.splitlines():
            if line.startswith("Ran "):
                run = int(line.split()[1])
            if line.startswith("FAIL: ") or line.startswith("ERROR: "):
                # FAIL: test_x (test_report.ReportTests.test_x)
                inner = line.split("(", 1)[1].rstrip(")")
                failed.append(inner)
            if line.startswith("AssertionError"):
                fmsgs.append(line.strip())
        errors = sum(1 for l in out.splitlines() if l.startswith("ERROR: "))
        failures = sum(1 for l in out.splitlines() if l.startswith("FAIL: "))
        return {"status": "passed" if proc.returncode == 0 else "failed",
                "tests_run": run, "failures": failures, "errors": errors,
                "failed_tests": failed, "failure_messages": fmsgs,
                "error_messages": errs, "exit_code": proc.returncode}

    reg.register("read_file", "Read a text file in the workspace.", {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"], "additionalProperties": False}, read_file)
    reg.register("replace_text", "Replace one exact occurrence of text in a workspace file.", {
        "type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                         "new": {"type": "string"}},
        "required": ["path", "old", "new"], "additionalProperties": False}, replace_text)
    reg.register("run_tests", "Run the selected Python test files.", {
        "type": "object", "properties": {"tests": {"type": "array", "items": {"type": "string"}}},
        "required": ["tests"], "additionalProperties": False}, run_tests)
    return reg


def run_agent(model, tools, messages, limit=12, append_observations=True, verifier=None):
    definitions = tools.definitions()
    for turn in range(limit):
        reply = model.generate(messages, definitions)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            if verifier is not None:
                verdict = verifier(messages)
                if not verdict["complete"]:
                    messages.append({"role": "user", "content": verdict["message"]})
                    continue
            return reply["content"]
        for call in calls:
            observation = tools.execute_checked(call)
            if append_observations:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": observation})
    raise RuntimeError("Model-call limit reached")


def write_run(outdir, name, obj):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            f.write(dump(obj) + "\n")
