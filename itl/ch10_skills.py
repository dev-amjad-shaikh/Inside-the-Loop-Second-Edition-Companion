"""Chapter 10 helpers: a minimal Agent Skills client in pure Python.

Pieces, in the order a run uses them:

parse_frontmatter  a small YAML *subset* parser for SKILL.md frontmatter
                   (scalars, quoted scalars, > and | block scalars, one level
                   of nested string mapping). It is not a YAML parser: no
                   lists, anchors, tags, flow collections or escapes.
validate_skill     the checks the skills-ref reference validator documents
                   (agentskills/skills-ref, validator.py), reimplemented.
lint_skill         this chapter's own quality and authority checks, which the
                   format validator does not attempt.
discover           scan a skills directory for <dir>/SKILL.md.
render_catalog     the tier-1 metadata block placed in the model input.
LexicalActivator   a deterministic stand-in for the model's choice of skill.
SkillExecutor      routes every proposed tool or script use through the run's
                   policy and the active skills' declarations, and runs skill
                   scripts under an audit-hook guard.
Registry           versions, content hashes, release and withdrawal.
run_skill_agent    the Chapter 3 loop, with activation and the executor.
"""
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Frontmatter: a deliberately small YAML subset
# ---------------------------------------------------------------------------


class FrontmatterError(Exception):
    pass


def _unquote(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        inner = value[1:-1]
        return inner.replace("''", "'") if value[0] == "'" else inner.replace('\\"', '"')
    return value


def _strip_comment(value):
    if value.startswith(("'", '"')):
        return value
    m = re.search(r"\s#", value)
    return value[:m.start()].rstrip() if m else value


def split_skill_md(text):
    """Return (frontmatter_text, body). Raises FrontmatterError."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise FrontmatterError("SKILL.md must start with YAML frontmatter (---)")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---" and not lines[i].startswith((" ", "\t")):
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:]).strip()
    raise FrontmatterError("frontmatter not closed with ---")


def parse_frontmatter_block(block):
    """Parse the supported subset into a dict of str -> str | dict[str, str]."""
    data = {}
    lines = block.split("\n")
    i = 0

    def indent(s):
        return len(s) - len(s.lstrip(" "))

    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if indent(line) != 0:
            raise FrontmatterError(f"unexpected indentation: {line!r}")
        if ":" not in line:
            raise FrontmatterError(f"line is not 'key: value': {line!r}")
        key, value = line.split(":", 1)
        key, value = key.strip(), _strip_comment(value.strip())
        if key in data:
            raise FrontmatterError(f"duplicate key {key!r}")
        i += 1
        if value and value[0] in ">|":
            # block scalar: take following lines indented deeper than the key
            chunk = []
            while i < len(lines) and (not lines[i].strip() or indent(lines[i]) > 0):
                chunk.append(lines[i])
                i += 1
            while chunk and not chunk[-1].strip():
                chunk.pop()
            base = min((indent(c) for c in chunk if c.strip()), default=0)
            body = [c[base:] for c in chunk]
            if value[0] == "|":
                data[key] = "\n".join(body)
            else:  # folded: join lines of a paragraph with spaces
                paras, cur = [], []
                for c in body:
                    if c.strip():
                        cur.append(c.strip())
                    else:
                        paras.append(" ".join(cur))
                        cur = []
                paras.append(" ".join(cur))
                data[key] = "\n".join(p for p in paras if p)
        elif value == "":
            # nested mapping of scalars (one level), e.g. metadata:
            nested = {}
            while i < len(lines) and (not lines[i].strip() or indent(lines[i]) > 0):
                sub = lines[i]
                i += 1
                if not sub.strip() or sub.strip().startswith("#"):
                    continue
                if ":" not in sub or sub.strip().startswith("- "):
                    raise FrontmatterError(f"unsupported nested form: {sub.strip()!r}")
                k, v = sub.split(":", 1)
                v = _strip_comment(v.strip())
                if v == "" or v[0] in ">|[{":
                    raise FrontmatterError(f"unsupported nested value for {k.strip()!r}")
                nested[k.strip()] = _unquote(v)
            data[key] = nested
        else:
            if value[0] in "[{":
                raise FrontmatterError(f"flow collections are not supported ({key})")
            # plain multi-line scalars: continuation lines indented deeper
            parts = [_unquote(value)]
            while i < len(lines) and lines[i].strip() and indent(lines[i]) > 0:
                parts.append(lines[i].strip())
                i += 1
            data[key] = " ".join(parts)
    return data


def parse_frontmatter(text):
    block, body = split_skill_md(text)
    return parse_frontmatter_block(block), body


# ---------------------------------------------------------------------------
# Format validation: the skills-ref checks, reimplemented
# ---------------------------------------------------------------------------

ALLOWED_FIELDS = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}


def validate_skill(skill_dir):
    """Reimplements agentskills/skills-ref validator.py: returns a list of errors."""
    import unicodedata
    if not os.path.isdir(skill_dir):
        return [f"Not a directory: {skill_dir}"]
    md = None
    for n in ("SKILL.md", "skill.md"):
        if os.path.exists(os.path.join(skill_dir, n)):
            md = os.path.join(skill_dir, n)
            break
    if md is None:
        return ["Missing required file: SKILL.md"]
    try:
        meta, _ = parse_frontmatter(open(md, encoding="utf-8").read())
    except FrontmatterError as exc:
        return [str(exc)]
    errors = []
    extra = set(meta) - ALLOWED_FIELDS
    if extra:
        errors.append(f"Unexpected fields in frontmatter: {', '.join(sorted(extra))}")
    name = meta.get("name")
    if name is None:
        errors.append("Missing required field in frontmatter: name")
    elif not isinstance(name, str) or not name.strip():
        errors.append("Field 'name' must be a non-empty string")
    else:
        name = unicodedata.normalize("NFKC", name.strip())
        if len(name) > 64:
            errors.append("Skill name exceeds 64 character limit")
        if name != name.lower():
            errors.append("Skill name must be lowercase")
        if name.startswith("-") or name.endswith("-"):
            errors.append("Skill name cannot start or end with a hyphen")
        if "--" in name:
            errors.append("Skill name cannot contain consecutive hyphens")
        if not all(c.isalnum() or c == "-" for c in name):
            errors.append("Skill name contains invalid characters")
        if unicodedata.normalize("NFKC", os.path.basename(os.path.normpath(skill_dir))) != name:
            errors.append("Directory name must match skill name")
    desc = meta.get("description")
    if desc is None:
        errors.append("Missing required field in frontmatter: description")
    elif not isinstance(desc, str) or not desc.strip():
        errors.append("Field 'description' must be a non-empty string")
    elif len(desc) > 1024:
        errors.append("Description exceeds 1024 character limit")
    comp = meta.get("compatibility")
    if comp is not None and (not isinstance(comp, str) or len(comp) > 500):
        errors.append("Compatibility must be a string of at most 500 characters")
    return errors


# ---------------------------------------------------------------------------
# Package reading, hashing and linting
# ---------------------------------------------------------------------------


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def package_files(skill_dir):
    out = {}
    for root, dirs, files in os.walk(skill_dir):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".pyc") or "__pycache__" in root:
                continue
            p = os.path.join(root, f)
            out[os.path.relpath(p, skill_dir).replace(os.sep, "/")] = open(p, "rb").read()
    return out


def package_hash(skill_dir):
    lines = [f"{rel} {sha256_bytes(data)}" for rel, data in sorted(package_files(skill_dir).items())]
    return sha256_bytes("\n".join(lines).encode())


def approx_tokens(text):
    """Rough size: ceil(chars / 4). Codex's catalog renderer uses the same ratio."""
    return math.ceil(len(text) / 4)


NETWORK_PATTERNS = [r"\burllib\.request\b", r"\brequests\.", r"\bsocket\.", r"\bhttp\.client\b",
                    r"\bcurl\b", r"\bwget\b"]


def parse_pins(meta):
    pins = {}
    raw = (meta.get("metadata") or {}).get("pinned-scripts", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if entry:
            path, _, digest = entry.partition(" sha256:")
            pins[path.strip()] = digest.strip()
    return pins


def lint_skill(skill_dir):
    """Quality and authority checks the format validator does not make.

    These are this chapter's heuristics, not part of the specification."""
    meta, body = parse_frontmatter(open(os.path.join(skill_dir, "SKILL.md"), encoding="utf-8").read())
    files = package_files(skill_dir)
    desc = meta.get("description", "")
    md = meta.get("metadata") or {}
    findings = []

    def f(check, ok, detail):
        findings.append({"check": check, "ok": ok, "detail": detail})

    f("description names a trigger", bool(re.search(r"\bUse (when|for)\b", desc)),
      "description says when to use the skill")
    f("description names a non-trigger", "Do not use" in desc,
      "description says when not to use the skill")
    f("maintainer declared", "maintainer" in md, md.get("maintainer", "missing"))
    f("version declared", "version" in md, md.get("version", "missing"))
    steps = re.findall(r"^\d+\.\s", body, flags=re.M)
    f("stepwise procedure", len(steps) >= 3, f"{len(steps)} numbered steps")
    f("verification step", bool(re.search(r"\b(confirm|verify|must be ok|must pass)\b", body, re.I)),
      "instructions ask for evidence before finishing")
    refs = set(re.findall(r"`?((?:scripts|references|assets)/[\w./-]+)`?", body))
    missing = sorted(r for r in refs if r not in files)
    f("referenced resources exist", not missing, missing or "all present")
    scripts = sorted(p for p in files if p.startswith("scripts/"))
    pins = parse_pins(meta)
    unpinned = [s for s in scripts if s not in pins]
    f("scripts pinned by hash", not unpinned, unpinned or "all pinned")
    stale = [s for s, d in pins.items() if s in files and sha256_bytes(files[s]) != d]
    f("pins match files", not stale, stale or "all match")
    declared_net = md.get("network", "undeclared")
    net_use = sorted(s for s in scripts
                     if any(re.search(p, files[s].decode("utf-8", "replace")) for p in NETWORK_PATTERNS))
    f("network use declared", not (net_use and declared_net in ("none", "undeclared")),
      {"declared": declared_net, "scripts_using_network": net_use})
    return findings


# ---------------------------------------------------------------------------
# Discovery and the tier-1 catalog
# ---------------------------------------------------------------------------


def discover(root):
    """Scan root for <dir>/SKILL.md. Lenient: warn on cosmetic issues, skip unusable skills."""
    records, diagnostics = [], []
    root = os.path.abspath(root)
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        md = os.path.join(d, "SKILL.md")
        if not os.path.isfile(md):
            continue
        try:
            meta, body = parse_frontmatter(open(md, encoding="utf-8").read())
        except FrontmatterError as exc:
            diagnostics.append({"skill_dir": entry, "level": "error", "message": str(exc)})
            continue
        if not meta.get("description"):
            diagnostics.append({"skill_dir": entry, "level": "error", "message": "no description; skipped"})
            continue
        if meta.get("name") != entry:
            diagnostics.append({"skill_dir": entry, "level": "warning", "message": "name differs from directory"})
        records.append({
            "name": meta.get("name", entry),
            "description": " ".join(meta["description"].split()),
            "location": md,
            "dir": d,
            "version": (meta.get("metadata") or {}).get("version", "unversioned"),
            "hash": package_hash(d),
            "allowed_tools": (meta.get("allowed-tools") or "").split(),
            "meta": meta,
        })
    return records, diagnostics


CATALOG_INSTRUCTIONS = (
    "The following skills provide specialized instructions for specific tasks. "
    "When a task matches a skill's description, call activate_skill with the skill's "
    "name to load its full instructions.")


def render_catalog(records):
    if not records:
        return ""
    parts = [CATALOG_INSTRUCTIONS, "<available_skills>"]
    for r in records:
        parts.append(f"<skill><name>{r['name']}</name><description>{r['description']}</description></skill>")
    parts.append("</available_skills>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Activation: a deterministic lexical stand-in for the model's choice
# ---------------------------------------------------------------------------

STOP = set("""a an the and or of to for in on at by with from as is are be it its this that these those
use used using when which who what how do does not no if into than then there their them they
you your can will would should about any anything else such such as me my our we us i
""".split())


def stem(word):
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def terms(text):
    return {stem(w) for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOP and len(w) > 2}


class LexicalActivator:
    """Scores each catalog entry by term overlap with the request.

    Positive terms come from the name and from description sentences that say
    what the skill does or when to use it. Terms that appear only in a
    sentence starting "Do not use" count against the skill. This models the
    intended effect of a non-trigger clause; it cannot show whether a given
    model honors one."""

    def __init__(self, records, threshold=2, margin=1):
        self.threshold, self.margin = threshold, margin
        self.index = []
        for r in records:
            sentences = re.split(r"(?<=\.)\s+", r["description"])
            pos, neg = set(terms(r["name"].replace("-", " "))), set()
            for s in sentences:
                (neg if s.strip().startswith("Do not use") else pos).update(terms(s))
            self.index.append((r["name"], pos, neg - pos))

    def score(self, request):
        q = terms(request)
        out = []
        for name, pos, neg in self.index:
            hits, against = sorted(q & pos), sorted(q & neg)
            out.append({"skill": name, "score": len(hits) - 2 * len(against),
                        "matched": hits, "against": against})
        return sorted(out, key=lambda s: -s["score"])  # stable: catalog order breaks ties

    def choose(self, request):
        ranked = self.score(request)
        top = ranked[0]
        chosen = top["skill"] if top["score"] >= self.threshold else None
        contested = [s["skill"] for s in ranked[1:]
                     if s["score"] >= self.threshold and top["score"] - s["score"] <= self.margin]
        return {"chosen": chosen, "contested_with": contested if chosen else [], "ranking": ranked}


# ---------------------------------------------------------------------------
# Registry: versions, release, withdrawal
# ---------------------------------------------------------------------------


class Registry:
    def __init__(self):
        self.skills = {}
        self.log = []

    def publish(self, name, version, source_dir, maintainer, at, evidence=None):
        entry = self.skills.setdefault(name, {"maintainer": maintainer, "versions": []})
        v = {"version": version, "hash": package_hash(source_dir), "status": "released",
             "released_at": at, "source": source_dir, "evidence": evidence or {}}
        entry["versions"].append(v)
        self.log.append({"at": at, "event": "released", "skill": name, "version": version,
                         "hash": v["hash"][:12], "by": maintainer})
        return v

    def withdraw(self, name, version, at, by, reason):
        for v in self.skills[name]["versions"]:
            if v["version"] == version:
                v["status"] = "withdrawn"
                v["withdrawn_at"], v["withdrawn_reason"] = at, reason
                self.log.append({"at": at, "event": "withdrawn", "skill": name, "version": version,
                                 "by": by, "reason": reason})
                return v
        raise KeyError(version)

    def resolve(self, name):
        """Newest released version; withdrawn versions are never resolved."""
        live = [v for v in self.skills[name]["versions"] if v["status"] == "released"]
        return live[-1] if live else None

    def snapshot(self):
        return {n: {"maintainer": e["maintainer"],
                    "versions": [{k: v[k] for k in v if k != "source"} for v in e["versions"]]}
                for n, e in self.skills.items()}


def materialize(registry, names, dest):
    """Copy the resolved version of each skill into dest/<name>/ (a skills directory)."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    for n in names:
        v = registry.resolve(n)
        if v is not None:
            shutil.copytree(v["source"], os.path.join(dest, n))
    return dest


# ---------------------------------------------------------------------------
# The executor: run policy, skill declarations, guarded scripts
# ---------------------------------------------------------------------------

GUARD = r'''
import json, runpy, sys
AUDIT, SCRIPT = sys.argv[1], sys.argv[2]
DENIED = ("socket.getaddrinfo", "socket.connect", "socket.gethostbyname",
          "subprocess.Popen", "os.system", "os.posix_spawn", "os.exec")
def hook(event, args):
    if event.startswith(DENIED):
        with open(AUDIT, "a") as f:
            f.write(json.dumps({"event": event, "args": [str(a) for a in args][:2]}) + "\n")
        raise PermissionError("guard: " + event + " denied by run policy")
sys.addaudithook(hook)
sys.argv = [SCRIPT] + sys.argv[3:]
runpy.run_path(SCRIPT, run_name="__main__")
'''


class RunPolicy:
    def __init__(self, allowed_tools, network=False, require_pinned_scripts=False, policy_id="run-policy"):
        self.allowed_tools = set(allowed_tools)
        self.network = network
        self.require_pinned_scripts = require_pinned_scripts
        self.policy_id = policy_id

    def describe(self):
        return {"policy_id": self.policy_id, "allowed_tools": sorted(self.allowed_tools),
                "network": self.network, "require_pinned_scripts": self.require_pinned_scripts}


class SkillExecutor:
    """Every proposed operation passes here. Order of checks:

    1. the run policy permits the tool (identity and scope come from the run);
    2. if active skills declare allowed-tools, the tool is in their union
       (a skill can narrow what the run may do, never widen it);
    3. argument shape (Chapter 3's execute_checked);
    4. for run_skill_script: the script belongs to an active skill, its hash
       matches the skill's pin, and it runs under the guard.
    """

    def __init__(self, registry, policy, catalog_records, workspace):
        self.tools = registry
        self.policy = policy
        self.catalog = {r["name"]: r for r in catalog_records}
        self.ws = workspace
        self.active = {}
        self.decisions = []
        self.loaded_resources = []
        self.scripts = []

    # -- activation --------------------------------------------------------
    def activate(self, name, mode):
        rec = self.catalog.get(name)
        if rec is None:
            return {"status": "input_error", "message": f"unknown skill {name}; available: {sorted(self.catalog)}"}
        if name in self.active:
            return {"status": "ok", "note": f"{name} already active; not injected again"}
        _, body = parse_frontmatter(open(rec["location"], encoding="utf-8").read())
        not_permitted = sorted(set(rec["allowed_tools"]) - self.policy.allowed_tools)
        self.active[name] = rec
        resources = sorted(p for p in package_files(rec["dir"]) if p != "SKILL.md")
        record = {"event": "activate", "skill": name, "version": rec["version"], "hash": rec["hash"][:12],
                  "mode": mode, "declared_tools": rec["allowed_tools"],
                  "declared_but_not_permitted": not_permitted,
                  "body_tokens": approx_tokens(body), "resources_listed": resources}
        self.decisions.append(record)
        content = (f'<skill_content name="{name}" version="{rec["version"]}">\n{body}\n\n'
                   f"Skill directory: skills/{name}\n<skill_resources>\n"
                   + "\n".join(f"<file>{p}</file>" for p in resources)
                   + "\n</skill_resources>\n</skill_content>")
        if not_permitted:
            content += ("\nRuntime note: this run does not permit " + ", ".join(not_permitted)
                        + "; steps that need it will be refused.")
        return {"status": "ok", "skill": name, "version": rec["version"], "content": content}

    # -- the authority boundary -------------------------------------------
    def execute(self, call):
        name = call["name"]
        base = {"call_id": call["id"], "tool": name}
        if name not in self.policy.allowed_tools:
            d = dict(base, decision="refused", rule="not permitted by run policy",
                     policy_id=self.policy.policy_id)
            self.decisions.append(d)
            return {"status": "refused", "message": f"{name} is not permitted in this run"}
        declared = set()
        for rec in self.active.values():
            declared |= set(rec["allowed_tools"])
        runtime_tools = {"activate_skill"}
        if declared and name not in declared | runtime_tools:
            d = dict(base, decision="refused", rule="outside active skills' declared tools",
                     declared=sorted(declared))
            self.decisions.append(d)
            return {"status": "refused", "message": f"{name} is outside the active skills' declared tools"}
        if name == "activate_skill":
            out = self.activate(call["arguments"].get("name", ""), mode="model")
            self.decisions.append(dict(base, decision="executed"))
            return out
        if name == "read_skill_resource":
            out = self._read_resource(**call["arguments"])
        elif name == "run_skill_script":
            out = self._run_script(call["id"], **call["arguments"])
        else:
            out = self.tools.execute_checked(call)
        self.decisions.append(dict(base, decision="executed" if out.get("status") not in
                                   ("input_error", "refused") else out["status"]))
        return out

    def _read_resource(self, skill, path):
        rec = self.active.get(skill)
        if rec is None:
            return {"status": "refused", "message": f"skill {skill} is not active"}
        files = package_files(rec["dir"])
        if path not in files:
            return {"status": "input_error", "message": f"{path} not in {skill}; available: {sorted(files)}"}
        text = files[path].decode("utf-8")
        self.loaded_resources.append({"skill": skill, "path": path, "tokens": approx_tokens(text)})
        return {"status": "ok", "skill": skill, "path": path, "text": text}

    def _run_script(self, call_id, skill, script, args):
        rec = self.active.get(skill)
        entry = {"call_id": call_id, "skill": skill, "version": rec["version"] if rec else None,
                 "script": script, "args": args}
        if rec is None:
            entry.update(decision="refused", rule="skill not active")
            self.scripts.append(entry)
            return {"status": "refused", "message": f"skill {skill} is not active"}
        files = package_files(rec["dir"])
        if script not in files or not script.startswith("scripts/"):
            entry.update(decision="refused", rule="script not in skill package")
            self.scripts.append(entry)
            return {"status": "refused", "message": f"{script} is not a script of {skill}"}
        pins = parse_pins(rec["meta"])
        actual = sha256_bytes(files[script])
        entry["sha256"] = actual[:12]
        if script in pins and pins[script] != actual:
            entry.update(decision="refused", rule="script hash does not match pin",
                         pinned=pins[script][:12])
            self.scripts.append(entry)
            return {"status": "refused",
                    "message": f"{script} changed since {skill} {rec['version']} was released"}
        if script not in pins:
            if self.policy.require_pinned_scripts:
                entry.update(decision="refused", rule="unpinned script")
                self.scripts.append(entry)
                return {"status": "refused", "message": f"{script} is not pinned by {skill}"}
            entry["warning"] = "unpinned script"
        with tempfile.TemporaryDirectory(prefix="itl-guard-") as tmp:
            guard = os.path.join(tmp, "guard.py")
            audit = os.path.join(tmp, "audit.jsonl")
            open(guard, "w").write(GUARD)
            open(audit, "w").close()
            # a clean environment: no proxy settings, credentials or tokens reach the script
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
            proc = subprocess.run([sys.executable, guard, audit, os.path.join(rec["dir"], script), *args],
                                  cwd=self.ws.root, capture_output=True, text=True, timeout=60, env=env)
            denied = [json.loads(l) for l in open(audit) if l.strip()]
        tail = proc.stderr.strip().splitlines()[-1:] if proc.stderr.strip() else []
        entry.update(decision="executed", exit_code=proc.returncode, denied_effects=denied)
        self.scripts.append(entry)
        out = {"status": "ok" if proc.returncode == 0 else "failed", "exit_code": proc.returncode,
               "stdout": proc.stdout.strip(), "stderr_tail": tail}
        if denied:
            out["denied_effects"] = denied
        return out


def skill_tool_definitions(catalog_records):
    names = [r["name"] for r in catalog_records]
    return [
        {"name": "activate_skill", "description": "Load a skill's full instructions by name.",
         "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": names}},
                        "required": ["name"], "additionalProperties": False}},
        {"name": "read_skill_resource", "description": "Read one file from an active skill.",
         "parameters": {"type": "object", "properties": {"skill": {"type": "string"},
                                                         "path": {"type": "string"}},
                        "required": ["skill", "path"], "additionalProperties": False}},
        {"name": "run_skill_script", "description": "Run a script bundled with an active skill.",
         "parameters": {"type": "object", "properties": {"skill": {"type": "string"},
                                                         "script": {"type": "string"},
                                                         "args": {"type": "array", "items": {"type": "string"}}},
                        "required": ["skill", "script", "args"], "additionalProperties": False}},
    ]


def run_skill_agent(model, executor, definitions, messages, limit=14):
    """Chapter 3's loop. The only change: calls go through the SkillExecutor,
    and an activation result's content is appended as the tool observation."""
    for _ in range(limit):
        reply = model.generate(messages, definitions)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            return reply["content"]
        for call in calls:
            observation = executor.execute(call)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
    raise RuntimeError("Model-call limit reached")


def input_accounting(model):
    """Approximate size of each recorded model input, split by origin."""
    rows = []
    for i, req in enumerate(model.requests, 1):
        catalog = skills = resources = other = 0
        for m in req["messages"]:
            c = m.get("content")
            text = c if isinstance(c, str) else json.dumps(c)
            t = approx_tokens(text)
            if m["role"] == "system" and "<available_skills>" in text:
                catalog += approx_tokens(text[text.index(CATALOG_INSTRUCTIONS):])
                other += t - approx_tokens(text[text.index(CATALOG_INSTRUCTIONS):])
            elif "<skill_content" in text:
                skills += t
            elif m["role"] == "tool" and isinstance(c, dict) and "skill" in c and "path" in c:
                resources += t
            else:
                other += t
        rows.append({"call": i, "catalog": catalog, "skill_body": skills,
                     "skill_resources": resources, "other": other})
    return rows
