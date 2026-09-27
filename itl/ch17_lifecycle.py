"""Chapter 17 helpers: a shared catalog, release manifests, pointers, a fleet
simulation and withdrawal.

The catalog is a SQLite database of versioned parts (tools, skills, prompts,
models, policies, context policies, controllers, evaluators, runtime builds).
Each row carries an owner, a semantic version, a content hash, declared
permissions and data scope, dependencies and release evidence. A release is an
immutable manifest that pins every part of one agent by hash. Environments
bind manifests through pointers (active plus an optional canary). Runs record
the manifest they were admitted under.

Everything here is deterministic. Simulated time is in hours from
2026-09-01 00:00 (hour 0); nothing reads the wall clock.
"""
import hashlib
import json
import sqlite3

SIM_START = "2026-09-01"


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(obj):
    data = obj if isinstance(obj, bytes) else canon(obj)
    return hashlib.sha256(data).hexdigest()


def hour_label(h):
    """Hour offset -> 'Sep 08 09:00' (simulated clock, minutes rounded down)."""
    m = int(round(h * 60, 6))
    day, rest = divmod(m, 24 * 60)
    return f"Sep {day + 1:02d} {rest // 60:02d}:{rest % 60:02d}"


def semver(v):
    return tuple(int(x) for x in v.split("."))


class Refused(Exception):
    pass


CATALOG_SCHEMA = """
CREATE TABLE parts (
  kind TEXT, name TEXT, version TEXT, content_hash TEXT, owner TEXT,
  status TEXT, released_at INTEGER, status_at INTEGER, status_reason TEXT,
  replacement TEXT, deprecated_until TEXT,
  permissions TEXT, data_scope TEXT, content TEXT,
  PRIMARY KEY (kind, name, version));
CREATE TABLE depends (
  consumer TEXT, consumer_owner TEXT, kind TEXT, name TEXT, version TEXT);
CREATE TABLE evidence (
  subject TEXT, kind TEXT, result TEXT, at INTEGER, by TEXT);
CREATE TABLE manifests (
  manifest_id TEXT PRIMARY KEY, agent TEXT, label TEXT, parts TEXT, created_at INTEGER,
  created_by TEXT);
CREATE TABLE manifest_parts (
  manifest_id TEXT, kind TEXT, name TEXT, version TEXT, content_hash TEXT);
CREATE TABLE pointers (
  env TEXT, agent TEXT, active TEXT, canary TEXT, fraction REAL,
  PRIMARY KEY (env, agent));
CREATE TABLE log (
  at INTEGER, event TEXT, subject TEXT, by TEXT, detail TEXT);
CREATE TABLE notices (
  at INTEGER, recipient TEXT, subject TEXT, message TEXT);
"""


class Catalog:
    """The shared catalog. Parts are immutable once released; only their
    status (released, deprecated, withdrawn) changes, and every change is logged."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(CATALOG_SCHEMA)

    # ---- parts -----------------------------------------------------------
    def register(self, kind, name, version, content, owner, at, permissions=None, data_scope=None,
                 compatibility=None, content_hash=None):
        """Register one part version. Refuses: a changed hash under an existing
        version, a name owned by another team, and a breaking change without a
        major version bump (when a compatibility function is supplied).
        content_hash overrides the default (skills use Chapter 10's package hash)."""
        h = content_hash or sha(content)
        row = self.db.execute("SELECT * FROM parts WHERE kind=? AND name=? AND version=?",
                              (kind, name, version)).fetchone()
        if row is not None:
            if row["content_hash"] == h:
                return h
            raise Refused(f"{kind} {name}@{version} is released with hash {row['content_hash'][:12]}; "
                          f"new content hashes to {h[:12]}. Released versions are immutable: "
                          f"publish a new version.")
        other = self.db.execute("SELECT DISTINCT owner FROM parts WHERE kind=? AND name=? AND owner<>?",
                                (kind, name, owner)).fetchone()
        if other is not None:
            raise Refused(f"{kind} name {name} is owned by {other['owner']}; {owner} cannot publish "
                          f"under it. Depend on the existing entry or choose a distinct name.")
        if compatibility is not None:
            prev = self.latest(kind, name)
            if prev is not None:
                verdict = compatibility(json.loads(prev["content"]), content)
                if verdict["breaking"] and semver(version)[0] <= semver(prev["version"])[0]:
                    raise Refused(f"{kind} {name}@{version} is a breaking change from "
                                  f"{prev['version']} ({'; '.join(c['change'] for c in verdict['changes'] if c['breaking'])}); "
                                  f"a breaking change needs a new major version.")
        self.db.execute("INSERT INTO parts VALUES (?,?,?,?,?,'released',?,?,NULL,NULL,NULL,?,?,?)",
                        (kind, name, version, h, owner, at, at, json.dumps(permissions or {}),
                         json.dumps(data_scope or {}), canon(content).decode()))
        self.log(at, "released", f"{kind}:{name}@{version}", owner, {"hash": h[:12]})
        return h

    def latest(self, kind, name, statuses=("released", "deprecated")):
        rows = self.db.execute(f"SELECT * FROM parts WHERE kind=? AND name=? AND status IN "
                               f"({','.join('?' * len(statuses))})", (kind, name, *statuses)).fetchall()
        return max(rows, key=lambda r: semver(r["version"])) if rows else None

    def part(self, kind, name, version):
        return self.db.execute("SELECT * FROM parts WHERE kind=? AND name=? AND version=?",
                               (kind, name, version)).fetchone()

    def depend(self, consumer, consumer_owner, kind, name, version):
        self.db.execute("INSERT INTO depends VALUES (?,?,?,?,?)", (consumer, consumer_owner, kind, name, version))

    def repin(self, consumer, kind, name, version, at, on=None):
        old = self.db.execute("SELECT version FROM depends WHERE consumer=? AND kind=? AND name=?",
                              (consumer, kind, name)).fetchone()
        self.db.execute("UPDATE depends SET version=? WHERE consumer=? AND kind=? AND name=?",
                        (version, consumer, kind, name))
        self.log(at, "repinned", f"{kind}:{name}", consumer, {"from": old["version"], "to": version, "on": on})

    def dependents(self, kind, name, version=None):
        q = "SELECT consumer, consumer_owner, version FROM depends WHERE kind=? AND name=?"
        args = [kind, name]
        if version:
            q += " AND version=?"
            args.append(version)
        return [dict(r) for r in self.db.execute(q + " ORDER BY consumer", args)]

    def add_evidence(self, subject, kind, result, at, by):
        self.db.execute("INSERT INTO evidence VALUES (?,?,?,?,?)", (subject, kind, json.dumps(result), at, by))

    def set_status(self, kind, name, version, status, at, by, reason, replacement=None, until=None):
        self.db.execute("UPDATE parts SET status=?, status_at=?, status_reason=?, replacement=?, "
                        "deprecated_until=? WHERE kind=? AND name=? AND version=?",
                        (status, at, reason, replacement, until, kind, name, version))
        self.log(at, status, f"{kind}:{name}@{version}", by, {"reason": reason, "replacement": replacement,
                                                              "until": until})

    def is_withdrawn(self, kind, name, version):
        r = self.part(kind, name, version)
        return r is not None and r["status"] == "withdrawn"

    def notify(self, at, recipient, subject, message):
        self.db.execute("INSERT INTO notices VALUES (?,?,?,?)", (at, recipient, subject, message))

    def log(self, at, event, subject, by, detail):
        self.db.execute("INSERT INTO log VALUES (?,?,?,?,?)", (at, event, subject, by, json.dumps(detail)))

    # ---- manifests -------------------------------------------------------
    def build_manifest(self, agent, label, refs, at, by):
        """Pin every (kind, name, version) by hash. Refuses withdrawn or unknown
        parts. The id is the hash of the sorted pin list, so the same parts
        always give the same manifest."""
        pins = []
        for kind, name, version in refs:
            r = self.part(kind, name, version)
            if r is None:
                raise Refused(f"{kind} {name}@{version} is not in the catalog")
            if r["status"] == "withdrawn":
                raise Refused(f"{kind} {name}@{version} is withdrawn: {r['status_reason']}")
            pins.append({"kind": kind, "name": name, "version": version, "hash": r["content_hash"]})
        pins.sort(key=lambda p: (p["kind"], p["name"]))
        mid = "m-" + sha({"agent": agent, "parts": pins})[:12]
        if self.db.execute("SELECT 1 FROM manifests WHERE manifest_id=?", (mid,)).fetchone() is None:
            self.db.execute("INSERT INTO manifests VALUES (?,?,?,?,?,?)",
                            (mid, agent, label, json.dumps(pins), at, by))
            for p in pins:
                self.db.execute("INSERT INTO manifest_parts VALUES (?,?,?,?,?)",
                                (mid, p["kind"], p["name"], p["version"], p["hash"]))
            self.log(at, "manifest_built", mid, by, {"agent": agent, "label": label})
        return mid

    def manifest(self, mid):
        r = self.db.execute("SELECT * FROM manifests WHERE manifest_id=?", (mid,)).fetchone()
        return dict(r, parts=json.loads(r["parts"])) if r else None

    def manifests_with(self, kind, name, version):
        return [r["manifest_id"] for r in self.db.execute(
            "SELECT DISTINCT manifest_id FROM manifest_parts WHERE kind=? AND name=? AND version=? "
            "ORDER BY manifest_id", (kind, name, version))]

    def withdrawn_parts(self, mid):
        return [dict(r) for r in self.db.execute(
            "SELECT mp.kind, mp.name, mp.version, p.status_reason FROM manifest_parts mp JOIN parts p "
            "ON p.kind=mp.kind AND p.name=mp.name AND p.version=mp.version "
            "WHERE mp.manifest_id=? AND p.status='withdrawn'", (mid,))]

    # ---- pointers --------------------------------------------------------
    def point(self, env, agent, at, by, cause, active=None, canary=None, fraction=None):
        cur = self.db.execute("SELECT * FROM pointers WHERE env=? AND agent=?", (env, agent)).fetchone()
        a = active if active is not None else (cur["active"] if cur else None)
        self.db.execute("INSERT OR REPLACE INTO pointers VALUES (?,?,?,?,?)", (env, agent, a, canary, fraction))
        self.log(at, "pointer_moved", f"{env}:{agent}", by,
                 {"active": a, "canary": canary, "fraction": fraction, "cause": cause})

    def pointer(self, env, agent):
        r = self.db.execute("SELECT * FROM pointers WHERE env=? AND agent=?", (env, agent)).fetchone()
        return dict(r) if r else None


def canary_draw(surface, manifest_id, run_id):
    """Seeded, reproducible canary assignment: the first eight bytes of
    SHA-256 over surface, candidate and run id, as a fraction of 2**64.
    The same construction as Rusty's canary_admits (see the chapter)."""
    material = "\n".join(["itl-canary-v1", surface, manifest_id, run_id]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2 ** 64


# ---------------------------------------------------------------------------
# Overlap between skill descriptions (catalog-wide discovery check)
# ---------------------------------------------------------------------------

def overlap(activator_index, name):
    """Jaccard similarity of positive description terms between `name` and
    every other entry in a LexicalActivator index (Chapter 10)."""
    mine = next(pos for n, pos, _ in activator_index if n == name)
    out = []
    for n, pos, _ in activator_index:
        if n != name:
            out.append({"with": n, "jaccard": round(len(mine & pos) / len(mine | pos), 3),
                        "shared": sorted(mine & pos)})
    return sorted(out, key=lambda x: -x["jaccard"])
