"""Chapter 20 helpers: fleet-level security operations.

SIMULATION. Everything here is a labeled stand-in. No real network, no real
sandbox, no real MCP server. The pieces model a small agent platform so the
chapter can show four operations as inspectable evidence:

    catalog + pinning      a tool/MCP-server catalog whose entries pin a
                           content hash; an update is admitted only when a
                           reviewer approves the new hash. A flawed
                           auto-updater trusts the server's own version string.
    incident               anomalous outbound attempts read from records, a
                           per-tool/tenant kill switch, affected-run
                           enumeration, credential revocation, incident report.
    data inventory         the stores a value lands in across a fleet, and a
                           deletion that visits each one with verification.
    boundary regression    a boundary suite run against two runtime versions,
                           catching a path-check regression before release.

The runtime pieces reuse the shape of Chapter 16 (principal from session,
deterministic rules, decision records) at fleet scale: many tenants, many
runs, a shared catalog. Standard library only; deterministic.
"""
import copy
import hashlib
import json
import os
import sqlite3
import time

RULESET = "ch20-rules-1"


def h12(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# The catalog: tool and MCP-server entries, pinned by content hash
# ---------------------------------------------------------------------------

class Catalog:
    """A shared catalog of tool servers. Each entry pins the exact manifest
    the platform admitted, by hash. Adding or changing an entry is a review
    event with a named approver, recorded in an append-only log."""

    def __init__(self):
        self.entries = {}      # name -> {version, manifest, manifest_hash, approver, at, effects}
        self.reviews = []      # append-only review log

    def _hash(self, manifest):
        return h12(canonical(manifest))

    def add(self, name, version, manifest, approver, at):
        mh = self._hash(manifest)
        self.entries[name] = {"name": name, "version": version, "manifest": copy.deepcopy(manifest),
                              "manifest_hash": mh, "approver": approver, "at": at,
                              "effects": sorted(manifest.get("effects", []))}
        self.reviews.append({"at": at, "event": "add", "name": name, "version": version,
                             "manifest_hash": mh, "approver": approver})
        return mh

    def resolve(self, name):
        return self.entries.get(name)

    # --- Governed update: a new manifest is admitted only after review of its hash.
    def propose_update(self, name, new_version, new_manifest, at):
        mh = self._hash(new_manifest)
        old = self.entries[name]
        diff = manifest_diff(old["manifest"], new_manifest)
        proposal = {"at": at, "event": "update_proposed", "name": name,
                    "from_version": old["version"], "to_version": new_version,
                    "from_hash": old["manifest_hash"], "to_hash": mh,
                    "diff": diff, "risk_flags": risk_flags(diff)}
        self.reviews.append(proposal)
        return proposal

    def review_update(self, proposal, new_manifest, decision, reviewer, at):
        entry = {"at": at, "event": "update_reviewed", "name": proposal["name"],
                 "to_version": proposal["to_version"], "to_hash": proposal["to_hash"],
                 "decision": decision, "reviewer": reviewer,
                 "reasons": proposal["risk_flags"]}
        self.reviews.append(entry)
        if decision == "approved":
            self.add(proposal["name"], proposal["to_version"], new_manifest, reviewer, at)
        return entry


class FlawedAutoUpdater:
    """The flawed operation: an auto-updater that trusts the server's own
    version string and installs whatever manifest the server now advertises,
    without re-review of the content hash. This is the pattern that lets a
    compromised or rug-pulled server change its behavior in place."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.log = []

    def poll_and_update(self, name, advertised_version, advertised_manifest, at):
        old = self.catalog.entries[name]
        installed = False
        if advertised_version != old["version"]:      # only check: version changed
            mh = self.catalog._hash(advertised_manifest)
            self.catalog.entries[name] = {
                "name": name, "version": advertised_version,
                "manifest": copy.deepcopy(advertised_manifest), "manifest_hash": mh,
                "approver": "auto-updater", "at": at,
                "effects": sorted(advertised_manifest.get("effects", []))}
            self.catalog.reviews.append({"at": at, "event": "auto_updated", "name": name,
                                         "to_version": advertised_version, "to_hash": mh,
                                         "approver": "auto-updater"})
            installed = True
        self.log.append({"at": at, "name": name, "advertised_version": advertised_version,
                         "installed": installed})
        return installed


def manifest_diff(old, new):
    diff = {}
    keys = set(old) | set(new)
    for k in sorted(keys):
        if old.get(k) != new.get(k):
            diff[k] = {"from": old.get(k), "to": new.get(k)}
    return diff


def risk_flags(diff):
    """Deterministic flags an update review would raise. These describe the
    kind of change, not a probabilistic judgment of intent."""
    flags = []
    if "effects" in diff:
        added = set(diff["effects"].get("to") or []) - set(diff["effects"].get("from") or [])
        if added:
            flags.append(f"new effect classes: {sorted(added)}")
    if "description" in diff:
        old_d = (diff["description"].get("from") or "").lower()
        new_d = (diff["description"].get("to") or "").lower()
        # A description that starts instructing the agent is a known injection vector.
        for phrase in ("ignore", "send", "forward", "exfiltrat", "@", "curl ", "http://", "https://"):
            if phrase in new_d and phrase not in old_d:
                flags.append(f"description now contains instruction-shaped text ({phrase!r})")
    if "endpoint" in diff:
        flags.append("network endpoint changed")
    return flags


# ---------------------------------------------------------------------------
# The fleet: tenants, agents, runs, and a records store
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, tenant TEXT, agent TEXT, agent_version TEXT,
  started_at INTEGER, tool_name TEXT, tool_hash TEXT, credential_id TEXT);
CREATE TABLE decisions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, tenant TEXT, tool TEXT,
  destination TEXT, decision TEXT, rule TEXT, data_left INTEGER, at INTEGER);
CREATE TABLE outbound (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, tenant TEXT,
  destination TEXT, bytes INTEGER, at INTEGER);
"""


class FleetRecords:
    """The shared records store: every run's identity, every policy decision,
    every delivered outbound message. This is what an incident is
    reconstructed from."""

    def __init__(self, path):
        if os.path.exists(path):
            os.remove(path)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def start_run(self, run):
        self.db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
            (run["run_id"], run["tenant"], run["agent"], run["agent_version"],
             run["started_at"], run["tool_name"], run["tool_hash"], run.get("credential_id")))
        self.db.commit()

    def record_decision(self, run_id, tenant, tool, destination, decision, rule, data_left, at):
        self.db.execute(
            "INSERT INTO decisions (run_id, tenant, tool, destination, decision, rule, data_left, at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, tenant, tool, destination, decision, rule, int(data_left), at))
        self.db.commit()

    def record_outbound(self, run_id, tenant, destination, nbytes, at):
        self.db.execute("INSERT INTO outbound (run_id, tenant, destination, bytes, at) VALUES (?,?,?,?,?)",
                        (run_id, tenant, destination, nbytes, at))
        self.db.commit()

    def rows(self, sql, args=()):
        return [dict(r) for r in self.db.execute(sql, args)]

    def close(self):
        self.db.close()


# ---------------------------------------------------------------------------
# Egress policy: an allowlist of destinations per tenant, deny by default
# ---------------------------------------------------------------------------

class EgressPolicy:
    def __init__(self, allow_by_tenant):
        self.allow = {t: set(v) for t, v in allow_by_tenant.items()}
        self.version = h12(canonical({t: sorted(v) for t, v in allow_by_tenant.items()}))

    def evaluate(self, tenant, destination):
        allowed = self.allow.get(tenant, set())
        domain = destination.split("@", 1)[-1] if "@" in destination else destination
        if destination in allowed or domain in allowed:
            return "allow", "egress.destination_allowlist"
        return "deny", "egress.destination_allowlist"


# ---------------------------------------------------------------------------
# Detection: anomalous outbound attempts from the records
# ---------------------------------------------------------------------------

def detect_anomalies(records, policy):
    """Read the records for two signals. First, denied outbound attempts in the
    decision log: runs whose model tried a destination the allowlist refused.
    Second, delivered outbound in the outbound log to a domain no tenant has
    registered: bytes that actually left to an unexpected place. Both are read
    from records, not from a live feed."""
    findings = []

    denied = records.rows(
        "SELECT run_id, tenant, tool, destination, at FROM decisions "
        "WHERE decision='deny' AND rule='egress.destination_allowlist' ORDER BY at")
    by_tool = {}
    for d in denied:
        by_tool.setdefault((d["tenant"], d["tool"]), []).append(d)
    for (tenant, tool), attempts in sorted(by_tool.items()):
        findings.append({"signal": "denied_outbound_attempt", "tenant": tenant, "tool": tool,
                         "attempts": len(attempts),
                         "distinct_destinations": sorted({a["destination"] for a in attempts}),
                         "runs": sorted({a["run_id"] for a in attempts})})

    registered = {dom for doms in policy.allow.values() for dom in doms}
    sent = records.rows("SELECT run_id, tenant, destination, bytes FROM outbound ORDER BY seq")
    unexpected = {}
    for o in sent:
        dom = o["destination"].split("@", 1)[-1]
        if dom not in registered:
            unexpected.setdefault(dom, {"bytes": 0, "runs": set()})
            unexpected[dom]["bytes"] += o["bytes"]
            unexpected[dom]["runs"].add(o["run_id"])
    for dom, agg in sorted(unexpected.items()):
        findings.append({"signal": "outbound_to_unregistered_domain", "domain": dom,
                         "bytes": agg["bytes"], "runs": sorted(agg["runs"])})
    return findings


# ---------------------------------------------------------------------------
# Response: the kill switch
# ---------------------------------------------------------------------------

class KillSwitch:
    """A per-tool, per-tenant, per-skill disable list checked at admission,
    outside the model. Flipping a switch stops future admissions immediately;
    it does not undo effects already delivered."""

    def __init__(self):
        self.disabled = set()     # ("tool", name) | ("tenant", name) | ("tool@tenant", "name@tenant")
        self.log = []

    def disable(self, kind, target, reason, by, at):
        self.disabled.add((kind, target))
        self.log.append({"at": at, "action": "disable", "kind": kind, "target": target,
                         "reason": reason, "by": by})

    def enable(self, kind, target, by, at):
        self.disabled.discard((kind, target))
        self.log.append({"at": at, "action": "enable", "kind": kind, "target": target, "by": by})

    def admits(self, tool, tenant):
        if ("tool", tool) in self.disabled:
            return False, ("tool", tool)
        if ("tenant", tenant) in self.disabled:
            return False, ("tenant", tenant)
        if ("tool@tenant", f"{tool}@{tenant}") in self.disabled:
            return False, ("tool@tenant", f"{tool}@{tenant}")
        return True, None


# ---------------------------------------------------------------------------
# Response: enumerate affected runs and revoke credentials
# ---------------------------------------------------------------------------

def affected_runs(records, tool_hash):
    """Every run that used the compromised tool build, by its pinned hash."""
    return records.rows(
        "SELECT run_id, tenant, agent, agent_version, credential_id, started_at "
        "FROM runs WHERE tool_hash=? ORDER BY started_at", (tool_hash,))


def data_left_in_runs(records, run_ids):
    """What actually left the environment in those runs, from the outbound log."""
    if not run_ids:
        return []
    marks = ",".join("?" for _ in run_ids)
    return records.rows(
        f"SELECT run_id, tenant, destination, bytes, at FROM outbound "
        f"WHERE run_id IN ({marks}) ORDER BY at", tuple(run_ids))


class CredentialRegistry:
    """Which credential each affected run held, and revocation. Revocation
    marks the credential; a revoked credential fails at its next use."""

    def __init__(self):
        self.status = {}       # credential_id -> "active" | "revoked"
        self.log = []

    def register(self, cred_id):
        self.status.setdefault(cred_id, "active")

    def revoke(self, cred_id, reason, by, at):
        self.status[cred_id] = "revoked"
        self.log.append({"at": at, "credential_id": cred_id, "action": "revoke",
                         "reason": reason, "by": by})

    def usable(self, cred_id):
        return self.status.get(cred_id) == "active"


def incident_report(records, catalog_entry, findings, affected, data_left, killed, revoked, timeline):
    """Assemble the incident report from records only. Every field is a query
    result, not a narrative."""
    return {
        "incident_id": "INC-" + h12(canonical({"tool": catalog_entry["name"],
                                               "hash": catalog_entry["manifest_hash"]})),
        "component": {"name": catalog_entry["name"], "version": catalog_entry["version"],
                      "manifest_hash": catalog_entry["manifest_hash"]},
        "detection": findings,
        "affected_runs": [r["run_id"] for r in affected],
        "affected_tenants": sorted({r["tenant"] for r in affected}),
        "data_left": data_left,
        "records_exfiltrated_bytes": sum(d["bytes"] for d in data_left),
        "containment": {"kill_switches": killed, "credentials_revoked": revoked},
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# Data inventory and deletion across stores
# ---------------------------------------------------------------------------

class Store:
    """A named store that can hold copies of a subject's values, report what it
    holds, delete on request, and verify the deletion by reading raw bytes."""

    def __init__(self, name, reachable_by_deletion=True, retention_days=None):
        self.name = name
        self.reachable = reachable_by_deletion
        self.retention_days = retention_days
        self.blob = bytearray()      # the raw store, so verification can read bytes

    def put(self, text):
        self.blob += text.encode("utf-8") + b"\n"

    def occurrences(self, needle):
        return bytes(self.blob).count(needle.encode("utf-8"))

    def delete(self, needle):
        if not self.reachable:
            return {"store": self.name, "reachable": False,
                    "retention_days": self.retention_days, "removed": 0,
                    "remaining": self.occurrences(needle)}
        lines = bytes(self.blob).split(b"\n")
        kept = [ln for ln in lines if needle.encode("utf-8") not in ln]
        removed = len(lines) - len(kept)
        self.blob = bytearray(b"\n".join(kept))
        return {"store": self.name, "reachable": True, "retention_days": self.retention_days,
                "removed": removed, "remaining": self.occurrences(needle)}


class DataInventory:
    """The set of stores a subject's data can land in across the fleet."""

    def __init__(self):
        self.stores = {}

    def add(self, store):
        self.stores[store.name] = store
        return store

    def inventory(self, needle):
        return [{"store": s.name, "occurrences": s.occurrences(needle),
                 "reachable_by_deletion": s.reachable, "retention_days": s.retention_days}
                for s in self.stores.values()]

    def delete_everywhere(self, needle):
        results = [self.stores[n].delete(needle) for n in self.stores]
        verified = all(r["remaining"] == 0 for r in results if r["reachable"])
        out_of_reach = [r["store"] for r in results if not r["reachable"]]
        return {"results": results, "verified_in_reachable_stores": verified,
                "out_of_reach": out_of_reach}


# ---------------------------------------------------------------------------
# Boundary regression across two runtime versions
# ---------------------------------------------------------------------------

def path_within_root_correct(root, path):
    """Resolve then compare: the check the runtime should make."""
    lexical = os.path.normpath(os.path.join(root, path))
    # os.path.realpath needs the fs; for the suite we simulate one resolution
    # step for a known symlink name, then compare canonical prefixes.
    resolved = _simulated_realpath(root, lexical)
    try:
        inside = os.path.commonpath([os.path.normpath(root), resolved]) == os.path.normpath(root)
    except ValueError:
        inside = False
    return inside


def path_within_root_regressed(root, path):
    """The regression: compare the lexical string prefix, before resolution.
    Admits a symlink escape and refuses a harmless dot-dot that stays inside."""
    lexical = os.path.normpath(os.path.join(root, path))
    return lexical.startswith(os.path.normpath(root) + os.sep) or lexical == os.path.normpath(root)


# A fixed symlink map for the simulation: 'shared' under acme points outside.
_SYMLINKS = {
    "/fleet/tenants/acme/workspace/shared": "/fleet/tenants/globex/workspace",
}


def _simulated_realpath(root, lexical):
    for link, target in _SYMLINKS.items():
        if lexical == link or lexical.startswith(link + os.sep):
            return os.path.normpath(target + lexical[len(link):])
    return os.path.normpath(lexical)


BOUNDARY_SUITE = [
    ("P01 own file", "/fleet/tenants/acme/workspace", "report.txt", "allow"),
    ("P02 dot-dot traversal", "/fleet/tenants/acme/workspace", "../../globex/workspace/customers.csv", "deny"),
    ("P03 dot-dot that stays inside", "/fleet/tenants/acme/workspace", "sub/../report.txt", "allow"),
    ("P04 symlink escape", "/fleet/tenants/acme/workspace", "shared/customers.csv", "deny"),
    ("P05 absolute path", "/fleet/tenants/acme/workspace", "/etc/passwd", "deny"),
]


def run_boundary_suite(check_fn):
    rows = []
    for cid, root, path, expected in BOUNDARY_SUITE:
        if os.path.isabs(path):
            got = "deny"       # both versions reject absolute paths before the prefix test
        else:
            got = "allow" if check_fn(root, path) else "deny"
        rows.append({"case": cid, "path": path, "expected": expected, "got": got,
                     "pass": got == expected})
    return rows
