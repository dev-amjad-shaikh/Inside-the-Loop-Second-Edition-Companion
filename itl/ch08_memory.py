"""Chapter 8 helpers: a governed memory store and a deliberately flawed one.

MemoryStore     sqlite3 store with typed records, an audit table and two derived
                artifacts (a per-subject summary and a term index) that record
                which memory records they were built from.
WritePolicy     decides whether a proposed record may be stored, and how.
retrieve        scope-, status- and expiry-aware retrieval with a trace of every
                record it excluded and why.
FlawedStore     the same schema, used the way many first versions are: every
                proposal becomes an active global record, corrections update
                the record table only, derived artifacts are rebuilt nightly.

Nothing here calls a model. Where a model's choice would matter, the chapter
script uses an explicitly labeled deterministic stand-in.
"""
import hashlib
import json
import os
import re
import sqlite3

SCHEMA_VERSION = 2
POLICY_VERSION = "wp-3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records (
  id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  subject TEXT NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  scope_type TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  source_event TEXT,
  source_kind TEXT,
  assertion TEXT NOT NULL,
  confidence REAL NOT NULL,
  sensitivity TEXT NOT NULL,
  created_at TEXT NOT NULL,
  valid_from TEXT NOT NULL,
  expires_at TEXT,
  status TEXT NOT NULL,
  supersedes TEXT,
  policy_version TEXT
);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  record_id TEXT,
  detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
  subject TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  built_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_index (
  term TEXT NOT NULL,
  record_id TEXT NOT NULL,
  snippet TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_sources (
  artifact TEXT NOT NULL,
  record_id TEXT NOT NULL
);
"""

SCOPE_RANK = {"session": 0, "meeting": 0, "project": 1, "user": 2, "global": 3}


def record_id(subject, key, value, source_event):
    h = hashlib.sha256(f"{subject}|{key}|{value}|{source_event}".encode()).hexdigest()
    return "mem-" + h[:8]


def terms_of(key, value):
    return sorted(set(t for t in re.split(r"[^a-z0-9]+", f"{key} {value}".lower()) if t))


class WritePolicy:
    """Version wp-3. Every rule is evaluated so a rejection lists all its reasons."""

    ALLOWED = {
        # kind: (widest scope allowed, default scope when the proposal gives none)
        "response_preference": ("user", "session"),
        "display_timezone": ("user", "user"),
    }
    OWNED_ELSEWHERE = {
        "report_timezone": "project configuration (settings.json)",
        "approval_rule": "Policy and authority",
        "order_history": "the order system, read through Retrieval",
    }
    TRUSTED_AUTHORS = {"user_message"}
    SENSITIVE_KINDS = {"health_note"}
    INFERRED_MAX_DAYS = 7

    def evaluate(self, c):
        reasons = []
        if not c.get("source_event") or not c.get("source_kind"):
            reasons.append("no provenance: source_event and source_kind are required")
        kind = c["kind"]
        if kind in self.OWNED_ELSEWHERE:
            reasons.append(f"kind {kind} is owned by {self.OWNED_ELSEWHERE[kind]}")
        elif kind in self.SENSITIVE_KINDS:
            reasons.append(f"kind {kind} is sensitive and never stored")
        elif kind not in self.ALLOWED:
            reasons.append(f"kind {kind} is not an allowed memory kind")
        if c.get("source_kind") == "retrieved_document":
            reasons.append("a retrieved document cannot author a durable record")
        if reasons:
            return {"decision": "reject", "reasons": reasons}
        widest, default = self.ALLOWED[kind]
        scope_type = c.get("scope_type") or default
        notes = []
        if not c.get("scope_type"):
            notes.append(f"no scope given; narrowest default {default} applied")
        if SCOPE_RANK[scope_type] > SCOPE_RANK[widest]:
            return {"decision": "reject", "reasons": [f"scope {scope_type} is wider than {widest}"]}
        if c["assertion"] in ("inferred", "unknown"):
            if c.get("source_kind") not in self.TRUSTED_AUTHORS | {"model_inference", "legacy_import"}:
                return {"decision": "reject", "reasons": ["inference from an untrusted source"]}
            notes.append("inferred or unverified: held until the user confirms")
            if c.get("source_kind") == "model_inference" and scope_type != "session":
                notes.append(f"inference narrowed from {scope_type} to session scope")
                scope_type = "session"
            return {"decision": "hold", "scope_type": scope_type, "notes": notes,
                    "max_days": self.INFERRED_MAX_DAYS}
        if c.get("source_kind") not in self.TRUSTED_AUTHORS:
            return {"decision": "reject", "reasons": [f"stated records need a user_message source"]}
        if scope_type in ("meeting", "session") and not c.get("expires_at"):
            return {"decision": "reject", "reasons": [f"{scope_type} scope requires expires_at"]}
        return {"decision": "activate", "scope_type": scope_type, "notes": notes}


class MemoryStore:
    def __init__(self, path, actor="memory-service"):
        if os.path.exists(path):
            os.remove(path)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.db.execute("INSERT INTO meta VALUES ('policy_version', ?)", (POLICY_VERSION,))
        self.db.commit()
        self.actor = actor
        self.policy = WritePolicy()

    # ---------------------------------------------------------------- audit
    def audit(self, at, action, rid, detail, actor=None):
        # Detail names fields and reasons, never record values.
        self.db.execute("INSERT INTO audit (at, actor, action, record_id, detail) VALUES (?,?,?,?,?)",
                        (at, actor or self.actor, action, rid, json.dumps(detail, sort_keys=True)))

    # ---------------------------------------------------------------- writes
    def _insert(self, c, rid, status, scope_type, expires_at):
        self.db.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, SCHEMA_VERSION, c["subject"], c["kind"], c["key"], c["value"], scope_type,
             c.get("scope_id") or (c["subject"] if scope_type == "user" else c.get("session", "")),
             c.get("source_event"), c.get("source_kind"), c["assertion"], c.get("confidence", 1.0),
             c.get("sensitivity", "low"), c["at"], c.get("valid_from", c["at"]), expires_at, status,
             c.get("supersedes"), POLICY_VERSION))

    def propose(self, c):
        """Propose, validate, and activate/hold/reject in one transaction."""
        rid = record_id(c["subject"], c["key"], c["value"], c.get("source_event"))
        with self.db:
            decision = self.policy.evaluate(c)
            self.audit(c["at"], "proposed", rid, {"kind": c["kind"], "source_kind": c.get("source_kind"),
                                                  "assertion": c["assertion"]}, actor=c.get("proposer"))
            if decision["decision"] == "reject":
                self.audit(c["at"], "rejected", rid, {"reasons": decision["reasons"]})
                return {"record_id": rid, "status": "rejected", **decision}
            expires = c.get("expires_at")
            if decision["decision"] == "hold":
                expires = expires or add_days(c["at"], decision["max_days"])
                status = "proposed"
            else:
                status = "active"
            if decision["scope_type"] != c.get("scope_type"):
                c = dict(c, scope_type=decision["scope_type"], scope_id=None)
            self._insert(c, rid, status, decision["scope_type"], expires)
            self.audit(c["at"], "validated", rid, {"policy": POLICY_VERSION, "notes": decision.get("notes", [])})
            if status == "active":
                self.audit(c["at"], "activated", rid, {"scope": decision["scope_type"]})
                self._index_record(rid)
                self._rebuild_summary(c["subject"], c["at"])
            else:
                self.audit(c["at"], "held", rid, {"expires_at": expires})
        return {"record_id": rid, "status": status, **decision, "expires_at": expires}

    def correct(self, old_id, c):
        """Supersede old_id with a new stated record; propagate to every derived artifact."""
        new = dict(c, supersedes=old_id)
        rid = record_id(new["subject"], new["key"], new["value"], new.get("source_event"))
        with self.db:
            decision = self.policy.evaluate(new)
            if decision["decision"] != "activate":
                raise ValueError(decision)
            self._insert(new, rid, "active", decision["scope_type"], new.get("expires_at"))
            self.db.execute("UPDATE records SET status='superseded' WHERE id=?", (old_id,))
            self.audit(new["at"], "superseded", old_id, {"by": rid, "source_event": new["source_event"]})
            self.audit(new["at"], "activated", rid, {"scope": decision["scope_type"], "supersedes": old_id})
            dependents = self.dependents(old_id)
            self._unindex_record(old_id)
            self._index_record(rid)
            self._rebuild_summary(new["subject"], new["at"])
            self.audit(new["at"], "derived_rebuilt", rid, {"artifacts": dependents})
        return {"record_id": rid, "superseded": old_id, "rebuilt": dependents}

    def correct_records_only(self, old_id, c):
        """The stale path: the record table is corrected, derived artifacts are not."""
        new = dict(c, supersedes=old_id)
        rid = record_id(new["subject"], new["key"], new["value"], new.get("source_event"))
        with self.db:
            self._insert(new, rid, "active", new["scope_type"], new.get("expires_at"))
            self.db.execute("UPDATE records SET status='superseded' WHERE id=?", (old_id,))
            self.audit(new["at"], "superseded", old_id, {"by": rid})
            self.audit(new["at"], "activated", rid, {"scope": new["scope_type"]})
        return {"record_id": rid, "superseded": old_id, "rebuilt": []}

    def erase(self, subject, key, at, reason, secure=True):
        """Delete every record for subject/key (current and history) and its derived traces."""
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM records WHERE subject=? AND key=? ORDER BY id", (subject, key))]
        dependents = sorted(set(a for i in ids for a in self.dependents(i)))
        # Some SQLite builds default secure_delete to ON; set it explicitly either way.
        self.db.execute(f"PRAGMA secure_delete={'ON' if secure else 'OFF'}")
        with self.db:
            for i in ids:
                self._unindex_record(i)
                self.db.execute("DELETE FROM records WHERE id=?", (i,))
                self.audit(at, "deleted", i, {"reason": reason})
            self._rebuild_summary(subject, at)
            self.audit(at, "derived_rebuilt", None, {"artifacts": dependents, "after": "erasure"})
        if secure:
            self.db.execute("VACUUM")
        return {"deleted": ids, "rebuilt": dependents}

    def sweep_expired(self, now):
        rows = list(self.db.execute(
            "SELECT id, subject FROM records WHERE status IN ('active','proposed') "
            "AND expires_at IS NOT NULL AND expires_at <= ?", (now,)))
        with self.db:
            for r in rows:
                self.db.execute("UPDATE records SET status='expired' WHERE id=?", (r["id"],))
                self._unindex_record(r["id"])
                self.audit(now, "expired", r["id"], {})
            for s in sorted(set(r["subject"] for r in rows)):
                self._rebuild_summary(s, now)
        return [r["id"] for r in rows]

    def confirm(self, rid, at, source_event):
        with self.db:
            self.db.execute("UPDATE records SET status='active', assertion='stated', confidence=1.0, "
                            "source_event=? WHERE id=? AND status='proposed'", (source_event, rid))
            self.audit(at, "confirmed", rid, {"source_event": source_event})
            self._index_record(rid)

    # ---------------------------------------------------------------- derived
    def _index_record(self, rid):
        r = self.db.execute("SELECT * FROM records WHERE id=?", (rid,)).fetchone()
        for t in terms_of(r["key"], r["value"]):
            self.db.execute("INSERT INTO search_index VALUES (?,?,?)", (t, rid, f"{r['key']}={r['value']}"))
        self.db.execute("INSERT INTO artifact_sources VALUES (?,?)", ("index", rid))

    def _unindex_record(self, rid):
        self.db.execute("DELETE FROM search_index WHERE record_id=?", (rid,))
        self.db.execute("DELETE FROM artifact_sources WHERE artifact='index' AND record_id=?", (rid,))

    def _rebuild_summary(self, subject, at):
        rows = list(self.db.execute(
            "SELECT * FROM records WHERE subject=? AND status='active' AND scope_type='user' ORDER BY key",
            (subject,)))
        art = f"summary:{subject}"
        self.db.execute("DELETE FROM artifact_sources WHERE artifact=?", (art,))
        if not rows:
            self.db.execute("DELETE FROM summaries WHERE subject=?", (subject,))
            return
        text = "; ".join(f"{r['key']}={r['value']}" for r in rows)
        self.db.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?)", (subject, text, at))
        for r in rows:
            self.db.execute("INSERT INTO artifact_sources VALUES (?,?)", (art, r["id"]))

    def dependents(self, rid):
        return sorted(set(a["artifact"] for a in self.db.execute(
            "SELECT artifact FROM artifact_sources WHERE record_id=?", (rid,))))

    # ---------------------------------------------------------------- reads
    def rows(self, sql, args=()):
        return [dict(r) for r in self.db.execute(sql, args)]

    def summary(self, subject):
        r = self.db.execute("SELECT * FROM summaries WHERE subject=?", (subject,)).fetchone()
        return dict(r) if r else None

    def index_lookup(self, term):
        return self.rows("SELECT term, record_id, snippet FROM search_index WHERE term=? ORDER BY record_id",
                         (term,))

    def close(self):
        self.db.close()


def retrieve(store, subject, ctx, now):
    """Records that apply to this subject in this context at this time, with a trace."""
    included, excluded = [], []
    for r in store.rows("SELECT * FROM records WHERE subject=? ORDER BY created_at, id", (subject,)):
        reason = None
        if r["status"] != "active":
            reason = f"status {r['status']}"
        elif r["valid_from"] > now:
            reason = "not yet valid"
        elif r["expires_at"] and r["expires_at"] <= now:
            reason = f"expired at {r['expires_at']}"
        elif r["scope_type"] == "user" and r["scope_id"] != ctx.get("user"):
            reason = "scope mismatch"
        elif r["scope_type"] in ("meeting", "session", "project") and \
                ctx.get(r["scope_type"]) != r["scope_id"]:
            reason = f"scope mismatch: record {r['scope_type']}:{r['scope_id']}, " + (
                f"context {r['scope_type']}:{ctx[r['scope_type']]}" if ctx.get(r["scope_type"])
                else f"context has no {r['scope_type']}")
        entry = {"id": r["id"], "key": r["key"], "value": r["value"],
                 "scope": f"{r['scope_type']}:{r['scope_id']}", "assertion": r["assertion"]}
        if reason:
            excluded.append(dict(entry, reason=reason))
        else:
            included.append(entry)
    return {"included": included, "excluded": excluded}


class FlawedStore(MemoryStore):
    """Accepts every proposal as an active, global, stated record with no expiry."""

    def propose(self, c):
        rid = record_id(c["subject"], c["key"], c["value"], c.get("source_event"))
        flat = dict(c, scope_type="global", scope_id="*", assertion="stated", confidence=1.0,
                    expires_at=None)
        with self.db:
            self._insert(flat, rid, "active", "global", None)
            self.audit(c["at"], "stored", rid, {})
            self._index_record(rid)
        return {"record_id": rid, "status": "active", "decision": "accept-all"}

    def nightly_rebuild(self, subject, at):
        with self.db:
            rows = list(self.db.execute(
                "SELECT * FROM records WHERE subject=? AND status='active' ORDER BY key", (subject,)))
            text = "; ".join(f"{r['key']}={r['value']}" for r in rows)
            self.db.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?)", (subject, text, at))


def retrieve_flawed(store, subject):
    rows = store.rows("SELECT * FROM records WHERE subject=? AND status='active' ORDER BY created_at, id",
                      (subject,))
    return {"included": [{"id": r["id"], "key": r["key"], "value": r["value"], "scope": "global:*",
                          "assertion": r["assertion"]} for r in rows], "excluded": []}


def add_days(iso, days):
    from datetime import datetime, timedelta
    t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ") + timedelta(days=days)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def raw_count(path, needle):
    with open(path, "rb") as f:
        return f.read().count(needle.encode())
