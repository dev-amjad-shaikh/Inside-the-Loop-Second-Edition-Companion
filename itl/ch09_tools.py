"""Chapter 9 helpers: an export service in sqlite, two toolsets over it, a
lexical selector used as a labeled stand-in for model tool selection, a
checked executor and a naive one, and a schema compatibility check.

Nothing here calls a model. Tool choice in the experiments comes from
`lexical_select`, which scores a task phrasing against tool names and
descriptions by keyword overlap. It is a stand-in: it shows how contract text
separates or confuses operations for any selector that relies on that text,
and it cannot show how a trained model weighs context, examples or intent.
"""
import copy
import datetime as dt
import hashlib
import json
import re
import sqlite3

SERVICE_TODAY = dt.date(2026, 9, 12)
RETENTION_DAYS = 90

# ---------------------------------------------------------------------------
# The export service (the external system the tools act on)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE tenants(tenant TEXT PRIMARY KEY, name TEXT);
CREATE TABLE reports(
    tenant TEXT, report TEXT, title TEXT, topic TEXT, version INTEGER,
    source_revision TEXT, restricted INTEGER, PRIMARY KEY(tenant, report));
CREATE TABLE exports(
    export_id TEXT PRIMARY KEY, tenant TEXT, report TEXT, report_version INTEGER,
    day TEXT, format TEXT, destination TEXT, status TEXT, rows INTEGER,
    requested_by TEXT, call_id TEXT);
CREATE TABLE export_log(export_id TEXT, line_no INTEGER, text TEXT);
"""

REPORTS = [
    # tenant, report, title, topic, version, source revision, restricted
    ("acme", "daily_orders", "Daily orders", "orders placed per local calendar day", 5,
     "339d3a9f2656", 0),
    ("acme", "weekly_revenue", "Weekly revenue", "revenue by week and channel", 3,
     "5b1e07c2d9aa", 0),
    ("acme", "refunds_by_region", "Refunds by region", "refund counts and amounts per region", 2,
     "a0c4e6f19b37", 0),
    ("acme", "payroll_summary", "Payroll summary", "salary totals per department", 7,
     "e93d11a0c5f2", 1),
    ("globex", "daily_orders", "Daily orders", "orders placed per local calendar day", 2,
     "0d7f3b8a6e21", 0),
]

# Destinations and the formats each one accepts.
DESTINATIONS = {"finance_sftp": ["csv"], "analytics_bucket": ["csv", "parquet"]}


def failed_upload_log(export_id, day, rows=1184, batch=2):
    """A long, deterministic job log whose only failure is at the very end."""
    lines = [f"2026-09-11T06:00:00Z export {export_id} started: daily_orders day={day} "
             f"format=csv destination=finance_sftp"]
    t = 6 * 3600
    for start in range(1, rows + 1, batch):
        end = min(start + batch - 1, rows)
        t += 1
        stamp = f"2026-09-11T{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}Z"
        lines.append(f"{stamp} wrote rows {start}-{end} to staging file (ok)")
    lines += [
        f"2026-09-11T06:09:55Z staging complete: {rows} rows, checksum sha256:4be1c09e77d2",
        "2026-09-11T06:09:56Z uploading to finance_sftp:/inbound/acme/daily_orders_"
        f"{day}.csv",
        "2026-09-11T06:10:26Z ERROR upload failed: permission denied (SFTP status 3) "
        "on /inbound/acme/; file not delivered",
        f"2026-09-11T06:10:26Z export {export_id} status=failed",
    ]
    return lines


def open_service(path, daily_orders=(5, "339d3a9f2656")):
    """Create the service database. `daily_orders` sets acme's report version
    and source revision: (4, "78721bd28acc") is the report before the
    timezone fix from Chapter 3, (5, "339d3a9f2656") after it."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO tenants VALUES(?,?)", [("acme", "Acme Retail"),
                                                        ("globex", "Globex Foods")])
    con.executemany("INSERT INTO reports VALUES(?,?,?,?,?,?,?)", REPORTS)
    con.execute("UPDATE reports SET version=?, source_revision=? WHERE tenant='acme' AND "
                "report='daily_orders'", daily_orders)
    # Two earlier exports: one delivered, one failed at upload.
    con.execute("INSERT INTO exports VALUES('exp_0411','acme','daily_orders',4,'2026-09-09',"
                "'csv','finance_sftp','delivered',1102,'maria@acme','seed')")
    con.execute("INSERT INTO exports VALUES('exp_0412','acme','daily_orders',4,'2026-09-10',"
                "'csv','finance_sftp','failed',1184,'maria@acme','seed')")
    for i, line in enumerate(failed_upload_log("exp_0412", "2026-09-10"), 1):
        con.execute("INSERT INTO export_log VALUES('exp_0412',?,?)", (i, line))
    con.commit()
    return con


def next_export_id(con):
    n = con.execute("SELECT COUNT(*) FROM exports").fetchone()[0]
    return f"exp_{411 + n:04d}"


def uuidish(*parts):
    """Deterministic UUID-shaped identifiers for the flawed toolset's results."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def export_rows(con):
    cur = con.execute("SELECT export_id, tenant, report, report_version, day, format, "
                      "destination, status, requested_by, call_id FROM exports ORDER BY export_id")
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def publish_report_version(con, tenant, report, version, source_revision):
    """An event outside the agent: the reporting team ships a new version."""
    con.execute("UPDATE reports SET version=?, source_revision=? WHERE tenant=? AND report=?",
                (version, source_revision, tenant, report))
    con.commit()


def insert_export(con, tenant, report, version, day, fmt, dest, user, call_id, status="delivered"):
    eid = next_export_id(con)
    rows = 1204 if report == "daily_orders" else 96
    con.execute("INSERT INTO exports VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (eid, tenant, report, version, day, fmt, dest, status, rows, user, call_id))
    con.commit()
    return eid


# ---------------------------------------------------------------------------
# Trusted context: who is acting. Never taken from model arguments.
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, tenant, user, grants):
        self.tenant = tenant
        self.user = user
        self.grants = set(grants)

    def as_dict(self):
        return {"tenant": self.tenant, "user": self.user, "grants": sorted(self.grants)}


# ---------------------------------------------------------------------------
# Lexical selector: the stand-in for model tool selection
# ---------------------------------------------------------------------------

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a all an and are as at be by can do does each every for from get given i in is it its "
    "of on one or other that the this to use used was what when where which who why with "
    "yet you your".split())


def tokenize(text):
    """Lowercase alphanumeric words, minus common English words, with a
    trailing plural s removed (orders -> order, exports -> export; not status)."""
    out = set()
    for w in TOKEN.findall(text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.add(w)
    return out


def lexical_select(task, definitions):
    """Score each tool by how many task keywords occur in its name or
    description, the rule of pydantic-ai's keywords_search_fn. Ties keep
    registration order (a stable sort), which is how an arbitrary order
    becomes a decision."""
    terms = tokenize(task)
    cands = []
    for order, d in enumerate(definitions):
        tool_terms = tokenize(d["name"] + " " + d["description"])
        matched = sorted(terms & tool_terms)
        cands.append({"tool": d["name"], "score": len(matched), "matched": matched,
                      "registration_order": order})
    cands.sort(key=lambda c: -c["score"])
    top = cands[0]
    tied = [c["tool"] for c in cands if c["score"] == top["score"]]
    runner = next((c for c in cands[1:]), None)
    return {
        "task": task,
        "selected": top["tool"],
        "margin": top["score"] - (runner["score"] if runner else 0),
        "tied_at_top": tied if len(tied) > 1 else [],
        "candidates": cands,
    }


# ---------------------------------------------------------------------------
# A small JSON Schema validator returning every violation with a path
# ---------------------------------------------------------------------------

def _type_ok(kind, v):
    return {"string": isinstance(v, str), "integer": isinstance(v, int) and not isinstance(v, bool),
            "boolean": isinstance(v, bool), "object": isinstance(v, dict),
            "array": isinstance(v, list)}.get(kind, True)


def validate(schema, args):
    out = []
    if not isinstance(args, dict):
        return [{"path": "", "rule": "type", "message": "arguments must be a JSON object"}]
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            out.append({"path": name, "rule": "required", "message": f"{name} is required"})
    if schema.get("additionalProperties") is False:
        for name in sorted(set(args) - set(props)):
            hint = schema.get("x-unexpected-hints", {}).get(name)
            msg = f"unexpected argument {name}"
            if hint:
                msg += f": {hint}"
            out.append({"path": name, "rule": "additionalProperties", "message": msg})
    for name, value in args.items():
        spec = props.get(name)
        if spec is None:
            continue
        kind = spec.get("type")
        if kind and not _type_ok(kind, value):
            out.append({"path": name, "rule": "type",
                        "message": f"{name} must be {kind}, got {type(value).__name__}"})
            continue
        if "enum" in spec and value not in spec["enum"]:
            out.append({"path": name, "rule": "enum",
                        "message": f"{name} must be one of {', '.join(spec['enum'])}; "
                                   f"got {value!r}"})
        if "pattern" in spec and isinstance(value, str) and not re.fullmatch(spec["pattern"], value):
            out.append({"path": name, "rule": "pattern",
                        "message": f"{name} must match {spec.get('x-format-name', spec['pattern'])}; "
                                   f"got {value!r}"})
        if "minimum" in spec and isinstance(value, int) and value < spec["minimum"]:
            out.append({"path": name, "rule": "minimum",
                        "message": f"{name} must be >= {spec['minimum']}; got {value}"})
        if "maximum" in spec and isinstance(value, int) and value > spec["maximum"]:
            out.append({"path": name, "rule": "maximum",
                        "message": f"{name} must be <= {spec['maximum']}; got {value}"
                                   + (f". {spec['x-range-hint']}" if "x-range-hint" in spec else "")})
    return out


# ---------------------------------------------------------------------------
# The flawed toolset
# ---------------------------------------------------------------------------

def flawed_definitions():
    s = lambda *names: {"type": "object",
                        "properties": {n: {"type": "string"} for n in names},
                        "required": list(names)}
    return [
        {"name": "process_data", "description":
            "Process data for reports. Use this to handle report data, exports, logs and "
            "other data operations.", "parameters": s("data", "action")},
        {"name": "export_report", "description": "Export a report with the given options.",
         "parameters": s("report", "options")},
        {"name": "export_data", "description": "Export data in a format.",
         "parameters": s("id", "fmt")},
        {"name": "create_export", "description": "Create an export.",
         "parameters": s("report_id", "tenant", "day", "format", "destination")},
        {"name": "get_export", "description": "Get export info. Pass all to get every export.",
         "parameters": s("id")},
        {"name": "get_logs", "description": "Get the logs.", "parameters": s("id")},
        {"name": "get_data", "description": "Get data.", "parameters": s("query")},
    ]


class FlawedTools:
    """Implementations that mirror common faults: free-string options,
    identity from arguments, opaque failures, UUID-heavy and unbounded results."""

    def __init__(self, con, session):
        self.con = con
        self.session = session

    def process_data(self, data, action, call_id):
        if "export" in action.lower():
            eid = insert_export(self.con, self.session.tenant, data.strip().lower().replace(" ", "_"),
                                self._version(self.session.tenant, data), None, "json",
                                "shared_drive", self.session.user, call_id)
            return {"status": "ok", "result": uuidish("job", eid)}
        return {"status": "ok", "result": "processed"}

    def export_report(self, report, options, call_id):
        # Options are "key=value" pairs; anything else is ignored silently.
        opts = dict(p.split("=", 1) for p in options.replace(";", ",").split(",") if "=" in p)
        rep = report.strip().lower().replace(" ", "_")
        eid = insert_export(self.con, self.session.tenant, rep, self._version(self.session.tenant, rep),
                            opts.get("date"), opts.get("format", "json"),
                            opts.get("dest", "shared_drive"), self.session.user, call_id)
        return {"status": "ok", "id": uuidish("export", eid)}

    def export_data(self, id, fmt, call_id):
        eid = insert_export(self.con, self.session.tenant, id, 0, None, fmt, "shared_drive",
                            self.session.user, call_id)
        return {"status": "ok", "id": uuidish("export", eid)}

    def create_export(self, report_id, tenant, day, format, destination, call_id):
        dt.date.fromisoformat(day)  # raises on "Sept 11"; the runtime hides the message
        version = self._version(tenant, report_id)
        eid = insert_export(self.con, tenant, report_id, version, day, format, destination,
                            self.session.user, call_id)
        return {"status": "ok", "id": uuidish("export", eid)}

    def get_export(self, id, call_id):
        rows = export_rows(self.con)
        if id != "all":
            rows = [r for r in rows if uuidish("export", r["export_id"]) == id or r["export_id"] == id]
        return [{
            "export_uuid": uuidish("export", r["export_id"]),
            "tenant_uuid": uuidish("tenant", r["tenant"]),
            "report_uuid": uuidish("report", r["tenant"], r["report"]),
            "job_uuid": uuidish("job", r["export_id"]),
            "destination_uuid": uuidish("dest", r["destination"]),
            "requested_by_uuid": uuidish("user", r["requested_by"]),
            "state_code": {"delivered": 3, "failed": 4}.get(r["status"], 1),
            "report_version": r["report_version"], "day": r["day"], "fmt": r["format"],
        } for r in rows]

    def get_logs(self, id, call_id):
        lines = [t for (t,) in self.con.execute(
            "SELECT text FROM export_log WHERE export_id=? ORDER BY line_no", (id,))]
        return {"status": "ok", "log": "\n".join(lines)}

    def get_data(self, query, call_id):
        return {"status": "ok", "rows": [list(r) for r in self.con.execute(
            "SELECT * FROM reports")]}

    def _version(self, tenant, report):
        row = self.con.execute("SELECT version FROM reports WHERE tenant=? AND report=?",
                               (tenant, report)).fetchone()
        return row[0] if row else 0


OPAQUE = "An error occurred while running the tool. Please try again."


class NaiveExecutor:
    """Type-only checks, identity wherever the tool puts it, every exception
    mapped to one fixed message, and head-only truncation with no marker."""

    def __init__(self, tools, definitions, observation_limit=4000):
        self.tools = tools
        self.defs = {d["name"]: d for d in definitions}
        self.limit = observation_limit

    def execute(self, call):
        d = self.defs.get(call["name"])
        if d is None:
            return {"status": "error", "message": OPAQUE}
        bad = [v for v in validate(d["parameters"], call["arguments"]) if v["rule"] in ("type", "required")]
        if bad:
            return {"status": "error", "message": OPAQUE}
        try:
            result = getattr(self.tools, call["name"])(call_id=call["id"], **call["arguments"])
        except Exception:
            return {"status": "error", "message": OPAQUE}
        text = json.dumps(result)
        return {"status": "ok", "content": text[: self.limit]}


# ---------------------------------------------------------------------------
# The revised toolset
# ---------------------------------------------------------------------------

DAY = {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}", "x-format-name": "YYYY-MM-DD",
       "description": "Calendar day of the report in the report's own timezone, YYYY-MM-DD."}


def revised_definitions(version=1):
    create_props = {
        "report": {"type": "string", "description": "Report name from reports_search, e.g. daily_orders."},
        "day": DAY,
        "format": {"type": "string", "enum": ["csv", "parquet"]},
        "destination": {"type": "string", "enum": ["finance_sftp", "analytics_bucket"],
                        "description": "finance_sftp accepts csv only."},
        "expected_report_version": {"type": "integer", "minimum": 1, "description":
            "The report version the user saw. The call fails if the report has changed since."},
    }
    required = ["report", "day", "format", "destination", "expected_report_version"]
    if version == 2:
        create_props["report_date"] = create_props.pop("day")
        required = ["report", "report_date", "format", "destination", "expected_report_version"]
    return [
        {"name": "reports_search", "version": 1, "effect": "read_only", "owner": "reporting-platform",
         "description": "Find which report covers a topic, such as orders, refunds or revenue, in "
                        "the report catalog of the signed-in tenant. Returns report names, titles "
                        "and current version. Read-only. Use it to find a report name before "
                        "exporting.",
         "parameters": {"type": "object", "additionalProperties": False, "required": ["query"],
                        "properties": {"query": {"type": "string"},
                                       "limit": {"type": "integer", "minimum": 1, "maximum": 20}}}},
        {"name": "exports_create", "version": version, "effect": "creates_external_record",
         "owner": "reporting-platform",
         "description": "Create one export file of one report for one calendar day and deliver "
                        "it to a destination. Changes external state: every successful call "
                        "creates a new export and a delivery; to re-run a failed export, call it "
                        "again, which creates a new export. Formats csv or parquet; "
                        "finance_sftp accepts csv only. Pass the report version the user saw as "
                        "expected_report_version. The tenant comes from the signed-in session.",
         "parameters": {"type": "object", "additionalProperties": False, "required": required,
                        "properties": create_props,
                        "x-unexpected-hints": {"tenant": "the tenant is taken from the signed-in "
                                               "session and cannot be chosen per call"}}},
        {"name": "exports_get_status", "version": 1, "effect": "read_only",
         "owner": "reporting-platform",
         "description": "Check whether one known export has finished: returns its status "
                        "(queued, running, delivered or failed) with report, day, format and "
                        "destination. Needs the export_id; use exports_list when you do not have "
                        "it. Read-only. response_format concise (default) or detailed.",
         "parameters": {"type": "object", "additionalProperties": False, "required": ["export_id"],
                        "properties": {"export_id": {"type": "string", "pattern": r"exp_\d{4}",
                                                     "x-format-name": "exp_NNNN"},
                                       "response_format": {"type": "string",
                                                           "enum": ["concise", "detailed"]}}}},
        {"name": "exports_list", "version": 1, "effect": "read_only", "owner": "reporting-platform",
         "description": "List recent exports of the signed-in tenant, newest first, optionally "
                        "only those with one status or since a day. Use it to find an export_id. "
                        "Read-only. Returns at most limit items and a next_cursor when more "
                        "exist.",
         "parameters": {"type": "object", "additionalProperties": False, "properties": {
             "status": {"type": "string", "enum": ["queued", "running", "delivered", "failed"]},
             "since": DAY, "limit": {"type": "integer", "minimum": 1, "maximum": 50},
             "cursor": {"type": "string"}}}},
        {"name": "exports_read_log", "version": 1, "effect": "read_only",
         "owner": "reporting-platform",
         "description": "Read lines from the job log of one export. Read-only. Defaults to the "
                        "last 40 lines, where failures are reported. The result gives total_lines, "
                        "the range returned and a cursor for the lines omitted.",
         "parameters": {"type": "object", "additionalProperties": False, "required": ["export_id"],
                        "properties": {"export_id": {"type": "string", "pattern": r"exp_\d{4}",
                                                     "x-format-name": "exp_NNNN"},
                                       "from_end": {"type": "boolean"},
                                       "lines": {"type": "integer", "minimum": 1, "maximum": 200,
                                                 "x-range-hint": "Page with cursor for more."},
                                       "cursor": {"type": "string"}}}},
        {"name": "exports_cancel", "version": 1, "effect": "changes_external_state",
         "owner": "reporting-platform",
         "description": "Cancel a queued or running export by export_id. Changes external state. "
                        "A delivered export cannot be cancelled.",
         "parameters": {"type": "object", "additionalProperties": False, "required": ["export_id"],
                        "properties": {"export_id": {"type": "string", "pattern": r"exp_\d{4}",
                                                     "x-format-name": "exp_NNNN"}}}},
    ]


def model_view(definitions):
    """What the model is sent: name, description, parameters. Governance
    metadata (owner, version, effect) stays in the runtime's catalog."""
    out = []
    for d in definitions:
        params = copy.deepcopy(d["parameters"])
        params.pop("x-unexpected-hints", None)
        for spec in params.get("properties", {}).values():
            for k in [k for k in spec if k.startswith("x-")]:
                spec.pop(k)
        out.append({"name": d["name"], "description": d["description"], "parameters": params})
    return out


class ToolFailure(Exception):
    def __init__(self, status, message, retryable, **extra):
        super().__init__(message)
        self.status, self.message, self.retryable, self.extra = status, message, retryable, extra


class RevisedTools:
    def __init__(self, con):
        self.con = con

    # semantic checks that a schema cannot express --------------------------
    def check_exports_create(self, args, session):
        day = args.get("day", args.get("report_date"))
        try:
            d = dt.date.fromisoformat(day)
        except ValueError:
            raise ToolFailure("input_error", f"day {day!r} is not a calendar date", True)
        if d >= SERVICE_TODAY:
            raise ToolFailure("input_error", f"day must be before {SERVICE_TODAY.isoformat()} "
                              f"(the service's current day); got {day}", True)
        if (SERVICE_TODAY - d).days > RETENTION_DAYS:
            raise ToolFailure("input_error", f"day must be within the last {RETENTION_DAYS} "
                              f"days; got {day}", True)
        allowed = DESTINATIONS[args["destination"]]
        if args["format"] not in allowed:
            raise ToolFailure("input_error", f"{args['destination']} accepts "
                              f"{' or '.join(allowed)} only; got format {args['format']}. Use "
                              f"format csv, or destination analytics_bucket for parquet.", True)

    def authorize_exports_create(self, args, session):
        row = self.con.execute("SELECT restricted FROM reports WHERE tenant=? AND report=?",
                               (session.tenant, args["report"])).fetchone()
        if row is None:
            raise ToolFailure("input_error", f"no report named {args['report']} in tenant "
                              f"{session.tenant}; use reports_search to find the name", True)
        if row[0] and "reports.restricted" not in session.grants:
            raise ToolFailure("not_authorized", f"report {args['report']} is restricted and "
                              f"{session.user} lacks the reports.restricted grant. Do not retry; "
                              f"tell the user who can grant access.", False)
        if "exports.create" not in session.grants:
            raise ToolFailure("not_authorized", f"{session.user} lacks exports.create", False)

    def precondition_exports_create(self, args, session):
        ver, rev = self.con.execute("SELECT version, source_revision FROM reports WHERE tenant=? "
                                    "AND report=?", (session.tenant, args["report"])).fetchone()
        if ver != args["expected_report_version"]:
            raise ToolFailure("precondition_failed", f"report {args['report']} is at version "
                              f"{ver} (source {rev}), not {args['expected_report_version']}. "
                              f"Confirm the version with the user or re-read it with "
                              f"reports_search before exporting.", False,
                              current_version=ver)

    # operations ---------------------------------------------------------------
    def reports_search(self, session, query, limit=5):
        words = tokenize(query)
        out = []
        for report, title, topic, version in self.con.execute(
                "SELECT report, title, topic, version FROM reports WHERE tenant=?", (session.tenant,)):
            score = len(words & tokenize(f"{report} {title} {topic}"))
            if score:
                out.append((score, {"report": report, "title": title, "current_version": version}))
        out.sort(key=lambda x: -x[0])
        return {"matches": [m for _, m in out[:limit]]}

    def exports_create(self, session, call_id, report, format, destination,
                       expected_report_version, day=None, report_date=None):
        day = day or report_date
        # The service exports its current version; the precondition stage is
        # what ties that version to the one the caller expected.
        version = self.con.execute("SELECT version FROM reports WHERE tenant=? AND report=?",
                                   (session.tenant, report)).fetchone()[0]
        eid = insert_export(self.con, session.tenant, report, version, day,
                            format, destination, session.user, call_id)
        return {"export_id": eid, "status": "delivered",
                "summary": f"{report} v{version} for {day} as {format} "
                           f"delivered to {destination}"}

    def exports_get_status(self, session, export_id, response_format="concise"):
        r = self._row(session, export_id)
        if response_format == "concise":
            return {"export_id": export_id, "status": r["status"],
                    "summary": f"{r['report']} v{r['report_version']} {r['day']} {r['format']} "
                               f"to {r['destination']}: {r['status']}"}
        n = self.con.execute("SELECT COUNT(*) FROM export_log WHERE export_id=?",
                             (export_id,)).fetchone()[0]
        return dict(r, log_lines=n)

    def exports_list(self, session, status=None, since=None, limit=10, cursor=None):
        rows = [r for r in export_rows(self.con) if r["tenant"] == session.tenant]
        rows = [r for r in rows if (status is None or r["status"] == status)
                and (since is None or (r["day"] or "") >= since)]
        rows.sort(key=lambda r: r["export_id"], reverse=True)
        start = int(cursor) if cursor else 0
        page = rows[start:start + limit]
        nxt = str(start + limit) if start + limit < len(rows) else None
        return {"exports": [{"export_id": r["export_id"], "report": r["report"], "day": r["day"],
                             "status": r["status"]} for r in page],
                "returned": len(page), "total": len(rows), "next_cursor": nxt}

    def exports_read_log(self, session, export_id, from_end=True, lines=40, cursor=None):
        self._row(session, export_id)
        all_lines = [t for (t,) in self.con.execute(
            "SELECT text FROM export_log WHERE export_id=? ORDER BY line_no", (export_id,))]
        total = len(all_lines)
        if cursor:
            start = int(cursor)
        else:
            start = max(0, total - lines) if from_end else 0
        chunk = all_lines[start:start + lines]
        first, last = start + 1, start + len(chunk)
        omitted_before = start
        return {"export_id": export_id, "total_lines": total, "returned_lines": f"{first}-{last}",
                "omitted": {"before": omitted_before, "after": total - last},
                "cursor_before": str(max(0, start - lines)) if start > 0 else None,
                "lines": chunk}

    def exports_cancel(self, session, export_id):
        r = self._row(session, export_id)
        if r["status"] in ("delivered", "failed"):
            raise ToolFailure("input_error", f"{export_id} is {r['status']} and cannot be "
                              f"cancelled", False)
        self.con.execute("UPDATE exports SET status='cancelled' WHERE export_id=?", (export_id,))
        self.con.commit()
        return {"export_id": export_id, "status": "cancelled"}

    def _row(self, session, export_id):
        for r in export_rows(self.con):
            if r["export_id"] == export_id and r["tenant"] == session.tenant:
                return r
        raise ToolFailure("input_error", f"no export {export_id} in tenant {session.tenant}; "
                          f"use exports_list to find it", True)

    # post-condition: read the effect back from the service --------------------
    def verify_exports_create(self, session, args, result):
        r = self._row(session, result["export_id"])
        want = {"tenant": session.tenant, "report": args["report"],
                "report_version": args["expected_report_version"],
                "day": args.get("day", args.get("report_date")), "format": args["format"],
                "destination": args["destination"], "status": "delivered"}
        diff = {k: {"wanted": v, "found": r[k]} for k, v in want.items() if r[k] != v}
        return {"verified": not diff, "checked": sorted(want), "differences": diff}


class CheckedExecutor:
    """resolve -> shape -> semantics -> authorize -> precondition -> execute
    -> verify -> bound. Each stage that refuses names its failure class."""

    ORDER = ["resolve", "shape", "semantics", "authorize", "precondition", "execute", "verify",
             "bound"]

    def __init__(self, tools, definitions, session, observation_limit=4000):
        self.tools = tools
        self.defs = {d["name"]: d for d in definitions}
        self.session = session
        self.limit = observation_limit
        self.records = []

    def execute(self, call):
        stages = []
        rec = {"call_id": call["id"], "tool": call["name"], "arguments": call["arguments"],
               "session": self.session.as_dict(), "stages": stages}
        self.records.append(rec)

        def refuse(stage, status, message, retryable, **extra):
            stages.append({"stage": stage, "result": status})
            obs = {"status": status, "stage": stage, "retryable": retryable, "message": message}
            obs.update(extra)
            rec["observation"] = obs
            return obs

        d = self.defs.get(call["name"])
        if d is None:
            return refuse("resolve", "unknown_tool", f"no tool {call['name']}; available: "
                          f"{', '.join(sorted(self.defs))}", True)
        stages.append({"stage": "resolve", "result": "ok", "version": d["version"],
                       "effect": d["effect"]})
        violations = validate(d["parameters"], call["arguments"])
        if violations:
            return refuse("shape", "input_error", "; ".join(v["message"] for v in violations),
                          True, violations=violations)
        stages.append({"stage": "shape", "result": "ok"})
        name = call["name"]
        current = "semantics"
        try:
            for stage, prefix in (("semantics", "check_"), ("authorize", "authorize_"),
                                  ("precondition", "precondition_")):
                current = stage
                fn = getattr(self.tools, prefix + name, None)
                if fn:
                    fn(call["arguments"], self.session)
                    stages.append({"stage": stage, "result": "ok"})
            current = "execute"
            kwargs = dict(call["arguments"])
            if name == "exports_create":
                kwargs["call_id"] = call["id"]
            result = getattr(self.tools, name)(self.session, **kwargs)
            stages.append({"stage": "execute", "result": "ok"})
        except ToolFailure as f:
            return refuse(current, f.status, f.message, f.retryable, **f.extra)
        obs = {"status": "ok", "effect": d["effect"], "result": result}
        verify = getattr(self.tools, "verify_" + name, None)
        if verify:
            v = verify(self.session, call["arguments"], result)
            stages.append({"stage": "verify", "result": "ok" if v["verified"] else "mismatch"})
            obs["verification"] = v
            if not v["verified"]:
                # The effect happened and differs from the request: report it
                # as its own outcome, never as plain success.
                obs["status"] = "effect_mismatch"
        text = json.dumps(obs)
        if len(text) > self.limit:
            obs = {"status": "ok", "truncated": True, "original_chars": len(text),
                   "content": bounded(text, self.limit)}
            stages.append({"stage": "bound", "result": "truncated"})
        rec["observation"] = obs
        return obs


def bounded(text, limit):
    """Keep head and tail, say what was removed (the Codex policy, simplified)."""
    head = limit // 2
    tail = limit - head
    removed = len(text) - head - tail
    return text[:head] + f"\n...[{removed} characters omitted from the middle]...\n" + text[-tail:]


# ---------------------------------------------------------------------------
# Reading a result: a deterministic stand-in for how a reader concludes
# ---------------------------------------------------------------------------

def read_outcome(text):
    """Conclude an export's outcome from the text a reader was given. Stand-in
    for a model reading an observation: it can only use what it sees."""
    low = text.lower()
    if "error" in low or "status=failed" in low or '"status": "failed"' in low:
        return "failed"
    if "omitted" in low or "truncated" in low:
        return "unknown: part of the log was omitted"
    if "wrote rows" in low or "delivered" in low:
        return "succeeded"
    return "unknown"


# ---------------------------------------------------------------------------
# Schema compatibility: classify a change before it ships
# ---------------------------------------------------------------------------

def schema_changes(old, new):
    """Compare two parameter schemas from the caller's side. Anything an
    existing valid call could fail, or be read differently, is breaking."""
    changes = []
    op, np_ = old.get("properties", {}), new.get("properties", {})
    oreq, nreq = set(old.get("required", [])), set(new.get("required", []))
    for name in sorted(set(op) - set(np_)):
        strict = new.get("additionalProperties") is False
        changes.append({"change": f"property {name} removed", "breaking": True,
                        "why": "existing calls that send it are rejected" if strict else
                        "existing calls that send it are accepted and the value is ignored"})
    for name in sorted(set(np_) - set(op)):
        req = name in nreq
        changes.append({"change": f"property {name} added" + (" as required" if req else ""),
                        "breaking": req,
                        "why": "existing calls lack it" if req else "optional; old calls still valid"})
    for name in sorted(set(op) & set(np_)):
        a, b = op[name], np_[name]
        if a.get("type") != b.get("type"):
            changes.append({"change": f"{name} type {a.get('type')} -> {b.get('type')}",
                            "breaking": True, "why": "old values fail the new type"})
        if "enum" in a and "enum" in b and set(b["enum"]) < set(a["enum"]):
            changes.append({"change": f"{name} enum narrowed", "breaking": True,
                            "why": f"values {sorted(set(a['enum']) - set(b['enum']))} now rejected"})
    for name in sorted((nreq - oreq) & set(op)):
        changes.append({"change": f"{name} became required", "breaking": True,
                        "why": "existing calls may omit it"})
    return {"breaking": any(c["breaking"] for c in changes), "changes": changes}
