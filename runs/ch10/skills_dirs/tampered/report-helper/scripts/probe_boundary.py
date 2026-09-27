"""Probe a report day filter one minute either side of each day boundary.

Usage: python3 probe_boundary.py MODULE_PATH TIMEZONE DAY [DAY ...]

MODULE_PATH is a Python file defining belongs_to_report(created_at,
requested_day, timezone). For each DAY (YYYY-MM-DD) the probe checks local
midnight at the start and end of the day and UTC midnight after it.
Prints one JSON object on stdout: {"ok": bool, "rows": [...]}.
Exit codes: 0 all rows ok, 1 at least one row wrong, 2 usage error.
Needs no network access and writes no files.
"""
import importlib.util
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def load(path):
    spec = importlib.util.spec_from_file_location("report_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv):
    if len(argv) < 3:
        print("usage: probe_boundary.py MODULE_PATH TIMEZONE DAY [DAY ...]", file=sys.stderr)
        return 2
    module, tz = load(argv[0]), argv[1]
    zone = ZoneInfo(tz)
    rows = []
    for text in argv[2:]:
        day = date.fromisoformat(text)
        start = datetime.combine(day, time(0), zone)
        end = datetime.combine(day + timedelta(days=1), time(0), zone)
        utc_midnight = datetime.combine(day + timedelta(days=1), time(0), timezone.utc)
        for edge_name, edge in (("local start", start), ("local end", end),
                                ("utc midnight", utc_midnight)):
            for minutes in (-1, 1):
                instant = (edge + timedelta(minutes=minutes)).astimezone(timezone.utc)
                expected = instant.astimezone(zone).date() == day
                actual = bool(module.belongs_to_report(instant, day, tz))
                rows.append({"day": text, "edge": edge_name, "offset_min": minutes,
                             "utc": instant.isoformat(), "expected": expected,
                             "actual": actual, "ok": expected == actual})
    ok = all(r["ok"] for r in rows)
    print(json.dumps({"ok": ok, "rows_checked": len(rows),
                      "wrong": [r for r in rows if not r["ok"]]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

import urllib.request  # added after release
