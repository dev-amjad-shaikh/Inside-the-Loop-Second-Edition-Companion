"""Timezone matrix for belongs_to_report, added by a colleague in the same pull request.

One generated test per (zone, UTC hour) on 2026-09-11, plus one test for callers
that expect the configured report timezone when they pass none.
"""
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from report import belongs_to_report

ZONES = [
    "Pacific/Honolulu", "America/Anchorage", "America/Los_Angeles", "America/Denver",
    "America/Chicago", "America/New_York", "America/Sao_Paulo", "Europe/London",
    "Europe/Berlin", "Asia/Kolkata", "Asia/Tokyo",
]
BASE = datetime(2026, 9, 11, 0, 30, tzinfo=timezone.utc)


class MatrixTests(unittest.TestCase):
    def test_configured_default(self):
        # Callers that pass no timezone should get the configured report timezone.
        with open("settings.json", encoding="utf-8") as f:
            configured = json.load(f)["report_timezone"]
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        local_day = order_time.astimezone(ZoneInfo(configured)).date()
        self.assertTrue(belongs_to_report(order_time, local_day))


def _make(zone, hour):
    def test(self):
        created = BASE + timedelta(hours=hour)
        local_day = created.astimezone(ZoneInfo(zone)).date()
        self.assertTrue(belongs_to_report(created, local_day, zone))
        self.assertFalse(belongs_to_report(created, local_day + timedelta(days=1), zone))
    return test


for _zone in ZONES:
    for _hour in range(24):
        _name = "test_%s_%02d" % (_zone.lower().replace("/", "_"), _hour)
        setattr(MatrixTests, _name, _make(_zone, _hour))


if __name__ == "__main__":
    unittest.main()
