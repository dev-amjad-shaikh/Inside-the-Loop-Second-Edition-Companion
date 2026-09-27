import unittest
from datetime import date, datetime

from report import belongs_to_report

TIMEZONE = "America/Los_Angeles"


class ReportTests(unittest.TestCase):
    def test_order_at_midday(self):
        order_time = datetime.fromisoformat("2026-09-11T19:00:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))

    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))

    def test_order_near_midnight(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(
            order_time, date(2026, 9, 11), TIMEZONE
        ))


if __name__ == "__main__":
    unittest.main()
