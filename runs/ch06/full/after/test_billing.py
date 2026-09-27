"""The billing service calls belongs_to_report positionally, with an explicit timezone."""
import unittest
from datetime import date, datetime

from report import belongs_to_report


class BillingCallerTests(unittest.TestCase):
    def test_billing_positional_call(self):
        order_time = datetime.fromisoformat("2026-09-12T00:30:00+00:00")
        self.assertTrue(belongs_to_report(order_time, date(2026, 9, 11), "America/Los_Angeles"))


if __name__ == "__main__":
    unittest.main()
