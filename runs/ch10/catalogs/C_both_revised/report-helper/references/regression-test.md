# Near-midnight regression test

Choose an instant that falls between UTC midnight and local midnight, so the
UTC date and the local date differ. For a report timezone west of UTC, take
00:30 UTC on the day after the local report day. The record belongs to the
earlier local day.

```
def test_order_near_midnight(self):
    order_time = datetime.fromisoformat("<next-day>T00:30:00+00:00")
    self.assertTrue(belongs_to_report(
        order_time, date(<report-day>), TIMEZONE
    ))
```

Add the method to the existing test class, above the `if __name__` block.
The expected value comes from the report timezone in settings.json, never
from the implementation under test.
