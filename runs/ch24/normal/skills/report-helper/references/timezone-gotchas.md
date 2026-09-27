# Timezone gotchas for report filters

- A timezone east of UTC moves the gap to the other side: records between
  local midnight and UTC midnight appear on the next UTC day. Pick the test
  instant at 23:30 UTC on the day before the local report day.
- Daylight-saving switches change the offset inside a day. A local day can
  be 23 or 25 hours long. Probe the switch day explicitly.
- `datetime.date()` on an aware datetime returns the date in that datetime's
  own offset. Convert with `astimezone(ZoneInfo(name))` first.
- Store and pass timezone names (`America/Los_Angeles`), never fixed offsets
  such as `-07:00`, which are wrong for half the year.
