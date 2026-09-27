from zoneinfo import ZoneInfo

from timeutil import calendar_day


def belongs_to_report(created_at, requested_day, timezone):
    """created_at is an aware UTC datetime; requested_day is a local date."""
    local_time = created_at.astimezone(ZoneInfo(timezone))
    return calendar_day(local_time) == requested_day
