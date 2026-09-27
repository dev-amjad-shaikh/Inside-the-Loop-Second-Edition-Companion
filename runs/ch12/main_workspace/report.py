from zoneinfo import ZoneInfo


def belongs_to_report(created_at, requested_day, timezone):
    """created_at is an aware UTC datetime; requested_day is a date in the
    report timezone."""
    local_time = created_at.astimezone(ZoneInfo(timezone))
    return local_time.date() == requested_day
