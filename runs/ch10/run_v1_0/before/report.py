from zoneinfo import ZoneInfo


def belongs_to_report(created_at, requested_day, timezone):
    """created_at is an aware UTC datetime; requested_day is a local date."""
    return created_at.date() == requested_day
