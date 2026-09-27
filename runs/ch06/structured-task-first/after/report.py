import json
from zoneinfo import ZoneInfo


def belongs_to_report(created_at, requested_day, timezone):
    """created_at is an aware UTC datetime; requested_day is a local date."""
    local_time = created_at.astimezone(ZoneInfo(timezone))
    return local_time.date() == requested_day


def belongs_to_configured_report(created_at, requested_day, settings_path="settings.json"):
    """Like belongs_to_report, using the configured report timezone."""
    with open(settings_path, encoding="utf-8") as f:
        timezone = json.load(f)["report_timezone"]
    return belongs_to_report(created_at, requested_day, timezone)
