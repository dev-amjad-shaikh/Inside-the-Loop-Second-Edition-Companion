import json
from zoneinfo import ZoneInfo


def belongs_to_report(created_at, requested_day, settings_path="settings.json", timezone=None):
    """created_at is an aware UTC datetime; requested_day is a local date."""
    if timezone is None:
        with open(settings_path, encoding="utf-8") as f:
            timezone = json.load(f)["report_timezone"]
    local_time = created_at.astimezone(ZoneInfo(timezone))
    return local_time.date() == requested_day
