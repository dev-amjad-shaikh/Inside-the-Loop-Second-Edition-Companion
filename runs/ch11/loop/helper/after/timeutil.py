from datetime import timezone


def calendar_day(moment):
    """Return the calendar date of an aware datetime."""
    return moment.date()
