"""Notifications helpers for the order pipeline."""

from decimal import Decimal


def notifications_step_1(record, rate=Decimal("1.60")):
    """Apply notifications rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["notifications_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("notifications_step_1")
    return updated


def notifications_step_2(record, rate=Decimal("1.61")):
    """Apply notifications rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["notifications_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("notifications_step_2")
    return updated


def notifications_step_3(record, rate=Decimal("1.62")):
    """Apply notifications rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["notifications_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("notifications_step_3")
    return updated


def notifications_step_4(record, rate=Decimal("1.63")):
    """Apply notifications rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["notifications_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("notifications_step_4")
    return updated


def notifications_step_5(record, rate=Decimal("1.64")):
    """Apply notifications rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["notifications_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("notifications_step_5")
    return updated


def notifications_step_6(record, rate=Decimal("1.65")):
    """Apply notifications rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["notifications_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("notifications_step_6")
    return updated
