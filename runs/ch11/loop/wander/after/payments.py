"""Payments helpers for the order pipeline."""

from decimal import Decimal


def payments_step_1(record, rate=Decimal("1.70")):
    """Apply payments rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["payments_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("payments_step_1")
    return updated


def payments_step_2(record, rate=Decimal("1.71")):
    """Apply payments rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["payments_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("payments_step_2")
    return updated


def payments_step_3(record, rate=Decimal("1.72")):
    """Apply payments rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["payments_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("payments_step_3")
    return updated


def payments_step_4(record, rate=Decimal("1.73")):
    """Apply payments rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["payments_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("payments_step_4")
    return updated


def payments_step_5(record, rate=Decimal("1.74")):
    """Apply payments rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["payments_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("payments_step_5")
    return updated


def payments_step_6(record, rate=Decimal("1.75")):
    """Apply payments rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["payments_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("payments_step_6")
    return updated
