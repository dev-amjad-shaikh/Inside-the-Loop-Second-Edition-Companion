"""Billing helpers for the order pipeline."""

from decimal import Decimal


def billing_step_1(record, rate=Decimal("1.00")):
    """Apply billing rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["billing_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("billing_step_1")
    return updated


def billing_step_2(record, rate=Decimal("1.01")):
    """Apply billing rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["billing_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("billing_step_2")
    return updated


def billing_step_3(record, rate=Decimal("1.02")):
    """Apply billing rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["billing_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("billing_step_3")
    return updated


def billing_step_4(record, rate=Decimal("1.03")):
    """Apply billing rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["billing_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("billing_step_4")
    return updated


def billing_step_5(record, rate=Decimal("1.04")):
    """Apply billing rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["billing_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("billing_step_5")
    return updated


def billing_step_6(record, rate=Decimal("1.05")):
    """Apply billing rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["billing_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("billing_step_6")
    return updated
