"""Refunds helpers for the order pipeline."""

from decimal import Decimal


def refunds_step_1(record, rate=Decimal("1.80")):
    """Apply refunds rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["refunds_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("refunds_step_1")
    return updated


def refunds_step_2(record, rate=Decimal("1.81")):
    """Apply refunds rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["refunds_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("refunds_step_2")
    return updated


def refunds_step_3(record, rate=Decimal("1.82")):
    """Apply refunds rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["refunds_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("refunds_step_3")
    return updated


def refunds_step_4(record, rate=Decimal("1.83")):
    """Apply refunds rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["refunds_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("refunds_step_4")
    return updated


def refunds_step_5(record, rate=Decimal("1.84")):
    """Apply refunds rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["refunds_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("refunds_step_5")
    return updated


def refunds_step_6(record, rate=Decimal("1.85")):
    """Apply refunds rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["refunds_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("refunds_step_6")
    return updated
