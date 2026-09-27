"""Taxes helpers for the order pipeline."""

from decimal import Decimal


def taxes_step_1(record, rate=Decimal("1.100")):
    """Apply taxes rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["taxes_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("taxes_step_1")
    return updated


def taxes_step_2(record, rate=Decimal("1.101")):
    """Apply taxes rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["taxes_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("taxes_step_2")
    return updated


def taxes_step_3(record, rate=Decimal("1.102")):
    """Apply taxes rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["taxes_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("taxes_step_3")
    return updated


def taxes_step_4(record, rate=Decimal("1.103")):
    """Apply taxes rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["taxes_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("taxes_step_4")
    return updated


def taxes_step_5(record, rate=Decimal("1.104")):
    """Apply taxes rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["taxes_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("taxes_step_5")
    return updated


def taxes_step_6(record, rate=Decimal("1.105")):
    """Apply taxes rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["taxes_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("taxes_step_6")
    return updated
