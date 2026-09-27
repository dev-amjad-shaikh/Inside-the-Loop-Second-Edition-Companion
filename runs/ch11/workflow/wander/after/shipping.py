"""Shipping helpers for the order pipeline."""

from decimal import Decimal


def shipping_step_1(record, rate=Decimal("1.90")):
    """Apply shipping rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["shipping_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("shipping_step_1")
    return updated


def shipping_step_2(record, rate=Decimal("1.91")):
    """Apply shipping rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["shipping_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("shipping_step_2")
    return updated


def shipping_step_3(record, rate=Decimal("1.92")):
    """Apply shipping rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["shipping_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("shipping_step_3")
    return updated


def shipping_step_4(record, rate=Decimal("1.93")):
    """Apply shipping rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["shipping_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("shipping_step_4")
    return updated


def shipping_step_5(record, rate=Decimal("1.94")):
    """Apply shipping rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["shipping_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("shipping_step_5")
    return updated


def shipping_step_6(record, rate=Decimal("1.95")):
    """Apply shipping rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["shipping_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("shipping_step_6")
    return updated
