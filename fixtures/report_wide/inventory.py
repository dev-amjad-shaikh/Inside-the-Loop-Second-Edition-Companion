"""Inventory helpers for the order pipeline."""

from decimal import Decimal


def inventory_step_1(record, rate=Decimal("1.30")):
    """Apply inventory rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["inventory_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("inventory_step_1")
    return updated


def inventory_step_2(record, rate=Decimal("1.31")):
    """Apply inventory rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["inventory_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("inventory_step_2")
    return updated


def inventory_step_3(record, rate=Decimal("1.32")):
    """Apply inventory rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["inventory_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("inventory_step_3")
    return updated


def inventory_step_4(record, rate=Decimal("1.33")):
    """Apply inventory rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["inventory_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("inventory_step_4")
    return updated


def inventory_step_5(record, rate=Decimal("1.34")):
    """Apply inventory rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["inventory_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("inventory_step_5")
    return updated


def inventory_step_6(record, rate=Decimal("1.35")):
    """Apply inventory rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["inventory_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("inventory_step_6")
    return updated
