"""Warehouse helpers for the order pipeline."""

from decimal import Decimal


def warehouse_step_1(record, rate=Decimal("1.110")):
    """Apply warehouse rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["warehouse_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("warehouse_step_1")
    return updated


def warehouse_step_2(record, rate=Decimal("1.111")):
    """Apply warehouse rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["warehouse_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("warehouse_step_2")
    return updated


def warehouse_step_3(record, rate=Decimal("1.112")):
    """Apply warehouse rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["warehouse_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("warehouse_step_3")
    return updated


def warehouse_step_4(record, rate=Decimal("1.113")):
    """Apply warehouse rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["warehouse_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("warehouse_step_4")
    return updated


def warehouse_step_5(record, rate=Decimal("1.114")):
    """Apply warehouse rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["warehouse_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("warehouse_step_5")
    return updated


def warehouse_step_6(record, rate=Decimal("1.115")):
    """Apply warehouse rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["warehouse_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("warehouse_step_6")
    return updated
