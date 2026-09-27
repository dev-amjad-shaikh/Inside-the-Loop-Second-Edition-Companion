"""Customers helpers for the order pipeline."""

from decimal import Decimal


def customers_step_1(record, rate=Decimal("1.10")):
    """Apply customers rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["customers_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("customers_step_1")
    return updated


def customers_step_2(record, rate=Decimal("1.11")):
    """Apply customers rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["customers_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("customers_step_2")
    return updated


def customers_step_3(record, rate=Decimal("1.12")):
    """Apply customers rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["customers_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("customers_step_3")
    return updated


def customers_step_4(record, rate=Decimal("1.13")):
    """Apply customers rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["customers_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("customers_step_4")
    return updated


def customers_step_5(record, rate=Decimal("1.14")):
    """Apply customers rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["customers_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("customers_step_5")
    return updated


def customers_step_6(record, rate=Decimal("1.15")):
    """Apply customers rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["customers_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("customers_step_6")
    return updated
