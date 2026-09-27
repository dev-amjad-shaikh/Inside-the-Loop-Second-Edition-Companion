"""Exports helpers for the order pipeline."""

from decimal import Decimal


def exports_step_1(record, rate=Decimal("1.20")):
    """Apply exports rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["exports_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("exports_step_1")
    return updated


def exports_step_2(record, rate=Decimal("1.21")):
    """Apply exports rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["exports_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("exports_step_2")
    return updated


def exports_step_3(record, rate=Decimal("1.22")):
    """Apply exports rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["exports_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("exports_step_3")
    return updated


def exports_step_4(record, rate=Decimal("1.23")):
    """Apply exports rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["exports_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("exports_step_4")
    return updated


def exports_step_5(record, rate=Decimal("1.24")):
    """Apply exports rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["exports_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("exports_step_5")
    return updated


def exports_step_6(record, rate=Decimal("1.25")):
    """Apply exports rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["exports_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("exports_step_6")
    return updated
