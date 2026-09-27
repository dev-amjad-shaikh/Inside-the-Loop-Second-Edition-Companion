"""Invoices helpers for the order pipeline."""

from decimal import Decimal


def invoices_step_1(record, rate=Decimal("1.40")):
    """Apply invoices rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["invoices_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("invoices_step_1")
    return updated


def invoices_step_2(record, rate=Decimal("1.41")):
    """Apply invoices rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["invoices_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("invoices_step_2")
    return updated


def invoices_step_3(record, rate=Decimal("1.42")):
    """Apply invoices rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["invoices_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("invoices_step_3")
    return updated


def invoices_step_4(record, rate=Decimal("1.43")):
    """Apply invoices rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["invoices_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("invoices_step_4")
    return updated


def invoices_step_5(record, rate=Decimal("1.44")):
    """Apply invoices rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["invoices_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("invoices_step_5")
    return updated


def invoices_step_6(record, rate=Decimal("1.45")):
    """Apply invoices rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["invoices_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("invoices_step_6")
    return updated
