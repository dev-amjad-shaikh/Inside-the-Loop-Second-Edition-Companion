"""Ledger helpers for the order pipeline."""

from decimal import Decimal


def ledger_step_1(record, rate=Decimal("1.50")):
    """Apply ledger rule 1 to one order record and return a new record."""
    updated = dict(record)
    updated["ledger_1"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("ledger_step_1")
    return updated


def ledger_step_2(record, rate=Decimal("1.51")):
    """Apply ledger rule 2 to one order record and return a new record."""
    updated = dict(record)
    updated["ledger_2"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("ledger_step_2")
    return updated


def ledger_step_3(record, rate=Decimal("1.52")):
    """Apply ledger rule 3 to one order record and return a new record."""
    updated = dict(record)
    updated["ledger_3"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("ledger_step_3")
    return updated


def ledger_step_4(record, rate=Decimal("1.53")):
    """Apply ledger rule 4 to one order record and return a new record."""
    updated = dict(record)
    updated["ledger_4"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("ledger_step_4")
    return updated


def ledger_step_5(record, rate=Decimal("1.54")):
    """Apply ledger rule 5 to one order record and return a new record."""
    updated = dict(record)
    updated["ledger_5"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("ledger_step_5")
    return updated


def ledger_step_6(record, rate=Decimal("1.55")):
    """Apply ledger rule 6 to one order record and return a new record."""
    updated = dict(record)
    updated["ledger_6"] = (Decimal(str(record.get("amount", 0))) * rate).quantize(Decimal("0.01"))
    updated.setdefault("history", []).append("ledger_step_6")
    return updated
