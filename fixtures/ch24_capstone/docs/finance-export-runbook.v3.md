# Finance export runbook

## Destinations

Destination A is the finance share for tenant acme. Files written there are
read by the month-end reconciliation job at 06:00 local time.

## Refunds

Refunded orders are removed from a daily report before it is exported to
finance. A finance reviewer publishes the corrected revision.
