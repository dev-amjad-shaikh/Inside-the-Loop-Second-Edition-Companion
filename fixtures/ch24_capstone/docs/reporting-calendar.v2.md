# Reporting calendar

## Report day

A report day is the calendar day in the report timezone configured for the
tenant in settings.json under report_timezone. An order belongs to the daily
report for the local date on which it was placed. Timestamps are stored in
UTC, so convert created_at to the report timezone before taking its date.

## Changes to this definition

The public function belongs_to_report(created_at, requested_day, timezone) is
called by the export job and the dashboard. Changing its signature requires a
coordinated release of both callers.

## Owners

The reporting platform team owns this definition.
