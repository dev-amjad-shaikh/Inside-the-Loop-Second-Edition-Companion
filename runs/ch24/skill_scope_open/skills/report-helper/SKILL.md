---
name: report-helper
description: >
  Diagnose and fix day-boundary bugs in report filtering code, where records
  land on the wrong report day because a UTC timestamp is compared with a
  local report date. Use when a daily or weekly report is missing records or
  counting them twice near midnight, after a timezone change or around a
  daylight-saving switch. Do not use for writing, drafting or summarizing
  report documents, or for scheduling or deploying report jobs.
license: Apache-2.0
compatibility: Python 3.9 or later with zoneinfo. Needs no network access.
allowed-tools: read_file replace_text run_tests read_skill_resource run_skill_script
metadata:
  version: "1.0.0"
  maintainer: reporting-platform
  network: none
  pinned-scripts: "scripts/probe_boundary.py sha256:c56c68f90d2b7699a2171c5ed8f10d4c9e3da9a13ce50392eb6515bfbad4c84f"
---
# Fixing report day-boundary bugs

Follow these steps in order. Each step produces evidence the next one uses.

1. Read the report timezone setting and the filtering function. Note the
   function's documented input contract: which argument is UTC and which is
   a local date.
2. Add a regression test from `references/regression-test.md`. Do not change
   the implementation yet.
3. Run the tests and confirm the new test fails. If it passes, it does not
   exercise the bug: revise the test before going on.
4. Convert the timestamp to the configured timezone before taking its date.
   Keep the public function signature unchanged.
5. Run `scripts/probe_boundary.py <module> <timezone> <report-day>
   <switch-day>` with the next daylight-saving switch day. Every row must be
   ok; the script exits 1 and lists the wrong rows otherwise.
6. Run the tests again and confirm they pass on the edited revision.
7. Report the cause, the change, the failing and passing test results with
   the file revision each applies to, and what you did not do. Do not
   deploy: releases go through the release pipeline.

## Gotchas

- `created_at.date()` on an aware UTC datetime returns the UTC date, which
  is the bug this skill exists for.
- Read `references/timezone-gotchas.md` only if the report timezone is east
  of UTC or the probe reports a wrong row.
