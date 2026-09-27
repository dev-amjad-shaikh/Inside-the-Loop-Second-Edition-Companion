---
name: report-timezone-fixer
description: >
  Fix timezone bugs in daily and weekly report code when orders placed near
  midnight are missing from a report or counted twice after a
  daylight-saving switch. Use when report filtering compares UTC timestamps
  with the local report day.
allowed-tools: read_file replace_text run_tests
metadata:
  version: "1.0.0"
  maintainer: data-platform
---
# Report timezone fixer

1. Read the report settings and the filtering function.
2. Convert timestamps to the report timezone before taking the date.
3. Run the tests.
