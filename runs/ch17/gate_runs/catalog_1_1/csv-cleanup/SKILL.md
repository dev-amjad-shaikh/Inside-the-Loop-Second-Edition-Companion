---
name: csv-cleanup
description: Clean exported CSV files that will not load, fixing delimiters, encodings, quoting and duplicated header rows. Use when a CSV export fails to import or parse.
metadata:
  version: "1.0.0"
  maintainer: data-platform
---
# CSV cleanup

1. Detect the delimiter and encoding from the first 64 KB.
2. Remove repeated header rows and report how many were removed.
3. Write the cleaned file beside the original; never overwrite it.
