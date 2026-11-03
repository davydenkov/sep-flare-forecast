#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper from the files in data/ (no network access).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY scripts/01_build_clean_sample.py
$PY scripts/02_run_analysis.py
$PY scripts/03_make_figures_tables.py
