#!/usr/bin/env bash
# Usage: ./run.sh demo | check | app | test
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
case "${1:-}" in
  demo)  shift; $PY scripts/demo.py "$@" ;;
  check) $PY scripts/demo.py --check ;;
  gif)   $PY scripts/make_gif.py ;;
  test)  $PY -m pytest tests/ -q ;;
  *)     echo "usage: $0 {demo|check|gif|test}" ; exit 1 ;;
esac
