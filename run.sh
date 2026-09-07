#!/usr/bin/env bash
# Convenience launcher. Usage: ./run.sh m1 | m2 | test
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
case "${1:-}" in
  m1)   $PY scripts/m1_move.py ;;
  m2)   $PY scripts/m2_ik.py ;;
  m3)   $PY scripts/m3_pick.py ;;
  test) $PY -m pytest tests/ -q ;;
  *)    echo "usage: $0 {m1|m2|m3|test}" ; exit 1 ;;
esac
