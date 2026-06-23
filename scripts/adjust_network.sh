#!/usr/bin/env bash
set -euo pipefail

PLAN="${1:-plans/principal.gin}"
CKPT="${2:-runs/principal/encoder.ckpt}"

python -m schemafree.sheets.probe "${PLAN}" --checkpoint "${CKPT}"
python -m schemafree.sheets.evaluate "${PLAN}" --checkpoint "${CKPT}"
