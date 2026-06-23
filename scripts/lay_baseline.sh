#!/usr/bin/env bash
set -euo pipefail

PLAN="${1:-plans/principal.gin}"
DATA_ROOT="${2:-data}"
NPROC="${NPROC:-4}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    -m schemafree.sheets.pretrain "${PLAN}" --data-root "${DATA_ROOT}"
