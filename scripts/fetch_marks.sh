#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data}"
for cohort in herlev sipakmed mendeley_lbc cric; do
    mkdir -p "${DATA_ROOT}/${cohort}"
    count=$(find "${DATA_ROOT}/${cohort}" -type f | wc -l | tr -d ' ')
    echo "${cohort}: ${count} images at ${DATA_ROOT}/${cohort}"
done
