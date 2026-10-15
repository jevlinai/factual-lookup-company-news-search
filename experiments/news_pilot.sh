#!/usr/bin/env bash
# Company-news adaptation, pilot: every frozen question x every arm, one cache-aware pass
# per invocation of this script (rerun with the same RUN_DIR to resume; PASSES>1 creates
# pass-N run dirs under RUN_DIR). Searches bill the Jevlin project key and the direct vendor
# keys; extraction and the Claude fallback judge bill the Claude subscription.
# NOT RUN during preparation: a pilot is a live pass and needs authorisation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

: "${PASSES:=1}"
: "${DIRECT:=exa_fast,perplexity_low,parallel_fast}"
: "${JEVLIN:=jevlin_balanced,jevlin_fast,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged}"
: "${LAYER_B:=codex}"
: "${CACHE_GAP_MIN:=16}"
: "${RUN_DIR:=runs/news-pilot}"
: "${DRY_RUN:=}"
mkdir -p "$RUN_DIR"

"$PY" experiments/catalog_freeze.py verify

for pass in $(seq 1 "$PASSES"); do
  DIR="$RUN_DIR/pass-$pass"
  echo "== pass $pass / $PASSES  dir=$DIR  direct=$DIRECT  jevlin=$JEVLIN  layer_b=$LAYER_B"
  if ! PYTHONPATH=scripts "$PY" experiments/news_pass.py \
      --run-dir "$DIR" --direct "$DIRECT" --jevlin "$JEVLIN" \
      --cache-gap-min "$CACHE_GAP_MIN" --case-gap-s 1.0 --abort-after-search-errors 12 --retry-errors \
      --layer-b "$LAYER_B" ${DRY_RUN:+--dry-run}; then
    echo "pass $pass stopped; rerun news_pilot.sh with RUN_DIR=$RUN_DIR to resume."
    exit 1
  fi
done
