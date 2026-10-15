#!/usr/bin/env bash
# Company-news adaptation, smoke: a few questions x every arm, one cache-aware pass.
# Searches bill the Jevlin project key and the direct vendor keys; extraction and the
# Claude fallback judge bill the Claude subscription; Codex bills the ChatGPT login.
# NOT RUN during preparation: a smoke is a live pass and needs authorisation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
[ -z "$PY" ] && { [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python3; }

: "${LIMIT:=3}"
: "${DIRECT:=exa_fast,perplexity_low,parallel_fast}"
: "${JEVLIN:=jevlin_balanced,jevlin_fast,jevlin_single,jevlin_fanout_raw,jevlin_fanout_merged}"
: "${LAYER_B:=codex}"
: "${CACHE_GAP_MIN:=16}"
: "${RUN_DIR:=runs/news-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${DRY_RUN:=}"

"$PY" experiments/catalog_freeze.py verify

PYTHONPATH=scripts "$PY" experiments/news_pass.py \
  --run-dir "$RUN_DIR" --limit "$LIMIT" --direct "$DIRECT" --jevlin "$JEVLIN" \
  --cache-gap-min "$CACHE_GAP_MIN" --case-gap-s 1.0 --abort-after-search-errors 6 \
  --layer-b "$LAYER_B" ${DRY_RUN:+--dry-run}

echo "smoke run dir: $RUN_DIR (measures.md / measures.json inside)"
