#!/usr/bin/env python3
"""Build the data files the harness reads from the public Hugging Face rows.

The harness reads ``data/company-news/samples.json`` (a JSON array of cases) and
``data/company-news/manifest.json`` (``{"ids": [...]}``, the scored subset). Both are
gitignored and neither is distributed; the public dataset
``openbenchmarks/OB-News-Websearch`` (``data/eval.jsonl``) is the only source we have.

What the public rows carry, and what they do not (inspected at revision
5cc87497, 2026-10-15): exactly two fields per row, ``question`` and ``answer``. No ``id``,
no ``cells``, no source URL, no recipe/company/domain. The dataset card defines ``answer``
as "the gold cell (amount, person, product, acquirer, location, or headcount)", so each
case gets **one** cell whose value is the answer — a one-to-one mapping sanctioned by the
card, not an invented decomposition. ``ground_truth_url`` is written as the empty string:
the card says source URLs are omitted on purpose, and nothing here guesses one. The
``recipe`` is a keyword heuristic over the question's opening (for per-recipe tables only)
and is labelled as derived.

Pinned: the HF revision and the SHA-256 of ``eval.jsonl``. A download whose hash differs
is refused unless ``--allow-sha-mismatch`` is passed, so a quiet upstream edit cannot move
the question set under a frozen run.

  PYTHONPATH=scripts .venv/bin/python scripts/websearch/build_public_samples.py            # download + build
  PYTHONPATH=scripts .venv/bin/python scripts/websearch/build_public_samples.py --source path/to/eval.jsonl
  PYTHONPATH=scripts .venv/bin/python experiments/catalog_freeze.py freeze                   # then freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "company-news"
PUBLIC_DIR = DATA_DIR / "public"
SAMPLES = DATA_DIR / "samples.json"
MANIFEST = DATA_DIR / "manifest.json"
SOURCE_JSON = PUBLIC_DIR / "SOURCE.json"
EVAL_JSONL = PUBLIC_DIR / "eval.jsonl"

DATASET_ID = "openbenchmarks/OB-News-Websearch"
DATASET_FILE = "data/eval.jsonl"
PINNED_REVISION = "5cc874975289202f2bd113e9808cecd412ce3d0d"  # HF commit, lastModified 2026-09-03
PINNED_SHA256 = "7cc5b1b4d24c3c2bb1c9deab341d2289219f9a8f78d74fe93c4450572fcdf2f3"  # eval.jsonl at that revision
EXPECTED_ROWS = 100

RECIPE_RULES: tuple[tuple[str, str], ...] = (
    (r"^how much (did|was|has)\b.*\b(raise|raised|funding|round)", "financing"),
    (r"^how much (did|was)\b", "financing"),
    (r"^how many (employees|people|jobs|staff|workers)\b", "headcount_cut"),
    (r"\b(laying off|layoffs?|lay off|job cuts?)\b", "headcount_cut"),
    (r"^(who|whom)\b.*\b(appoint|appointed|became|named|elected|promoted|hired|joins?|joined)\b", "hire"),
    (r"^which company\b.*\b(acquir|acquisition|takeover|merge|buy|bought|purchase)", "acquisition"),
    (r"\b(acquire|acquired|acquisition|takeover offer)\b", "acquisition"),
    (r"^where did\b.*\b(open|opened|expand|headquarters|facility|office)", "new_office"),
    (r"\b(launch|launched|unveil|unveiled|introduc|generally available|announce[d]? .* (product|platform|offering|program))\b", "product_launch"),
    (r"^what\b.*\b(product|offering|platform|system|module|engine|documentary|effort|machine)\b", "product_launch"),
)


def derive_recipe(question: str) -> str:
    q = re.sub(r"\s+", " ", question or "").strip().lower()
    for pattern, recipe in RECIPE_RULES:
        if re.search(pattern, q):
            return recipe
    return "other"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(revision: str) -> bytes:
    import requests  # the harness's own dependency

    url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{DATASET_FILE}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def parse_rows(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def field_findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the rows carry, counted — so the README can state it rather than assume it."""
    keys: dict[str, int] = {}
    for row in rows:
        for key in row:
            keys[key] = keys.get(key, 0) + 1

    def present(*names: str) -> int:
        return sum(1 for row in rows if any(row.get(n) not in (None, "", [], {}) for n in names))

    return {
        "rows": len(rows),
        "keys": keys,
        "with_question": present("question"),
        "with_answer": present("answer", "expected_answer", "ground_truth"),
        "with_id": present("id"),
        "with_cells": present("cells"),
        "with_gold_object": present("gold"),
        "with_source_url": present("ground_truth_url", "source_url", "url", "primary_url"),
        "with_recipe": present("recipe", "pattern"),
        "with_company": present("company", "company_domain"),
        "gold_token_in_question": sum(
            1 for row in rows if str(row.get("answer") or "").strip() and str(row["answer"]).lower() in str(row.get("question") or "").lower()
        ),
        "duplicate_questions": len(rows) - len({str(row.get("question") or "") for row in rows}),
    }


def build_cases(rows: list[dict[str, Any]], *, revision: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            raise SystemExit(f"row {index}: question or answer missing: {row!r}")
        cases.append(
            {
                "id": f"cn-{index:03d}",
                "question": question,
                "expected_answer": answer,
                "ground_truth": answer,
                # The card defines `answer` as the single gold cell; nothing is split or added.
                "cells": [{"value": answer, "source": "hf:answer"}],
                # Source URLs are omitted from the public set by design. Never guessed.
                "ground_truth_url": "",
                "recipe": derive_recipe(question),
                "recipe_source": "derived_from_question_heuristic",
                "company": None,
                "company_domain": None,
                "source": {
                    "dataset": DATASET_ID,
                    "revision": revision,
                    "file": DATASET_FILE,
                    "row": index,
                    "question_sha256": sha256_bytes(question.encode("utf-8")),
                },
            }
        )
    return cases


def write_outputs(cases: list[dict[str, Any]], raw: bytes, *, revision: str, findings: dict[str, Any], sha: str) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_JSONL.write_bytes(raw)
    SAMPLES.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    MANIFEST.write_text(json.dumps({"ids": [case["id"] for case in cases]}, indent=2) + "\n")
    SOURCE_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "file": DATASET_FILE,
                "revision": revision,
                "eval_jsonl_sha256": sha,
                "pinned_sha256": PINNED_SHA256,
                "sha_matches_pin": sha == PINNED_SHA256,
                "rows": len(cases),
                "findings": findings,
                "recipes": {r: sum(1 for c in cases if c["recipe"] == r) for r in sorted({c["recipe"] for c in cases})},
                "cells_policy": "one cell per case, value = the public `answer` field (dataset card: `answer` is the gold cell)",
                "gold_url_policy": "absent in the public set; written as empty string; never inferred",
            },
            indent=2,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=None, help="Local eval.jsonl instead of downloading.")
    parser.add_argument("--revision", default=PINNED_REVISION, help="HF revision to download (default: pinned).")
    parser.add_argument("--allow-sha-mismatch", action="store_true", help="Build even if eval.jsonl's SHA-256 is not the pinned one.")
    args = parser.parse_args(argv)

    raw = args.source.read_bytes() if args.source else download(args.revision)
    sha = sha256_bytes(raw)
    if sha != PINNED_SHA256 and not args.allow_sha_mismatch:
        print(f"eval.jsonl sha256 {sha} != pinned {PINNED_SHA256}; refusing to build (use --allow-sha-mismatch)", file=sys.stderr)
        return 1
    rows = parse_rows(raw)
    findings = field_findings(rows)
    if len(rows) != EXPECTED_ROWS:
        print(f"warning: {len(rows)} rows, expected {EXPECTED_ROWS}", file=sys.stderr)
    cases = build_cases(rows, revision=args.revision)
    write_outputs(cases, raw, revision=args.revision, findings=findings, sha=sha)
    print(json.dumps({"samples": str(SAMPLES), "manifest": str(MANIFEST), "sha256": sha, **findings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
