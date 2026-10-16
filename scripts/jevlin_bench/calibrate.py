#!/usr/bin/env python3
"""Calibrate the two-layer judge on a frozen set of graded (question, gold, answer) triples.

Runs Layer A (deterministic) and, when a backend is named, Layer B (semantic) on every item
of scripts/jevlin_bench/testdata/calibration.json and reports agreement: each layer against
the expected verdict, and the two layers against each other, overall and by item kind. The
output is the evidence the README cites for "the judge was calibrated"; the rubric's SHA-256
is recorded so a later rubric change is visible.

  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/calibrate.py --backend none                 # Layer A only
  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/calibrate.py --backend claude-code --out DIR  # ~25 subscription calls
  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/calibrate.py --backend codex --out DIR        # when codex is operational
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _shared import load_environment  # noqa: E402
from jevlin_bench import judge  # noqa: E402

CALIBRATION = HERE / "testdata" / "calibration.json"


def load_items(path: Path = CALIBRATION) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["items"]


def run_calibration(items: list[dict[str, Any]], backend: judge.Backend | None, *, workdir: Path) -> dict[str, Any]:
    rows = []
    for item in items:
        cells = [{"value": v} for v in (item.get("cells") or [item["gold"]])]
        gold = judge.judge_answer(question=item["question"], expected=item["gold"], cells=cells, answer=item.get("answer") or "",
                                 backend=backend, workdir=workdir / item["id"])
        expected = item["expected"] == "correct"
        a = bool(gold["layer_a"]["correct"])
        b_block = gold.get("layer_b")
        b = bool(b_block["correct"]) if (b_block and b_block.get("ok")) else None
        rows.append({
            "id": item["id"], "kind": item["kind"], "expected": expected, "layer_a": a, "layer_b": b, "final": bool(gold["correct"]),
            "a_matches_expected": a == expected, "b_matches_expected": (b == expected) if b is not None else None,
            "ab_agree": (a == b) if b is not None else None, "b_error": (b_block or {}).get("error") if b_block else None,
            "a_methods": [c["method"] for c in gold["layer_a"].get("cells") or []],
            "b_justification": (b_block or {}).get("justification") if b_block else None,
            "b_cost_usd": ((b_block or {}).get("meta") or {}).get("total_cost_usd") if b_block else None,
            # The backend's own accounting: for codex, tool items (must be 0), tokens, latency, version.
            "b_meta": {k: ((b_block or {}).get("meta") or {}).get(k)
                       for k in ("backend", "model", "effort", "codex_version", "tool_items", "items", "usage", "elapsed_ms")} if b_block else None,
        })
    return {"rows": rows, "summary": summarize(rows, backend)}


def _backend_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metas = [r["b_meta"] for r in rows if r.get("b_meta") and r["b_meta"].get("elapsed_ms") is not None]
    if not metas:
        return {"calls": 0}
    usages = [m["usage"] for m in metas if isinstance(m.get("usage"), dict)]
    elapsed = sorted(int(m["elapsed_ms"]) for m in metas)
    return {
        "calls": len(metas),
        "tool_items": sum(int(m.get("tool_items") or 0) for m in metas),
        "model": metas[0].get("model"),
        "effort": metas[0].get("effort"),
        "codex_version": metas[0].get("codex_version"),
        "tokens": {key: sum(int(u.get(key) or 0) for u in usages) for key in ("input_tokens", "cached_input_tokens", "output_tokens")} if usages else None,
        "elapsed_ms": {"median": elapsed[len(elapsed) // 2], "max": elapsed[-1]},
    }


def _rate(flags: list[bool | None]) -> dict[str, Any]:
    known = [f for f in flags if f is not None]
    return {"n": len(known), "agree": sum(1 for f in known if f), "rate": round(sum(1 for f in known if f) / len(known), 3) if known else None}


def summarize(rows: list[dict[str, Any]], backend: judge.Backend | None) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(sub),
            "layer_a_vs_expected": _rate([r["a_matches_expected"] for r in sub]),
            "layer_b_vs_expected": _rate([r["b_matches_expected"] for r in sub]),
            "a_vs_b": _rate([r["ab_agree"] for r in sub]),
        }
    return {
        "n": len(rows),
        "backend": backend.label if backend else "none",
        "backend_family": backend.family if backend else None,
        "independent_of_extractor": backend.independent_of_extractor if backend else None,
        "rubric_sha256": judge.RUBRIC_SHA256,
        "layer_a_vs_expected": _rate([r["a_matches_expected"] for r in rows]),
        "layer_b_vs_expected": _rate([r["b_matches_expected"] for r in rows]),
        "a_vs_b": _rate([r["ab_agree"] for r in rows]),
        "final_vs_expected": _rate([r["final"] == r["expected"] for r in rows]),
        "layer_b_failures": sum(1 for r in rows if r["b_error"]),
        "layer_b_cost_usd": round(sum(float(r["b_cost_usd"] or 0) for r in rows), 4),
        "disagreements": [r["id"] for r in rows if r["ab_agree"] is False],
        "layer_a_misses": [r["id"] for r in rows if not r["a_matches_expected"]],
        "layer_b_misses": [r["id"] for r in rows if r["b_matches_expected"] is False],
        "layer_b_accounting": _backend_accounting(rows),
        "by_kind": by_kind,
    }


def to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# Judge calibration — backend `{s['backend']}` ({s['backend_family'] or 'n/a'}; independent of extractor: {s['independent_of_extractor']})",
        "",
        f"Items: {s['n']}. Rubric SHA-256: `{s['rubric_sha256'][:16]}…`.",
        "",
        "| agreement | n | agree | rate |",
        "|---|---|---|---|",
    ]
    for key in ("layer_a_vs_expected", "layer_b_vs_expected", "a_vs_b", "final_vs_expected"):
        r = s[key]
        lines.append(f"| {key} | {r['n']} | {r['agree']} | {r['rate'] if r['rate'] is not None else '—'} |")
    acc = s.get("layer_b_accounting") or {}
    accounting = f"Layer B calls: {acc.get('calls', 0)}."
    if acc.get("calls"):
        tokens = acc.get("tokens") or {}
        accounting += (f" Tool items used by the judge: **{acc.get('tool_items', 0)}**. Model `{acc.get('model')}` (effort {acc.get('effort')}), "
                       f"`{acc.get('codex_version')}`. Tokens in/cached/out: {tokens.get('input_tokens')}/{tokens.get('cached_input_tokens')}/"
                       f"{tokens.get('output_tokens')}. Latency median/max: {acc['elapsed_ms']['median']}/{acc['elapsed_ms']['max']} ms.")
    lines += ["", f"Layer B failures: {s['layer_b_failures']}. Layer B API-equivalent cost: ${s['layer_b_cost_usd']}. {accounting}",
              f"Layer A misses: {', '.join(s['layer_a_misses']) or 'none'}. Layer B misses: {', '.join(s['layer_b_misses']) or 'none'}. "
              f"A/B disagreements: {', '.join(s['disagreements']) or 'none'}.", "",
              "| id | kind | expected | A | B | final | B justification |", "|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        fmt = lambda v: "—" if v is None else ("correct" if v else "incorrect")  # noqa: E731
        lines.append(f"| {r['id']} | {r['kind']} | {fmt(r['expected'])} | {fmt(r['layer_a'])} | {fmt(r['layer_b'])} | {fmt(r['final'])} | {(r['b_justification'] or '')[:90]} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    load_environment()  # .env: CODEX_BIN for the default --codex-bin
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("none", "codex", "claude-code", "fake"), default="none")
    parser.add_argument("--items", type=Path, default=CALIBRATION)
    parser.add_argument("--out", type=Path, default=None, help="Directory for calibration-<backend>-<stamp>.json/.md")
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN") or "codex", help="default $CODEX_BIN or `codex` on PATH")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--workdir", type=Path, default=ROOT / "runs" / "calibration-work")
    args = parser.parse_args(argv)

    items = load_items(args.items)
    if args.backend == "codex":
        ok, status = judge.CodexBackend.available(args.codex_bin)
        if not ok:
            print(f"codex not operational: {status}", file=sys.stderr)
            return 2
    backend = judge.make_backend(args.backend, codex_bin=args.codex_bin, claude_bin=args.claude_bin,
                                 fake_script=lambda payload: {"verdict": "correct", "cells_hit": 1, "justification": "fake"})
    report = run_calibration(items, backend, workdir=args.workdir)
    report["ran_at"] = datetime.now(timezone.utc).isoformat()
    report["items_file"] = str(args.items)
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (args.out / f"calibration-{args.backend}-{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        (args.out / f"calibration-{args.backend}-{stamp}.md").write_text(to_markdown(report))
        print(f"wrote {args.out}/calibration-{args.backend}-{stamp}.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
