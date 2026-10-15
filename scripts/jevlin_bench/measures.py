#!/usr/bin/env python3
"""Measures over a searched, extracted and judged run. Reads artifacts only; calls nothing.

Per configuration (endpoint), from <run-dir>/run.json after cc_extract.py --apply and
judge.py --apply:

  accuracy              final verdict (layer_b_primary), plus A-only, B-only, A∧B, A∨B;
                        every (question, configuration) row is in the denominator, including
                        failed searches, failed extractions and failed judge calls
  AR@1/5/10             answer recall on the stored snippets, by the deterministic layer and,
                        when run, the semantic layer — independent of the extractor
  search cost / query   Jevlin: OBSERVED mean of usage.cost_micros over non-cached searches
                        (a floor where pending>0); direct: ESTIMATE from the dated rate card.
                        Never summed, always labelled.
  $ / 1k correct        cost per 1,000 queries ÷ accuracy (the board's metric), per the above
  latency               harness wall time of the search request (retry sleeps included, as
                        upstream); Jevlin cached searches excluded; gateway-reported latency
                        shown beside it
  snippet tokens        title+snippet, ~4 chars/token (the upstream stand-in)
  empty-answer rate, citation validity (cited_url ∈ supplied hits), upstream calls, cache
  hits, failures by stage, extractor/judge disagreement, retrieval vs extraction failure

  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/measures.py --run-dir DIR --md DIR/measures.md --json DIR/measures.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_RATE_CARD = ROOT / "experiments" / "rates.json"
KS = (1, 5, 10)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _pctile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] * (1 - (idx - lo)) + ordered[hi] * (idx - lo), 1)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _tokens(text: str) -> int:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return max(1, round(len(cleaned) / 4)) if cleaned else 0


def is_jevlin(name: str) -> bool:
    return name.startswith("jevlin_")


def load_rate_card(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"usd_per_search": {}, "_filled_on": None, "_file": str(path) if path else None}
    card = json.loads(path.read_text())
    card["_file"] = str(path)
    return card


def score_endpoint(cases: list[dict[str, Any]], name: str, rate_card: dict[str, Any]) -> dict[str, Any]:
    rows = [(case, (case.get("vendors") or {})[name]) for case in cases if name in (case.get("vendors") or {})]
    n = len(rows)
    ok_rows = [(c, b) for c, b in rows if not b.get("error")]
    err_rows = [(c, b) for c, b in rows if b.get("error")]
    statuses: dict[str, int] = {}
    for _, b in err_rows:
        key = str(b.get("status_code") or "no-response")
        statuses[key] = statuses.get(key, 0) + 1

    def gold(b: dict[str, Any]) -> dict[str, Any]:
        return b.get("gold") if isinstance(b.get("gold"), dict) else {}

    judged = [(c, b) for c, b in rows if gold(b).get("correct") is not None]
    final_correct = sum(1 for _, b in rows if gold(b).get("correct") is True)
    a_correct = sum(1 for _, b in rows if (gold(b).get("layer_a") or {}).get("correct") is True)
    b_rows = [(c, b) for c, b in rows if (gold(b).get("layer_b") or {}).get("ok")]
    b_correct = sum(1 for _, b in b_rows if (gold(b).get("layer_b") or {}).get("correct") is True)
    both = sum(1 for _, b in b_rows if (gold(b).get("layer_a") or {}).get("correct") and (gold(b).get("layer_b") or {}).get("correct"))
    either = sum(1 for _, b in b_rows if (gold(b).get("layer_a") or {}).get("correct") or (gold(b).get("layer_b") or {}).get("correct"))
    agreement_known = [(c, b) for c, b in rows if gold(b).get("agreement") is not None]
    disagree = sum(1 for _, b in agreement_known if gold(b).get("agreement") is False)
    judge_failed = sum(1 for _, b in rows if gold(b).get("judge_failed"))
    policy = next((gold(b).get("policy") for _, b in judged), None)

    def ar(b: dict[str, Any]) -> dict[str, Any]:
        return b.get("ar") if isinstance(b.get("ar"), dict) else {}

    ar_a = {k: sum(1 for _, b in rows if (ar(b).get("layer_a") or {}).get(f"ar{k}")) for k in KS}
    ar_b_rows = [(c, b) for c, b in rows if (ar(b).get("layer_b") or {}).get("ok")]
    ar_b = {k: sum(1 for _, b in ar_b_rows if (ar(b).get("layer_b") or {}).get(f"ar{k}")) for k in KS}
    ar_final = {k: sum(1 for _, b in rows if ar(b).get(f"ar{k}")) for k in KS}

    answers = [(c, b) for c, b in ok_rows if (b.get("answer") or "").strip()]
    empty = len(ok_rows) - len(answers)
    extract_status: dict[str, int] = {}
    for _, b in rows:
        st = (b.get("extract") or {}).get("status") or ("not-extracted" if "extract" not in b else "?")
        extract_status[st] = extract_status.get(st, 0) + 1
    cited = [(c, b) for c, b in answers if (b.get("cited_url") or "").strip()]
    cite_valid = sum(1 for _, b in cited if b.get("citation_valid") is True)

    tokens = [float(sum(_tokens(f"{h.get('title') or ''} {h.get('snippet') or ''}") for h in (b.get("hits") or [])[:10])) for _, b in ok_rows]
    hits_n = [float(len((b.get("hits") or [])[:10])) for _, b in ok_rows]

    meta_rows = [(c, b, b.get("vendor_meta") or {}) for c, b in ok_rows]
    cached = [(c, b, m) for c, b, m in meta_rows if m.get("jevlin_cache_hit")]
    fresh = [(c, b, m) for c, b, m in meta_rows if not m.get("jevlin_cache_hit")]
    latency_rows = fresh if is_jevlin(name) else [(c, b, b.get("vendor_meta") or {}) for c, b in ok_rows]
    latencies = [float(b.get("latency_ms") or 0) for _, b, _ in latency_rows]
    gateway_latency = [float(m.get("jevlin_latency_ms") or 0) for _, _, m in fresh if m.get("jevlin_latency_ms") is not None]

    cost: dict[str, Any]
    if is_jevlin(name):
        micros = [float(m.get("jevlin_cost_micros")) for _, _, m in fresh if isinstance(m.get("jevlin_cost_micros"), (int, float))]
        pending = sum(1 for _, _, m in fresh if int(m.get("jevlin_pending") or 0) > 0)
        per_query = (sum(micros) / len(micros) / 1_000_000) if micros else None
        cost = {
            "basis": "observed (usage.cost_micros, non-cached searches)",
            "usd_per_query": round(per_query, 6) if per_query is not None else None,
            "usd_total_observed": round(sum(micros) / 1_000_000, 4),
            "searches_priced": len(micros),
            "pending_floor_searches": pending,
            "cached_searches_excluded": len(cached),
        }
    else:
        card_value = (rate_card.get("usd_per_search") or {}).get(name)
        billed = sum(1 for _, b in rows if not b.get("error") or (b.get("status_code") not in (401, 402, 403) and b.get("status_code") is not None))
        per_query = float(card_value) if card_value is not None else None
        cost = {
            "basis": f"estimate (rate card {rate_card.get('_filled_on')})",
            "usd_per_query": per_query,
            "usd_total_estimate": round(per_query * billed, 4) if per_query is not None else None,
            "searches_billed": billed,
        }
    accuracy = _rate(final_correct, n)
    per_1k = round(cost["usd_per_query"] * 1000, 4) if cost.get("usd_per_query") is not None else None
    cost["usd_per_1k_queries"] = per_1k
    cost["usd_per_1k_correct"] = round(per_1k / accuracy, 2) if (per_1k is not None and accuracy) else None

    # Where the incorrect rows were lost. "evidence present" = some stored snippet was judged
    # answer-bearing (final AR source); with it, a wrong or empty answer is an extraction failure.
    taxonomy = {"search_error": 0, "retrieval_failure_empty": 0, "retrieval_failure_wrong": 0, "extraction_failure_empty": 0,
                "extraction_failure_wrong": 0, "not_judged": 0}
    for _, b in rows:
        g = gold(b)
        if g.get("correct") is None:
            taxonomy["not_judged"] += 1
            continue
        if g.get("correct"):
            continue
        if b.get("error"):
            taxonomy["search_error"] += 1
            continue
        evidence = bool(ar(b).get("ar10"))
        empty_answer = not (b.get("answer") or "").strip()
        key = ("extraction_failure" if evidence else "retrieval_failure") + ("_empty" if empty_answer else "_wrong")
        taxonomy[key] += 1

    return {
        "n": n,
        "searches_ok": len(ok_rows),
        "search_errors": len(err_rows),
        "search_error_statuses": statuses,
        "judged": len(judged),
        "accuracy": accuracy,
        "correct": final_correct,
        "accuracy_policy": policy,
        "accuracy_layer_a_only": _rate(a_correct, n),
        "accuracy_layer_b_only": _rate(b_correct, n) if b_rows else None,
        "accuracy_a_and_b": _rate(both, n) if b_rows else None,
        "accuracy_a_or_b": _rate(either, n) if b_rows else None,
        "layer_b_rows": len(b_rows),
        "judge_failed": judge_failed,
        "ab_disagreements": disagree,
        "ab_disagreement_rate": _rate(disagree, len(agreement_known)),
        "ar_final": {str(k): _rate(ar_final[k], n) for k in KS},
        "ar_layer_a": {str(k): _rate(ar_a[k], n) for k in KS},
        "ar_layer_b": {str(k): _rate(ar_b[k], n) for k in KS} if ar_b_rows else None,
        "empty_answers": empty,
        "empty_answer_rate": _rate(empty, len(ok_rows)),
        "extract_status": extract_status,
        "answers": len(answers),
        "cited": len(cited),
        "citation_valid": cite_valid,
        "citation_validity_rate": _rate(cite_valid, len(answers)),
        "hits_mean": _mean(hits_n),
        "snippet_tokens_mean": _mean(tokens),
        "snippet_tokens_p50": _pctile(tokens, 0.5),
        "latency_ms_mean": _mean(latencies),
        "latency_ms_p50": _pctile(latencies, 0.5),
        "latency_ms_p95": _pctile(latencies, 0.95),
        "latency_rows": len(latencies),
        "gateway_latency_ms_mean": _mean(gateway_latency) if is_jevlin(name) else None,
        "upstream_calls_mean": _mean([float(m.get("jevlin_upstream_calls")) for _, _, m in meta_rows if isinstance(m.get("jevlin_upstream_calls"), (int, float))]) if is_jevlin(name) else (1.0 if ok_rows else None),
        "upstream_arms_ok_mean": _mean([float(m.get("jevlin_arms_ok")) for _, _, m in meta_rows if isinstance(m.get("jevlin_arms_ok"), (int, float))]) if is_jevlin(name) else None,
        "cache_hits": len(cached) if is_jevlin(name) else None,
        "cache_hit_rate": _rate(len(cached), len(ok_rows)) if is_jevlin(name) else None,
        "deadline_hits": sum(1 for _, _, m in meta_rows if m.get("jevlin_deadline_hit")) if is_jevlin(name) else None,
        "no_usable_result": sum(1 for _, _, m in meta_rows if m.get("jevlin_no_usable_result")) if is_jevlin(name) else None,
        "cost": cost,
        "failure_taxonomy": taxonomy,
    }


def measure(payload: dict[str, Any], rate_card: dict[str, Any]) -> dict[str, Any]:
    cases = payload.get("cases") or []
    names = list(payload.get("endpoints") or [])
    for case in cases:
        for name in (case.get("vendors") or {}):
            if name not in names:
                names.append(name)
    return {
        "n_cases": len(cases),
        "extractor": payload.get("extractor"),
        "judge": payload.get("judge"),
        "rate_card": {"file": rate_card.get("_file"), "filled_on": rate_card.get("_filled_on")},
        "endpoints": {name: score_endpoint(cases, name, rate_card) for name in names},
        "definitions": {
            "accuracy": "final verdict share over every (question, configuration) row; failed searches, extractions and judge calls count as incorrect",
            "ar_at_k": "share of rows with an answer-bearing snippet in the top K of the stored hits (layer_a: deterministic matcher; layer_b: semantic judge when run)",
            "cost.jevlin": "observed mean of usage.cost_micros over non-cached searches; a floor where pending>0",
            "cost.direct": "rate-card estimate per billed search; never summed with observed Jevlin cost",
            "latency": "harness wall time of the search request including retry sleeps (upstream definition); Jevlin cached searches excluded",
            "retrieval_failure": "incorrect and no stored snippet judged answer-bearing (final AR source)",
            "extraction_failure": "incorrect although some stored snippet was judged answer-bearing",
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    cols = ["arm", "n", "acc", "acc A", "acc B", "A/B dis.", "AR@1", "AR@5", "AR@10", "$/query", "$/1k correct", "basis", "lat ms", "p95", "snip tok",
            "empty", "cite ok", "upstream", "cache", "srch err", "retr. fail", "extr. fail"]
    lines = [f"# Measures — {report['n_cases']} cases", "", "Jevlin cost is observed; direct cost is a rate-card estimate. Not comparable to the public board.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v:.1%}"

    def num(v: Any, fmt: str = "{:.0f}") -> str:
        return "—" if v is None else fmt.format(v)

    for name, m in report["endpoints"].items():
        c = m["cost"]
        t = m["failure_taxonomy"]
        lines.append("| " + " | ".join([
            name, str(m["n"]), pct(m["accuracy"]), pct(m["accuracy_layer_a_only"]), pct(m["accuracy_layer_b_only"]), pct(m["ab_disagreement_rate"]),
            pct(m["ar_final"]["1"]), pct(m["ar_final"]["5"]), pct(m["ar_final"]["10"]),
            num(c.get("usd_per_query"), "${:.4f}"), num(c.get("usd_per_1k_correct"), "${:.2f}"), "obs" if c["basis"].startswith("observed") else "est",
            num(m["latency_ms_mean"]), num(m["latency_ms_p95"]), num(m["snippet_tokens_mean"]), pct(m["empty_answer_rate"]), pct(m["citation_validity_rate"]),
            num(m["upstream_calls_mean"], "{:.1f}"), num(m["cache_hits"]), str(m["search_errors"]),
            str(t["retrieval_failure_empty"] + t["retrieval_failure_wrong"]), str(t["extraction_failure_empty"] + t["extraction_failure_wrong"]),
        ]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rate-card", type=Path, default=DEFAULT_RATE_CARD)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    args = parser.parse_args(argv)
    run_path = args.run_dir / "run.json"
    if not run_path.exists():
        run_path = args.run_dir / "run.partial.json"
    payload = json.loads(run_path.read_text())
    report = measure(payload, load_rate_card(args.rate_card))
    report["run"] = str(run_path)
    md = to_markdown(report)
    print(md)
    if args.md:
        args.md.write_text(md)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
