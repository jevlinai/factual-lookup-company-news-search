"""The calibration harness with a fake backend, and the measures script over synthetic artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jevlin_bench import calibrate, judge, measures


class Calibration(unittest.TestCase):
    def test_items_are_well_formed_and_cover_every_kind(self):
        items = calibrate.load_items()
        self.assertGreaterEqual(len(items), 20)
        kinds = {i["kind"] for i in items}
        for kind in ("exact", "paraphrase", "numeric_equivalent", "date_equivalent", "missing_fact", "contradiction", "empty", "alias", "hedge", "multi_cell"):
            self.assertIn(kind, kinds)
        self.assertEqual(len({i["id"] for i in items}), len(items), "ids unique")
        for item in items:
            self.assertIn(item["expected"], ("correct", "incorrect"))

    def test_layer_a_only_report(self):
        items = calibrate.load_items()
        with tempfile.TemporaryDirectory() as tmp:
            report = calibrate.run_calibration(items, None, workdir=Path(tmp))
        s = report["summary"]
        self.assertEqual(s["backend"], "none")
        self.assertIsNone(s["layer_b_vs_expected"]["rate"])
        self.assertGreaterEqual(s["layer_a_vs_expected"]["rate"], 0.8, "Layer A alone is expected to miss only the semantic cases")
        self.assertIn("alias-state-abbrev", s["layer_a_misses"])
        md = calibrate.to_markdown(report)
        self.assertIn("| exact-money |", md)

    def test_fake_backend_report_and_agreement(self):
        items = calibrate.load_items()

        def oracle(payload):  # a perfect judge: substring of gold in the answer, except the hedge item
            gold = payload["gold_answer"].lower()
            ans = payload["extracted_answer"].lower()
            ok = gold in ans and "either" not in ans
            return {"verdict": "correct" if ok else "incorrect", "cells_hit": 1 if ok else 0, "justification": "oracle"}

        with tempfile.TemporaryDirectory() as tmp:
            report = calibrate.run_calibration(items, judge.FakeBackend(oracle), workdir=Path(tmp))
        s = report["summary"]
        self.assertEqual(s["backend"], "fake")
        self.assertEqual(s["layer_b_vs_expected"]["n"], len(items))
        self.assertIsNotNone(s["a_vs_b"]["rate"])
        self.assertEqual(s["layer_b_failures"], 0)
        self.assertEqual(len(report["rows"]), len(items))
        empty = next(r for r in report["rows"] if r["id"] == "empty")
        self.assertFalse(empty["final"])
        self.assertIn("by_kind", s)


def _synthetic_run() -> dict:
    hits_hit = [{"url": "https://e.com/a", "title": "A", "snippet": "MaxQ raised $31.5 million " + "x" * 100}]
    hits_miss = [{"url": "https://e.com/b", "title": "B", "snippet": "nothing useful"}]

    def gold(final, a, b=None, failed=False, agreement=None):
        return {"correct": final, "layer_a": {"correct": a}, "layer_b": ({"ok": True, "correct": b} if b is not None else ({"ok": False} if failed else None)),
                "judge_failed": failed, "agreement": agreement, "policy": "layer_b_primary"}

    def ar(a10, b10=None):
        out = {"layer_a": {"ar1": a10, "ar5": a10, "ar10": a10}, "ar1": a10, "ar5": a10, "ar10": a10}
        if b10 is not None:
            out["layer_b"] = {"ok": True, "ar1": b10, "ar5": b10, "ar10": b10}
            out.update({"ar1": b10, "ar5": b10, "ar10": b10})
        return out

    def meta(cost, cache=False, pending=0, latency=5000, upstream=11):
        return {"jevlin_cost_micros": cost, "jevlin_cache_hit": cache, "jevlin_pending": pending, "jevlin_latency_ms": latency,
                "jevlin_upstream_calls": upstream, "jevlin_arms_ok": upstream - 1, "jevlin_deadline_hit": False, "jevlin_no_usable_result": False}

    cases = [
        {"id": "c1", "question": "q", "vendors": {
            "jevlin_balanced": {"error": None, "latency_ms": 6000, "hits": hits_hit, "answer": "$31.5M", "cited_url": "https://e.com/a", "citation_valid": True,
                                "extract": {"status": "ok"}, "gold": gold(True, True, True, agreement=True), "ar": ar(True), "vendor_meta": meta(42000)},
            "exa_fast": {"error": None, "latency_ms": 700, "hits": hits_hit, "answer": "$31.5 million", "cited_url": "https://x.com/z", "citation_valid": False,
                         "extract": {"status": "ok"}, "gold": gold(True, True, True, agreement=True), "ar": ar(True)}}},
        {"id": "c2", "question": "q", "vendors": {
            "jevlin_balanced": {"error": None, "latency_ms": 100, "hits": hits_hit, "answer": "$30 million", "cited_url": "", "citation_valid": None,
                                "extract": {"status": "ok"}, "gold": gold(False, False, False, agreement=True), "ar": ar(True), "vendor_meta": meta(0, cache=True, latency=90)},
            "exa_fast": {"error": "HTTP 429", "status_code": 429, "latency_ms": 24000, "hits": [], "answer": "", "cited_url": "", "extract": {"status": "skipped"},
                         "gold": gold(False, False), "ar": ar(False)}}},
        {"id": "c3", "question": "q", "vendors": {
            "jevlin_balanced": {"error": None, "latency_ms": 4000, "hits": hits_miss, "answer": "", "cited_url": "", "citation_valid": None,
                                "extract": {"status": "ok"}, "gold": gold(False, False), "ar": ar(False), "vendor_meta": meta(44000, pending=1)},
            "exa_fast": {"error": None, "latency_ms": 500, "hits": hits_miss, "answer": "Nylund", "cited_url": "https://e.com/b", "citation_valid": True,
                         "extract": {"status": "ok"}, "gold": gold(True, False, True, agreement=False), "ar": ar(False)}}},
        {"id": "c4", "question": "q", "vendors": {
            "jevlin_balanced": {"error": None, "latency_ms": 5000, "hits": hits_hit, "answer": "$31.5 million", "cited_url": "https://e.com/a", "citation_valid": True,
                                "extract": {"status": "ok"}, "gold": gold(True, True, failed=True), "ar": ar(True), "vendor_meta": meta(40000)},
            "exa_fast": {"error": None, "latency_ms": 600, "hits": hits_hit, "answer": "", "cited_url": "", "extract": {"status": "invalid"},
                         "gold": gold(False, False), "ar": ar(True)}}},
    ]
    return {"endpoints": ["jevlin_balanced", "exa_fast"], "extractor": {"kind": "claude-code"}, "judge": {"layer_b": "fake"}, "cases": cases}


class Measures(unittest.TestCase):
    def setUp(self):
        self.card = {"usd_per_search": {"exa_fast": 0.007}, "_filled_on": "2026-10-15", "_file": "rates.json"}
        self.report = measures.measure(_synthetic_run(), self.card)

    def test_accuracy_denominator_includes_every_row(self):
        jev = self.report["endpoints"]["jevlin_balanced"]
        exa = self.report["endpoints"]["exa_fast"]
        self.assertEqual(jev["n"], 4)
        self.assertEqual(jev["correct"], 2)
        self.assertEqual(jev["accuracy"], 0.5)
        self.assertEqual(jev["accuracy_layer_a_only"], 0.5)
        self.assertEqual(jev["accuracy_layer_b_only"], 0.25, "B-only counts only rows where B produced a verdict")
        self.assertEqual(jev["judge_failed"], 1)
        self.assertEqual(exa["search_errors"], 1)
        self.assertEqual(exa["search_error_statuses"], {"429": 1})
        self.assertEqual(exa["accuracy"], 0.5)
        self.assertEqual(exa["ab_disagreements"], 1)
        self.assertEqual(exa["ab_disagreement_rate"], 0.5)

    def test_jevlin_cost_and_latency_exclude_cached(self):
        jev = self.report["endpoints"]["jevlin_balanced"]
        self.assertEqual(jev["cache_hits"], 1)
        self.assertEqual(jev["cost"]["searches_priced"], 3)
        self.assertEqual(jev["cost"]["usd_per_query"], 0.042)
        self.assertEqual(jev["cost"]["pending_floor_searches"], 1)
        self.assertEqual(jev["cost"]["usd_per_1k_queries"], 42.0)
        self.assertEqual(jev["cost"]["usd_per_1k_correct"], 84.0)
        self.assertTrue(jev["cost"]["basis"].startswith("observed"))
        self.assertEqual(jev["latency_rows"], 3, "the cached search's 100 ms is not in the latency statistics")
        self.assertEqual(jev["latency_ms_mean"], 5000.0)
        self.assertEqual(jev["upstream_calls_mean"], 11.0)

    def test_direct_cost_is_a_rate_card_estimate(self):
        exa = self.report["endpoints"]["exa_fast"]
        self.assertTrue(exa["cost"]["basis"].startswith("estimate"))
        self.assertEqual(exa["cost"]["usd_per_query"], 0.007)
        self.assertEqual(exa["cost"]["searches_billed"], 4, "a 429 is still billed, as upstream counts it")
        self.assertEqual(exa["cost"]["usd_per_1k_correct"], 14.0)
        self.assertIsNone(exa["cache_hits"])
        self.assertEqual(exa["latency_rows"], 3, "failed searches carry no latency row")

    def test_empty_citation_and_taxonomy(self):
        jev = self.report["endpoints"]["jevlin_balanced"]
        exa = self.report["endpoints"]["exa_fast"]
        self.assertEqual(jev["empty_answers"], 1)
        self.assertEqual(jev["empty_answer_rate"], 0.25)
        self.assertEqual(jev["citation_validity_rate"], round(2 / 3, 4))
        self.assertEqual(exa["citation_validity_rate"], 0.5)
        self.assertEqual(exa["extract_status"], {"ok": 2, "skipped": 1, "invalid": 1})
        self.assertEqual(jev["failure_taxonomy"], {"search_error": 0, "retrieval_failure_empty": 1, "retrieval_failure_wrong": 0,
                                                   "extraction_failure_empty": 0, "extraction_failure_wrong": 1, "not_judged": 0})
        self.assertEqual(exa["failure_taxonomy"]["search_error"], 1)
        self.assertEqual(exa["failure_taxonomy"]["extraction_failure_empty"], 1)
        self.assertEqual(jev["ar_layer_a"]["10"], 0.75)

    def test_markdown_and_cli(self):
        md = measures.to_markdown(self.report)
        self.assertIn("| jevlin_balanced |", md)
        self.assertIn("obs", md)
        self.assertIn("est", md)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run.json").write_text(json.dumps(_synthetic_run()))
            card = run_dir / "rates.json"
            card.write_text(json.dumps(self.card))
            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                rc = measures.main(["--run-dir", str(run_dir), "--rate-card", str(card), "--md", str(run_dir / "m.md"), "--json", str(run_dir / "m.json")])
            self.assertEqual(rc, 0)
            out = json.loads((run_dir / "m.json").read_text())
            self.assertEqual(out["endpoints"]["jevlin_balanced"]["accuracy"], 0.5)
            self.assertEqual(out["rate_card"]["filled_on"], "2026-10-15")


if __name__ == "__main__":
    unittest.main()
