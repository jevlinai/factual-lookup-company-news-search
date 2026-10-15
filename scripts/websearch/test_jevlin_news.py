"""Offline contracts for the Jevlin arms in the company-news harness. No network, no keys.

The HTTP wrapper is patched; the fixture is the coding-agents fork's merged response at
f01b032 (scripts/websearch/testdata/jevlin/merged_response.json), so presentation behaviour
is checked against the same bytes as the other suite.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from websearch import jevlin_news
from websearch import run_official_news_probe as runner
from websearch.score_official_search_metrics import ENDPOINTS as SCORED_ENDPOINTS
from websearch.score_official_search_metrics import UPSTREAM_ENDPOINTS as SCORED_UPSTREAM

HERE = Path(__file__).resolve().parent
MERGED = json.loads((HERE / "testdata" / "jevlin" / "merged_response.json").read_text())

VERSIONS_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/versions-api.html"
SHARED_URL = "https://servicenow.com/docs/r/api-reference/rest-apis/cdm-shared-libraries-api.html"
CDM_URL = "https://developer.servicenow.com/dev.do/reference/api/latest/rest/cdm-api"
THREAD_URL = "https://community.servicenow.com/cdm-snapshots-thread"

ENV = {"JEVLIN_API_KEY": "test-key", "JEVLIN_API_URL": "https://jevlin.test", "JEVLIN_PROVIDERS": "exa,brave,serpapi,tinyfish,you"}


def _env(**extra: str):
    return patch.dict(os.environ, {**ENV, **extra}, clear=False)


def _ok(payload=MERGED):
    return {"ok": True, "status_code": 200, "response": payload, "latency_ms": 123, "error": None, "attempts": [{"n": 1}]}


class Rosters(unittest.TestCase):
    def test_every_arm_is_registered_everywhere(self):
        for name in jevlin_news.ARMS:
            with self.subTest(arm=name):
                self.assertIn(name, runner.ENDPOINTS)
                self.assertIn(name, runner.ENDPOINT_MAP)
                self.assertIn(name, runner.ENDPOINT_ENV)
                self.assertIn(name, runner.UNIT_COST_USD)
                self.assertIn(name, SCORED_ENDPOINTS)
                self.assertEqual(runner._parse_endpoints(name), [name])
                self.assertEqual(runner.UNIT_COST_USD[name], 0.0, "metered arm: no list price")

    def test_arms_are_exactly_the_conventions_five(self):
        self.assertEqual(jevlin_news.ARMS, ("jevlin_fast", "jevlin_balanced", "jevlin_single", "jevlin_fanout_raw", "jevlin_fanout_merged"))
        self.assertNotIn("jevlin_cli", runner.ENDPOINTS)
        self.assertNotIn("jevlin_max", runner.ENDPOINTS)

    def test_upstream_roster_is_untouched_and_first(self):
        self.assertEqual(len(runner.UPSTREAM_ENDPOINTS), 20)
        self.assertEqual(runner.ENDPOINTS[:20], runner.UPSTREAM_ENDPOINTS)
        self.assertEqual(SCORED_ENDPOINTS[:20], SCORED_UPSTREAM)
        self.assertEqual(runner.UNIT_COST_USD["exa_fast"], 0.007)
        self.assertEqual(runner.ENDPOINT_ENV["exa_fast"], ("EXA_API_KEY",))

    def test_fanout_arms_require_pinned_providers_in_env_check(self):
        with patch.dict(os.environ, {"JEVLIN_API_KEY": "k"}, clear=False):
            os.environ.pop("JEVLIN_PROVIDERS", None)
            self.assertIn("JEVLIN_PROVIDERS", runner._missing_env(["jevlin_fanout_raw"], "none"))
            self.assertNotIn("JEVLIN_PROVIDERS", runner._missing_env(["jevlin_balanced"], "none"))

    def test_extractor_none_drops_openai_requirement(self):
        self.assertNotIn("OPENAI_API_KEY", runner._required_env(["jevlin_fast"], "none"))
        self.assertIn("OPENAI_API_KEY", runner._required_env(["jevlin_fast"], "openai"))
        self.assertIn("OPENAI_API_KEY", runner._required_env(["exa_fast"]))


class RequestShapes(unittest.TestCase):
    def _call(self, arm: str, question: str = "Unmodified question?"):
        with _env(), patch.object(runner, "_http", return_value=_ok()) as http:
            hits, raw = runner.call_endpoint(arm, question, {})
        return hits, raw, http

    def test_fast_and_balanced_send_tier_without_providers(self):
        for arm, tier in (("jevlin_fast", "fast"), ("jevlin_balanced", "balanced")):
            with self.subTest(arm=arm):
                _, _, http = self._call(arm)
                kw = http.call_args.kwargs
                self.assertEqual(kw["method"], "POST")
                self.assertEqual(kw["url"], "https://jevlin.test/v1/search")
                self.assertEqual(kw["body"], {"query": "Unmodified question?", "max_results": 10, "view": "merged", "tier": tier})
                self.assertEqual(kw["headers"]["Authorization"], "Bearer test-key")
                self.assertEqual(kw["params"], None)
                self.assertEqual(kw["timeout"], 45)

    def test_single_sends_one_provider_and_no_tier(self):
        _, _, http = self._call("jevlin_single")
        self.assertEqual(http.call_args.kwargs["body"], {"query": "Unmodified question?", "max_results": 10, "view": "merged", "providers": ["exa"]})
        with _env(JEVLIN_SINGLE_PROVIDER="brave"), patch.object(runner, "_http", return_value=_ok()) as http:
            runner.call_endpoint("jevlin_single", "q", {})
        self.assertEqual(http.call_args.kwargs["body"]["providers"], ["brave"])

    def test_fanout_arms_pin_providers_with_balanced_tier(self):
        for arm in ("jevlin_fanout_raw", "jevlin_fanout_merged"):
            with self.subTest(arm=arm):
                _, _, http = self._call(arm)
                self.assertEqual(http.call_args.kwargs["body"], {
                    "query": "Unmodified question?", "max_results": 10, "view": "merged", "tier": "balanced",
                    "providers": ["exa", "brave", "serpapi", "tinyfish", "you"],
                })

    def test_fanout_refuses_without_pinned_set(self):
        with _env(JEVLIN_PROVIDERS=""), patch.object(runner, "_http", return_value=_ok()):
            with self.assertRaises(RuntimeError):
                runner.call_endpoint("jevlin_fanout_raw", "q", {})

    def test_missing_key_is_refused_before_any_call(self):
        with patch.dict(os.environ, {"JEVLIN_API_KEY": ""}, clear=False), patch.object(runner, "_http") as http:
            with self.assertRaises(RuntimeError):
                runner.call_endpoint("jevlin_fast", "q", {})
            http.assert_not_called()

    def test_question_goes_out_verbatim(self):
        q = "  How much did  MaxQ Medical raise in its latest funding round?  "
        _, _, http = self._call("jevlin_balanced", q)
        self.assertEqual(http.call_args.kwargs["body"]["query"], q)

    def test_timeout_env_is_respected(self):
        with _env(JEVLIN_TIMEOUT_S="20"), patch.object(runner, "_http", return_value=_ok()) as http:
            runner.call_endpoint("jevlin_fast", "q", {})
        self.assertEqual(http.call_args.kwargs["timeout"], 20)

    def test_upstream_adapter_is_untouched(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "e"}, clear=False), patch.object(runner, "_http", return_value=_ok({"results": []})) as http:
            runner.call_endpoint("exa_fast", "Q", {})
        self.assertEqual(http.call_args.kwargs["body"], {"query": "Q", "type": "fast", "numResults": 10, "contents": {"highlights": True}})
        self.assertEqual(http.call_args.kwargs["url"], "https://api.exa.ai/search")


class Presentation(unittest.TestCase):
    def test_merged_order_in_harness_hit_shape(self):
        with _env(), patch.object(runner, "_http", return_value=_ok()):
            hits, raw = runner.call_endpoint("jevlin_balanced", "q", {})
        self.assertEqual([h["url"] for h in hits], [VERSIONS_URL, SHARED_URL, CDM_URL, THREAD_URL])
        self.assertEqual(set(hits[0]), {"url", "title", "snippet", "host"})
        self.assertEqual(hits[0]["host"], "servicenow.com")
        self.assertTrue(hits[0]["snippet"].startswith("The Versions API provides endpoints"))
        self.assertEqual(raw["vendor_meta"]["jevlin_presentation"], "merged")
        self.assertEqual(raw["vendor_meta"]["jevlin_hits_shown"], 4)
        self.assertEqual(raw["vendor_meta"]["jevlin_merged_total"], 4)

    def test_raw_is_round_robin_over_ok_providers(self):
        with _env(), patch.object(runner, "_http", return_value=_ok()):
            hits, raw = runner.call_endpoint("jevlin_fanout_raw", "q", {})
        self.assertEqual([h["url"] for h in hits], [VERSIONS_URL, CDM_URL, SHARED_URL, THREAD_URL])
        self.assertEqual(hits[0]["snippet"], "PUT /api/sn_cdm/versions/publish marks a snapshot version as published for export and CI/CD use.")
        self.assertEqual(raw["vendor_meta"]["jevlin_presentation"], "raw")

    def test_cap_at_ten_and_host_path_dedupe(self):
        pages = [{"url": f"https://example.com/p{i}", "title": f"t{i}", "excerpt": "x  y\n z"} for i in range(15)]
        pages.insert(1, {"url": "https://www.example.com/p0/", "title": "dup", "excerpt": "dup"})
        payload = {**MERGED, "merged": pages}
        with _env(), patch.object(runner, "_http", return_value=_ok(payload)):
            hits, _ = runner.call_endpoint("jevlin_fanout_merged", "q", {})
        self.assertEqual(len(hits), 10)
        self.assertEqual(len({h["url"] for h in hits}), 10)
        self.assertNotIn("https://www.example.com/p0/", [h["url"] for h in hits])
        self.assertEqual(hits[0]["snippet"], "x y z", "whitespace collapsed like every vendor")

    def test_snippet_cap_is_1200(self):
        payload = {**MERGED, "merged": [{"url": "https://e.com/a", "title": "t", "excerpt": "a" * 5000}]}
        with _env(), patch.object(runner, "_http", return_value=_ok(payload)):
            hits, _ = runner.call_endpoint("jevlin_fast", "q", {})
        self.assertEqual(len(hits[0]["snippet"]), 1200)

    def test_failed_call_yields_no_hits_and_an_error_meta(self):
        failed = {"ok": False, "status_code": 429, "response": {"error": "rate limit"}, "latency_ms": 9000, "error": "HTTP 429", "attempts": []}
        with _env(), patch.object(runner, "_http", return_value=failed):
            hits, raw = runner.call_endpoint("jevlin_balanced", "q", {})
        self.assertEqual(hits, [])
        self.assertEqual(raw["status_code"], 429)
        self.assertEqual(raw["vendor_meta"]["jevlin_error"], "HTTP 429")
        self.assertIsNone(raw["vendor_meta"]["jevlin_cost_micros"])


class Metadata(unittest.TestCase):
    def test_vendor_meta_carries_observed_cost_cache_pending_and_upstream_calls(self):
        with _env(), patch.object(runner, "_http", return_value=_ok()):
            _, raw = runner.call_endpoint("jevlin_balanced", "q", {})
        meta = raw["vendor_meta"]
        self.assertEqual(meta["jevlin_cost_micros"], 10000)
        self.assertFalse(meta["jevlin_cache_hit"])
        self.assertEqual(meta["jevlin_pending"], 1)
        self.assertEqual(meta["jevlin_upstream_calls"], 4, "skipped is the only not-called status")
        self.assertEqual(meta["jevlin_arms_ok"], 2)
        self.assertEqual(meta["jevlin_request_id"], "req_fixture_0001")
        self.assertEqual(meta["jevlin_tier"], "balanced")
        self.assertIn("f01b032", meta["jevlin_core_source"])
        statuses = {arm["provider"]: arm["status"] for arm in meta["jevlin_arms"]}
        self.assertEqual(statuses, {"exa": "ok", "brave": "ok", "serpapi": "pending", "tinyfish": "timeout", "you": "skipped"})

    def test_cost_event_fields_never_touch_usd(self):
        fields = jevlin_news.cost_event_fields({"jevlin_cost_micros": 10000, "jevlin_cache_hit": True, "jevlin_pending": 1,
                                                "jevlin_upstream_calls": 4, "jevlin_arms": [{"provider": "exa", "status": "ok"}]})
        self.assertEqual(fields["usd_observed"], 0.01)
        self.assertTrue(fields["cache_hit"])
        self.assertEqual(fields["upstream_statuses"], {"exa": "ok"})
        self.assertNotIn("usd", fields)
        self.assertIsNone(jevlin_news.cost_event_fields({})["usd_observed"])


class RunCaseWithoutExtractor(unittest.TestCase):
    def _run(self, endpoints):
        case = {"id": "cn-001", "question": "Q?", "expected_answer": "$1 million", "cells": [{"value": "$1 million"}], "ground_truth_url": "", "recipe": "financing"}

        def fake_call(name, question, _case):
            if name.startswith("jevlin"):
                return jevlin_news.call_jevlin(name, question, http=lambda **kw: _ok(), hit=runner._hit, dedupe=runner._dedupe, max_results=10)
            return [runner._hit("https://x.com/a", "t", "s")], _ok({"results": []})

        with _env(), patch.object(runner, "call_endpoint", side_effect=fake_call), tempfile.TemporaryDirectory() as tmp:
            row = runner.run_case(None, "none", case, endpoints, Path(tmp))
        return row

    def test_blocks_are_deferred_and_no_openai_event_is_written(self):
        row = self._run(["jevlin_balanced", "exa_fast"])
        for name in ("jevlin_balanced", "exa_fast"):
            block = row["vendors"][name]
            self.assertEqual(block["answer"], "")
            self.assertIsNone(block["gold"]["correct"])
            self.assertIn("deferred", block["gold"]["note"])
        self.assertEqual([e["kind"] for e in row["cost_events"]], ["vendor_search", "vendor_search"])

    def test_jevlin_cost_event_has_observed_fields_and_nominal_usd(self):
        row = self._run(["jevlin_balanced", "exa_fast"])
        events = {e["vendor"]: e for e in row["cost_events"]}
        self.assertEqual(events["jevlin_balanced"]["usd"], 0.0)
        self.assertEqual(events["jevlin_balanced"]["usd_observed"], 0.01)
        self.assertEqual(events["jevlin_balanced"]["upstream_calls"], 4)
        self.assertNotIn("usd_observed", events["exa_fast"])
        self.assertEqual(events["exa_fast"]["usd"], 0.007)
        self.assertIn("vendor_meta", row["vendors"]["jevlin_balanced"])
        self.assertNotIn("vendor_meta", row["vendors"]["exa_fast"], "other vendors' blocks keep their upstream shape")

    def test_cost_rollups_keep_observed_separate(self):
        row = self._run(["jevlin_balanced", "exa_fast"])
        roll = runner._cost_rollups([row])
        self.assertEqual(roll["by_endpoint"]["jevlin_balanced"]["usd"], 0.0)
        self.assertEqual(roll["by_endpoint"]["jevlin_balanced"]["usd_observed"], 0.01)
        self.assertEqual(roll["by_endpoint"]["jevlin_balanced"]["pending_calls"], 1)
        self.assertNotIn("usd_observed", roll["by_endpoint"]["exa_fast"])
        self.assertEqual(roll["total_usd"], 0.007, "observed spend is never summed into the flat total")


class Guards(unittest.TestCase):
    def test_family_failure_counter_aborts_on_a_storm(self):
        counts: dict[str, int] = {}
        failing = {"vendors": {"jevlin_fast": {"error": "HTTP 429"}, "exa_fast": {"error": None}}}
        self.assertIsNone(runner._update_family_failures(counts, failing, ["jevlin_fast", "exa_fast"], 2))
        self.assertEqual(counts, {"jevlin": 1, "exa_fast": 0})
        reason = runner._update_family_failures(counts, failing, ["jevlin_fast", "exa_fast"], 2)
        self.assertIn("jevlin", reason)
        healthy = {"vendors": {"jevlin_fast": {"error": None}, "exa_fast": {"error": None}}}
        runner._update_family_failures(counts, healthy, ["jevlin_fast", "exa_fast"], 2)
        self.assertEqual(counts["jevlin"], 0)

    def test_retryable_failure_classification(self):
        self.assertTrue(runner._retryable_failure({"error": "HTTP 429", "status_code": 429}))
        self.assertTrue(runner._retryable_failure({"error": "HTTP 503", "status_code": 503}))
        self.assertTrue(runner._retryable_failure({"error": "ConnectionError", "status_code": None}))
        self.assertFalse(runner._retryable_failure({"error": "HTTP 401", "status_code": 401}))
        self.assertFalse(runner._retryable_failure({"error": None, "status_code": 200}))

    def test_endpoint_family(self):
        self.assertEqual(runner.endpoint_family("jevlin_fanout_raw"), "jevlin")
        self.assertEqual(runner.endpoint_family("exa_fast"), "exa_fast")


if __name__ == "__main__":
    unittest.main()
