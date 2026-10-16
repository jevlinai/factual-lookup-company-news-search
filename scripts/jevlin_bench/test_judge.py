"""Layer A matching, Layer B through fake backends, the final policy, AR@K, citation validity, run/apply."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jevlin_bench import judge
from jevlin_bench.fake_claude import make_fake_claude, make_fake_codex, recorded_calls


class LayerA(unittest.TestCase):
    def assert_match(self, gold, answer, method=None):
        matched, how = judge.match_cell(gold, answer)
        self.assertTrue(matched, f"{gold!r} vs {answer!r} -> {how}")
        if method:
            self.assertEqual(how, method)

    def assert_no_match(self, gold, answer):
        matched, how = judge.match_cell(gold, answer)
        self.assertFalse(matched, f"{gold!r} vs {answer!r} matched by {how}")

    def test_exact_and_contains(self):
        self.assert_match("$31.5 million", "$31.5 million", "exact")
        self.assert_match("CEVIRA®", "Cevira", "exact")
        self.assert_match("Pilgrim's Europe", "Pilgrim’s Europe", "exact")
        self.assert_match("$31.5 million", "MaxQ raised $31.5 million in a Series B", "contains")
        self.assert_match("R1", "R1 RCM", "contains")

    def test_numeric_equivalents(self):
        for ans in ("$31.5M", "USD 31.5 million", "31.5 million dollars", "$31,500,000", "US$31.5m"):
            self.assert_match("$31.5 million", ans, "numeric")
        self.assert_match("$11 million", "eleven million dollars", "numeric")
        self.assert_match("$6 million", "6 million", "numeric")
        self.assert_match("Approximately 45 employees", "about 45 staff", "numeric")
        self.assert_match("Approximately 45 employees", "45", "numeric")
        self.assert_no_match("$31.5 million", "$30 million")
        self.assert_no_match("$31.5 million", "eleven million")
        self.assert_no_match("$400 million", "€400 million")
        self.assert_no_match("Approximately 45 employees", "450 employees")
        self.assert_no_match("15%", "15 million")

    def test_dates(self):
        self.assert_match("March 3, 2026", "3 March 2026", "date")
        self.assert_match("March 3, 2026", "2026-03-03", "date")
        self.assert_match("August 2026", "Aug 2026", "date")
        self.assert_no_match("March 3, 2026", "March 4, 2026")
        self.assert_no_match("March 3, 2026", "March 3, 2025")

    def test_aliases_names_and_suffixes(self):
        self.assert_match("Dr. Rama Kondru", "Rama Kondru", "alias")
        self.assert_match("Jonathan M. Pratt", "Jonathan Pratt", "alias")
        self.assert_match("William E. Evans, Pharm.D.", "William Evans", "alias")
        self.assert_match('Wei "Dimi" Du', "Wei Du", "alias")
        self.assert_match("Chao-Yang Wang, Ph.D.", "Chao-Yang Wang", "alias")
        self.assert_match("Angel S. Angelov, M.D., MBA", "Dr. Angel Angelov", "alias")
        self.assert_match("Nayax Ltd.", "Nayax", "alias")
        self.assert_match("TruGolf Holdings, Inc.", "TruGolf", "alias")
        self.assert_match("Nutanix Enterprise AI (NAI) 2.8", "Nutanix Enterprise AI 2.8", "alias")
        self.assert_match("Climate Technology Fund II (CTF II)", "Climate Technology Fund II", "alias")
        self.assert_no_match("Nutanix Enterprise AI (NAI) 2.8", "Nutanix Enterprise AI 2.7")
        self.assert_no_match("Steve Nylund", "Nylund")
        self.assert_no_match("Steve Nylund", "Michael Nylund")
        self.assert_no_match("CVC DIF", "CVC")
        self.assert_no_match("Peter Hall", "Peter")

    def test_verdict_empty_and_multi_cell(self):
        empty = judge.layer_a_verdict("", [{"value": "$1"}])
        self.assertFalse(empty["correct"])
        self.assertEqual(empty["note"], "empty answer")
        both = judge.layer_a_verdict("Yaron Eldad was named CFO after the $12M round.", [{"value": "Yaron Eldad"}, {"value": "$12 million"}])
        self.assertTrue(both["correct"])
        self.assertEqual(both["cells_hit"], 2)
        one = judge.layer_a_verdict("Yaron Eldad was named CFO.", [{"value": "Yaron Eldad"}, {"value": "$12 million"}])
        self.assertFalse(one["correct"])
        self.assertEqual(one["cells_hit"], 1)
        fallback = judge.layer_a_verdict("$5 million", [], expected="$5 million")
        self.assertTrue(fallback["correct"], "expected_answer stands in when no cells exist")

    def test_bearing_ranks_and_flags(self):
        hits = [{"title": "A", "snippet": "nothing"}, {"title": "B", "snippet": "raised $31.5M today"}, {"title": "C", "snippet": "$31.5 million round"}]
        ranks = judge.layer_a_bearing_ranks(hits, [{"value": "$31.5 million"}])
        self.assertEqual(ranks, [2, 3])
        self.assertEqual(judge.ar_flags(ranks), {"ar1": False, "ar5": True, "ar10": True})
        self.assertEqual(judge.layer_a_bearing_ranks(hits, []), [])

    def test_citation_valid(self):
        hits = [{"url": "https://www.example.com/a/"}, {"url": "https://b.com/x?y=1"}]
        self.assertTrue(judge.citation_valid("https://example.com/a", hits))
        self.assertTrue(judge.citation_valid("http://b.com/x", hits))
        self.assertFalse(judge.citation_valid("https://c.com/z", hits))
        self.assertIsNone(judge.citation_valid("", hits))


class LayerBAndPolicy(unittest.TestCase):
    def _judge(self, answer, backend, cells=None):
        with tempfile.TemporaryDirectory() as tmp:
            return judge.judge_answer(question="Q?", expected="$31.5 million", cells=cells or [{"value": "$31.5 million"}], answer=answer,
                                     backend=backend, workdir=Path(tmp))

    def test_layer_b_primary_and_agreement(self):
        fake = judge.FakeBackend({"verdict": "correct", "cells_hit": 1, "justification": "same amount"})
        out = self._judge("thirty-one point five million dollars", fake)
        self.assertEqual(out["policy"], "layer_b_primary")
        self.assertTrue(out["correct"])
        self.assertFalse(out["layer_a"]["correct"], "Layer A cannot parse this paraphrase")
        self.assertFalse(out["agreement"])
        self.assertEqual(out["note"], "same amount")
        fake2 = judge.FakeBackend({"verdict": "incorrect", "cells_hit": 0, "justification": "different amount"})
        out2 = self._judge("$31.5 million", fake2)
        self.assertFalse(out2["correct"], "B is primary even when A matched; the disagreement is recorded")
        self.assertFalse(out2["agreement"])

    def test_layer_b_failure_falls_back_to_a_and_is_flagged(self):
        out = self._judge("$31.5 million", judge.FakeBackend(fail=True))
        self.assertTrue(out["correct"])
        self.assertTrue(out["judge_failed"])
        self.assertIsNone(out["agreement"])
        self.assertFalse(out["layer_b"]["ok"])
        invalid = self._judge("$31.5 million", judge.FakeBackend({"verdict": "maybe"}))
        self.assertTrue(invalid["judge_failed"])
        self.assertIn("verdict", invalid["layer_b"]["error"])

    def test_no_backend_is_layer_a_only(self):
        out = self._judge("$31.5 million", None)
        self.assertEqual(out["policy"], "layer_a_only")
        self.assertIsNone(out["layer_b"])
        self.assertTrue(out["correct"])

    def test_empty_answer_never_reaches_the_backend(self):
        fake = judge.FakeBackend({"verdict": "correct", "cells_hit": 1, "justification": "x"})
        out = self._judge("", fake)
        self.assertFalse(out["correct"])
        self.assertEqual(fake.calls, [])
        self.assertEqual(out["layer_b"]["skipped"], "empty answer")

    def test_backend_is_blind_to_provider_and_snippets(self):
        fake = judge.FakeBackend({"verdict": "correct", "cells_hit": 1, "justification": "x"})
        self._judge("$31.5 million", fake)
        payload = json.loads(fake.calls[0]["user_prompt"])
        self.assertEqual(set(payload), {"question", "gold_answer", "required_facts", "extracted_answer"})
        self.assertEqual(fake.calls[0]["system_prompt"], judge.ACCURACY_RUBRIC)
        self.assertEqual(fake.calls[0]["schema"], judge.ACCURACY_SCHEMA)
        self.assertNotIn("jevlin", json.dumps(fake.calls[0]).lower())

    def test_rubric_is_frozen_by_hash(self):
        self.assertEqual(judge.RUBRIC_SHA256, judge.sha256_text(judge.ACCURACY_RUBRIC))
        self.assertIn("Do not use outside knowledge", judge.ACCURACY_RUBRIC)
        self.assertIn("not told which search provider", judge.ACCURACY_RUBRIC)

    def test_semantic_ar_overrides_deterministic_when_valid(self):
        hits = [{"url": "u1", "title": "A", "snippet": "nothing"}, {"url": "u2", "title": "B", "snippet": "thirty-one point five million"}]
        fake = judge.FakeBackend({"bearing_ranks": [2, 99], "note": "paraphrase"})
        with tempfile.TemporaryDirectory() as tmp:
            out = judge.judge_ar(question="Q", expected="$31.5 million", cells=[{"value": "$31.5 million"}], hits=hits, backend=fake, workdir=Path(tmp))
        self.assertEqual(out["layer_a_ranks"], [])
        self.assertEqual(out["layer_b_ranks"], [2], "out-of-range ranks dropped")
        self.assertEqual(out["source"], "layer_b")
        self.assertTrue(out["ar5"])
        self.assertFalse(out["ar1"])
        payload = json.loads(fake.calls[0]["user_prompt"])
        self.assertEqual(set(payload), {"question", "gold_answer", "required_facts", "snippets"})


class Backends(unittest.TestCase):
    def test_codex_argv_shape_and_clean_env(self):
        backend = judge.CodexBackend(codex_bin="/usr/local/bin/codex", model="gpt-x", effort="low")
        argv = backend.argv(prompt="P", schema_path=Path("/t/schema.json"), out_path=Path("/t/out.json"), workdir=Path("/t"))
        self.assertEqual(argv[:3], ["/usr/local/bin/codex", "exec", "--json"])
        for flag in ("--skip-git-repo-check", "--sandbox", "--output-schema", "--output-last-message", "-C", "-m",
                     "--ignore-user-config", "--ignore-rules", "--ephemeral"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-x")
        self.assertIn("model_reasoning_effort=low", argv)
        self.assertIn("web_search=disabled", argv)
        disabled = [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"]
        self.assertEqual(set(disabled), {"apps", "browser_use", "computer_use", "hooks", "code_mode_host"})
        self.assertEqual(argv[-1], "P")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-leak", "JEVLIN_API_KEY": "j"}):
            self.assertEqual(judge.cc_isolated.forbidden_env_keys(judge.CodexBackend.env()), [])

    def test_codex_defaults_pin_model_and_effort(self):
        with patch.dict(os.environ, {"CODEX_JUDGE_MODEL": "", "CODEX_JUDGE_EFFORT": ""}):
            backend = judge.CodexBackend()
        self.assertEqual((backend.model, backend.effort), (judge.CODEX_DEFAULT_MODEL, judge.CODEX_DEFAULT_EFFORT))
        with patch.dict(os.environ, {"CODEX_JUDGE_MODEL": "gpt-env", "CODEX_JUDGE_EFFORT": "high"}):
            backend = judge.CodexBackend()
        self.assertEqual((backend.model, backend.effort), ("gpt-env", "high"))

    def test_codex_event_stream_is_summarised(self):
        stream = "\n".join([
            "codex-cli banner line",
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}),
            json.dumps({"type": "item.completed", "item": {"type": "error", "message": "Code Mode is unavailable"}}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"verdict": "correct"}'}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 1}}),
        ])
        summary = judge.CodexBackend.read_events(stream)
        self.assertEqual(summary["items"], {"command_execution": 1, "error": 1, "agent_message": 1})
        self.assertEqual(summary["breaches"], {"command_execution": 1})
        self.assertEqual(summary["errors"], ["Code Mode is unavailable"])
        self.assertEqual(summary["usage"], {"input_tokens": 7, "output_tokens": 1})
        self.assertEqual(summary["last_text"], '{"verdict": "correct"}')

    def test_codex_tool_use_fails_the_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = judge.CodexBackend(codex_bin=str(make_fake_codex(Path(tmp) / "bin", "web")))
            out = judge.judge_answer(question="Q", expected="x", cells=[{"value": "x"}], answer="x", backend=backend, workdir=Path(tmp) / "w")
            self.assertFalse(out["layer_b"]["ok"])
            self.assertIn("isolation breach", out["layer_b"]["error"])
            self.assertIn("web_search", out["layer_b"]["error"])
            self.assertTrue(out["judge_failed"])
            self.assertEqual(out["layer_b"]["meta"]["tool_items"], 1)

    def test_codex_backend_against_fake(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = make_fake_codex(Path(tmp) / "bin", "ok")
            backend = judge.CodexBackend(codex_bin=str(fake))
            ok, status = judge.CodexBackend.available(str(fake))
            self.assertTrue(ok)
            self.assertIn("fake", status)
            out = judge.judge_answer(question="Q", expected="$31.5 million", cells=[{"value": "$31.5 million"}], answer="$31.5 million",
                                     backend=backend, workdir=Path(tmp) / "w")
            self.assertTrue(out["layer_b"]["ok"])
            self.assertTrue(out["correct"])
            self.assertEqual(out["layer_b"]["family"], "openai")
            self.assertTrue(out["layer_b"]["independent_of_extractor"])
            call = recorded_calls(Path(tmp) / "bin")[0]
            self.assertIn("--output-last-message", call["argv"])
            self.assertTrue(call["stdin_is_devnull"], "codex exec blocks on an inherited non-TTY stdin")
            self.assertEqual(judge.cc_isolated.forbidden_env_keys(call["env"]), [])
            self.assertEqual(out["layer_b"]["meta"]["tool_items"], 0)
            self.assertEqual(out["layer_b"]["meta"]["usage"]["input_tokens"], 1000)
            self.assertIn("fake", out["layer_b"]["meta"]["codex_version"])
            self.assertIn("INPUT:", call["argv"][-1])
            self.assertTrue(call["argv"][-1].startswith(judge.ACCURACY_RUBRIC[:40]))
            garbage = judge.CodexBackend(codex_bin=str(make_fake_codex(Path(tmp) / "bin2", "garbage")))
            out2 = judge.judge_answer(question="Q", expected="x", cells=[{"value": "x"}], answer="x", backend=garbage, workdir=Path(tmp) / "w2")
            self.assertTrue(out2["judge_failed"])
            self.assertTrue(out2["correct"], "A decides when B fails")

    def test_codex_unavailable_is_reported_not_raised(self):
        ok, status = judge.CodexBackend.available("/nonexistent/codex-binary")
        self.assertFalse(ok)
        self.assertTrue(status)

    def test_claude_code_backend_is_labelled_dependent(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = make_fake_claude(Path(tmp) / "bin", "ok")
            backend = judge.ClaudeCodeBackend(claude_bin=str(fake))
            out = judge.judge_answer(question="Q", expected="$31.5 million", cells=[{"value": "$31.5 million"}], answer="$31.5 million",
                                     backend=backend, workdir=Path(tmp) / "w")
            self.assertTrue(out["layer_b"]["ok"])
            self.assertEqual(out["layer_b"]["family"], "anthropic")
            self.assertFalse(out["layer_b"]["independent_of_extractor"])
            call = recorded_calls(Path(tmp) / "bin")[0]
            self.assertEqual(call["argv"][call["argv"].index("--tools") + 1], "")
            self.assertEqual(call["argv"][call["argv"].index("--system-prompt") + 1], judge.ACCURACY_RUBRIC)
            self.assertEqual(judge.cc_isolated.forbidden_env_keys(call["env"]), [])

    def test_make_backend(self):
        self.assertIsNone(judge.make_backend("none"))
        self.assertIsInstance(judge.make_backend("codex"), judge.CodexBackend)
        self.assertIsInstance(judge.make_backend("claude-code"), judge.ClaudeCodeBackend)
        self.assertIsInstance(judge.make_backend("fake"), judge.FakeBackend)
        with self.assertRaises(ValueError):
            judge.make_backend("bedrock")


class RunAndApply(unittest.TestCase):
    def test_run_apply_over_a_small_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            hits = [{"url": "https://e.com/a", "title": "A", "snippet": "MaxQ raised $31.5 million"}]
            payload = {"endpoints": ["jevlin_fast", "exa_fast"], "cases": [{
                "id": "cn-001", "question": "Q?", "expected_answer": "$31.5 million", "cells": [{"value": "$31.5 million"}],
                "vendors": {
                    "jevlin_fast": {"error": None, "hits": hits, "answer": "$31.5M", "cited_url": "https://e.com/a", "extract": {"status": "ok"}, "gold": {"correct": None}},
                    "exa_fast": {"error": None, "hits": hits, "answer": "", "cited_url": "", "extract": {"status": "ok"}, "gold": {"correct": None}},
                }, "gold_scores": {}}]}
            (run_dir / "run.json").write_text(json.dumps(payload))
            samples = Path(tmp) / "samples.json"
            samples.write_text(json.dumps([{"id": "cn-001", "expected_answer": "$31.5 million", "cells": [{"value": "$31.5 million"}]}]))
            with redirect_stdout(io.StringIO()):
                rc = judge.main(["--run-dir", str(run_dir), "--samples", str(samples), "--layer-b", "fake", "--run", "--apply"])
            self.assertEqual(rc, 0)
            out = json.loads((run_dir / "run.json").read_text())
            jev = out["cases"][0]["vendors"]["jevlin_fast"]
            exa = out["cases"][0]["vendors"]["exa_fast"]
            self.assertTrue(jev["gold"]["layer_a"]["correct"])
            self.assertIn("layer_b", jev["gold"])
            self.assertTrue(jev["citation_valid"])
            self.assertTrue(jev["ar"]["ar1"])
            self.assertTrue(jev["answer_in_excerpt"])
            self.assertFalse(exa["gold"]["correct"])
            self.assertEqual(exa["gold"]["layer_a"]["note"], "empty answer")
            self.assertEqual(out["judge"]["layer_b"], "fake")
            self.assertEqual(out["cases"][0]["gold_scores"]["exa_fast"]["correct"], False)
            self.assertTrue((run_dir / "run.before-judge.json").exists())

    def test_codex_not_operational_stops_unless_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(json.dumps({"endpoints": [], "cases": []}))
            with redirect_stdout(io.StringIO()):
                rc = judge.main(["--run-dir", str(run_dir), "--samples", str(Path(tmp) / "none.json"), "--layer-b", "codex", "--codex-bin", "/nonexistent/codex", "--run"])
            self.assertEqual(rc, 2)
            fake = make_fake_claude(Path(tmp) / "bin", "ok")
            with redirect_stdout(io.StringIO()):
                rc = judge.main(["--run-dir", str(run_dir), "--samples", str(Path(tmp) / "none.json"), "--layer-b", "codex", "--codex-bin", "/nonexistent/codex",
                                 "--claude-bin", str(fake), "--allow-fallback", "--run", "--apply"])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads((run_dir / "run.json").read_text())["judge"]["layer_b"], "claude-code")


if __name__ == "__main__":
    unittest.main()
