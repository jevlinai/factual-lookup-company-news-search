"""The extractor runner against a fake `claude`: isolation flags, clean environment, prompt contents, statuses, apply."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jevlin_bench import cc_extract, cc_isolated
from jevlin_bench.fake_claude import make_fake_claude, recorded_calls

HITS = [
    {"url": "https://example.com/maxq", "title": "MaxQ raises", "snippet": "MaxQ Medical raised $31.5 million.", "host": "example.com"},
    {"url": "https://example.com/other", "title": "Other", "snippet": "x" * 2000, "host": "example.com"},
]


def _run_json(tmp: Path) -> Path:
    run_dir = tmp / "run"
    (run_dir / "raw_calls").mkdir(parents=True)
    payload = {
        "protocol": "official_atomic_single_query", "model": None, "endpoints": ["jevlin_fast", "exa_fast"], "cost": {},
        "cases": [
            {"id": "cn-001", "question": "How much did MaxQ Medical raise?", "expected_answer": "$31.5 million", "cells": [{"value": "$31.5 million"}],
             "vendors": {
                 "jevlin_fast": {"endpoint": "jevlin_fast", "error": None, "hits": HITS, "answer": "", "cited_url": "", "gold": {"correct": None}},
                 "exa_fast": {"endpoint": "exa_fast", "error": "HTTP 429", "hits": [], "answer": "", "cited_url": "", "gold": {"correct": None}},
             }, "gold_scores": {}, "cost_events": []},
            {"id": "cn-002", "question": "Who became CEO?", "expected_answer": "Peter Hall", "cells": [{"value": "Peter Hall"}],
             "vendors": {
                 "jevlin_fast": {"endpoint": "jevlin_fast", "error": None, "hits": [], "answer": "", "cited_url": "", "gold": {"correct": None}},
                 "exa_fast": {"endpoint": "exa_fast", "error": None, "hits": HITS[:1], "answer": "", "cited_url": "", "gold": {"correct": None}},
             }, "gold_scores": {}, "cost_events": []},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(payload))
    return run_dir


class Argv(unittest.TestCase):
    def test_isolation_flags_present(self):
        argv = cc_isolated.claude_argv("claude", prompt="p", system_prompt="s", schema=cc_extract.SCHEMA, model="m", isolation="both")
        joined = " ".join(argv)
        for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--json-schema",
                     "--system-prompt", "--permission-mode dontAsk", "--output-format json", "--model m", "--disallowedTools"):
            self.assertIn(flag, joined)
        self.assertIn("--tools", argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "", '`--tools ""` is the flag that removes every built-in tool')
        self.assertNotIn("--append-system-prompt", argv, "the instruction replaces the agent prompt; nothing is appended to it")
        self.assertNotIn("--bare", argv, "--bare would force API-key auth")

    def test_single_isolation_forms(self):
        only_tools = cc_isolated.claude_argv("claude", prompt="p", system_prompt="s", schema={}, isolation="tools-empty")
        self.assertIn("--tools", only_tools)
        self.assertNotIn("--disallowedTools", only_tools)
        only_deny = cc_isolated.claude_argv("claude", prompt="p", system_prompt="s", schema={}, isolation="disallowed")
        self.assertNotIn("--tools", only_deny)
        self.assertIn("--disallowedTools", only_deny)
        with self.assertRaises(ValueError):
            cc_isolated.claude_argv("claude", prompt="p", system_prompt="s", schema={}, isolation="nope")

    def test_stream_json_adds_verbose(self):
        argv = cc_isolated.claude_argv("claude", prompt="p", system_prompt="s", schema={}, output_format="stream-json")
        self.assertIn("--verbose", argv)


class Environment(unittest.TestCase):
    def test_clean_env_strips_keys_and_markers(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-x", "JEVLIN_API_KEY": "j", "JEVLIN_PROVIDERS": "a,b", "CLAUDECODE": "1",
                                     "CLAUDE_CODE_ENTRYPOINT": "cli", "OPENAI_API_KEY": "o", "AWS_ACCESS_KEY_ID": "a", "EXA_API_KEY": "e"}):
            env = cc_isolated.clean_env("claude")
        self.assertEqual(cc_isolated.forbidden_env_keys(env), [])
        self.assertEqual(set(env) - {"HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "TMPDIR", "PATH", "DISABLE_AUTOUPDATER"}, set())
        self.assertIn("/usr/bin", env["PATH"])

    def test_forbidden_detector(self):
        self.assertEqual(cc_isolated.forbidden_env_keys({"ANTHROPIC_API_KEY": "x", "JEVLIN_API_URL": "y", "CLAUDECODE": "1", "HOME": "h"}),
                         ["ANTHROPIC_API_KEY", "CLAUDECODE", "JEVLIN_API_URL"])


class Parsing(unittest.TestCase):
    def test_structured_output_preferred_then_text(self):
        self.assertEqual(cc_isolated.extract_structured({"structured_output": {"a": 1}, "result": "{\"a\": 2}"}), {"a": 1})
        self.assertEqual(cc_isolated.extract_structured({"result": "```json\n{\"a\": 2}\n```"}), {"a": 2})
        self.assertEqual(cc_isolated.extract_structured({"result": "prose {\"a\": 3} trailing"}), {"a": 3})
        self.assertIsNone(cc_isolated.extract_structured({"result": "nothing"}))

    def test_tool_use_counting_excludes_structured_output_channel(self):
        events = [{"type": "system", "tools": ["StructuredOutput"]},
                  {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "StructuredOutput"}]}},
                  {"type": "result"}]
        self.assertEqual(cc_isolated.count_tool_uses(events), 0)
        self.assertEqual(cc_isolated.tool_use_names(events), ["StructuredOutput"])
        self.assertEqual(cc_isolated.available_tools(events), ["StructuredOutput"])
        events[1]["message"]["content"].insert(0, {"type": "tool_use", "name": "WebSearch"})
        self.assertEqual(cc_isolated.count_tool_uses(events), 1)
        self.assertIsNone(cc_isolated.count_tool_uses([{"type": "result"}]), "plain json output has no assistant events")

    def test_validate_follows_upstream_extractout_defaults(self):
        self.assertEqual(cc_extract.validate({"answer": "x"}), (True, {"answer": "x", "cited_url": ""}, None))
        self.assertEqual(cc_extract.validate({"answer": " x ", "cited_url": None}), (True, {"answer": "x", "cited_url": ""}, None))
        self.assertFalse(cc_extract.validate({"answer": 31.5, "cited_url": ""})[0])
        self.assertFalse(cc_extract.validate({"answer": "x", "cited_url": "", "extra": 1})[0])
        self.assertFalse(cc_extract.validate(None)[0])

    def test_pack_hits_is_url_title_snippet_only_capped(self):
        packed = cc_extract.pack_hits(HITS + [dict(HITS[0])] * 12)
        self.assertEqual(len(packed), 10)
        self.assertEqual(set(packed[0]), {"url", "title", "snippet"})
        self.assertEqual(len(packed[1]["snippet"]), 1200)


class EndToEnd(unittest.TestCase):
    def _main(self, run_dir: Path, fake: Path, *extra: str) -> int:
        with redirect_stdout(io.StringIO()):
            return cc_extract.main(["--run-dir", str(run_dir), "--run", "--apply", "--claude-bin", str(fake), "--rate-limit-wait", "0", *extra])

    def test_run_and_apply_against_fake(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-leak", "JEVLIN_API_KEY": "jk", "CLAUDECODE": "1"}):
            tmp_path = Path(tmp)
            run_dir = _run_json(tmp_path)
            fake = make_fake_claude(tmp_path / "bin", "ok")
            self.assertEqual(self._main(run_dir, fake), 0)
            calls = recorded_calls(tmp_path / "bin")
            self.assertEqual(len(calls), 2, "one session per (case, configuration) with hits; skipped jobs make no call")
            for call in calls:
                argv = call["argv"]
                self.assertIn("--safe-mode", argv)
                self.assertEqual(argv[argv.index("--tools") + 1], "")
                self.assertEqual(argv[argv.index("--model") + 1], cc_extract.DEFAULT_MODEL)
                self.assertEqual(argv[argv.index("--system-prompt") + 1], cc_extract.INSTRUCTION)
                self.assertEqual(json.loads(argv[argv.index("--json-schema") + 1]), cc_extract.SCHEMA)
                self.assertEqual(cc_isolated.forbidden_env_keys(call["env"]), [], "no key or marker reaches the session")
                prompt = json.loads(argv[argv.index("-p") + 1])
                self.assertEqual(set(prompt), {"question", "results"})
                self.assertNotIn("$31.5 million", prompt["question"])
                for hit in prompt["results"]:
                    self.assertEqual(set(hit), {"url", "title", "snippet"})
                self.assertNotIn("expected", json.dumps(prompt))
                self.assertTrue(str(Path(call["cwd"]).resolve()).startswith(str(run_dir.resolve())),
                                "an empty per-job working directory, no CLAUDE.md to find")
            rows = {p.stem: json.loads(p.read_text()) for p in (run_dir / "cc-extract").glob("*.json")}
            self.assertEqual(rows["cn-001__jevlin_fast"]["status"], "ok")
            self.assertEqual(rows["cn-001__jevlin_fast"]["answer"], "$31.5 million")
            self.assertEqual(rows["cn-001__jevlin_fast"]["cited_url"], "https://example.com/maxq")
            self.assertEqual(rows["cn-001__exa_fast"]["status"], "skipped")
            self.assertEqual(rows["cn-001__exa_fast"]["skipped"], "search error")
            self.assertEqual(rows["cn-002__jevlin_fast"]["skipped"], "no hits")
            self.assertEqual(rows["cn-002__exa_fast"]["status"], "ok")
            self.assertEqual(rows["cn-001__jevlin_fast"]["claude"]["num_turns"], 2)
            payload = json.loads((run_dir / "run.json").read_text())
            block = payload["cases"][0]["vendors"]["jevlin_fast"]
            self.assertEqual(block["answer"], "$31.5 million")
            self.assertEqual(block["extract"]["status"], "ok")
            self.assertEqual(block["extract"]["kind"], "claude-code")
            self.assertIsNone(block["gold"]["correct"], "the judge decides gold")
            self.assertEqual(payload["extractor"]["kind"], "claude-code")
            self.assertEqual(payload["model"], cc_extract.DEFAULT_MODEL)
            self.assertTrue((run_dir / "run.before-cc-extract.json").exists())
            # Rerun resumes from checkpoints: no new sessions.
            self.assertEqual(self._main(run_dir, fake), 0)
            self.assertEqual(len(recorded_calls(tmp_path / "bin")), 2)

    def test_invalid_output_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_json(Path(tmp))
            fake = make_fake_claude(Path(tmp) / "bin", "invalid")
            self.assertEqual(self._main(run_dir, fake), 0)
            row = json.loads((run_dir / "cc-extract" / "cn-001__jevlin_fast.json").read_text())
            self.assertEqual(row["status"], "invalid")
            self.assertEqual(row["answer"], "")
            self.assertEqual(row["attempts"], 1)
            self.assertFalse(row["schema_valid"])
            self.assertEqual(len(recorded_calls(Path(tmp) / "bin")), 2)
            block = json.loads((run_dir / "run.json").read_text())["cases"][0]["vendors"]["jevlin_fast"]
            self.assertEqual(block["extract"]["status"], "invalid")
            self.assertEqual(block["answer"], "")

    def test_garbage_output_is_a_failed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_json(Path(tmp))
            fake = make_fake_claude(Path(tmp) / "bin", "garbage")
            self.assertEqual(self._main(run_dir, fake), 0)
            row = json.loads((run_dir / "cc-extract" / "cn-001__jevlin_fast.json").read_text())
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["answer"], "")

    def test_rate_limit_waits_and_reruns_same_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_json(Path(tmp))
            fake = make_fake_claude(Path(tmp) / "bin", "rate-limit-first")
            self.assertEqual(self._main(run_dir, fake, "--ids", "cn-001", "--endpoints", "jevlin_fast"), 0)
            row = json.loads((run_dir / "cc-extract" / "cn-001__jevlin_fast.json").read_text())
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["attempts"], 2)
            self.assertEqual(len(recorded_calls(Path(tmp) / "bin")), 2)

    def test_consecutive_failures_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_json(Path(tmp))
            fake = make_fake_claude(Path(tmp) / "bin", "crash")
            with redirect_stdout(io.StringIO()):
                rc = cc_extract.main(["--run-dir", str(run_dir), "--run", "--claude-bin", str(fake), "--abort-after-failures", "1"])
            self.assertEqual(rc, 3)

    def test_stream_json_records_tool_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _run_json(Path(tmp))
            fake = make_fake_claude(Path(tmp) / "bin", "ok")
            self.assertEqual(self._main(run_dir, fake, "--output-format", "stream-json"), 0)
            row = json.loads((run_dir / "cc-extract" / "cn-001__jevlin_fast.json").read_text())
            self.assertEqual(row["claude"]["tool_uses"], 0)
            self.assertEqual(row["claude"]["available_tools"], ["StructuredOutput"])
            self.assertEqual(row["claude"]["structured_output_calls"], 1)
            fake2 = make_fake_claude(Path(tmp) / "bin2", "tool-user")
            run2 = _run_json(Path(tmp) / "two")
            self.assertEqual(self._main(run2, fake2, "--output-format", "stream-json"), 0)
            row2 = json.loads((run2 / "cc-extract" / "cn-001__jevlin_fast.json").read_text())
            self.assertEqual(row2["claude"]["tool_uses"], 1, "an external tool call would be visible in the row")
            self.assertIn("WebSearch", row2["claude"]["available_tools"])


if __name__ == "__main__":
    unittest.main()
