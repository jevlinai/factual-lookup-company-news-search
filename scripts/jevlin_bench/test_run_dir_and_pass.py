"""Run-dir parametrisation of the official chain, and the cache-aware pass planner (no subprocesses)."""

from __future__ import annotations

import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from websearch import rejudge_official_ar_opus as ar_opus
from websearch import rejudge_official_extract_and_gold as extract_gold
from websearch import rejudge_official_gold_opus as gold_opus
from websearch import score_official_search_metrics as score

ROOT = Path(__file__).resolve().parents[2]


def _load_pass():
    import sys

    spec = importlib.util.spec_from_file_location("news_pass", ROOT / "experiments" / "news_pass.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


class RunDir(unittest.TestCase):
    def test_official_stamp_is_still_the_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEWS_RUN_DIR", None)
            self.assertEqual(score.resolve_run_dir(None).name, "20260816T020806Z")
        for module in (extract_gold, gold_opus, ar_opus):
            self.assertEqual(module.DEFAULT_RUN_DIR.name, "20260816T020806Z")

    def test_env_and_flag_override(self):
        with patch.dict(os.environ, {"NEWS_RUN_DIR": "/tmp/env-run"}):
            self.assertEqual(score.resolve_run_dir(None), Path("/tmp/env-run"))
            self.assertEqual(score.resolve_run_dir("/tmp/flag-run"), Path("/tmp/flag-run"))
        for module, ckpt in ((extract_gold, "rejudge-gpt56"), (gold_opus, "rejudge-opus5"), (ar_opus, "rejudge-ar-opus5")):
            with self.subTest(module=module.__name__):
                module.set_run_dir("/tmp/other-run")
                self.assertEqual(module.RUN_DIR, Path("/tmp/other-run"))
                self.assertEqual(module.RUN_PATH, Path("/tmp/other-run/run.json"))
                self.assertEqual(module.CKPT_DIR, Path("/tmp/other-run") / ckpt)
                module.set_run_dir(module.DEFAULT_RUN_DIR)

    def test_chain_models_unchanged(self):
        self.assertEqual(extract_gold.EXTRACT_MODEL, "gpt-5.6-terra")
        self.assertEqual(extract_gold.JUDGE_MODEL, "gpt-5.6-sol")
        self.assertEqual(gold_opus.JUDGE_LABEL, "claude-opus-5")
        self.assertEqual(ar_opus.JUDGE_LABEL, "claude-opus-5")


class Planner(unittest.TestCase):
    def setUp(self):
        self.mod = _load_pass()

    def test_default_plan_separates_same_key_arms(self):
        plan = self.mod.plan_invocations(list(self.mod.DEFAULT_DIRECT), list(self.mod.DEFAULT_JEVLIN), cases=100, case_gap_s=1.0)
        labels = [inv.endpoints for inv in plan]
        self.assertEqual(labels, [["jevlin_balanced", "exa_fast"], ["jevlin_fast", "perplexity_low"], ["jevlin_single", "parallel_fast"],
                                  ["jevlin_fanout_raw"], ["jevlin_fanout_merged"]])
        self.assertEqual([inv.key_class for inv in plan], ["balanced", "fast", "single", "balanced", "balanced"])
        self.assertEqual([inv.wait_for_class for inv in plan], [None, None, None, "balanced", "balanced"])
        for inv in plan:
            self.assertLessEqual(sum(1 for e in inv.endpoints if e.startswith("jevlin_")), 1)
        self.assertGreater(plan[0].estimate_s, plan[1].estimate_s, "balanced is the slow tier")

    def test_leftover_direct_arms_get_their_own_invocation(self):
        plan = self.mod.plan_invocations(["exa_fast", "perplexity_low"], ["jevlin_fast"], cases=3, case_gap_s=0.4)
        self.assertEqual([inv.endpoints for inv in plan], [["jevlin_fast", "exa_fast"], ["perplexity_low"]])
        self.assertIsNone(plan[1].key_class)

    def test_two_same_key_arms_in_one_invocation_is_refused(self):
        with self.assertRaises(SystemExit):
            self.mod.check_plan([self.mod.Invocation(endpoints=["jevlin_balanced", "jevlin_fanout_raw"], key_class="balanced")])
        with self.assertRaises(SystemExit):
            self.mod.check_plan([self.mod.Invocation(endpoints=["jevlin_balanced", "jevlin_fast"], key_class="balanced")])

    def test_wait_arithmetic(self):
        now = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
        state = {"invocations": [{"label": "jevlin_balanced+exa_fast", "key_class": "balanced", "ended_at": (now - timedelta(minutes=10)).isoformat(), "returncode": 0}]}
        inv = self.mod.Invocation(endpoints=["jevlin_fanout_raw"], key_class="balanced", wait_for_class="balanced")
        self.assertAlmostEqual(self.mod.wait_needed_s(state, inv, 16, now), 6 * 60, delta=1)
        self.assertEqual(self.mod.wait_needed_s(state, inv, 5, now), 0.0)
        fresh = self.mod.Invocation(endpoints=["jevlin_fast"], key_class="fast")
        self.assertEqual(self.mod.wait_needed_s(state, fresh, 16, now), 0.0)
        self.assertEqual(self.mod.wait_needed_s({"invocations": []}, inv, 16, now), 0.0)

    def test_runner_argv_fresh_then_attach(self):
        class Args:
            case_gap_s = 1.0
            abort_after_search_errors = 12
            limit = 3
            ids = ""
            samples = None
            retry_errors = True

        inv = self.mod.Invocation(endpoints=["jevlin_balanced", "exa_fast"], key_class="balanced")
        fresh = self.mod.runner_argv("py", inv, run_dir=Path("/r"), fresh=True, args=Args)
        self.assertIn("--run-dir", fresh)
        self.assertNotIn("--attach", fresh)
        self.assertEqual(fresh[fresh.index("--extractor") + 1], "none", "no OpenAI call anywhere")
        self.assertEqual(fresh[fresh.index("--endpoints") + 1], "jevlin_balanced,exa_fast")
        self.assertEqual(fresh[fresh.index("--limit") + 1], "3")
        attach = self.mod.runner_argv("py", inv, run_dir=Path("/r"), fresh=False, args=Args)
        self.assertIn("--attach", attach)
        self.assertIn("--attach-retry-errors", attach)
        self.assertNotIn("--run-dir", attach)

    def test_dry_run_prints_commands_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples.json"
            samples.write_text("[{}, {}, {}]")
            buf = io.StringIO()
            with redirect_stdout(buf), patch("subprocess.run") as run:
                rc = self.mod.main(["--run-dir", str(Path(tmp) / "run"), "--samples", str(samples), "--dry-run", "--cache-gap-min", "16"])
            self.assertEqual(rc, 0)
            run.assert_not_called()
            out = buf.getvalue()
            self.assertIn("5 invocations over 3 cases", out)
            self.assertIn("catalog_freeze.py verify", out)
            self.assertIn("cc_extract.py", out)
            self.assertIn("judge.py", out)
            self.assertIn("--layer-b codex", out)
            self.assertIn("measures.py", out)
            self.assertNotIn("WARNING", out)

    def test_short_gap_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stdout(buf), patch("subprocess.run"):
                self.mod.main(["--run-dir", str(Path(tmp) / "run"), "--limit", "2", "--dry-run", "--cache-gap-min", "5"])
            self.assertIn("WARNING", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
