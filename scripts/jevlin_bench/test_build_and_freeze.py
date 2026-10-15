"""Dataset build from public rows, and the freeze/verify guard. Offline; no download."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from websearch import build_public_samples as build

ROOT = Path(__file__).resolve().parents[2]
ROWS = [
    {"question": "How much did MaxQ Medical raise in its latest funding round?", "answer": "$31.5 million"},
    {"question": "Who became the new CEO of Kettle Cuisine in August 2026?", "answer": "Peter Hall"},
    {"question": "Which company agreed to acquire OWYN?", "answer": "Good Foods"},
]


def _load_freeze():
    spec = importlib.util.spec_from_file_location("catalog_freeze", ROOT / "experiments" / "catalog_freeze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildCases(unittest.TestCase):
    def test_one_cell_per_row_and_no_invented_gold(self):
        cases = build.build_cases(ROWS, revision="rev")
        self.assertEqual([c["id"] for c in cases], ["cn-001", "cn-002", "cn-003"])
        for case, row in zip(cases, ROWS):
            self.assertEqual(case["question"], row["question"])
            self.assertEqual(case["expected_answer"], row["answer"])
            self.assertEqual(case["ground_truth"], row["answer"])
            self.assertEqual(case["cells"], [{"value": row["answer"], "source": "hf:answer"}])
            self.assertEqual(case["ground_truth_url"], "", "source URLs are omitted upstream; never inferred")
            self.assertNotIn("gold", case)
            self.assertIsNone(case["company"])
            self.assertEqual(case["source"]["revision"], "rev")
            self.assertEqual(len(case["source"]["question_sha256"]), 64)

    def test_recipe_heuristic(self):
        cases = build.build_cases(ROWS, revision="rev")
        self.assertEqual([c["recipe"] for c in cases], ["financing", "hire", "acquisition"])
        self.assertEqual(build.derive_recipe("How many employees did DOOR announce it was laying off?"), "headcount_cut")
        self.assertEqual(build.derive_recipe("Where did Galderma open its new U.S. headquarters?"), "new_office")
        self.assertEqual(build.derive_recipe("What product did vAuto launch in August 2026?"), "product_launch")
        self.assertEqual(build.derive_recipe("Which company selected Caplight to provide pricing?"), "other")
        self.assertEqual(cases[0]["recipe_source"], "derived_from_question_heuristic")

    def test_field_findings_count_what_is_there(self):
        findings = build.field_findings(ROWS)
        self.assertEqual(findings["rows"], 3)
        self.assertEqual(findings["keys"], {"question": 3, "answer": 3})
        self.assertEqual(findings["with_cells"], 0)
        self.assertEqual(findings["with_source_url"], 0)
        self.assertEqual(findings["with_id"], 0)
        self.assertEqual(findings["gold_token_in_question"], 0)

    def test_blank_row_is_refused(self):
        with self.assertRaises(SystemExit):
            build.build_cases([{"question": "x", "answer": ""}], revision="rev")


class MainAndSha(unittest.TestCase):
    def _patched(self, tmp: Path):
        data = tmp / "data"
        public = data / "public"
        return patch.multiple(build, DATA_DIR=data, PUBLIC_DIR=public, SAMPLES=data / "samples.json", MANIFEST=data / "manifest.json",
                              SOURCE_JSON=public / "SOURCE.json", EVAL_JSONL=public / "eval.jsonl")

    def test_sha_mismatch_is_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "eval.jsonl"
            src.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
            with self._patched(Path(tmp)), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(build.main(["--source", str(src)]), 1)
                self.assertFalse((Path(tmp) / "data" / "samples.json").exists())
                self.assertEqual(build.main(["--source", str(src), "--allow-sha-mismatch"]), 0)
            samples = json.loads((Path(tmp) / "data" / "samples.json").read_text())
            manifest = json.loads((Path(tmp) / "data" / "manifest.json").read_text())
            source = json.loads((Path(tmp) / "data" / "public" / "SOURCE.json").read_text())
            self.assertEqual(len(samples), 3)
            self.assertEqual(manifest["ids"], ["cn-001", "cn-002", "cn-003"])
            self.assertFalse(source["sha_matches_pin"])
            self.assertEqual(source["findings"]["with_source_url"], 0)
            self.assertNotIn("downloaded_at", source, "no timestamp: the frozen tree must be reproducible")

    def test_pinned_constants(self):
        self.assertEqual(build.PINNED_REVISION, "5cc874975289202f2bd113e9808cecd412ce3d0d")
        self.assertEqual(build.PINNED_SHA256, "7cc5b1b4d24c3c2bb1c9deab341d2289219f9a8f78d74fe93c4450572fcdf2f3")
        self.assertEqual(build.EXPECTED_ROWS, 100)


class Freeze(unittest.TestCase):
    def test_freeze_verify_and_tamper(self):
        freeze = _load_freeze()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            (data / "public").mkdir(parents=True)
            (data / "samples.json").write_text("[1]")
            (data / "manifest.json").write_text('{"ids":[]}')
            (data / "public" / "eval.jsonl").write_text("{}\n")
            (data / "public" / "SOURCE.json").write_text("{}")
            (data / "official-runs").mkdir()
            (data / "official-runs" / "run.json").write_text("output, not an input")
            frozen = Path(tmp) / "FROZEN.sha256"
            with patch.multiple(freeze, DATA=data, FROZEN=frozen), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(freeze.main(["x", "verify"]), 1, "not frozen yet")
                self.assertEqual(freeze.main(["x", "freeze"]), 0)
                self.assertEqual(freeze.main(["x", "verify"]), 0)
                (data / "official-runs" / "run.json").write_text("outputs may change")
                self.assertEqual(freeze.main(["x", "verify"]), 0)
                (data / "samples.json").write_text("[2]")
                self.assertEqual(freeze.main(["x", "verify"]), 1)
            lines = frozen.read_text().splitlines()
            self.assertTrue(lines[0].startswith("tree "))
            self.assertEqual(lines[1], "files 4")


if __name__ == "__main__":
    unittest.main()
