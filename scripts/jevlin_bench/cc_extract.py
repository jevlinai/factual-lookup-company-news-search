#!/usr/bin/env python3
"""Tool-isolated headless Claude Code extractor for the company-news adaptation.

Replaces the upstream extractor (`gpt-5.6-terra` through the OpenAI API) with one fresh
`claude -p` session per (question, configuration), billed to the Claude subscription. The
search protocol is untouched: this reads the hits the runner stored and writes an answer.

Per job the model receives exactly: the fixed instruction (the upstream extract prompt,
plus one sentence saying it has no tools) as the whole system prompt, and the user payload
`{"question": ..., "results": [{url, title, snippet}, ...]}` — the <= 10 normalised hits the
runner stored for that configuration, nothing else. No gold, no other question, no
provider name. Same model and flags for every configuration. See cc_isolated.py for the
isolation contract (empty environment, no tools, safe mode, structured output).

Rows are never selectively retried: an invalid or missing output is recorded as a row with
`status: invalid|failed` and an empty answer, which the judge scores incorrect (the
upstream rule for an empty answer). The only re-run is a subscription window exhaustion,
which is a run-level condition, not a model output. A search that failed, or returned no
hits, is `skipped` without a model call (the upstream re-extract rule).

  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/cc_extract.py --run-dir DIR            # plan
  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/cc_extract.py --run-dir DIR --run      # extract
  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/cc_extract.py --run-dir DIR --apply    # write back

Checkpoints: <run-dir>/cc-extract/<case>__<endpoint>.json (resume = rerun); Claude Code's
raw stdout per job under <run-dir>/cc-extract/raw/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from jevlin_bench import cc_isolated  # noqa: E402

DEFAULT_MODEL = cc_isolated.DEFAULT_MODEL
SNIPPET_CHARS = 1200
MAX_HITS = 10

# The upstream extract prompt (run_atomic_news_probe.extract_answer, with the
# rejudge script's "Do not use outside knowledge." sentence), plus one sentence that names
# the isolation the flags enforce. Identical for every configuration.
INSTRUCTION = (
    "Answer only from the provided search snippets. No extra research. "
    "If the snippets do not contain the fact, answer empty. "
    "Do not use outside knowledge. "
    "You have no tools: do not attempt to search, browse, read files or run commands. "
    'Return JSON {"answer": string, "cited_url": string}.'
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "cited_url": {"type": "string"}},
    "required": ["answer", "cited_url"],
    "additionalProperties": False,
}


def pack_hits(hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The <= 10 hits in the shape the upstream extractor saw: url, title, snippet (<= 1,200 chars)."""
    packed = []
    for hit in list(hits or [])[:MAX_HITS]:
        if not isinstance(hit, dict):
            continue
        packed.append({
            "url": str(hit.get("url") or ""),
            "title": str(hit.get("title") or ""),
            "snippet": str(hit.get("snippet") or "")[:SNIPPET_CHARS],
        })
    return packed


def user_payload(question: str, hits: list[dict[str, Any]]) -> str:
    return json.dumps({"question": question, "results": pack_hits(hits)}, ensure_ascii=False)


def validate(parsed: dict[str, Any] | None) -> tuple[bool, dict[str, str], str | None]:
    """Upstream `ExtractOut` semantics: both fields default to "" when absent; a non-string is invalid."""
    if not isinstance(parsed, dict):
        return False, {"answer": "", "cited_url": ""}, "output is not an object"
    out = {"answer": "", "cited_url": ""}
    for key in ("answer", "cited_url"):
        value = parsed.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            return False, {"answer": "", "cited_url": ""}, f"{key} is not a string"
        out[key] = value.strip()
    extra = sorted(set(parsed) - {"answer", "cited_url"})
    if extra:
        return False, {"answer": "", "cited_url": ""}, f"unexpected keys {extra}"
    return True, out, None


def ckpt_path(run_dir: Path, case_id: str, endpoint: str) -> Path:
    return run_dir / "cc-extract" / f"{case_id}__{endpoint}.json"


def load_run(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    for name in ("run.json", "run.partial.json"):
        path = run_dir / name
        if path.exists():
            return path, json.loads(path.read_text())
    raise SystemExit(f"no run.json or run.partial.json in {run_dir}")


def jobs_from_run(payload: dict[str, Any], *, endpoints: set[str] | None, ids: set[str] | None) -> list[dict[str, Any]]:
    jobs = []
    for case in payload.get("cases") or []:
        if ids and case.get("id") not in ids:
            continue
        for endpoint, block in (case.get("vendors") or {}).items():
            if endpoints and endpoint not in endpoints:
                continue
            jobs.append({
                "case_id": case["id"],
                "endpoint": endpoint,
                "question": case["question"],
                "hits": pack_hits(block.get("hits") or []),
                "search_error": block.get("error"),
            })
    return jobs


def run_job(job: dict[str, Any], *, run_dir: Path, args: Any, version: str) -> dict[str, Any]:
    dest = ckpt_path(run_dir, job["case_id"], job["endpoint"])
    if dest.exists():
        return json.loads(dest.read_text())
    base = {
        "case_id": job["case_id"],
        "endpoint": job["endpoint"],
        "extractor": {"kind": "claude-code", "model": args.model, "version": version, "isolation": args.isolation,
                      "output_format": args.output_format, "instruction_sha256": _sha(INSTRUCTION), "schema": SCHEMA},
        "n_hits": len(job["hits"]),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    if job["search_error"]:
        row = {**base, "status": "skipped", "skipped": "search error", "answer": "", "cited_url": "", "schema_valid": None}
    elif not job["hits"]:
        row = {**base, "status": "skipped", "skipped": "no hits", "answer": "", "cited_url": "", "schema_valid": None}
    else:
        row = _extract(job, base, run_dir=run_dir, args=args)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    return row


def _extract(job: dict[str, Any], base: dict[str, Any], *, run_dir: Path, args: Any) -> dict[str, Any]:
    raw_dir = run_dir / "cc-extract" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    job_dir = run_dir / "cc-extract" / "jobs" / f"{job['case_id']}__{job['endpoint']}"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    prompt = user_payload(job["question"], job["hits"])
    attempts = 0
    waited_s = 0
    while True:
        attempts += 1
        result = cc_isolated.run(
            claude_bin=args.claude_bin, prompt=prompt, system_prompt=INSTRUCTION, schema=SCHEMA, model=args.model,
            cwd=job_dir, timeout_s=args.timeout, output_format=args.output_format, isolation=args.isolation,
        )
        stem = f"{job['case_id']}__{job['endpoint']}.attempt-{attempts}"
        (raw_dir / f"{stem}.stdout.json").write_text(result.stdout or "")
        if result.stderr:
            (raw_dir / f"{stem}.stderr.txt").write_text(result.stderr)
        if result.rate_limited and attempts <= args.rate_limit_retries:
            print(f"  rate limited on {job['case_id']} {job['endpoint']}; waiting {args.rate_limit_wait}s (attempt {attempts})", flush=True)
            time.sleep(args.rate_limit_wait)
            waited_s += args.rate_limit_wait
            continue
        break
    shutil.rmtree(job_dir, ignore_errors=True)
    valid, fields, why = validate(result.parsed) if result.ok else (False, {"answer": "", "cited_url": ""}, result.error)
    if result.ok and valid:
        status = "ok"
    elif result.ok:
        status = "invalid"
    else:
        status = "failed"
    return {
        **base,
        "status": status,
        "error": None if status == "ok" else why,
        "answer": fields["answer"] if status == "ok" else "",
        "cited_url": fields["cited_url"] if status == "ok" else "",
        "raw_output": result.parsed if status != "ok" else None,
        "schema_valid": valid if result.ok else None,
        "attempts": attempts,
        "rate_limit_wait_s": waited_s,
        "claude": result.accounting(),
        "prompt_sha256": _sha(prompt),
    }


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply(run_dir: Path, *, model: str, version: str) -> dict[str, Any]:
    run_path, payload = load_run(run_dir)
    backup = run_dir / "run.before-cc-extract.json"
    if not backup.exists():
        shutil.copy2(run_path, backup)
    applied = missing = 0
    statuses: dict[str, int] = {}
    for case in payload.get("cases") or []:
        for endpoint, block in (case.get("vendors") or {}).items():
            path = ckpt_path(run_dir, case["id"], endpoint)
            if not path.exists():
                missing += 1
                continue
            row = json.loads(path.read_text())
            block["answer"] = row.get("answer") or ""
            block["cited_url"] = row.get("cited_url") or ""
            block["extract"] = {
                "kind": "claude-code",
                "model": (row.get("extractor") or {}).get("model") or model,
                "version": (row.get("extractor") or {}).get("version") or version,
                "status": row.get("status"),
                "skipped": row.get("skipped"),
                "error": row.get("error"),
                "schema_valid": row.get("schema_valid"),
                "num_turns": (row.get("claude") or {}).get("num_turns"),
                "tool_uses": (row.get("claude") or {}).get("tool_uses"),
                "total_cost_usd": (row.get("claude") or {}).get("total_cost_usd"),
                "session_id": (row.get("claude") or {}).get("session_id"),
                "extracted_at": row.get("extracted_at"),
            }
            # The judge decides `gold`; mark it pending so a stale verdict is never read as current.
            block["gold"] = {"correct": None, "cells_hit": 0, "note": "pending: run jevlin_bench/judge.py"}
            statuses[row.get("status") or "?"] = statuses.get(row.get("status") or "?", 0) + 1
            applied += 1
    payload["model"] = model
    payload["extractor"] = {"kind": "claude-code", "model": model, "version": version, "instruction": INSTRUCTION,
                            "schema": SCHEMA, "applied_at": datetime.now(timezone.utc).isoformat()}
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {"applied": applied, "missing": missing, "statuses": statuses, "backup": str(backup), "run": str(run_dir / "run.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Call Claude Code. Default prints the plan.")
    parser.add_argument("--apply", action="store_true", help="Write checkpoints back onto run.json.")
    parser.add_argument("--endpoints", default="", help="Comma list; default every endpoint in the run.")
    parser.add_argument("--ids", default="", help="Comma list of case ids; default all.")
    parser.add_argument("--limit", type=int, default=0, help="At most this many pending jobs.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--isolation", choices=("tools-empty", "disallowed", "both"), default="both")
    parser.add_argument("--output-format", choices=("json", "stream-json"), default="json")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per Claude Code session.")
    parser.add_argument("--rate-limit-wait", type=int, default=600)
    parser.add_argument("--rate-limit-retries", type=int, default=6)
    parser.add_argument("--abort-after-failures", type=int, default=8,
                        help="Stop after this many consecutive failed (not invalid) sessions; 0 disables.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    _, payload = load_run(run_dir)
    endpoints = {e.strip() for e in args.endpoints.split(",") if e.strip()} or None
    ids = {i.strip() for i in args.ids.split(",") if i.strip()} or None
    jobs = jobs_from_run(payload, endpoints=endpoints, ids=ids)
    done = [j for j in jobs if ckpt_path(run_dir, j["case_id"], j["endpoint"]).exists()]
    pending = [j for j in jobs if not ckpt_path(run_dir, j["case_id"], j["endpoint"]).exists()]
    if args.limit:
        pending = pending[: args.limit]
    version = cc_isolated.claude_version(args.claude_bin) if (args.run or args.apply) else "n/a"
    plan = {
        "run_dir": str(run_dir),
        "jobs": len(jobs),
        "checkpointed": len(done),
        "pending": len(pending),
        "pending_with_hits": sum(1 for j in pending if j["hits"] and not j["search_error"]),
        "pending_skips": sum(1 for j in pending if not j["hits"] or j["search_error"]),
        "model": args.model,
        "claude_code_version": version,
        "isolation": args.isolation,
        "output_format": args.output_format,
        "instruction_sha256": _sha(INSTRUCTION),
        "api_key_calls": 0,
        "vendor_calls": 0,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        failures_in_a_row = 0
        counts: dict[str, int] = {}
        for n, job in enumerate(pending, 1):
            row = run_job(job, run_dir=run_dir, args=args, version=version)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(json.dumps({"n": n, "case": job["case_id"], "endpoint": job["endpoint"], "status": row["status"],
                              "answer": (row.get("answer") or "")[:80], "tool_uses": (row.get("claude") or {}).get("tool_uses"),
                              "num_turns": (row.get("claude") or {}).get("num_turns")}, ensure_ascii=False), flush=True)
            failures_in_a_row = failures_in_a_row + 1 if row["status"] == "failed" else 0
            if args.abort_after_failures and failures_in_a_row >= args.abort_after_failures:
                print(f"ABORTING: {failures_in_a_row} consecutive failed sessions; last: {row.get('error')}", flush=True)
                return 3
        print(json.dumps({"finished": True, **counts}), flush=True)
    if args.apply:
        print(json.dumps(apply(run_dir, model=args.model, version=version), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
