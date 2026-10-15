#!/usr/bin/env python3
"""One cache-aware pass of the company-news adaptation: search, extract, judge, measure.

The gateway cache (15 min TTL) is keyed by query + tier + provider set and is shared across
every platform-paid tenant, so `jevlin_balanced`, `jevlin_fanout_raw` and
`jevlin_fanout_merged` share a key for the same verbatim question (the pinned set has
equalled the resolved balanced set on every measured deployment). The upstream runner fires
every endpoint of a case concurrently, which would zero-rate two of the three. This driver
therefore runs the arms **invocation by invocation** against one run directory
(`--attach`), one Jevlin key class per invocation with a direct arm beside it, and before a
same-key invocation waits until at least CACHE_GAP minutes have passed since the previous
same-key invocation *ended* — which bounds every question's gap from below, since the case
order is the same in every invocation. This is a design choice: the arms do not see the
question at the same instant (a deviation from upstream, stated in the README), in exchange
for fresh evidence per arm. `cache_hit` is recorded on every search regardless, and cached
searches are excluded from cost and latency statistics by measures.py.

Resume: rerunning with the same --run-dir skips invocations recorded as complete in
pass-state.json and re-attaches the rest; with --retry-errors, stored searches that failed
for a retryable reason (429/5xx/no response) are searched again. Abort: the runner stops an
invocation after N consecutive cases in which every search of a backend family failed; this
driver then stops too (exit 3) and the rerun resumes.

  PYTHONPATH=scripts .venv/bin/python experiments/news_pass.py --run-dir runs/news-smoke --limit 3 --dry-run
  PYTHONPATH=scripts .venv/bin/python experiments/news_pass.py --run-dir runs/news-smoke --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import jevlin_core  # noqa: E402

DEFAULT_DIRECT = ("exa_fast", "perplexity_low", "parallel_fast")
DEFAULT_JEVLIN = ("jevlin_balanced", "jevlin_fast", "jevlin_single", "jevlin_fanout_raw", "jevlin_fanout_merged")
# Seconds per case for the duration estimate (dry run): the slowest arm in the invocation + case gap.
EST_CASE_S = {"balanced": 6.0, "fast": 3.5, "single": 3.0, "direct": 2.0}


@dataclass
class Invocation:
    endpoints: list[str]
    key_class: str | None  # the Jevlin cache key class this invocation searches, if any
    wait_for_class: str | None = None  # must start >= gap after the last invocation of this class ended
    index: int = 0
    estimate_s: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return "+".join(self.endpoints)


def key_class(arm: str) -> str | None:
    return jevlin_core.ARMS[arm].cache_key_class() if arm in jevlin_core.ARMS else None


def plan_invocations(direct: list[str], jevlin: list[str], *, cases: int, case_gap_s: float) -> list[Invocation]:
    """One invocation per Jevlin arm, same-key arms never together, direct arms spread between them.

    Order: the first arm of every key class (fresh keys, no wait), then the remaining arms, each
    waiting for its class. Direct arms are paired with the earliest invocations so vendor load is
    bounded; leftover direct arms get an invocation of their own.
    """
    classes: dict[str, list[str]] = {}
    for arm in jevlin:
        cls = key_class(arm)
        if cls is None:
            raise SystemExit(f"{arm} is not a Jevlin arm")
        classes.setdefault(cls, []).append(arm)
    first_round = [arms[0] for arms in classes.values()]
    rest = [arm for arms in classes.values() for arm in arms[1:]]
    directs = list(direct)
    out: list[Invocation] = []
    for arm in first_round:
        endpoints = [arm] + ([directs.pop(0)] if directs else [])
        out.append(Invocation(endpoints=endpoints, key_class=key_class(arm)))
    for arm in rest:
        endpoints = [arm] + ([directs.pop(0)] if directs else [])
        out.append(Invocation(endpoints=endpoints, key_class=key_class(arm), wait_for_class=key_class(arm)))
    if directs:
        out.append(Invocation(endpoints=directs, key_class=None))
    for i, inv in enumerate(out):
        inv.index = i + 1
        per_case = max([EST_CASE_S.get(inv.key_class or "direct", 2.0)] + [EST_CASE_S["direct"] for e in inv.endpoints if not e.startswith("jevlin_")])
        inv.estimate_s = cases * (per_case + case_gap_s)
    check_plan(out)
    return out


def check_plan(invocations: list[Invocation]) -> None:
    for inv in invocations:
        jevlin_arms = [e for e in inv.endpoints if e.startswith("jevlin_")]
        classes = {key_class(e) for e in jevlin_arms}
        if len(jevlin_arms) > len(classes):
            raise SystemExit(f"invocation {inv.label()} would search two same-key Jevlin arms concurrently")
        if len(jevlin_arms) > 1:
            raise SystemExit(f"invocation {inv.label()} carries two Jevlin arms; one per invocation keeps the request rate bounded")


def runner_argv(python: str, inv: Invocation, *, run_dir: Path, fresh: bool, args: Any) -> list[str]:
    argv = [python, "-u", str(ROOT / "scripts" / "websearch" / "run_official_news_probe.py"), "--extractor", "none",
            "--endpoints", ",".join(inv.endpoints), "--case-gap-s", str(args.case_gap_s),
            "--abort-after-search-errors", str(args.abort_after_search_errors)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.ids:
        argv += ["--ids", args.ids]
    if args.samples:
        argv += ["--samples", str(args.samples)]
    if fresh:
        argv += ["--run-dir", str(run_dir)]
    else:
        argv += ["--attach", str(run_dir)]
        if args.retry_errors:
            argv.append("--attach-retry-errors")
    return argv


def load_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "pass-state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"invocations": []}


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pass-state.json").write_text(json.dumps(state, indent=2) + "\n")


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def last_end_for_class(state: dict[str, Any], cls: str) -> datetime | None:
    ends = [_parse(rec.get("ended_at")) for rec in state["invocations"] if rec.get("key_class") == cls and rec.get("ended_at")]
    ends = [e for e in ends if e]
    return max(ends) if ends else None


def wait_needed_s(state: dict[str, Any], inv: Invocation, gap_min: float, now: datetime) -> float:
    if not inv.wait_for_class:
        return 0.0
    last_end = last_end_for_class(state, inv.wait_for_class)
    if last_end is None:
        return 0.0
    return max(0.0, (last_end + timedelta(minutes=gap_min) - now).total_seconds())


def cache_hits_in(run_dir: Path, endpoints: list[str]) -> dict[str, int]:
    for name in ("run.json", "run.partial.json"):
        path = run_dir / name
        if path.exists():
            payload = json.loads(path.read_text())
            break
    else:
        return {}
    out = {e: 0 for e in endpoints if e.startswith("jevlin_")}
    for case in payload.get("cases") or []:
        for e in out:
            meta = ((case.get("vendors") or {}).get(e) or {}).get("vendor_meta") or {}
            out[e] += 1 if meta.get("jevlin_cache_hit") else 0
    return out


def describe(invocations: list[Invocation], *, gap_min: float, cases: int) -> str:
    lines = [f"plan: {len(invocations)} invocations over {cases} cases; cache gap {gap_min} min between same-key invocations"]
    total = 0.0
    for inv in invocations:
        wait = f"  (wait >= {gap_min} min after the last '{inv.wait_for_class}' invocation ended)" if inv.wait_for_class else ""
        lines.append(f"  {inv.index}. {inv.label():45} key_class={inv.key_class or '-':9} ~{inv.estimate_s/60:.1f} min{wait}")
        total += inv.estimate_s
    lines.append(f"  estimated search wall time ~{total/60:.0f} min plus cache waits (each same-key wait is up to {gap_min} min minus the time already elapsed)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--direct", default=",".join(DEFAULT_DIRECT))
    parser.add_argument("--jevlin", default=",".join(DEFAULT_JEVLIN))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--cache-gap-min", type=float, default=float(os.environ.get("CACHE_GAP_MIN") or 16))
    parser.add_argument("--case-gap-s", type=float, default=1.0)
    parser.add_argument("--abort-after-search-errors", type=int, default=12)
    parser.add_argument("--retry-errors", action="store_true", help="On resume, re-search stored 429/5xx/no-response failures.")
    parser.add_argument("--layer-b", choices=("codex", "claude-code", "none"), default="codex")
    parser.add_argument("--allow-fallback", action="store_true", help="Let judge.py fall back to claude-code when codex is not operational.")
    parser.add_argument("--ar-semantic", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-freeze-check", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and every command; run nothing.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    direct = [e.strip() for e in args.direct.split(",") if e.strip()]
    jevlin = [e.strip() for e in args.jevlin.split(",") if e.strip()]
    samples_path = args.samples or (ROOT / "data" / "company-news" / "samples.json")
    n_cases = args.limit or (len(json.loads(samples_path.read_text())) if samples_path.exists() else 0)
    if args.ids:
        n_cases = len([i for i in args.ids.split(",") if i.strip()])
    invocations = plan_invocations(direct, jevlin, cases=n_cases, case_gap_s=args.case_gap_s)
    print(describe(invocations, gap_min=args.cache_gap_min, cases=n_cases), flush=True)

    if args.cache_gap_min < 15:
        print(f"WARNING: cache gap {args.cache_gap_min} min is below the gateway TTL (15 min); same-key arms may share evidence", flush=True)

    freeze = [args.python, str(ROOT / "experiments" / "catalog_freeze.py"), "verify"]
    extract = [args.python, str(ROOT / "scripts" / "jevlin_bench" / "cc_extract.py"), "--run-dir", str(run_dir), "--run", "--apply"]
    judge_cmd = [args.python, str(ROOT / "scripts" / "jevlin_bench" / "judge.py"), "--run-dir", str(run_dir), "--layer-b", args.layer_b, "--run", "--apply"]
    if args.allow_fallback:
        judge_cmd.append("--allow-fallback")
    if args.ar_semantic:
        judge_cmd.append("--ar-semantic")
    measures = [args.python, str(ROOT / "scripts" / "jevlin_bench" / "measures.py"), "--run-dir", str(run_dir),
                "--md", str(run_dir / "measures.md"), "--json", str(run_dir / "measures.json")]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts")}

    if args.dry_run:
        print("\ncommands (dry run):")
        print("  " + " ".join(freeze))
        state = load_state(run_dir) if run_dir.exists() else {"invocations": []}
        fresh = not (run_dir / "run.json").exists() and not (run_dir / "run.partial.json").exists()
        for inv in invocations:
            done = any(rec.get("label") == inv.label() and rec.get("returncode") == 0 for rec in state["invocations"])
            print(f"  {'[done] ' if done else ''}" + " ".join(runner_argv(args.python, inv, run_dir=run_dir, fresh=fresh, args=args)))
            fresh = False
        print("  " + " ".join(extract))
        print("  " + " ".join(judge_cmd))
        print("  " + " ".join(measures))
        return 0

    if not args.skip_freeze_check:
        if subprocess.run(freeze, check=False).returncode != 0:
            print("frozen question tree does not verify; refusing to run", file=sys.stderr)
            return 1

    state = load_state(run_dir)
    if not args.skip_search:
        for inv in invocations:
            done = any(rec.get("label") == inv.label() and rec.get("returncode") == 0 for rec in state["invocations"])
            if done:
                print(f"== invocation {inv.index} {inv.label()}: already complete, skipping", flush=True)
                continue
            wait = wait_needed_s(state, inv, args.cache_gap_min, datetime.now(timezone.utc))
            if wait > 0:
                print(f"== invocation {inv.index} {inv.label()}: waiting {wait/60:.1f} min for the '{inv.wait_for_class}' cache key to expire", flush=True)
                time.sleep(wait)
            fresh = not (run_dir / "run.json").exists() and not (run_dir / "run.partial.json").exists()
            cmd = runner_argv(args.python, inv, run_dir=run_dir, fresh=fresh, args=args)
            started = datetime.now(timezone.utc)
            print(f"== invocation {inv.index} {inv.label()} at {started.isoformat()}\n   " + " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, env=env, check=False)
            ended = datetime.now(timezone.utc)
            rec = {"index": inv.index, "label": inv.label(), "endpoints": inv.endpoints, "key_class": inv.key_class,
                   "started_at": started.isoformat(), "ended_at": ended.isoformat(), "returncode": proc.returncode,
                   "cache_hits": cache_hits_in(run_dir, inv.endpoints)}
            state["invocations"] = [r for r in state["invocations"] if r.get("label") != inv.label()] + [rec]
            save_state(run_dir, state)
            if any(v for v in rec["cache_hits"].values()):
                print(f"   WARNING: cache hits recorded: {rec['cache_hits']} (these searches are excluded from cost/latency)", flush=True)
            if proc.returncode != 0:
                print(f"invocation {inv.index} exited {proc.returncode}; stopping. Rerun the same command to resume.", file=sys.stderr)
                return 3
    if not args.skip_extract:
        if subprocess.run(extract, env=env, check=False).returncode != 0:
            return 4
    if not args.skip_judge:
        rc = subprocess.run(judge_cmd, env=env, check=False).returncode
        if rc != 0:
            print("judge did not complete (codex not operational? rerun with --layer-b claude-code or --allow-fallback)", file=sys.stderr)
            return 5
    subprocess.run(measures, env=env, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
