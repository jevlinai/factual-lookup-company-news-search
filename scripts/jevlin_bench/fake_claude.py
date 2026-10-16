"""A fake `claude` launcher for offline tests.

`make_fake_claude(dir, behaviour)` writes an executable Python script that records every
invocation (argv, environment, cwd) to <dir>/calls/NNN.json and prints a Claude Code
`--output-format json` result object — or stream-json lines — shaped like 2.1.300's. The
behaviours cover a valid structured answer, an invalid one, garbage output, a first-call
rate-limit error, a crash, and a session that announces tools and makes an external tool
call (so the isolation checks have something to catch). `make_fake_codex` does the same for
`codex exec` with `--output-last-message`.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

SCRIPT = r'''#!{python}
import json, os, sys
from pathlib import Path
HERE = Path({here!r})
CALLS = HERE / "calls"
CALLS.mkdir(exist_ok=True)
n = len(list(CALLS.glob("*.json"))) + 1
argv = sys.argv[1:]
if argv == ["--version"]:
    print("9.9.9 (Claude Code fake)")
    sys.exit(0)
(CALLS / f"{{n:03d}}.json").write_text(json.dumps({{"argv": argv, "env": dict(os.environ), "cwd": os.getcwd()}}))
behaviour = {behaviour!r}
fmt = argv[argv.index("--output-format") + 1] if "--output-format" in argv else "json"
prompt = argv[argv.index("-p") + 1] if "-p" in argv else ""
def answer_for(prompt):
    try:
        payload = json.loads(prompt)
    except ValueError:
        payload = {{}}
    if "extracted_answer" in payload:  # accuracy judge payload
        gold = str(payload.get("gold_answer") or "").lower()
        ans = str(payload.get("extracted_answer") or "").lower()
        ok = bool(gold) and gold in ans
        return {{"verdict": "correct" if ok else "incorrect", "cells_hit": 1 if ok else 0, "justification": "fake: substring rule"}}
    if "snippets" in payload:  # AR judge payload
        ranks = [s["rank"] for s in payload["snippets"] if str(payload.get("gold_answer") or "").lower() in (s.get("snippet") or "").lower()]
        return {{"bearing_ranks": ranks, "note": "fake"}}
    hits = payload.get("results") or []
    return {{"answer": "$31.5 million", "cited_url": hits[0]["url"] if hits else ""}}
structured = answer_for(prompt)
if behaviour == "invalid":
    structured = {{"answer": 31.5, "cited_url": ""}}
if behaviour == "rate-limit-first" and n == 1:
    print(json.dumps({{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "Rate limit reached. Your limit resets at 3pm.", "num_turns": 0}}))
    sys.exit(1)
if behaviour == "garbage":
    print("not json at all")
    sys.exit(0)
if behaviour == "crash":
    print("boom", file=sys.stderr)
    sys.exit(2)
result = {{"type": "result", "subtype": "success", "is_error": False, "duration_ms": 1200, "duration_api_ms": 900, "num_turns": 2,
          "result": json.dumps(structured), "structured_output": structured, "session_id": f"fake-{{n}}", "total_cost_usd": 0.004,
          "usage": {{"input_tokens": 2, "output_tokens": 50, "server_tool_use": {{"web_search_requests": 0, "web_fetch_requests": 0}}}},
          "permission_denials": []}}
if behaviour == "tool-user":
    result["num_turns"] = 3
if fmt == "stream-json":
    tools = ["StructuredOutput"] if behaviour != "tool-user" else ["Bash", "WebSearch", "StructuredOutput"]
    content = [{{"type": "tool_use", "name": "StructuredOutput", "input": structured}}]
    if behaviour == "tool-user":
        content.insert(0, {{"type": "tool_use", "name": "WebSearch", "input": {{"query": "x"}}}})
    print(json.dumps({{"type": "system", "subtype": "init", "tools": tools, "model": "fake"}}))
    print(json.dumps({{"type": "assistant", "message": {{"content": content}}}}))
    print(json.dumps(result))
else:
    print(json.dumps(result))
'''

CODEX_SCRIPT = r'''#!{python}
import json, os, sys
from pathlib import Path
HERE = Path({here!r})
CALLS = HERE / "calls"
CALLS.mkdir(exist_ok=True)
argv = sys.argv[1:]
if argv == ["--version"]:
    print("codex-cli 0.0.0-fake")
    sys.exit(0)
n = len(list(CALLS.glob("*.json"))) + 1
# The real `codex exec` blocks on a non-/dev/null stdin that is not a TTY; record what it was given.
stdin_is_devnull = os.fstat(0).st_rdev == os.stat(os.devnull).st_rdev
(CALLS / f"{{n:03d}}.json").write_text(json.dumps({{"argv": argv, "env": dict(os.environ), "cwd": os.getcwd(),
                                                   "stdin_is_devnull": stdin_is_devnull}}))
behaviour = {behaviour!r}
out = Path(argv[argv.index("--output-last-message") + 1]) if "--output-last-message" in argv else None
prompt = argv[-1]
payload = {{}}
try:
    payload = json.loads(prompt.split("INPUT:", 1)[1].strip())
except (IndexError, ValueError):
    pass
gold = str(payload.get("gold_answer") or "").lower()
ans = str(payload.get("extracted_answer") or "").lower()
ok = bool(gold) and gold in ans
verdict = {{"verdict": "correct" if ok else "incorrect", "cells_hit": 1 if ok else 0, "justification": "fake codex"}}
if behaviour == "garbage":
    if out: out.write_text("no json here")
    sys.exit(0)
if behaviour == "crash":
    print("codex exploded", file=sys.stderr)
    sys.exit(2)
# The `--json` event stream of codex-cli 0.160.0, abridged; "web" adds the tool item a judge must not use.
print(json.dumps({{"type": "thread.started", "thread_id": "fake"}}))
print(json.dumps({{"type": "turn.started"}}))
if behaviour == "web":
    print(json.dumps({{"type": "item.completed", "item": {{"id": "item_1", "type": "web_search", "query": "gold fact"}}}}))
print(json.dumps({{"type": "item.completed", "item": {{"id": "item_2", "type": "agent_message", "text": json.dumps(verdict)}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 1000, "cached_input_tokens": 500, "output_tokens": 50}}}}))
if out: out.write_text(json.dumps(verdict))
'''


def _write_exec(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make_fake_claude(directory: Path, behaviour: str = "ok") -> Path:
    directory = Path(directory)
    return _write_exec(directory / "claude", SCRIPT.format(python=sys.executable, here=str(directory), behaviour=behaviour))


def make_fake_codex(directory: Path, behaviour: str = "ok") -> Path:
    directory = Path(directory)
    return _write_exec(directory / "codex", CODEX_SCRIPT.format(python=sys.executable, here=str(directory), behaviour=behaviour))


def recorded_calls(directory: Path) -> list[dict]:
    calls = sorted((Path(directory) / "calls").glob("*.json"))
    return [json.loads(p.read_text()) for p in calls]


if __name__ == "__main__":
    print(make_fake_claude(Path(os.environ.get("FAKE_DIR") or "fake-claude")))
