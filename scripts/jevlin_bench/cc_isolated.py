"""One tool-isolated headless Claude Code call, billed to the subscription.

Shared by the extractor (cc_extract.py) and the Claude Code fallback of the semantic judge
(judge.py). Every call is a fresh `claude -p` session with:

  * an **empty environment** — no ANTHROPIC_API_KEY (the subscription login must be used),
    no JEVLIN_*, no CLAUDECODE markers from a parent session, PATH limited to the launcher's
    directory and the system directories;
  * **no tools** — `--tools ""` disables every built-in tool, and `--disallowedTools` names
    them again so a tool that slips past the first flag is denied by the second; which form
    yields zero tool uses is verified live and recorded in the company-news README;
  * **no customisation** — `--safe-mode` (no CLAUDE.md, skills, plugins, hooks, MCP servers,
    custom commands), `--strict-mcp-config` with no config, `--disable-slash-commands`,
    `--no-session-persistence`; the working directory is an empty job directory;
  * a **replaced system prompt** (`--system-prompt`): the fixed instruction is the whole system
    prompt, so the model sees instruction + user payload and nothing of Claude Code's agent
    prompt;
  * **structured output** via `--json-schema`, read back from the result's `structured_output`
    when present, else parsed from the `result` text.

The result carries Claude Code's own accounting (num_turns, usage, total_cost_usd,
permission_denials) and, when `--output-format stream-json` is used, a count of `tool_use`
blocks across the assistant messages — the direct evidence of zero tool use.

An exhausted subscription window surfaces as a Claude Code error; `run()` reports it as
`rate_limited=True` so the caller can wait and re-run the same job. That is the only retry
anywhere: an invalid output is recorded as it is, never re-asked.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"
RATE_LIMIT_RE = re.compile(
    r"rate.?limit|usage limit|limit reached|hit your limit|resets at|too many requests|overloaded|\b529\b", re.I
)
# Every built-in tool name we know of. Unknown names are harmless; the list is a second
# fence behind `--tools ""`.
ALL_TOOLS = (
    "Bash,Edit,Write,Read,Glob,Grep,WebSearch,WebFetch,Task,Agent,TodoWrite,NotebookEdit,MultiEdit,LS,"
    "Skill,KillShell,BashOutput,SlashCommand,ExitPlanMode,EnterPlanMode,AskUserQuestion,ToolSearch,"
    "Monitor,Workflow,Artifact,PowerShell,Computer,SendMessage,TaskStop,EnterWorktree,ExitWorktree"
)


def resolve_claude(claude_bin: str) -> str:
    """Absolute path of the launcher, found on the parent's PATH once; the session gets no such PATH."""
    found = shutil.which(claude_bin) if not os.path.isabs(claude_bin) else claude_bin
    return os.path.abspath(found or claude_bin)


def claude_version(claude_bin: str) -> str:
    try:
        return subprocess.run([resolve_claude(claude_bin), "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def clean_env(claude_bin: str) -> dict[str, str]:
    """Enough to run `claude` under the subscription login, nothing else.

    Explicitly absent: ANTHROPIC_API_KEY and every other *_API_KEY, every JEVLIN_* variable,
    CLAUDECODE / CLAUDE_CODE_* markers, AWS_*, OPENAI_*.
    """
    claude_dir = str(Path(resolve_claude(claude_bin)).parent)
    dirs = [claude_dir, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    env = {
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LOGNAME": os.environ.get("LOGNAME", os.environ.get("USER", "")),
        "SHELL": "/bin/sh",
        "TERM": "xterm-256color",
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PATH": ":".join(dict.fromkeys(dirs)),
        "DISABLE_AUTOUPDATER": "1",
    }
    return {k: v for k, v in env.items() if v}


def forbidden_env_keys(env: dict[str, str]) -> list[str]:
    """Keys that must never reach the session. Used by tests and by run() as an assertion."""
    bad = []
    for key in env:
        upper = key.upper()
        if upper == "ANTHROPIC_API_KEY" or upper.endswith("_API_KEY") or upper.startswith(("JEVLIN_", "CLAUDECODE", "CLAUDE_CODE_", "AWS_", "OPENAI_")):
            bad.append(key)
    return sorted(bad)


def claude_argv(
    claude_bin: str,
    *,
    prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    model: str = DEFAULT_MODEL,
    output_format: str = "json",
    isolation: str = "both",
) -> list[str]:
    """`isolation`: "tools-empty" (`--tools ""`), "disallowed" (`--disallowedTools <all>`), or "both"."""
    argv = [
        resolve_claude(claude_bin), "-p", prompt,
        "--model", model,
        "--output-format", output_format,
        "--json-schema", json.dumps(schema, separators=(",", ":")),
        "--system-prompt", system_prompt,
        "--permission-mode", "dontAsk",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if output_format == "stream-json":
        argv.append("--verbose")
    if isolation in ("tools-empty", "both"):
        argv += ["--tools", ""]
    if isolation in ("disallowed", "both"):
        argv += ["--disallowedTools", ALL_TOOLS]
    if isolation not in ("tools-empty", "disallowed", "both"):
        raise ValueError(f"unknown isolation mode {isolation!r}")
    return argv


@dataclass
class IsolatedResult:
    ok: bool
    parsed: dict[str, Any] | None
    result: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    tool_uses: int | None = None
    num_turns: int | None = None
    error: str | None = None
    rate_limited: bool = False
    elapsed_ms: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def accounting(self) -> dict[str, Any]:
        usage = self.result.get("usage") if isinstance(self.result.get("usage"), dict) else {}
        names = tool_use_names(self.events)
        return {
            "session_id": self.result.get("session_id"),
            "subtype": self.result.get("subtype"),
            "is_error": bool(self.result.get("is_error")),
            "num_turns": self.num_turns,
            "tool_uses": self.tool_uses,
            "tool_use_names": names,
            "structured_output_calls": (sum(1 for n in names if n == STRUCTURED_OUTPUT_TOOL) if names is not None else None),
            "available_tools": available_tools(self.events),
            "server_tool_use": usage.get("server_tool_use"),
            "structured_output_present": isinstance(self.result.get("structured_output"), dict),
            "duration_ms": self.result.get("duration_ms"),
            "duration_api_ms": self.result.get("duration_api_ms"),
            "total_cost_usd": self.result.get("total_cost_usd"),
            "usage": self.result.get("usage"),
            "model_usage": self.result.get("modelUsage"),
            "permission_denials": self.result.get("permission_denials"),
            "elapsed_ms": self.elapsed_ms,
            "returncode": self.returncode,
        }


def parse_stdout(stdout: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The final `result` object and, for stream-json, every event line."""
    text = (stdout or "").strip()
    if not text:
        return {}, []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data, [data]
        if isinstance(data, list):
            events = [item for item in data if isinstance(item, dict)]
            results = [item for item in events if item.get("type") == "result"]
            return (results[-1] if results else (events[-1] if events else {})), events
        return {}, []
    except ValueError:
        pass
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    results = [item for item in events if item.get("type") == "result"]
    return (results[-1] if results else {}), events


# With --json-schema Claude Code injects one pseudo-tool, `StructuredOutput`, and the model
# emits its answer by calling it (observed live, 2.1.300: the init event lists it as the only
# tool). It is the output channel, not a capability, and is counted separately.
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"


def tool_use_names(events: list[dict[str, Any]]) -> list[str] | None:
    """Names of every `tool_use` block across assistant messages; None when no assistant events were
    seen (plain `json` output carries only the result object)."""
    seen_assistant = False
    names: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        seen_assistant = True
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(str(block.get("name") or "?"))
    return names if seen_assistant else None


def count_tool_uses(events: list[dict[str, Any]]) -> int | None:
    """External tool uses: every `tool_use` block except the StructuredOutput channel."""
    names = tool_use_names(events)
    if names is None:
        return None
    return sum(1 for name in names if name != STRUCTURED_OUTPUT_TOOL)


def available_tools(events: list[dict[str, Any]]) -> list[str] | None:
    """The tool list Claude Code announced in its `system`/`init` event (stream-json only)."""
    for event in events:
        if event.get("type") == "system" and isinstance(event.get("tools"), list):
            return [str(t) for t in event["tools"]]
    return None


def extract_structured(result: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer Claude Code's validated `structured_output`; else parse the result text as JSON."""
    structured = result.get("structured_output")
    if isinstance(structured, dict):
        return structured
    text = result.get("result")
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        data = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(body[start : end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def is_rate_limited(result: dict[str, Any], stderr: str, returncode: int) -> bool:
    text = " ".join(str(result.get(k) or "") for k in ("result", "error", "subtype")) + " " + (stderr or "")
    return bool(result.get("is_error") or returncode != 0) and bool(RATE_LIMIT_RE.search(text))


def run(
    *,
    claude_bin: str,
    prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    model: str = DEFAULT_MODEL,
    cwd: Path,
    timeout_s: int = 300,
    output_format: str = "json",
    isolation: str = "both",
) -> IsolatedResult:
    argv = claude_argv(claude_bin, prompt=prompt, system_prompt=system_prompt, schema=schema, model=model,
                       output_format=output_format, isolation=isolation)
    env = clean_env(claude_bin)
    leaked = forbidden_env_keys(env)
    if leaked:  # cannot happen with clean_env; a guard against a future edit
        raise RuntimeError(f"isolated session environment carries forbidden keys: {leaked}")
    cwd.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False)
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, returncode = "", f"claude timed out after {timeout_s}s", 124
    except OSError as exc:
        stdout, stderr, returncode = "", f"{type(exc).__name__}: {exc}", 127
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result, events = parse_stdout(stdout)
    tool_uses = count_tool_uses(events)
    num_turns = result.get("num_turns") if isinstance(result.get("num_turns"), int) else None
    out = IsolatedResult(ok=False, parsed=None, result=result, stdout=stdout, stderr=stderr, returncode=returncode,
                         tool_uses=tool_uses, num_turns=num_turns, elapsed_ms=elapsed_ms, events=events)
    if is_rate_limited(result, stderr, returncode):
        out.rate_limited = True
        out.error = f"rate limited: {(result.get('result') or stderr or '')[:300]}"
        return out
    if returncode != 0 and not result:
        out.error = f"claude exit {returncode}: {(stderr or '').strip()[:500]}"
        return out
    if result.get("is_error"):
        out.error = f"claude error ({result.get('subtype')}): {str(result.get('result') or result.get('error') or '')[:500]}"
        return out
    if not result:
        out.error = "claude produced no JSON result"
        return out
    parsed = extract_structured(result)
    if parsed is None:
        out.error = "no structured output in result"
        return out
    out.ok = True
    out.parsed = parsed
    return out
