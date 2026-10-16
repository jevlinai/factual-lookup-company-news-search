#!/usr/bin/env python3
"""Two-layer judge for the company-news adaptation. No vendor calls, no API keys.

Replaces the upstream Opus-on-Bedrock judge with:

  Layer A — deterministic. Normalised exact / containment / numeric / date / alias matching
            of every gold cell against the extracted answer. Offline, auditable, the same for
            every run. Also scores AR@K on the stored snippets (per-snippet, same matcher).
  Layer B — an independent semantic judge behind a small backend interface:
              codex        `codex exec` (OpenAI model family via the ChatGPT login, the
                           preferred independent family) — tool-less and config-independent,
                           verified live 2026-10-16 on codex-cli 0.160.0, see CodexBackend;
              claude-code  a tool-isolated `claude -p` session (cc_isolated) — the labelled
                           fallback; it is the extractor's family, so the independence the
                           upstream chain has is lost and every report says so;
              fake         scripted verdicts for tests and calibration dry runs.
            Fixed rubric, structured verdict with a one-line justification, blind to the
            provider that produced the answer and to the search results: it sees question,
            gold answer, required facts and the extracted answer, nothing else.

Both layers' outputs are kept on every row. The final verdict policy is `layer_b_primary`:
Layer B's verdict when it produced a valid one, Layer A's otherwise (the row is flagged
`judge_failed` and stays in the denominator). With `--layer-b none` the policy is
`layer_a_only`. An empty answer is incorrect without any judge call (the upstream rule).
Disagreement between layers is recorded per row and reported by measures.py.

  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/judge.py --run-dir DIR                       # plan
  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/judge.py --run-dir DIR --layer-b codex --run  # judge
  PYTHONPATH=scripts .venv/bin/python scripts/jevlin_bench/judge.py --run-dir DIR --apply                # write back

Checkpoints: <run-dir>/judge/<case>__<endpoint>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from jevlin_bench import cc_isolated  # noqa: E402

KS = (1, 5, 10)

# ═══════════════════════════════════════════════════════════════════════════
# Layer A — deterministic matching
# ═══════════════════════════════════════════════════════════════════════════

HONORIFICS = {"dr", "mr", "mrs", "ms", "miss", "prof", "professor", "sir", "dame", "hon", "the"}
CREDENTIAL_RE = re.compile(
    r"[,\s]+(?:ph\.?\s?d\.?|m\.?\s?d\.?|m\.?b\.?a\.?|c\.?p\.?a\.?|c\.?f\.?a\.?|j\.?d\.?|pharm\.?\s?d\.?|d\.?o\.?|r\.?n\.?|p\.?e\.?|"
    r"esq\.?|jr\.?|sr\.?|ii|iii|iv|d\.?d\.?s\.?|d\.?v\.?m\.?|m\.?sc\.?|b\.?sc\.?|ll\.?m\.?|m\.?p\.?h\.?|c\.?f\.?p\.?)(?=[,\s]|$)",
    re.I,
)
CORP_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "plc", "ag", "gmbh", "corp", "corporation", "co", "company",
    "sa", "nv", "se", "spa", "pty", "lp", "llp", "bv", "oy", "ab", "kk", "holdings", "group",
}
# `holdings` and `group` are stripped only as a *trailing* token and only when at least one
# other token remains; "TruGolf Holdings, Inc." -> "trugolf", "CVC DIF" is untouched.
QUALIFIERS = {
    "approximately", "approx", "about", "around", "roughly", "nearly", "almost", "over", "more", "than", "up", "to",
    "some", "an", "estimated", "total", "of", "in", "its", "a", "the", "at", "least", "circa", "c",
}
COUNT_UNITS = {"employees", "employee", "people", "jobs", "job", "staff", "workers", "worker", "positions", "roles",
               "percent", "pct", "%", "units", "stores", "locations", "offices"}
WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
CURRENCY_ALIASES = {
    "$": "usd", "us$": "usd", "usd": "usd", "dollar": "usd", "dollars": "usd",
    "€": "eur", "eur": "eur", "euro": "eur", "euros": "eur",
    "£": "gbp", "gbp": "gbp", "pound": "gbp", "pounds": "gbp",
    "c$": "cad", "ca$": "cad", "cad": "cad", "a$": "aud", "au$": "aud", "aud": "aud",
    "chf": "chf", "¥": "jpy", "jpy": "jpy", "yen": "jpy", "₹": "inr", "inr": "inr", "rs": "inr", "rupees": "inr",
    "sek": "sek", "nok": "nok", "dkk": "dkk", "hk$": "hkd", "hkd": "hkd", "s$": "sgd", "sgd": "sgd",
    "cny": "cny", "rmb": "cny", "yuan": "cny", "krw": "krw", "nz$": "nzd", "nzd": "nzd", "r$": "brl", "brl": "brl",
    "zar": "zar", "mxn": "mxn", "pln": "pln", "ils": "ils", "aed": "aed", "sar": "sar",
}
SCALES = {
    "k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "mm": 1e6, "mil": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12,
    "crore": 1e7, "crores": 1e7, "lakh": 1e5, "lakhs": 1e5,
}
_CUR_PRE = r"(?P<cur_pre>us\$|ca\$|c\$|au\$|a\$|hk\$|s\$|nz\$|r\$|[$€£¥₹]|\b(?:usd|eur|gbp|cad|aud|chf|jpy|inr|sek|nok|dkk|hkd|sgd|cny|rmb|krw|nzd|brl|zar|mxn|pln|ils|aed|sar|rs)\b\.?)"
_SCALE = r"(?P<scale>thousand|millions?|billions?|trillion|crores?|lakhs?|mn|mm|mil|bn|tn|k|m|b|t)"
_CUR_POST = r"(?P<cur_post>dollars?|euros?|pounds?|yen|rupees?|yuan|usd|eur|gbp|cad|aud|chf)"
NUMBER_RE = re.compile(
    rf"{_CUR_PRE}?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?:{_SCALE}\b\.?)?\s*(?P<pct>%|percent)?\s*(?:{_CUR_POST}\b)?",
    re.I,
)
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH = r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
DATE_RES = (
    re.compile(rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH},?\s+(?P<year>\d{{4}})\b", re.I),
    re.compile(rf"\b{_MONTH}\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<year>\d{{4}})\b", re.I),
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b"),
    re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\b"),
    re.compile(rf"\b{_MONTH},?\s+(?P<year>\d{{4}})\b", re.I),
)


def norm_text(text: str) -> str:
    """NFKC, casefold, trademark/quote cleanup, `&`->`and`, dashes->space, dots kept only inside numbers."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.casefold()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[®™©]", "", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[-–—/]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    s = re.sub(r"[^\w\s$€£¥₹%]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains(needle_norm: str, hay_norm: str) -> bool:
    return bool(needle_norm) and f" {needle_norm} " in f" {hay_norm} "


def words_to_digits(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        parts = re.split(r"[\s-]+", match.group(0).strip())
        total = 0
        for part in parts:
            value = WORD_NUMBERS.get(part.lower())
            if value is None:
                return match.group(0)
            total = total * 100 if value == 100 and total else total + value
        return str(total)

    pattern = r"\b(?:" + "|".join(WORD_NUMBERS) + r")(?:[\s-](?:" + "|".join(WORD_NUMBERS) + r"))*\b"
    return re.sub(pattern, repl, text, flags=re.I)


def extract_numbers(text: str) -> list[tuple[float, str | None, str | None]]:
    """(value, currency, unit) for every quantity in the text; `unit` is 'pct' or None."""
    out = []
    source = words_to_digits(unicodedata.normalize("NFKC", str(text or "")))
    for match in NUMBER_RE.finditer(source):
        raw = match.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        scale = (match.group("scale") or "").lower()
        if scale:
            value *= SCALES.get(scale, 1)
        cur = (match.group("cur_pre") or match.group("cur_post") or "").lower().rstrip(".")
        currency = CURRENCY_ALIASES.get(cur) if cur else None
        unit = "pct" if match.group("pct") else None
        out.append((value, currency, unit))
    return out


def extract_dates(text: str) -> list[tuple[int, int, int | None]]:
    out = []
    s = unicodedata.normalize("NFKC", str(text or ""))
    for regex in DATE_RES:
        for match in regex.finditer(s):
            month_raw = match.group("month").lower().rstrip(".")
            month = int(month_raw) if month_raw.isdigit() else MONTHS.get(month_raw[:3] if len(month_raw) > 4 else month_raw) or MONTHS.get(month_raw[:3])
            if not month or not 1 <= month <= 12:
                continue
            year = int(match.group("year"))
            day = int(match.group("day")) if "day" in match.groupdict() and match.group("day") else None
            if day is not None and not 1 <= day <= 31:
                continue
            out.append((year, month, day))
    return out


def _is_quantity(gold: str) -> bool:
    """A gold cell that is a quantity (money, count, percentage) rather than a name that happens to carry a digit."""
    s = norm_text(words_to_digits(gold))
    s = NUMBER_RE.sub(" ", unicodedata.normalize("NFKC", str(gold or "")).lower())
    tokens = [t for t in re.split(r"[^a-z%]+", s) if t]
    return all(t in QUALIFIERS or t in COUNT_UNITS or t in SCALES or t in CURRENCY_ALIASES for t in tokens)


def _values_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b)))


def numeric_match(gold: str, answer: str) -> bool | None:
    """None when the gold has no number; True when every gold quantity has an equal one in the answer
    (currency must agree when both state one; `%` must agree)."""
    gold_nums = extract_numbers(gold)
    if not gold_nums:
        return None
    answer_nums = extract_numbers(answer)
    for value, currency, unit in gold_nums:
        found = False
        for a_value, a_currency, a_unit in answer_nums:
            if not _values_close(value, a_value):
                continue
            if currency and a_currency and currency != a_currency:
                continue
            if (unit == "pct") != (a_unit == "pct"):
                continue
            found = True
            break
        if not found:
            return False
    return True


def date_match(gold: str, answer: str) -> bool | None:
    gold_dates = extract_dates(gold)
    if not gold_dates:
        return None
    answer_dates = extract_dates(answer)
    for year, month, day in gold_dates:
        ok = any(y == year and m == month and (day is None or d is None or d == day) for y, m, d in answer_dates)
        if not ok:
            return False
    return True


def core_name(text: str) -> str:
    """Name without honorifics, credentials, initials, quoted nicknames, parentheticals, corporate suffixes."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = re.sub(r"[\"“”']([^\"“”']{1,40})[\"“”']", " ", s)  # "Dimi"
    s = re.sub(r"\([^)]*\)", " ", s)  # (NAI), (CTF II)
    s = CREDENTIAL_RE.sub(" ", s)
    s = norm_text(s)
    tokens = s.split()
    while tokens and tokens[0] in HONORIFICS:
        tokens = tokens[1:]
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    while len(tokens) > 1 and tokens[-1] in CORP_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def match_cell(gold_value: str, answer: str) -> tuple[bool, str]:
    """Match one gold cell against the answer; returns (matched, method)."""
    gold_n = norm_text(gold_value)
    answer_n = norm_text(answer)
    if not gold_n:
        return True, "empty-cell"
    if gold_n == answer_n:
        return True, "exact"
    if contains(gold_n, answer_n):
        return True, "contains"
    if _is_quantity(gold_value):
        verdict = numeric_match(gold_value, answer)
        if verdict is not None:
            return verdict, "numeric"
    dates = date_match(gold_value, answer)
    if dates is not None and not extract_numbers(re.sub(r"\d", "", gold_value)):
        if dates:
            return True, "date"
    gold_core = core_name(gold_value)
    if gold_core and len(gold_core) >= 3 and contains(gold_core, core_name(answer)):
        # A name match must not contradict a number the gold carries (e.g. "NAI 2.8" vs "NAI 2.7").
        numbers_ok = numeric_match(gold_value, answer)
        if numbers_ok is None or numbers_ok:
            return True, "alias"
    return False, "none"


def layer_a_verdict(answer: str, cells: list[dict[str, Any]], expected: str = "") -> dict[str, Any]:
    """Deterministic verdict: correct iff every non-empty gold cell matches. Empty answer: incorrect, no matching."""
    values = [str(c.get("value") or "").strip() for c in (cells or []) if str(c.get("value") or "").strip()]
    if not values and expected:
        values = [expected]
    if not (answer or "").strip():
        return {"correct": False, "cells_hit": 0, "cells": [{"value": v, "matched": False, "method": "empty-answer"} for v in values],
                "note": "empty answer", "layer": "A"}
    per_cell = []
    for value in values:
        matched, method = match_cell(value, answer)
        per_cell.append({"value": value, "matched": matched, "method": method})
    hits = sum(1 for c in per_cell if c["matched"])
    return {
        "correct": bool(values) and hits == len(values),
        "cells_hit": hits,
        "cells": per_cell,
        "note": ",".join(c["method"] for c in per_cell) or "no cells",
        "layer": "A",
    }


def layer_a_bearing_ranks(hits: list[dict[str, Any]], cells: list[dict[str, Any]], expected: str = "") -> list[int]:
    """Ranks (1-indexed) of the stored snippets that alone contain every gold cell, by the Layer A matcher."""
    values = [str(c.get("value") or "").strip() for c in (cells or []) if str(c.get("value") or "").strip()]
    if not values and expected:
        values = [expected]
    if not values:
        return []
    ranks = []
    for index, hit in enumerate(list(hits or [])[:10], 1):
        text = f"{hit.get('title') or ''} {hit.get('snippet') or ''}"
        if all(match_cell(value, text)[0] for value in values):
            ranks.append(index)
    return ranks


def ar_flags(ranks: list[int]) -> dict[str, bool]:
    return {f"ar{k}": any(r <= k for r in ranks) for k in KS}


# ═══════════════════════════════════════════════════════════════════════════
# Layer B — semantic judge behind a backend interface
# ═══════════════════════════════════════════════════════════════════════════

ACCURACY_RUBRIC = """You grade one extracted answer against human-labelled gold for a factual-lookup benchmark.

Rules:
1. verdict is "correct" only if EVERY required fact (gold cell) is present in the extracted answer.
2. Synonyms, abbreviations, and equivalent numeric or date forms count as present (eleven million = $11 million = $11M; 3 March 2026 = March 3, 2026; "Dr. Jane Q. Doe" = "Jane Doe").
3. A different value, a different person or company, a partial name that could be someone else, or an answer that hedges between candidates is "incorrect".
4. Extra correct context in the answer does not make it incorrect; a contradiction of a gold fact does.
5. Use ONLY the question, the gold answer, the required facts and the extracted answer. Do not use outside knowledge. You are not told which search provider produced the answer; do not guess or mention one.
6. justification is one short sentence.

Return ONLY a JSON object: {"verdict": "correct" | "incorrect", "cells_hit": <int>, "justification": "<one line>"}"""

AR_RUBRIC = """Decide which search snippets already contain the gold fact.

A snippet is answer-bearing only if a reader could recover every required fact (gold cell) from
that snippet alone, without outside knowledge. Synonyms and equivalent numeric forms are
allowed (eleven million = $11 million = $11M). Digit coincidence is not enough: do not treat
"2.0" as matching "2026", or "$20M" as matching "$11 million". Stale directory pages that name
a different person, company, or round are not answer-bearing. Ranks are 1-indexed in the
order the snippets are listed. You are not told which search provider produced them.

Return ONLY a JSON object: {"bearing_ranks": [<int>, ...], "note": "<at most 12 words>"}"""

ACCURACY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "cells_hit": {"type": "integer"},
        "justification": {"type": "string"},
    },
    "required": ["verdict", "cells_hit", "justification"],
    "additionalProperties": False,
}
AR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"bearing_ranks": {"type": "array", "items": {"type": "integer"}}, "note": {"type": "string"}},
    "required": ["bearing_ranks", "note"],
    "additionalProperties": False,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


RUBRIC_SHA256 = sha256_text(ACCURACY_RUBRIC)
AR_RUBRIC_SHA256 = sha256_text(AR_RUBRIC)


def accuracy_payload(question: str, expected: str, cells: list[dict[str, Any]], answer: str) -> str:
    """Blind to provider and to search results by construction: these four fields only."""
    return json.dumps(
        {
            "question": question,
            "gold_answer": expected,
            "required_facts": [str(c.get("value") or "") for c in (cells or []) if str(c.get("value") or "").strip()] or [expected],
            "extracted_answer": answer,
        },
        ensure_ascii=False,
    )


def ar_payload(question: str, expected: str, cells: list[dict[str, Any]], hits: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "question": question,
            "gold_answer": expected,
            "required_facts": [str(c.get("value") or "") for c in (cells or []) if str(c.get("value") or "").strip()] or [expected],
            "snippets": [
                {"rank": i, "title": str(h.get("title") or "")[:300], "url": str(h.get("url") or ""), "snippet": str(h.get("snippet") or "")[:1200]}
                for i, h in enumerate(list(hits or [])[:10], 1)
            ],
        },
        ensure_ascii=False,
    )


class BackendResult:
    def __init__(self, ok: bool, parsed: dict[str, Any] | None, *, error: str | None = None, meta: dict[str, Any] | None = None,
                 rate_limited: bool = False) -> None:
        self.ok = ok
        self.parsed = parsed
        self.error = error
        self.meta = meta or {}
        self.rate_limited = rate_limited


class Backend(Protocol):
    label: str
    family: str
    independent_of_extractor: bool

    def run(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any], workdir: Path) -> BackendResult: ...


class FakeBackend:
    """Scripted verdicts: a callable (user_prompt_json) -> dict, or a fixed dict. Tests and dry runs."""

    label = "fake"
    family = "none"
    independent_of_extractor = True

    def __init__(self, script: Any = None, *, fail: bool = False) -> None:
        self.script = script
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def run(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any], workdir: Path) -> BackendResult:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "schema": schema})
        if self.fail:
            return BackendResult(False, None, error="fake failure")
        if callable(self.script):
            return BackendResult(True, self.script(json.loads(user_prompt)), meta={"backend": "fake"})
        return BackendResult(True, dict(self.script or {"verdict": "incorrect", "cells_hit": 0, "justification": "fake"}), meta={"backend": "fake"})


CODEX_DEFAULT_MODEL = "gpt-6.1-sol"
CODEX_DEFAULT_EFFORT = "medium"
# Item types in the `--json` event stream that mean the model reached outside its prompt. A verdict
# formed with any of them is not a judgement of the extracted text and is discarded.
CODEX_BREACH_ITEM_TYPES = ("web_search", "mcp_tool_call", "command_execution", "file_change")
# Everything that turns `codex exec` into a tool-less, config-independent judge. Order matters only
# for readability; every flag was accepted by codex-cli 0.160.0.
CODEX_LOCK_FLAGS = (
    "--ignore-user-config",   # the operator's config.toml (model, plugins, MCP servers) stays out
    "--ignore-rules",         # and so do user/project execpolicy .rules files
    "--ephemeral",            # no session rollout under ~/.codex/sessions per judged row
    "-c", "web_search=disabled",  # web search is ON by default; the sandbox does not govern it
    "--disable", "apps",          # the bundled ChatGPT-apps connector (search_service.web_run)
    "--disable", "browser_use",
    "--disable", "computer_use",
    "--disable", "hooks",
    "--disable", "code_mode_host",  # the shell tool fails closed: nothing runs in the workdir
)


class CodexBackend:
    """`codex exec` — OpenAI model family through the operator's ChatGPT login; no OPENAI_API_KEY ever
    reaches the process (the environment is rebuilt without any *_API_KEY).

    Verified live on 2026-10-16 against the Codex CLI bundled with the ChatGPT desktop app
    (`codex-cli 0.160.0`, `/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex`).
    Three findings from the probes shape this backend:

    * `codex exec` reads stdin whenever stdin is not a TTY ("Reading additional input from
      stdin...") and an open pipe blocks it indefinitely — the first probe sat for six hours.
      The child therefore gets /dev/null as stdin.
    * Web search is on by default and `--sandbox read-only` does not touch it. With
      `-c web_search=disabled` the model reached for the bundled ChatGPT-apps connector instead
      (`codex_apps` → `search_service.web_run`), so `apps` is disabled too, with browser use,
      computer use and hooks; `--ignore-user-config` keeps the operator's plugins and MCP servers
      out. The read-only sandbox blocks the network for shell commands ("Could not resolve host")
      but lets them read world-readable files, so the shell is closed as well (`code_mode_host`
      off) and every tool item in the `--json` event stream is counted: a row judged with any
      web_search, MCP call, command or file change fails rather than being trusted.
    * Model and reasoning effort are pinned on the command line (`-m`, `model_reasoning_effort`)
      so the judge does not follow whatever the operator's config.toml says that day.

    Remaining flags: `--skip-git-repo-check` (the workdir is not a repository), `-C <workdir>`,
    `--output-schema <file>` (structured final message), `--output-last-message <file>`; the
    prompt is the positional argument. A sanity probe takes ≈ 11–15 s and ≈ 21k input tokens.
    """

    label = "codex"
    family = "openai"
    independent_of_extractor = True

    def __init__(self, codex_bin: str = "codex", model: str | None = None, timeout_s: int = 300,
                 effort: str | None = None) -> None:
        self.codex_bin = codex_bin
        self.model = model or (os.environ.get("CODEX_JUDGE_MODEL") or "").strip() or CODEX_DEFAULT_MODEL
        self.effort = effort or (os.environ.get("CODEX_JUDGE_EFFORT") or "").strip() or CODEX_DEFAULT_EFFORT
        self.timeout_s = timeout_s
        self.version: str | None = None

    @staticmethod
    def available(codex_bin: str = "codex") -> tuple[bool, str]:
        exe = shutil.which(codex_bin) if not os.path.isabs(codex_bin) else codex_bin
        if not exe:
            return False, f"{codex_bin} not on PATH"
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "").strip()[:300]
        return True, (proc.stdout or "").strip()

    def argv(self, *, prompt: str, schema_path: Path, out_path: Path, workdir: Path) -> list[str]:
        exe = shutil.which(self.codex_bin) if not os.path.isabs(self.codex_bin) else self.codex_bin
        argv = [exe or self.codex_bin, "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(workdir),
                *CODEX_LOCK_FLAGS, "--output-schema", str(schema_path), "--output-last-message", str(out_path),
                "-m", self.model, "-c", f"model_reasoning_effort={self.effort}"]
        argv.append(prompt)
        return argv

    @staticmethod
    def read_events(stdout: str) -> dict[str, Any]:
        """Summarise a `--json` event stream: item counts by type, token usage, error messages and
        the last agent message (the fallback when the last-message file is missing)."""
        items: dict[str, int] = {}
        errors: list[str] = []
        usage: dict[str, Any] | None = None
        last_text = ""
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get("item") if isinstance(event.get("item"), dict) else None
            if event.get("type") == "item.completed" and item:
                kind = str(item.get("type") or "")
                items[kind] = items.get(kind, 0) + 1
                if kind == "agent_message":
                    last_text = str(item.get("text") or "")
                elif kind == "error":
                    errors.append(str(item.get("message") or "")[:300])
            elif event.get("type") in ("error", "turn.failed"):
                errors.append(json.dumps(event)[:300])
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
        breaches = {k: v for k, v in items.items() if k in CODEX_BREACH_ITEM_TYPES}
        return {"items": items, "breaches": breaches, "errors": errors, "usage": usage, "last_text": last_text}

    @staticmethod
    def env() -> dict[str, str]:
        env = {k: os.environ.get(k, "") for k in ("HOME", "USER", "LOGNAME", "LANG", "TMPDIR", "TERM")}
        env["PATH"] = ":".join(dict.fromkeys(os.environ.get("PATH", "").split(":") + ["/usr/bin", "/bin"]))
        env["SHELL"] = "/bin/sh"
        return {k: v for k, v in env.items() if v}

    def run(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any], workdir: Path) -> BackendResult:
        workdir.mkdir(parents=True, exist_ok=True)
        schema_path = workdir / "schema.json"
        out_path = workdir / "last_message.json"
        schema_path.write_text(json.dumps(schema))
        if out_path.exists():
            out_path.unlink()
        prompt = f"{system_prompt}\n\nINPUT:\n{user_prompt}"
        argv = self.argv(prompt=prompt, schema_path=schema_path, out_path=out_path, workdir=workdir)
        env = self.env()
        assert not cc_isolated.forbidden_env_keys(env), "codex environment carries an API key"
        if self.version is None:
            ok, status = self.available(self.codex_bin)
            self.version = status if ok else f"unavailable: {status}"
        started = time.perf_counter()
        try:
            # stdin must be closed: `codex exec` reads a non-TTY stdin for "additional input" and an
            # inherited pipe never ends.
            proc = subprocess.run(argv, cwd=workdir, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=self.timeout_s, check=False)
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr, returncode = "", f"codex timed out after {self.timeout_s}s", 124
        except OSError as exc:
            stdout, stderr, returncode = "", f"{type(exc).__name__}: {exc}", 127
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        events = self.read_events(stdout)
        meta = {"backend": "codex", "model": self.model, "effort": self.effort, "codex_version": self.version,
                "returncode": returncode, "elapsed_ms": elapsed_ms, "usage": events["usage"], "items": events["items"],
                "tool_items": sum(events["breaches"].values()), "event_errors": events["errors"][:5],
                "stderr": (stderr or "")[:1000], "stdout_tail": (stdout or "")[-1000:]}
        text = out_path.read_text() if out_path.exists() else ""
        parsed = _parse_json_object(text) or _parse_json_object(events["last_text"])
        rate_limited = parsed is None and bool(cc_isolated.RATE_LIMIT_RE.search((stderr or "") + (stdout or "")))
        if returncode != 0 and parsed is None:
            return BackendResult(False, None, error=f"codex exit {returncode}: {(stderr or stdout or '').strip()[:300]}", meta=meta,
                                 rate_limited=rate_limited)
        if parsed is None:
            return BackendResult(False, None, error="no JSON object in codex output", meta=meta, rate_limited=rate_limited)
        if events["breaches"]:
            used = ", ".join(f"{k}×{v}" for k, v in sorted(events["breaches"].items()))
            return BackendResult(False, None, error=f"isolation breach: codex used {used}", meta=meta)
        return BackendResult(True, parsed, meta=meta)


class ClaudeCodeBackend:
    """Tool-isolated `claude -p` (cc_isolated). The FALLBACK: same model family as the extractor."""

    label = "claude-code"
    family = "anthropic"
    independent_of_extractor = False

    def __init__(self, claude_bin: str = "claude", model: str = cc_isolated.DEFAULT_MODEL, timeout_s: int = 300,
                 isolation: str = "both", output_format: str = "json") -> None:
        self.claude_bin = claude_bin
        self.model = model
        self.timeout_s = timeout_s
        self.isolation = isolation
        self.output_format = output_format
        self.version = None

    def run(self, *, system_prompt: str, user_prompt: str, schema: dict[str, Any], workdir: Path) -> BackendResult:
        result = cc_isolated.run(claude_bin=self.claude_bin, prompt=user_prompt, system_prompt=system_prompt, schema=schema,
                                 model=self.model, cwd=workdir, timeout_s=self.timeout_s, output_format=self.output_format,
                                 isolation=self.isolation)
        shutil.rmtree(workdir, ignore_errors=True)
        meta = {"backend": "claude-code", "model": self.model, **result.accounting()}
        if not result.ok:
            return BackendResult(False, None, error=result.error, meta=meta, rate_limited=result.rate_limited)
        return BackendResult(True, result.parsed, meta=meta)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    body = (text or "").strip()
    if not body:
        return None
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    try:
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(body[start : end + 1])
        except ValueError:
            return None
        return data if isinstance(data, dict) else None


def validate_accuracy(parsed: dict[str, Any] | None) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not isinstance(parsed, dict):
        return False, None, "not an object"
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        return False, None, f"verdict {parsed.get('verdict')!r} not in correct|incorrect"
    try:
        cells_hit = int(parsed.get("cells_hit") or 0)
    except (TypeError, ValueError):
        return False, None, "cells_hit not an int"
    justification = str(parsed.get("justification") or "").strip().splitlines()[0] if parsed.get("justification") else ""
    return True, {"correct": verdict == "correct", "cells_hit": cells_hit, "justification": justification[:300]}, None


def validate_ar(parsed: dict[str, Any] | None, n_snippets: int) -> tuple[bool, list[int] | None, str, str | None]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("bearing_ranks"), list):
        return False, None, "", "bearing_ranks missing"
    ranks = []
    for item in parsed["bearing_ranks"]:
        try:
            rank = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= rank <= n_snippets:
            ranks.append(rank)
    return True, sorted(set(ranks)), str(parsed.get("note") or "")[:200], None


def make_backend(name: str, **kwargs: Any) -> Backend | None:
    if name == "none":
        return None
    if name == "codex":
        return CodexBackend(codex_bin=kwargs.get("codex_bin", "codex"), model=kwargs.get("codex_model"), timeout_s=kwargs.get("timeout", 300))
    if name == "claude-code":
        return ClaudeCodeBackend(claude_bin=kwargs.get("claude_bin", "claude"), model=kwargs.get("claude_model", cc_isolated.DEFAULT_MODEL),
                                 timeout_s=kwargs.get("timeout", 300))
    if name == "fake":
        return FakeBackend(kwargs.get("fake_script"))
    raise ValueError(f"unknown backend {name!r}")


def judge_answer(*, question: str, expected: str, cells: list[dict[str, Any]], answer: str, backend: Backend | None,
                 workdir: Path, rate_limit_wait: int = 0, rate_limit_retries: int = 0) -> dict[str, Any]:
    """Both layers for one answer, plus the final verdict under the stated policy."""
    layer_a = layer_a_verdict(answer, cells, expected)
    layer_b: dict[str, Any] | None = None
    if backend is not None and (answer or "").strip():
        attempts = 0
        while True:
            attempts += 1
            result = backend.run(system_prompt=ACCURACY_RUBRIC, user_prompt=accuracy_payload(question, expected, cells, answer),
                                 schema=ACCURACY_SCHEMA, workdir=workdir)
            if result.rate_limited and attempts <= rate_limit_retries:
                time.sleep(rate_limit_wait)
                continue
            break
        valid, verdict, why = validate_accuracy(result.parsed) if result.ok else (False, None, result.error)
        layer_b = {
            "layer": "B",
            "backend": backend.label,
            "family": backend.family,
            "independent_of_extractor": backend.independent_of_extractor,
            "rubric_sha256": RUBRIC_SHA256,
            "ok": bool(result.ok and valid),
            "error": None if (result.ok and valid) else why,
            "raw": result.parsed if not (result.ok and valid) else None,
            "attempts": attempts,
            **(verdict or {"correct": None, "cells_hit": None, "justification": ""}),
            "meta": result.meta,
        }
    elif backend is not None:
        layer_b = {"layer": "B", "backend": backend.label, "family": backend.family, "independent_of_extractor": backend.independent_of_extractor,
                   "rubric_sha256": RUBRIC_SHA256, "ok": True, "correct": False, "cells_hit": 0, "justification": "empty answer: not sent to the judge",
                   "error": None, "skipped": "empty answer", "attempts": 0, "meta": {}}
    return finalize(layer_a, layer_b, backend is not None)


def finalize(layer_a: dict[str, Any], layer_b: dict[str, Any] | None, has_backend: bool) -> dict[str, Any]:
    if not has_backend:
        return {"correct": layer_a["correct"], "cells_hit": layer_a["cells_hit"], "note": layer_a["note"], "policy": "layer_a_only",
                "layer_a": layer_a, "layer_b": None, "agreement": None, "judge_failed": False}
    assert layer_b is not None
    if layer_b.get("ok"):
        final = bool(layer_b["correct"])
        agreement = final == bool(layer_a["correct"])
        return {"correct": final, "cells_hit": layer_b.get("cells_hit"), "note": layer_b.get("justification") or layer_a["note"],
                "policy": "layer_b_primary", "layer_a": layer_a, "layer_b": layer_b, "agreement": agreement, "judge_failed": False}
    return {"correct": layer_a["correct"], "cells_hit": layer_a["cells_hit"], "note": f"layer B failed ({layer_b.get('error')}); layer A decides",
            "policy": "layer_b_primary", "layer_a": layer_a, "layer_b": layer_b, "agreement": None, "judge_failed": True}


def judge_ar(*, question: str, expected: str, cells: list[dict[str, Any]], hits: list[dict[str, Any]], backend: Backend | None,
             workdir: Path) -> dict[str, Any]:
    hits = list(hits or [])[:10]
    ranks_a = layer_a_bearing_ranks(hits, cells, expected)
    out: dict[str, Any] = {"layer_a_ranks": ranks_a, "layer_a": ar_flags(ranks_a), "n_snippets": len(hits), "layer_b_ranks": None, "layer_b": None}
    if backend is not None and hits:
        result = backend.run(system_prompt=AR_RUBRIC, user_prompt=ar_payload(question, expected, cells, hits), schema=AR_SCHEMA, workdir=workdir)
        valid, ranks_b, note, why = validate_ar(result.parsed, len(hits)) if result.ok else (False, None, "", result.error)
        out["layer_b_ranks"] = ranks_b
        out["layer_b"] = {**(ar_flags(ranks_b) if ranks_b is not None else {}), "ok": bool(result.ok and valid), "error": None if valid else why,
                          "note": note, "backend": backend.label, "rubric_sha256": AR_RUBRIC_SHA256, "meta": result.meta}
    ranks = out["layer_b_ranks"] if out["layer_b_ranks"] is not None else ranks_a
    out["bearing_ranks"] = ranks
    out.update(ar_flags(ranks))
    out["source"] = "layer_b" if out["layer_b_ranks"] is not None else "layer_a"
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Citation validity and the run-level driver
# ═══════════════════════════════════════════════════════════════════════════


def norm_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{parsed.path.rstrip('/')}"


def citation_valid(cited_url: str, hits: list[dict[str, Any]]) -> bool | None:
    """True when the cited URL is one of the supplied hits (host+path); None when nothing was cited."""
    if not (cited_url or "").strip():
        return None
    wanted = norm_url(cited_url)
    return any(norm_url(str(h.get("url") or "")) == wanted for h in (hits or []))


def ckpt_path(run_dir: Path, case_id: str, endpoint: str) -> Path:
    return run_dir / "judge" / f"{case_id}__{endpoint}.json"


def load_run(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    for name in ("run.json", "run.partial.json"):
        path = run_dir / name
        if path.exists():
            return path, json.loads(path.read_text())
    raise SystemExit(f"no run.json in {run_dir}")


def jobs_from_run(payload: dict[str, Any], samples: dict[str, dict[str, Any]] | None, *, endpoints: set[str] | None, ids: set[str] | None) -> list[dict[str, Any]]:
    jobs = []
    for case in payload.get("cases") or []:
        if ids and case.get("id") not in ids:
            continue
        sample = (samples or {}).get(case.get("id")) or {}
        cells = sample.get("cells") or (sample.get("gold") or {}).get("cells") or case.get("cells") or []
        expected = sample.get("expected_answer") or sample.get("ground_truth") or case.get("expected_answer") or case.get("ground_truth") or ""
        for endpoint, block in (case.get("vendors") or {}).items():
            if endpoints and endpoint not in endpoints:
                continue
            jobs.append({"case_id": case["id"], "endpoint": endpoint, "question": case["question"], "expected": expected, "cells": cells,
                         "answer": (block.get("answer") or "").strip(), "cited_url": (block.get("cited_url") or "").strip(),
                         "hits": list(block.get("hits") or [])[:10], "search_error": block.get("error"),
                         "extract_status": (block.get("extract") or {}).get("status")})
    return jobs


def run_job(job: dict[str, Any], *, run_dir: Path, backend: Backend | None, ar_semantic: bool, rate_limit_wait: int, rate_limit_retries: int) -> dict[str, Any]:
    dest = ckpt_path(run_dir, job["case_id"], job["endpoint"])
    if dest.exists():
        return json.loads(dest.read_text())
    workdir = run_dir / "judge" / "jobs" / f"{job['case_id']}__{job['endpoint']}"
    gold = judge_answer(question=job["question"], expected=job["expected"], cells=job["cells"], answer=job["answer"], backend=backend,
                        workdir=workdir / "acc", rate_limit_wait=rate_limit_wait, rate_limit_retries=rate_limit_retries)
    ar = judge_ar(question=job["question"], expected=job["expected"], cells=job["cells"], hits=job["hits"],
                  backend=backend if ar_semantic else None, workdir=workdir / "ar")
    shutil.rmtree(workdir, ignore_errors=True)
    row = {
        "case_id": job["case_id"],
        "endpoint": job["endpoint"],
        "answer": job["answer"],
        "cited_url": job["cited_url"],
        "citation_valid": citation_valid(job["cited_url"], job["hits"]),
        "gold": gold,
        "ar": ar,
        "search_error": bool(job["search_error"]),
        "extract_status": job["extract_status"],
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    return row


def apply(run_dir: Path, *, backend_label: str) -> dict[str, Any]:
    run_path, payload = load_run(run_dir)
    backup = run_dir / "run.before-judge.json"
    if not backup.exists():
        shutil.copy2(run_path, backup)
    applied = missing = 0
    for case in payload.get("cases") or []:
        for endpoint, block in (case.get("vendors") or {}).items():
            path = ckpt_path(run_dir, case["id"], endpoint)
            if not path.exists():
                missing += 1
                continue
            row = json.loads(path.read_text())
            block["gold"] = row["gold"]
            block["ar"] = row["ar"]
            block["answer_in_excerpt"] = bool(row["ar"].get("ar10"))
            block["citation_valid"] = row.get("citation_valid")
            block["judged_at"] = row.get("judged_at")
            case.setdefault("gold_scores", {})[endpoint] = {"correct": row["gold"]["correct"], "cells_hit": row["gold"].get("cells_hit"),
                                                             "note": row["gold"].get("note")}
            applied += 1
    payload["judge"] = {"layer_a": "deterministic (jevlin_bench/judge.py)", "layer_b": backend_label, "policy": "layer_b_primary" if backend_label != "none" else "layer_a_only",
                        "rubric_sha256": RUBRIC_SHA256, "ar_rubric_sha256": AR_RUBRIC_SHA256, "applied_at": datetime.now(timezone.utc).isoformat()}
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return {"applied": applied, "missing": missing, "backup": str(backup)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=ROOT / "data" / "company-news" / "samples.json")
    parser.add_argument("--layer-b", choices=("codex", "claude-code", "none", "fake"), default="codex")
    parser.add_argument("--ar-semantic", action="store_true", help="Also ask Layer B which snippets are answer-bearing.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--endpoints", default="")
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN") or "codex",
                        help="codex executable; default $CODEX_BIN or `codex` on PATH (the ChatGPT app bundles one under "
                             "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex)")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--claude-model", default=cc_isolated.DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--rate-limit-wait", type=int, default=600)
    parser.add_argument("--rate-limit-retries", type=int, default=6)
    parser.add_argument("--allow-fallback", action="store_true", help="If codex is not operational, use claude-code instead of stopping.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    _, payload = load_run(run_dir)
    samples = {s["id"]: s for s in json.loads(args.samples.read_text())} if args.samples.exists() else None
    endpoints = {e.strip() for e in args.endpoints.split(",") if e.strip()} or None
    ids = {i.strip() for i in args.ids.split(",") if i.strip()} or None
    jobs = jobs_from_run(payload, samples, endpoints=endpoints, ids=ids)
    pending = [j for j in jobs if not ckpt_path(run_dir, j["case_id"], j["endpoint"]).exists()]
    if args.limit:
        pending = pending[: args.limit]
    layer_b = args.layer_b
    codex_status = None
    if layer_b == "codex":
        ok, codex_status = CodexBackend.available(args.codex_bin)
        if not ok:
            if args.allow_fallback:
                print(f"codex not operational ({codex_status}); FALLING BACK to claude-code (same family as the extractor)", flush=True)
                layer_b = "claude-code"
            elif args.run:
                print(f"codex not operational: {codex_status}. Use --layer-b claude-code (labelled fallback) or --allow-fallback.", file=sys.stderr)
                return 2
    plan = {"run_dir": str(run_dir), "jobs": len(jobs), "checkpointed": len(jobs) - len([j for j in jobs if not ckpt_path(run_dir, j['case_id'], j['endpoint']).exists()]),
            "pending": len(pending), "pending_nonempty_answers": sum(1 for j in pending if j["answer"]), "layer_b": layer_b, "codex_status": codex_status,
            "ar_semantic": args.ar_semantic, "rubric_sha256": RUBRIC_SHA256, "ar_rubric_sha256": AR_RUBRIC_SHA256, "api_key_calls": 0, "vendor_calls": 0}
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        backend = make_backend(layer_b, codex_bin=args.codex_bin, codex_model=args.codex_model, claude_bin=args.claude_bin,
                               claude_model=args.claude_model, timeout=args.timeout)
        counts = {"correct": 0, "incorrect": 0, "judge_failed": 0, "disagree": 0}
        for n, job in enumerate(pending, 1):
            row = run_job(job, run_dir=run_dir, backend=backend, ar_semantic=args.ar_semantic, rate_limit_wait=args.rate_limit_wait,
                          rate_limit_retries=args.rate_limit_retries)
            gold = row["gold"]
            counts["correct" if gold["correct"] else "incorrect"] += 1
            counts["judge_failed"] += 1 if gold.get("judge_failed") else 0
            counts["disagree"] += 1 if gold.get("agreement") is False else 0
            print(json.dumps({"n": n, "case": job["case_id"], "endpoint": job["endpoint"], "correct": gold["correct"], "A": gold["layer_a"]["correct"],
                              "B": (gold.get("layer_b") or {}).get("correct"), "ar10": row["ar"].get("ar10"), "note": str(gold.get("note"))[:80]},
                             ensure_ascii=False), flush=True)
        print(json.dumps({"finished": True, **counts}), flush=True)
    if args.apply:
        print(json.dumps(apply(run_dir, backend_label=layer_b), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
