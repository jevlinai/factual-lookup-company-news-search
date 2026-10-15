"""Jevlin core, vendored for the company-news harness. Benchmark-only code.

Source: ``jevlinai/web-search-for-coding-agents`` branch ``jevlin-backend`` at commit
**f01b032**, file ``scripts/coding_search/jevlin.py``. Copied, not imported, per the shared
conventions ("copy, do not import across repositories; record the source commit in the
vendored module's docstring"). The request body, the pinned-provider rule, both
presentations (``present_merged`` / ``present_raw``), ``arm_summary`` and ``jevlin_meta``
are byte-for-byte the f01b032 logic. What is *not* vendored: the coding harness's
``JevlinSearch.search`` (bound to that harness's ``vendor_call``), ``JevlinMax`` (not an arm
in this suite) and ``JevlinCli`` (agent suites only; this suite has no agent).

Jevlin is a multi-provider search gateway: one ``POST /v1/search`` fans out to a
tier-defined set of upstream providers and returns every provider's results side by side;
with ``view:"merged"`` it also returns one deduplicated list of pages ordered by how many
providers returned each page.

Arms (conventions.md):

| arm                  | request                                   | presentation                          |
|----------------------|-------------------------------------------|---------------------------------------|
| jevlin_fast          | tier=fast, account default providers      | gateway merged order                  |
| jevlin_balanced      | tier=balanced, account default providers  | gateway merged order                  |
| jevlin_single        | providers=[JEVLIN_SINGLE_PROVIDER], no tier | that provider's own order (raw)     |
| jevlin_fanout_raw    | tier=balanced, providers=JEVLIN_PROVIDERS | round-robin over providers, exact-URL dedup |
| jevlin_fanout_merged | tier=balanced, providers=JEVLIN_PROVIDERS | gateway merged order                  |

``jevlin_fanout_*`` refuse to run without ``JEVLIN_PROVIDERS``: the ablation must pin the
set so a configuration change on the server cannot move it between passes. The tier arms
deliberately do not pin, because they measure what a customer gets.

Environment:
  JEVLIN_API_KEY          bearer key for the HTTP arms
  JEVLIN_API_URL          base URL, default https://api.jevlin.ai
  JEVLIN_PROVIDERS        comma list, required by jevlin_fanout_*
  JEVLIN_SINGLE_PROVIDER  provider name for jevlin_single, default exa
  JEVLIN_TIMEOUT_S        HTTP timeout, default 45 (Jevlin's own fan-out deadline is 30 s)
"""

from __future__ import annotations

import os
from typing import Any

SOURCE_COMMIT = "f01b032"
SOURCE_REPO = "jevlinai/web-search-for-coding-agents@jevlin-backend scripts/coding_search/jevlin.py"

DEFAULT_API_URL = "https://api.jevlin.ai"
SEARCH_PATH = "/v1/search"
DEFAULT_TIMEOUT_S = 45.0
SNIPPET_CHARS = 1200
# Statuses the gateway assigns an arm it never called. Everything else was an
# upstream request, billed or not.
NOT_CALLED = frozenset({"skipped"})


def api_url() -> str:
    base = (os.environ.get("JEVLIN_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    return base + SEARCH_PATH


def timeout_s() -> float:
    raw = (os.environ.get("JEVLIN_TIMEOUT_S") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S


def pinned_providers() -> list[str]:
    raw = os.environ.get("JEVLIN_PROVIDERS") or ""
    out: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in out:
            out.append(name)
    return out


# ── presentation: Jevlin response → url / title / snippet ──────────────────


def _hit(url: str, title: Any, snippet: Any) -> dict[str, str]:
    return {"url": url, "title": str(title or ""), "snippet": str(snippet or "")[:SNIPPET_CHARS]}


def present_merged(payload: Any, *, max_results: int) -> list[dict[str, str]]:
    """Jevlin's `merged[]`, in Jevlin's order, cut at max_results.

    The order is the gateway's: pages returned by more providers first, then
    best per-provider rank, then provider order. The excerpt is the longest
    snippet any provider gave for the page; when only `excerpts` (quoted
    evidence) exist they are joined.
    """
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in (payload or {}).get("merged") or []:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = page.get("excerpt") or "\n".join(
            part for part in (page.get("excerpts") or []) if isinstance(part, str)
        )
        hits.append(_hit(url, page.get("title"), snippet))
        if len(hits) >= max_results:
            break
    return hits


def present_raw(payload: Any, *, max_results: int) -> list[dict[str, str]]:
    """Fan-out with the least processing that yields one list.

    Round-robin over the providers that answered, in the gateway's dispatch
    order: every provider's first result, then every provider's second, and
    so on. Duplicates are dropped only on the exact URL string the providers
    returned (each adapter has already canonicalised its own citations). No
    vote count, no re-ranking, no snippet choice: the first provider to show a
    page supplies its title and snippet.
    """
    lists: list[list[dict[str, Any]]] = []
    for cand in (payload or {}).get("candidates") or []:
        if not isinstance(cand, dict) or cand.get("status") != "ok":
            continue
        cits = [c for c in (cand.get("citations") or []) if isinstance(c, dict) and c.get("url")]
        if cits:
            lists.append(cits)
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    depth = max((len(cits) for cits in lists), default=0)
    for rank in range(depth):
        for cits in lists:
            if rank >= len(cits):
                continue
            cit = cits[rank]
            url = str(cit.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            hits.append(_hit(url, cit.get("title"), cit.get("snippet")))
            if len(hits) >= max_results:
                return hits
    return hits


PRESENTATIONS = {"merged": present_merged, "raw": present_raw}


# ── metadata: what the harness's own fields cannot carry ───────────────────


def arm_summary(payload: Any) -> list[dict[str, Any]]:
    """One row per dispatched arm: provider, status, results, cost, latency."""
    reasons: dict[str, str] = {}
    indexes = (payload or {}).get("indexes") or {}
    for row in indexes.get("by_provider") or []:
        if isinstance(row, dict) and row.get("provider") and row.get("reason"):
            reasons[str(row["provider"])] = str(row["reason"])
    arms: list[dict[str, Any]] = []
    for cand in (payload or {}).get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        latency = cand.get("latency") if isinstance(cand.get("latency"), dict) else {}
        provider = str(cand.get("provider") or "")
        arms.append({
            "provider": provider,
            "kind": cand.get("kind"),
            "status": cand.get("status"),
            "reason": reasons.get(provider),
            "results": len(cand.get("citations") or []),
            "has_answer": bool(cand.get("answer")),
            "cost_micros": cand.get("cost_micros"),
            "latency_ms": latency.get("total_ms"),
            "error": cand.get("error"),
        })
    return arms


def jevlin_meta(payload: Any, hits: list[dict[str, Any]], *, presentation: str) -> dict[str, Any]:
    decision = (payload or {}).get("decision") or {}
    usage = (payload or {}).get("usage") or {}
    arms = arm_summary(payload)
    merged = (payload or {}).get("merged")
    return {
        "jevlin_request_id": str((payload or {}).get("request_id") or ""),
        "jevlin_presentation": presentation,
        "jevlin_tier": decision.get("tier"),
        "jevlin_mode": decision.get("mode"),
        "jevlin_provider_set": decision.get("provider_set"),
        "jevlin_providers": list(decision.get("providers") or []),
        "jevlin_trimmed": list(decision.get("trimmed") or []),
        "jevlin_requested": bool(decision.get("requested")),
        "jevlin_router_version": decision.get("router_version"),
        "jevlin_ranker_version": decision.get("ranker_version"),
        "jevlin_cost_micros": usage.get("cost_micros"),
        "jevlin_latency_ms": usage.get("latency_ms"),
        "jevlin_cache_hit": bool(usage.get("cache_hit")),
        "jevlin_deadline_hit": bool(usage.get("deadline_hit")),
        "jevlin_pending": int(usage.get("pending") or 0),
        "jevlin_no_usable_result": bool((payload or {}).get("no_usable_result")),
        "jevlin_chosen": (payload or {}).get("chosen"),
        "jevlin_upstream_calls": sum(1 for arm in arms if arm["status"] not in NOT_CALLED),
        "jevlin_arms_ok": sum(1 for arm in arms if arm["status"] == "ok"),
        "jevlin_arms": arms,
        "jevlin_merged_total": len(merged) if isinstance(merged, list) else None,
        "jevlin_hits_shown": len(hits),
    }


# ── arms: the request each one sends (JevlinSearch subclasses at f01b032) ──


class Arm:
    """One Jevlin request shape. ``request_body`` and ``providers`` are f01b032's."""

    def __init__(self, name: str, *, tier: str | None, presentation: str, pin_providers: bool = False,
                 single: bool = False) -> None:
        self.name = name
        self.tier = tier
        self.presentation = presentation
        self.pin_providers = pin_providers
        self.single = single

    def key(self) -> str:
        key = (os.environ.get("JEVLIN_API_KEY") or "").strip()
        if not key:
            raise RuntimeError(f"JEVLIN_API_KEY is required for {self.name}")
        return key

    def providers(self) -> list[str]:
        if self.single:
            return [(os.environ.get("JEVLIN_SINGLE_PROVIDER") or "exa").strip()]
        if not self.pin_providers:
            return []
        names = pinned_providers()
        if not names:
            raise RuntimeError(
                f"JEVLIN_PROVIDERS must name the pinned provider set for {self.name}"
            )
        return names

    def request_body(self, query: str, max_results: int) -> dict[str, Any]:
        # view:"merged" adds `merged` and `indexes` and changes nothing else in
        # the response, so every arm asks for it and the raw presentation
        # reads `candidates` from the same body.
        body: dict[str, Any] = {"query": query, "max_results": int(max_results), "view": "merged"}
        if self.tier:
            body["tier"] = self.tier
        providers = self.providers()
        if providers:
            body["providers"] = providers
        return body

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key()}", "Content-Type": "application/json"}

    def cache_key_class(self) -> str:
        """Which arms collide on the gateway cache (query + tier + provider set).

        The pinned fan-out set has equalled the resolved balanced set on every deployment
        measured so far (coding suite README §E2, §E7), so the three balanced-tier arms are
        treated as one key class. ``single`` has no tier and one provider; ``fast`` is its
        own tier.
        """
        if self.single:
            return "single"
        return str(self.tier)


ARMS: dict[str, Arm] = {
    "jevlin_fast": Arm("jevlin_fast", tier="fast", presentation="merged"),
    "jevlin_balanced": Arm("jevlin_balanced", tier="balanced", presentation="merged"),
    "jevlin_single": Arm("jevlin_single", tier=None, presentation="raw", single=True),
    "jevlin_fanout_raw": Arm("jevlin_fanout_raw", tier="balanced", presentation="raw", pin_providers=True),
    "jevlin_fanout_merged": Arm("jevlin_fanout_merged", tier="balanced", presentation="merged", pin_providers=True),
}

ARM_NAMES: tuple[str, ...] = tuple(ARMS)
