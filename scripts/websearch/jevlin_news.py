"""Jevlin arms for the company-news harness: one more vendor in ``call_endpoint``.

The harness has no vendor protocol; a vendor is a branch in
``run_official_news_probe.call_endpoint(name, question, case) -> (hits, raw)``. This module
is that branch for the five Jevlin arms. It uses the harness's own HTTP wrapper (``_http``:
retries, redaction, ``attempts[]``, ``latency_ms``), the harness's hit constructor
(``_hit``: ``host`` field, whitespace collapse, 1,200-char snippet) and the harness's
dedup (``_dedupe``: normalised host+path, cut at ``MAX_RESULTS``), so a Jevlin hit is
shaped and capped exactly like every other vendor's. The gateway is asked for
``max_results = MAX_RESULTS`` per provider (the harness's per-vendor count).

What the harness's vendor block cannot carry — which upstream providers ran, what each
cost, whether the cache answered — is attached to the raw dict as ``vendor_meta`` (from the
vendored ``jevlin_core.jevlin_meta``) and copied onto the block and the cost event by the
runner. Jevlin's observed cost (``usage.cost_micros``) is recorded beside the harness's flat
``usd`` and never summed with it: ``UNIT_COST_USD`` for these arms is ``0.0`` because there is
no list price, the gateway meters each search.

Jevlin core is vendored in ``scripts/jevlin_core`` from the coding-agents fork at f01b032.
"""

from __future__ import annotations

from typing import Any, Callable

import jevlin_core

ARMS: tuple[str, ...] = jevlin_core.ARM_NAMES

ENDPOINT_MAP: dict[str, str] = {
    "jevlin_fast": "POST {JEVLIN_API_URL}/v1/search tier=fast view=merged max_results=10 · merged order",
    "jevlin_balanced": "POST {JEVLIN_API_URL}/v1/search tier=balanced view=merged max_results=10 · merged order",
    "jevlin_single": "POST {JEVLIN_API_URL}/v1/search providers=[JEVLIN_SINGLE_PROVIDER] view=merged max_results=10 · provider order",
    "jevlin_fanout_raw": "POST {JEVLIN_API_URL}/v1/search tier=balanced providers=JEVLIN_PROVIDERS view=merged max_results=10 · round-robin, exact-URL dedup",
    "jevlin_fanout_merged": "POST {JEVLIN_API_URL}/v1/search tier=balanced providers=JEVLIN_PROVIDERS view=merged max_results=10 · merged order",
}

ENDPOINT_ENV: dict[str, tuple[str, ...]] = {
    "jevlin_fast": ("JEVLIN_API_KEY",),
    "jevlin_balanced": ("JEVLIN_API_KEY",),
    "jevlin_single": ("JEVLIN_API_KEY",),
    "jevlin_fanout_raw": ("JEVLIN_API_KEY", "JEVLIN_PROVIDERS"),
    "jevlin_fanout_merged": ("JEVLIN_API_KEY", "JEVLIN_PROVIDERS"),
}

# No list price: the gateway meters every search and the runner records the observed
# micro-dollars on the cost event (`usd_observed`). The flat figure must exist for the
# harness's accounting to run; it is nominal and is never used for a Jevlin $/1k figure.
UNIT_COST_USD: dict[str, float] = {name: 0.0 for name in ARMS}

HttpFn = Callable[..., dict[str, Any]]
HitFn = Callable[[str, str | None, str | None], dict[str, Any]]
DedupeFn = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


def is_jevlin(name: str) -> bool:
    return name in jevlin_core.ARMS


def cache_key_class(name: str) -> str:
    return jevlin_core.ARMS[name].cache_key_class()


def call_jevlin(
    name: str,
    question: str,
    *,
    http: HttpFn,
    hit: HitFn,
    dedupe: DedupeFn,
    max_results: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One Jevlin search in the harness's shape. Raises KeyError for a non-Jevlin name."""
    arm = jevlin_core.ARMS[name]
    body = arm.request_body(question, max_results)
    raw = http(
        method="POST",
        url=jevlin_core.api_url(),
        headers=arm.headers(),
        body=body,
        params=None,
        timeout=int(jevlin_core.timeout_s()),
    )
    hits: list[dict[str, Any]] = []
    payload = raw.get("response") if raw.get("ok") else None
    if raw.get("ok") and isinstance(payload, dict):
        # Present every page the gateway returned, then let the harness's own dedup
        # (normalised host+path) cut at MAX_RESULTS, exactly as it does for every vendor
        # whose parser collects all results before `_dedupe`.
        shown = jevlin_core.PRESENTATIONS[arm.presentation](payload, max_results=1 << 30)
        hits = dedupe([hit(h["url"], h["title"], h["snippet"]) for h in shown])
        raw["vendor_meta"] = jevlin_core.jevlin_meta(payload, hits, presentation=arm.presentation)
    else:
        raw["vendor_meta"] = {
            "jevlin_request_id": "",
            "jevlin_presentation": arm.presentation,
            "jevlin_tier": arm.tier,
            "jevlin_cost_micros": None,
            "jevlin_cache_hit": False,
            "jevlin_pending": 0,
            "jevlin_upstream_calls": None,
            "jevlin_hits_shown": 0,
            "jevlin_error": raw.get("error"),
        }
    raw["vendor_meta"]["jevlin_core_source"] = f"{jevlin_core.SOURCE_REPO}@{jevlin_core.SOURCE_COMMIT}"
    return hits, raw


def cost_event_fields(vendor_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Observed-cost fields for the harness's `vendor_search` cost event. Never summed into `usd`."""
    meta = vendor_meta or {}
    micros = meta.get("jevlin_cost_micros")
    return {
        "cost_micros": micros,
        "usd_observed": (round(int(micros) / 1_000_000, 6) if isinstance(micros, (int, float)) else None),
        "cache_hit": bool(meta.get("jevlin_cache_hit")),
        "pending": int(meta.get("jevlin_pending") or 0),
        "deadline_hit": bool(meta.get("jevlin_deadline_hit")),
        "upstream_calls": meta.get("jevlin_upstream_calls"),
        "arms_ok": meta.get("jevlin_arms_ok"),
        "no_usable_result": bool(meta.get("jevlin_no_usable_result")),
        "upstream_statuses": {
            str(arm.get("provider")): arm.get("status") for arm in (meta.get("jevlin_arms") or []) if isinstance(arm, dict)
        },
    }
