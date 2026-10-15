"""Jevlin × OpenBenchmarks Company News — internal Claude/Codex adaptation (jevlin-backend fork).

Benchmark-only code that replaces the two paid model stages of the upstream chain
(gpt-5.6-terra extractor, Opus-on-Bedrock judge) with subscription-billed, tool-isolated
local runners, and adds the measures the harness's own columns cannot carry for a metered
gateway arm. Nothing here changes the search protocol or the upstream vendor adapters.
"""
