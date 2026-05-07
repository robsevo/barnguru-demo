"""Unit tests for stream_resolver — offline only."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from dashboard.api.stream_resolver.health import HealthStore
from dashboard.api.stream_resolver.patterns import PatternStore


def test_pattern_store_loads_default() -> None:
    """The shipped provider_patterns.json loads cleanly with all 4 providers."""
    ps = PatternStore()
    providers = ps.all_providers()
    assert "vidsrc" in providers
    assert "bflix" in providers
    assert "fmoviesz" in providers
    assert "yesmovies" in providers


def test_pattern_store_returns_empty_for_unknown() -> None:
    ps = PatternStore()
    assert ps.get("nonexistent") == {}


def test_pattern_store_patch_bumps_version(tmp_path: Path) -> None:
    """Patching a provider increments pattern_version + persists atomically."""
    p = tmp_path / "patterns.json"
    p.write_text(json.dumps({
        "schema_version": 2,
        "providers": {"x": {"pattern_version": 5, "base_urls": ["a"]}},
    }))
    ps = PatternStore(path=p)
    new_v = ps.patch("x", {"base_urls": ["b"]}, reason="test")
    assert new_v == 6
    # Re-read from disk to confirm persistence
    saved = json.loads(p.read_text())
    assert saved["providers"]["x"]["pattern_version"] == 6
    assert saved["providers"]["x"]["base_urls"] == ["b"]
    assert saved["providers"]["x"]["last_patch_reason"] == "test"


def test_pattern_store_patch_creates_provider_if_missing(tmp_path: Path) -> None:
    p = tmp_path / "patterns.json"
    p.write_text(json.dumps({"schema_version": 2, "providers": {}}))
    ps = PatternStore(path=p)
    new_v = ps.patch("brand_new", {"base_urls": ["a"]}, reason="bootstrap")
    assert new_v == 1
    saved = json.loads(p.read_text())
    assert saved["providers"]["brand_new"]["base_urls"] == ["a"]


def test_health_store_records(tmp_path: Path) -> None:
    """Health store rolls up recent successes/failures correctly."""
    os.environ["GRETZKY_DATA_DIR"] = str(tmp_path)
    h = HealthStore()
    h.record_success("p", base_url="https://a")
    h.record_success("p", base_url="https://a")
    h.record_failure("p", "boom")
    pH = h.get("p")
    assert pH.rolling_successes == 2
    assert pH.rolling_failures == 1
    assert pH.last_failure_reason == "boom"
    assert pH.consecutive_silent_empty == 0


def test_health_store_silent_empty_count(tmp_path: Path) -> None:
    os.environ["GRETZKY_DATA_DIR"] = str(tmp_path)
    h = HealthStore()
    h.record_failure("p", "no_sources", silent_empty=True)
    h.record_failure("p", "no_sources", silent_empty=True)
    h.record_failure("p", "no_sources", silent_empty=True)
    assert h.needs_silent_empty_recycle("p", threshold=3)
    h.record_success("p")
    assert not h.needs_silent_empty_recycle("p", threshold=3)


def test_health_store_persists_atomic(tmp_path: Path) -> None:
    os.environ["GRETZKY_DATA_DIR"] = str(tmp_path)
    h = HealthStore()
    h.record_success("p", base_url="https://a")
    state_path = tmp_path / "stream_resolver" / "state.json"
    assert state_path.exists()
    raw = json.loads(state_path.read_text())
    assert raw["providers"]["p"]["rolling_window"]["successes"] == 1


def test_health_success_rate_uses_bayes_prior(tmp_path: Path) -> None:
    """A provider with no history reports near 50% so it gets a fair shot.
    A 2-failure provider reports below 50% but not zero."""
    os.environ["GRETZKY_DATA_DIR"] = str(tmp_path)
    h = HealthStore()
    fresh = h.get("never_tried")
    assert 0.4 < fresh.success_rate < 0.6
    h.record_failure("p", "boom")
    h.record_failure("p", "boom")
    after_two = h.get("p").success_rate
    assert after_two < 0.5
    assert after_two > 0.2


def test_health_needs_rediscovery_only_with_enough_attempts(tmp_path: Path) -> None:
    os.environ["GRETZKY_DATA_DIR"] = str(tmp_path)
    h = HealthStore()
    # 3 failures shouldn't trigger rediscovery (below min_attempts)
    for _ in range(3):
        h.record_failure("p", "boom")
    assert not h.needs_rediscovery("p", min_attempts=20, rate_threshold=0.30)
    # 25 failures: should trigger
    for _ in range(22):
        h.record_failure("p", "boom")
    assert h.needs_rediscovery("p", min_attempts=20, rate_threshold=0.30)
