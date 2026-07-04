"""Unit tests for stream_filter.is_expired_stream_url — offline only."""
from __future__ import annotations

from dashboard.api.stream_resolver.stream_filter import is_expired_stream_url

# Fixed "now" so the tests never rot: 2026-07-04T00:00:00Z.
NOW = 1_782_950_400.0
PAST_S = "1765241578"      # 2025-12-09 — the observed months-dead carvoni link
FUTURE_S = "1799999999"    # 2027-01
PAST_MS = "1765241578000"
FUTURE_MS = "1799999999000"


def test_expired_seconds_epoch_dropped() -> None:
    url = f"https://sq3.carvoni.cyou/v4/mik/cf-master.txt?t=abc&e={PAST_S}"
    assert is_expired_stream_url(url, now=NOW) is True


def test_future_seconds_epoch_kept() -> None:
    url = f"https://cdn.example.net/master.m3u8?token=abc&expires={FUTURE_S}"
    assert is_expired_stream_url(url, now=NOW) is False


def test_expired_milliseconds_epoch_dropped() -> None:
    url = f"https://cdn.example.net/master.m3u8?exp={PAST_MS}"
    assert is_expired_stream_url(url, now=NOW) is True


def test_future_milliseconds_epoch_kept() -> None:
    url = f"https://cdn.example.net/master.m3u8?exp={FUTURE_MS}"
    assert is_expired_stream_url(url, now=NOW) is False


def test_no_expiry_params_kept() -> None:
    assert is_expired_stream_url(
        "https://cdn.example.net/master.m3u8?token=abc&user=x", now=NOW
    ) is False


def test_non_numeric_expiry_value_kept() -> None:
    assert is_expired_stream_url(
        "https://cdn.example.net/master.m3u8?e=abcdef1234", now=NOW
    ) is False


def test_implausible_epoch_length_kept() -> None:
    # 6 digits is not a plausible epoch — e= means something else here.
    assert is_expired_stream_url(
        "https://cdn.example.net/master.m3u8?e=123456", now=NOW
    ) is False


def test_unrelated_past_looking_param_kept() -> None:
    # Past epoch under a NON-expiry param name must not reject.
    assert is_expired_stream_url(
        f"https://cdn.example.net/master.m3u8?start={PAST_S}", now=NOW
    ) is False
