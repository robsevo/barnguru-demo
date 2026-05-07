"""Offline tests for stream_filter.is_real_m3u8_url.

Locks in the rejection of tracker false positives that bit us during
Phase C development."""
from __future__ import annotations

from dashboard.api.stream_resolver.stream_filter import is_real_m3u8_url


def test_accepts_neonhorizon() -> None:
    assert is_real_m3u8_url(
        "https://tmstr4.neonhorizonworkshops.com/pl/ABC/master.m3u8"
    )


def test_accepts_path_ending_in_m3u8() -> None:
    assert is_real_m3u8_url("https://x.streamtape.com/abc.m3u8")


def test_accepts_master_m3u8_in_path() -> None:
    assert is_real_m3u8_url("https://cdn.bunnycdn.com/foo/master.m3u8?token=xyz")


def test_accepts_playlist_m3u8() -> None:
    assert is_real_m3u8_url("https://cdn.b-cdn.net/foo/playlist.m3u8")


def test_rejects_jwplayer_ping_gif_with_m3u8_in_query() -> None:
    """The exact false positive that we observed — JW Player tracker
    embeds the segment URL in its analytics ping query string."""
    assert not is_real_m3u8_url(
        "https://prd.jwpltx.com/v1/jwplayer6/ping.gif?h=-951059448&e=pa&n=98&u=master.m3u8"
    )


def test_rejects_unknown_host_even_if_path_correct() -> None:
    """Unknown hosts (not on our allowlist) get rejected. The point of
    the allowlist is to gate which CDNs we trust as legitimate streams,
    even when the URL path looks correct."""
    assert not is_real_m3u8_url("https://made-up-tracker.example/hls/abc.m3u8")


def test_accepts_ployan_me() -> None:
    """ployan.me has been observed serving real HLS for vidsrc resolutions.
    Locked in the allowlist."""
    assert is_real_m3u8_url("https://ployan.me/hls/abc.m3u8")


def test_rejects_when_no_m3u8_at_all() -> None:
    assert not is_real_m3u8_url("https://example.com/foo.bar")


def test_rejects_path_only_m3u8_extension_substring() -> None:
    """Path contains 'm3u8' but doesn't end in '.m3u8' — reject."""
    assert not is_real_m3u8_url(
        "https://prd.jwpltx.com/m3u8_tracker.gif?p=master.m3u8"
    )


def test_accepts_neonhorizon_with_subdomain_pattern() -> None:
    """The actual production captured host has a numbered subdomain."""
    assert is_real_m3u8_url(
        "https://tmstr5.neonhorizonworkshops.com/pl/H4sIAAAA/master.m3u8"
    )


def test_rejects_doubleclick_even_with_path() -> None:
    """doubleclick.net is in TRACKER_HOST_PATTERNS — must reject."""
    assert not is_real_m3u8_url(
        "https://stats.doubleclick.net/ping/master.m3u8?ad=1"
    )
