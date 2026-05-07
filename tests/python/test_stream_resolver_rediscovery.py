"""Offline tests for rediscovery._diff_endpoints.

Covers the three diff outcomes:
  - clean (no drift)
  - safe patch (URL params rotated, structurally same)
  - structural change (path moved or response shape changed)

The browser-driven trace capture is exercised in the Phase F live tests,
not here — this module locks the diff logic.
"""
from __future__ import annotations

from dashboard.api.stream_resolver.rediscovery import (
    Rediscovery, RediscoveryDiff, _DiffResult,
)


def _make_rediscovery() -> Rediscovery:
    """Build a Rediscovery with stub deps. We only need _diff_endpoints
    for these tests — no real browser/health/patterns needed."""
    return Rediscovery(browser=None, patterns=None, health=None)  # type: ignore[arg-type]


def test_diff_clean_when_no_drift() -> None:
    r = _make_rediscovery()
    cfg = {
        "ajax_chain": [
            {"step": "episodes", "path": "/ajax/episodes/list",
             "params": {"id": "{film_id}", "_": "{token}"}},
        ],
    }
    observed = ["https://x.com/ajax/episodes/list?id=abc&_=t"]
    result = r._diff_endpoints(cfg, observed, "x")
    assert not result.patches
    assert not result.structural_changes


def test_diff_no_chain_baseline() -> None:
    """Provider with no stored chain — rediscovery is a no-op."""
    r = _make_rediscovery()
    result = r._diff_endpoints({}, ["https://x.com/anything"], "x")
    assert "no_chain_baseline" in result.patches
    assert not result.structural_changes


def test_diff_token_rotation_patches() -> None:
    """A new param appeared on a known endpoint — safe to patch."""
    r = _make_rediscovery()
    cfg = {
        "ajax_chain": [
            {"step": "view", "path": "/ajax/links/view",
             "params": {"id": "{link_id}"}},
        ],
    }
    # Site rotated to require a `nonce=` param
    observed = ["https://x.com/ajax/links/view?id=abc&nonce=xyz&_=t"]
    result = r._diff_endpoints(cfg, observed, "x")
    assert any("added params" in p for p in result.patches)
    # Patch payload should preserve existing params and add the new one
    chain = result.patch_payload["ajax_chain"]
    assert chain[0]["path"] == "/ajax/links/view"
    assert "nonce" in chain[0]["params"]
    assert chain[0]["params"]["id"] == "{link_id}"


def test_diff_path_moved_is_structural() -> None:
    """A known path is gone but a sibling appeared — operator-paged."""
    r = _make_rediscovery()
    cfg = {
        "ajax_chain": [
            {"step": "list", "path": "/ajax/episodes/list", "params": {}},
        ],
    }
    # Site moved /ajax/episodes/list → /api/v2/episodes/list
    observed = ["https://x.com/api/v2/episodes/list?id=abc"]
    result = r._diff_endpoints(cfg, observed, "x")
    assert any("path moved" in s for s in result.structural_changes)
    assert not result.patches  # no auto-patch for structural change


def test_diff_path_missing_is_structural() -> None:
    r = _make_rediscovery()
    cfg = {
        "ajax_chain": [
            {"step": "list", "path": "/ajax/episodes/list", "params": {}},
        ],
    }
    # Nothing matching that path or its suffix
    observed = ["https://x.com/some/unrelated/url"]
    result = r._diff_endpoints(cfg, observed, "x")
    assert any("path missing" in s for s in result.structural_changes)


def test_diff_param_removed_is_structural() -> None:
    """A param we used to send is no longer accepted — operator-paged."""
    r = _make_rediscovery()
    cfg = {
        "ajax_chain": [
            {"step": "list", "path": "/ajax/episodes/list",
             "params": {"id": "{film_id}", "format": "json"}},
        ],
    }
    # Site dropped the format= param requirement
    observed = ["https://x.com/ajax/episodes/list?id=abc"]
    result = r._diff_endpoints(cfg, observed, "x")
    assert any("removed" in s for s in result.structural_changes)


def test_rediscovery_diff_to_dict_serializes_cleanly() -> None:
    d = RediscoveryDiff(provider_id="x", pattern_was_patched=True,
                         patches=["a"], structural_changes=["b"],
                         captured_endpoints=["c"], summary="patched")
    out = d.to_dict()
    assert out["provider_id"] == "x"
    assert out["patches"] == ["a"]
    assert out["structural_changes"] == ["b"]
    assert out["summary"] == "patched"


def test_rediscovery_diff_is_clean_only_when_empty() -> None:
    assert RediscoveryDiff(provider_id="x", pattern_was_patched=False).is_clean()
    assert not RediscoveryDiff(provider_id="x", pattern_was_patched=False,
                                patches=["a"]).is_clean()
    assert not RediscoveryDiff(provider_id="x", pattern_was_patched=False,
                                structural_changes=["b"]).is_clean()
