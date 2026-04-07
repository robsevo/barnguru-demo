"""Tests for TransactionSyncer — TTL-based Parquet persistence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from data.transaction_parser import TRANSACTION_SCHEMA, TransactionEvent
from data.transaction_sync import TransactionSyncer, TransactionSyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = "2026-03-21"
_DATE2 = "2026-03-22"


def _event(
    date: str = _DATE,
    description: str = "TOR traded Player A to EDM.",
    team: str = "TOR",
) -> TransactionEvent:
    return TransactionEvent(
        date=date,
        event_type="trade",
        team=team,
        secondary_team="EDM",
        player_or_executive="Player A",
        player_id_espn="12345",
        description=description,
        notes=None,
        fetched_at=f"{date}T10:00:00+00:00",
    )


def _make_client(raw: dict | None = None, raise_exc: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if raise_exc is not None:
        client.fetch_transactions.side_effect = raise_exc
    else:
        client.fetch_transactions.return_value = raw or {}
    return client


def _make_parser(
    events: list[TransactionEvent] | None = None,
    raise_exc: Exception | None = None,
) -> MagicMock:
    parser = MagicMock()
    if raise_exc is not None:
        parser.parse.side_effect = raise_exc
    else:
        parser.parse.return_value = events or []
    return parser


# ---------------------------------------------------------------------------
# 1 — sync ok
# ---------------------------------------------------------------------------


def test_sync_ok(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    client = _make_client()
    parser = _make_parser([_event()])

    result = syncer.sync(client, parser, _DATE)

    assert result.status == "ok"
    assert result.event_count == 1
    parquet_path = tmp_path / f"transactions_{_DATE}.parquet"
    assert parquet_path.exists()


# ---------------------------------------------------------------------------
# 2 — skipped within TTL
# ---------------------------------------------------------------------------


def test_sync_skipped_within_ttl(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path, ttl_hours=4.0)
    client = _make_client()
    parser = _make_parser([_event()])

    syncer.sync(client, parser, _DATE)
    assert client.fetch_transactions.call_count == 1

    result = syncer.sync(client, parser, _DATE)
    assert result.status == "skipped"
    assert client.fetch_transactions.call_count == 1  # no second fetch


# ---------------------------------------------------------------------------
# 3 — force bypasses TTL
# ---------------------------------------------------------------------------


def test_sync_force_bypasses_ttl(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path, ttl_hours=4.0)
    client = _make_client()
    parser = _make_parser([_event()])

    syncer.sync(client, parser, _DATE)
    result = syncer.sync(client, parser, _DATE, force=True)

    assert result.status == "ok"
    assert client.fetch_transactions.call_count == 2


# ---------------------------------------------------------------------------
# 4 — error_api
# ---------------------------------------------------------------------------


def test_sync_error_api(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    client = _make_client(raise_exc=RuntimeError("timeout"))
    parser = _make_parser()

    result = syncer.sync(client, parser, _DATE)

    assert result.status == "error_api"
    assert result.event_count == 0
    assert "timeout" in (result.error_message or "")

    manifest = syncer.get_manifest()
    assert manifest["dates"][_DATE]["status"] == "error_api"


# ---------------------------------------------------------------------------
# 5 — empty (0 events)
# ---------------------------------------------------------------------------


def test_sync_empty(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    client = _make_client()
    parser = _make_parser([])

    result = syncer.sync(client, parser, _DATE)

    assert result.status == "empty"
    assert result.event_count == 0


# ---------------------------------------------------------------------------
# 6 — get_events returns DataFrame after ok sync
# ---------------------------------------------------------------------------


def test_get_events_returns_df(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    syncer.sync(_make_client(), _make_parser([_event()]), _DATE)

    df = syncer.get_events(_DATE)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "event_type" in df.columns


# ---------------------------------------------------------------------------
# 7 — get_events: no file → empty schema
# ---------------------------------------------------------------------------


def test_get_events_no_file(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    df = syncer.get_events(_DATE)
    assert len(df) == 0
    for col in TRANSACTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 8 — get_events_range
# ---------------------------------------------------------------------------


def test_get_events_range(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    syncer.sync(_make_client(), _make_parser([_event(date=_DATE)]), _DATE)
    syncer.sync(_make_client(), _make_parser([_event(date=_DATE2, description="Day 2 trade.")]), _DATE2)

    df = syncer.get_events_range(_DATE, _DATE2)
    assert len(df) == 2


def test_get_events_range_no_files(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    df = syncer.get_events_range(_DATE, _DATE2)
    assert len(df) == 0
    for col in TRANSACTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 9 — get_manifest empty
# ---------------------------------------------------------------------------


def test_get_manifest_empty(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    manifest = syncer.get_manifest()
    assert manifest == {"format_version": 1, "dates": {}}


# ---------------------------------------------------------------------------
# 10 — clear_errors
# ---------------------------------------------------------------------------


def test_clear_errors_removes_error_entries(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    manifest = {
        "format_version": 1,
        "dates": {
            "2026-03-21": {"status": "error_api"},
            "2026-03-20": {"status": "ok"},
            "2026-03-19": {"status": "error_parse"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    removed = syncer.clear_errors()
    assert removed == 2
    updated = syncer.get_manifest()
    assert "2026-03-20" in updated["dates"]
    assert "2026-03-21" not in updated["dates"]
    assert "2026-03-19" not in updated["dates"]


# ---------------------------------------------------------------------------
# 11 — expired TTL re-syncs
# ---------------------------------------------------------------------------


def test_expired_ttl_re_syncs(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path, ttl_hours=1.0)
    client = _make_client()
    parser = _make_parser([_event()])

    # Inject stale manifest (3 hours ago)
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    manifest = {
        "format_version": 1,
        "dates": {_DATE: {"status": "ok", "event_count": 1, "synced_at": stale_time}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = syncer.sync(client, parser, _DATE)
    assert result.status == "ok"
    assert client.fetch_transactions.call_count == 1  # re-fetched


# ---------------------------------------------------------------------------
# 12 — Parquet deduplication
# ---------------------------------------------------------------------------


def test_parquet_deduplication(tmp_path: Path) -> None:
    syncer = TransactionSyncer(tmp_path)
    client = _make_client()
    parser = _make_parser([_event()])

    syncer.sync(client, parser, _DATE)
    syncer.sync(client, parser, _DATE, force=True)

    df = syncer.get_events(_DATE)
    # same (date, description, team) → deduplicated to 1 row
    assert len(df) == 1


# ---------------------------------------------------------------------------
# 13 — storage_dir created automatically
# ---------------------------------------------------------------------------


def test_storage_dir_created_automatically(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested" / "transactions"
    assert not new_dir.exists()
    TransactionSyncer(new_dir)
    assert new_dir.exists()
