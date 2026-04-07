"""Tests for PressConferenceSyncer — TTL-based Parquet persistence."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from data.pbp_parser import DataMissingWarning
from data.press_conference_parser import (
    MENTION_SCHEMA,
    InjuryMention,
    InjurySeverity,
    PressConferenceParser,
    PressConferenceReport,
)
from data.press_conference_sync import PressConferenceSyncer, PressConferenceSyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE = "2026-03-21"
_TEAM = "EDM"


def _make_parser(
    mentions: list[dict] | None = None,
    raise_exc: Exception | None = None,
) -> PressConferenceParser:
    """Return a PressConferenceParser whose parse() returns controlled output."""
    parser = MagicMock(spec=PressConferenceParser)

    if raise_exc is not None:
        parser.parse.side_effect = raise_exc
    else:
        mention_objs = []
        for m in (mentions or []):
            mention_objs.append(
                InjuryMention(
                    player_name=m.get("player_name", "Player A"),
                    severity=InjurySeverity(m.get("severity", "day_to_day")),
                    confidence=m.get("confidence", 0.8),
                    speaker=m.get("speaker"),
                    raw_quote=m.get("raw_quote"),
                    injury_location=m.get("injury_location"),
                    notes=m.get("notes"),
                )
            )
        report = PressConferenceReport(
            date=_DATE,
            team_abbrev=_TEAM,
            raw_text="raw text",
            mentions=mention_objs,
            fetched_at=f"{_DATE}T10:00:00+00:00",
            llm_model="claude-haiku-4-5-20251001",
            warning_count=0,
            input_tokens=150,
            output_tokens=80,
        )
        parser.parse.return_value = report

    return parser


def _default_mention() -> dict:
    return {"player_name": "Connor McDavid", "severity": "day_to_day", "confidence": 0.85}


# ---------------------------------------------------------------------------
# 1 — sync ok
# ---------------------------------------------------------------------------


def test_sync_ok(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    parser = _make_parser([_default_mention()])
    result = syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)

    assert result.status == "ok"
    assert result.mention_count == 1
    parquet_path = tmp_path / f"press_conference_{_DATE}_{_TEAM}.parquet"
    assert parquet_path.exists()


# ---------------------------------------------------------------------------
# 2 — skipped within TTL
# ---------------------------------------------------------------------------


def test_sync_skipped_within_ttl(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path, ttl_hours=4.0)
    parser = _make_parser([_default_mention()])

    # First sync
    syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)
    assert parser.parse.call_count == 1

    # Second sync within TTL
    result = syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)
    assert result.status == "skipped"
    assert parser.parse.call_count == 1  # no second API call


# ---------------------------------------------------------------------------
# 3 — force bypasses TTL
# ---------------------------------------------------------------------------


def test_sync_force_bypasses_ttl(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path, ttl_hours=4.0)
    parser = _make_parser([_default_mention()])

    syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)
    result = syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM, force=True)

    assert result.status == "ok"
    assert parser.parse.call_count == 2


# ---------------------------------------------------------------------------
# 4 — error_api
# ---------------------------------------------------------------------------


def test_sync_error_api(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    parser = _make_parser(raise_exc=RuntimeError("API timeout"))

    result = syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)

    assert result.status == "error_api"
    assert result.mention_count == 0
    assert "API timeout" in (result.error_message or "")

    manifest = syncer.get_manifest()
    assert manifest["dates"][f"{_DATE}:{_TEAM}"]["status"] == "error_api"


# ---------------------------------------------------------------------------
# 5 — empty (0 mentions)
# ---------------------------------------------------------------------------


def test_sync_empty(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    parser = _make_parser([])  # no mentions
    result = syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)

    assert result.status == "empty"
    assert result.mention_count == 0


# ---------------------------------------------------------------------------
# 6 — get_mentions returns DataFrame after ok sync
# ---------------------------------------------------------------------------


def test_get_mentions_returns_df(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    parser = _make_parser([_default_mention()])
    syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)

    df = syncer.get_mentions(_DATE, team_abbrev=_TEAM)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1
    assert "player_name" in df.columns


# ---------------------------------------------------------------------------
# 7 — get_mentions: no file → empty schema
# ---------------------------------------------------------------------------


def test_get_mentions_no_file(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    df = syncer.get_mentions(_DATE, team_abbrev=_TEAM)
    assert len(df) == 0
    for col in MENTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 8 — get_manifest: empty returns correct structure
# ---------------------------------------------------------------------------


def test_get_manifest_empty(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    manifest = syncer.get_manifest()
    assert manifest == {"format_version": 1, "dates": {}}


# ---------------------------------------------------------------------------
# 9 — clear_errors removes error entries
# ---------------------------------------------------------------------------


def test_clear_errors_removes_error_entries(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    # Inject error entries manually
    manifest = {
        "format_version": 1,
        "dates": {
            "2026-03-21:EDM": {"status": "error_api"},
            "2026-03-21:TOR": {"status": "ok"},
            "2026-03-20:EDM": {"status": "error_parse"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    removed = syncer.clear_errors()
    assert removed == 2
    updated = syncer.get_manifest()
    assert "2026-03-21:TOR" in updated["dates"]
    assert "2026-03-21:EDM" not in updated["dates"]
    assert "2026-03-20:EDM" not in updated["dates"]


# ---------------------------------------------------------------------------
# 10 — expired TTL re-parses
# ---------------------------------------------------------------------------


def test_sync_expired_ttl_re_parses(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path, ttl_hours=1.0)
    parser = _make_parser([_default_mention()])

    # Write a stale manifest entry (3 hours ago)
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    manifest = {
        "format_version": 1,
        "dates": {
            f"{_DATE}:{_TEAM}": {
                "status": "ok",
                "mention_count": 1,
                "warning_count": 0,
                "synced_at": stale_time,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)
    assert result.status == "ok"
    assert parser.parse.call_count == 1  # re-parsed


# ---------------------------------------------------------------------------
# 11 — Parquet deduplication
# ---------------------------------------------------------------------------


def test_parquet_deduplication(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    parser = _make_parser([_default_mention()])

    syncer.sync(parser, "first text", date=_DATE, team_abbrev=_TEAM)
    # Second sync for same player (force=True to bypass TTL)
    syncer.sync(parser, "second text", date=_DATE, team_abbrev=_TEAM, force=True)

    df = syncer.get_mentions(_DATE, team_abbrev=_TEAM)
    # Same player+severity+date+team should be deduplicated to 1 row
    assert len(df) == 1


# ---------------------------------------------------------------------------
# 12 — manifest stores token counts
# ---------------------------------------------------------------------------


def test_manifest_stores_token_counts(tmp_path: Path) -> None:
    syncer = PressConferenceSyncer(tmp_path)
    parser = _make_parser([_default_mention()])
    syncer.sync(parser, "raw text", date=_DATE, team_abbrev=_TEAM)

    manifest = syncer.get_manifest()
    entry = manifest["dates"][f"{_DATE}:{_TEAM}"]
    assert entry.get("input_tokens", 0) > 0
    assert entry.get("output_tokens", 0) > 0


# ---------------------------------------------------------------------------
# 13 — no team_abbrev writes _all suffix
# ---------------------------------------------------------------------------


def test_team_abbrev_none_writes_all_suffix(tmp_path: Path) -> None:
    # Build parser that returns report with team_abbrev=None
    parser = MagicMock(spec=PressConferenceParser)
    report = PressConferenceReport(
        date=_DATE,
        team_abbrev=None,
        raw_text="text",
        mentions=[
            InjuryMention(
                player_name="Player X",
                severity=InjurySeverity.DAY_TO_DAY,
                confidence=0.8,
            )
        ],
        fetched_at=f"{_DATE}T10:00:00+00:00",
        llm_model="claude-haiku-4-5-20251001",
        warning_count=0,
        input_tokens=100,
        output_tokens=50,
    )
    parser.parse.return_value = report

    syncer = PressConferenceSyncer(tmp_path)
    result = syncer.sync(parser, "text", date=_DATE, team_abbrev=None)

    assert result.status == "ok"
    assert (tmp_path / f"press_conference_{_DATE}_all.parquet").exists()


# ---------------------------------------------------------------------------
# 14 — storage_dir created automatically
# ---------------------------------------------------------------------------


def test_storage_dir_created_automatically(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested" / "press_conference"
    assert not new_dir.exists()
    syncer = PressConferenceSyncer(new_dir)
    assert new_dir.exists()
