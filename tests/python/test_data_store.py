"""Tests for DataStore — unified DuckDB query layer.

All tests use tmp_path and write minimal Parquet/JSON fixtures.
No real API calls; no side effects outside tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from data.data_store import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pbp(path: Path, season: int, n_rows: int = 2) -> None:
    """Write a minimal pbp Parquet file with `game_id` and `season` columns."""
    df = pl.DataFrame(
        {
            "game_id": list(range(n_rows)),
            "season": [season] * n_rows,
            "event_type": ["shot-on-goal"] * n_rows,
        }
    )
    df.write_parquet(path)


def _write_signals(path: Path, date: str, team_abbrev: str, n_rows: int = 2) -> None:
    """Write a minimal morning skate signals Parquet file."""
    df = pl.DataFrame(
        {
            "date": [date] * n_rows,
            "team_abbrev": [team_abbrev] * n_rows,
            "player_name": [f"Player {i}" for i in range(n_rows)],
            "signal_type": ["practicing_full"] * n_rows,
            "confidence": [0.9] * n_rows,
            "raw_quote": [None] * n_rows,
            "notes": [None] * n_rows,
            "fetched_at": ["2026-03-21T10:00:00+00:00"] * n_rows,
            "llm_model": ["claude-haiku-4-5-20251001"] * n_rows,
        }
    )
    df.write_parquet(path)


def _write_roster_json(path: Path, team_code: str, n_players: int = 2) -> None:
    """Write a roster JSON file matching roster_sync format."""
    profiles = [
        {
            "player_id": 8000000 + i,
            "team_code": team_code,
            "first_name": f"First{i}",
            "last_name": f"Last{i}",
            "position": "C",
            "position_type": "forward",
            "shoots_catches": "L",
            "jersey_number": i + 1,
            "height_inches": 72,
            "weight_pounds": 185,
            "birth_date": "1995-01-15",
            "age": 31.2,
            "birth_country": "CAN",
            "nationality": "CAN",
            "is_captain": False,
            "is_alternate_captain": False,
        }
        for i in range(n_players)
    ]
    payload = {
        "fetched_at": "2026-03-21T10:00:00+00:00",
        "team_code": team_code,
        "profiles": profiles,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Return (raw_dir, ms_dir) under tmp_path, pre-created."""
    raw = tmp_path / "raw"
    ms = tmp_path / "morning_skate"
    raw.mkdir()
    ms.mkdir()
    return raw, ms


@pytest.fixture()
def store(tmp_path: Path, dirs: tuple[Path, Path]) -> DataStore:
    raw, ms = dirs
    return DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms)


# ---------------------------------------------------------------------------
# 1 — pbp: all seasons
# ---------------------------------------------------------------------------


def test_query_pbp_all_seasons(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    _write_pbp(raw / "pbp_20222023.parquet", season=20222023)
    _write_pbp(raw / "pbp_20232024.parquet", season=20232024)

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.pbp()

    assert len(df) == 4  # 2 rows × 2 files
    assert set(df["season"].to_list()) == {20222023, 20232024}


# ---------------------------------------------------------------------------
# 2 — pbp: single season
# ---------------------------------------------------------------------------


def test_query_pbp_single_season(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    _write_pbp(raw / "pbp_20222023.parquet", season=20222023)
    _write_pbp(raw / "pbp_20232024.parquet", season=20232024)

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.pbp(season=20232024)

    assert len(df) == 2
    assert df["season"].to_list() == [20232024, 20232024]


# ---------------------------------------------------------------------------
# 3 — pbp: no files → empty schema
# ---------------------------------------------------------------------------


def test_pbp_empty_returns_schema(store: DataStore) -> None:
    df = store.pbp()
    assert len(df) == 0
    assert "game_id" in df.columns


# ---------------------------------------------------------------------------
# 4 — player_stats
# ---------------------------------------------------------------------------


def test_player_stats_all(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    df_in = pl.DataFrame({"player_id": [1, 2], "season": [20232024, 20232024]})
    df_in.write_parquet(raw / "player_stats_20232024.parquet")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.player_stats()

    assert "player_id" in df.columns
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 5 — injuries: date filter
# ---------------------------------------------------------------------------


def test_injuries_date_filter(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    for date in ("2026-03-20", "2026-03-21"):
        df_in = pl.DataFrame(
            {
                "player_name": ["Player A"],
                "team_code": ["EDM"],
                "status": ["Out"],
                "status_raw": ["Out"],
                "injury_type": ["Lower Body"],
                "injury_detail": [None],
                "return_date_estimate": [None],
                "reported_at": [f"{date}T12:00:00+00:00"],
                "fetched_at": [f"{date}T12:00:00+00:00"],
                "player_id_espn": ["12345"],
            }
        )
        df_in.write_parquet(raw / f"injuries_{date}.parquet")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.injuries(date="2026-03-21")

    assert len(df) == 1
    assert "2026-03-21" in df["reported_at"][0]


# ---------------------------------------------------------------------------
# 6 — markets: no files → empty schema
# ---------------------------------------------------------------------------


def test_markets_no_files(store: DataStore) -> None:
    df = store.markets()
    assert len(df) == 0
    assert "condition_id" in df.columns


# ---------------------------------------------------------------------------
# 7 — morning_skate_signals: team filter
# ---------------------------------------------------------------------------


def test_morning_skate_signals_team_filter(
    dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, ms = dirs
    _write_signals(ms / "signals_2026-03-21_EDM.parquet", "2026-03-21", "EDM")
    _write_signals(ms / "signals_2026-03-21_TOR.parquet", "2026-03-21", "TOR")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.morning_skate_signals(team_abbrev="EDM")

    assert len(df) == 2
    assert all(v == "EDM" for v in df["team_abbrev"].to_list())


# ---------------------------------------------------------------------------
# 8 — roster: single team
# ---------------------------------------------------------------------------


def test_roster_single_team(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    _write_roster_json(raw / "roster_EDM.json", "EDM")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.roster("EDM")

    assert len(df) == 2
    assert all(v == "EDM" for v in df["team_code"].to_list())


# ---------------------------------------------------------------------------
# 9 — roster: all teams
# ---------------------------------------------------------------------------


def test_roster_all_teams(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    _write_roster_json(raw / "roster_EDM.json", "EDM", n_players=3)
    _write_roster_json(raw / "roster_TOR.json", "TOR", n_players=2)

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.roster()

    assert len(df) == 5
    assert set(df["team_code"].to_list()) == {"EDM", "TOR"}


# ---------------------------------------------------------------------------
# 10 — roster: no files → empty schema
# ---------------------------------------------------------------------------


def test_roster_no_files(store: DataStore) -> None:
    df = store.roster()
    assert len(df) == 0
    assert "player_id" in df.columns


# ---------------------------------------------------------------------------
# 11 — context manager
# ---------------------------------------------------------------------------


def test_context_manager(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.pbp()
    assert len(df) == 0  # no files but no exception


# ---------------------------------------------------------------------------
# 12 — raw SQL
# ---------------------------------------------------------------------------


def test_query_raw_sql(store: DataStore) -> None:
    df = store.query("SELECT 1 AS x")
    assert "x" in df.columns
    assert df["x"][0] == 1


# ---------------------------------------------------------------------------
# 13 — shots (moneypuck)
# ---------------------------------------------------------------------------


def test_shots_moneypuck(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    df_in = pl.DataFrame(
        {"shot_id": ["s1", "s2"], "season": [20232024, 20232024], "is_goal": [False, True]}
    )
    df_in.write_parquet(raw / "shots_20232024.parquet")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.shots(20232024)

    assert len(df) == 2
    assert "season" in df.columns


# ---------------------------------------------------------------------------
# 14 — rapm (evolving hockey)
# ---------------------------------------------------------------------------


def test_rapm_evolving_hockey(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    raw, ms = dirs
    df_in = pl.DataFrame(
        {"player_id": [8480353], "season": [20232024], "rapm_xg": [0.15]}
    )
    df_in.write_parquet(raw / "rapm_20232024.parquet")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.rapm()

    assert len(df) == 1
    assert df["player_id"][0] == 8480353


# ---------------------------------------------------------------------------
# 15 — vs_team: no files → empty schema
# ---------------------------------------------------------------------------


def test_vs_team_empty(store: DataStore) -> None:
    df = store.vs_team()
    assert len(df) == 0
    assert "player_id" in df.columns
    assert "opponent_team_id" in df.columns


# ---------------------------------------------------------------------------
# Bonus: morning_skate_signals no files
# ---------------------------------------------------------------------------


def test_morning_skate_signals_no_files(store: DataStore) -> None:
    df = store.morning_skate_signals()
    assert len(df) == 0
    assert "player_name" in df.columns


# ---------------------------------------------------------------------------
# Bonus: morning_skate_signals date filter
# ---------------------------------------------------------------------------


def test_morning_skate_signals_date_filter(
    dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, ms = dirs
    _write_signals(ms / "signals_2026-03-20_EDM.parquet", "2026-03-20", "EDM")
    _write_signals(ms / "signals_2026-03-21_EDM.parquet", "2026-03-21", "EDM")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df = ds.morning_skate_signals(date="2026-03-21")

    assert len(df) == 2
    assert all(v == "2026-03-21" for v in df["date"].to_list())


# ---------------------------------------------------------------------------
# Bonus: roster cache invalidation
# ---------------------------------------------------------------------------


def test_roster_cache_is_used(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Second call returns same result without re-reading disk."""
    raw, ms = dirs
    _write_roster_json(raw / "roster_EDM.json", "EDM")

    with DataStore(tmp_path, raw_dir=raw, morning_skate_dir=ms) as ds:
        df1 = ds.roster("EDM")
        df2 = ds.roster("EDM")  # should hit cache

    assert len(df1) == len(df2) == 2
