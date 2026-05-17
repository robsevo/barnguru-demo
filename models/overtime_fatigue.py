"""Overtime fatigue tracker — Feature 3.10.

Per-player-per-game rolling 7-day count of overtime games played, plus a
synthetic "equivalent TOI" fatigue load: each OT game in the window adds
``OT_TOI_EQUIVALENT_SECS`` (default 360 s ≈ 6 min) of effective wear on
top of what regular TOI already captures.

OT minutes are not "just more TOI" — they are 3-on-3 sprint hockey,
overlap heavily with the top six / top pair, and arrive on top of a
full 60 minutes of regulation work in the same calendar day. A team
that plays three OT games inside a week is carrying a fatigue payload
no rest-day count can see. This channel feeds the composite FI (3.17).

Inputs
------
``ot_df`` (Polars) — one row per (game, player) with::

    game_id          Int64
    game_date        Utf8    ("YYYY-MM-DD")
    player_id        Int64
    team_id          Int64
    toi_ot_secs      Int64   seconds played in the OT period(s)
    played_ot        Boolean True iff the game reached overtime
                              (independent of whether this player skated)

The compute script derives ``toi_ot_secs`` from the TOI Parquet table
and ``played_ot`` from per-game PBP `period_type == "OT"` membership.

Outputs
-------
One row per (player, game) with OVERTIME_FATIGUE_SCHEMA::

    player_id                  — Int64
    game_id                    — Int64
    game_date                  — Utf8
    team_id                    — Int64
    games_played_to_date       — Int64
    played_ot                  — Boolean   this game
    toi_ot_secs                — Int64     this game
    ot_games_7day              — Int64     count over last 7 calendar days
                                           including this game
    ot_secs_actual_7day        — Int64     sum of toi_ot_secs over window
    ot_load_equiv_secs         — Float64   ot_games_7day × OT_TOI_EQUIVALENT_SECS
    ot_fatigue_score           — Float64   ot_load_equiv_secs + ot_secs_actual_7day
                                           — total OT-attributable fatigue
                                           load in seconds, equivalent units

Conventions
-----------
- "Last 7 days" is a calendar-day window inclusive of the current game's date.
- ``played_ot=True`` counts the game even if the player did not skate in OT
  — team-level OT exposure (extended bench shortening, late finish,
  short travel turnaround) still propagates to all players.
- ``toi_ot_secs`` is summed *actual* OT seconds, used additively with the
  equivalent-TOI penalty: real burn + synthetic equivalent.
- The equivalent constant ``OT_TOI_EQUIVALENT_SECS`` (default 360 s) is
  the audit knob; pass another value to the constructor to override.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "overtime_fatigue_v1"

OT_FATIGUE_REQUIRED_COLS = (
    "game_id",
    "game_date",
    "player_id",
    "team_id",
    "toi_ot_secs",
    "played_ot",
)

OT_FATIGUE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":              pl.Int64,
    "game_id":                pl.Int64,
    "game_date":              pl.Utf8,
    "team_id":                pl.Int64,
    "games_played_to_date":   pl.Int64,
    "played_ot":              pl.Boolean,
    "toi_ot_secs":            pl.Int64,
    "ot_games_7day":          pl.Int64,
    "ot_secs_actual_7day":    pl.Int64,
    "ot_load_equiv_secs":     pl.Float64,
    "ot_fatigue_score":       pl.Float64,
}


OT_TOI_EQUIVALENT_SECS = 360       # ~6 min midpoint of 5–7 min range
ROLLING_WINDOW_DAYS    = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in OT_FATIGUE_SCHEMA.items()}
    )


def _validate(ot_df: pl.DataFrame) -> bool:
    missing = [c for c in OT_FATIGUE_REQUIRED_COLS if c not in ot_df.columns]
    if missing:
        warnings.warn(
            f"ot_df missing columns: {missing}. Required: "
            f"{list(OT_FATIGUE_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _annotate_player(block: pl.DataFrame, equiv_secs: int) -> pl.DataFrame:
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return _empty_output()

    dates       = [_parse_date(r["game_date"])      for r in rows]
    played_ots  = [bool(r["played_ot"])             for r in rows]
    ot_secs     = [int(r["toi_ot_secs"] or 0)      for r in rows]

    games_played    = list(range(1, n + 1))
    ot_games_7day:  list[int]   = [0] * n
    ot_secs_7day:   list[int]   = [0] * n
    ot_equiv_secs:  list[float] = [0.0] * n
    ot_score:       list[float] = [0.0] * n

    # Two-pointer left edge over date-sorted rows.
    lo = 0
    for i in range(n):
        cutoff = dates[i] - timedelta(days=ROLLING_WINDOW_DAYS - 1)
        while lo <= i and dates[lo] < cutoff:
            lo += 1
        count = 0
        secs_sum = 0
        for j in range(lo, i + 1):
            if played_ots[j]:
                count   += 1
                secs_sum += ot_secs[j]
        ot_games_7day[i]  = count
        ot_secs_7day[i]   = secs_sum
        ot_equiv_secs[i]  = float(count) * float(equiv_secs)
        ot_score[i]       = ot_equiv_secs[i] + float(secs_sum)

    return block.with_columns(
        pl.Series("games_played_to_date",  games_played, dtype=pl.Int64),
        pl.Series("ot_games_7day",         ot_games_7day, dtype=pl.Int64),
        pl.Series("ot_secs_actual_7day",   ot_secs_7day,  dtype=pl.Int64),
        pl.Series("ot_load_equiv_secs",    ot_equiv_secs, dtype=pl.Float64),
        pl.Series("ot_fatigue_score",      ot_score,      dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# OvertimeFatigueTracker
# ---------------------------------------------------------------------------

class OvertimeFatigueTracker:
    """Overtime fatigue tracker (Feature 3.10). Stateless transformer."""

    def __init__(
        self,
        equivalent_toi_secs: int = OT_TOI_EQUIVALENT_SECS,
    ) -> None:
        if equivalent_toi_secs < 0:
            raise ValueError("equivalent_toi_secs must be non-negative")
        self._equiv_secs = int(equivalent_toi_secs)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, ot_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (player, game) with OT-fatigue features."""
        if not _validate(ot_df):
            return _empty_output()
        if len(ot_df) == 0:
            return _empty_output()

        sorted_df = (
            ot_df.select(list(OT_FATIGUE_REQUIRED_COLS))
            .with_columns(
                pl.col("played_ot").cast(pl.Boolean),
                pl.col("toi_ot_secs").cast(pl.Int64),
            )
            .sort(["player_id", "game_date", "game_id"])
        )
        blocks = [
            _annotate_player(group, self._equiv_secs)
            for _, group in sorted_df.group_by("player_id", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        return result.select(list(OT_FATIGUE_SCHEMA.keys()))

    def compute_player(self, ot_df: pl.DataFrame, player_id: int) -> pl.DataFrame:
        return self.compute(ot_df).filter(pl.col("player_id") == player_id)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_overtime_fatigue(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write overtime-fatigue DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"overtime_fatigue_{as_of_date}.parquet"
    for col, dtype in OT_FATIGUE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(OT_FATIGUE_SCHEMA.keys())).write_parquet(path)
    return path
