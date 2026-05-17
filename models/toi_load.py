"""TOI load tracker — Feature 3.7.

Per-player-per-game time-on-ice load: rolling 5-game average, season-to-date
average, signed spike vs. season baseline, and a Boolean spike flag. Captures
the "minutes burned" fatigue channel that 3.1 (schedule density) and 3.3
(travel) compound: a defenceman who has played 27 minutes for three straight
nights enters the next game already cooked, even on three days' rest.

Inputs
------
``toi_df`` (Polars) — one row per (game, player) with these columns::

    game_id          Int64
    game_date        Utf8    ("YYYY-MM-DD")
    player_id        Int64
    team_id          Int64
    toi_total_secs   Int64    seconds — sum of shift durations for the game

The compute script joins ``ShiftParser`` TOI output with the schedule to
attach ``game_date``. This model is intentionally schedule-agnostic: the
schedule itself is not needed here — just the dated TOI rows.

Outputs
-------
One row per (player, game) with TOI_LOAD_SCHEMA::

    player_id                — Int64
    game_id                  — Int64
    game_date                — Utf8
    team_id                  — Int64
    toi_secs                 — Int64    this game's TOI
    games_played_to_date     — Int64    1-indexed
    toi_5game_avg_secs       — Float64  inclusive of current game
    toi_season_avg_secs      — Float64  season-to-date mean (inclusive)
    toi_spike_delta_secs     — Float64  current − season avg
    toi_spike_z              — Float64  delta / season-std-to-date;
                                        0.0 when std=0 or games<2
    is_toi_spike             — Boolean  True iff z ≥ 1.5 AND
                                        delta_secs ≥ 60 (1 minute)

Conventions
-----------
- "Last 5 games" includes the current game. A player's first game has a
  5-game average equal to that game's TOI.
- Season-to-date averages are running and only ever reference games at or
  before the current row.
- The spike flag requires both a ≥1.5σ z-score AND at least one full minute
  of excess to suppress noise on light-TOI players who hit the threshold on
  variance alone (e.g. a 4th-line forward jumping from 8 to 10 minutes).
- Empty input or a player with only one game gets z = 0.0, spike = False.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "toi_load_v1"

TOI_LOAD_REQUIRED_COLS = (
    "game_id",
    "game_date",
    "player_id",
    "team_id",
    "toi_total_secs",
)

TOI_LOAD_SCHEMA: dict[str, pl.DataType] = {
    "player_id":            pl.Int64,
    "game_id":              pl.Int64,
    "game_date":            pl.Utf8,
    "team_id":              pl.Int64,
    "toi_secs":             pl.Int64,
    "games_played_to_date": pl.Int64,
    "toi_5game_avg_secs":   pl.Float64,
    "toi_season_avg_secs":  pl.Float64,
    "toi_spike_delta_secs": pl.Float64,
    "toi_spike_z":          pl.Float64,
    "is_toi_spike":         pl.Boolean,
}


SPIKE_Z_THRESHOLD     = 1.5
SPIKE_MIN_DELTA_SECS  = 60      # 1 minute of excess minimum
ROLLING_WINDOW_GAMES  = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in TOI_LOAD_SCHEMA.items()}
    )


def _validate(toi_df: pl.DataFrame) -> bool:
    missing = [c for c in TOI_LOAD_REQUIRED_COLS if c not in toi_df.columns]
    if missing:
        warnings.warn(
            f"toi_df missing columns: {missing}. Required: "
            f"{list(TOI_LOAD_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _annotate_player(block: pl.DataFrame) -> pl.DataFrame:
    """Run the rolling-window pass for one player's chronological games."""
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return _empty_output()

    games_played = list(range(1, n + 1))
    five_avg:    list[float] = [0.0] * n
    season_avg:  list[float] = [0.0] * n
    delta:       list[float] = [0.0] * n
    z_score:     list[float] = [0.0] * n
    is_spike:    list[bool]  = [False] * n

    season_sum    = 0.0
    season_sq_sum = 0.0
    five_sum      = 0
    toi_vals      = [int(r["toi_total_secs"]) for r in rows]

    for i, toi in enumerate(toi_vals):
        season_sum    += toi
        season_sq_sum += toi * toi

        # Five-game window: prefix sum trick using a small deque-ish slice.
        # Slicing on a list is cheap at this size (n ≈ 82).
        window_lo = max(0, i - (ROLLING_WINDOW_GAMES - 1))
        five_sum = sum(toi_vals[window_lo : i + 1])
        five_n = i - window_lo + 1
        five_avg[i] = five_sum / five_n

        season_n = i + 1
        mean = season_sum / season_n
        season_avg[i] = mean
        delta[i] = float(toi) - mean

        if season_n >= 2:
            var = max(0.0, (season_sq_sum / season_n) - (mean * mean))
            std = math.sqrt(var)
            if std > 0.0:
                z_score[i] = delta[i] / std
        is_spike[i] = (
            z_score[i] >= SPIKE_Z_THRESHOLD
            and delta[i] >= float(SPIKE_MIN_DELTA_SECS)
        )

    return block.with_columns(
        pl.Series("toi_secs",             toi_vals,     dtype=pl.Int64),
        pl.Series("games_played_to_date", games_played, dtype=pl.Int64),
        pl.Series("toi_5game_avg_secs",   five_avg,     dtype=pl.Float64),
        pl.Series("toi_season_avg_secs",  season_avg,   dtype=pl.Float64),
        pl.Series("toi_spike_delta_secs", delta,        dtype=pl.Float64),
        pl.Series("toi_spike_z",          z_score,      dtype=pl.Float64),
        pl.Series("is_toi_spike",         is_spike,     dtype=pl.Boolean),
    )


# ---------------------------------------------------------------------------
# TOILoadTracker
# ---------------------------------------------------------------------------

class TOILoadTracker:
    """TOI load tracker (Feature 3.7). Stateless transformer."""

    def __init__(self) -> None:
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, toi_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (player, game) with TOI-load features."""
        if not _validate(toi_df):
            return _empty_output()
        if len(toi_df) == 0:
            return _empty_output()

        sorted_df = toi_df.select(list(TOI_LOAD_REQUIRED_COLS)).sort(
            ["player_id", "game_date", "game_id"]
        )
        blocks = [
            _annotate_player(group)
            for _, group in sorted_df.group_by("player_id", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        # Drop the source toi_total_secs column (replaced by typed toi_secs).
        result = result.drop("toi_total_secs")
        return result.select(list(TOI_LOAD_SCHEMA.keys()))

    def compute_player(self, toi_df: pl.DataFrame, player_id: int) -> pl.DataFrame:
        return self.compute(toi_df).filter(pl.col("player_id") == player_id)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_toi_load(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write TOI-load DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"toi_load_{as_of_date}.parquet"
    for col, dtype in TOI_LOAD_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(TOI_LOAD_SCHEMA.keys())).write_parquet(path)
    return path
