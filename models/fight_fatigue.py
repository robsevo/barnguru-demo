"""Fight fatigue tracker — Feature 3.11.

Per-player-per-game rolling tally of fighting majors with an exponential
adrenal-toll decay. A fight is not a hit, not a check, not a regular
penalty: it is a sustained physiological event — heart rate spike,
adrenal dump, knuckle/wrist micro-trauma, sleep disruption from
post-game soreness. The cost decays over several days but does not
vanish overnight.

Inputs
------
``fight_df`` (Polars) — one row per (game, player) with::

    game_id          Int64
    game_date        Utf8    ("YYYY-MM-DD")
    player_id        Int64
    team_id          Int64
    fights_this_game Int64   number of fighting majors this player took

The compute script derives this from PBP rows with
``penalty_type == "MAJ"`` and ``penalty_description ~ "fighting"``,
counted per ``committed_by_id`` per game.

Outputs
-------
One row per (player, game) with FIGHT_FATIGUE_SCHEMA::

    player_id                 — Int64
    game_id                   — Int64
    game_date                 — Utf8
    team_id                   — Int64
    games_played_to_date      — Int64
    fights_this_game          — Int64
    fights_14day              — Int64    raw fight count over 14-day window
    days_since_last_fight     — Int64    integer days since most recent fight
                                         in window; -1 if no prior fight
    fight_load_score          — Float64  Σ over fights in window of
                                         exp(−ln 2 × days_since / HALF_LIFE)
                                         — adrenal/physical toll units

Conventions
-----------
- "Last 14 days" is calendar-inclusive: a fight on the same day counts at
  full weight (1.0), one 3 days ago counts at 0.5, one 6 days ago at 0.25.
- A player who fights twice in one game gets ``fights_this_game = 2``;
  both contribute to ``fight_load_score`` at the current-day weight (1.0).
- ``days_since_last_fight = -1`` is the sentinel for "no fight in window".
- ``HALF_LIFE_DAYS`` is the audit knob on the constructor — adrenal cost
  half-life. Default 3 days reflects published recovery data on combat
  athletes (heart-rate variability returns to baseline within ~72 h).
- Empty input or a player who has never fought gets score = 0.0.
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "fight_fatigue_v1"

FIGHT_REQUIRED_COLS = (
    "game_id",
    "game_date",
    "player_id",
    "team_id",
    "fights_this_game",
)

FIGHT_FATIGUE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":              pl.Int64,
    "game_id":                pl.Int64,
    "game_date":              pl.Utf8,
    "team_id":                pl.Int64,
    "games_played_to_date":   pl.Int64,
    "fights_this_game":       pl.Int64,
    "fights_14day":           pl.Int64,
    "days_since_last_fight":  pl.Int64,
    "fight_load_score":       pl.Float64,
}


HALF_LIFE_DAYS         = 3.0      # adrenal/physical recovery half-life
ROLLING_WINDOW_DAYS    = 14       # contributions older than this are dropped
NO_RECENT_FIGHT        = -1       # sentinel for days_since_last_fight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in FIGHT_FATIGUE_SCHEMA.items()}
    )


def _validate(fight_df: pl.DataFrame) -> bool:
    missing = [c for c in FIGHT_REQUIRED_COLS if c not in fight_df.columns]
    if missing:
        warnings.warn(
            f"fight_df missing columns: {missing}. Required: "
            f"{list(FIGHT_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fight_decay_weight(days_since: int, half_life: float = HALF_LIFE_DAYS) -> float:
    """Exponential decay weight for a fight ``days_since`` days ago.

    Same-day fight gets weight 1.0; weight halves every ``half_life`` days.
    Negative days_since is clipped to 0 (treated as same-day).
    """
    d = max(0, int(days_since))
    if half_life <= 0:
        return 1.0 if d == 0 else 0.0
    return math.exp(-math.log(2.0) * d / half_life)


def _annotate_player(
    block: pl.DataFrame,
    half_life: float,
    window_days: int,
) -> pl.DataFrame:
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return _empty_output()

    dates  = [_parse_date(r["game_date"])      for r in rows]
    fights = [int(r["fights_this_game"] or 0)  for r in rows]

    games_played          = list(range(1, n + 1))
    fights_window:  list[int]   = [0] * n
    days_since:     list[int]   = [NO_RECENT_FIGHT] * n
    load_score:     list[float] = [0.0] * n

    # Two-pointer left edge on the date-sorted block.
    lo = 0
    for i in range(n):
        cutoff = dates[i] - timedelta(days=window_days - 1)
        while lo <= i and dates[lo] < cutoff:
            lo += 1

        count = 0
        score = 0.0
        last_fight_day = NO_RECENT_FIGHT
        for j in range(lo, i + 1):
            f = fights[j]
            if f <= 0:
                continue
            gap = (dates[i] - dates[j]).days
            count += f
            score += f * fight_decay_weight(gap, half_life)
            last_fight_day = gap if last_fight_day == NO_RECENT_FIGHT else min(last_fight_day, gap)

        fights_window[i] = count
        days_since[i]    = last_fight_day
        load_score[i]    = score

    return block.with_columns(
        pl.Series("games_played_to_date",   games_played,  dtype=pl.Int64),
        pl.Series("fights_14day",           fights_window, dtype=pl.Int64),
        pl.Series("days_since_last_fight",  days_since,    dtype=pl.Int64),
        pl.Series("fight_load_score",       load_score,    dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# FightFatigueTracker
# ---------------------------------------------------------------------------

class FightFatigueTracker:
    """Fight fatigue tracker (Feature 3.11). Stateless transformer."""

    def __init__(
        self,
        half_life_days: float = HALF_LIFE_DAYS,
        window_days: int = ROLLING_WINDOW_DAYS,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        self._half_life = float(half_life_days)
        self._window    = int(window_days)
        self._version   = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, fight_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (player, game) with fight-fatigue features."""
        if not _validate(fight_df):
            return _empty_output()
        if len(fight_df) == 0:
            return _empty_output()

        sorted_df = (
            fight_df.select(list(FIGHT_REQUIRED_COLS))
            .with_columns(pl.col("fights_this_game").cast(pl.Int64))
            .sort(["player_id", "game_date", "game_id"])
        )
        blocks = [
            _annotate_player(group, self._half_life, self._window)
            for _, group in sorted_df.group_by("player_id", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        return result.select(list(FIGHT_FATIGUE_SCHEMA.keys()))

    def compute_player(self, fight_df: pl.DataFrame, player_id: int) -> pl.DataFrame:
        return self.compute(fight_df).filter(pl.col("player_id") == player_id)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_fight_fatigue(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write fight-fatigue DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fight_fatigue_{as_of_date}.parquet"
    for col, dtype in FIGHT_FATIGUE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(FIGHT_FATIGUE_SCHEMA.keys())).write_parquet(path)
    return path
