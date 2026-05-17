"""Special teams load tracker — Feature 3.8.

Per-player-per-game accumulation of power-play and penalty-kill minutes
across a rolling 5-game window. Special-teams shifts are run substantially
harder than 5v5: PP units chase set plays at full intensity; PK units skate
hard board-to-board on every clearance. A defenceman who spent 12 minutes
killing penalties over three games is materially more cooked than one who
spent the same 12 minutes at 5v5.

Inputs
------
``st_df`` (Polars) — one row per (game, player) with::

    game_id        Int64
    game_date      Utf8   ("YYYY-MM-DD")
    player_id      Int64
    team_id        Int64
    pptoi_secs     Int64   power-play seconds this game
    shtoi_secs     Int64   penalty-kill seconds this game

Outputs
-------
One row per (player, game) with ST_LOAD_SCHEMA::

    player_id                — Int64
    game_id                  — Int64
    game_date                — Utf8
    team_id                  — Int64
    games_played_to_date     — Int64
    pptoi_secs               — Int64    this game's PP TOI
    shtoi_secs               — Int64    this game's PK TOI
    pptoi_5game_sum_secs     — Int64    PP TOI summed over last 5 games
    shtoi_5game_sum_secs     — Int64    PK TOI summed over last 5 games
    st_total_5game_sum_secs  — Int64    PP + PK summed over last 5 games
    st_intensity_score       — Float64  weighted seconds:
                                        PP × PP_INTENSITY_WEIGHT
                                        + PK × PK_INTENSITY_WEIGHT
                                        summed over last 5 games

Conventions
-----------
- "Last 5 games" includes the current game.
- Intensity weights are deliberately conservative; they capture the relative
  fatigue load of ST minutes vs. 5v5 minutes. PK gets the higher weight —
  desperate defending, board scrums, less ice to recover on. These weights
  are auditable knobs (Bob can dial them) — they live as module constants.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "special_teams_load_v1"

ST_LOAD_REQUIRED_COLS = (
    "game_id",
    "game_date",
    "player_id",
    "team_id",
    "pptoi_secs",
    "shtoi_secs",
)

ST_LOAD_SCHEMA: dict[str, pl.DataType] = {
    "player_id":                pl.Int64,
    "game_id":                  pl.Int64,
    "game_date":                pl.Utf8,
    "team_id":                  pl.Int64,
    "games_played_to_date":     pl.Int64,
    "pptoi_secs":               pl.Int64,
    "shtoi_secs":               pl.Int64,
    "pptoi_5game_sum_secs":     pl.Int64,
    "shtoi_5game_sum_secs":     pl.Int64,
    "st_total_5game_sum_secs":  pl.Int64,
    "st_intensity_score":       pl.Float64,
}


PP_INTENSITY_WEIGHT  = 1.5     # PP minutes ≈ 1.5× the fatigue load of EV
PK_INTENSITY_WEIGHT  = 1.8     # PK minutes ≈ 1.8× — highest intensity
ROLLING_WINDOW_GAMES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in ST_LOAD_SCHEMA.items()}
    )


def _validate(st_df: pl.DataFrame) -> bool:
    missing = [c for c in ST_LOAD_REQUIRED_COLS if c not in st_df.columns]
    if missing:
        warnings.warn(
            f"st_df missing columns: {missing}. Required: "
            f"{list(ST_LOAD_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _annotate_player(block: pl.DataFrame) -> pl.DataFrame:
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return _empty_output()

    pp = [int(r["pptoi_secs"]) for r in rows]
    pk = [int(r["shtoi_secs"]) for r in rows]

    games_played       = list(range(1, n + 1))
    pp_5sum:  list[int]   = [0] * n
    pk_5sum:  list[int]   = [0] * n
    st_5sum:  list[int]   = [0] * n
    intensity: list[float] = [0.0] * n

    for i in range(n):
        lo = max(0, i - (ROLLING_WINDOW_GAMES - 1))
        pp_window = pp[lo : i + 1]
        pk_window = pk[lo : i + 1]
        pp_sum = sum(pp_window)
        pk_sum = sum(pk_window)
        pp_5sum[i] = pp_sum
        pk_5sum[i] = pk_sum
        st_5sum[i] = pp_sum + pk_sum
        intensity[i] = (
            PP_INTENSITY_WEIGHT * pp_sum
            + PK_INTENSITY_WEIGHT * pk_sum
        )

    return block.with_columns(
        pl.Series("games_played_to_date",    games_played, dtype=pl.Int64),
        pl.Series("pptoi_5game_sum_secs",    pp_5sum,      dtype=pl.Int64),
        pl.Series("shtoi_5game_sum_secs",    pk_5sum,      dtype=pl.Int64),
        pl.Series("st_total_5game_sum_secs", st_5sum,      dtype=pl.Int64),
        pl.Series("st_intensity_score",      intensity,    dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# SpecialTeamsLoadTracker
# ---------------------------------------------------------------------------

class SpecialTeamsLoadTracker:
    """Special teams load tracker (Feature 3.8). Stateless transformer."""

    def __init__(
        self,
        pp_intensity_weight: float = PP_INTENSITY_WEIGHT,
        pk_intensity_weight: float = PK_INTENSITY_WEIGHT,
    ) -> None:
        self._pp_w = float(pp_intensity_weight)
        self._pk_w = float(pk_intensity_weight)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, st_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (player, game) with special-teams load features."""
        if not _validate(st_df):
            return _empty_output()
        if len(st_df) == 0:
            return _empty_output()

        sorted_df = st_df.select(list(ST_LOAD_REQUIRED_COLS)).sort(
            ["player_id", "game_date", "game_id"]
        )
        blocks: list[pl.DataFrame] = []
        for _, group in sorted_df.group_by("player_id", maintain_order=True):
            annotated = _annotate_player(group)
            # Allow per-instance weight overrides without rebuilding the loop.
            if (
                self._pp_w != PP_INTENSITY_WEIGHT
                or self._pk_w != PK_INTENSITY_WEIGHT
            ):
                pp = annotated["pptoi_5game_sum_secs"].to_list()
                pk = annotated["shtoi_5game_sum_secs"].to_list()
                annotated = annotated.with_columns(
                    pl.Series(
                        "st_intensity_score",
                        [self._pp_w * p + self._pk_w * k for p, k in zip(pp, pk)],
                        dtype=pl.Float64,
                    )
                )
            blocks.append(annotated)
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        return result.select(list(ST_LOAD_SCHEMA.keys()))

    def compute_player(self, st_df: pl.DataFrame, player_id: int) -> pl.DataFrame:
        return self.compute(st_df).filter(pl.col("player_id") == player_id)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_special_teams_load(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write ST-load DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"special_teams_load_{as_of_date}.parquet"
    for col, dtype in ST_LOAD_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(ST_LOAD_SCHEMA.keys())).write_parquet(path)
    return path
