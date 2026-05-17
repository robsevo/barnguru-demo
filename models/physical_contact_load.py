"""Physical contact load — Feature 3.9.

Per-player-per-game rolling tally of hits absorbed and shots blocked across
the last 5 games. These are the two PBP-observable signals for accumulated
physical wear: the cumulative pounding a defenceman absorbs from a heavy
forecheck, or the broken-stick / bruised-quad legacy of a shot-blocking
specialist. Stacks on the schedule, TOI, and ST channels in the composite
Fatigue Index (3.17).

Inputs
------
``contact_df`` (Polars) — one row per (game, player) with::

    game_id          Int64
    game_date        Utf8    ("YYYY-MM-DD")
    player_id        Int64
    team_id          Int64
    hits_taken       Int64   number of hits the player absorbed this game
    blocked_shots    Int64   number of shots the player blocked this game

The compute script derives these from PBP via ``hittee_id`` and
``blocker_id`` counts; this model is intentionally storage-agnostic.

Outputs
-------
One row per (player, game) with PHYSICAL_LOAD_SCHEMA::

    player_id                  — Int64
    game_id                    — Int64
    game_date                  — Utf8
    team_id                    — Int64
    games_played_to_date       — Int64
    hits_taken                 — Int64
    blocked_shots              — Int64
    hits_5game_sum             — Int64
    blocks_5game_sum           — Int64
    contact_score              — Float64  this game's score:
                                          hits + HIT_WEIGHT × blocks
    contact_5game_score        — Float64  contact_score summed over last 5

Conventions
-----------
- "Last 5 games" includes the current game.
- Blocking a shot is weighted higher than absorbing a hit (default 1.5×):
  a blocked shot is a guaranteed direct collision with a hard projectile,
  often delivered to a less-padded body part; hits vary in delivery quality.
  The weight is an audit knob on the constructor.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "physical_contact_load_v1"

PHYSICAL_LOAD_REQUIRED_COLS = (
    "game_id",
    "game_date",
    "player_id",
    "team_id",
    "hits_taken",
    "blocked_shots",
)

PHYSICAL_LOAD_SCHEMA: dict[str, pl.DataType] = {
    "player_id":             pl.Int64,
    "game_id":               pl.Int64,
    "game_date":             pl.Utf8,
    "team_id":               pl.Int64,
    "games_played_to_date":  pl.Int64,
    "hits_taken":            pl.Int64,
    "blocked_shots":         pl.Int64,
    "hits_5game_sum":        pl.Int64,
    "blocks_5game_sum":      pl.Int64,
    "contact_score":         pl.Float64,
    "contact_5game_score":   pl.Float64,
}


HIT_WEIGHT           = 1.0     # baseline
BLOCK_WEIGHT         = 1.5     # blocks carry more concentrated impact
ROLLING_WINDOW_GAMES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in PHYSICAL_LOAD_SCHEMA.items()}
    )


def _validate(contact_df: pl.DataFrame) -> bool:
    missing = [c for c in PHYSICAL_LOAD_REQUIRED_COLS if c not in contact_df.columns]
    if missing:
        warnings.warn(
            f"contact_df missing columns: {missing}. Required: "
            f"{list(PHYSICAL_LOAD_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def contact_score(
    hits_taken: int,
    blocked_shots: int,
    hit_weight: float = HIT_WEIGHT,
    block_weight: float = BLOCK_WEIGHT,
) -> float:
    """Single-game physical-contact score for one player."""
    return hit_weight * float(hits_taken) + block_weight * float(blocked_shots)


def _annotate_player(
    block: pl.DataFrame,
    hit_w: float,
    block_w: float,
) -> pl.DataFrame:
    rows = block.to_dicts()
    n = len(rows)
    if n == 0:
        return _empty_output()

    hits   = [int(r["hits_taken"])    for r in rows]
    blocks = [int(r["blocked_shots"]) for r in rows]

    games_played = list(range(1, n + 1))
    hits_5sum:    list[int]   = [0] * n
    blocks_5sum:  list[int]   = [0] * n
    score_each:   list[float] = [0.0] * n
    score_5sum:   list[float] = [0.0] * n

    for i in range(n):
        lo = max(0, i - (ROLLING_WINDOW_GAMES - 1))
        h_win = hits[lo : i + 1]
        b_win = blocks[lo : i + 1]
        hits_5sum[i]   = sum(h_win)
        blocks_5sum[i] = sum(b_win)
        score_each[i]  = contact_score(hits[i], blocks[i], hit_w, block_w)
        score_5sum[i]  = hit_w * hits_5sum[i] + block_w * blocks_5sum[i]

    return block.with_columns(
        pl.Series("games_played_to_date",  games_played, dtype=pl.Int64),
        pl.Series("hits_5game_sum",        hits_5sum,    dtype=pl.Int64),
        pl.Series("blocks_5game_sum",      blocks_5sum,  dtype=pl.Int64),
        pl.Series("contact_score",         score_each,   dtype=pl.Float64),
        pl.Series("contact_5game_score",   score_5sum,   dtype=pl.Float64),
    )


# ---------------------------------------------------------------------------
# PhysicalContactLoadTracker
# ---------------------------------------------------------------------------

class PhysicalContactLoadTracker:
    """Physical contact load tracker (Feature 3.9). Stateless transformer."""

    def __init__(
        self,
        hit_weight: float = HIT_WEIGHT,
        block_weight: float = BLOCK_WEIGHT,
    ) -> None:
        self._hit_w = float(hit_weight)
        self._block_w = float(block_weight)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, contact_df: pl.DataFrame) -> pl.DataFrame:
        """Return one row per (player, game) with physical contact features."""
        if not _validate(contact_df):
            return _empty_output()
        if len(contact_df) == 0:
            return _empty_output()

        sorted_df = contact_df.select(list(PHYSICAL_LOAD_REQUIRED_COLS)).sort(
            ["player_id", "game_date", "game_id"]
        )
        blocks = [
            _annotate_player(group, self._hit_w, self._block_w)
            for _, group in sorted_df.group_by("player_id", maintain_order=True)
        ]
        if not blocks:
            return _empty_output()

        result = pl.concat(blocks)
        return result.select(list(PHYSICAL_LOAD_SCHEMA.keys()))

    def compute_player(self, contact_df: pl.DataFrame, player_id: int) -> pl.DataFrame:
        return self.compute(contact_df).filter(pl.col("player_id") == player_id)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_physical_contact_load(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write physical-contact-load DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"physical_contact_load_{as_of_date}.parquet"
    for col, dtype in PHYSICAL_LOAD_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(PHYSICAL_LOAD_SCHEMA.keys())).write_parquet(path)
    return path
