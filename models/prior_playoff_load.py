"""Prior playoff load — Feature 3.15.

A player who finished the prior season with a deep Cup run paid for it
twice: more games in May–June (each one as physical as a regular-season
B2B), and a compressed summer. Less time to heal, less time to rebuild
strength, training camp comes early. The result is a measurable
season-start penalty that bleeds off as the new season's normal rhythm
restores the player.

This module is the *static* per-player load — the prior-season number
of playoff games and the round reached — combined with a *current* count
of games played into the new season to produce a time-varying penalty
that starts at its maximum at season opener and decays linearly to zero
by ``DECAY_GAMES``.

Form
----
``base_penalty = min(PENALTY_MAX, PENALTY_PER_GAME × prior_playoff_games)``

``decay_weight = max(0.0, 1.0 − games_into_season / DECAY_GAMES)``

``current_penalty = base_penalty × decay_weight``

At defaults (PENALTY_PER_GAME=0.002, PENALTY_MAX=0.05, DECAY_GAMES=25):

    prior_playoff_games=0  → 0 penalty all season
    prior_playoff_games=10 → opener 0.020, game 13 ≈ 0.010, game 25 = 0
    prior_playoff_games=26 → opener 0.05  (cap), game 13 ≈ 0.025, game 25 = 0

The penalty is *additive* into the composite FI (3.17), not a direct
rating subtraction.

Inputs
------
``load_df`` (Polars) — one row per player::

    player_id                  Int64
    prior_playoff_games        Int64   GP last spring; 0 if missed playoffs
    prior_playoff_round        Int64   (optional) 0..4: 0=missed,
                                       1=R1, 2=R2, 3=Conf Final, 4=Cup Final
    games_played_this_season   Int64   GP so far this season; 0 at opener

``prior_playoff_round`` is carried through to the output for downstream
context but does not affect the penalty in v1 (games played already
encodes intensity reasonably well). It is optional; defaults to 0.

``as_of_date`` (str, ``"YYYY-MM-DD"``) — snapshot date, written through
to every row.

Outputs
-------
One row per player with PRIOR_PLAYOFF_LOAD_SCHEMA::

    player_id                  — Int64
    as_of_date                 — Utf8
    prior_playoff_games        — Int64
    prior_playoff_round        — Int64
    games_played_this_season   — Int64
    base_playoff_penalty       — Float64   pre-decay penalty
    playoff_load_penalty       — Float64   current penalty after decay
    decay_weight               — Float64   in [0, 1]

Calibration knobs (constructor):
- ``penalty_per_game``  default 0.002
- ``penalty_max``       default 0.05
- ``decay_games``       default 25
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "prior_playoff_load_v1"

LOAD_REQUIRED_COLS = (
    "player_id",
    "prior_playoff_games",
    "games_played_this_season",
)

PRIOR_PLAYOFF_LOAD_SCHEMA: dict[str, pl.DataType] = {
    "player_id":                pl.Int64,
    "as_of_date":               pl.Utf8,
    "prior_playoff_games":      pl.Int64,
    "prior_playoff_round":      pl.Int64,
    "games_played_this_season": pl.Int64,
    "base_playoff_penalty":     pl.Float64,
    "playoff_load_penalty":     pl.Float64,
    "decay_weight":             pl.Float64,
}


PENALTY_PER_GAME = 0.002    # per prior playoff game played
PENALTY_MAX      = 0.05     # cap on base penalty (≈ a Cup-final winner)
DECAY_GAMES      = 25       # games into the new season for penalty to reach 0


# ---------------------------------------------------------------------------
# Public formulas
# ---------------------------------------------------------------------------

def base_penalty(
    prior_playoff_games: int,
    per_game: float = PENALTY_PER_GAME,
    cap:      float = PENALTY_MAX,
) -> float:
    """``min(cap, per_game × max(0, prior_playoff_games))``."""
    g = max(0, int(prior_playoff_games))
    return min(float(cap), float(per_game) * g)


def decay_weight_for(
    games_into_season: int,
    decay_games:       int = DECAY_GAMES,
) -> float:
    """Linear decay weight: 1.0 at opener, 0.0 at/after ``decay_games``."""
    g = max(0, int(games_into_season))
    span = max(1, int(decay_games))
    return max(0.0, 1.0 - g / span)


def current_penalty(
    prior_playoff_games: int,
    games_into_season:   int,
    per_game:    float = PENALTY_PER_GAME,
    cap:         float = PENALTY_MAX,
    decay_games: int   = DECAY_GAMES,
) -> float:
    """Composite: ``base_penalty × decay_weight``."""
    return base_penalty(prior_playoff_games, per_game, cap) * \
           decay_weight_for(games_into_season, decay_games)


def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in PRIOR_PLAYOFF_LOAD_SCHEMA.items()}
    )


def _validate(load_df: pl.DataFrame) -> bool:
    missing = [c for c in LOAD_REQUIRED_COLS if c not in load_df.columns]
    if missing:
        warnings.warn(
            f"load_df missing columns: {missing}. Required: "
            f"{list(LOAD_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# PriorPlayoffLoad
# ---------------------------------------------------------------------------

class PriorPlayoffLoad:
    """Prior-playoff-load → season-start penalty (Feature 3.15)."""

    def __init__(
        self,
        penalty_per_game: float = PENALTY_PER_GAME,
        penalty_max:      float = PENALTY_MAX,
        decay_games:      int   = DECAY_GAMES,
    ) -> None:
        if penalty_per_game < 0.0:
            raise ValueError("penalty_per_game must be >= 0")
        if penalty_max < 0.0:
            raise ValueError("penalty_max must be >= 0")
        if decay_games < 1:
            raise ValueError("decay_games must be >= 1")
        self._per_game    = float(penalty_per_game)
        self._max         = float(penalty_max)
        self._decay_games = int(decay_games)
        self._version     = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, load_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        """Return one row per player with the time-varying playoff penalty."""
        if not _validate(load_df):
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(load_df) == 0:
            return _empty_output()

        rows = load_df.to_dicts()
        seen: set[int] = set()
        out_rows: list[dict] = []
        for r in rows:
            pid_raw = r.get("player_id")
            if pid_raw is None:
                continue
            pid = int(pid_raw)
            if pid in seen:
                continue
            seen.add(pid)

            try:
                pp_games = int(r.get("prior_playoff_games") or 0)
                gp_now   = int(r.get("games_played_this_season") or 0)
            except (TypeError, ValueError):
                continue
            if pp_games < 0 or gp_now < 0:
                continue

            round_raw = r.get("prior_playoff_round")
            try:
                pp_round = int(round_raw) if round_raw is not None else 0
            except (TypeError, ValueError):
                pp_round = 0
            pp_round = max(0, min(4, pp_round))

            base = base_penalty(pp_games, self._per_game, self._max)
            w    = decay_weight_for(gp_now, self._decay_games)

            out_rows.append({
                "player_id":                pid,
                "as_of_date":               as_of_date,
                "prior_playoff_games":      pp_games,
                "prior_playoff_round":      pp_round,
                "games_played_this_season": gp_now,
                "base_playoff_penalty":     float(base),
                "playoff_load_penalty":     float(base * w),
                "decay_weight":             float(w),
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=PRIOR_PLAYOFF_LOAD_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_prior_playoff_load(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write prior-playoff-load DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"prior_playoff_load_{as_of_date}.parquet"
    for col, dtype in PRIOR_PLAYOFF_LOAD_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(PRIOR_PLAYOFF_LOAD_SCHEMA.keys())).write_parquet(path)
    return path
