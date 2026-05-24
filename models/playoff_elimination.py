"""Playoff Elimination Fatigue — Feature 4.20.

Team-level motivational regression when ``playoff_prob < 25%`` AND
``games_remaining < 30``.  Gradual ramp — full activation at ``<10%``
with ``<15`` games left.

Approach
--------
V1 builds a lightweight playoff probability calculator from current
standings (W/L/OT → points%) and games remaining.  Not a full Monte
Carlo season simulator (that's Phase 10.3) — just a logistic fit:

    ``playoff_prob = sigmoid(k × (points_pct − threshold) × sqrt(games_remaining))``

where ``threshold ≈ 0.560`` is the historical 82-game playoff cutoff and
``k`` is calibrated so that a team 5 points% below threshold with 20 GP
left has ~15% probability.

Output
------
One row per team:

- team, season, gp, points_pct, games_remaining
- playoff_prob ∈ [0, 1]
- elimination_drag ∈ [0, 1] — 0 when not eliminated, ramps up to 1.0
- efficiency_multiplier — ``1 − elimination_drag × MAX_DRAG``

Output: ``playoff_elimination/playoff_elimination_{season}.parquet``
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import polars as pl


MODEL_VERSION = "playoff_elimination_v1"

REGULAR_SEASON_GP = 82
PLAYOFF_THRESHOLD = 0.560
LOGISTIC_K        = 12.0
MAX_DRAG          = 0.08


class DataMissingWarning(UserWarning):
    pass


PLAYOFF_ELIMINATION_SCHEMA: dict[str, pl.DataType] = {
    "team":                   pl.Utf8,
    "season":                 pl.Int64,
    "gp":                     pl.Int64,
    "points_pct":             pl.Float64,
    "games_remaining":        pl.Int64,
    "playoff_prob":           pl.Float64,
    "elimination_drag":       pl.Float64,
    "efficiency_multiplier":  pl.Float64,
    "model_version":          pl.Utf8,
}


def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _playoff_prob(points_pct: float, games_remaining: int) -> float:
    gap = points_pct - PLAYOFF_THRESHOLD
    return _sigmoid(LOGISTIC_K * gap * max(1.0, games_remaining ** 0.5))


def _elimination_drag(playoff_prob: float, games_remaining: int) -> float:
    if playoff_prob >= 0.25 or games_remaining >= 30:
        return 0.0
    # Ramp: full at <10% with <15 left; partial between 10-25% / 15-30 left
    prob_factor = max(0.0, min(1.0, (0.25 - playoff_prob) / 0.15))
    games_factor = max(0.0, min(1.0, (30 - games_remaining) / 15.0))
    return round(prob_factor * games_factor, 4)


def compute_playoff_elimination(
    team_stats_df: pl.DataFrame,
    team_lookup:   dict[int, str],
    season:        int,
) -> pl.DataFrame:
    required = {"team_id", "regulation_wins", "regulation_losses", "ot_games"}
    if team_stats_df.is_empty():
        warnings.warn("compute_playoff_elimination: empty team_stats.", DataMissingWarning, stacklevel=2)
        return pl.DataFrame(schema=PLAYOFF_ELIMINATION_SCHEMA)
    missing = required - set(team_stats_df.columns)
    if missing:
        raise ValueError(f"team_stats_df missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for r in team_stats_df.iter_rows(named=True):
        tid = int(r["team_id"])
        abbrev = team_lookup.get(tid)
        if abbrev is None:
            continue
        w  = int(r["regulation_wins"] or 0)
        l  = int(r["regulation_losses"] or 0)
        ot = int(r["ot_games"] or 0)
        gp = w + l + ot
        pts = 2 * w + ot
        pts_pct = pts / (2 * gp) if gp > 0 else 0.0
        remaining = max(0, REGULAR_SEASON_GP - gp)

        prob = _playoff_prob(pts_pct, remaining)
        drag = _elimination_drag(prob, remaining)
        eff  = round(1.0 - drag * MAX_DRAG, 4)

        rows.append({
            "team":                  abbrev,
            "season":                int(season),
            "gp":                    gp,
            "points_pct":            round(pts_pct, 4),
            "games_remaining":       remaining,
            "playoff_prob":          round(prob, 4),
            "elimination_drag":      drag,
            "efficiency_multiplier": eff,
            "model_version":         MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=PLAYOFF_ELIMINATION_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in PLAYOFF_ELIMINATION_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(PLAYOFF_ELIMINATION_SCHEMA.keys())).sort("elimination_drag", descending=True)


def write_playoff_elimination(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"playoff_elimination_{season}.parquet"
    df.write_parquet(path)
    return path


def read_playoff_elimination(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "playoff_elimination"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"playoff_elimination_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("playoff_elimination_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
