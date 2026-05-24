"""Seller Motivation State — Feature 4.16.

When a team is a confirmed **seller** (4.15) and the trade deadline has
passed, the team's competitive effort drops measurably:

- TOI distribution shifts to youth (prospects get NHL auditions).
- Line quality drops (top-6 traded, replaced by AHL depth).
- Clutch-time compete level falls (nothing to play for).

This module produces a per-team ``seller_drag`` ∈ [0, 1] that the game
setup builder (7.1) applies as a team efficiency baseline multiplier:

    effective_baseline = baseline × (1 − seller_drag × MAX_DRAG)

MAX_DRAG is capped at 0.06 (~6% efficiency loss at maximum drag).

Decay
-----
``seller_drag`` is strongest in the first 2 weeks after the deadline
(~6 games) and decays linearly to zero over ``DECAY_GAMES`` (default
20).  ``games_since_deadline`` is estimated from the team's GP relative
to the deadline date.

Inputs
------
- ``buyer_seller_{season}.parquet`` — classification + confidence.
- Team game count (from team_stats or coach_profile GP).

Output
------
One row per team with:

- team, season, classification, seller_confidence
- games_since_deadline (estimated), seller_drag, efficiency_multiplier
- contextual_flag (str — human-readable for "Ask GRETZKY")

Output: ``seller_motivation/seller_motivation_{season}.parquet``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "seller_motivation_v1"

DECAY_GAMES = 20
MAX_DRAG    = 0.06

# NHL trade deadline is typically early March.  As a proxy, we estimate
# games_since_deadline from GP: a team at ~65 GP is roughly at the
# deadline; at 82 GP the season is over.  This is crude but sufficient
# for v1 — the real date will come from the schedule ingester in v2.
DEADLINE_GP_PROXY = 65


class DataMissingWarning(UserWarning):
    """Raised when seller motivation data is absent."""


SELLER_MOTIVATION_SCHEMA: dict[str, pl.DataType] = {
    "team":                   pl.Utf8,
    "season":                 pl.Int64,
    "classification":         pl.Utf8,
    "seller_confidence":      pl.Float64,
    "gp":                     pl.Int64,
    "games_since_deadline":   pl.Int64,
    "seller_drag":            pl.Float64,
    "efficiency_multiplier":  pl.Float64,
    "contextual_flag":        pl.Utf8,
    "model_version":          pl.Utf8,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_seller_motivation(
    buyer_seller_df: pl.DataFrame,
    season:          int,
    deadline_gp:     int = DEADLINE_GP_PROXY,
    decay_games:     int = DECAY_GAMES,
    max_drag:        float = MAX_DRAG,
) -> pl.DataFrame:
    """Compute per-team seller motivation drag.

    Args:
        buyer_seller_df: output of ``compute_buyer_seller`` (4.15).
        season:          NHL season start year.
        deadline_gp:     approximate team GP at the trade deadline.
        decay_games:     games over which seller_drag decays to zero.
        max_drag:        maximum efficiency reduction (fraction).
    """
    required = {"team", "classification", "confidence", "gp"}
    if buyer_seller_df.is_empty():
        warnings.warn(
            "compute_seller_motivation: empty buyer_seller frame.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=SELLER_MOTIVATION_SCHEMA)

    missing = required - set(buyer_seller_df.columns)
    if missing:
        raise ValueError(f"buyer_seller_df missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for r in buyer_seller_df.iter_rows(named=True):
        team  = str(r["team"])
        cls   = str(r["classification"])
        conf  = float(r.get("confidence") or 0.0)
        gp    = int(r.get("gp") or 0)

        games_since = max(0, gp - deadline_gp)

        if cls == "seller" and games_since > 0:
            # Linear decay: 1.0 at deadline → 0.0 at deadline + decay_games.
            raw_drag = max(0.0, 1.0 - games_since / decay_games) * conf
            drag     = round(min(1.0, raw_drag), 4)
            eff_mult = round(1.0 - drag * max_drag, 4)
            if drag >= 0.5:
                flag = f"ACTIVE SELLER — {team} post-deadline drag {drag:.2f}, efficiency ×{eff_mult:.3f}"
            elif drag > 0:
                flag = f"FADING SELLER — {team} drag {drag:.2f}, efficiency ×{eff_mult:.3f}"
            else:
                flag = ""
        else:
            drag     = 0.0
            eff_mult = 1.0
            flag     = ""

        rows.append({
            "team":                  team,
            "season":                int(season),
            "classification":        cls,
            "seller_confidence":     conf,
            "gp":                    gp,
            "games_since_deadline":  games_since,
            "seller_drag":           drag,
            "efficiency_multiplier": eff_mult,
            "contextual_flag":       flag,
            "model_version":         MODEL_VERSION,
        })

    df = pl.DataFrame(rows)
    for col, dtype in SELLER_MOTIVATION_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(SELLER_MOTIVATION_SCHEMA.keys())).sort("seller_drag", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_seller_motivation(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"seller_motivation_{season}.parquet"
    df.write_parquet(path)
    return path


def read_seller_motivation(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "seller_motivation"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"seller_motivation_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("seller_motivation_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
