"""Trade Deadline Buyer/Seller Classifier — Feature 4.15.

Per-team ``buyer | seller | neutral`` classification based on standings
position and season progress.  Intended to run weekly starting ~Feb 1
through the trade deadline (~March 7).

Inputs
------
- ``team_stats_{season}.parquet`` — regulation W/L/OT for points% calc.
- Season calendar position: games played / 82 (proxy from W+L+OT).

Approach
--------
Lightweight heuristic classifier (v1) — not ML:

1. Compute points% = ``points / (2 * GP)`` where
   ``points = 2*W + OT`` and ``GP = W + L + OT``.
2. Estimate playoff threshold: the 8th-best points% in each conference
   is a rough cutoff.  Without conference mapping in v1, use a
   league-wide 16th-best proxy (i.e. top half of the league by pts%).
3. ``gap = points_pct - threshold``
4. Classification rules:
   - ``gap >= +0.04`` → **buyer** (comfortably in; willing to add)
   - ``gap <= -0.04`` → **seller** (comfortably out; selling pieces)
   - else → **neutral** (on the bubble; unpredictable)
5. ``confidence = min(1.0, abs(gap) / 0.10)`` — linear ramp from
   0.0 at the threshold to 1.0 at ±10 pts% gap.

Adjustments (v2 — planned but not wired in v1):
- Cap space (PuckPedia scrape from 2.25) — tight-cap buyers are
  constrained even if standings say buy.
- Pending UFA count — many expiring deals → more likely seller.
- Prospect pipeline depth — thin pipeline → less willing to sell.

Output
------
One row per team:

- team, season, gp, points, points_pct
- classification (buyer | seller | neutral)
- confidence ∈ [0, 1]
- gap (signed — positive = above threshold)
- threshold (the league 16th-best pts% used)

Output: ``buyer_seller/buyer_seller_{season}.parquet``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "buyer_seller_v1"

BUYER_GAP  = +0.04
SELLER_GAP = -0.04
CONFIDENCE_SCALE = 0.10


class DataMissingWarning(UserWarning):
    """Raised when buyer/seller data is absent or insufficient."""


BUYER_SELLER_SCHEMA: dict[str, pl.DataType] = {
    "team":             pl.Utf8,
    "season":           pl.Int64,
    "gp":               pl.Int64,
    "wins":             pl.Int64,
    "ot_games":         pl.Int64,
    "losses":           pl.Int64,
    "points":           pl.Int64,
    "points_pct":       pl.Float64,
    "threshold":        pl.Float64,
    "gap":              pl.Float64,
    "classification":   pl.Utf8,
    "confidence":       pl.Float64,
    "model_version":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_buyer_seller(
    team_stats_df: pl.DataFrame,
    team_lookup:   dict[int, str],
    season:        int,
) -> pl.DataFrame:
    """Classify every team as buyer / seller / neutral.

    Args:
        team_stats_df: raw team_stats parquet with columns
                       ``team_id, regulation_wins, regulation_losses, ot_games``.
        team_lookup:   team_id → abbrev.
        season:        NHL season start year.
    """
    required = {"team_id", "regulation_wins", "regulation_losses", "ot_games"}
    missing = required - set(team_stats_df.columns)
    if missing:
        raise ValueError(f"team_stats_df missing columns: {sorted(missing)}")

    if team_stats_df.is_empty():
        warnings.warn(
            "compute_buyer_seller: empty team_stats — no classification possible.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=BUYER_SELLER_SCHEMA)

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
        rows.append({
            "team":       abbrev,
            "gp":         gp,
            "wins":       w,
            "losses":     l,
            "ot_games":   ot,
            "points":     pts,
            "points_pct": pts_pct,
        })

    if not rows:
        return pl.DataFrame(schema=BUYER_SELLER_SCHEMA)

    # Compute threshold = 16th-best points% (top half of 32 teams).
    sorted_pcts = sorted([r["points_pct"] for r in rows], reverse=True)
    threshold_idx = min(15, len(sorted_pcts) - 1)
    threshold = sorted_pcts[threshold_idx]

    for r in rows:
        gap = r["points_pct"] - threshold
        if gap >= BUYER_GAP:
            classification = "buyer"
        elif gap <= SELLER_GAP:
            classification = "seller"
        else:
            classification = "neutral"
        conf = min(1.0, abs(gap) / CONFIDENCE_SCALE) if CONFIDENCE_SCALE > 0 else 0.0
        r["threshold"]      = round(threshold, 4)
        r["gap"]            = round(gap, 4)
        r["classification"] = classification
        r["confidence"]     = round(conf, 4)
        r["season"]         = int(season)
        r["model_version"]  = MODEL_VERSION

    df = pl.DataFrame(rows)
    for col, dtype in BUYER_SELLER_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(BUYER_SELLER_SCHEMA.keys())).sort("points_pct", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_buyer_seller(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"buyer_seller_{season}.parquet"
    df.write_parquet(path)
    return path


def read_buyer_seller(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "buyer_seller"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"buyer_seller_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("buyer_seller_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
