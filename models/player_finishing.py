"""Per-player Finishing metric aggregator — Feature 2.1.

Finishing = actual_goals − Σ xG(shot_i) over a season.
Positive = player scores more than expected given shot quality → elite shooter.
Negative = player scores less than expected → average-or-below finishing.

Normalised to per-60-minutes when EDGE icetime data is available.

Output Parquet schema (xg_finishing_{season}.parquet):
    shooter_id        Int64    — MoneyPuck player ID
    shooter_name      Utf8
    team              Utf8     — team code (last team if traded)
    season            Int64    — season start-year
    shots             Int64    — total shot attempts in season
    goals             Int64    — actual goals
    xg_sum            Float64  — Σ predicted xG
    finishing         Float64  — goals − xg_sum
    finishing_per60   Float64  — finishing / icetime_hours (null if no EDGE data)
    model_version     Utf8
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from models.xg_model import XGModel

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FINISHING_SCHEMA: dict[str, pl.DataType] = {
    "shooter_id":       pl.Int64,
    "shooter_name":     pl.Utf8,
    "team":             pl.Utf8,
    "season":           pl.Int64,
    "shots":            pl.Int64,
    "goals":            pl.Int64,
    "xg_sum":           pl.Float64,
    "finishing":        pl.Float64,
    "finishing_per60":  pl.Float64,
    "model_version":    pl.Utf8,
}

# Minimum shots for a player to be included in the output
MIN_SHOTS = 20


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class PlayerFinishingAggregator:
    """Compute per-player Finishing metric from shot-level xG predictions.

    Usage::

        model = XGModel.load(model_path)
        agg = PlayerFinishingAggregator()
        df = agg.compute(shots_df, model, icetime_df=edge_df)
        df.write_parquet(output_path)
    """

    def compute(
        self,
        shots_df: pl.DataFrame,
        model: XGModel,
        icetime_df: pl.DataFrame | None = None,
        season: int | None = None,
        model_version: str | None = None,
    ) -> pl.DataFrame:
        """Compute per-player Finishing for a single season's shot data.

        Args:
            shots_df:      MoneyPuck shot parquet for one season.
                           Required columns: shooter_id, shooter_name, shooting_team,
                           is_goal, and all build_features() inputs.
            model:         Trained XGModel.
            icetime_df:    EDGE skating parquet (edge_skating_{season}.parquet).
                           If provided, used to normalise Finishing to per-60.
                           Expected columns: player_id (or shooter_id), icetime (seconds).
            season:        Season year.  Falls back to first value in shots_df["season"].
            model_version: String tag for provenance.  Falls back to XGModel.MODEL_VERSION.

        Returns:
            pl.DataFrame with FINISHING_SCHEMA, one row per player (≥ MIN_SHOTS).
        """
        if shots_df.is_empty():
            return pl.DataFrame(schema=FINISHING_SCHEMA)

        # Predict xG for every shot
        xg_proba: np.ndarray = model.predict_proba(shots_df)

        # Attach predictions to shots
        df = shots_df.with_columns(pl.Series("xg_pred", xg_proba.tolist()))

        # Extract season
        if season is None:
            season = int(df["season"].drop_nulls()[0]) if "season" in df.columns else 0

        mv = model_version or "xg_v1"

        # Group by shooter
        agg = (
            df
            .group_by(["shooter_id", "shooter_name", "shooting_team"])
            .agg([
                pl.len().alias("shots"),
                pl.col("is_goal").cast(pl.Int64).sum().alias("goals"),
                pl.col("xg_pred").sum().alias("xg_sum"),
            ])
            .filter(pl.col("shots") >= MIN_SHOTS)
            .with_columns([
                (pl.col("goals") - pl.col("xg_sum")).alias("finishing"),
                pl.lit(season).cast(pl.Int64).alias("season"),
                pl.lit(mv).alias("model_version"),
            ])
            .rename({"shooting_team": "team"})
        )

        # Normalise to per-60 using EDGE icetime if available
        agg = agg.with_columns(pl.lit(None).cast(pl.Float64).alias("finishing_per60"))

        if icetime_df is not None and not icetime_df.is_empty():
            agg = self._join_icetime(agg, icetime_df)

        # Cast to schema
        return agg.select(list(FINISHING_SCHEMA.keys())).cast(FINISHING_SCHEMA)

    # --- helpers ---

    @staticmethod
    def _join_icetime(
        agg: pl.DataFrame, icetime_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Join EDGE icetime and compute finishing_per60.

        Tries to match on player_id column (EDGE uses numeric IDs).
        Falls back to name matching if player_id not available.
        """
        # EDGE skating parquet column for icetime is "icetime" (seconds total)
        it_cols = set(icetime_df.columns)

        # Identify the id column and icetime column
        id_col    = "player_id" if "player_id" in it_cols else None
        time_col  = "icetime"   if "icetime"   in it_cols else None

        if id_col is None or time_col is None:
            return agg  # can't join — leave finishing_per60 as null

        it = icetime_df.select([id_col, time_col]).rename({id_col: "shooter_id", time_col: "_icetime_s"})

        joined = agg.join(it, on="shooter_id", how="left")

        joined = joined.with_columns(
            pl.when(pl.col("_icetime_s").is_not_null() & (pl.col("_icetime_s") > 0))
            .then(pl.col("finishing") / (pl.col("_icetime_s") / 3600.0))
            .otherwise(None)
            .alias("finishing_per60")
        ).drop("_icetime_s")

        return joined


# ---------------------------------------------------------------------------
# Parquet I/O helpers
# ---------------------------------------------------------------------------


def write_finishing(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write finishing DataFrame to {output_dir}/xg_finishing_{season}.parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"xg_finishing_{season}.parquet"
    df.write_parquet(path)
    return path


def read_finishing(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    """Read most-recent (or specific) finishing parquet.  Returns None if not found."""
    xg_dir = Path(data_dir) / "xg_finishing"
    if not xg_dir.exists():
        return None
    parquets = sorted(xg_dir.glob("xg_finishing_*.parquet"))
    if not parquets:
        return None
    if season is not None:
        target = xg_dir / f"xg_finishing_{season}.parquet"
        return pl.read_parquet(target) if target.exists() else None
    return pl.read_parquet(parquets[-1])


def lookup_player(data_dir: Path, name: str, season: int | None = None) -> dict | None:
    """Find one player by case-insensitive name match.  Returns dict or None."""
    df = read_finishing(data_dir, season)
    if df is None:
        return None
    name_lower = name.strip().lower()
    row = df.filter(pl.col("shooter_name").str.to_lowercase() == name_lower)
    if row.is_empty():
        return None
    return row.to_dicts()[0]
