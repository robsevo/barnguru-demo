"""Special Teams Ratings — Feature 2.7.

Computes per-player Power Play (PP) and Penalty Kill (PK) ratings from
stint-level xG data produced by build_stints() in rapm_model.py.

Rating formula
--------------
PP rating  = 0.6 × z(pp_xgf60)   + 0.4 × z(rapm_pp)
PK rating  = 0.6 × z(-pk_xga60)  + 0.4 × z(rapm_pk)

Both ratings are z-scored across all qualified players so they are
mean≈0, roughly normal; positive = better than league average.

Qualification
-------------
- PP rating: player must have ≥ MIN_PP_TOI_SECS seconds of power-play TOI.
- PK rating: player must have ≥ MIN_PK_TOI_SECS seconds of penalty-kill TOI.

Inputs
------
- stints_df  from models.rapm_model.build_stints()
- rapm_df    from models.rapm_model output parquet
- season     integer (e.g. 2025)

Outputs
-------
Polars DataFrame matching SPECIAL_TEAMS_SCHEMA, one row per player
who qualifies for at least one situation (PP or PK).  Players who only
qualify for one situation have NaN ratings for the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION    = "special_teams_v1"
MIN_PP_TOI_SECS  = 120.0   # 2 minutes of PP ice time required
MIN_PK_TOI_SECS  = 120.0   # 2 minutes of PK ice time required


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

SPECIAL_TEAMS_SCHEMA: dict[str, pl.DataType] = {
    "player_id":     pl.Int64,
    "player_name":   pl.Utf8,
    "team":          pl.Utf8,
    "season":        pl.Int64,
    "gp":            pl.Int64,
    # Power play
    "toi_pp":        pl.Float64,
    "pp_xgf":        pl.Float64,
    "pp_xga":        pl.Float64,
    "pp_xgf60":      pl.Float64,
    "pp_xga60":      pl.Float64,
    "rapm_pp":       pl.Float64,
    "pp_rating":     pl.Float64,
    # Penalty kill
    "toi_pk":        pl.Float64,
    "pk_xgf":        pl.Float64,
    "pk_xga":        pl.Float64,
    "pk_xgf60":      pl.Float64,
    "pk_xga60":      pl.Float64,
    "rapm_pk":       pl.Float64,
    "pk_rating":     pl.Float64,
    # Metadata
    "model_version": pl.Utf8,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score an array; constant arrays return zeros; NaN → mean before scoring."""
    arr = arr.copy().astype(np.float64)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = np.nanmean(arr)
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _compute_situation_rates(
    stints_df: pl.DataFrame,
    situation: str,
    focal_side: str,
    xgf_col: str,
    xga_col: str,
) -> pl.DataFrame:
    """Aggregate per-player TOI + xGF + xGA for a given situation and side.

    Args:
        stints_df:  Output of build_stints() with columns:
                    situation, duration_secs, home_player_ids, away_player_ids,
                    xgf_home, xgf_away.
        situation:  "PP" or "PK".
        focal_side: "home_player_ids" or "away_player_ids" — which side gets credit.
        xgf_col:    Column name to use as xGF for the focal side ("xgf_home"/"xgf_away").
        xga_col:    Column name to use as xGA for the focal side ("xgf_away"/"xgf_home").

    Returns:
        DataFrame with columns: player_id (Int64), duration_secs (Float64),
        xgf (Float64), xga (Float64).
    """
    filtered = stints_df.filter(pl.col("situation") == situation)
    if filtered.is_empty():
        return pl.DataFrame(schema={
            "player_id":     pl.Int64,
            "duration_secs": pl.Float64,
            "xgf":           pl.Float64,
            "xga":           pl.Float64,
        })

    # Explode the focal side's player list so each player gets one row per stint
    expanded = (
        filtered
        .select([
            pl.col(focal_side).alias("player_ids"),
            pl.col("duration_secs"),
            pl.col(xgf_col).alias("xgf"),
            pl.col(xga_col).alias("xga"),
        ])
        .explode("player_ids")
        .rename({"player_ids": "player_id"})
        .drop_nulls("player_id")
        .with_columns(pl.col("player_id").cast(pl.Int64))
    )

    return (
        expanded
        .group_by("player_id")
        .agg([
            pl.col("duration_secs").sum(),
            pl.col("xgf").sum(),
            pl.col("xga").sum(),
        ])
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_special_teams(
    stints_df: pl.DataFrame,
    rapm_df: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Compute PP and PK ratings for all qualified players.

    Args:
        stints_df:  Stint-level DataFrame from models.rapm_model.build_stints().
        rapm_df:    RAPM output parquet for this season.  Required columns:
                    player_id (Int64), player_name (Utf8), team (Utf8),
                    gp (Int64), rapm_pp (Float64), rapm_pk (Float64).
        season:     Season integer (e.g. 2025).

    Returns:
        DataFrame matching SPECIAL_TEAMS_SCHEMA, sorted descending by pp_rating
        (NaN last).  Players with insufficient TOI in a situation receive NaN
        for that situation's rating.

    Raises:
        ValueError: if stints_df or rapm_df are missing required columns.
    """
    # --- Validate inputs -------------------------------------------------------
    required_stints = {
        "situation", "duration_secs",
        "home_player_ids", "away_player_ids",
        "xgf_home", "xgf_away",
    }
    missing_stints = required_stints - set(stints_df.columns)
    if missing_stints:
        raise ValueError(
            f"stints_df missing required columns: {sorted(missing_stints)}"
        )

    required_rapm = {"player_id", "player_name", "team", "gp", "rapm_pp", "rapm_pk"}
    missing_rapm = required_rapm - set(rapm_df.columns)
    if missing_rapm:
        raise ValueError(
            f"rapm_df missing required columns: {sorted(missing_rapm)}"
        )

    # --- Build PP stats --------------------------------------------------------
    # Home team on PP (situation == "PP"):   home gets PP credit, away gets PK
    pp_home = _compute_situation_rates(
        stints_df, "PP", "home_player_ids", "xgf_home", "xgf_away"
    ).rename({"duration_secs": "_toi", "xgf": "_xgf", "xga": "_xga"})

    # Away team on PP (situation == "PK" from HOME perspective means AWAY is on PP):
    # situation == "PK" → away is on PP (more skaters), home is on PK
    pp_away = _compute_situation_rates(
        stints_df, "PK", "away_player_ids", "xgf_away", "xgf_home"
    ).rename({"duration_secs": "_toi", "xgf": "_xgf", "xga": "_xga"})

    pp_combined = pl.concat([pp_home, pp_away], how="vertical")
    if not pp_combined.is_empty():
        pp_stats = (
            pp_combined
            .group_by("player_id")
            .agg([
                pl.col("_toi").sum().alias("toi_pp"),
                pl.col("_xgf").sum().alias("pp_xgf"),
                pl.col("_xga").sum().alias("pp_xga"),
            ])
        )
    else:
        pp_stats = pl.DataFrame(schema={
            "player_id": pl.Int64,
            "toi_pp":    pl.Float64,
            "pp_xgf":    pl.Float64,
            "pp_xga":    pl.Float64,
        })

    # --- Build PK stats --------------------------------------------------------
    # Home team on PK (situation == "PK" from HOME perspective):
    # home has fewer skaters → home is on PK
    pk_home = _compute_situation_rates(
        stints_df, "PK", "home_player_ids", "xgf_home", "xgf_away"
    ).rename({"duration_secs": "_toi", "xgf": "_xgf", "xga": "_xga"})

    # Away team on PK (situation == "PP" → home is on PP, away is on PK):
    pk_away = _compute_situation_rates(
        stints_df, "PP", "away_player_ids", "xgf_away", "xgf_home"
    ).rename({"duration_secs": "_toi", "xgf": "_xgf", "xga": "_xga"})

    pk_combined = pl.concat([pk_home, pk_away], how="vertical")
    if not pk_combined.is_empty():
        pk_stats = (
            pk_combined
            .group_by("player_id")
            .agg([
                pl.col("_toi").sum().alias("toi_pk"),
                pl.col("_xgf").sum().alias("pk_xgf"),
                pl.col("_xga").sum().alias("pk_xga"),
            ])
        )
    else:
        pk_stats = pl.DataFrame(schema={
            "player_id": pl.Int64,
            "toi_pk":    pl.Float64,
            "pk_xgf":    pl.Float64,
            "pk_xga":    pl.Float64,
        })

    # --- Compute per-60 rates --------------------------------------------------
    if not pp_stats.is_empty():
        pp_stats = pp_stats.with_columns([
            (pl.col("pp_xgf") / (pl.col("toi_pp") / 3600.0)).alias("pp_xgf60"),
            (pl.col("pp_xga") / (pl.col("toi_pp") / 3600.0)).alias("pp_xga60"),
        ])

    if not pk_stats.is_empty():
        pk_stats = pk_stats.with_columns([
            (pl.col("pk_xgf") / (pl.col("toi_pk") / 3600.0)).alias("pk_xgf60"),
            (pl.col("pk_xga") / (pl.col("toi_pk") / 3600.0)).alias("pk_xga60"),
        ])

    # --- Merge with RAPM -------------------------------------------------------
    # Filter rapm_df to this season if a season column is present
    rapm = rapm_df
    if "season" in rapm.columns:
        rapm = rapm.filter(pl.col("season") == season)

    rapm_slim = rapm.select(
        ["player_id", "player_name", "team", "gp", "rapm_pp", "rapm_pk"]
    ).with_columns(pl.col("player_id").cast(pl.Int64))

    # Start from RAPM player universe; left join ST rates
    df = rapm_slim
    if not pp_stats.is_empty():
        df = df.join(
            pp_stats.with_columns(pl.col("player_id").cast(pl.Int64)),
            on="player_id", how="left",
        )
    else:
        for col, dtype in [("toi_pp", pl.Float64), ("pp_xgf", pl.Float64),
                           ("pp_xga", pl.Float64), ("pp_xgf60", pl.Float64),
                           ("pp_xga60", pl.Float64)]:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    if not pk_stats.is_empty():
        df = df.join(
            pk_stats.with_columns(pl.col("player_id").cast(pl.Int64)),
            on="player_id", how="left",
        )
    else:
        for col, dtype in [("toi_pk", pl.Float64), ("pk_xgf", pl.Float64),
                           ("pk_xga", pl.Float64), ("pk_xgf60", pl.Float64),
                           ("pk_xga60", pl.Float64)]:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Fill zeros for null TOI / rates (players with no ST time at all)
    df = df.with_columns([
        pl.col("toi_pp").fill_null(0.0),
        pl.col("pp_xgf").fill_null(0.0),
        pl.col("pp_xga").fill_null(0.0),
        pl.col("pp_xgf60").fill_null(0.0),
        pl.col("pp_xga60").fill_null(0.0),
        pl.col("toi_pk").fill_null(0.0),
        pl.col("pk_xgf").fill_null(0.0),
        pl.col("pk_xga").fill_null(0.0),
        pl.col("pk_xgf60").fill_null(0.0),
        pl.col("pk_xga60").fill_null(0.0),
    ])

    # --- Z-score and compute ratings ------------------------------------------
    # PP rating (only for players with enough PP TOI)
    pp_qualified = df["toi_pp"].to_numpy() >= MIN_PP_TOI_SECS
    pk_qualified = df["toi_pk"].to_numpy() >= MIN_PK_TOI_SECS

    pp_xgf60_arr = df["pp_xgf60"].to_numpy()
    rapm_pp_arr  = df["rapm_pp"].to_numpy()
    pk_xga60_arr = df["pk_xga60"].to_numpy()
    rapm_pk_arr  = df["rapm_pk"].to_numpy()

    # Z-score over qualified subset only, then assign back
    pp_rating = np.full(len(df), np.nan)
    if pp_qualified.sum() >= 2:
        z_pp_xgf60 = _zscore(pp_xgf60_arr[pp_qualified])
        z_rapm_pp  = _zscore(rapm_pp_arr[pp_qualified])
        pp_rating[pp_qualified] = 0.6 * z_pp_xgf60 + 0.4 * z_rapm_pp

    pk_rating = np.full(len(df), np.nan)
    if pk_qualified.sum() >= 2:
        z_neg_pk_xga60 = _zscore(-pk_xga60_arr[pk_qualified])
        z_rapm_pk      = _zscore(rapm_pk_arr[pk_qualified])
        pk_rating[pk_qualified] = 0.6 * z_neg_pk_xga60 + 0.4 * z_rapm_pk

    # --- Assemble output ------------------------------------------------------
    result = df.with_columns([
        pl.lit(season).cast(pl.Int64).alias("season"),
        pl.Series("pp_rating", pp_rating, dtype=pl.Float64),
        pl.Series("pk_rating", pk_rating, dtype=pl.Float64),
        pl.lit(MODEL_VERSION).alias("model_version"),
    ])

    # Cast to schema types
    for col, dtype in SPECIAL_TEAMS_SCHEMA.items():
        if col in result.columns:
            result = result.with_columns(pl.col(col).cast(dtype))

    # Select only schema columns in schema order
    result = result.select(list(SPECIAL_TEAMS_SCHEMA.keys()))

    # Sort by pp_rating descending (NaN last), then pk_rating
    result = result.sort("pp_rating", descending=True, nulls_last=True)

    return result


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def top_pp_players(st_df: pl.DataFrame, n: int = 20) -> pl.DataFrame:
    """Return top-n players by PP rating (highest = best power play)."""
    return (
        st_df
        .filter(pl.col("pp_rating").is_not_nan() & pl.col("pp_rating").is_not_null())
        .sort("pp_rating", descending=True)
        .head(n)
    )


def top_pk_players(st_df: pl.DataFrame, n: int = 20) -> pl.DataFrame:
    """Return top-n players by PK rating (highest = best penalty killer)."""
    return (
        st_df
        .filter(pl.col("pk_rating").is_not_nan() & pl.col("pk_rating").is_not_null())
        .sort("pk_rating", descending=True)
        .head(n)
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_special_teams(
    df: pl.DataFrame,
    output_dir: Path,
    season: int,
) -> Path:
    """Write special teams ratings parquet.

    Args:
        df:         DataFrame matching SPECIAL_TEAMS_SCHEMA.
        output_dir: Directory to write into (created if absent).
        season:     Season integer used in filename.

    Returns:
        Path to the written parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"special_teams_{season}.parquet"
    df.write_parquet(path)
    return path


def read_special_teams(
    data_dir: Path,
    season: int | None = None,
) -> pl.DataFrame | None:
    """Read special teams ratings parquet.

    Args:
        data_dir: Root data directory (e.g. ~/.gretzky/data).
                  Looks for files in data_dir/special_teams/.
        season:   Season integer.  If None, returns the most recent available.

    Returns:
        DataFrame or None if no file is found.
    """
    st_dir = Path(data_dir) / "special_teams"
    if not st_dir.exists():
        return None

    if season is not None:
        path = st_dir / f"special_teams_{season}.parquet"
        return pl.read_parquet(path) if path.exists() else None

    # Find the most recent season available
    candidates = sorted(st_dir.glob("special_teams_*.parquet"))
    if not candidates:
        return None
    return pl.read_parquet(candidates[-1])
