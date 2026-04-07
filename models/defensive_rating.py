"""Composite Defensive Rating (CDR) — Feature 2.26.

Combines outcome and process defensive signals into a single z-scored rating per player.

Formula:
    CDR = 0.60 × z(EV_Def_RAPM_xGA)
        + 0.25 × z(TK/GV ratio)
        + 0.15 × z(–xGA_60_dzs_adj)

Components:
    EV_Def_RAPM_xGA  — Evolving Hockey RAPM defensive component; isolates individual
                        defensive impact from teammates and opponents. Higher = better.
    TK/GV ratio      — Takeaways ÷ max(Giveaways, 1) per player; active puck recovery.
                        Higher = better.
    xGA_60_dzs_adj   — On-ice expected goals against per 60, adjusted for defensive zone
                        start share; inverted before z-scoring so higher CDR = better.
                        DZS adjustment: xga_60_adj = xga_60 − (DZS% − 0.50) × 0.10

All components z-scored across qualified players (≥ MIN_GP games played).
Output CDR is mean≈0, roughly normal; positive = better than average defensively.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_GP = 15  # minimum games played to qualify

_WEIGHTS = {
    "rapm": 0.60,
    "tk_gv": 0.25,
    "xga": 0.15,
}

CDR_SCHEMA: dict[str, pl.DataType] = {
    "player_id":   pl.Int64,
    "player_name": pl.Utf8,
    "team":        pl.Utf8,
    "season":      pl.Int64,
    "gp":          pl.Int64,
    "toi":         pl.Float64,
    "takeaways":   pl.Int64,
    "giveaways":   pl.Int64,
    "tk_gv_ratio": pl.Float64,
    "xga_60":      pl.Float64,
    "xga_60_adj":  pl.Float64,
    "rapm_xga":    pl.Float64,
    "rapm_xga_z":  pl.Float64,
    "tk_gv_z":     pl.Float64,
    "xga_adj_z":   pl.Float64,
    "cdr":         pl.Float64,
}


# ---------------------------------------------------------------------------
# CompositeDefensiveRating
# ---------------------------------------------------------------------------


class CompositeDefensiveRating:
    """Compute CDR from RAPM and PBP player stats DataFrames."""

    def compute(
        self,
        rapm_df: pl.DataFrame,
        pbp_stats_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Compute CDR for all qualified players.

        Args:
            rapm_df:      Evolving Hockey RAPM parquet (columns: player_name, player_id,
                          team, gp, toi, rapm_xga, xga_60, dzs_pct).
                          dzs_pct is optional — defaults to 0.50 if missing.
            pbp_stats_df: Player stats from historical ingestor (columns: player_id or
                          player_name, takeaway_count, giveaway_count).
            season:       Season integer (e.g. 2025).

        Returns:
            Polars DataFrame matching CDR_SCHEMA, one row per qualified player,
            sorted descending by CDR.

        Raises:
            ValueError: if rapm_df is missing required columns or has no qualified rows.
        """
        required = {"player_name", "gp", "toi", "rapm_xga", "xga_60"}
        missing = required - set(rapm_df.columns)
        if missing:
            raise ValueError(f"rapm_df missing required columns: {sorted(missing)}")

        # Filter to qualified players and this season if season column present
        df = rapm_df
        if "season" in df.columns:
            df = df.filter(pl.col("season") == season)
        df = df.filter(pl.col("gp") >= MIN_GP)

        if df.is_empty():
            raise ValueError(
                f"No qualified players found (season={season}, min_gp={MIN_GP}). "
                "Ensure EH RAPM data has been synced for this season."
            )

        # Normalise player_id — may be null in RAPM data
        if "player_id" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("player_id"))

        # Normalise team — use first team entry if multiple rows per player (multi-team trades)
        if "team" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("team"))

        # DZS adjustment — subtract deployment bias before z-scoring xGA
        if "dzs_pct" in df.columns:
            df = df.with_columns(
                (pl.col("xga_60") - (pl.col("dzs_pct") - 0.50) * 0.10).alias("xga_60_adj")
            )
        else:
            df = df.with_columns(pl.col("xga_60").alias("xga_60_adj"))

        # ── Join T/G data ─────────────────────────────────────────────────────
        tk_gv = self._prepare_tk_gv(pbp_stats_df)
        if not tk_gv.is_empty():
            # Join on player_id if available, else player_name
            if "player_id" in tk_gv.columns and df["player_id"].drop_nulls().len() > 0:
                df = df.join(tk_gv.select(["player_id", "takeaway_count", "giveaway_count"]),
                             on="player_id", how="left")
            else:
                df = df.join(
                    tk_gv.select(["player_name", "takeaway_count", "giveaway_count"]),
                    on="player_name", how="left",
                )
        else:
            df = df.with_columns(
                pl.lit(None).cast(pl.Int64).alias("takeaway_count"),
                pl.lit(None).cast(pl.Int64).alias("giveaway_count"),
            )

        # Fill missing T/G with league medians (don't penalise for missing data)
        median_tk = int(df["takeaway_count"].drop_nulls().median() or 30)
        median_gv = int(df["giveaway_count"].drop_nulls().median() or 30)
        df = df.with_columns(
            pl.col("takeaway_count").fill_null(median_tk),
            pl.col("giveaway_count").fill_null(median_gv),
        )

        # TK/GV ratio — clamp denominator to 1 to avoid division by zero
        df = df.with_columns(
            (pl.col("takeaway_count").cast(pl.Float64) /
             pl.col("giveaway_count").cast(pl.Float64).clip(lower_bound=1.0))
            .alias("tk_gv_ratio")
        )

        # ── Z-score each component ────────────────────────────────────────────
        rapm_arr  = df["rapm_xga"].to_numpy()
        tk_gv_arr = df["tk_gv_ratio"].to_numpy()
        xga_arr   = df["xga_60_adj"].to_numpy()

        rapm_z  = _zscore(rapm_arr)       # higher RAPM xGA = worse; EH publishes it as positive = better
        tk_gv_z = _zscore(tk_gv_arr)      # higher TK/GV = better
        xga_z   = _zscore(-xga_arr)       # lower xGA = better → invert before z-scoring

        cdr = (
            _WEIGHTS["rapm"] * rapm_z
            + _WEIGHTS["tk_gv"] * tk_gv_z
            + _WEIGHTS["xga"] * xga_z
        )

        # ── Assemble output ───────────────────────────────────────────────────
        result = df.with_columns([
            pl.lit(season).alias("season"),
            pl.Series("rapm_xga_z", rapm_z),
            pl.Series("tk_gv_z",    tk_gv_z),
            pl.Series("xga_adj_z",  xga_z),
            pl.Series("cdr",        cdr),
        ]).rename({
            "takeaway_count": "takeaways",
            "giveaway_count": "giveaways",
        })

        # Select and cast to schema
        cols = list(CDR_SCHEMA.keys())
        result = result.select([c for c in cols if c in result.columns])
        # Add any missing schema columns as null
        for col, dtype in CDR_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))
        result = result.select(cols)

        return result.sort("cdr", descending=True)

    @staticmethod
    def _prepare_tk_gv(pbp_stats_df: pl.DataFrame) -> pl.DataFrame:
        """Extract takeaway/giveaway counts from player stats DataFrame.

        Accepts DataFrames with either:
        - Columns: player_id, takeaway_count, giveaway_count
        - Columns: player_name, takeaway_count, giveaway_count
        """
        if pbp_stats_df.is_empty():
            return pl.DataFrame()
        needed = {"takeaway_count", "giveaway_count"}
        if not needed.issubset(set(pbp_stats_df.columns)):
            return pl.DataFrame()
        keep = list(needed)
        if "player_id" in pbp_stats_df.columns:
            keep = ["player_id"] + keep
        if "player_name" in pbp_stats_df.columns:
            keep = ["player_name"] + keep
        return pbp_stats_df.select(keep).filter(
            pl.col("takeaway_count").is_not_null() | pl.col("giveaway_count").is_not_null()
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_cdr(df: pl.DataFrame, out_dir: Path, season: int) -> Path:
    """Write CDR parquet to out_dir/cdr_{season}.parquet. Creates dir if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cdr_{season}.parquet"
    df.write_parquet(path)
    return path


def read_cdr(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    """Read CDR parquet. If season is None, reads the most recent available."""
    cdr_dir = data_dir / "cdr"
    if not cdr_dir.exists():
        return None
    parquets = sorted(cdr_dir.glob("cdr_*.parquet"))
    if not parquets:
        return None
    if season is not None:
        path = cdr_dir / f"cdr_{season}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)
    return pl.read_parquet(parquets[-1])


def lookup_player(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Case-insensitive player name lookup. Returns matching rows (may be empty)."""
    return df.filter(pl.col("player_name").str.to_lowercase() == name.strip().lower())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score an array. Returns zeros if std is 0 (all values identical)."""
    arr = arr.astype(float)
    # Replace NaN with mean before z-scoring so missing values land at 0
    mean = float(np.nanmean(arr))
    std  = float(np.nanstd(arr))
    arr = np.where(np.isnan(arr), mean, arr)
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - mean) / std
