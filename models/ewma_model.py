"""EWMA Form Metric — Feature 2.13.

20-game rolling EWMA (λ=0.96, ~17-game half-life) on xGF/60, Corsi%,
and save% (goalies).  Produces a fast hot/cold signal that runs in
parallel to the Bayesian posterior — it does not replace it.

EWMA formula (Polars convention: α = 1 − λ = 0.04):
    ewma_t = (1−α) × ewma_{t-1} + α × value_t
           = 0.96  × ewma_{t-1} + 0.04 × value_t

Hot/cold thresholds:
    hot:     ewma > season_mean + HOT_SIGMA_THRESHOLD × season_std
    cold:    ewma < season_mean − COLD_SIGMA_THRESHOLD × season_std
    neutral: otherwise

Half-life derivation:
    (1−α)^h = 0.5  →  h = ln(0.5) / ln(1−α) = ln(0.5) / ln(0.96) ≈ 17 games

Outputs
-------
``player_form_{season}.parquet`` — one row per (player, game):
    player_id, game_id, game_number, ewma_xgf60, ewma_corsi_pct,
    ewma_sv_pct (nullable for skaters), form_flag, season, model_version

``player_form_summary_{season}.parquet`` — one row per player (latest snapshot):
    player_id, player_name, season, games_played, ewma_xgf60,
    ewma_corsi_pct, ewma_sv_pct, form_flag, season_mean_xgf60,
    season_std_xgf60, model_version
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION          = "ewma_v1"
EWMA_LAMBDA            = 0.96         # decay factor (weight on previous EWMA)
EWMA_ALPHA             = 1.0 - EWMA_LAMBDA   # = 0.04 (Polars ewm_mean alpha)
EWMA_HALF_LIFE_GAMES   = np.log(0.5) / np.log(EWMA_LAMBDA)   # ≈ 17 games
HOT_SIGMA_THRESHOLD    = 0.75         # std devs above season mean → hot
COLD_SIGMA_THRESHOLD   = 0.75         # std devs below season mean → cold
MIN_GAMES_FOR_FLAG     = 5            # minimum GP before assigning hot/cold


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

PLAYER_FORM_SCHEMA: dict[str, pl.DataType] = {
    "player_id":      pl.Int64,
    "game_id":        pl.Int64,
    "game_number":    pl.Int64,
    "season":         pl.Int64,
    "ewma_xgf60":     pl.Float64,
    "ewma_corsi_pct": pl.Float64,
    "ewma_sv_pct":    pl.Float64,   # null for non-goalies
    "form_flag":      pl.Utf8,      # "hot" | "cold" | "neutral"
    "model_version":  pl.Utf8,
}

PLAYER_FORM_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "player_id":          pl.Int64,
    "player_name":        pl.Utf8,
    "season":             pl.Int64,
    "games_played":       pl.Int64,
    "ewma_xgf60":         pl.Float64,
    "ewma_corsi_pct":     pl.Float64,
    "ewma_sv_pct":        pl.Float64,
    "form_flag":          pl.Utf8,
    "season_mean_xgf60":  pl.Float64,
    "season_std_xgf60":   pl.Float64,
    "model_version":      pl.Utf8,
}


# ---------------------------------------------------------------------------
# Core EWMA computation (single player, pure NumPy)
# ---------------------------------------------------------------------------

def compute_ewma_series(
    values: np.ndarray,
    alpha: float = EWMA_ALPHA,
    warm_start: float | None = None,
) -> np.ndarray:
    """Compute EWMA for a 1-D array of values.

    Args:
        values:     Ordered observations (oldest first).
        alpha:      Smoothing factor (0 < α ≤ 1).  α=0.04 for λ=0.96.
        warm_start: Initial EWMA value.  Defaults to values[0] if None.

    Returns:
        EWMA array of same length as values.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)

    result = np.empty(n, dtype=np.float64)
    ewma = float(warm_start) if warm_start is not None else float(values[0])
    for i, v in enumerate(values):
        if np.isnan(v):
            result[i] = ewma
        else:
            ewma = (1.0 - alpha) * ewma + alpha * v
            result[i] = ewma
    return result


def _form_flag(
    ewma_val: float,
    season_mean: float,
    season_std: float,
    n_games: int,
) -> str:
    """Classify form as hot / cold / neutral."""
    if n_games < MIN_GAMES_FOR_FLAG or season_std < 1e-9:
        return "neutral"
    z = (ewma_val - season_mean) / season_std
    if z >= HOT_SIGMA_THRESHOLD:
        return "hot"
    if z <= -COLD_SIGMA_THRESHOLD:
        return "cold"
    return "neutral"


# ---------------------------------------------------------------------------
# EWMAFormModel — Feature 2.13
# ---------------------------------------------------------------------------

class EWMAFormModel:
    """EWMA form metric model (Feature 2.13).

    Computes rolling EWMA across each player's game log and emits
    hot/cold/neutral form flags.

    Usage::

        model = EWMAFormModel()
        form_df = model.compute(player_game_df, season=2025)
        summary_df = model.summary(player_game_df, season=2025)
    """

    def __init__(
        self,
        alpha: float = EWMA_ALPHA,
        hot_threshold:  float = HOT_SIGMA_THRESHOLD,
        cold_threshold: float = COLD_SIGMA_THRESHOLD,
    ) -> None:
        self._alpha     = alpha
        self._lambda    = 1.0 - alpha
        self._hot_thr   = hot_threshold
        self._cold_thr  = cold_threshold
        self._version   = MODEL_VERSION

    # ------------------------------------------------------------------
    # Game-by-game EWMA across all players
    # ------------------------------------------------------------------

    def compute(
        self,
        player_game_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Compute per-game EWMA for every player in the game log.

        Args:
            player_game_df: One row per (player, game). Required columns:
                player_id, game_id, game_number (or date ordering),
                xgf_per60.
                Optional columns: corsi_pct (CF%), sv_pct (goalies).

        Returns:
            DataFrame matching PLAYER_FORM_SCHEMA, sorted by
            (player_id, game_number).
        """
        required = {"player_id", "game_id", "xgf_per60"}
        missing = required - set(player_game_df.columns)
        if missing:
            raise ValueError(f"player_game_df missing columns: {sorted(missing)}")

        df = player_game_df

        # Add game_number if missing (use row order per player)
        if "game_number" not in df.columns:
            df = df.with_columns(
                pl.lit(0).alias("game_number")   # placeholder; filled below
            )

        has_corsi = "corsi_pct" in df.columns
        has_sv    = "sv_pct"    in df.columns

        # Process each player independently
        out_rows: list[dict] = []

        player_ids = df["player_id"].unique().to_list()

        for pid in player_ids:
            pdata = df.filter(pl.col("player_id") == pid)

            # Sort by game_id (proxy for chronological order)
            pdata = pdata.sort("game_id")

            n = len(pdata)
            xgf_vals    = pdata["xgf_per60"].cast(pl.Float64).to_numpy()
            game_ids    = pdata["game_id"].cast(pl.Int64).to_numpy()

            # Season-level stats for hot/cold threshold
            valid_xgf   = xgf_vals[~np.isnan(xgf_vals)]
            season_mean = float(np.mean(valid_xgf)) if len(valid_xgf) > 0 else 0.0
            season_std  = float(np.std(valid_xgf))  if len(valid_xgf) > 1 else 1.0

            ewma_xgf = compute_ewma_series(xgf_vals, self._alpha, warm_start=season_mean)

            ewma_corsi: np.ndarray | None = None
            if has_corsi:
                corsi_vals = pdata["corsi_pct"].cast(pl.Float64).to_numpy()
                valid_corsi = corsi_vals[~np.isnan(corsi_vals)]
                corsi_mean = float(np.mean(valid_corsi)) if len(valid_corsi) > 0 else 0.5
                ewma_corsi = compute_ewma_series(corsi_vals, self._alpha, warm_start=corsi_mean)

            ewma_sv: np.ndarray | None = None
            if has_sv:
                sv_vals   = pdata["sv_pct"].cast(pl.Float64).to_numpy()
                valid_sv  = sv_vals[~np.isnan(sv_vals)]
                sv_mean   = float(np.mean(valid_sv)) if len(valid_sv) > 0 else 0.910
                ewma_sv   = compute_ewma_series(sv_vals, self._alpha, warm_start=sv_mean)

            for i in range(n):
                flag = _form_flag(ewma_xgf[i], season_mean, season_std, i + 1)
                out_rows.append({
                    "player_id":      int(pid),
                    "game_id":        int(game_ids[i]),
                    "game_number":    i + 1,
                    "season":         season,
                    "ewma_xgf60":     float(ewma_xgf[i]),
                    "ewma_corsi_pct": float(ewma_corsi[i]) if ewma_corsi is not None else None,
                    "ewma_sv_pct":    float(ewma_sv[i])    if ewma_sv    is not None else None,
                    "form_flag":      flag,
                    "model_version":  self._version,
                })

        if not out_rows:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYER_FORM_SCHEMA.items()}
            )

        result = pl.DataFrame(out_rows).cast(
            {col: dt for col, dt in PLAYER_FORM_SCHEMA.items() if col in pl.DataFrame(out_rows).columns}
        )

        # Ensure all schema columns present
        for col, dtype in PLAYER_FORM_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))

        return result.select(list(PLAYER_FORM_SCHEMA.keys())).sort(["player_id", "game_number"])

    # ------------------------------------------------------------------
    # Per-player summary (latest EWMA snapshot)
    # ------------------------------------------------------------------

    def summary(
        self,
        player_game_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Return one-row-per-player summary with latest EWMA values.

        Args:
            player_game_df: Same input as compute().  Optional extra columns:
                player_name.

        Returns:
            DataFrame matching PLAYER_FORM_SUMMARY_SCHEMA, sorted by
            ewma_xgf60 descending.
        """
        form_df = self.compute(player_game_df, season)

        if form_df.is_empty():
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYER_FORM_SUMMARY_SCHEMA.items()}
            )

        # Latest row per player
        latest = form_df.group_by("player_id").agg([
            pl.col("game_number").max().alias("games_played"),
            pl.col("ewma_xgf60").last().alias("ewma_xgf60"),
            pl.col("ewma_corsi_pct").last().alias("ewma_corsi_pct"),
            pl.col("ewma_sv_pct").last().alias("ewma_sv_pct"),
            pl.col("form_flag").last().alias("form_flag"),
        ])

        # Season-level stats for context
        season_stats = (
            player_game_df
            .group_by("player_id")
            .agg([
                pl.col("xgf_per60").mean().alias("season_mean_xgf60"),
                pl.col("xgf_per60").std().alias("season_std_xgf60"),
            ])
        )

        result = latest.join(season_stats, on="player_id", how="left")

        # Join player_name if available
        if "player_name" in player_game_df.columns:
            names = (
                player_game_df
                .group_by("player_id")
                .agg(pl.col("player_name").first())
            )
            result = result.join(names, on="player_id", how="left")
        else:
            result = result.with_columns(pl.lit("").alias("player_name"))

        result = result.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit(self._version).alias("model_version"),
        ])

        # Enforce schema
        for col, dtype in PLAYER_FORM_SUMMARY_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))

        return (
            result
            .select(list(PLAYER_FORM_SUMMARY_SCHEMA.keys()))
            .sort("ewma_xgf60", descending=True)
        )

    # ------------------------------------------------------------------
    # Single-player incremental update
    # ------------------------------------------------------------------

    def update_player(
        self,
        current_ewma: float,
        new_observation: float,
    ) -> float:
        """Apply one EWMA step (used for live streaming updates).

        Args:
            current_ewma:    Current EWMA value.
            new_observation: Latest game's xGF/60 (or Corsi%, etc.).

        Returns:
            Updated EWMA value.
        """
        if np.isnan(new_observation):
            return current_ewma
        return (1.0 - self._alpha) * current_ewma + self._alpha * new_observation

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lambda_(self) -> float:
        return self._lambda

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def half_life_games(self) -> float:
        """Approximate half-life in games."""
        return float(np.log(0.5) / np.log(self._lambda))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_player_form(
    form_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_dir: Path,
    season: int,
) -> tuple[Path, Path]:
    """Write per-game form and summary to parquets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_form    = output_dir / f"player_form_{season}.parquet"
    p_summary = output_dir / f"player_form_summary_{season}.parquet"

    form_df.write_parquet(p_form)
    summary_df.write_parquet(p_summary)

    return p_form, p_summary


def read_player_form_summary(data_dir: Path, season: int) -> pl.DataFrame:
    """Read per-player form summary."""
    path = Path(data_dir) / "form" / f"player_form_summary_{season}.parquet"
    return pl.read_parquet(path)
