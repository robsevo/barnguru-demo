"""Strength of Schedule Normalizer — Feature 2.15.

Weights each current-season game observation by the quality of the
opponent faced.  High-quality opponents → evidence is more informative.
Low-quality opponents → evidence gets discounted.

This prevents easy-schedule mirages from contaminating the Bayesian
posterior: a player on a hot streak against bottom-5 defences should
update their posterior less than the same streak against elite defences.

Weight formula:
    w_i = clip(q_opp / q_league_avg,  W_MIN,  W_MAX)

where:
    q_opp        = opponent quality metric (e.g. posterior_mean xGF/60
                   allowed, or opponent defensive RAPM)
    q_league_avg = median q across all opponents seen this season
    W_MIN = 0.50  (weakest opponents get half weight)
    W_MAX = 2.00  (strongest opponents get double weight)

Integration with InSeasonBayesianPipeline:
    The effective obs_sigma for a weighted game is:
        effective_sigma = base_obs_sigma / sqrt(w_i)
    (higher weight = lower effective sigma = more informative observation)

Outputs
-------
``sos_weights_{season}.parquet`` — one row per player-game:
    player_id, game_id, opponent_id, opponent_quality,
    sos_weight, effective_sigma, season, model_version
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl

from models.bayesian_player_rating import OBS_SIGMA_XGF60
from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION     = "sos_v1"
SOS_W_MIN         = 0.50    # minimum weight (vs. weakest opponents)
SOS_W_MAX         = 2.00    # maximum weight (vs. strongest opponents)
SOS_QUALITY_COL   = "posterior_mean"   # preferred column name in ratings_df


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

SOS_WEIGHT_SCHEMA: dict[str, pl.DataType] = {
    "player_id":         pl.Int64,
    "game_id":           pl.Int64,
    "opponent_id":       pl.Int64,
    "opponent_quality":  pl.Float64,
    "sos_weight":        pl.Float64,
    "effective_sigma":   pl.Float64,
    "season":            pl.Int64,
    "model_version":     pl.Utf8,
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_sos_weights(
    opponent_qualities: np.ndarray,
    base_obs_sigma: float = OBS_SIGMA_XGF60,
    w_min: float = SOS_W_MIN,
    w_max: float = SOS_W_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SoS weights from an array of opponent quality scores.

    Args:
        opponent_qualities: Array of opponent quality values per game.
        base_obs_sigma:     Baseline per-game observation noise.
        w_min:              Minimum weight (floor).
        w_max:              Maximum weight (ceiling).

    Returns:
        (weights, effective_sigmas) — both arrays of shape (n,).
        weights ∈ [w_min, w_max].
        effective_sigma = base_obs_sigma / sqrt(weight).
    """
    n = len(opponent_qualities)
    if n == 0:
        return np.array([]), np.array([])

    valid = opponent_qualities[~np.isnan(opponent_qualities)]
    if len(valid) == 0:
        weights = np.ones(n)
    else:
        league_avg = float(np.median(valid))
        if abs(league_avg) < 1e-9:
            weights = np.ones(n)
        else:
            raw_weights = opponent_qualities / league_avg
            weights = np.where(
                np.isnan(raw_weights),
                1.0,
                np.clip(raw_weights, w_min, w_max),
            )

    effective_sigmas = base_obs_sigma / np.sqrt(weights)
    return weights, effective_sigmas


# ---------------------------------------------------------------------------
# StrengthOfScheduleNormalizer — Feature 2.15
# ---------------------------------------------------------------------------

class StrengthOfScheduleNormalizer:
    """Strength of schedule normalizer (Feature 2.15).

    Computes per-game opponent quality weights for use with the
    InSeasonBayesianPipeline.

    Usage::

        normalizer = StrengthOfScheduleNormalizer()
        # Load opponent ratings (defensive quality or posterior_mean):
        normalizer.update_opponent_ratings(ratings_df)

        # For a player's game log:
        weighted_df = normalizer.add_weights(player_game_df, season=2025)

        # Get effective obs_sigma for one game:
        sigma = normalizer.effective_sigma(opponent_id=8471215)
    """

    def __init__(
        self,
        quality_col:    str   = SOS_QUALITY_COL,
        w_min:          float = SOS_W_MIN,
        w_max:          float = SOS_W_MAX,
        base_obs_sigma: float = OBS_SIGMA_XGF60,
    ) -> None:
        self._quality_col    = quality_col
        self._w_min          = w_min
        self._w_max          = w_max
        self._base_obs_sigma = base_obs_sigma
        self._version        = MODEL_VERSION
        # opponent_id → quality score
        self._opponent_quality: dict[int, float] = {}
        self._league_avg: float = 1.0

    # ------------------------------------------------------------------
    # Opponent quality registry
    # ------------------------------------------------------------------

    def update_opponent_ratings(self, ratings_df: pl.DataFrame) -> None:
        """Load or refresh opponent quality scores from a ratings DataFrame.

        Args:
            ratings_df: Must have columns player_id and quality_col
                        (default: posterior_mean).  Use defensive RAPM,
                        goalie GSAx/60, or any quality-inverted metric.
        """
        if "player_id" not in ratings_df.columns:
            raise ValueError("ratings_df must have 'player_id' column.")

        q_col = self._quality_col
        if q_col not in ratings_df.columns:
            available = [c for c in ["posterior_mean", "off_rapm", "def_rapm", "gsax_per60"]
                         if c in ratings_df.columns]
            if not available:
                warnings.warn(
                    f"ratings_df has no quality column '{q_col}' or fallbacks. "
                    "Using uniform weights (SoS disabled).",
                    DataMissingWarning,
                    stacklevel=2,
                )
                return
            q_col = available[0]

        for row in ratings_df.select(["player_id", q_col]).to_dicts():
            pid = int(row["player_id"])
            val = row.get(q_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                self._opponent_quality[pid] = float(val)

        # Update league average
        if self._opponent_quality:
            vals = list(self._opponent_quality.values())
            self._league_avg = float(np.median(vals))
            if abs(self._league_avg) < 1e-9:
                self._league_avg = float(np.mean(np.abs(vals))) or 1.0

    def opponent_quality(self, opponent_id: int) -> float | None:
        """Return quality score for an opponent, or None if unknown."""
        return self._opponent_quality.get(opponent_id)

    def effective_sigma(
        self,
        opponent_id: int | None = None,
        opponent_quality: float | None = None,
    ) -> float:
        """Return effective obs_sigma for one game.

        Provide either opponent_id (looked up in registry) or
        direct opponent_quality float.
        """
        if opponent_quality is None and opponent_id is not None:
            opponent_quality = self._opponent_quality.get(opponent_id)

        if opponent_quality is None or abs(self._league_avg) < 1e-9:
            return self._base_obs_sigma

        raw_w = opponent_quality / self._league_avg
        w = float(np.clip(raw_w, self._w_min, self._w_max))
        return self._base_obs_sigma / np.sqrt(w)

    # ------------------------------------------------------------------
    # Add weights to a game log
    # ------------------------------------------------------------------

    def add_weights(
        self,
        game_df: pl.DataFrame,
        season: int,
        opponent_col: str = "opponent_id",
    ) -> pl.DataFrame:
        """Attach SoS weights to a per-player game log.

        Args:
            game_df:      One row per (player, game).  Must have
                          player_id, game_id.  Optional: opponent_col.
            season:       NHL season start-year.
            opponent_col: Column name for opponent identifier.

        Returns:
            game_df with additional columns:
                opponent_quality, sos_weight, effective_sigma.
        """
        required = {"player_id", "game_id"}
        missing = required - set(game_df.columns)
        if missing:
            raise ValueError(f"game_df missing columns: {sorted(missing)}")

        has_opponent = opponent_col in game_df.columns

        # Resolve quality per row
        opponent_qualities: list[float] = []
        opponent_ids: list[int] = []

        for row in game_df.to_dicts():
            opp_id = int(row.get(opponent_col) or 0) if has_opponent else 0
            opp_q  = self._opponent_quality.get(opp_id)
            if opp_q is None:
                opp_q = float("nan")
            opponent_qualities.append(opp_q)
            opponent_ids.append(opp_id)

        opp_q_arr  = np.array(opponent_qualities)
        weights, eff_sigmas = compute_sos_weights(
            opp_q_arr, self._base_obs_sigma, self._w_min, self._w_max
        )

        return game_df.with_columns([
            pl.Series("opponent_id",      opponent_ids,            dtype=pl.Int64),
            pl.Series("opponent_quality", opp_q_arr.tolist(),      dtype=pl.Float64),
            pl.Series("sos_weight",       weights.tolist(),        dtype=pl.Float64),
            pl.Series("effective_sigma",  eff_sigmas.tolist(),     dtype=pl.Float64),
        ])

    def to_parquet_df(
        self,
        game_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Return SoS weight DataFrame matching SOS_WEIGHT_SCHEMA.

        Args:
            game_df: Output of add_weights().
            season:  NHL season start-year.
        """
        weighted = self.add_weights(game_df, season)
        base = weighted.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit(self._version).alias("model_version"),
        ])
        for col, dtype in SOS_WEIGHT_SCHEMA.items():
            if col not in base.columns:
                base = base.with_columns(pl.lit(None).cast(dtype).alias(col))
        return base.select(list(SOS_WEIGHT_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Season-level SoS summary
    # ------------------------------------------------------------------

    def season_sos_summary(
        self,
        game_df: pl.DataFrame,
        season: int,
        opponent_col: str = "opponent_id",
    ) -> pl.DataFrame:
        """Compute per-player season-level SoS summary statistics.

        Returns DataFrame with player_id, n_games, mean_sos_weight,
        min_sos_weight, max_sos_weight — useful for auditing schedule strength.
        """
        weighted = self.add_weights(game_df, season, opponent_col)
        return (
            weighted
            .group_by("player_id")
            .agg([
                pl.col("sos_weight").count().alias("n_games"),
                pl.col("sos_weight").mean().alias("mean_sos_weight"),
                pl.col("sos_weight").min().alias("min_sos_weight"),
                pl.col("sos_weight").max().alias("max_sos_weight"),
            ])
            .sort("mean_sos_weight", descending=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "quality_col":       self._quality_col,
            "w_min":             self._w_min,
            "w_max":             self._w_max,
            "base_obs_sigma":    self._base_obs_sigma,
            "opponent_quality":  self._opponent_quality,
            "league_avg":        self._league_avg,
            "version":           self._version,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "StrengthOfScheduleNormalizer":
        import joblib
        payload = joblib.load(Path(path))
        obj = cls(
            quality_col    = payload.get("quality_col",    SOS_QUALITY_COL),
            w_min          = payload.get("w_min",          SOS_W_MIN),
            w_max          = payload.get("w_max",          SOS_W_MAX),
            base_obs_sigma = payload.get("base_obs_sigma", OBS_SIGMA_XGF60),
        )
        obj._opponent_quality = payload.get("opponent_quality", {})
        obj._league_avg       = payload.get("league_avg",       1.0)
        obj._version          = payload.get("version",          MODEL_VERSION)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_sos_weights(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write SoS weight table to ``output_dir/sos_weights_{season}.parquet``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sos_weights_{season}.parquet"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    for col, dtype in SOS_WEIGHT_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(SOS_WEIGHT_SCHEMA.keys())).write_parquet(path)
    return path
