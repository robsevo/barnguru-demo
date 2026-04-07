"""Goalie Fatigue Sub-model — Feature 2.6.

Linear regression model that estimates save% degradation due to fatigue.
Coefficients are intentionally auditable (linear model, named features).

Inputs (per-goalie, per-game):
    is_b2b          0/1 — back-to-back game
    rest_days       days since last game (clipped 0–7)
    gp_last_7       games played in last 7 days
    gp_last_14      games played in last 14 days
    road_game_num   consecutive road game number (0 if home game)
    saves_last_5    total saves in last 5 appearances (workload)
    age             player age in years
    toi_last_5_avg  average TOI (seconds) in last 5 games

Output:
    fatigue_sv_delta  — expected change in save% (typically −0.02 to 0.0)
                        negative = worse performance than rested baseline
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "goalie_fatigue_v1"

# Feature ordering — MUST stay in sync with _build_fatigue_features
FATIGUE_FEATURE_NAMES: list[str] = [
    "is_b2b",
    "rest_days",
    "gp_last_7",
    "gp_last_14",
    "road_game_num",
    "saves_last_5",
    "age",
    "toi_last_5_avg",
]

# Interpretable priors for default coefficients (hand-tuned, Bob-auditable)
# Unit: save% change per unit of each feature
DEFAULT_COEFFICIENTS: dict[str, float] = {
    "is_b2b":         -0.0080,   # B2B game: −0.8% save%
    "rest_days":      +0.0012,   # each rest day: +0.12%  (recovery)
    "gp_last_7":      -0.0035,   # each extra game in 7 days: −0.35%
    "gp_last_14":     -0.0018,   # each extra game in 14 days: −0.18%
    "road_game_num":  -0.0020,   # each consecutive road game: −0.20%
    "saves_last_5":   -0.000030, # each save in last 5: −0.003% (volume toll)
    "age":            -0.00050,  # each year over 30 adds extra drag (centered)
    "toi_last_5_avg": -0.000010, # each extra second TOI/game: −0.001%
}

# Ridge regularisation — keeps coefficients stable on small samples
DEFAULT_ALPHA = 1.0

# Minimum qualifying appearances to include a goalie in a training sample
MIN_APPEARANCES = 5


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

GOALIE_FATIGUE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":       pl.Int64,
    "player_name":     pl.Utf8,
    "season":          pl.Int64,
    "game_id":         pl.Int64,
    "is_b2b":          pl.Int8,
    "rest_days":       pl.Float64,
    "gp_last_7":       pl.Int64,
    "road_game_num":   pl.Int64,
    "age":             pl.Float64,
    "fatigue_sv_delta": pl.Float64,
    "model_version":   pl.Utf8,
}


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def build_fatigue_features(
    schedule_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build feature matrix from a per-goalie game-log DataFrame.

    Args:
        schedule_df: One row per (goalie, game). Required columns:
            player_id, game_id, game_date (date or str), is_b2b, rest_days,
            gp_last_7, gp_last_14, road_game_num, saves_last_5, age,
            toi_last_5_avg.
            Optional: sv_delta (float) — observed save% vs. season baseline.

    Returns:
        (X float32 of shape (n, 8), y float32 or None)
    """
    required = {
        "is_b2b", "rest_days", "gp_last_7", "gp_last_14",
        "road_game_num", "saves_last_5", "age", "toi_last_5_avg",
    }
    missing = required - set(schedule_df.columns)
    if missing:
        raise ValueError(f"schedule_df missing columns: {sorted(missing)}")

    df = schedule_df

    X_raw = (
        df.select(FATIGUE_FEATURE_NAMES)
        .to_numpy()
        .astype(np.float64)
    )
    # Clip / clean
    X_raw[:, 0] = _clip(X_raw[:, 0], 0, 1)          # is_b2b
    X_raw[:, 1] = _clip(X_raw[:, 1], 0, 7)           # rest_days
    X_raw[:, 2] = _clip(X_raw[:, 2], 0, 7)           # gp_last_7
    X_raw[:, 3] = _clip(X_raw[:, 3], 0, 14)          # gp_last_14
    X_raw[:, 4] = _clip(X_raw[:, 4], 0, 14)          # road_game_num
    X_raw[:, 5] = _clip(X_raw[:, 5], 0, 250)         # saves_last_5
    X_raw[:, 6] = _clip(X_raw[:, 6], 18, 45)         # age
    X_raw[:, 7] = _clip(X_raw[:, 7], 0, 5400)        # toi_last_5_avg (seconds)

    X = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)

    y: np.ndarray | None = None
    if "sv_delta" in df.columns:
        y_raw = df["sv_delta"].cast(pl.Float64).to_numpy().astype(np.float32)
        valid = ~np.isnan(y_raw)
        X = X[valid]
        y = y_raw[valid]

    return X, y


# ---------------------------------------------------------------------------
# GoalieFatigueModel
# ---------------------------------------------------------------------------

class GoalieFatigueModel:
    """Linear ridge regression goalie fatigue model (Feature 2.6).

    Predicts expected save% delta (negative = degraded) due to fatigue
    signals.  Linear model = fully auditable coefficients.

    Default coefficients are hand-calibrated estimates. Call ``fit()`` to
    learn data-driven coefficients from a historical game log.

    Usage::

        model = GoalieFatigueModel()
        # Use defaults (auditable priors):
        delta = model.predict(X)

        # Or fit from data:
        metrics = model.fit(schedule_df)
        delta = model.predict(X)
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self._alpha = alpha
        self._model: Ridge | None = None
        self._version = MODEL_VERSION
        self._fitted = False
        # Always initialise default coefficient vector
        self._default_coef = np.array(
            [DEFAULT_COEFFICIENTS[f] for f in FATIGUE_FEATURE_NAMES],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Coefficient access (auditable)
    # ------------------------------------------------------------------

    def coefficients(self) -> dict[str, float]:
        """Return current model coefficients (fitted or default).

        These are the auditable numbers Bob can inspect. Negative coefficient
        on ``is_b2b`` means B2B games reduce expected save%.
        """
        if self._fitted and self._model is not None:
            coef = self._model.coef_
        else:
            coef = self._default_coef
        return {name: float(v) for name, v in zip(FATIGUE_FEATURE_NAMES, coef)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        schedule_df: pl.DataFrame,
    ) -> dict[str, float]:
        """Fit model from a per-goalie game-log DataFrame.

        Args:
            schedule_df: must contain FATIGUE_FEATURE_NAMES + ``sv_delta``.

        Returns:
            {"rmse": float, "n_games": int, "r2": float}

        Raises:
            ValueError: if no valid training rows.
        """
        X, y = build_fatigue_features(schedule_df)

        if y is None or len(y) == 0:
            raise ValueError(
                "schedule_df must contain 'sv_delta' column with observed "
                "save% deltas to fit the model."
            )

        if len(y) < MIN_APPEARANCES:
            warnings.warn(
                f"Only {len(y)} rows in training data. "
                f"Default coefficients may be more reliable.",
                DataMissingWarning,
                stacklevel=2,
            )

        self._model = Ridge(alpha=self._alpha, fit_intercept=True)
        self._model.fit(X, y)
        self._fitted = True

        preds = self._model.predict(X)
        residuals = preds - y
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return {"rmse": rmse, "r2": r2, "n_games": int(len(y))}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return fatigue_sv_delta array for input feature matrix X.

        Falls back to default coefficients if model not fitted.

        Args:
            X: float array of shape (n, 8) matching FATIGUE_FEATURE_NAMES order.

        Returns:
            float64 array of shape (n,), values typically in [−0.05, 0.01].
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(FATIGUE_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FATIGUE_FEATURE_NAMES)} features, got {X.shape[1]}"
            )

        if self._fitted and self._model is not None:
            return self._model.predict(X).astype(np.float64)

        # Default coefficients + intercept = 0
        return (X.astype(np.float64) @ self._default_coef)

    def predict_season(
        self,
        schedule_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Compute fatigue deltas for all goalie-game rows in schedule_df.

        Returns DataFrame matching (a subset of) GOALIE_FATIGUE_SCHEMA.
        """
        required_out = {"player_id", "game_id"}
        if not required_out.issubset(schedule_df.columns):
            missing = required_out - set(schedule_df.columns)
            raise ValueError(f"schedule_df missing: {sorted(missing)}")

        X, _ = build_fatigue_features(schedule_df)
        deltas = self.predict(X)

        player_name_col = (
            schedule_df["player_name"]
            if "player_name" in schedule_df.columns
            else pl.Series("player_name", [""] * len(schedule_df), dtype=pl.Utf8)
        )
        age_col = (
            schedule_df["age"].cast(pl.Float64)
            if "age" in schedule_df.columns
            else pl.Series("age", [0.0] * len(schedule_df), dtype=pl.Float64)
        )

        result = pl.DataFrame({
            "player_id":        schedule_df["player_id"].cast(pl.Int64),
            "player_name":      player_name_col,
            "season":           pl.Series("season", [season] * len(schedule_df), dtype=pl.Int64),
            "game_id":          schedule_df["game_id"].cast(pl.Int64),
            "is_b2b":           schedule_df["is_b2b"].cast(pl.Int8),
            "rest_days":        schedule_df["rest_days"].cast(pl.Float64),
            "gp_last_7":        schedule_df["gp_last_7"].cast(pl.Int64),
            "road_game_num":    schedule_df["road_game_num"].cast(pl.Int64),
            "age":              age_col,
            "fatigue_sv_delta": pl.Series("fatigue_sv_delta", deltas, dtype=pl.Float64),
            "model_version":    pl.Series("model_version", [self._version] * len(schedule_df), dtype=pl.Utf8),
        })
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model to path (joblib)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":          self._model,
            "version":        self._version,
            "alpha":          self._alpha,
            "fitted":         self._fitted,
            "default_coef":   self._default_coef,
            "feature_names":  FATIGUE_FEATURE_NAMES,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "GoalieFatigueModel":
        """Load model from path."""
        payload = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._model        = payload["model"]
        obj._version      = payload.get("version", MODEL_VERSION)
        obj._alpha        = payload.get("alpha", DEFAULT_ALPHA)
        obj._fitted       = payload.get("fitted", False)
        obj._default_coef = payload.get("default_coef", np.array(
            [DEFAULT_COEFFICIENTS[f] for f in FATIGUE_FEATURE_NAMES]
        ))
        return obj


# ---------------------------------------------------------------------------
# Convenience: apply fatigue delta to goalie ratings
# ---------------------------------------------------------------------------

def apply_fatigue_to_ratings(
    ratings_df: pl.DataFrame,
    fatigue_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join fatigue deltas onto a goalie ratings DataFrame.

    Args:
        ratings_df: output of GoalieModel.predict_season() — must have player_id.
        fatigue_df: output of GoalieFatigueModel.predict_season() — must have
                    player_id and fatigue_sv_delta.

    Returns:
        ratings_df with additional ``adjusted_gsax_per60`` column
        (model_gsax_per60 adjusted by fatigue_sv_delta * 60).
    """
    if "player_id" not in ratings_df.columns or "player_id" not in fatigue_df.columns:
        raise ValueError("Both DataFrames must have 'player_id' column.")

    # Aggregate fatigue deltas to season-level (mean across games)
    fatigue_agg = (
        fatigue_df
        .group_by("player_id")
        .agg(pl.col("fatigue_sv_delta").mean().alias("mean_fatigue_sv_delta"))
    )

    result = ratings_df.join(fatigue_agg, on="player_id", how="left")

    # Save% delta × ~30 shots/game × 60min → approximate GSAx/60 effect
    # Conservative: 1 save%pt × 30 shots/game = 0.3 goals/game = 0.3 GSAx/game
    # Per 60 min ≈ game, so multiply by 30
    SHOTS_PER_GAME_PROXY = 30.0
    result = result.with_columns([
        (
            pl.col("model_gsax_per60")
            + pl.col("mean_fatigue_sv_delta").fill_null(0.0) * SHOTS_PER_GAME_PROXY
        ).alias("adjusted_gsax_per60")
    ])

    return result.drop("mean_fatigue_sv_delta")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_goalie_fatigue(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write fatigue predictions to ``output_dir/goalie_fatigue_{season}.parquet``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"goalie_fatigue_{season}.parquet"
    df.write_parquet(path)
    return path


def read_goalie_fatigue(data_dir: Path, season: int) -> pl.DataFrame:
    """Read fatigue predictions from ``data_dir/goalie_fatigue/goalie_fatigue_{season}.parquet``."""
    path = Path(data_dir) / "goalie_fatigue" / f"goalie_fatigue_{season}.parquet"
    return pl.read_parquet(path)
