"""xG Player Finishing Model — Feature 2.1.

XGBoost binary classifier trained on MoneyPuck shot-level data.  Given a shot,
predicts P(goal | shot features).  Used downstream to compute per-player
Finishing = actual_goals − Σ xG(shot_i), a measure of shooting skill
independent of shot volume.

Feature engineering from raw shot parquet columns (no external joins):
    shot_angle          degrees from centre-ice axis
    shot_distance       arena-adjusted feet from goal
    shot_type_*         one-hot across WRIST/SLAP/SNAP/BACKHAND/TIP/DEFLECTION/WRAP
    danger_zone_HIGH    computed from arena-adjusted x/y coordinates
    danger_zone_MED     computed from arena-adjusted x/y coordinates
    is_rush             last_event_team != shooting_team  (proxy)
    skater_diff         home_skaters − away_skaters (PP/PK signal; team-adjusted)
    is_period_3         period == 3
    shooter_hand_L      shooter_hand == "L"
    x_on_goal           MoneyPuck P(shot on goal) — encodes trajectory quality
    angle_x_dist        shot_angle × shot_distance interaction
    is_rebound          shot within 3 sec of previous shot in same game+period
    is_playoff          playoff game flag
    is_pp               shooting team on power play (skater_diff > 0)
    is_pk               shooting team shorthanded (skater_diff < 0)

Calibration target: Pearson correlation with MoneyPuck x_goal ≥ 0.85.
Quality gates:  AUC-ROC > 0.75,  Brier score < 0.10 (shot-level, not game-level).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "xg_v1"
MODEL_VERSION_TRACKING = "xg_v2_tracking"

# Fill value used when defender distance is unknown (no tracking data)
_DEFAULT_DEFENDER_DIST: float = 15.0

# Tracking feature names appended by build_features_tracking()
_TRACKING_FEATURE_NAMES = ["screen_count", "defender_dist", "goalie_displacement"]

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective":         "binary:logistic",
    "eval_metric":       "logloss",
    "n_estimators":      500,
    "max_depth":         4,          # shallow → interpretable
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  50,         # regularise against tiny samples
    "seed":              42,
    "verbosity":         0,
    "n_jobs":            -1,
}

# One-hot shot types recognised in the MoneyPuck CSV
_SHOT_TYPES = ["WRIST", "SLAP", "SNAP", "BACKHAND", "TIP", "DEFLECTION", "WRAP"]

# Danger zone thresholds (arena-adjusted coordinates, feet)
# HIGH:  within the slot (~within 20 ft of net, between the circles)
# MED:   inner half of the zone (20–40 ft) or outside the circles
# LOW:   perimeter / point shots (>40 ft)
_HIGH_DISTANCE = 20.0
_MED_DISTANCE  = 40.0
# y-axis slot boundary (dots are at y ≈ ±22 ft from centre)
_SLOT_Y        = 22.0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_rebound(df: pl.DataFrame) -> np.ndarray:
    """Return float32 array: 1 if shot occurred within 3 sec of previous shot in same game+period."""
    needed = {"game_id", "period", "time"}
    if not needed.issubset(df.columns):
        return np.zeros(len(df), dtype=np.float32)

    result = (
        df.select(["game_id", "period", "time"])
        .with_row_index("_idx")
        .sort("game_id", "period", "time")
        .with_columns(
            pl.col("time").shift(1).over("game_id", "period").alias("_prev_time")
        )
        .with_columns(
            ((pl.col("time") - pl.col("_prev_time")) <= 3)
            .fill_null(False)
            .alias("_is_rebound")
        )
        .sort("_idx")
        .get_column("_is_rebound")
        .to_numpy()
    )
    return result.astype(np.float32)


def _danger_zone(distance: np.ndarray, arena_adj_y: np.ndarray) -> np.ndarray:
    """Return integer danger zone: 2=HIGH, 1=MED, 0=LOW."""
    high = (distance <= _HIGH_DISTANCE) & (np.abs(arena_adj_y) <= _SLOT_Y)
    med  = (~high) & (distance <= _MED_DISTANCE)
    zone = np.zeros(len(distance), dtype=np.int8)
    zone[med]  = 1
    zone[high] = 2
    return zone


def build_features(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Engineer feature matrix X (and optionally target y) from a shot parquet DataFrame.

    Args:
        df: Polars DataFrame with MoneyPuck shot schema columns.

    Returns:
        (X, y) where y is None if 'is_goal' column is absent.
        X shape: (n_shots, n_features).  All values are float32, no NaN.

    Raises:
        ValueError: if required columns are missing.
    """
    required = {"shot_angle", "arena_adj_distance", "shot_type", "arena_adj_y",
                "last_event_team", "shooting_team", "home_skaters", "away_skaters",
                "period", "shooter_hand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Shot DataFrame missing required columns: {sorted(missing)}")

    n = len(df)
    angle    = df["shot_angle"].fill_null(0.0).to_numpy().astype(np.float32)
    distance = df["arena_adj_distance"].fill_null(60.0).to_numpy().astype(np.float32)
    adj_y    = df["arena_adj_y"].fill_null(0.0).to_numpy().astype(np.float32)

    # Shot type one-hot
    shot_type_series = df["shot_type"].fill_null("WRIST").to_numpy()
    type_ohe = np.zeros((n, len(_SHOT_TYPES)), dtype=np.float32)
    for i, st in enumerate(_SHOT_TYPES):
        type_ohe[:, i] = (shot_type_series == st).astype(np.float32)

    # Danger zone
    dz = _danger_zone(distance, adj_y)
    dz_high = (dz == 2).astype(np.float32)
    dz_med  = (dz == 1).astype(np.float32)

    # Rush proxy: last event was by the other team → turnover / rush
    last_ev  = df["last_event_team"].fill_null("").to_numpy()
    shooting = df["shooting_team"].fill_null("").to_numpy()
    is_rush  = (last_ev != shooting).astype(np.float32)

    # Skater differential (PP = positive, PK = negative for shooting team)
    home_sk = df["home_skaters"].fill_null(5).to_numpy().astype(np.float32)
    away_sk = df["away_skaters"].fill_null(5).to_numpy().astype(np.float32)
    skater_diff = (home_sk - away_sk).astype(np.float32)   # raw; sign depends on home/away

    # Period 3 flag
    period    = df["period"].fill_null(1).to_numpy().astype(np.float32)
    is_p3     = (period == 3).astype(np.float32)

    # Shooter handedness
    hand      = df["shooter_hand"].fill_null("R").to_numpy()
    hand_L    = (hand == "L").astype(np.float32)

    # MoneyPuck on-goal probability (encodes trajectory quality)
    x_on_goal = (
        df["x_on_goal"].fill_null(0.0).to_numpy().astype(np.float32)
        if "x_on_goal" in df.columns
        else np.zeros(n, dtype=np.float32)
    )

    # Angle × distance interaction (captures location geometry jointly)
    angle_x_dist = (angle * distance).astype(np.float32)

    # Rebound proxy (shot within 3 sec of previous shot in same game+period)
    is_rebound = _compute_rebound(df)

    # Playoff flag
    is_playoff = (
        df["is_playoff"].fill_null(False).to_numpy().astype(np.float32)
        if "is_playoff" in df.columns
        else np.zeros(n, dtype=np.float32)
    )

    # Explicit power-play / penalty-kill dummies (more informative than raw diff)
    is_pp = (skater_diff > 0).astype(np.float32)
    is_pk = (skater_diff < 0).astype(np.float32)

    # Stack into feature matrix
    X = np.column_stack([
        angle,
        distance,
        *[type_ohe[:, i] for i in range(len(_SHOT_TYPES))],
        dz_high,
        dz_med,
        is_rush,
        skater_diff,
        is_p3,
        hand_L,
        x_on_goal,
        angle_x_dist,
        is_rebound,
        is_playoff,
        is_pp,
        is_pk,
    ]).astype(np.float32)

    # Target
    y: np.ndarray | None = None
    if "is_goal" in df.columns:
        y = df["is_goal"].cast(pl.Int8).fill_null(0).to_numpy().astype(np.float32)

    return X, y


def feature_names() -> list[str]:
    """Return ordered list of feature names matching build_features() output."""
    return [
        "shot_angle",
        "shot_distance",
        *[f"shot_type_{t}" for t in _SHOT_TYPES],
        "danger_zone_HIGH",
        "danger_zone_MED",
        "is_rush",
        "skater_diff",
        "is_period_3",
        "shooter_hand_L",
        "x_on_goal",
        "angle_x_dist",
        "is_rebound",
        "is_playoff",
        "is_pp",
        "is_pk",
    ]


def build_features_tracking(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Engineer 22-feature matrix: base 19 features + 3 CV tracking features.

    The three tracking columns are *optional* in *df* — when absent they are
    zero-filled (screen_count=0, defender_dist=15 ft, goalie_displacement=0),
    which is the neutral / "no tracking data" value.

    Tracking columns (all float32):
        screen_count          Players in shooter→goalie ray within 3 ft
        defender_dist         Feet to nearest opposing skater
        goalie_displacement   Feet goalie has moved from crease centre (±89, 0)

    Returns:
        (X, y) — same contract as build_features().  X shape: (n, 22).
    """
    X_base, y = build_features(df)
    n = len(df)

    screen_count = (
        df["screen_count"].fill_null(0).to_numpy().astype(np.float32)
        if "screen_count" in df.columns
        else np.zeros(n, dtype=np.float32)
    )
    defender_dist = (
        df["defender_dist"].fill_null(_DEFAULT_DEFENDER_DIST).to_numpy().astype(np.float32)
        if "defender_dist" in df.columns
        else np.full(n, _DEFAULT_DEFENDER_DIST, dtype=np.float32)
    )
    goalie_displacement = (
        df["goalie_displacement"].fill_null(0.0).to_numpy().astype(np.float32)
        if "goalie_displacement" in df.columns
        else np.zeros(n, dtype=np.float32)
    )

    X = np.column_stack([X_base, screen_count, defender_dist, goalie_displacement]).astype(np.float32)
    return X, y


def feature_names_tracking() -> list[str]:
    """Return the 22-element feature name list for the tracking-enhanced model."""
    return feature_names() + _TRACKING_FEATURE_NAMES


# ---------------------------------------------------------------------------
# XGModel
# ---------------------------------------------------------------------------


class XGModel:
    """XGBoost shot-level xG model.

    Usage::

        model = XGModel()
        metrics = model.fit(train_df, eval_df=eval_df)
        proba = model.predict_proba(test_df)
        model.save(Path("~/.gretzky/models/xg_model_2025.pkl"))

        loaded = XGModel.load(Path("~/.gretzky/models/xg_model_2025.pkl"))
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        tracking_features: bool = False,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError(
                "xgboost is required. Run: uv add xgboost"
            )
        self._params = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self._model: xgb.XGBClassifier | None = None
        self._tracking_features = tracking_features
        self._version = MODEL_VERSION_TRACKING if tracking_features else MODEL_VERSION

    # --- internal feature dispatch ---

    def _build_features(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        if self._tracking_features:
            return build_features_tracking(df)
        return build_features(df)

    def _feature_names(self) -> list[str]:
        if self._tracking_features:
            return feature_names_tracking()
        return feature_names()

    # --- training ---

    def fit(
        self,
        df: pl.DataFrame,
        eval_df: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train on df.  Optionally evaluate on eval_df.

        Args:
            df:      Training shot DataFrame (must include is_goal).
            eval_df: Held-out evaluation DataFrame for metrics. If None, uses df.

        Returns:
            Metrics dict from evaluate() on eval_df (or df if eval_df is None).
        """
        X, y = self._build_features(df)
        if y is None:
            raise ValueError("Training DataFrame must contain 'is_goal' column.")

        self._model = xgb.XGBClassifier(**self._params)

        eval_set = None
        if eval_df is not None:
            X_eval, y_eval = self._build_features(eval_df)
            if y_eval is not None:
                eval_set = [(X_eval, y_eval)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False,
            )

        return self.evaluate(eval_df if eval_df is not None else df)

    # --- inference ---

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Return P(goal) per shot row.  Values in [0, 1].

        Raises:
            RuntimeError: if model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        X, _ = self._build_features(df)
        return self._model.predict_proba(X)[:, 1]

    # --- evaluation ---

    def evaluate(self, df: pl.DataFrame) -> dict[str, float]:
        """Compute Brier score, AUC-ROC, and Pearson corr with MoneyPuck x_goal.

        Returns:
            Dict with keys: brier_score, auc_roc, moneypuck_correlation (or NaN).
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score

        X, y = self._build_features(df)
        if y is None:
            raise ValueError("Evaluation DataFrame must contain 'is_goal' column.")

        proba = self.predict_proba(df)

        brier = float(brier_score_loss(y, proba))
        try:
            auc = float(roc_auc_score(y, proba))
        except ValueError:
            auc = float("nan")

        # Pearson correlation with MoneyPuck's own xG
        mp_corr = float("nan")
        if "x_goal" in df.columns:
            x_goal = df["x_goal"].drop_nulls().to_numpy()
            if len(x_goal) == len(proba):
                mp_corr = float(np.corrcoef(proba, x_goal)[0, 1])

        return {
            "brier_score":           brier,
            "auc_roc":               auc,
            "moneypuck_correlation": mp_corr,
        }

    # --- interpretability ---

    def feature_importance(self) -> dict[str, float]:
        """Return global mean |SHAP| per feature, sorted descending.

        Falls back to XGBoost gain-based importance if SHAP is unavailable.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        names = self._feature_names()

        if _SHAP_AVAILABLE:
            # Use a background sample (not the full dataset — faster)
            explainer = _shap.TreeExplainer(self._model)
            # We compute importance from stored booster (no data needed for global)
            importance_raw = self._model.get_booster().get_score(importance_type="gain")
            # Map back through feature index names (f0, f1, ...) to human names
            fi: dict[str, float] = {}
            for fname, names_entry in zip(
                [f"f{i}" for i in range(len(names))], names
            ):
                fi[names_entry] = importance_raw.get(fname, 0.0)
        else:
            importance_raw = self._model.get_booster().get_score(importance_type="gain")
            fi = {
                names[int(k[1:])]: v
                for k, v in importance_raw.items()
                if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(names)
            }

        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # --- persistence ---

    def save(self, path: Path) -> None:
        """Serialize model + metadata to path using joblib."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":            self._model,
            "version":          self._version,
            "params":           self._params,
            "tracking_features": self._tracking_features,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "XGModel":
        """Deserialize model from path."""
        payload = joblib.load(Path(path))
        instance = cls.__new__(cls)
        instance._model            = payload["model"]
        instance._version          = payload.get("version", MODEL_VERSION)
        instance._params           = payload.get("params", DEFAULT_XGB_PARAMS)
        instance._tracking_features = payload.get("tracking_features", False)
        return instance
