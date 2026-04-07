"""Goalie Model — Feature 2.5.

XGBoost + SHAP model that rates goalie quality from shot-quality mix and
observed save performance.  Smooths out small-sample noise and produces an
interpretable per-goalie GSAx/60 rating.

Outputs
-------
``goalie_ratings_{season}.parquet`` — one row per goalie:
    model_gsax_per60  XGBoost-smoothed goals-saved-above-expected per 60 min
    gsax_delta        model_gsax_per60 − 0.0  (excess above replacement)
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

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION   = "goalie_v1"
MIN_GAMES       = 10      # minimum GP to include a goalie-season
SHAP_SAMPLE_SIZE = 500    # max rows for SHAP computation

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "n_estimators":     300,
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed":             42,
    "verbosity":        0,
    "n_jobs":           -1,
}

_FEATURE_NAMES = [
    "hd_shot_pct",    # hd_shots / shots — danger zone mix
    "md_shot_pct",    # md_shots / shots
    "xga_per60",      # xga / (icetime_seconds / 3600)
    "shots_per60",    # shots / (icetime_seconds / 3600)
    "hdsv_pct",       # high-danger save %
    "mdsv_pct",       # medium-danger save %
    "ldsv_pct",       # low-danger save %
    "sv_pct",         # overall save %
    "log1p_gp",       # log1p(games_played) — sample size
    "sv_minus_xsv",   # sv_pct - (1 - xga/shots) — skill above shot quality
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

GOALIE_RATING_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "player_name":      pl.Utf8,
    "team":             pl.Utf8,
    "season":           pl.Int64,
    "games_played":     pl.Int64,
    "icetime_seconds":  pl.Float64,
    "sv_pct":           pl.Float64,
    "xga":              pl.Float64,
    "gsax":             pl.Float64,
    "gsax_per60":       pl.Float64,       # raw observed
    "model_gsax_per60": pl.Float64,       # XGBoost-smoothed rating
    "gsax_delta":       pl.Float64,       # model_gsax_per60 − 0.0
    "model_version":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_goalie_features(
    goalie_df: pl.DataFrame,
    season: int,
) -> tuple[np.ndarray, np.ndarray | None, pl.DataFrame]:
    """Build XGBoost feature matrix from goalie summary data.

    Args:
        goalie_df: MoneyPuck goalie parquet (GOALIE_SCHEMA).
        season:    Season integer for filtering.

    Returns:
        (X float32, y float32 or None, meta DataFrame)
        meta has: player_id, player_name, team, games_played, icetime_seconds,
                  sv_pct, xga, gsax, gsax_per60
    """
    required = {"player_id", "player_name", "team", "situation", "games_played",
                "icetime_seconds", "shots", "saves", "xga", "gsax",
                "hd_shots", "md_shots", "hdsv_pct", "mdsv_pct", "ldsv_pct",
                "sv_pct"}
    missing = required - set(goalie_df.columns)
    if missing:
        raise ValueError(f"goalie_df missing columns: {sorted(missing)}")

    df = goalie_df.filter(pl.col("situation") == "all")

    # Season filter
    if "season" in df.columns:
        df = df.filter(pl.col("season") == season)

    df = df.filter(pl.col("games_played") >= MIN_GAMES)

    if df.is_empty():
        warnings.warn(
            f"Season {season}: no goalie rows with games_played >= {MIN_GAMES}.",
            DataMissingWarning,
            stacklevel=3,
        )
        meta = pl.DataFrame(schema={
            "player_id": pl.Int64, "player_name": pl.Utf8, "team": pl.Utf8,
            "games_played": pl.Int64, "icetime_seconds": pl.Float64,
            "sv_pct": pl.Float64, "xga": pl.Float64, "gsax": pl.Float64,
            "gsax_per60": pl.Float64,
        })
        return np.zeros((0, len(_FEATURE_NAMES)), dtype=np.float32), None, meta

    toi60 = (df["icetime_seconds"].cast(pl.Float64) / 3600.0).clip(lower_bound=1e-9)
    shots_safe = df["shots"].cast(pl.Float64).clip(lower_bound=1)

    df = df.with_columns([
        (pl.col("hd_shots").cast(pl.Float64) / shots_safe).alias("hd_shot_pct"),
        (pl.col("md_shots").cast(pl.Float64) / shots_safe).alias("md_shot_pct"),
        pl.Series("xga_per60",   (df["xga"].cast(pl.Float64).to_numpy() / toi60.to_numpy()), dtype=pl.Float64),
        pl.Series("shots_per60", (df["shots"].cast(pl.Float64).to_numpy() / toi60.to_numpy()), dtype=pl.Float64),
        pl.Series("log1p_gp",    np.log1p(df["games_played"].cast(pl.Float64).to_numpy()), dtype=pl.Float64),
        pl.Series(
            "sv_minus_xsv",
            (
                df["sv_pct"].cast(pl.Float64).to_numpy()
                - (1.0 - df["xga"].cast(pl.Float64).to_numpy() / shots_safe.to_numpy())
            ),
            dtype=pl.Float64,
        ),
        pl.Series("gsax_per60", df["gsax"].cast(pl.Float64).to_numpy() / toi60.to_numpy(), dtype=pl.Float64),
    ])

    X = (
        df.select(_FEATURE_NAMES)
        .to_numpy()
        .astype(np.float32)
    )
    X = np.nan_to_num(X, nan=0.0)

    y = df["gsax_per60"].cast(pl.Float64).to_numpy().astype(np.float32)
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    df = df.filter(pl.col("gsax_per60").is_not_null())

    meta = df.select([
        "player_id", "player_name", "team", "games_played", "icetime_seconds",
        "sv_pct", "xga", "gsax", "gsax_per60",
    ])
    return X, y, meta


# ---------------------------------------------------------------------------
# GoalieModel
# ---------------------------------------------------------------------------

class GoalieModel:
    """XGBoost goalie quality model (Feature 2.5).

    Predicts GSAx/60 (goals saved above expected per 60 min) for each goalie,
    smoothing out small-sample noise using shot-quality mix and performance features.

    Usage::

        model = GoalieModel()
        metrics = model.fit([goalie_2023, goalie_2024], seasons=[2023, 2024])
        ratings = model.predict_season(goalie_2025, season=2025)
        model.save(Path("~/.gretzky/data/goalie_ratings/goalie_model.pkl"))
    """

    def __init__(self, xgb_params: dict[str, Any] | None = None) -> None:
        self._params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self._model: Any = None
        self._version = MODEL_VERSION

    def fit(
        self,
        goalie_dfs: list[pl.DataFrame],
        seasons: list[int],
    ) -> dict[str, float]:
        """Train on goalie summary data across multiple seasons.

        Args:
            goalie_dfs: One MoneyPuck goalie parquet DataFrame per season.
            seasons:    Season integers matching the above list.

        Returns:
            {"rmse": float, "n_goalies": int}

        Raises:
            ImportError: if xgboost is not installed.
            ValueError: if no training data.
        """
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for GoalieModel. Run: uv add xgboost")

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for goalie_df, season in zip(goalie_dfs, seasons):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_i, y_i, _ = _build_goalie_features(goalie_df, season)
            if y_i is not None and len(y_i) > 0:
                X_parts.append(X_i)
                y_parts.append(y_i)

        if not X_parts:
            raise ValueError("No training data produced — check goalie data.")

        X_train = np.vstack(X_parts)
        y_train  = np.concatenate(y_parts)

        self._model = xgb.XGBRegressor(**self._params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_train, y_train, verbose=False)

        preds = self._model.predict(X_train)
        rmse  = float(np.sqrt(np.mean((preds - y_train) ** 2)))
        return {"rmse": rmse, "n_goalies": len(X_train)}

    def predict_season(
        self,
        goalie_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Generate goalie ratings for one season.

        Returns:
            DataFrame matching GOALIE_RATING_SCHEMA, sorted descending by model_gsax_per60.

        Raises:
            RuntimeError: if model not trained.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, _, meta = _build_goalie_features(goalie_df, season)

        if meta.is_empty():
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in GOALIE_RATING_SCHEMA.items()}
            )

        model_preds = self._model.predict(X).astype(np.float64)

        result = meta.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.Series("model_gsax_per60", model_preds, dtype=pl.Float64),
            pl.Series("gsax_delta", model_preds - 0.0, dtype=pl.Float64),
            pl.lit(self._version).alias("model_version"),
        ])

        for col, dtype in GOALIE_RATING_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))

        return result.select(list(GOALIE_RATING_SCHEMA.keys())).sort(
            "model_gsax_per60", descending=True
        )

    def feature_importance(self) -> dict[str, float]:
        """Return gain-based feature importances, sorted descending."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        fi_arr = self._model.feature_importances_
        fi = {name: float(v) for name, v in zip(_FEATURE_NAMES, fi_arr)}
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    def feature_importance_from_data(self, X: np.ndarray) -> dict[str, float]:
        """Return SHAP mean |value| importances over X. Falls back to gain if shap unavailable."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        if _SHAP_AVAILABLE:
            sample = X[:SHAP_SAMPLE_SIZE]
            explainer = _shap.TreeExplainer(self._model)
            shap_vals = explainer.shap_values(sample)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            fi = {name: float(mean_abs[i]) for i, name in enumerate(_FEATURE_NAMES)}
            return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        return self.feature_importance()

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":         self._model,
            "version":       self._version,
            "params":        self._params,
            "feature_names": _FEATURE_NAMES,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "GoalieModel":
        payload = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._model   = payload["model"]
        obj._version = payload.get("version", MODEL_VERSION)
        obj._params  = payload.get("params", DEFAULT_XGB_PARAMS)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_goalie_ratings(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write goalie ratings parquet to ``output_dir/goalie_ratings_{season}.parquet``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"goalie_ratings_{season}.parquet"
    # Enforce schema
    for col, dtype in GOALIE_RATING_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(GOALIE_RATING_SCHEMA.keys())).write_parquet(path)
    return path


def read_goalie_ratings(data_dir: Path, season: int) -> pl.DataFrame:
    """Read goalie ratings from ``data_dir/goalie_ratings/goalie_ratings_{season}.parquet``."""
    path = Path(data_dir) / "goalie_ratings" / f"goalie_ratings_{season}.parquet"
    return pl.read_parquet(path)
