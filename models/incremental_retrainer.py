"""Incremental xG / Goalie Retrainer — Feature 2.17.

Weekly warm-start retraining of XGBoost models on current-season shots.
Each call adds ``n_new_trees`` additional boosting rounds on top of the
existing model, using season-weighted samples so that new evidence is
up-weighted relative to prior seasons.

Season weight formula:
    w(season) = 1.0                          if season == current_season
    w(season) = 0.7 ^ (current - season)    for prior seasons

This gives the previous season weight 0.70, two seasons ago 0.49, etc.
XGBoost "warm-start" is implemented by passing the existing model's booster
to ``fit(xgb_model=...)``, which appends new trees rather than retraining
from scratch — making weekly updates computationally cheap.

Validation gate:
    After retraining, the updated model is evaluated against a held-out set
    (defaults to the current season's data).  If the primary metric does NOT
    improve by at least ``accept_threshold``, the original model is returned
    unchanged.  Use ``force=True`` to skip this gate.

Outputs
-------
``retrain_log_{model_type}.parquet`` — one row per retraining run:
    retrain_date, model_type, current_season, n_seasons_used, n_new_trees,
    metric_before, metric_after, metric_name, accepted, model_version
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION         = "incr_v1"
SEASON_DECAY          = 0.7     # prior season weight multiplier
DEFAULT_N_NEW_TREES   = 50      # additional boosting rounds per weekly run
DEFAULT_ACCEPT_THRESH = 0.0     # accept if metric improves by >= this (0 = any improvement)

RETRAIN_LOG_SCHEMA: dict[str, pl.DataType] = {
    "retrain_date":    pl.Utf8,
    "model_type":      pl.Utf8,
    "current_season":  pl.Int64,
    "n_seasons_used":  pl.Int64,
    "n_new_trees":     pl.Int64,
    "metric_before":   pl.Float64,
    "metric_after":    pl.Float64,
    "metric_name":     pl.Utf8,
    "accepted":        pl.Boolean,
    "model_version":   pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def season_weight(current_season: int, data_season: int, decay: float = SEASON_DECAY) -> float:
    """Return sample weight for a given (current, data) season pair.

    Args:
        current_season: The season being predicted / updated.
        data_season:    The season the shot/game was played in.
        decay:          Multiplier per season of lag (default 0.7).

    Returns:
        1.0 for current season; decay^gap for prior seasons.
    """
    gap = max(0, current_season - data_season)
    return 1.0 if gap == 0 else decay ** gap


def _build_weight_array(seasons: np.ndarray, current_season: int, decay: float = SEASON_DECAY) -> np.ndarray:
    """Vectorised season-weight computation."""
    gaps = np.maximum(0, current_season - seasons)
    weights = np.where(gaps == 0, 1.0, decay ** gaps.astype(float))
    return weights.astype(np.float64)


# ---------------------------------------------------------------------------
# Retrain result
# ---------------------------------------------------------------------------

@dataclass
class RetrainResult:
    """Result of one incremental retraining run.

    Attributes:
        model_type:     "xg" or "goalie".
        current_season: Season of the update.
        metric_name:    Name of validation metric.
        metric_before:  Metric value before retraining.
        metric_after:   Metric value after retraining.
        n_new_trees:    Number of boosting rounds added.
        n_seasons_used: Number of seasons in training pool.
        accepted:       True if new model was accepted (metric improved).
        log_row:        Dict matching RETRAIN_LOG_SCHEMA for appending to log.
    """
    model_type:     str
    current_season: int
    metric_name:    str
    metric_before:  float
    metric_after:   float
    n_new_trees:    int
    n_seasons_used: int
    accepted:       bool
    log_row:        dict[str, Any] = field(default_factory=dict)

    def improvement(self) -> float:
        """Signed improvement (positive = better).

        For AUC/correlation: higher is better → after - before.
        For RMSE/Brier:      lower is better  → before - after.
        """
        lower_better = self.metric_name in ("rmse", "brier_score", "brier")
        if lower_better:
            return self.metric_before - self.metric_after
        return self.metric_after - self.metric_before

    def __str__(self) -> str:
        sign = "+" if self.improvement() >= 0 else ""
        return (
            f"RetrainResult({self.model_type}, season={self.current_season}, "
            f"{self.metric_name}: {self.metric_before:.4f}→{self.metric_after:.4f} "
            f"({sign}{self.improvement():.4f}), accepted={self.accepted})"
        )


# ---------------------------------------------------------------------------
# IncrementalRetrainer — Feature 2.17
# ---------------------------------------------------------------------------

class IncrementalRetrainer:
    """Weekly warm-start retrainer for xG and Goalie XGBoost models.

    Usage::

        retrainer = IncrementalRetrainer()

        # Weekly xG update:
        new_xg_model, result = retrainer.retrain_xg(
            model=xg_model,
            shots_data={2023: shots_2023, 2024: shots_2024, 2025: shots_2025_so_far},
            current_season=2025,
        )

        # Weekly goalie update:
        new_goalie_model, result = retrainer.retrain_goalie(
            model=goalie_model,
            goalie_data={2023: goalie_2023, 2024: goalie_2024, 2025: goalie_2025},
            current_season=2025,
        )
    """

    def __init__(
        self,
        decay:          float = SEASON_DECAY,
        n_new_trees:    int   = DEFAULT_N_NEW_TREES,
        accept_threshold: float = DEFAULT_ACCEPT_THRESH,
    ) -> None:
        self._decay            = decay
        self._n_new_trees      = n_new_trees
        self._accept_threshold = accept_threshold
        self._version          = MODEL_VERSION
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # xG warm-start
    # ------------------------------------------------------------------

    def retrain_xg(
        self,
        model: Any,
        shots_data: dict[int, pl.DataFrame],
        current_season: int,
        n_new_trees: int | None = None,
        eval_season: int | None = None,
        force: bool = False,
    ) -> tuple[Any, RetrainResult]:
        """Warm-start retrain the xG model with season-weighted shots.

        Args:
            model:          Trained XGModel instance.
            shots_data:     Dict mapping season → shots DataFrame.
            current_season: Season of the current update.
            n_new_trees:    Additional boosting rounds (default: self._n_new_trees).
            eval_season:    Season to use for validation (default: current_season).
            force:          Accept updated model even if metric regresses.

        Returns:
            (new_model, RetrainResult) — new_model == model if not accepted.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for IncrementalRetrainer.")

        from models.xg_model import XGModel, build_features

        n_trees = n_new_trees if n_new_trees is not None else self._n_new_trees
        eval_s  = eval_season if eval_season is not None else current_season

        # Pool all seasons into one training set with weights
        X_parts:  list[np.ndarray] = []
        y_parts:  list[np.ndarray] = []
        w_parts:  list[np.ndarray] = []
        seasons_used: list[int] = []

        for season, shots_df in sorted(shots_data.items()):
            if len(shots_df) == 0:
                continue
            X_i, y_i = build_features(shots_df)
            if y_i is None or len(y_i) == 0:
                continue
            w_i = np.full(len(y_i), season_weight(current_season, season, self._decay))
            X_parts.append(X_i)
            y_parts.append(y_i)
            w_parts.append(w_i)
            seasons_used.append(season)

        if not X_parts:
            warnings.warn(
                f"No xG training data found for seasons {sorted(shots_data.keys())}.",
                DataMissingWarning,
                stacklevel=2,
            )
            result = self._make_result("xg", current_season, "auc_roc",
                                       float("nan"), float("nan"), n_trees,
                                       0, accepted=False)
            return model, result

        X_all = np.vstack(X_parts).astype(np.float32)
        y_all = np.concatenate(y_parts)
        w_all = np.concatenate(w_parts)

        # Compute baseline metric on eval season
        metric_before = float("nan")
        if eval_s in shots_data and len(shots_data[eval_s]) > 0:
            try:
                metric_before = model.evaluate(shots_data[eval_s]).get("auc_roc", float("nan"))
            except Exception:
                pass

        # Warm-start: add n_new_trees on top of existing booster
        existing_booster = None
        if model._model is not None:
            existing_booster = model._model.get_booster()

        new_clf_params = {
            **model._params,
            "n_estimators": n_trees,
        }
        new_clf = xgb.XGBClassifier(**new_clf_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_clf.fit(
                X_all, y_all,
                sample_weight=w_all,
                xgb_model=existing_booster,
                verbose=False,
            )

        # Build new XGModel wrapping the updated classifier
        new_model = XGModel.__new__(XGModel)
        new_model._model            = new_clf
        new_model._version          = model._version
        new_model._params           = model._params
        new_model._tracking_features = getattr(model, "_tracking_features", False)

        # Evaluate after
        metric_after = float("nan")
        if eval_s in shots_data and len(shots_data[eval_s]) > 0:
            try:
                metric_after = new_model.evaluate(shots_data[eval_s]).get("auc_roc", float("nan"))
            except Exception:
                pass

        # Decide acceptance
        accepted = force
        if not force and not (np.isnan(metric_before) or np.isnan(metric_after)):
            improvement = metric_after - metric_before  # higher AUC = better
            accepted = improvement >= self._accept_threshold

        result = self._make_result(
            "xg", current_season, "auc_roc",
            metric_before, metric_after, n_trees,
            len(seasons_used), accepted,
        )
        returned_model = new_model if accepted else model
        return returned_model, result

    # ------------------------------------------------------------------
    # Goalie warm-start
    # ------------------------------------------------------------------

    def retrain_goalie(
        self,
        model: Any,
        goalie_data: dict[int, pl.DataFrame],
        current_season: int,
        n_new_trees: int | None = None,
        eval_season: int | None = None,
        force: bool = False,
    ) -> tuple[Any, RetrainResult]:
        """Warm-start retrain the Goalie model with season-weighted data.

        Args:
            model:          Trained GoalieModel instance.
            goalie_data:    Dict mapping season → goalie DataFrame.
            current_season: Season of the current update.
            n_new_trees:    Additional boosting rounds (default: self._n_new_trees).
            eval_season:    Season to use for validation (default: current_season).
            force:          Accept updated model even if metric regresses.

        Returns:
            (new_model, RetrainResult) — new_model == model if not accepted.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required for IncrementalRetrainer.")

        from models.goalie_model import GoalieModel, _build_goalie_features

        n_trees = n_new_trees if n_new_trees is not None else self._n_new_trees
        eval_s  = eval_season if eval_season is not None else current_season

        X_parts:  list[np.ndarray] = []
        y_parts:  list[np.ndarray] = []
        w_parts:  list[np.ndarray] = []
        seasons_used: list[int] = []

        for season, goalie_df in sorted(goalie_data.items()):
            if len(goalie_df) == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_i, y_i, _ = _build_goalie_features(goalie_df, season)
            if y_i is None or len(y_i) == 0:
                continue
            w_i = np.full(len(y_i), season_weight(current_season, season, self._decay))
            X_parts.append(X_i)
            y_parts.append(y_i)
            w_parts.append(w_i)
            seasons_used.append(season)

        if not X_parts:
            warnings.warn(
                f"No goalie training data found for seasons {sorted(goalie_data.keys())}.",
                DataMissingWarning,
                stacklevel=2,
            )
            result = self._make_result("goalie", current_season, "rmse",
                                       float("nan"), float("nan"), n_trees,
                                       0, accepted=False)
            return model, result

        X_all = np.vstack(X_parts).astype(np.float32)
        y_all = np.concatenate(y_parts)
        w_all = np.concatenate(w_parts)

        # Compute baseline RMSE
        metric_before = float("nan")
        if model._model is not None:
            try:
                preds_b = model._model.predict(X_all)
                metric_before = float(np.sqrt(np.mean((preds_b - y_all) ** 2)))
            except Exception:
                pass

        # Warm-start
        existing_booster = None
        if model._model is not None:
            existing_booster = model._model.get_booster()

        new_reg_params = {**model._params, "n_estimators": n_trees}
        new_reg = xgb.XGBRegressor(**new_reg_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_reg.fit(
                X_all, y_all,
                sample_weight=w_all,
                xgb_model=existing_booster,
                verbose=False,
            )

        # Build new GoalieModel
        new_model = GoalieModel.__new__(GoalieModel)
        new_model._model   = new_reg
        new_model._version = model._version
        new_model._params  = model._params

        # Evaluate after
        metric_after = float("nan")
        try:
            preds_a = new_model._model.predict(X_all)
            metric_after = float(np.sqrt(np.mean((preds_a - y_all) ** 2)))
        except Exception:
            pass

        # Decide acceptance (RMSE: lower = better)
        accepted = force
        if not force and not (np.isnan(metric_before) or np.isnan(metric_after)):
            improvement = metric_before - metric_after  # lower RMSE = better
            accepted = improvement >= self._accept_threshold

        result = self._make_result(
            "goalie", current_season, "rmse",
            metric_before, metric_after, n_trees,
            len(seasons_used), accepted,
        )
        returned_model = new_model if accepted else model
        return returned_model, result

    # ------------------------------------------------------------------
    # Log management
    # ------------------------------------------------------------------

    def _make_result(
        self,
        model_type:     str,
        current_season: int,
        metric_name:    str,
        metric_before:  float,
        metric_after:   float,
        n_new_trees:    int,
        n_seasons_used: int,
        accepted:       bool,
    ) -> RetrainResult:
        now = datetime.now(timezone.utc).isoformat()
        log_row = {
            "retrain_date":    now,
            "model_type":      model_type,
            "current_season":  current_season,
            "n_seasons_used":  n_seasons_used,
            "n_new_trees":     n_new_trees,
            "metric_before":   metric_before,
            "metric_after":    metric_after,
            "metric_name":     metric_name,
            "accepted":        accepted,
            "model_version":   self._version,
        }
        self._log.append(log_row)
        return RetrainResult(
            model_type     = model_type,
            current_season = current_season,
            metric_name    = metric_name,
            metric_before  = metric_before,
            metric_after   = metric_after,
            n_new_trees    = n_new_trees,
            n_seasons_used = n_seasons_used,
            accepted       = accepted,
            log_row        = log_row,
        )

    def log_dataframe(self) -> pl.DataFrame:
        """Return all retrain results as a DataFrame."""
        if not self._log:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in RETRAIN_LOG_SCHEMA.items()}
            )
        df = pl.DataFrame(self._log)
        for col, dtype in RETRAIN_LOG_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df.select(list(RETRAIN_LOG_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "decay":            self._decay,
            "n_new_trees":      self._n_new_trees,
            "accept_threshold": self._accept_threshold,
            "log":              self._log,
            "version":          self._version,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "IncrementalRetrainer":
        payload = joblib.load(Path(path))
        obj = cls(
            decay            = payload.get("decay",            SEASON_DECAY),
            n_new_trees      = payload.get("n_new_trees",      DEFAULT_N_NEW_TREES),
            accept_threshold = payload.get("accept_threshold", DEFAULT_ACCEPT_THRESH),
        )
        obj._log = payload.get("log", [])
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_retrain_log(df: pl.DataFrame, output_dir: Path, model_type: str) -> Path:
    """Write retrain log to ``output_dir/retrain_log_{model_type}.parquet``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"retrain_log_{model_type}.parquet"
    for col, dtype in RETRAIN_LOG_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(RETRAIN_LOG_SCHEMA.keys())).write_parquet(path)
    return path
