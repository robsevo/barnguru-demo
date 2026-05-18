"""Line Chemistry Model — Feature 2.4.

XGBoost + SHAP model that captures non-linear linemate interaction effects.
Given two players who share EV ice time on the SAME team, predicts whether
their combined xGF% outperforms or underperforms the sum of their individual
ratings.  The "chemistry delta" is the excess above/below what individual
RAPM predicts.

Complement to Feature 2.3 (Matchup / opposition pairs):
    Feature 2.3 → cross-team pairs (opposition)
    Feature 2.4 → same-team pairs (linemates)

Outputs
-------
``pair_chemistry_{season}.parquet``   — one row per (player_a, player_b) linemate pair:
    chemistry_delta = model_xgf_pct - LEAGUE_AVG_XGF_PCT

``player_chemistry_{season}.parquet`` — one row per player:
    chemistry_score = co_toi_ev-weighted avg chemistry_delta across all pairs

Blending rule (games_together = games where pair shared EV ice time):
    games_together < 10  → model prediction only  (archetype prior = 0.500 xGF%)
    games_together ≥ 10  → 0.70 × model_xgf_pct + 0.30 × historical_xgf_pct
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

MODEL_VERSION        = "chemistry_v1"
MIN_PAIR_GP          = 10
HIST_WEIGHT          = 0.30
LEAGUE_AVG_XGF_PCT   = 0.500
MIN_SHARED_TOI_SECS  = 60.0   # higher threshold than matchup — linemates need quality signal

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "n_estimators":     300,
    "max_depth":        3,           # shallow — keep it interpretable
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,          # regularise against small pair samples
    "seed":             42,
    "verbosity":        0,
    "n_jobs":           -1,
}

_FEATURE_NAMES = [
    "a_rapm_ev_off",
    "a_rapm_ev_def",
    "b_rapm_ev_off",
    "b_rapm_ev_def",
    "rapm_off_sum",        # a_rapm_ev_off + b_rapm_ev_off
    "rapm_def_sum",        # a_rapm_ev_def + b_rapm_ev_def
    "rapm_composite_sum",  # (a_off+a_def) + (b_off+b_def)
    "rapm_balance_abs",    # abs((a_off+a_def) - (b_off+b_def))  — role asymmetry
    "a_finishing",
    "b_finishing",
    "a_ozs_pct",
    "b_ozs_pct",
    "hist_xgf_pct",
    "log1p_gp",
]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

PAIR_CHEMISTRY_SCHEMA: dict[str, pl.DataType] = {
    "player_a_id":      pl.Int64,
    "player_b_id":      pl.Int64,
    "season":           pl.Int64,
    "co_toi_ev":        pl.Float64,
    "games_together":   pl.Int64,
    "hist_xgf_pct":     pl.Float64,   # observed xGF% together (null if GP < MIN_PAIR_GP)
    "model_xgf_pct":    pl.Float64,   # XGBoost prediction
    "chemistry_delta":  pl.Float64,   # model_xgf_pct - LEAGUE_AVG_XGF_PCT
    "model_version":    pl.Utf8,
}

PLAYER_CHEMISTRY_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "player_name":      pl.Utf8,
    "team":             pl.Utf8,
    "season":           pl.Int64,
    "gp":               pl.Int64,
    "toi_ev":           pl.Float64,
    "chemistry_score":  pl.Float64,   # co_toi_ev-weighted avg chemistry_delta across all pairs
    "chemistry_pairs":  pl.Int64,     # number of pairs used
    "model_version":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Linemate pair building
# ---------------------------------------------------------------------------

def _empty_pair_chemistry(season: int) -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in PAIR_CHEMISTRY_SCHEMA.items()}
    ).with_columns(pl.lit(season).cast(pl.Int64).alias("season"))


def build_linemate_pairs_with_xgf(
    stints_df: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Build canonical (player_a < player_b) linemate pairs with xGF accumulation.

    For each pair of players who share EV ice time on the same team, accumulates:
    - co_toi_ev: total shared EV seconds
    - games_together: number of distinct games
    - xgf_for: team xG while both players are on ice
    - xgf_against: opponent xG while both players are on ice
    - xgf_pct: xgf_for / (xgf_for + xgf_against)

    Only pairs with co_toi_ev >= MIN_SHARED_TOI_SECS are included.

    Args:
        stints_df: Output of ``build_stints()``; required columns:
            game_id, situation, duration_secs,
            home_player_ids (List[Int64]), away_player_ids (List[Int64]),
            xgf_home (Float64), xgf_away (Float64)
        season: Season integer for labelling output rows.

    Returns:
        Polars DataFrame with columns:
            player_a_id, player_b_id, season, co_toi_ev, games_together,
            xgf_for, xgf_against, xgf_pct

    Raises:
        ValueError: if required columns are missing.
        DataMissingWarning: if no EV stints found.
    """
    required = {
        "game_id", "situation", "duration_secs",
        "home_player_ids", "away_player_ids",
        "xgf_home", "xgf_away",
    }
    missing = required - set(stints_df.columns)
    if missing:
        raise ValueError(f"stints_df missing columns: {sorted(missing)}")

    ev = stints_df.filter(pl.col("situation") == "EV")
    if ev.is_empty():
        warnings.warn(
            "build_linemate_pairs_with_xgf: no EV stints found — returning empty DataFrame.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema={
            "player_a_id":    pl.Int64,
            "player_b_id":    pl.Int64,
            "season":         pl.Int64,
            "co_toi_ev":      pl.Float64,
            "games_together": pl.Int64,
            "xgf_for":        pl.Float64,
            "xgf_against":    pl.Float64,
            "xgf_pct":        pl.Float64,
        })

    ev = ev.with_row_index("_stint_idx")

    # Explode home and away player lists into long format.
    # For home: xgf_for = xgf_home (own team), xgf_against = xgf_away (opponent)
    # For away: xgf_for = xgf_away (own team), xgf_against = xgf_home (opponent)
    home_long = (
        ev.select([
            "_stint_idx", "game_id", "duration_secs",
            pl.col("xgf_home").alias("xgf_for"),
            pl.col("xgf_away").alias("xgf_against"),
            "home_player_ids",
        ])
        .explode("home_player_ids")
        .rename({"home_player_ids": "player_id"})
        .filter(pl.col("player_id").is_not_null())
    )
    away_long = (
        ev.select([
            "_stint_idx", "game_id", "duration_secs",
            pl.col("xgf_away").alias("xgf_for"),
            pl.col("xgf_home").alias("xgf_against"),
            "away_player_ids",
        ])
        .explode("away_player_ids")
        .rename({"away_player_ids": "player_id"})
        .filter(pl.col("player_id").is_not_null())
    )

    parts: list[pl.DataFrame] = []

    for side_df in (home_long, away_long):
        # Self-join on _stint_idx with canonical ordering (player_id < player_b_id)
        # xgf_for and xgf_against come from the LEFT side (identical for both players
        # in the same stint, since they're on the same team)
        left = side_df.select([
            "_stint_idx", "game_id", "duration_secs",
            "xgf_for", "xgf_against", "player_id",
        ])
        right = (
            side_df.select(["_stint_idx", "player_id"])
                   .rename({"player_id": "player_b_id"})
        )
        pair = (
            left.join(right, on="_stint_idx")
            .filter(pl.col("player_id") < pl.col("player_b_id"))
            .group_by(["player_id", "player_b_id"])
            .agg([
                pl.col("duration_secs").sum().alias("co_toi_ev"),
                pl.col("game_id").n_unique().alias("games_together"),
                pl.col("xgf_for").sum(),
                pl.col("xgf_against").sum(),
            ])
            .filter(pl.col("co_toi_ev") >= MIN_SHARED_TOI_SECS)
        )
        if not pair.is_empty():
            parts.append(pair)

    if not parts:
        return pl.DataFrame(schema={
            "player_a_id":    pl.Int64,
            "player_b_id":    pl.Int64,
            "season":         pl.Int64,
            "co_toi_ev":      pl.Float64,
            "games_together": pl.Int64,
            "xgf_for":        pl.Float64,
            "xgf_against":    pl.Float64,
            "xgf_pct":        pl.Float64,
        })

    # Deduplicate after concat: a traded player can appear as both home AND away
    # on the same team in different portions of the season.  Aggregate those rows
    # so each canonical pair appears exactly once with co_toi/xGF summed.
    lm = (
        pl.concat(parts)
        .group_by(["player_id", "player_b_id"])
        .agg([
            pl.col("co_toi_ev").sum(),
            pl.col("games_together").sum(),
            pl.col("xgf_for").sum(),
            pl.col("xgf_against").sum(),
        ])
        .filter(pl.col("co_toi_ev") >= MIN_SHARED_TOI_SECS)
        .rename({"player_id": "player_a_id"})
        .with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            (
                pl.col("xgf_for") /
                (pl.col("xgf_for") + pl.col("xgf_against")).clip(lower_bound=1e-9)
            ).alias("xgf_pct"),
        ])
        .select([
            "player_a_id", "player_b_id", "season", "co_toi_ev",
            "games_together", "xgf_for", "xgf_against", "xgf_pct",
        ])
    )
    return lm


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_chemistry_features(
    linemate_df: pl.DataFrame,
    rapm_df: pl.DataFrame,
    finishing_df: pl.DataFrame | None,
    player_stats_df: pl.DataFrame | None,
    season: int,
) -> tuple[np.ndarray, np.ndarray | None, pl.DataFrame]:
    """Build XGBoost feature matrix from linemate pair data.

    Args:
        linemate_df:      Output of ``build_linemate_pairs_with_xgf()``.
        rapm_df:          RAPM output for the season.
        finishing_df:     FINISHING_SCHEMA DataFrame (optional; provides finishing scores).
        player_stats_df:  player_stats DataFrame (optional; zone start %s).
        season:           Season integer.

    Returns:
        (X, y, meta) where:
            X    — float32 array, shape (n_pairs, len(_FEATURE_NAMES))
            y    — float32 xgf_pct targets, or None if not available
            meta — DataFrame with player_a_id, player_b_id, games_together, co_toi_ev
    """
    rapm_s = (
        rapm_df.filter(pl.col("season") == season)
        if "season" in rapm_df.columns else rapm_df.clone()
    )

    def _season_filter(df: pl.DataFrame | None) -> pl.DataFrame | None:
        if df is None or df.is_empty():
            return None
        return df.filter(pl.col("season") == season) if "season" in df.columns else df.clone()

    finishing_s   = _season_filter(finishing_df)
    player_stats_s = _season_filter(player_stats_df)

    pairs = linemate_df.select(
        ["player_a_id", "player_b_id", "games_together", "co_toi_ev", "xgf_pct"]
    )

    # RAPM for A
    pairs = pairs.join(
        rapm_s.select(["player_id", "rapm_ev_off", "rapm_ev_def"])
              .rename({"player_id":    "player_a_id",
                       "rapm_ev_off":  "a_rapm_ev_off",
                       "rapm_ev_def":  "a_rapm_ev_def"}),
        on="player_a_id", how="inner",
    )

    # RAPM for B
    pairs = pairs.join(
        rapm_s.select(["player_id", "rapm_ev_off", "rapm_ev_def"])
              .rename({"player_id":    "player_b_id",
                       "rapm_ev_off":  "b_rapm_ev_off",
                       "rapm_ev_def":  "b_rapm_ev_def"}),
        on="player_b_id", how="inner",
    )

    # Finishing for A and B (key: shooter_id; fill null with 0.0)
    if (finishing_s is not None
            and {"shooter_id", "finishing"}.issubset(finishing_s.columns)):
        fin = finishing_s.select(["shooter_id", "finishing"])
        pairs = (
            pairs
            .join(fin.rename({"shooter_id": "player_a_id",
                               "finishing":  "a_finishing"}),
                  on="player_a_id", how="left")
            .join(fin.rename({"shooter_id": "player_b_id",
                               "finishing":  "b_finishing"}),
                  on="player_b_id", how="left")
        )
    else:
        pairs = pairs.with_columns([
            pl.lit(0.0).alias("a_finishing"),
            pl.lit(0.0).alias("b_finishing"),
        ])

    pairs = pairs.with_columns([
        pl.col("a_finishing").fill_null(0.0),
        pl.col("b_finishing").fill_null(0.0),
    ])

    # Zone starts for A and B (fill null with 0.5)
    if (player_stats_s is not None
            and {"player_id", "zone_start_oz_pct"}.issubset(player_stats_s.columns)):
        ozs = player_stats_s.select(["player_id", "zone_start_oz_pct"])
        pairs = (
            pairs
            .join(ozs.rename({"player_id":          "player_a_id",
                               "zone_start_oz_pct":  "a_ozs_pct"}),
                  on="player_a_id", how="left")
            .join(ozs.rename({"player_id":          "player_b_id",
                               "zone_start_oz_pct":  "b_ozs_pct"}),
                  on="player_b_id", how="left")
        )
    else:
        pairs = pairs.with_columns([
            pl.lit(0.5).alias("a_ozs_pct"),
            pl.lit(0.5).alias("b_ozs_pct"),
        ])

    pairs = pairs.with_columns([
        pl.col("a_ozs_pct").fill_null(0.5),
        pl.col("b_ozs_pct").fill_null(0.5),
    ])

    # Derived RAPM features
    pairs = pairs.with_columns([
        (pl.col("a_rapm_ev_off") + pl.col("b_rapm_ev_off")).alias("rapm_off_sum"),
        (pl.col("a_rapm_ev_def") + pl.col("b_rapm_ev_def")).alias("rapm_def_sum"),
        (
            (pl.col("a_rapm_ev_off") + pl.col("a_rapm_ev_def")) +
            (pl.col("b_rapm_ev_off") + pl.col("b_rapm_ev_def"))
        ).alias("rapm_composite_sum"),
        (
            (pl.col("a_rapm_ev_off") + pl.col("a_rapm_ev_def")) -
            (pl.col("b_rapm_ev_off") + pl.col("b_rapm_ev_def"))
        ).abs().alias("rapm_balance_abs"),
    ])

    # Historical xGF% feature: actual when GP >= MIN_PAIR_GP, else archetype prior
    pairs = pairs.with_columns(
        pl.when(pl.col("games_together") >= MIN_PAIR_GP)
        .then(pl.col("xgf_pct"))
        .otherwise(pl.lit(LEAGUE_AVG_XGF_PCT))
        .alias("hist_xgf_pct")
    )

    # log(1 + games_together) — sample size signal
    pairs = pairs.with_columns(
        pl.Series(
            "log1p_gp",
            np.log1p(pairs["games_together"].cast(pl.Float64).to_numpy()),
            dtype=pl.Float64,
        )
    )

    X = (
        pairs.select(_FEATURE_NAMES)
        .to_numpy()
        .astype(np.float32)
    )
    X = np.nan_to_num(X, nan=0.0)

    y: np.ndarray | None = None
    if "xgf_pct" in pairs.columns:
        raw_y = pairs["xgf_pct"].to_numpy().astype(np.float32)
        valid  = ~np.isnan(raw_y)
        X      = X[valid]
        y      = raw_y[valid]
        pairs  = pairs.filter(pl.col("xgf_pct").is_not_null())

    meta = pairs.select(["player_a_id", "player_b_id", "games_together", "co_toi_ev"])
    return X, y, meta


# ---------------------------------------------------------------------------
# LineChemistryModel
# ---------------------------------------------------------------------------

class LineChemistryModel:
    """XGBoost line chemistry model (Feature 2.4).

    Predicts xGF% for same-team linemate pairs.  The chemistry_delta
    (model_xgf_pct - LEAGUE_AVG_XGF_PCT) captures non-linear interaction
    effects above/below individual RAPM predictions.

    Blends with historical xGF% for pairs with games_together >= MIN_PAIR_GP.

    Usage::

        model = LineChemistryModel()
        metrics = model.fit(
            stints_list=[stints_2122, stints_2223],
            rapm_list=[rapm_2122, rapm_2223],
            finishing_list=[fin_2122, fin_2223],
            player_stats_list=[stats_2122, stats_2223],
            seasons=[20212022, 20222023],
        )
        pairs = model.predict_season(
            stints_df=stints_2324,
            rapm_df=rapm_2324,
            finishing_df=fin_2324,
            player_stats_df=stats_2324,
            season=20232024,
        )
        model.save(Path("~/.gretzky/data/chemistry/chemistry_model.pkl"))
        loaded = LineChemistryModel.load(Path("~/.gretzky/data/chemistry/chemistry_model.pkl"))
    """

    def __init__(self, xgb_params: dict[str, Any] | None = None) -> None:
        self._params  = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self._model: Any = None
        self._version = MODEL_VERSION

    # --- training ---

    def fit(
        self,
        stints_list:       list[pl.DataFrame],
        rapm_list:         list[pl.DataFrame],
        finishing_list:    list[pl.DataFrame | None],
        player_stats_list: list[pl.DataFrame | None],
        seasons:           list[int],
    ) -> dict[str, float]:
        """Train on historical linemate pairs across multiple seasons.

        Args:
            stints_list:       One build_stints() DataFrame per season.
            rapm_list:         One RAPM DataFrame per season.
            finishing_list:    One finishing DataFrame per season (None = skip).
            player_stats_list: One player_stats DataFrame per season (None = skip).
            seasons:           Season integers matching the above lists.

        Returns:
            {"rmse": float, "n_pairs": int}

        Raises:
            ImportError: if xgboost is not installed.
            ValueError: if no training data is produced.
        """
        if not _XGB_AVAILABLE:
            raise ImportError(
                "xgboost is required for LineChemistryModel. "
                "Run: uv add xgboost"
            )

        if not (len(stints_list) == len(rapm_list) == len(seasons)):
            raise ValueError("stints_list, rapm_list, and seasons must have equal length.")

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for stints_df, rapm_df, finishing_df, stats_df, season in zip(
            stints_list, rapm_list, finishing_list, player_stats_list, seasons
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                linemate_df = build_linemate_pairs_with_xgf(stints_df, season)

            if linemate_df.is_empty():
                continue

            X_i, y_i, _ = _build_chemistry_features(
                linemate_df, rapm_df, finishing_df, stats_df, season
            )
            if y_i is not None and len(y_i) > 0:
                X_parts.append(X_i)
                y_parts.append(y_i)

        if not X_parts:
            raise ValueError("No training data produced — check stints/RAPM data.")

        X_train = np.vstack(X_parts)
        y_train  = np.concatenate(y_parts)

        self._model = xgb.XGBRegressor(**self._params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_train, y_train, verbose=False)

        preds = self._model.predict(X_train)
        rmse  = float(np.sqrt(np.mean((preds - y_train) ** 2)))

        return {"rmse": rmse, "n_pairs": len(X_train)}

    # --- inference ---

    def predict_season(
        self,
        stints_df:       pl.DataFrame,
        rapm_df:         pl.DataFrame,
        finishing_df:    pl.DataFrame | None,
        player_stats_df: pl.DataFrame | None,
        season:          int,
    ) -> pl.DataFrame:
        """Generate blended xGF% predictions for all EV linemate pairs in a season.

        Blending rule:
            games_together < MIN_PAIR_GP  → model_xgf_pct  (archetype prior)
            games_together >= MIN_PAIR_GP → 0.70 × model + 0.30 × historical

        chemistry_delta = model_xgf_pct - LEAGUE_AVG_XGF_PCT

        Returns:
            DataFrame matching PAIR_CHEMISTRY_SCHEMA, sorted descending by chemistry_delta.

        Raises:
            RuntimeError: if model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linemate_df = build_linemate_pairs_with_xgf(stints_df, season)

        if linemate_df.is_empty():
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PAIR_CHEMISTRY_SCHEMA.items()}
            )

        X, _, meta = _build_chemistry_features(
            linemate_df, rapm_df, finishing_df, player_stats_df, season
        )
        model_preds = np.clip(self._model.predict(X).astype(np.float64), 0.0, 1.0)

        # Pull historical xGF% from linemate_df for blending
        hist = (
            linemate_df.select(["player_a_id", "player_b_id", "xgf_pct"])
                       .rename({"xgf_pct": "_hist"})
        )
        meta = meta.join(hist, on=["player_a_id", "player_b_id"], how="left")

        hist_arr = meta["_hist"].fill_null(LEAGUE_AVG_XGF_PCT).to_numpy()
        gp_arr   = meta["games_together"].to_numpy()

        w_hist      = np.where(gp_arr >= MIN_PAIR_GP, HIST_WEIGHT, 0.0)
        blended     = (1.0 - w_hist) * model_preds + w_hist * hist_arr

        # hist_xgf_pct: actual when GP >= MIN_PAIR_GP, else null (Polars null, not NaN)
        hist_xgf_pct_arr = np.where(gp_arr >= MIN_PAIR_GP, hist_arr, np.nan)

        # Convert NaN sentinels to Polars null for hist_xgf_pct
        hist_series = (
            pl.Series("hist_xgf_pct", hist_xgf_pct_arr, dtype=pl.Float64)
            .set(pl.Series(np.isnan(hist_xgf_pct_arr)), None)
        )

        result = meta.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            hist_series,
            pl.Series("model_xgf_pct",   blended,                       dtype=pl.Float64),
            pl.Series("chemistry_delta",  blended - LEAGUE_AVG_XGF_PCT,  dtype=pl.Float64),
            pl.lit(self._version).alias("model_version"),
        ])

        for col, dtype in PAIR_CHEMISTRY_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))
        return result.select(list(PAIR_CHEMISTRY_SCHEMA.keys())).sort(
            "chemistry_delta", descending=True
        )

    # --- interpretability ---

    def feature_importance(self) -> dict[str, float]:
        """Return feature importances.

        Uses SHAP mean absolute values if shap is available, else XGBoost
        gain-based importance.

        Returns:
            dict mapping feature name → importance value, sorted descending.

        Raises:
            RuntimeError: if model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        # Gain-based feature importance (always available).
        # For SHAP values over a dataset, call feature_importance_from_data(X) instead.
        raw = self._model.get_booster().get_score(importance_type="gain")
        fi: dict[str, float] = {}
        for k, v in raw.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(_FEATURE_NAMES):
                    fi[_FEATURE_NAMES[idx]] = float(v)
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    def feature_importance_from_data(
        self,
        X: np.ndarray,
    ) -> dict[str, float]:
        """Return SHAP mean absolute feature importances computed over X.

        Requires shap to be installed.  Falls back to gain-based if not available.

        Args:
            X: Feature matrix (n_samples, n_features) in _FEATURE_NAMES order.

        Returns:
            dict mapping feature name → mean |SHAP value|, sorted descending.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        if _SHAP_AVAILABLE:
            explainer  = _shap.TreeExplainer(self._model)
            shap_vals  = explainer.shap_values(X)
            mean_abs   = np.abs(shap_vals).mean(axis=0)
            fi = {
                name: float(mean_abs[i])
                for i, name in enumerate(_FEATURE_NAMES)
            }
            return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

        return self.feature_importance()

    # --- persistence ---

    def save(self, path: Path) -> None:
        """Serialize model + metadata to path using joblib."""
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
    def load(cls, path: Path) -> "LineChemistryModel":
        """Deserialize model from path."""
        payload = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._model   = payload["model"]
        obj._version = payload.get("version", MODEL_VERSION)
        obj._params  = payload.get("params", DEFAULT_XGB_PARAMS)
        return obj


# ---------------------------------------------------------------------------
# Player-level chemistry aggregation
# ---------------------------------------------------------------------------

def compute_player_chemistry(
    pair_df: pl.DataFrame,
    rapm_df: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Aggregate pair chemistry to per-player scores.

    For each player, collects all their pairs (both as player_a and player_b),
    computes co_toi_ev-weighted avg chemistry_delta, and joins with RAPM for
    player metadata (player_name, team, gp, toi_ev).

    Args:
        pair_df: PAIR_CHEMISTRY_SCHEMA DataFrame from predict_season().
        rapm_df: RAPM output (must have player_id, player_name, team, gp, toi_ev).
        season:  Season integer for labelling output rows.

    Returns:
        Polars DataFrame matching PLAYER_CHEMISTRY_SCHEMA, sorted by chemistry_score desc.
    """
    if pair_df.is_empty():
        return pl.DataFrame(
            {col: pl.Series([], dtype=dt) for col, dt in PLAYER_CHEMISTRY_SCHEMA.items()}
        )

    rapm_s = (
        rapm_df.filter(pl.col("season") == season)
        if "season" in rapm_df.columns else rapm_df.clone()
    )

    # Build A-perspective rows: player_id = player_a_id
    side_a = pair_df.select([
        pl.col("player_a_id").alias("player_id"),
        pl.col("co_toi_ev"),
        pl.col("chemistry_delta"),
    ])

    # Build B-perspective rows: player_id = player_b_id
    side_b = pair_df.select([
        pl.col("player_b_id").alias("player_id"),
        pl.col("co_toi_ev"),
        pl.col("chemistry_delta"),
    ])

    all_sides = pl.concat([side_a, side_b])

    # Weighted avg chemistry_delta by co_toi_ev
    player_chem = (
        all_sides
        .with_columns(
            (pl.col("co_toi_ev") * pl.col("chemistry_delta")).alias("_wt")
        )
        .group_by("player_id")
        .agg([
            pl.col("_wt").sum().alias("_num"),
            pl.col("co_toi_ev").sum().alias("_den"),
            pl.col("chemistry_delta").count().alias("chemistry_pairs"),
        ])
        .with_columns(
            (pl.col("_num") / pl.col("_den").clip(lower_bound=1e-9))
            .alias("chemistry_score")
        )
        .select(["player_id", "chemistry_score", "chemistry_pairs"])
    )

    # Join RAPM metadata
    rapm_meta = rapm_s.select([
        "player_id", "player_name", "team", "gp", "toi_ev",
    ])

    result = (
        player_chem.join(rapm_meta, on="player_id", how="left")
        .with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit(MODEL_VERSION).alias("model_version"),
            pl.col("player_name").fill_null("Unknown"),
            pl.col("team").fill_null(""),
            pl.col("gp").fill_null(0).cast(pl.Int64),
            pl.col("toi_ev").fill_null(0.0),
            pl.col("chemistry_pairs").cast(pl.Int64),
        ])
        .select(list(PLAYER_CHEMISTRY_SCHEMA.keys()))
    )
    return result.sort("chemistry_score", descending=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _enforce_pair_schema(df: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in PAIR_CHEMISTRY_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    return df.select(list(PAIR_CHEMISTRY_SCHEMA.keys()))


def _enforce_player_schema(df: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in PLAYER_CHEMISTRY_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    return df.select(list(PLAYER_CHEMISTRY_SCHEMA.keys()))


def write_pair_chemistry(
    df: pl.DataFrame, output_dir: Path, season: int, season_type: str = "regular"
) -> Path:
    """Write pair chemistry parquet. season_type='playoffs' appends _playoffs suffix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_playoffs" if season_type == "playoffs" else ""
    path = output_dir / f"pair_chemistry_{season}{suffix}.parquet"
    _enforce_pair_schema(df).write_parquet(path)
    return path


def read_pair_chemistry(data_dir: Path, season: int) -> pl.DataFrame:
    """Read pair chemistry parquet from ``data_dir/chemistry/pair_chemistry_{season}.parquet``."""
    path = Path(data_dir) / "chemistry" / f"pair_chemistry_{season}.parquet"
    return pl.read_parquet(path)


def write_player_chemistry(
    df: pl.DataFrame, output_dir: Path, season: int, season_type: str = "regular"
) -> Path:
    """Write player chemistry parquet. season_type='playoffs' appends _playoffs suffix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_playoffs" if season_type == "playoffs" else ""
    path = output_dir / f"player_chemistry_{season}{suffix}.parquet"
    _enforce_player_schema(df).write_parquet(path)
    return path


def read_player_chemistry(data_dir: Path, season: int) -> pl.DataFrame:
    """Read player chemistry parquet from ``data_dir/chemistry/player_chemistry_{season}.parquet``."""
    path = Path(data_dir) / "chemistry" / f"player_chemistry_{season}.parquet"
    return pl.read_parquet(path)
