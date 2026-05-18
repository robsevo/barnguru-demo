"""Matchup Efficiency Matrix — Feature 2.3.

XGBoost model on matchup pair history, producing explicit QoC and QoT scores.

Outputs
-------
``qot_qoc_{season}.parquet`` — one row per player, per season:
    QoT  Quality of Teammates: avg composite RAPM of linemates, weighted by shared EV TOI.
         Surfaces deployment context ("2.4 xGF/60 with McDavid" ≠ "4th-line 2.4 xGF/60").
    QoC  Quality of Competition: avg composite RAPM of opponents, weighted by shared EV TOI.

``matchup_preds_{season}.parquet`` — one row per (player_a, player_b) opposition pair:
    Predicted xGF% for player A's team when A and B are on ice simultaneously.

    Blending rule (pair_gp = games where A and B shared EV ice time):
        pair_gp < 10  → model prediction only  (archetype prior = 0.500 xGF%)
        pair_gp ≥ 10  → 0.70 × model_xgf_pct + 0.30 × historical_xgf_pct

Note: composite RAPM = rapm_ev_off + rapm_ev_def is used as the player-value weight for
QoT/QoC until feature 2.25 (WAR unification) is implemented.  Substitute WAR there.

Architecture
------------
1. build_cotoi_from_stints(stints_df, season) → (linemate_df, opponent_df)
2. compute_qot_qoc(stints_df, rapm_df, season) → QoT/QoC DataFrame
3. MatchupModel.fit(stints_list, rapm_list, ...)  → trained XGBoost
4. MatchupModel.predict_season(...)               → blended xGF% per pair
"""

from __future__ import annotations

import math
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

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION       = "matchup_v1"
MIN_PAIR_GP         = 10      # pair games together before blending in historical data
HIST_WEIGHT         = 0.30    # weight on historical xGF% when pair_gp >= MIN_PAIR_GP
LEAGUE_AVG_XGF_PCT  = 0.500   # archetype prior; replace with 2.11 archetypes when built
LEAGUE_AVG_SH_PCT   = 0.100   # fallback shooting pct vs team when no vs_team history
MIN_SHARED_TOI_SECS = 30.0    # minimum shared EV seconds to include a pair

# NHL team 3-letter code → NHL API integer team_id (inverse of rapm_model._NHL_TEAM_IDS)
_TEAM_CODE_TO_ID: dict[str, int] = {
    "NJD": 1,  "NYI": 2,  "NYR": 3,  "PHI": 4,  "PIT": 5,
    "BOS": 6,  "BUF": 7,  "MTL": 8,  "OTT": 9,  "TOR": 10,
    "CAR": 12, "FLA": 13, "TBL": 14, "WSH": 15, "CHI": 16,
    "DET": 17, "NSH": 18, "STL": 19, "CGY": 20, "COL": 21,
    "EDM": 22, "VAN": 23, "ANA": 24, "DAL": 25, "LAK": 26,
    "SJS": 28, "CBJ": 29, "MIN": 30, "WPG": 52, "ARI": 53,
    "VGK": 54, "SEA": 55, "UTH": 59,
}

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
    "a_ozs_pct",
    "b_ozs_pct",
    "vs_team_sh_pct",
    "hist_xgf_pct",
    "log1p_gp",
]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

CO_TOI_SCHEMA: dict[str, pl.DataType] = {
    "player_a_id":    pl.Int64,
    "player_b_id":    pl.Int64,
    "season":         pl.Int64,
    "side":           pl.Utf8,    # "linemate" | "opponent"
    "co_toi_ev":      pl.Float64, # shared EV seconds
    "games_together": pl.Int64,
    "xgf_for":        pl.Float64, # A-team xG in shared stints (opponent pairs only)
    "xgf_against":    pl.Float64, # opponent xG in shared stints (opponent pairs only)
    "xgf_pct":        pl.Float64, # xgf_for / (xgf_for + xgf_against) (opponent pairs only)
}

QOT_QOC_SCHEMA: dict[str, pl.DataType] = {
    "player_id":     pl.Int64,
    "player_name":   pl.Utf8,
    "team":          pl.Utf8,
    "season":        pl.Int64,
    "gp":            pl.Int64,
    "toi_ev":        pl.Float64,
    "qot":           pl.Float64,  # avg composite RAPM of linemates, co-TOI weighted
    "qoc":           pl.Float64,  # avg composite RAPM of opponents, co-TOI weighted
    "qot_toi":       pl.Float64,  # total linemate co-TOI (sample size indicator)
    "qoc_toi":       pl.Float64,  # total opponent co-TOI
    "model_version": pl.Utf8,
}

MATCHUP_PRED_SCHEMA: dict[str, pl.DataType] = {
    "player_a_id":    pl.Int64,
    "player_b_id":    pl.Int64,
    "season":         pl.Int64,
    "games_together": pl.Int64,
    "co_toi_ev":      pl.Float64,
    "hist_xgf_pct":   pl.Float64, # observed xGF% from shared stints
    "model_xgf_pct":  pl.Float64, # raw XGBoost prediction
    "xgf_pct":        pl.Float64, # final blended prediction
    "model_version":  pl.Utf8,
}


# ---------------------------------------------------------------------------
# Pairwise co-TOI from stints
# ---------------------------------------------------------------------------

def _empty_cotoi(season: int) -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in CO_TOI_SCHEMA.items()}
    ).with_columns(pl.lit(season).cast(pl.Int64).alias("season"))


def _enforce_cotoi_schema(df: pl.DataFrame, season: int) -> pl.DataFrame:
    for col, dtype in CO_TOI_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df = df.select(list(CO_TOI_SCHEMA.keys()))
    return df.with_columns(pl.col("season").fill_null(season).cast(pl.Int64))


def build_cotoi_from_stints(
    stints_df: pl.DataFrame,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute pairwise EV co-TOI and xGF from a stint DataFrame.

    Args:
        stints_df: Output of ``build_stints()``; required columns:
            game_id, situation, duration_secs,
            home_player_ids (List[Int64]), away_player_ids (List[Int64]),
            xgf_home (Float64), xgf_away (Float64)
        season: Season integer for labelling output rows.

    Returns:
        (linemate_df, opponent_df) — Polars DataFrames matching CO_TOI_SCHEMA.

        *linemate_df*: same-team pairs (player_a_id < player_b_id for uniqueness);
        xgf columns are null.

        *opponent_df*: cross-team pairs in **both orientations** (A=home vs B=away
        AND A=away vs B=home) so every player appears as player_a; xgf_pct is
        A-team's xGF% when A and B are on ice together.

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
            "build_cotoi_from_stints: no EV stints found — returning empty DataFrames.",
            DataMissingWarning,
            stacklevel=2,
        )
        return _empty_cotoi(season), _empty_cotoi(season)

    ev = ev.with_row_index("_stint_idx")

    # Explode home and away player lists into long format, each with their
    # team's xGF perspective (xgf_for = own team's xG, xgf_against = opponent xG).
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

    linemate_df = _build_linemate_pairs(home_long, away_long, season)
    opponent_df = _build_opposition_pairs(home_long, away_long, season)
    return linemate_df, opponent_df


def _build_linemate_pairs(
    home_long: pl.DataFrame,
    away_long: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Build canonical (player_a < player_b) linemate co-TOI pairs."""
    parts: list[pl.DataFrame] = []

    for side_df in (home_long, away_long):
        left  = side_df.select(["_stint_idx", "game_id", "duration_secs", "player_id"])
        right = (
            left.select(["_stint_idx", "player_id"])
                .rename({"player_id": "player_b_id"})
        )
        pair = (
            left.join(right, on="_stint_idx")
            .filter(pl.col("player_id") < pl.col("player_b_id"))
            .group_by(["player_id", "player_b_id"])
            .agg([
                pl.col("duration_secs").sum().alias("co_toi_ev"),
                pl.col("game_id").n_unique().alias("games_together"),
            ])
            .filter(pl.col("co_toi_ev") >= MIN_SHARED_TOI_SECS)
        )
        if not pair.is_empty():
            parts.append(pair)

    if not parts:
        return _empty_cotoi(season)

    lm = (
        pl.concat(parts)
        .group_by(["player_id", "player_b_id"])
        .agg([
            pl.col("co_toi_ev").sum(),
            pl.col("games_together").sum(),
        ])
        .rename({"player_id": "player_a_id"})
        .with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit("linemate").alias("side"),
        ])
    )
    return _enforce_cotoi_schema(lm, season)


def _build_opposition_pairs(
    home_long: pl.DataFrame,
    away_long: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Build opposition pairs in both orientations (home→away and away→home)."""
    parts: list[pl.DataFrame] = []

    for focal_long, opp_long in ((home_long, away_long), (away_long, home_long)):
        left  = focal_long.select(
            ["_stint_idx", "game_id", "duration_secs", "xgf_for", "xgf_against", "player_id"]
        )
        right = (
            opp_long.select(["_stint_idx", "player_id"])
                    .rename({"player_id": "player_b_id"})
        )
        pair = (
            left.join(right, on="_stint_idx")
            .group_by(["player_id", "player_b_id"])
            .agg([
                pl.col("duration_secs").sum().alias("co_toi_ev"),
                pl.col("game_id").n_unique().alias("games_together"),
                pl.col("xgf_for").sum(),
                pl.col("xgf_against").sum(),
            ])
            .filter(pl.col("co_toi_ev") >= MIN_SHARED_TOI_SECS)
            .rename({"player_id": "player_a_id"})
        )
        if not pair.is_empty():
            parts.append(pair)

    if not parts:
        return _empty_cotoi(season)

    # Deduplicate after concat: a traded player can appear as both home AND away
    # against the same opponent in different portions of the season, producing two
    # rows for the same (player_a_id, player_b_id) pair.  Aggregate those rows so
    # each pair appears exactly once, with co_toi/xGF summed across both stints.
    opp = (
        pl.concat(parts)
        .group_by(["player_a_id", "player_b_id"])
        .agg([
            pl.col("co_toi_ev").sum(),
            pl.col("games_together").sum(),
            pl.col("xgf_for").sum(),
            pl.col("xgf_against").sum(),
        ])
        .filter(pl.col("co_toi_ev") >= MIN_SHARED_TOI_SECS)
        .with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit("opponent").alias("side"),
            (
                pl.col("xgf_for") /
                (pl.col("xgf_for") + pl.col("xgf_against")).clip(lower_bound=1e-9)
            ).alias("xgf_pct"),
        ])
    )
    return _enforce_cotoi_schema(opp, season)


# ---------------------------------------------------------------------------
# QoT / QoC computation
# ---------------------------------------------------------------------------

def compute_qot_qoc(
    stints_df: pl.DataFrame,
    rapm_df: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """Compute QoT and QoC for all players in a season.

    QoT = avg composite RAPM of linemates, weighted by shared EV TOI.
    QoC = avg composite RAPM of opponents, weighted by shared EV TOI.

    Composite RAPM = rapm_ev_off + rapm_ev_def (proxy for player value).
    Replace with WAR when feature 2.25 is implemented.

    Args:
        stints_df: Output of ``build_stints()``.
        rapm_df:   RAPM output (RAPM_OUTPUT_SCHEMA); if it has a ``season`` column,
                   rows are filtered to ``season`` before computing.
        season:    Season integer (e.g. 20232024).

    Returns:
        Polars DataFrame matching QOT_QOC_SCHEMA, sorted descending by QoT.

    Raises:
        ValueError: if rapm_df is missing required columns.
    """
    required = {"player_id", "player_name", "team", "gp", "toi_ev",
                "rapm_ev_off", "rapm_ev_def"}
    missing = required - set(rapm_df.columns)
    if missing:
        raise ValueError(f"rapm_df missing required columns: {sorted(missing)}")

    rapm_s = (
        rapm_df.filter(pl.col("season") == season)
        if "season" in rapm_df.columns
        else rapm_df.clone()
    )
    rapm_s = rapm_s.with_columns(
        (pl.col("rapm_ev_off") + pl.col("rapm_ev_def")).alias("rapm_composite")
    ).select(["player_id", "player_name", "team", "gp", "toi_ev", "rapm_composite"])

    linemate_df, opponent_df = build_cotoi_from_stints(stints_df, season)

    rapm_vals = rapm_s.select(["player_id", "rapm_composite"])

    # ── QoT: linemate_df is canonical (player_a < player_b); handle both sides ──
    def _weighted_rapm(
        pairs: pl.DataFrame,
        focal_col: str,
        partner_col: str,
    ) -> pl.DataFrame:
        """Return (focal_player_id, weighted_sum, co_toi_sum) for one orientation."""
        return (
            pairs
            .join(
                rapm_vals.rename({
                    "player_id":       partner_col,
                    "rapm_composite":  "partner_rapm",
                }),
                on=partner_col,
                how="inner",
            )
            .with_columns(
                (pl.col("co_toi_ev") * pl.col("partner_rapm")).alias("_wt")
            )
            .group_by(focal_col)
            .agg([
                pl.col("_wt").sum().alias("num"),
                pl.col("co_toi_ev").sum().alias("den"),
            ])
            .rename({focal_col: "player_id"})
        )

    qot_ab = _weighted_rapm(linemate_df, "player_a_id", "player_b_id")
    qot_ba = _weighted_rapm(linemate_df, "player_b_id", "player_a_id")

    qot_all = (
        pl.concat([qot_ab, qot_ba])
        .group_by("player_id")
        .agg([pl.col("num").sum(), pl.col("den").sum()])
        .with_columns(
            (pl.col("num") / pl.col("den").clip(lower_bound=1e-9)).alias("qot")
        )
        .rename({"den": "qot_toi"})
        .select(["player_id", "qot", "qot_toi"])
    )

    # ── QoC: opponent_df already has both orientations ──
    qoc_all = (
        _weighted_rapm(opponent_df, "player_a_id", "player_b_id")
        .with_columns(
            (pl.col("num") / pl.col("den").clip(lower_bound=1e-9)).alias("qoc")
        )
        .rename({"den": "qoc_toi"})
        .select(["player_id", "qoc", "qoc_toi"])
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    result = (
        rapm_s
        .join(qot_all, on="player_id", how="left")
        .join(qoc_all, on="player_id", how="left")
        .with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit(MODEL_VERSION).alias("model_version"),
            pl.col("qot").fill_null(0.0),
            pl.col("qoc").fill_null(0.0),
            pl.col("qot_toi").fill_null(0.0),
            pl.col("qoc_toi").fill_null(0.0),
        ])
        .select(list(QOT_QOC_SCHEMA.keys()))
    )
    return result.sort("qot", descending=True)


# ---------------------------------------------------------------------------
# Matchup feature engineering
# ---------------------------------------------------------------------------

def _build_matchup_features(
    opponent_df: pl.DataFrame,
    rapm_df: pl.DataFrame,
    vs_team_df: pl.DataFrame | None,
    player_stats_df: pl.DataFrame | None,
    season: int,
) -> tuple[np.ndarray, np.ndarray | None, pl.DataFrame]:
    """Build XGBoost feature matrix from opposition pair data.

    Args:
        opponent_df:      Opposition pairs from ``build_cotoi_from_stints()``.
        rapm_df:          RAPM output for the season.
        vs_team_df:       VS_TEAM_SCHEMA DataFrame (optional; provides historical
                          shooting pct vs opponent team).
        player_stats_df:  PLAYER_STATS_SCHEMA DataFrame (optional; zone start %s).
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

    # Filter stats helpers to season
    def _season_filter(df: pl.DataFrame | None) -> pl.DataFrame | None:
        if df is None or df.is_empty():
            return None
        return df.filter(pl.col("season") == season) if "season" in df.columns else df.clone()

    stats_s    = _season_filter(player_stats_df)
    vs_team_s  = _season_filter(vs_team_df)

    pairs = opponent_df.select(
        ["player_a_id", "player_b_id", "games_together", "co_toi_ev", "xgf_pct"]
    )

    # RAPM for A
    pairs = pairs.join(
        rapm_s.select(["player_id", "rapm_ev_off", "rapm_ev_def", "team"])
              .rename({"player_id": "player_a_id",
                       "rapm_ev_off": "a_rapm_ev_off",
                       "rapm_ev_def": "a_rapm_ev_def",
                       "team": "a_team"}),
        on="player_a_id", how="inner",
    )

    # RAPM for B (also carry B's team for vs_team lookup)
    pairs = pairs.join(
        rapm_s.select(["player_id", "rapm_ev_off", "rapm_ev_def", "team"])
              .rename({"player_id": "player_b_id",
                       "rapm_ev_off": "b_rapm_ev_off",
                       "rapm_ev_def": "b_rapm_ev_def",
                       "team": "b_team"}),
        on="player_b_id", how="inner",
    )

    # Zone starts for A and B
    if (stats_s is not None
            and {"player_id", "zone_start_oz_pct"}.issubset(stats_s.columns)):
        ozs = stats_s.select(["player_id", "zone_start_oz_pct"])
        pairs = (
            pairs
            .join(ozs.rename({"player_id": "player_a_id",
                               "zone_start_oz_pct": "a_ozs_pct"}),
                  on="player_a_id", how="left")
            .join(ozs.rename({"player_id": "player_b_id",
                               "zone_start_oz_pct": "b_ozs_pct"}),
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

    # vs_team shooting pct for A vs B's team
    if (vs_team_s is not None
            and {"player_id", "opponent_team_id", "shots_for", "goals_for"}
                .issubset(vs_team_s.columns)):
        team_id_map = pl.DataFrame({
            "b_team":    pl.Series(list(_TEAM_CODE_TO_ID.keys())),
            "b_team_id": pl.Series(list(_TEAM_CODE_TO_ID.values()), dtype=pl.Int64),
        })
        vs_sh = (
            vs_team_s
            .with_columns(
                (pl.col("goals_for").cast(pl.Float64) /
                 pl.col("shots_for").clip(lower_bound=1).cast(pl.Float64))
                .alias("vs_team_sh_pct")
            )
            .select(["player_id", "opponent_team_id", "vs_team_sh_pct"])
            .rename({"player_id": "player_a_id", "opponent_team_id": "b_team_id"})
        )
        pairs = (
            pairs
            .join(team_id_map, on="b_team", how="left")
            .join(vs_sh, on=["player_a_id", "b_team_id"], how="left")
        )
    else:
        pairs = pairs.with_columns(
            pl.lit(None).cast(pl.Float64).alias("vs_team_sh_pct")
        )

    pairs = pairs.with_columns(
        pl.col("vs_team_sh_pct").fill_null(LEAGUE_AVG_SH_PCT)
    )

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
# MatchupModel
# ---------------------------------------------------------------------------


class MatchupModel:
    """XGBoost matchup efficiency model (Feature 2.3).

    Predicts xGF% for player A's team when on EV ice against player B.
    Blends with historical xGF% for pairs with pair_gp ≥ MIN_PAIR_GP.

    Usage::

        model = MatchupModel()
        metrics = model.fit(
            stints_list=[stints_2122, stints_2223],
            rapm_list=[rapm_2122, rapm_2223],
            vs_team_list=[vs_2122, vs_2223],
            player_stats_list=[stats_2122, stats_2223],
            seasons=[20212022, 20222023],
        )
        preds = model.predict_season(
            stints_df=stints_2324,
            rapm_df=rapm_2324,
            vs_team_df=vs_2324,
            player_stats_df=stats_2324,
            season=20232024,
        )
        model.save(Path("~/.gretzky/models/matchup_v1.pkl"))
        loaded = MatchupModel.load(Path("~/.gretzky/models/matchup_v1.pkl"))
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required. Run: uv add xgboost")
        self._params  = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self._model: xgb.XGBRegressor | None = None
        self._version = MODEL_VERSION

    # --- training ---

    def fit(
        self,
        stints_list:       list[pl.DataFrame],
        rapm_list:         list[pl.DataFrame],
        vs_team_list:      list[pl.DataFrame | None],
        player_stats_list: list[pl.DataFrame | None],
        seasons:           list[int],
        eval_season_idx:   int | None = None,
    ) -> dict[str, float]:
        """Train on historical matchup pairs across multiple seasons.

        Args:
            stints_list:       One build_stints() DataFrame per season.
            rapm_list:         One RAPM DataFrame per season.
            vs_team_list:      One vs_team DataFrame per season (None = skip).
            player_stats_list: One player_stats DataFrame per season (None = skip).
            seasons:           Season integers matching the above lists.
            eval_season_idx:   List index to use as held-out eval set (optional).

        Returns:
            {"rmse": float, "n_pairs": int}
        """
        if not (len(stints_list) == len(rapm_list) == len(seasons)):
            raise ValueError("stints_list, rapm_list, and seasons must have equal length.")

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for stints_df, rapm_df, vs_team_df, stats_df, season in zip(
            stints_list, rapm_list, vs_team_list, player_stats_list, seasons
        ):
            _, opp_df = build_cotoi_from_stints(stints_df, season)
            if opp_df.is_empty():
                continue
            X_i, y_i, _ = _build_matchup_features(
                opp_df, rapm_df, vs_team_df, stats_df, season
            )
            if y_i is not None and len(y_i) > 0:
                X_parts.append(X_i)
                y_parts.append(y_i)

        if not X_parts:
            raise ValueError("No training data produced — check stints/RAPM data.")

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        eval_set: list | None = None
        if eval_season_idx is not None and eval_season_idx < len(X_parts):
            eval_set = [(X_parts[eval_season_idx], y_parts[eval_season_idx])]

        self._model = xgb.XGBRegressor(**self._params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        y_eval  = y_parts[eval_season_idx] if eval_season_idx is not None and eval_set else y_train
        X_eval  = X_parts[eval_season_idx] if eval_season_idx is not None and eval_set else X_train
        preds   = self._model.predict(X_eval)
        rmse    = float(np.sqrt(np.mean((preds - y_eval) ** 2)))

        return {"rmse": rmse, "n_pairs": len(X_train)}

    # --- inference ---

    def predict_season(
        self,
        stints_df:      pl.DataFrame,
        rapm_df:        pl.DataFrame,
        vs_team_df:     pl.DataFrame | None,
        player_stats_df: pl.DataFrame | None,
        season:         int,
    ) -> pl.DataFrame:
        """Generate blended xGF% predictions for all EV opposition pairs in a season.

        Blending rule:
            pair_gp < 10  → model_xgf_pct  (archetype prior)
            pair_gp ≥ 10  → 0.70 × model + 0.30 × historical

        Returns:
            DataFrame matching MATCHUP_PRED_SCHEMA, sorted descending by xgf_pct.

        Raises:
            RuntimeError: if model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        _, opp_df = build_cotoi_from_stints(stints_df, season)
        if opp_df.is_empty():
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in MATCHUP_PRED_SCHEMA.items()}
            )

        X, _, meta = _build_matchup_features(
            opp_df, rapm_df, vs_team_df, player_stats_df, season
        )
        model_preds = np.clip(self._model.predict(X).astype(np.float64), 0.0, 1.0)

        # Pull only the historical xGF% from opp_df for blending; games_together
        # and co_toi_ev are already in meta so we don't re-select them here.
        hist = (
            opp_df.select(["player_a_id", "player_b_id", "xgf_pct"])
                  .rename({"xgf_pct": "_hist"})
        )
        meta = meta.join(hist, on=["player_a_id", "player_b_id"], how="left")

        hist_arr = meta["_hist"].fill_null(LEAGUE_AVG_XGF_PCT).to_numpy()
        gp_arr   = meta["games_together"].to_numpy()

        w_hist      = np.where(gp_arr >= MIN_PAIR_GP, HIST_WEIGHT, 0.0)
        final_preds = (1.0 - w_hist) * model_preds + w_hist * hist_arr

        result = meta.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.Series("model_xgf_pct", model_preds,  dtype=pl.Float64),
            pl.Series("hist_xgf_pct",  hist_arr,      dtype=pl.Float64),
            pl.Series("xgf_pct",       final_preds,   dtype=pl.Float64),
            pl.lit(self._version).alias("model_version"),
        ])

        for col, dtype in MATCHUP_PRED_SCHEMA.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))
        return result.select(list(MATCHUP_PRED_SCHEMA.keys())).sort("xgf_pct", descending=True)

    # --- interpretability ---

    def feature_importance(self) -> dict[str, float]:
        """Return gain-based feature importance sorted descending."""
        if self._model is None:
            raise RuntimeError("Model not trained.")
        raw = self._model.get_booster().get_score(importance_type="gain")
        fi = {
            _FEATURE_NAMES[int(k[1:])]: v
            for k, v in raw.items()
            if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(_FEATURE_NAMES)
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
            "model":         self._model,
            "version":       self._version,
            "params":        self._params,
            "feature_names": _FEATURE_NAMES,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "MatchupModel":
        """Deserialize model from path."""
        payload = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._model   = payload["model"]
        obj._version = payload.get("version", MODEL_VERSION)
        obj._params  = payload.get("params", DEFAULT_XGB_PARAMS)
        return obj


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_qot_qoc(
    df: pl.DataFrame, out_dir: Path, season: int, season_type: str = "regular"
) -> Path:
    """Write QoT/QoC parquet. season_type='playoffs' appends _playoffs suffix."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_playoffs" if season_type == "playoffs" else ""
    path = out_dir / f"qot_qoc_{season}{suffix}.parquet"
    df.write_parquet(path)
    return path


def read_qot_qoc(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    """Read QoT/QoC parquet. If season is None, returns the most recent available."""
    d = Path(data_dir) / "qot_qoc"
    if not d.exists():
        return None
    parquets = sorted(d.glob("qot_qoc_*.parquet"))
    if not parquets:
        return None
    if season is not None:
        p = d / f"qot_qoc_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    return pl.read_parquet(parquets[-1])


def write_matchup_preds(
    df: pl.DataFrame, out_dir: Path, season: int, season_type: str = "regular"
) -> Path:
    """Write matchup predictions parquet. season_type='playoffs' appends suffix."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_playoffs" if season_type == "playoffs" else ""
    path = out_dir / f"matchup_preds_{season}{suffix}.parquet"
    df.write_parquet(path)
    return path


def read_matchup_preds(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    """Read matchup predictions parquet. If season is None, returns most recent."""
    d = Path(data_dir) / "matchup_preds"
    if not d.exists():
        return None
    parquets = sorted(d.glob("matchup_preds_*.parquet"))
    if not parquets:
        return None
    if season is not None:
        p = d / f"matchup_preds_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    return pl.read_parquet(parquets[-1])
