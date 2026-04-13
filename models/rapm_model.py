"""RAPM Model — Feature 2.2.

Prior-informed ridge regression with daisy-chain seasonal chaining.

Each season's posterior becomes the next season's regularization prior — veterans
carry forward strong career signal; rookies start at league-average (0.0).

Algorithm
---------
1. Build stints from shift data + MoneyPuck shot events:
   - Find all time-breakpoints (shift starts/ends) per game
   - For each interval [t_i, t_{i+1}]: which players are on ice + which shots fall inside
   - Aggregate xGF / xGA per stint, tag situation (EV / PP / PK)

2. Construct sparse design matrix (stints × players):
   - +1 if player is on offensive team for that stint
   - -1 if player is on defensive team
   - weight by stint duration in minutes

3. Solve weighted ridge with per-player prior means:
   min Σ wᵢ(yᵢ − Xᵢβ)² + λ Σ (βⱼ − μⱼ)²

   Prior centering trick:
   Let β' = β − μ, y' = y − X μ → standard Ridge (zero prior) on β', then shift back.

4. Daisy chain: season N posterior × PRIOR_WEIGHT → season N+1 prior mean.

Outputs
-------
Per-player per-season parquet with:
  rapm_ev_off, rapm_ev_def, rapm_pp, rapm_pk
  toi_ev, toi_pp, toi_pk, gp, player_name, team, season, model_version
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import polars as pl

try:
    from scipy.sparse import csr_matrix
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION   = "rapm_v1"
MIN_STINT_SECS  = 5      # discard stints shorter than 5 s (line-change noise)
MIN_TOI_SECS    = 900    # player must have ≥ 15 min to appear in output
LAMBDA_BASE     = 8.0    # ridge penalty λ (tuned for ~225 GP/season sample size)
PRIOR_WEIGHT    = 0.35   # fraction of previous posterior carried into next prior
SECS_PER_PERIOD = 1200   # 20 minutes

# NHL team_id (integer from api.nhle.com) → three-letter abbreviation.
# These IDs are stable; new franchises get a new entry here.
_NHL_TEAM_IDS: dict[int, str] = {
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT",
    6: "BOS", 7: "BUF", 8: "MTL", 9: "OTT", 10: "TOR",
    12: "CAR", 13: "FLA", 14: "TBL", 15: "WSH", 16: "CHI",
    17: "DET", 18: "NSH", 19: "STL", 20: "CGY", 21: "COL",
    22: "EDM", 23: "VAN", 24: "ANA", 25: "DAL", 26: "LAK",
    28: "SJS", 29: "CBJ", 30: "MIN", 52: "WPG", 53: "ARI",
    54: "VGK", 55: "SEA", 59: "UTH",
}


class DataMissingWarning(UserWarning):
    """Raised when required input data is absent or insufficient."""


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

RAPM_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "player_id":    pl.Int64,
    "player_name":  pl.Utf8,
    "team":         pl.Utf8,
    "season":       pl.Int64,
    "gp":           pl.Int64,
    "toi_ev":       pl.Float64,
    "toi_pp":       pl.Float64,
    "toi_pk":       pl.Float64,
    "rapm_ev_off":  pl.Float64,
    "rapm_ev_def":  pl.Float64,
    "rapm_pp":      pl.Float64,
    "rapm_pk":      pl.Float64,
    "xgf_60":       pl.Float64,
    "xga_60":       pl.Float64,
    "model_version": pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time_to_secs(time_val: object) -> float:
    """Convert a MoneyPuck ``time`` value to seconds within the period.

    Handles both float/int (already seconds) and "MM:SS" strings.
    """
    if isinstance(time_val, (int, float)):
        return float(time_val)
    s = str(time_val).strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def _game_seconds(period: int, time_secs: float) -> float:
    """Return absolute game seconds from period number and within-period seconds."""
    return (period - 1) * SECS_PER_PERIOD + time_secs


# ---------------------------------------------------------------------------
# Stint construction
# ---------------------------------------------------------------------------

def build_stints(
    shifts_df: pl.DataFrame,
    xg_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build stint-level xGF/xGA aggregations from shift and shot data.

    Args:
        shifts_df: Shift data with columns:
            game_id (Int64 or Utf8), player_id (Int64), team_id (Int64 or Utf8),
            game_seconds_start (Float64), game_seconds_end (Float64)
        xg_df: Shot/xG events with columns:
            game_id, period (Int64), time (time-in-period seconds or "MM:SS"),
            x_goal (Float64), shooting_team (Utf8),
            home_team (Utf8), away_team (Utf8),
            home_skaters (Int64), away_skaters (Int64)

    Returns:
        Polars DataFrame with columns:
            game_id, situation (EV/PP/PK), duration_secs (Float64),
            home_player_ids (List[Int64]), away_player_ids (List[Int64]),
            xgf_home (Float64), xgf_away (Float64)

    Raises:
        ValueError: if required columns are missing from either DataFrame.
        DataMissingWarning: if fewer than 10 games are found.
    """
    _require_cols(shifts_df, {"game_id", "player_id", "team_id",
                               "game_seconds_start", "game_seconds_end"}, "shifts_df")
    _require_cols(xg_df, {"game_id", "period", "time", "x_goal",
                           "shooting_team", "home_team", "away_team",
                           "home_skaters", "away_skaters"}, "xg_df")

    # Normalise game seconds to float
    shifts_df = shifts_df.with_columns([
        pl.col("game_seconds_start").cast(pl.Float64),
        pl.col("game_seconds_end").cast(pl.Float64),
    ])

    # Compute shot game_seconds.
    # MoneyPuck's `time` column is cumulative game-seconds (e.g. period-2 shots
    # have time ≈ 1200–2400), NOT seconds-within-period.  Use it directly.
    time_raw  = xg_df["time"].to_list()
    gsecs_arr = np.array([_parse_time_to_secs(t) for t in time_raw], dtype=np.float64)

    xg_df = xg_df.with_columns(
        pl.Series("game_seconds", gsecs_arr, dtype=pl.Float64)
    )

    # Map game_id to string for consistent joining
    shifts_df = shifts_df.with_columns(pl.col("game_id").cast(pl.Utf8))
    xg_df     = xg_df.with_columns(pl.col("game_id").cast(pl.Utf8))

    # Derive team abbreviation from team_id so we can match shots' team codes.
    # MoneyPuck shots use 3-letter codes ("TBL", "PIT"); NHL shifts use integer team_id (14, 5).
    # If team_id is already a string (abbrev), use it directly.
    if "team_abbrev" not in shifts_df.columns:
        if shifts_df["team_id"].dtype in (pl.Utf8, pl.String):
            # Already a 3-letter abbreviation — use directly
            shifts_df = shifts_df.with_columns(pl.col("team_id").alias("team_abbrev"))
        else:
            _lookup = pl.DataFrame({
                "team_id":    pl.Series(list(_NHL_TEAM_IDS.keys()), dtype=pl.Int64),
                "team_abbrev": pl.Series(list(_NHL_TEAM_IDS.values()), dtype=pl.Utf8),
            })
            shifts_df = shifts_df.with_columns(pl.col("team_id").cast(pl.Int64)).join(
                _lookup, on="team_id", how="left"
            ).with_columns(
                pl.col("team_abbrev").fill_null("")
            )

    game_ids = shifts_df["game_id"].unique().to_list()

    if len(game_ids) < 10:
        warnings.warn(
            f"build_stints: only {len(game_ids)} games found in shifts_df — "
            "RAPM estimates will be unreliable.",
            DataMissingWarning,
            stacklevel=2,
        )

    stints: list[dict] = []

    for gid in game_ids:
        g_shifts = shifts_df.filter(pl.col("game_id") == gid)
        g_shots  = xg_df.filter(pl.col("game_id") == gid)

        # Home / away team identifiers from shots (use first available shot)
        if g_shots.is_empty():
            home_team = ""
            away_team = ""
        else:
            home_team = g_shots["home_team"][0]
            away_team = g_shots["away_team"][0]

        # Build player→team mapping for this game
        player_ids   = g_shifts["player_id"].to_numpy()
        team_abbrevs = g_shifts["team_abbrev"].to_numpy()
        starts       = g_shifts["game_seconds_start"].to_numpy()
        ends         = g_shifts["game_seconds_end"].to_numpy()

        # Collect all breakpoints
        breakpoints = np.unique(np.concatenate([starts, ends]))
        if len(breakpoints) < 2:
            continue

        # Shot arrays for this game
        if not g_shots.is_empty():
            shot_gsecs   = g_shots["game_seconds"].to_numpy()
            shot_xg      = g_shots["x_goal"].to_numpy()
            shot_team    = g_shots["shooting_team"].to_numpy()
            shot_home_sk = g_shots["home_skaters"].to_numpy()
            shot_away_sk = g_shots["away_skaters"].to_numpy()
        else:
            shot_gsecs = np.array([], dtype=np.float64)
            shot_xg    = np.array([], dtype=np.float64)
            shot_team  = np.array([], dtype=object)
            shot_home_sk = np.array([], dtype=np.int64)
            shot_away_sk = np.array([], dtype=np.int64)

        for i in range(len(breakpoints) - 1):
            t_start = breakpoints[i]
            t_end   = breakpoints[i + 1]
            duration = t_end - t_start

            if duration < MIN_STINT_SECS:
                continue

            # Players on ice during [t_start, t_end]
            on_ice_mask   = (starts <= t_start) & (ends >= t_end)
            on_ice_pids   = player_ids[on_ice_mask]
            on_ice_codes  = team_abbrevs[on_ice_mask]

            home_pids = on_ice_pids[on_ice_codes == home_team].tolist()
            away_pids = on_ice_pids[on_ice_codes == away_team].tolist()

            if not home_pids and not away_pids:
                continue

            # Shots within this interval [t_start, t_end)
            shot_mask = (shot_gsecs >= t_start) & (shot_gsecs < t_end)
            if shot_mask.any():
                interval_xg   = shot_xg[shot_mask]
                interval_team = shot_team[shot_mask]
                interval_hsk  = shot_home_sk[shot_mask]
                interval_ask  = shot_away_sk[shot_mask]

                xgf_home = float(interval_xg[interval_team == home_team].sum())
                xgf_away = float(interval_xg[interval_team == away_team].sum())

                # Situation from first shot in interval
                h_sk = int(interval_hsk[0])
                a_sk = int(interval_ask[0])
            else:
                xgf_home = 0.0
                xgf_away = 0.0
                h_sk = 5
                a_sk = 5

            if h_sk == 5 and a_sk == 5:
                situation = "EV"
            elif h_sk > a_sk:
                situation = "PP"   # home on PP
            else:
                situation = "PK"   # home on PK

            stints.append({
                "game_id":         gid,
                "situation":       situation,
                "duration_secs":   float(duration),
                "home_player_ids": home_pids,
                "away_player_ids": away_pids,
                "xgf_home":        xgf_home,
                "xgf_away":        xgf_away,
            })

    if not stints:
        return pl.DataFrame(schema={
            "game_id":         pl.Utf8,
            "situation":       pl.Utf8,
            "duration_secs":   pl.Float64,
            "home_player_ids": pl.List(pl.Int64),
            "away_player_ids": pl.List(pl.Int64),
            "xgf_home":        pl.Float64,
            "xgf_away":        pl.Float64,
        })

    return pl.DataFrame(stints)


def _require_cols(df: pl.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def build_design_matrix(
    stints_df: pl.DataFrame,
    player_index: dict[int, int],
) -> tuple["csr_matrix", np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse ridge-regression inputs from stints.

    For each stint, offensive players get +1, defensive players get -1.
    Returns two separate (X, y, w) triples — one for offensive RAPM (xGF/60),
    one packed together as (X, y_off, y_def, w).

    Args:
        stints_df: Output of build_stints().
        player_index: {player_id → column_index} mapping.

    Returns:
        (X, y_off, y_def, weights) where:
            X       : (n_stints, n_players) sparse csr_matrix
            y_off   : (n_stints,) xGF per 60 for home team
            y_def   : (n_stints,) xGA per 60 for home team
            weights : (n_stints,) stint duration in minutes
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required. Run: uv add scipy")  # pragma: no cover

    n_players = len(player_index)
    n_stints  = len(stints_df)

    rows, cols, vals = [], [], []
    y_off    = np.zeros(n_stints, dtype=np.float64)
    y_def    = np.zeros(n_stints, dtype=np.float64)
    weights  = np.zeros(n_stints, dtype=np.float64)

    durations   = stints_df["duration_secs"].to_numpy()
    xgf_home    = stints_df["xgf_home"].to_numpy()
    xgf_away    = stints_df["xgf_away"].to_numpy()
    home_pids   = stints_df["home_player_ids"].to_list()
    away_pids   = stints_df["away_player_ids"].to_list()

    for i in range(n_stints):
        dur_min = durations[i] / 60.0
        if dur_min <= 0:
            continue

        weights[i] = dur_min
        per60 = 60.0 / dur_min

        # Home offensive perspective: xGF is home shots, xGA is away shots
        y_off[i] = xgf_home[i] * per60
        y_def[i] = xgf_away[i] * per60   # xGA for home = xGF for away

        for pid in home_pids[i]:
            col = player_index.get(int(pid))
            if col is not None:
                rows.append(i)
                cols.append(col)
                vals.append(1.0)

        for pid in away_pids[i]:
            col = player_index.get(int(pid))
            if col is not None:
                rows.append(i)
                cols.append(col)
                vals.append(-1.0)

    X = csr_matrix(
        (vals, (rows, cols)),
        shape=(n_stints, n_players),
        dtype=np.float64,
    )
    return X, y_off, y_def, weights


# ---------------------------------------------------------------------------
# Ridge with prior
# ---------------------------------------------------------------------------

def _ridge_with_prior(
    X: "csr_matrix",
    y: np.ndarray,
    weights: np.ndarray,
    prior_means: dict[int, float],
    player_ids: list[int],
    lambda_: float,
) -> dict[int, float]:
    """Weighted ridge regression with per-player prior means.

    Prior centering trick: subtract prior from target so standard Ridge
    (with zero mean) solves for the deviation from the prior, then shift back.

    Args:
        X:           (n_stints, n_players) sparse design matrix.
        y:           (n_stints,) target values (xGF/60 or xGA/60).
        weights:     (n_stints,) sample weights (stint minutes).
        prior_means: {player_id → prior mean} (0.0 for rookies).
        player_ids:  list of player IDs in column order of X.
        lambda_:     Ridge regularisation strength.

    Returns:
        {player_id → rapm_coefficient}
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. Run: uv add scikit-learn")  # pragma: no cover

    n_players = len(player_ids)
    mu = np.array([prior_means.get(pid, 0.0) for pid in player_ids], dtype=np.float64)

    # Shift target: y' = y − X @ mu
    y_shifted = y - X.dot(mu)

    # Mask out zero-weight rows
    mask = weights > 0
    if mask.sum() < 2:
        return {pid: mu[j] for j, pid in enumerate(player_ids)}

    ridge = Ridge(alpha=lambda_, fit_intercept=False, solver="sparse_cg")
    ridge.fit(X[mask], y_shifted[mask], sample_weight=weights[mask])

    beta = ridge.coef_ + mu  # shift back
    return {pid: float(beta[j]) for j, pid in enumerate(player_ids)}


# ---------------------------------------------------------------------------
# TOI helpers
# ---------------------------------------------------------------------------

def _compute_player_toi(
    shifts_df: pl.DataFrame,
    situation: str | None = None,
    xg_df: pl.DataFrame | None = None,
) -> dict[int, float]:
    """Compute total TOI in seconds per player, optionally filtered by situation.

    If situation is None, returns overall TOI from shifts_df directly.
    If situation is 'EV'/'PP'/'PK', needs xg_df to classify each stint.
    For simplicity in v1, we return overall TOI only (situation split via stints).
    """
    if shifts_df.is_empty():
        return {}
    toi = (
        shifts_df
        .drop_nulls(subset=["player_id"])
        .with_columns(
            (pl.col("game_seconds_end") - pl.col("game_seconds_start")).alias("dur")
        )
        .group_by("player_id")
        .agg(pl.col("dur").sum().alias("toi_secs"))
    )
    return {int(r["player_id"]): float(r["toi_secs"]) for r in toi.to_dicts()}


def _compute_gp(shifts_df: pl.DataFrame) -> dict[int, int]:
    """Games played per player from shifts."""
    if shifts_df.is_empty():
        return {}
    gp = (
        shifts_df
        .drop_nulls(subset=["player_id"])
        .group_by("player_id")
        .agg(pl.col("game_id").n_unique().alias("gp"))
    )
    return {int(r["player_id"]): int(r["gp"]) for r in gp.to_dicts()}


def _compute_player_names(shifts_df: pl.DataFrame, xg_df: pl.DataFrame) -> dict[int, str]:
    """Best-effort player_id → name from shooter_name in xg_df if available."""
    names: dict[int, str] = {}
    if "shooter_name" in xg_df.columns and "shooter_id" in xg_df.columns:
        pairs = (
            xg_df
            .select(["shooter_id", "shooter_name"])
            .drop_nulls()
            .unique(subset=["shooter_id"])
        )
        for r in pairs.to_dicts():
            if r["shooter_id"] is not None:
                names[int(r["shooter_id"])] = str(r["shooter_name"])
    return names


def _compute_player_teams(shifts_df: pl.DataFrame) -> dict[int, str]:
    """Most recent team per player (last game's team_id)."""
    if shifts_df.is_empty():
        return {}
    teams = (
        shifts_df
        .drop_nulls(subset=["player_id"])
        .sort("game_id", descending=True)
        .group_by("player_id")
        .agg(pl.col("team_id").first().alias("team"))
    )
    return {int(r["player_id"]): str(r["team"]) for r in teams.to_dicts()}


def _compute_player_xg_60(ev_stints: pl.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    """Per-player EV xGF/60 and xGA/60 from on-ice stint xG.

    Returns (xgf_60_map, xga_60_map) where each value is per-60-minute rate.
    xGF for a player = their team's xGF while they were on ice.
    xGA for a player = the opposing team's xGF while they were on ice.
    """
    if ev_stints.is_empty():
        return {}, {}

    xgf: dict[int, float] = {}
    xga: dict[int, float] = {}
    toi: dict[int, float] = {}

    for row in ev_stints.to_dicts():
        dur      = row["duration_secs"]
        xgf_home = row["xgf_home"]
        xgf_away = row["xgf_away"]

        for pid in row["home_player_ids"]:
            xgf[pid] = xgf.get(pid, 0.0) + xgf_home
            xga[pid] = xga.get(pid, 0.0) + xgf_away
            toi[pid] = toi.get(pid, 0.0) + dur

        for pid in row["away_player_ids"]:
            xgf[pid] = xgf.get(pid, 0.0) + xgf_away
            xga[pid] = xga.get(pid, 0.0) + xgf_home
            toi[pid] = toi.get(pid, 0.0) + dur

    return (
        {pid: (xgf.get(pid, 0.0) / t * 3600.0) if t > 0 else 0.0 for pid, t in toi.items()},
        {pid: (xga.get(pid, 0.0) / t * 3600.0) if t > 0 else 0.0 for pid, t in toi.items()},
    )


def _compute_player_xga_60(ev_stints: pl.DataFrame) -> dict[int, float]:
    """Legacy wrapper — returns only xGA/60. Use _compute_player_xg_60 for both."""
    _, xga_60 = _compute_player_xg_60(ev_stints)
    return xga_60


# ---------------------------------------------------------------------------
# RAPMModel
# ---------------------------------------------------------------------------

class RAPMModel:
    """Prior-informed RAPM model with cross-season daisy-chain regularisation.

    Usage::

        model = RAPMModel()
        df = model.fit_season(
            season=2025,
            shifts_df=shifts,
            shots_df=shots,
            prior_means={"ev_off": {}, "ev_def": {}, "pp": {}, "pk": {}},
        )
        model.save(Path("~/.gretzky/models/rapm_model.pkl"))

        loaded = RAPMModel.load(Path("~/.gretzky/models/rapm_model.pkl"))
        results = loaded.fit_daisy_chain(
            seasons=[2021, 2022, 2023, 2024, 2025],
            data_loader_fn=my_loader,
        )
    """

    def __init__(self, lambda_: float = LAMBDA_BASE) -> None:
        if not _SCIPY_AVAILABLE:
            raise ImportError("scipy is required. Run: uv add scipy")  # pragma: no cover
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Run: uv add scikit-learn")  # pragma: no cover
        self.lambda_ = lambda_
        self._version = MODEL_VERSION
        self._season_results: dict[int, pl.DataFrame] = {}
        # Cross-season player name lookup — populated by caller for seasons where
        # MoneyPuck shooter_id is null (typically pre-2024). Falls back to "player_{id}".
        self.name_fallback: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Single-season fit
    # ------------------------------------------------------------------

    def fit_season(
        self,
        season: int,
        shifts_df: pl.DataFrame,
        shots_df: pl.DataFrame,
        prior_means: dict[str, dict[int, float]] | None = None,
    ) -> pl.DataFrame:
        """Fit RAPM for one season and return per-player output DataFrame.

        Args:
            season:      Season year (e.g. 2025 = 2024-25 season).
            shifts_df:   Shift data (see build_stints for required columns).
            shots_df:    MoneyPuck shot data (see build_stints for required columns).
            prior_means: Dict with keys "ev_off", "ev_def", "pp", "pk", each mapping
                         player_id → prior mean coefficient. Missing players get 0.0.

        Returns:
            Polars DataFrame conforming to RAPM_OUTPUT_SCHEMA.

        Raises:
            DataMissingWarning: if fewer than 100 qualifying stints are found.
        """
        prior_means = prior_means or {}
        pm_ev_off = prior_means.get("ev_off", {})
        pm_ev_def = prior_means.get("ev_def", {})
        pm_pp     = prior_means.get("pp", {})
        pm_pk     = prior_means.get("pk", {})

        # ---- build stints ----
        stints_df = build_stints(shifts_df, shots_df)

        if len(stints_df) < 10:
            warnings.warn(
                f"Season {season}: only {len(stints_df)} stints found. "
                "Check that shifts_df and shots_df cover the same games.",
                DataMissingWarning,
                stacklevel=2,
            )

        # ---- compute per-player TOI and GP ----
        toi_all = _compute_player_toi(shifts_df)
        gp_all  = _compute_gp(shifts_df)
        names   = _compute_player_names(shifts_df, shots_df)
        teams   = _compute_player_teams(shifts_df)

        # ---- qualifying players (min TOI) ----
        qualifying = {
            pid for pid, toi in toi_all.items()
            if toi >= MIN_TOI_SECS
        }

        if not qualifying:
            warnings.warn(
                f"Season {season}: no players met minimum TOI threshold ({MIN_TOI_SECS}s). "
                "Returning empty DataFrame.",
                DataMissingWarning,
                stacklevel=2,
            )
            return pl.DataFrame(schema=RAPM_OUTPUT_SCHEMA)

        player_ids   = sorted(qualifying)
        player_index = {pid: i for i, pid in enumerate(player_ids)}

        # ---- EV stints ----
        ev_stints = stints_df.filter(pl.col("situation") == "EV")
        pp_stints = stints_df.filter(pl.col("situation") == "PP")
        pk_stints = stints_df.filter(pl.col("situation") == "PK")

        # ---- solve ridge per situation ----
        rapm_ev_off = self._solve(ev_stints, player_index, player_ids, pm_ev_off, "off")
        rapm_ev_def = self._solve(ev_stints, player_index, player_ids, pm_ev_def, "def")
        rapm_pp     = self._solve(pp_stints, player_index, player_ids, pm_pp,     "off")
        rapm_pk     = self._solve(pk_stints, player_index, player_ids, pm_pk,     "def")

        # ---- TOI per situation (from stints duration) ----
        toi_ev = self._toi_from_stints(ev_stints, player_ids)
        toi_pp = self._toi_from_stints(pp_stints, player_ids)
        toi_pk = self._toi_from_stints(pk_stints, player_ids)

        # ---- EV xGF/60 and xGA/60 from on-ice xG exposure ----
        xgf_60_all, xga_60_all = _compute_player_xg_60(ev_stints)

        # ---- assemble output ----
        rows = []
        for pid in player_ids:
            rows.append({
                "player_id":    pid,
                "player_name":  names.get(pid) or self.name_fallback.get(pid, f"player_{pid}"),
                "team":         teams.get(pid, ""),
                "season":       season,
                "gp":           gp_all.get(pid, 0),
                "toi_ev":       toi_ev.get(pid, 0.0),
                "toi_pp":       toi_pp.get(pid, 0.0),
                "toi_pk":       toi_pk.get(pid, 0.0),
                "rapm_ev_off":  rapm_ev_off.get(pid, 0.0),
                "rapm_ev_def":  rapm_ev_def.get(pid, 0.0),
                "rapm_pp":      rapm_pp.get(pid, 0.0),
                "rapm_pk":      rapm_pk.get(pid, 0.0),
                "xgf_60":       xgf_60_all.get(pid, 0.0),
                "xga_60":       xga_60_all.get(pid, 0.0),
                "model_version": self._version,
            })

        result = pl.DataFrame(rows, schema=RAPM_OUTPUT_SCHEMA)
        self._season_results[season] = result
        return result

    def _solve(
        self,
        stints_df: pl.DataFrame,
        player_index: dict[int, int],
        player_ids: list[int],
        prior_means: dict[int, float],
        direction: str,  # "off" or "def"
    ) -> dict[int, float]:
        """Solve ridge for one situation + direction. Returns {player_id: coef}."""
        if stints_df.is_empty():
            return {pid: prior_means.get(pid, 0.0) for pid in player_ids}

        X, y_off, y_def, weights = build_design_matrix(stints_df, player_index)
        y = y_off if direction == "off" else y_def
        return _ridge_with_prior(X, y, weights, prior_means, player_ids, self.lambda_)

    def _toi_from_stints(
        self,
        stints_df: pl.DataFrame,
        player_ids: list[int],
    ) -> dict[int, float]:
        """Sum stint durations for each player across home and away columns."""
        toi: dict[int, float] = {pid: 0.0 for pid in player_ids}
        if stints_df.is_empty():
            return toi

        for row in stints_df.iter_rows(named=True):
            dur = row["duration_secs"]
            for pid in row["home_player_ids"]:
                if pid in toi:
                    toi[pid] += dur
            for pid in row["away_player_ids"]:
                if pid in toi:
                    toi[pid] += dur
        return toi

    # ------------------------------------------------------------------
    # Daisy-chain multi-season fit
    # ------------------------------------------------------------------

    def fit_daisy_chain(
        self,
        seasons: list[int],
        data_loader_fn: Callable[[int], tuple[pl.DataFrame, pl.DataFrame]],
    ) -> dict[int, pl.DataFrame]:
        """Fit RAPM across multiple seasons with daisy-chain priors.

        Args:
            seasons:        Ordered list of seasons to fit (oldest → newest).
            data_loader_fn: Callable(season) → (shifts_df, shots_df).
                            Must raise DataMissingWarning (or return empty DFs)
                            if data is unavailable for that season.

        Returns:
            {season → result DataFrame}

        Notes:
            Season N posterior × PRIOR_WEIGHT becomes season N+1 prior mean.
            Dampening ensures old signal decays rather than compounding indefinitely.
        """
        prior_means: dict[str, dict[int, float]] = {
            "ev_off": {}, "ev_def": {}, "pp": {}, "pk": {}
        }
        results: dict[int, pl.DataFrame] = {}

        for season in seasons:
            shifts_df, shots_df = data_loader_fn(season)

            if shifts_df.is_empty() or shots_df.is_empty():
                warnings.warn(
                    f"Season {season}: empty DataFrames from data_loader_fn — skipping.",
                    DataMissingWarning,
                    stacklevel=2,
                )
                continue

            result = self.fit_season(
                season=season,
                shifts_df=shifts_df,
                shots_df=shots_df,
                prior_means=prior_means,
            )
            results[season] = result

            # Update priors for next season (daisy chain)
            prior_means = _build_next_prior(result, prior_means)

        self._season_results.update(results)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize model to path using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version":        self._version,
            "lambda_":        self.lambda_,
            "season_results": self._season_results,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "RAPMModel":
        """Deserialize model from path."""
        payload = joblib.load(Path(path))
        instance = cls.__new__(cls)
        instance._version        = payload.get("version", MODEL_VERSION)
        instance.lambda_         = payload.get("lambda_", LAMBDA_BASE)
        instance._season_results = payload.get("season_results", {})
        return instance


# ---------------------------------------------------------------------------
# Daisy-chain prior update
# ---------------------------------------------------------------------------

def _build_next_prior(
    result_df: pl.DataFrame,
    prev_prior: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    """Blend current season posterior into next season's prior.

    new_prior[pid] = PRIOR_WEIGHT * posterior[pid] + (1 - PRIOR_WEIGHT) * 0
    """
    new_prior: dict[str, dict[int, float]] = {
        "ev_off": {}, "ev_def": {}, "pp": {}, "pk": {}
    }
    col_map = {
        "ev_off": "rapm_ev_off",
        "ev_def": "rapm_ev_def",
        "pp":     "rapm_pp",
        "pk":     "rapm_pk",
    }
    for key, col in col_map.items():
        if col in result_df.columns:
            for r in result_df.select(["player_id", col]).to_dicts():
                pid = int(r["player_id"])
                posterior = float(r[col])
                new_prior[key][pid] = PRIOR_WEIGHT * posterior
    return new_prior


# ---------------------------------------------------------------------------
# Convenience: load latest rapm parquet
# ---------------------------------------------------------------------------

def read_rapm(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    """Load the most recent RAPM parquet from data_dir/rapm/.

    Args:
        data_dir: Root data directory (e.g. ~/.gretzky/data).
        season:   If provided, load that specific season; else load latest.

    Returns:
        Polars DataFrame or None if not found.
    """
    rapm_dir = Path(data_dir) / "rapm"
    if not rapm_dir.exists():
        return None

    if season is not None:
        path = rapm_dir / f"rapm_{season}.parquet"
        return pl.read_parquet(path) if path.exists() else None

    parquets = sorted(rapm_dir.glob("rapm_*.parquet"))
    if not parquets:
        return None
    return pl.read_parquet(parquets[-1])


def lookup_player(rapm_df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Case-insensitive player name lookup in a RAPM DataFrame."""
    return rapm_df.filter(
        pl.col("player_name").str.to_lowercase() == name.strip().lower()
    )
