"""Short-Burst Hot Hand Signal — Feature 2.28.

Distinct from EWMA (2.13) which is a 20-game trend. This is a 5-game burst:

    5-game rolling window → goals scored vs. xG expected
    Z-score vs. player's full-season variance
    Output: hot_hand_score ∈ [-2.0, +2.0] standard deviations from mean

Signal is applied as a multiplicative boost (~2–4% weight) to shooting
probability in the Rust shot-generation engine (5.4).

Bayesian shrinkage:
    Players with more career GP receive stronger signal weight.
    ``shrinkage_factor = min(career_gp / FULL_SIGNAL_GP, 1.0)``
    Rookies (0 career GP) default toward zero; veterans get full weight.

Complementary pair with Feature 2.13 (EWMA):
    EWMA  → slow 20-game trend  (λ=0.96, half-life ≈ 17 games)
    HotHand → fast 5-game burst (raw Z-score, Bayesian-shrunk)

Outputs
-------
``hot_hand_{season}.parquet`` — one row per (player, game):
    player_id, game_id, game_number, season,
    goals_5g, xg_5g, goals_vs_xg_5g,
    season_mean_gvx, season_std_gvx,
    raw_z, shrinkage_factor, hot_hand_score,
    model_version

``hot_hand_summary_{season}.parquet`` — one row per player (latest snapshot):
    player_id, player_name, season, games_played,
    goals_5g, xg_5g, goals_vs_xg_5g,
    hot_hand_score, shrinkage_factor,
    season_mean_gvx, season_std_gvx,
    model_version

Usage::

    from models.hot_hand_model import HotHandModel

    model = HotHandModel()
    game_df  = model.compute(player_game_df, season=2025)
    summ_df  = model.summary(player_game_df, season=2025)

    # At game time — single player lookup:
    score = model.get_score(player_id=8481600)
    # score ∈ [-2.0, +2.0]; 0.0 = no burst signal yet
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION      = "hot_hand_v1"

BURST_WINDOW       = 5           # games in the rolling window
FULL_SIGNAL_GP     = 82          # career GP needed for full shrinkage weight (≈ 1 NHL season)
MIN_SEASON_GP      = 5           # minimum games before computing a non-zero score
SCORE_MIN          = -2.0        # output clamp floor
SCORE_MAX          = +2.0        # output clamp ceiling

# Shooting probability multiplier weight applied in Rust engine (documentation only)
# Actual weight calibrated during Phase 5 engine integration
ENGINE_WEIGHT      = 0.03        # ~3% weight on shooting probability; midpoint of [0.02, 0.04]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

HOT_HAND_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "game_id":          pl.Int64,
    "game_number":      pl.Int64,
    "season":           pl.Int64,
    "goals_5g":         pl.Float64,   # goals in the last BURST_WINDOW games
    "xg_5g":            pl.Float64,   # expected goals in the last BURST_WINDOW games
    "goals_vs_xg_5g":   pl.Float64,   # goals_5g − xg_5g
    "season_mean_gvx":  pl.Float64,   # player's full-season mean of (goals − xg) per game
    "season_std_gvx":   pl.Float64,   # player's full-season std of (goals − xg) per game
    "raw_z":            pl.Float64,   # z-score of 5g burst vs season distribution
    "shrinkage_factor": pl.Float64,   # Bayesian shrinkage in [0.0, 1.0]
    "hot_hand_score":   pl.Float64,   # raw_z × shrinkage_factor, clamped to [-2, +2]
    "model_version":    pl.Utf8,
}

HOT_HAND_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "player_name":      pl.Utf8,
    "season":           pl.Int64,
    "games_played":     pl.Int64,
    "goals_5g":         pl.Float64,
    "xg_5g":            pl.Float64,
    "goals_vs_xg_5g":   pl.Float64,
    "hot_hand_score":   pl.Float64,
    "shrinkage_factor": pl.Float64,
    "season_mean_gvx":  pl.Float64,
    "season_std_gvx":   pl.Float64,
    "model_version":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Core computation helpers (pure NumPy, single player)
# ---------------------------------------------------------------------------

def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over *window* games (right-aligned, no min_periods guard).

    For the first (window-1) games we use the cumulative sum so the scorer
    still produces a finite output — those games are gated by MIN_SEASON_GP
    before being included in the final output.
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    cumsum = np.cumsum(np.where(np.isnan(values), 0.0, values))
    for i in range(n):
        if i < window:
            result[i] = cumsum[i]
        else:
            result[i] = cumsum[i] - cumsum[i - window]
    return result


def _shrinkage(career_gp: int, full_signal_gp: int = FULL_SIGNAL_GP) -> float:
    """Bayesian shrinkage factor ∈ [0.0, 1.0].

    career_gp == 0  → 0.0 (rookie, no signal)
    career_gp ≥ 82  → 1.0 (veteran, full signal)
    """
    return min(float(career_gp) / float(full_signal_gp), 1.0)


def compute_hot_hand_series(
    goals: np.ndarray,
    xg_expected: np.ndarray,
    career_gp: int = 0,
    window: int = BURST_WINDOW,
) -> dict[str, np.ndarray]:
    """Compute hot-hand arrays for a single player.

    Args:
        goals:       Per-game goals (oldest first).  NaN-safe.
        xg_expected: Per-game expected goals (oldest first).  NaN-safe.
        career_gp:   Total career games played at start of this season
                     (used for Bayesian shrinkage weight).
        window:      Rolling window size (default: 5 games).

    Returns:
        Dict of named arrays, each of length len(goals):
            goals_5g, xg_5g, goals_vs_xg_5g,
            season_mean_gvx, season_std_gvx,
            raw_z, shrinkage_factor, hot_hand_score
    """
    n = len(goals)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return {k: empty for k in (
            "goals_5g", "xg_5g", "goals_vs_xg_5g",
            "season_mean_gvx", "season_std_gvx",
            "raw_z", "shrinkage_factor", "hot_hand_score",
        )}

    goals_f = np.where(np.isnan(goals), 0.0, goals.astype(np.float64))
    xg_f    = np.where(np.isnan(xg_expected), 0.0, xg_expected.astype(np.float64))

    # Per-game goals-vs-xG (the underlying signal)
    gvx = goals_f - xg_f

    # Full-season moments (constant for every row — shrinkage anchored to season)
    valid_gvx   = gvx[~np.isnan(gvx)]
    season_mean = float(np.mean(valid_gvx)) if len(valid_gvx) > 0 else 0.0
    season_std  = float(np.std(valid_gvx))  if len(valid_gvx) > 1 else 1.0
    if season_std < 1e-9:
        season_std = 1.0   # prevent division by zero for players who never shoot

    goals_5g = _rolling_sum(goals_f, window)
    xg_5g    = _rolling_sum(xg_f,    window)
    gvx_5g   = goals_5g - xg_5g

    # Z-score: how far is the 5g burst from the player's own seasonal average?
    # We normalise the 5g window total against window × season_std so the z-scale
    # is in per-game standard deviation units, matching the [-2.0, +2.0] spec.
    raw_z = (gvx_5g - window * season_mean) / (np.sqrt(window) * season_std)

    # Bayesian shrinkage
    sf    = _shrinkage(career_gp)
    sf_arr = np.full(n, sf, dtype=np.float64)

    # Gate: no signal before MIN_SEASON_GP games
    hot_hand = raw_z * sf_arr
    hot_hand[:MIN_SEASON_GP] = 0.0

    # Clamp to [-2.0, +2.0]
    hot_hand = np.clip(hot_hand, SCORE_MIN, SCORE_MAX)
    raw_z_gated = raw_z.copy()
    raw_z_gated[:MIN_SEASON_GP] = 0.0

    return {
        "goals_5g":        goals_5g,
        "xg_5g":           xg_5g,
        "goals_vs_xg_5g":  gvx_5g,
        "season_mean_gvx": np.full(n, season_mean, dtype=np.float64),
        "season_std_gvx":  np.full(n, season_std,  dtype=np.float64),
        "raw_z":           raw_z_gated,
        "shrinkage_factor": sf_arr,
        "hot_hand_score":  hot_hand,
    }


# ---------------------------------------------------------------------------
# HotHandModel — Feature 2.28
# ---------------------------------------------------------------------------

class HotHandModel:
    """Short-burst hot hand signal model (Feature 2.28).

    Produces a 5-game rolling burst Z-score with Bayesian shrinkage by
    career games played.  Complements the 20-game EWMA form metric (2.13).

    Usage::

        model = HotHandModel()
        game_df  = model.compute(player_game_df, season=2025)
        summ_df  = model.summary(player_game_df, season=2025)

        # After compute/summary has been called:
        score = model.get_score(player_id=8481600)
    """

    def __init__(self, window: int = BURST_WINDOW) -> None:
        self._window  = window
        self._version = MODEL_VERSION
        # Latest scores keyed by player_id; populated by compute()
        self._latest_scores: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Game-by-game computation across all players
    # ------------------------------------------------------------------

    def compute(
        self,
        player_game_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Compute per-game hot-hand score for every player.

        Args:
            player_game_df: One row per (player, game), sorted by game order.
                Required columns: player_id, game_id, goals, xg_expected.
                Optional columns: career_gp (int — total GP before this season;
                    if absent, all players are treated as rookies with gp=0).

        Returns:
            DataFrame matching HOT_HAND_SCHEMA, sorted by (player_id, game_number).
        """
        required = {"player_id", "game_id", "goals", "xg_expected"}
        missing  = required - set(player_game_df.columns)
        if missing:
            raise ValueError(f"player_game_df missing columns: {sorted(missing)}")

        has_career_gp = "career_gp" in player_game_df.columns

        out_rows: list[dict] = []
        self._latest_scores  = {}

        player_ids = player_game_df["player_id"].unique().to_list()

        for pid in player_ids:
            pdata = (
                player_game_df
                .filter(pl.col("player_id") == pid)
                .sort("game_id")
            )

            n         = len(pdata)
            goals_arr = pdata["goals"].cast(pl.Float64).to_numpy()
            xg_arr    = pdata["xg_expected"].cast(pl.Float64).to_numpy()
            game_ids  = pdata["game_id"].cast(pl.Int64).to_numpy()

            career_gp = 0
            if has_career_gp:
                # Take the first non-null value (constant per player per season)
                gp_col = pdata["career_gp"].drop_nulls()
                if len(gp_col) > 0:
                    career_gp = int(gp_col[0])

            arrays = compute_hot_hand_series(
                goals_arr, xg_arr, career_gp, self._window
            )

            for i in range(n):
                out_rows.append({
                    "player_id":        int(pid),
                    "game_id":          int(game_ids[i]),
                    "game_number":      i + 1,
                    "season":           season,
                    "goals_5g":         float(arrays["goals_5g"][i]),
                    "xg_5g":            float(arrays["xg_5g"][i]),
                    "goals_vs_xg_5g":   float(arrays["goals_vs_xg_5g"][i]),
                    "season_mean_gvx":  float(arrays["season_mean_gvx"][i]),
                    "season_std_gvx":   float(arrays["season_std_gvx"][i]),
                    "raw_z":            float(arrays["raw_z"][i]),
                    "shrinkage_factor": float(arrays["shrinkage_factor"][i]),
                    "hot_hand_score":   float(arrays["hot_hand_score"][i]),
                    "model_version":    self._version,
                })

            # Cache latest score for fast lookup
            if n > 0:
                self._latest_scores[int(pid)] = float(arrays["hot_hand_score"][-1])

        if not out_rows:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in HOT_HAND_SCHEMA.items()}
            )

        result = pl.DataFrame(out_rows)

        # Cast to schema types
        for col, dtype in HOT_HAND_SCHEMA.items():
            if col in result.columns:
                result = result.with_columns(pl.col(col).cast(dtype))
            else:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))

        return result.select(list(HOT_HAND_SCHEMA.keys())).sort(["player_id", "game_number"])

    # ------------------------------------------------------------------
    # Per-player summary (latest snapshot)
    # ------------------------------------------------------------------

    def summary(
        self,
        player_game_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Return one-row-per-player with the latest hot-hand snapshot.

        Args:
            player_game_df: Same input as compute().
                Optional extra column: player_name.

        Returns:
            DataFrame matching HOT_HAND_SUMMARY_SCHEMA, sorted by
            hot_hand_score descending.
        """
        game_df = self.compute(player_game_df, season)

        if game_df.is_empty():
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in HOT_HAND_SUMMARY_SCHEMA.items()}
            )

        # Latest row per player (highest game_number)
        latest = game_df.group_by("player_id").agg([
            pl.col("game_number").max().alias("games_played"),
            pl.col("goals_5g").last().alias("goals_5g"),
            pl.col("xg_5g").last().alias("xg_5g"),
            pl.col("goals_vs_xg_5g").last().alias("goals_vs_xg_5g"),
            pl.col("hot_hand_score").last().alias("hot_hand_score"),
            pl.col("shrinkage_factor").last().alias("shrinkage_factor"),
            pl.col("season_mean_gvx").last().alias("season_mean_gvx"),
            pl.col("season_std_gvx").last().alias("season_std_gvx"),
        ])

        # Join player_name if available
        if "player_name" in player_game_df.columns:
            names = player_game_df.group_by("player_id").agg(
                pl.col("player_name").first()
            )
            latest = latest.join(names, on="player_id", how="left")
        else:
            latest = latest.with_columns(pl.lit("").alias("player_name"))

        latest = latest.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit(self._version).alias("model_version"),
        ])

        # Enforce schema
        for col, dtype in HOT_HAND_SUMMARY_SCHEMA.items():
            if col not in latest.columns:
                latest = latest.with_columns(pl.lit(None).cast(dtype).alias(col))

        return (
            latest
            .select(list(HOT_HAND_SUMMARY_SCHEMA.keys()))
            .sort("hot_hand_score", descending=True)
        )

    # ------------------------------------------------------------------
    # Single-player fast lookup (post-compute)
    # ------------------------------------------------------------------

    def get_score(self, player_id: int) -> float:
        """Return the cached latest hot-hand score for *player_id*.

        Must call ``compute()`` or ``summary()`` first.

        Returns:
            Float in ``[SCORE_MIN, SCORE_MAX]``.  0.0 if player not found.
        """
        return self._latest_scores.get(player_id, 0.0)

    def get_scores_batch(self, player_ids: list[int]) -> dict[int, float]:
        """Return {player_id: hot_hand_score} for a batch of players.

        Unknown players map to 0.0.
        """
        return {pid: self._latest_scores.get(pid, 0.0) for pid in player_ids}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def window(self) -> int:
        return self._window

    @property
    def version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_hot_hand(
    game_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_dir: Path,
    season: int,
) -> tuple[Path, Path]:
    """Write per-game and summary DataFrames to parquet.

    Args:
        game_df:    Output of ``HotHandModel.compute()``.
        summary_df: Output of ``HotHandModel.summary()``.
        output_dir: Directory to write into (created if absent).
        season:     Season year used in filenames.

    Returns:
        Tuple of (game_path, summary_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_game    = output_dir / f"hot_hand_{season}.parquet"
    p_summary = output_dir / f"hot_hand_summary_{season}.parquet"

    game_df.write_parquet(p_game)
    summary_df.write_parquet(p_summary)

    return p_game, p_summary


def read_hot_hand_summary(data_dir: Path, season: int) -> pl.DataFrame:
    """Read per-player hot-hand summary parquet."""
    path = Path(data_dir) / "hot_hand" / f"hot_hand_summary_{season}.parquet"
    return pl.read_parquet(path)
