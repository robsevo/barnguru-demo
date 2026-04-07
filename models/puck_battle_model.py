"""Puck Battle / Scrum Proxy Model — Feature 2.18.

Estimates team- and individual-level board battle win rates from observable
NHL stats that proxy physical dominance: hits, blocked shots, zone entry
success (carry%), and net-front shot frequency.

Motivation:
    Puck battles in the corners and in front of the net create the possession
    swings that generate high-danger chances. No direct stat captures "won
    board battle", so we construct a composite proxy score from four signals:

    1. hits_per60       — physical presence; players who win battles hit more
    2. blocks_per60     — defensive battle engagement (blocked attempts)
    3. carry_entry_pct  — controlled zone entries / total entries (carry = won
                          entry battle); source: EDGE skating data or MoneyPuck
    4. net_front_pct    — shots taken from net-front area (<20ft, centre slot);
                          requires going through / absorbing defenders

Battle score formula (all components z-scored across the player pool):
    battle_score = 0.30·z(hits/60)
                 + 0.25·z(blocks/60)
                 + 0.30·z(carry_entry_pct)
                 + 0.15·z(net_front_pct)

    controlled_entry_prob = sigmoid(battle_score)  ∈ (0, 1)

Team-level score is TOI-weighted average of player scores for the team's
forwards + defence, then converted to controlled_entry_prob.

Feeds:
    - P(controlled zone entry) in the Rust game state machine (3.2)
    - Coaching line-matching context for 4.x features
    - General team physical profile for fatigue indexing

Outputs
-------
``puck_battle_{season}.parquet``   — one row per player-season:
    player_id, team, season, hits_per60, blocks_per60,
    carry_entry_pct, net_front_pct, battle_score,
    battle_percentile, controlled_entry_prob, model_version

``team_battle_{season}.parquet``   — one row per team-season:
    team, season, team_battle_score, controlled_entry_prob,
    n_players, model_version
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "battle_v1"

# Net-front zone: shots within this arc are counted as net-front
# Coordinates: x_adjusted ≥ NET_FRONT_X_MIN, |y_adjusted| ≤ NET_FRONT_Y_MAX
# (MoneyPuck x_adjusted: 0=own net, 100=opponent's net; y: −42.5 to +42.5)
NET_FRONT_X_MIN = 78.0   # roughly 11 ft from goal line
NET_FRONT_Y_MAX = 12.0   # within the slot width

# Battle score weights — must sum to 1.0
BATTLE_WEIGHTS = {
    "hits_per60":       0.30,
    "blocks_per60":     0.25,
    "carry_entry_pct":  0.30,
    "net_front_pct":    0.15,
}

# Minimum TOI (seconds) for a player to be included
MIN_TOI_SEC = 300  # 5 minutes EV TOI


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

PLAYER_BATTLE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":           pl.Int64,
    "team":                pl.Utf8,
    "season":              pl.Int64,
    "toi_ev":              pl.Float64,
    "hits_per60":          pl.Float64,
    "blocks_per60":        pl.Float64,
    "carry_entry_pct":     pl.Float64,
    "net_front_pct":       pl.Float64,
    "battle_score":        pl.Float64,
    "battle_percentile":   pl.Float64,
    "controlled_entry_prob": pl.Float64,
    "model_version":       pl.Utf8,
}

TEAM_BATTLE_SCHEMA: dict[str, pl.DataType] = {
    "team":                    pl.Utf8,
    "season":                  pl.Int64,
    "team_battle_score":       pl.Float64,
    "controlled_entry_prob":   pl.Float64,
    "n_players":               pl.Int64,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score an array; returns zeros when std < 1e-9."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0 or np.std(valid) < 1e-9:
        return np.zeros_like(arr)
    mean = float(np.mean(valid))
    std  = float(np.std(valid))
    return (arr - mean) / std


def compute_net_front_pct(shots_df: pl.DataFrame) -> pl.DataFrame:
    """Compute net-front shot percentage per player from a shots DataFrame.

    Args:
        shots_df: Shot-level DataFrame.  Expected columns:
                  shooter_id, x_adjusted (or x_fixed), y_adjusted (or y_fixed).

    Returns:
        DataFrame with player_id and net_front_pct.
        If coordinate columns are absent, returns net_front_pct = NaN for all.
    """
    if "shooter_id" not in shots_df.columns:
        return pl.DataFrame({"player_id": pl.Series([], dtype=pl.Int64),
                             "net_front_pct": pl.Series([], dtype=pl.Float64)})

    # Resolve coordinate column names
    x_col = next((c for c in ["x_adjusted", "x_fixed", "xCordAdjusted"] if c in shots_df.columns), None)
    y_col = next((c for c in ["y_adjusted", "y_fixed", "yCordAdjusted"] if c in shots_df.columns), None)

    if x_col is None or y_col is None:
        # No coordinates — return NaN for all players
        result = (
            shots_df
            .filter(pl.col("shooter_id").is_not_null())
            .group_by("shooter_id")
            .agg(pl.col("shooter_id").count().alias("n_shots"))
            .select([
                pl.col("shooter_id").cast(pl.Int64).alias("player_id"),
                pl.lit(float("nan")).alias("net_front_pct"),
            ])
        )
        return result

    net_front_flag = (
        (pl.col(x_col) >= NET_FRONT_X_MIN) &
        (pl.col(y_col).abs() <= NET_FRONT_Y_MAX)
    )

    result = (
        shots_df
        .filter(pl.col("shooter_id").is_not_null())
        .with_columns(net_front_flag.alias("_is_net_front"))
        .group_by("shooter_id")
        .agg([
            pl.col("shooter_id").count().alias("n_shots"),
            pl.col("_is_net_front").sum().alias("n_net_front"),
        ])
        .with_columns(
            (pl.col("n_net_front") / pl.col("n_shots").clip(1)).alias("net_front_pct")
        )
        .select([
            pl.col("shooter_id").cast(pl.Int64).alias("player_id"),
            pl.col("net_front_pct"),
        ])
    )
    return result


def compute_battle_scores(
    player_df: pl.DataFrame,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Compute battle scores from a player-level DataFrame.

    Args:
        player_df: Must contain columns:
                   player_id, hits_per60, blocks_per60,
                   carry_entry_pct, net_front_pct.
                   Missing columns default to 0.0.
        weights:   Override BATTLE_WEIGHTS if provided.

    Returns:
        player_df with added columns:
        battle_score, battle_percentile, controlled_entry_prob.
    """
    w = weights or BATTLE_WEIGHTS
    n = len(player_df)
    if n == 0:
        return player_df.with_columns([
            pl.lit(0.0).alias("battle_score"),
            pl.lit(50.0).alias("battle_percentile"),
            pl.lit(0.5).alias("controlled_entry_prob"),
        ])

    # Extract component arrays (fill missing columns with zeros)
    def _col_or_zeros(col: str) -> np.ndarray:
        if col in player_df.columns:
            arr = player_df[col].to_numpy(allow_copy=True).astype(np.float64)
        else:
            arr = np.zeros(n)
        # Replace NaN with column mean (fallback to 0)
        valid = arr[~np.isnan(arr)]
        fill  = float(np.mean(valid)) if len(valid) > 0 else 0.0
        arr[np.isnan(arr)] = fill
        return arr

    hits_arr    = _col_or_zeros("hits_per60")
    blocks_arr  = _col_or_zeros("blocks_per60")
    carry_arr   = _col_or_zeros("carry_entry_pct")
    netfront_arr = _col_or_zeros("net_front_pct")

    # Z-score each component
    z_hits    = _zscore(hits_arr)
    z_blocks  = _zscore(blocks_arr)
    z_carry   = _zscore(carry_arr)
    z_netfront = _zscore(netfront_arr)

    # Weighted composite
    battle_score = (
        w.get("hits_per60",      0.30) * z_hits    +
        w.get("blocks_per60",    0.25) * z_blocks  +
        w.get("carry_entry_pct", 0.30) * z_carry   +
        w.get("net_front_pct",   0.15) * z_netfront
    )

    # Percentile rank (0–100)
    n_valid = np.sum(~np.isnan(battle_score))
    if n_valid > 0:
        ranks = np.argsort(np.argsort(battle_score))
        percentile = (ranks / max(n - 1, 1)) * 100.0
    else:
        percentile = np.full(n, 50.0)

    controlled_entry_prob = _sigmoid(battle_score)

    return player_df.with_columns([
        pl.Series("battle_score",          battle_score.tolist(),          dtype=pl.Float64),
        pl.Series("battle_percentile",     percentile.tolist(),            dtype=pl.Float64),
        pl.Series("controlled_entry_prob", controlled_entry_prob.tolist(), dtype=pl.Float64),
    ])


# ---------------------------------------------------------------------------
# PuckBattleModel — Feature 2.18
# ---------------------------------------------------------------------------

class PuckBattleModel:
    """Puck battle / scrum proxy model (Feature 2.18).

    Builds per-player and per-team board battle win rates from observable
    NHL stats (hits, blocks, carry entry%, net-front shot%).

    Usage::

        model = PuckBattleModel()

        # Load player stats (hits, blocks, toi) and shots (for net-front):
        player_df = model.fit(
            player_stats_df=player_stats,
            shots_df=shots,
            season=2025,
        )

        # Team-level controlled entry probability:
        team_df = model.team_battle_ratings(player_df)

        # Save / load:
        model.save(path)
        model2 = PuckBattleModel.load(path)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {**BATTLE_WEIGHTS}
        self._version = MODEL_VERSION
        # Store season-level normalisation stats for consistent inference
        self._fit_stats: dict[int, dict[str, dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # Fit / compute
    # ------------------------------------------------------------------

    def fit(
        self,
        player_stats_df: pl.DataFrame,
        season: int,
        shots_df: pl.DataFrame | None = None,
        carry_entry_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute per-player battle scores for a season.

        Args:
            player_stats_df: Per-player stats.  Required columns:
                             player_id.  Preferred: team, hits, blocked_shots,
                             toi_ev (or toi) in seconds.
                             Optional: carry_entry_pct (from EDGE/MoneyPuck).
            season:          NHL season start-year.
            shots_df:        Shot-level DataFrame to compute net_front_pct.
                             If None, net_front_pct defaults to NaN → 0 fill.
            carry_entry_df:  Optional separate carry-entry DataFrame with
                             player_id and carry_entry_pct columns.

        Returns:
            DataFrame matching PLAYER_BATTLE_SCHEMA (plus any extra cols
            from player_stats_df).
        """
        if "player_id" not in player_stats_df.columns:
            raise ValueError("player_stats_df must have 'player_id' column.")

        df = player_stats_df.clone()

        # --- Resolve TOI ---
        toi_col = next((c for c in ["toi_ev", "toi", "TOI"] if c in df.columns), None)
        if toi_col is None:
            warnings.warn(
                "No TOI column found in player_stats_df. Defaulting to 1800 sec.",
                DataMissingWarning, stacklevel=2,
            )
            df = df.with_columns(pl.lit(1800.0).alias("toi_ev"))
            toi_col = "toi_ev"

        if toi_col != "toi_ev":
            df = df.rename({toi_col: "toi_ev"})
        df = df.with_columns(pl.col("toi_ev").cast(pl.Float64))

        # Filter minimum TOI
        n_before = len(df)
        df = df.filter(pl.col("toi_ev") >= MIN_TOI_SEC)
        if len(df) < n_before:
            warnings.warn(
                f"Filtered {n_before - len(df)} players below MIN_TOI_SEC={MIN_TOI_SEC}.",
                DataMissingWarning, stacklevel=2,
            )

        if len(df) == 0:
            warnings.warn(
                f"No players meet minimum TOI threshold for season {season}.",
                DataMissingWarning, stacklevel=2,
            )
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYER_BATTLE_SCHEMA.items()}
            )

        # --- hits_per60 ---
        hits_col = next((c for c in ["hits", "Hits", "hits_ev"] if c in df.columns), None)
        if hits_col:
            df = df.with_columns(
                (pl.col(hits_col).cast(pl.Float64) / pl.col("toi_ev").clip(1) * 3600.0)
                .alias("hits_per60")
            )
        else:
            df = df.with_columns(pl.lit(float("nan")).alias("hits_per60"))

        # --- blocks_per60 ---
        block_col = next((c for c in ["blocked_shots", "blocked", "Blocked", "blocks"] if c in df.columns), None)
        if block_col:
            df = df.with_columns(
                (pl.col(block_col).cast(pl.Float64) / pl.col("toi_ev").clip(1) * 3600.0)
                .alias("blocks_per60")
            )
        else:
            df = df.with_columns(pl.lit(float("nan")).alias("blocks_per60"))

        # --- carry_entry_pct ---
        if "carry_entry_pct" not in df.columns:
            if carry_entry_df is not None and "carry_entry_pct" in carry_entry_df.columns:
                carry_slim = carry_entry_df.select(["player_id", "carry_entry_pct"])
                df = df.join(carry_slim, on="player_id", how="left")
            else:
                df = df.with_columns(pl.lit(float("nan")).alias("carry_entry_pct"))

        # --- net_front_pct ---
        if shots_df is not None and len(shots_df) > 0:
            netfront_df = compute_net_front_pct(shots_df)
            df = df.join(netfront_df, on="player_id", how="left")
        else:
            df = df.with_columns(pl.lit(float("nan")).alias("net_front_pct"))

        # --- Compute battle scores ---
        df = compute_battle_scores(df, weights=self._weights)
        df = df.with_columns([
            pl.lit(season).cast(pl.Int64).alias("season"),
            pl.lit(self._version).alias("model_version"),
        ])

        # Ensure schema columns exist
        for col, dtype in PLAYER_BATTLE_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

        return df.select(
            [c for c in PLAYER_BATTLE_SCHEMA.keys() if c in df.columns]
            + [c for c in df.columns if c not in PLAYER_BATTLE_SCHEMA]
        )

    def team_battle_ratings(
        self,
        player_battle_df: pl.DataFrame,
        season: int | None = None,
    ) -> pl.DataFrame:
        """Compute team-level battle ratings from player scores.

        Args:
            player_battle_df: Output of fit() — one row per player.
            season:           Season integer (used if not already in df).

        Returns:
            DataFrame matching TEAM_BATTLE_SCHEMA.
        """
        if "team" not in player_battle_df.columns or "battle_score" not in player_battle_df.columns:
            warnings.warn(
                "player_battle_df missing 'team' or 'battle_score' column.",
                DataMissingWarning, stacklevel=2,
            )
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in TEAM_BATTLE_SCHEMA.items()}
            )

        toi_col = "toi_ev" if "toi_ev" in player_battle_df.columns else None

        if toi_col:
            team_df = (
                player_battle_df
                .group_by("team")
                .agg([
                    (pl.col("battle_score") * pl.col(toi_col)).sum().alias("_weighted_score"),
                    pl.col(toi_col).sum().alias("_total_toi"),
                    pl.col("player_id").count().alias("n_players"),
                ])
                .with_columns(
                    (pl.col("_weighted_score") / pl.col("_total_toi").clip(1e-9))
                    .alias("team_battle_score")
                )
            )
        else:
            team_df = (
                player_battle_df
                .group_by("team")
                .agg([
                    pl.col("battle_score").mean().alias("team_battle_score"),
                    pl.col("player_id").count().alias("n_players"),
                ])
            )

        # Convert to controlled_entry_prob
        scores = team_df["team_battle_score"].to_numpy(allow_copy=True).astype(np.float64)
        cep = _sigmoid(scores)

        team_df = team_df.with_columns([
            pl.Series("controlled_entry_prob", cep.tolist(), dtype=pl.Float64),
            pl.lit(season).cast(pl.Int64).alias("season") if season is not None
            else pl.lit(None).cast(pl.Int64).alias("season"),
            pl.lit(self._version).alias("model_version"),
        ])

        for col, dtype in TEAM_BATTLE_SCHEMA.items():
            if col not in team_df.columns:
                team_df = team_df.with_columns(pl.lit(None).cast(dtype).alias(col))

        return team_df.select(list(TEAM_BATTLE_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "weights":   self._weights,
            "version":   self._version,
            "fit_stats": self._fit_stats,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PuckBattleModel":
        payload = joblib.load(Path(path))
        obj = cls(weights=payload.get("weights", BATTLE_WEIGHTS))
        obj._version   = payload.get("version",   MODEL_VERSION)
        obj._fit_stats = payload.get("fit_stats", {})
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_player_battle(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write player battle ratings to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"puck_battle_{season}.parquet"
    for col, dtype in PLAYER_BATTLE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select([c for c in PLAYER_BATTLE_SCHEMA.keys() if c in df.columns]).write_parquet(path)
    return path


def write_team_battle(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write team battle ratings to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"team_battle_{season}.parquet"
    for col, dtype in TEAM_BATTLE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(TEAM_BATTLE_SCHEMA.keys())).write_parquet(path)
    return path
