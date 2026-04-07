"""Player Skating Baseline Profile — Feature 2.20.

Per-player EDGE skating baselines when fully rested (FI ≈ 0.0–0.15).
Built from historical EDGE data conditioned on rest_days > 1 (or, when
rest data is unavailable, from all regular-season games as a conservative
season-aggregate baseline).

These are *ceiling* values — what this player produces when fully recovered.
The fatigue degradation model (Feature 3.21) will predict how far below
these ceilings a given FI score pushes the player in-game.

Model design:
    For each player across up to 3 seasons of EDGE data:
        1. Collect game-level rows where rest_days > MIN_REST_DAYS (default 1).
        2. Compute mean and std-dev for each metric (avg_speed_kmh,
           max_speed_kmh, distance_per_game_km, zone_time_pct_oz,
           zone_time_pct_dz, zone_time_pct_nz).
        3. Clip implausible outliers (speed > MAX_SPEED_KMH = 40, distance
           > MAX_DIST_KM = 10).
        4. Require MIN_GAMES_RESTED (default 5) rested-game observations;
           if fewer, fall back to season-aggregate mean.
        5. Attach a confidence flag: "rested" (used rested filter),
           "aggregate" (fell back), or "insufficient" (< MIN_GAMES_TOTAL).

Inputs
------
EDGE skating DataFrame matching EDGE_SKATING_SCHEMA from data/edge_parser.py:
    player_id, player_name, team, season, game_type, games_played,
    avg_speed_kmh, max_speed_kmh, total_distance_km, distance_per_game_km,
    zone_time_pct_oz, zone_time_pct_dz, zone_time_pct_nz

Optional: game-level EDGE data with additional column ``rest_days`` (int:
    number of rest days before this game).

Outputs
-------
``skating_baseline_{season}.parquet`` — one row per player:
    player_id, player_name, team, season,
    baseline_avg_speed_kmh, baseline_max_speed_kmh,
    baseline_distance_per_game_km,
    baseline_zone_time_pct_oz, baseline_zone_time_pct_dz, baseline_zone_time_pct_nz,
    speed_std, distance_std,
    n_games_rested, n_games_total, confidence, model_version
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

MODEL_VERSION       = "skate_v1"
MIN_REST_DAYS       = 1          # minimum rest days to count as "rested"
MIN_GAMES_RESTED    = 5          # min rested-game rows to use rested baseline
MIN_GAMES_TOTAL     = 3          # below this → confidence="insufficient"
MAX_SPEED_KMH       = 40.0       # implausible upper bound (clip)
MAX_DIST_KM         = 10.0       # implausible upper bound (clip)
MIN_SPEED_KMH       = 0.0

# Baseline metrics derived from EDGE data
BASELINE_METRICS: list[str] = [
    "avg_speed_kmh",
    "max_speed_kmh",
    "distance_per_game_km",
    "zone_time_pct_oz",
    "zone_time_pct_dz",
    "zone_time_pct_nz",
]

# Output schema
SKATING_BASELINE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":                    pl.Int64,
    "player_name":                  pl.Utf8,
    "team":                         pl.Utf8,
    "season":                       pl.Int64,
    "baseline_avg_speed_kmh":       pl.Float64,
    "baseline_max_speed_kmh":       pl.Float64,
    "baseline_distance_per_game_km": pl.Float64,
    "baseline_zone_time_pct_oz":    pl.Float64,
    "baseline_zone_time_pct_dz":    pl.Float64,
    "baseline_zone_time_pct_nz":    pl.Float64,
    "speed_std":                    pl.Float64,
    "distance_std":                 pl.Float64,
    "n_games_rested":               pl.Int64,
    "n_games_total":                pl.Int64,
    "confidence":                   pl.Utf8,
    "model_version":                pl.Utf8,
}

# Confidence levels (ordered best → worst)
CONFIDENCE_RESTED      = "rested"
CONFIDENCE_AGGREGATE   = "aggregate"
CONFIDENCE_INSUFFICIENT = "insufficient"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _clip_edge(df: pl.DataFrame) -> pl.DataFrame:
    """Clip implausible EDGE metric values in-place."""
    clips: dict[str, tuple[float, float]] = {
        "avg_speed_kmh":       (MIN_SPEED_KMH, MAX_SPEED_KMH),
        "max_speed_kmh":       (MIN_SPEED_KMH, MAX_SPEED_KMH),
        "distance_per_game_km": (0.0, MAX_DIST_KM),
        "zone_time_pct_oz":    (0.0, 1.0),
        "zone_time_pct_dz":    (0.0, 1.0),
        "zone_time_pct_nz":    (0.0, 1.0),
    }
    exprs = []
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            exprs.append(pl.col(col).clip(lo, hi).alias(col))
    if exprs:
        df = df.with_columns(exprs)
    return df


def _safe_mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def _safe_std(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.std(clean, ddof=0)) if len(clean) >= 2 else 0.0


def _player_baseline(
    rows: list[dict[str, Any]],
    rested_rows: list[dict[str, Any]],
    min_games_rested: int,
    min_games_total: int,
) -> dict[str, Any]:
    """Compute baseline dict for one player from their EDGE game rows.

    Handles both:
    - Game-level data: many rows, one per game (uses len(rows) for counts)
    - Season-aggregated data: one row with ``games_played`` column (uses that
      value for confidence thresholds; rest-day filtering is not applicable).
    """
    # Detect season-aggregated format: single row with a games_played column
    _is_aggregated = (
        len(rows) == 1
        and rows[0].get("games_played") is not None
        and int(rows[0].get("games_played", 0)) > 1
    )

    if _is_aggregated:
        # Use the games_played field for confidence thresholds
        n_total  = int(rows[0].get("games_played", 0))
        n_rested = 0  # rest-day info not available in aggregated data
    else:
        n_total  = len(rows)
        n_rested = len(rested_rows)

    if n_total < min_games_total:
        confidence = CONFIDENCE_INSUFFICIENT
        source = rows
    elif not _is_aggregated and n_rested >= min_games_rested:
        confidence = CONFIDENCE_RESTED
        source = rested_rows
    else:
        confidence = CONFIDENCE_AGGREGATE
        source = rows

    def _col(col: str) -> list[float | None]:
        return [r.get(col) for r in source]

    return {
        "baseline_avg_speed_kmh":        _safe_mean(_col("avg_speed_kmh")),
        "baseline_max_speed_kmh":        _safe_mean(_col("max_speed_kmh")),
        "baseline_distance_per_game_km": _safe_mean(_col("distance_per_game_km")),
        "baseline_zone_time_pct_oz":     _safe_mean(_col("zone_time_pct_oz")),
        "baseline_zone_time_pct_dz":     _safe_mean(_col("zone_time_pct_dz")),
        "baseline_zone_time_pct_nz":     _safe_mean(_col("zone_time_pct_nz")),
        "speed_std":                     _safe_std(_col("avg_speed_kmh")),
        "distance_std":                  _safe_std(_col("distance_per_game_km")),
        "n_games_rested":                n_rested,
        "n_games_total":                 n_total,
        "confidence":                    confidence,
    }


# ---------------------------------------------------------------------------
# SkatingBaselineModel — Feature 2.20
# ---------------------------------------------------------------------------

class SkatingBaselineModel:
    """Player skating baseline profile model (Feature 2.20).

    Derives per-player EDGE skating ceilings from historical EDGE data,
    conditioned on rest_days > MIN_REST_DAYS when available.

    Usage::

        model = SkatingBaselineModel()
        baseline_df = model.fit(edge_df, season=2025)

        # Multi-season (most recent season wins on player_id key)
        df_all = pl.concat([edge_2023, edge_2024, edge_2025])
        baseline_df = model.fit(df_all, season=2025)
    """

    def __init__(
        self,
        min_rest_days:    int   = MIN_REST_DAYS,
        min_games_rested: int   = MIN_GAMES_RESTED,
        min_games_total:  int   = MIN_GAMES_TOTAL,
        max_speed_kmh:    float = MAX_SPEED_KMH,
        max_dist_km:      float = MAX_DIST_KM,
    ) -> None:
        self._min_rest      = min_rest_days
        self._min_rested    = min_games_rested
        self._min_total     = min_games_total
        self._max_speed     = max_speed_kmh
        self._max_dist      = max_dist_km
        self._version       = MODEL_VERSION
        self._baselines: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit(
        self,
        edge_df: pl.DataFrame,
        season: int,
    ) -> pl.DataFrame:
        """Fit baseline profiles from EDGE data.

        Args:
            edge_df: EDGE skating DataFrame (EDGE_SKATING_SCHEMA columns).
                     If it contains a ``rest_days`` column, rested-game
                     filtering is applied; otherwise season-aggregate is used.
            season:  Season year to attach to output rows.

        Returns:
            DataFrame matching SKATING_BASELINE_SCHEMA.

        Raises:
            ValueError: If ``player_id`` column is missing.
        """
        if "player_id" not in (edge_df.columns if hasattr(edge_df, "columns") else []):
            raise ValueError("edge_df must contain 'player_id' column.")

        if len(edge_df) == 0:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in SKATING_BASELINE_SCHEMA.items()}
            )

        # Clip implausible values
        df = _clip_edge(edge_df)

        has_rest = "rest_days" in df.columns
        rows_all = df.to_dicts()

        # Group by player_id
        from collections import defaultdict
        player_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        player_rested: dict[int, list[dict[str, Any]]] = defaultdict(list)
        player_meta: dict[int, dict[str, str]] = {}  # player_id → {name, team}

        for row in rows_all:
            pid = row.get("player_id")
            if pid is None:
                continue
            pid = int(pid)
            player_rows[pid].append(row)
            player_meta[pid] = {
                "player_name": row.get("player_name", ""),
                "team":        row.get("team", ""),
            }
            if has_rest:
                rd = row.get("rest_days")
                if rd is not None and int(rd) > self._min_rest:
                    player_rested[pid].append(row)

        output: list[dict[str, Any]] = []
        for pid, all_rows in player_rows.items():
            rested = player_rested.get(pid, [])
            baseline = _player_baseline(
                all_rows, rested,
                self._min_rested, self._min_total,
            )
            meta = player_meta[pid]
            row_out = {
                "player_id":   pid,
                "player_name": meta["player_name"],
                "team":        meta["team"],
                "season":      season,
                **baseline,
                "model_version": self._version,
            }
            output.append(row_out)
            self._baselines[pid] = row_out

        if not output:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in SKATING_BASELINE_SCHEMA.items()}
            )

        df_out = pl.DataFrame(output)
        for col, dtype in SKATING_BASELINE_SCHEMA.items():
            if col not in df_out.columns:
                df_out = df_out.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df_out.select(list(SKATING_BASELINE_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_player_baseline(self, player_id: int) -> dict[str, Any] | None:
        """Return the baseline dict for a single player (after fit).

        Returns:
            Baseline dict or None if player was not in fit data.
        """
        return self._baselines.get(int(player_id))

    def speed_ceiling(self, player_id: int) -> float | None:
        """Return baseline avg_speed_kmh for a player.

        Returns:
            float or None if player not found / no speed data.
        """
        b = self.get_player_baseline(player_id)
        if b is None:
            return None
        return b.get("baseline_avg_speed_kmh")

    def distance_ceiling(self, player_id: int) -> float | None:
        """Return baseline distance_per_game_km for a player."""
        b = self.get_player_baseline(player_id)
        if b is None:
            return None
        return b.get("baseline_distance_per_game_km")

    def zone_profile(self, player_id: int) -> dict[str, float | None] | None:
        """Return zone time percentages for a player.

        Returns:
            Dict with keys oz, dz, nz or None if player not found.
        """
        b = self.get_player_baseline(player_id)
        if b is None:
            return None
        return {
            "oz": b.get("baseline_zone_time_pct_oz"),
            "dz": b.get("baseline_zone_time_pct_dz"),
            "nz": b.get("baseline_zone_time_pct_nz"),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, baseline_df: pl.DataFrame) -> dict[str, Any]:
        """Compute league-wide summary statistics from a baseline DataFrame.

        Returns:
            Dict with: n_players, n_rested, n_aggregate, n_insufficient,
            mean_speed, mean_distance, mean_oz_pct.
        """
        if len(baseline_df) == 0:
            return {
                "n_players": 0, "n_rested": 0, "n_aggregate": 0,
                "n_insufficient": 0, "mean_speed": None,
                "mean_distance": None, "mean_oz_pct": None,
            }
        conf = baseline_df["confidence"].to_list()
        speeds = baseline_df["baseline_avg_speed_kmh"].drop_nulls().to_list()
        dists  = baseline_df["baseline_distance_per_game_km"].drop_nulls().to_list()
        ozs    = baseline_df["baseline_zone_time_pct_oz"].drop_nulls().to_list()
        return {
            "n_players":     len(baseline_df),
            "n_rested":      conf.count(CONFIDENCE_RESTED),
            "n_aggregate":   conf.count(CONFIDENCE_AGGREGATE),
            "n_insufficient": conf.count(CONFIDENCE_INSUFFICIENT),
            "mean_speed":    float(np.mean(speeds)) if speeds else None,
            "mean_distance": float(np.mean(dists)) if dists else None,
            "mean_oz_pct":   float(np.mean(ozs)) if ozs else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "min_rest":     self._min_rest,
            "min_rested":   self._min_rested,
            "min_total":    self._min_total,
            "max_speed":    self._max_speed,
            "max_dist":     self._max_dist,
            "version":      self._version,
            "baselines":    self._baselines,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "SkatingBaselineModel":
        payload = joblib.load(Path(path))
        obj = cls(
            min_rest_days    = payload.get("min_rest",   MIN_REST_DAYS),
            min_games_rested = payload.get("min_rested", MIN_GAMES_RESTED),
            min_games_total  = payload.get("min_total",  MIN_GAMES_TOTAL),
            max_speed_kmh    = payload.get("max_speed",  MAX_SPEED_KMH),
            max_dist_km      = payload.get("max_dist",   MAX_DIST_KM),
        )
        obj._version   = payload.get("version",   MODEL_VERSION)
        obj._baselines = payload.get("baselines", {})
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_skating_baseline(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write skating baseline scores to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"skating_baseline_{season}.parquet"
    for col, dtype in SKATING_BASELINE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(SKATING_BASELINE_SCHEMA.keys())).write_parquet(path)
    return path
