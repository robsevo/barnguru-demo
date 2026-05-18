"""Player Positional Heat Map — Feature 2.21.

Per-player probability distribution over on-ice positioning zones:
slot%, half-wall%, circle%, point%, net-front%, corner% — computed
separately for offensive (shooting) and defensive (zone deployment) profiles.

Model design:
    Offensive zone profile (from shot coordinates):
        Each shot attempt made by the player is classified into one of six
        mutually exclusive zones using arena-adjusted x/y coordinates.
        The probability for each zone = shots_in_zone / total_shots.
        Requires MIN_SHOTS_FULL (20) for "full" confidence; 5–19 gives
        "partial"; fewer gives "insufficient" (zero profile stored).

    Defensive zone profile (from EDGE zone time):
        Taken directly from the EDGE skating baseline (Feature 2.20):
        zone_time_pct_oz, zone_time_pct_dz, zone_time_pct_nz.  If EDGE
        data is unavailable the defensive profile columns are null.

Zone boundaries (arena-adjusted coordinates, goal mouth at x ≈ 89):
    net_front  : x ≥ 78, |y| ≤ 12       (crease / high-danger)
    slot       : x ≥ 54, |y| ≤ 22        (prime scoring area, not net-front)
    circle     : x ≥ 54, 22 < |y| ≤ 36  (face-off circle area)
    corner     : x ≥ 54, |y| > 36        (behind/around the net)
    half_wall  : 25 ≤ x < 54             (inner perimeter, all y)
    point      : x < 25                  (blue-line / defenseman shots)

Inputs
------
shots_df  : DataFrame with shot attempts (MoneyPuck or PBP schema).
            Must contain shooter identifier + arena-adjusted x/y coordinates.
            Accepted column aliases:
              player: shooter_id | player_id
              x:      arena_adj_x | x_adjusted | x_fixed | xCordAdjusted
              y:      arena_adj_y | y_adjusted | y_fixed | yCordAdjusted

edge_df   : Optional EDGE skating DataFrame (EDGE_SKATING_SCHEMA) for
            defensive zone deployment profile.

Outputs
-------
``positional_heatmap_{season}.parquet`` — one row per player:
    player_id, player_name, team, season,
    off_net_front_pct, off_slot_pct, off_circle_pct,
    off_half_wall_pct, off_point_pct, off_corner_pct,
    zone_time_oz, zone_time_dz, zone_time_nz,
    n_shots, confidence, model_version
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "heatmap_v1"

# Zone boundary thresholds (arena-adjusted feet; goal mouth at x ≈ 89)
ZONE_X_NET_FRONT  = 78.0   # x ≥ this for net-front / crease zone
ZONE_Y_NET_FRONT  = 12.0   # |y| ≤ this for net-front
ZONE_X_INNER      = 54.0   # x ≥ this for slot / circle / corner
ZONE_Y_SLOT       = 22.0   # |y| ≤ this for slot (face-off dot boundary)
ZONE_Y_CIRCLE     = 36.0   # |y| ≤ this for circle (outer boundary)
ZONE_X_BLUE_LINE  = 25.0   # x ≥ this for half-wall; x < this for point

MIN_SHOTS_FULL    = 20  # ≥ this → confidence = "full"
MIN_SHOTS_PARTIAL = 5   # ≥ this → confidence = "partial"
# < MIN_SHOTS_PARTIAL → confidence = "insufficient"

CONFIDENCE_FULL        = "full"
CONFIDENCE_PARTIAL     = "partial"
CONFIDENCE_INSUFFICIENT = "insufficient"

ZONE_LABELS = ["net_front", "slot", "circle", "corner", "half_wall", "point"]

# Output schema
POSITIONAL_HEATMAP_SCHEMA: dict[str, pl.DataType] = {
    "player_id":          pl.Int64,
    "player_name":        pl.Utf8,
    "team":               pl.Utf8,
    "season":             pl.Int64,
    # Offensive zone shot placement (from PBP / MoneyPuck shots)
    "off_net_front_pct":  pl.Float64,
    "off_slot_pct":       pl.Float64,
    "off_circle_pct":     pl.Float64,
    "off_half_wall_pct":  pl.Float64,
    "off_point_pct":      pl.Float64,
    "off_corner_pct":     pl.Float64,
    # Defensive zone deployment (from EDGE zone time)
    "zone_time_oz":       pl.Float64,
    "zone_time_dz":       pl.Float64,
    "zone_time_nz":       pl.Float64,
    # Confidence metadata
    "n_shots":            pl.Int64,
    "confidence":         pl.Utf8,
    "model_version":      pl.Utf8,
}


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------

def classify_shot_zone(x: float, y: float) -> str:
    """Classify a shot's arena-adjusted coordinates into one of six zones.

    Zones are mutually exclusive and collectively exhaustive for all
    arena-adjusted coordinates in the offensive half (x ≥ 0).  Shots from
    the defensive zone (x < 0) return "point" as a conservative fallback.

    Args:
        x: Arena-adjusted x-coordinate (feet; goal mouth at x ≈ 89).
        y: Arena-adjusted y-coordinate (feet; centre-ice at y = 0).

    Returns:
        One of: "net_front", "slot", "circle", "corner", "half_wall", "point".
    """
    ay = abs(y)
    if x >= ZONE_X_NET_FRONT and ay <= ZONE_Y_NET_FRONT:
        return "net_front"
    if x >= ZONE_X_INNER and ay <= ZONE_Y_SLOT:
        return "slot"
    if x >= ZONE_X_INNER and ay <= ZONE_Y_CIRCLE:
        return "circle"
    if x >= ZONE_X_INNER:
        return "corner"
    if x >= ZONE_X_BLUE_LINE:
        return "half_wall"
    return "point"


def classify_shots_batch(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Vectorised zone classification.

    Args:
        x_arr: 1-D float array of arena-adjusted x-coordinates.
        y_arr: 1-D float array of arena-adjusted y-coordinates.

    Returns:
        1-D object array of zone label strings (same length as input).
    """
    ay = np.abs(y_arr)
    zones = np.full(len(x_arr), "point", dtype=object)
    mask_nf = (x_arr >= ZONE_X_NET_FRONT) & (ay <= ZONE_Y_NET_FRONT)
    mask_sl = (x_arr >= ZONE_X_INNER) & (ay <= ZONE_Y_SLOT) & ~mask_nf
    mask_ci = (x_arr >= ZONE_X_INNER) & (ay > ZONE_Y_SLOT) & (ay <= ZONE_Y_CIRCLE)
    mask_co = (x_arr >= ZONE_X_INNER) & (ay > ZONE_Y_CIRCLE)
    mask_hw = (x_arr >= ZONE_X_BLUE_LINE) & (x_arr < ZONE_X_INNER)
    zones[mask_nf] = "net_front"
    zones[mask_sl] = "slot"
    zones[mask_ci] = "circle"
    zones[mask_co] = "corner"
    zones[mask_hw] = "half_wall"
    return zones


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def _detect_col(df: pl.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].dtype != pl.Null:
            return col
    return None


def _detect_player_col(df: pl.DataFrame) -> str | None:
    # Try integer ID columns first (skip all-null / Null-dtype columns)
    for col in ["shooter_id", "player_id"]:
        if col in df.columns and df[col].dtype != pl.Null:
            return col
    # Fall back to string name columns
    for col in ["shooter_name", "player_name"]:
        if col in df.columns and df[col].dtype != pl.Null:
            return col
    return None


def _detect_x_col(df: pl.DataFrame) -> str | None:
    return _detect_col(df, ["arena_adj_x", "x_adjusted", "x_fixed", "xCordAdjusted"])


def _detect_y_col(df: pl.DataFrame) -> str | None:
    return _detect_col(df, ["arena_adj_y", "y_adjusted", "y_fixed", "yCordAdjusted"])


# ---------------------------------------------------------------------------
# PositionalHeatmapModel — Feature 2.21
# ---------------------------------------------------------------------------

class PositionalHeatmapModel:
    """Player positional heat map model (Feature 2.21).

    Computes per-player offensive shot zone distributions and (optionally)
    attaches EDGE zone deployment profiles.

    Usage::

        model = PositionalHeatmapModel()
        df = model.fit(shots_df, season=2025)
        profile = model.get_player_profile(player_id=8478402)
        # {'net_front': 0.18, 'slot': 0.31, 'circle': 0.22, ...}
    """

    def __init__(
        self,
        min_shots_full:    int = MIN_SHOTS_FULL,
        min_shots_partial: int = MIN_SHOTS_PARTIAL,
    ) -> None:
        self._min_full    = min_shots_full
        self._min_partial = min_shots_partial
        self._version     = MODEL_VERSION
        self._profiles: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit(
        self,
        shots_df: pl.DataFrame,
        season: int,
        edge_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fit positional profiles from shot data.

        Args:
            shots_df: Shot attempts DataFrame. Must have a player identifier
                      column and arena-adjusted x/y coordinate columns.
            season:   Season year to attach to output rows.
            edge_df:  Optional EDGE skating DataFrame to attach defensive
                      zone deployment percentages.

        Returns:
            DataFrame matching POSITIONAL_HEATMAP_SCHEMA.

        Raises:
            ValueError: If no player or coordinate columns are found.
        """
        player_col = _detect_player_col(shots_df)
        x_col      = _detect_x_col(shots_df)
        y_col      = _detect_y_col(shots_df)

        if player_col is None:
            raise ValueError(
                "shots_df must contain a player identifier column "
                "(shooter_id or player_id)."
            )

        if len(shots_df) == 0:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in POSITIONAL_HEATMAP_SCHEMA.items()}
            )

        if x_col is None or y_col is None:
            warnings.warn(
                "No arena-adjusted coordinate columns found in shots_df. "
                "Expected one of arena_adj_x/x_adjusted/x_fixed/xCordAdjusted and "
                "arena_adj_y/y_adjusted/y_fixed/yCordAdjusted. "
                "Returning empty profile.",
                DataMissingWarning,
                stacklevel=2,
            )
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in POSITIONAL_HEATMAP_SCHEMA.items()}
            )

        # Extract arrays — handle both integer ID and string name player columns
        player_series = shots_df[player_col]
        _str_to_int: dict[str, int] = {}
        if player_series.dtype in (pl.Utf8, pl.String):
            # String player column (e.g. shooter_name): build a stable name→int map
            unique_names = [n for n in player_series.unique().to_list() if n is not None]
            _str_to_int = {name: abs(hash(name)) % (2 ** 31) for name in unique_names}
            pids = np.array([
                _str_to_int.get(v, -1) if v is not None else -1
                for v in player_series.to_list()
            ], dtype=np.int64)
        else:
            pids = player_series.fill_null(-1).cast(pl.Int64).to_numpy()
        xs   = shots_df[x_col].fill_null(float("nan")).to_numpy().astype(np.float64)
        ys   = shots_df[y_col].fill_null(float("nan")).to_numpy().astype(np.float64)

        # Reverse map: int_id → string name (for player_name in output)
        _int_to_name: dict[int, str] = {v: k for k, v in _str_to_int.items()}

        # Optional player_name / team columns
        has_name = "player_name" in shots_df.columns and shots_df["player_name"].dtype != pl.Null
        has_team = "team" in shots_df.columns and shots_df["team"].dtype != pl.Null
        # For shooter_name column, use it as the name
        if not has_name and player_col in ("shooter_name",):
            raw_names = shots_df[player_col].to_list()
        else:
            raw_names = shots_df["player_name"].to_list() if has_name else [""] * len(pids)
        names = raw_names
        teams_col = "shooting_team" if "shooting_team" in shots_df.columns and shots_df.get_column("shooting_team").dtype != pl.Null else ("team" if has_team else None)
        teams = shots_df[teams_col].to_list() if teams_col else [""] * len(pids)

        # Classify zones (NaN coords → classify as "point" fallback)
        nan_mask = np.isnan(xs) | np.isnan(ys)
        xs_safe = np.where(nan_mask, 0.0, xs)
        ys_safe = np.where(nan_mask, 0.0, ys)
        zone_labels = classify_shots_batch(xs_safe, ys_safe)
        zone_labels[nan_mask] = "point"  # unknown coords → point

        # Accumulate per player
        player_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        player_meta:   dict[int, dict[str, str]] = {}

        for i, pid in enumerate(pids):
            if pid < 0:
                continue
            player_counts[pid][zone_labels[i]] += 1
            if pid not in player_meta:
                player_meta[pid] = {
                    "player_name": names[i] if names[i] else "",
                    "team":        teams[i] if teams[i] else "",
                }

        # Build EDGE zone time lookup
        edge_lookup: dict[int, dict[str, float | None]] = {}
        if edge_df is not None and len(edge_df) > 0:
            edge_pid_col = _detect_col(edge_df, ["player_id"])
            if edge_pid_col:
                for row in edge_df.to_dicts():
                    pid = row.get("player_id")
                    if pid is not None:
                        edge_lookup[int(pid)] = {
                            "zone_time_oz": row.get("zone_time_pct_oz"),
                            "zone_time_dz": row.get("zone_time_pct_dz"),
                            "zone_time_nz": row.get("zone_time_pct_nz"),
                        }

        # Build output rows
        output: list[dict[str, Any]] = []
        for pid, counts in player_counts.items():
            n_shots = sum(counts.values())
            if n_shots >= self._min_full:
                confidence = CONFIDENCE_FULL
            elif n_shots >= self._min_partial:
                confidence = CONFIDENCE_PARTIAL
            else:
                confidence = CONFIDENCE_INSUFFICIENT

            def pct(zone: str) -> float | None:
                if n_shots == 0:
                    return None
                return counts.get(zone, 0) / n_shots

            edge = edge_lookup.get(pid, {})
            meta = player_meta.get(pid, {})

            row_out = {
                "player_id":         pid,
                "player_name":       meta.get("player_name", ""),
                "team":              meta.get("team", ""),
                "season":            season,
                "off_net_front_pct": pct("net_front"),
                "off_slot_pct":      pct("slot"),
                "off_circle_pct":    pct("circle"),
                "off_half_wall_pct": pct("half_wall"),
                "off_point_pct":     pct("point"),
                "off_corner_pct":    pct("corner"),
                "zone_time_oz":      edge.get("zone_time_oz"),
                "zone_time_dz":      edge.get("zone_time_dz"),
                "zone_time_nz":      edge.get("zone_time_nz"),
                "n_shots":           n_shots,
                "confidence":        confidence,
                "model_version":     self._version,
            }
            output.append(row_out)
            self._profiles[pid] = row_out

        if not output:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in POSITIONAL_HEATMAP_SCHEMA.items()}
            )

        df_out = pl.DataFrame(output)
        for col, dtype in POSITIONAL_HEATMAP_SCHEMA.items():
            if col not in df_out.columns:
                df_out = df_out.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df_out.select(list(POSITIONAL_HEATMAP_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_player_profile(self, player_id: int) -> dict[str, Any] | None:
        """Return the full profile dict for a player (after fit).

        Returns:
            Dict with zone percentages, EDGE zone time, n_shots, confidence.
            None if player was not in fit data.
        """
        return self._profiles.get(int(player_id))

    def offensive_zone_distribution(self, player_id: int) -> dict[str, float] | None:
        """Return the offensive zone shot distribution for a player.

        Returns:
            Dict mapping zone label → fraction [0,1], or None if not found.
        """
        profile = self.get_player_profile(player_id)
        if profile is None:
            return None
        return {
            "net_front": profile["off_net_front_pct"] or 0.0,
            "slot":      profile["off_slot_pct"]      or 0.0,
            "circle":    profile["off_circle_pct"]    or 0.0,
            "half_wall": profile["off_half_wall_pct"] or 0.0,
            "point":     profile["off_point_pct"]     or 0.0,
            "corner":    profile["off_corner_pct"]    or 0.0,
        }

    def high_danger_pct(self, player_id: int) -> float | None:
        """Return the fraction of shots from high-danger zones (net_front + slot)."""
        dist = self.offensive_zone_distribution(player_id)
        if dist is None:
            return None
        return dist["net_front"] + dist["slot"]

    def summary(self, heatmap_df: pl.DataFrame) -> dict[str, Any]:
        """Compute league-wide summary statistics.

        Returns:
            Dict with n_players, confidence breakdown, mean high-danger pct.
        """
        if len(heatmap_df) == 0:
            return {
                "n_players": 0, "n_full": 0, "n_partial": 0,
                "n_insufficient": 0, "mean_high_danger_pct": None,
            }
        conf = heatmap_df["confidence"].to_list()
        hd = (
            (heatmap_df["off_net_front_pct"].fill_null(0.0) +
             heatmap_df["off_slot_pct"].fill_null(0.0))
            .drop_nulls()
            .to_list()
        )
        return {
            "n_players":           len(heatmap_df),
            "n_full":              conf.count(CONFIDENCE_FULL),
            "n_partial":           conf.count(CONFIDENCE_PARTIAL),
            "n_insufficient":      conf.count(CONFIDENCE_INSUFFICIENT),
            "mean_high_danger_pct": float(np.mean(hd)) if hd else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "min_full":    self._min_full,
            "min_partial": self._min_partial,
            "version":     self._version,
            "profiles":    self._profiles,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PositionalHeatmapModel":
        payload = joblib.load(Path(path))
        obj = cls(
            min_shots_full    = payload.get("min_full",    MIN_SHOTS_FULL),
            min_shots_partial = payload.get("min_partial", MIN_SHOTS_PARTIAL),
        )
        obj._version  = payload.get("version",  MODEL_VERSION)
        obj._profiles = payload.get("profiles", {})
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_positional_heatmap(
    df: pl.DataFrame, output_dir: Path, season: int, season_type: str = "regular"
) -> Path:
    """Write positional heatmap scores to parquet. season_type='playoffs' appends suffix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_playoffs" if season_type == "playoffs" else ""
    path = output_dir / f"positional_heatmap_{season}{suffix}.parquet"
    for col, dtype in POSITIONAL_HEATMAP_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(POSITIONAL_HEATMAP_SCHEMA.keys())).write_parquet(path)
    return path
