"""Playoff Performance Delta Model — Feature 2.23.

Career playoff vs. regular-season delta per player for goals/60, xGF/60,
CF%, and goalie SV%.  Bayesian shrinkage prevents over-weighting sparse
playoff samples:

    shrinkage_factor = min(playoff_gp / SHRINK_GAMES, 1.0)
    shrunk_delta     = raw_delta × shrinkage_factor

A player with 0 playoff games → shrinkage = 0, no modifier applied.
A player with 30+ playoff games (≈ 3 deep runs) → full delta applied.

Applied as additive modifier to player/goalie ratings when
``game_type=playoff`` in the Rust engine (Features 5.14, 5.5).

Input
-----
stats_df : DataFrame with one row per (player_id, game_type).
    Required columns: player_id, game_type, gp
    game_type values: 2 (regular season) or 3 (playoff); also accepts
                      "R" / "P" string codes.
    Optional metric columns (any subset):
        toi_seconds   total even-strength ice time in seconds
        goals         total goals scored
        xgf           expected goals for (on-ice)
        cf            Corsi for (shot attempts for)
        ca            Corsi against (shot attempts against)
        sv_pct        save percentage (goalies only)
        shots_against total shots against (goalies only)

Output
------
``playoff_delta_{season}.parquet`` — one row per player:
    player_id, player_name, team, position,
    reg_gp, reg_toi_sec, reg_goals_per60, reg_xgf_per60, reg_cf_pct, reg_sv_pct,
    playoff_gp, playoff_toi_sec, playoff_goals_per60, playoff_xgf_per60,
    playoff_cf_pct, playoff_sv_pct,
    delta_goals_per60, delta_xgf_per60, delta_cf_pct, delta_sv_pct,
    shrinkage_factor,
    shrunk_delta_goals_per60, shrunk_delta_xgf_per60,
    shrunk_delta_cf_pct, shrunk_delta_sv_pct,
    model_version
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

MODEL_VERSION     = "playoff_delta_v1"
SHRINK_GAMES      = 30      # playoff GP for full shrinkage (≈ 3 deep runs)
GAME_TYPE_REG     = 2
GAME_TYPE_PLAYOFF = 3
SECONDS_PER_HOUR  = 3600.0

# Output schema
PLAYOFF_DELTA_SCHEMA: dict[str, pl.DataType] = {
    "player_id":               pl.Int64,
    "player_name":             pl.Utf8,
    "team":                    pl.Utf8,
    "position":                pl.Utf8,
    # Regular season
    "reg_gp":                  pl.Int64,
    "reg_toi_sec":             pl.Float64,
    "reg_goals_per60":         pl.Float64,
    "reg_xgf_per60":           pl.Float64,
    "reg_cf_pct":              pl.Float64,
    "reg_sv_pct":              pl.Float64,
    # Playoffs
    "playoff_gp":              pl.Int64,
    "playoff_toi_sec":         pl.Float64,
    "playoff_goals_per60":     pl.Float64,
    "playoff_xgf_per60":       pl.Float64,
    "playoff_cf_pct":          pl.Float64,
    "playoff_sv_pct":          pl.Float64,
    # Raw deltas
    "delta_goals_per60":       pl.Float64,
    "delta_xgf_per60":         pl.Float64,
    "delta_cf_pct":            pl.Float64,
    "delta_sv_pct":            pl.Float64,
    # Shrinkage-adjusted deltas
    "shrinkage_factor":        pl.Float64,
    "shrunk_delta_goals_per60": pl.Float64,
    "shrunk_delta_xgf_per60":  pl.Float64,
    "shrunk_delta_cf_pct":     pl.Float64,
    "shrunk_delta_sv_pct":     pl.Float64,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def shrinkage_factor(playoff_gp: int | float, shrink_games: int = SHRINK_GAMES) -> float:
    """Compute Bayesian shrinkage factor from playoff games played.

    Args:
        playoff_gp:   Number of career playoff games played.
        shrink_games: Games for full shrinkage (default: SHRINK_GAMES = 30).

    Returns:
        Float in [0, 1].  0 = no playoff experience; 1 = full delta applied.
    """
    return float(min(max(playoff_gp, 0) / max(shrink_games, 1), 1.0))


def _safe_per60(value: float | None, toi_sec: float | None) -> float | None:
    """Convert a counting stat to a per-60-minutes rate."""
    if value is None or toi_sec is None:
        return None
    if toi_sec <= 0:
        return None
    return float(value / toi_sec * SECONDS_PER_HOUR)


def _safe_cf_pct(cf: float | None, ca: float | None) -> float | None:
    """Compute Corsi percentage from CF and CA."""
    if cf is None or ca is None:
        return None
    total = cf + ca
    return float(cf / total) if total > 0 else None


def _normalise_game_type(val: Any) -> int | None:
    """Normalise game_type to 2 (regular) or 3 (playoff)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = int(val)
        if v in (2, 3):
            return v
        return None
    s = str(val).strip().upper()
    if s in ("2", "R", "REG", "REGULAR"):
        return GAME_TYPE_REG
    if s in ("3", "P", "PO", "PLAYOFF", "PLAYOFFS"):
        return GAME_TYPE_PLAYOFF
    return None


# ---------------------------------------------------------------------------
# PlayoffDeltaModel — Feature 2.23
# ---------------------------------------------------------------------------

class PlayoffDeltaModel:
    """Playoff performance delta model (Feature 2.23).

    Usage::

        model = PlayoffDeltaModel()
        delta_df = model.fit(stats_df)

        # Get modifier for one player
        mod = model.get_modifier(player_id=8478402)
        # {'goals_per60': +0.3, 'xgf_per60': +0.2, 'cf_pct': +0.01, 'sv_pct': 0.0}
    """

    def __init__(self, shrink_games: int = SHRINK_GAMES) -> None:
        self._shrink = shrink_games
        self._version = MODEL_VERSION
        self._modifiers: dict[int, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit(
        self,
        stats_df: pl.DataFrame,
        min_reg_toi_sec: float = 300.0,
    ) -> pl.DataFrame:
        """Compute per-player playoff deltas from a split stats DataFrame.

        Args:
            stats_df:        DataFrame with one row per (player_id, game_type).
                             Required: player_id, game_type, gp.
                             Optional metrics: toi_seconds, goals, xgf, cf, ca,
                             sv_pct, shots_against, player_name, team, position.
            min_reg_toi_sec: Minimum regular-season ice time (seconds) to include
                             a player.  Players with less are returned with null
                             rates and no delta.

        Returns:
            DataFrame matching PLAYOFF_DELTA_SCHEMA.

        Raises:
            ValueError: If player_id or game_type columns are missing.
        """
        if "player_id" not in stats_df.columns:
            raise ValueError("stats_df must contain 'player_id' column.")
        if "game_type" not in stats_df.columns:
            raise ValueError("stats_df must contain 'game_type' column.")
        if len(stats_df) == 0:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYOFF_DELTA_SCHEMA.items()}
            )

        rows_dicts = stats_df.to_dicts()

        # Group rows by (player_id, normalised game_type)
        from collections import defaultdict
        by_pid_type: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        meta: dict[int, dict[str, str]] = {}

        for row in rows_dicts:
            pid = row.get("player_id")
            gt  = _normalise_game_type(row.get("game_type"))
            if pid is None or gt is None:
                continue
            pid = int(pid)
            by_pid_type[pid][gt].append(row)
            if pid not in meta:
                meta[pid] = {
                    "player_name": str(row.get("player_name") or ""),
                    "team":        str(row.get("team") or ""),
                    "position":    str(row.get("position") or ""),
                }

        output: list[dict[str, Any]] = []

        for pid, game_types in by_pid_type.items():
            reg_rows     = game_types.get(GAME_TYPE_REG, [])
            playoff_rows = game_types.get(GAME_TYPE_PLAYOFF, [])

            def _sum(rows: list[dict], key: str) -> float | None:
                vals = [r.get(key) for r in rows if r.get(key) is not None]
                return float(sum(vals)) if vals else None

            def _wmean(rows: list[dict], key: str, weight_key: str) -> float | None:
                pairs = [(r.get(key), r.get(weight_key)) for r in rows
                         if r.get(key) is not None and r.get(weight_key) is not None]
                if not pairs:
                    return None
                total_w = sum(w for _, w in pairs)
                if total_w <= 0:
                    return None
                return float(sum(v * w for v, w in pairs) / total_w)

            # Aggregate counting stats
            def agg(rows):
                return {
                    "gp":            int(_sum(rows, "gp") or 0),
                    "toi_seconds":   _sum(rows, "toi_seconds") or _sum(rows, "toi_sec"),
                    "goals":         _sum(rows, "goals"),
                    "xgf":           _sum(rows, "xgf"),
                    "cf":            _sum(rows, "cf"),
                    "ca":            _sum(rows, "ca"),
                    # sv_pct is a rate — weight by shots_against
                    "sv_pct":        _wmean(rows, "sv_pct", "shots_against"),
                }

            reg    = agg(reg_rows)
            po     = agg(playoff_rows)

            reg_toi = reg["toi_seconds"]
            po_toi  = po["toi_seconds"]

            # Skip players without enough regular-season data
            if reg_toi is None or reg_toi < min_reg_toi_sec:
                continue

            # Per-60 rates
            reg_g60  = _safe_per60(reg["goals"],  reg_toi)
            reg_x60  = _safe_per60(reg["xgf"],    reg_toi)
            reg_cf   = _safe_cf_pct(reg["cf"],     reg["ca"])
            reg_sv   = reg["sv_pct"]

            po_g60   = _safe_per60(po["goals"],  po_toi) if po_toi else None
            po_x60   = _safe_per60(po["xgf"],    po_toi) if po_toi else None
            po_cf    = _safe_cf_pct(po["cf"],     po["ca"])
            po_sv    = po["sv_pct"]

            # Raw deltas (None if playoff data missing)
            def _delta(po_val, reg_val):
                if po_val is None or reg_val is None:
                    return None
                return po_val - reg_val

            d_g60  = _delta(po_g60, reg_g60)
            d_x60  = _delta(po_x60, reg_x60)
            d_cf   = _delta(po_cf,  reg_cf)
            d_sv   = _delta(po_sv,  reg_sv)

            sf = shrinkage_factor(po["gp"], self._shrink)

            def _shrink(d):
                return d * sf if d is not None else None

            row_out = {
                "player_id":               pid,
                "player_name":             meta[pid]["player_name"],
                "team":                    meta[pid]["team"],
                "position":                meta[pid]["position"],
                "reg_gp":                  reg["gp"],
                "reg_toi_sec":             reg_toi,
                "reg_goals_per60":         reg_g60,
                "reg_xgf_per60":           reg_x60,
                "reg_cf_pct":              reg_cf,
                "reg_sv_pct":              reg_sv,
                "playoff_gp":              po["gp"],
                "playoff_toi_sec":         po_toi,
                "playoff_goals_per60":     po_g60,
                "playoff_xgf_per60":       po_x60,
                "playoff_cf_pct":          po_cf,
                "playoff_sv_pct":          po_sv,
                "delta_goals_per60":       d_g60,
                "delta_xgf_per60":         d_x60,
                "delta_cf_pct":            d_cf,
                "delta_sv_pct":            d_sv,
                "shrinkage_factor":        sf,
                "shrunk_delta_goals_per60": _shrink(d_g60),
                "shrunk_delta_xgf_per60":  _shrink(d_x60),
                "shrunk_delta_cf_pct":     _shrink(d_cf),
                "shrunk_delta_sv_pct":     _shrink(d_sv),
                "model_version":           self._version,
            }
            output.append(row_out)
            self._modifiers[pid] = {
                "goals_per60": _shrink(d_g60) or 0.0,
                "xgf_per60":   _shrink(d_x60) or 0.0,
                "cf_pct":      _shrink(d_cf)   or 0.0,
                "sv_pct":      _shrink(d_sv)   or 0.0,
            }

        if not output:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in PLAYOFF_DELTA_SCHEMA.items()}
            )

        df_out = pl.DataFrame(output)
        for col, dtype in PLAYOFF_DELTA_SCHEMA.items():
            if col not in df_out.columns:
                df_out = df_out.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df_out.select(list(PLAYOFF_DELTA_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_modifier(self, player_id: int) -> dict[str, float]:
        """Return the shrunk playoff delta modifiers for a player.

        Args:
            player_id: NHL player identifier.

        Returns:
            Dict with keys goals_per60, xgf_per60, cf_pct, sv_pct.
            Returns all-zero dict if player has no playoff history.
        """
        zero = {"goals_per60": 0.0, "xgf_per60": 0.0, "cf_pct": 0.0, "sv_pct": 0.0}
        return dict(self._modifiers.get(int(player_id), zero))

    def apply_modifier(
        self,
        player_id: int,
        base_goals_per60: float,
        base_xgf_per60: float | None = None,
    ) -> dict[str, float]:
        """Apply playoff modifier to a player's base ratings.

        Args:
            player_id:         NHL player identifier.
            base_goals_per60:  Regular-season goals/60 (or xGF/60).
            base_xgf_per60:    Regular-season xGF/60 (optional).

        Returns:
            Dict with playoff-adjusted goals_per60 and xgf_per60.
        """
        mod = self.get_modifier(player_id)
        return {
            "goals_per60": base_goals_per60 + mod["goals_per60"],
            "xgf_per60":   (base_xgf_per60 + mod["xgf_per60"])
                           if base_xgf_per60 is not None else None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "shrink":    self._shrink,
            "version":   self._version,
            "modifiers": self._modifiers,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PlayoffDeltaModel":
        payload = joblib.load(Path(path))
        obj = cls(shrink_games=payload.get("shrink", SHRINK_GAMES))
        obj._version   = payload.get("version",   MODEL_VERSION)
        obj._modifiers = payload.get("modifiers", {})
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_playoff_delta(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write playoff delta scores to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"playoff_delta_{season}.parquet"
    for col, dtype in PLAYOFF_DELTA_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(PLAYOFF_DELTA_SCHEMA.keys())).write_parquet(path)
    return path
