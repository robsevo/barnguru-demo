"""WAR (Wins Above Replacement) + Contract Efficiency Model — Feature 2.25.

Synthesises all GRTZKY player ratings into a single Wins Above Replacement
score per player per season, then computes contract efficiency relative to
a player's WAR-implied value.

WAR Formula
-----------
The model converts Goal-Contributions Above Replacement (GAR components) to
wins using the Pythagorean goals-per-win constant:

    goals_per_win ≈ 5.0   (Pythagorean exp ≈ 2.022, ~5.0 G/win for typical
                            NHL teams scoring ~2.8 goals/game)

    WAR = GAR / goals_per_win

GAR Components
--------------
For skaters:

    GAR = (rapm_ev_off + rapm_ev_def) × ev_toi_multiplier
          + rapm_pp × pp_toi_multiplier
          + rapm_pk × pk_toi_multiplier
          + finishing_contribution
          + penalty_contribution

For goalies:

    GAR = goalie_war_contribution
          (derived from GSAA — Goals Saved Above Average per 60 × TOI)

Replacement Level
-----------------
Replacement level is defined as the marginal NHL player — roughly the 13th
forward or 7th defenceman by even-strength TOI percentage across the league.
The replacement-level RAPM is calibrated to ≈ −2.0 goals/60 relative to the
average player, translating to ≈ −0.4 WAR per 82-game full season.

Contract Efficiency
-------------------
Contract efficiency measures how many WAR a team receives per dollar of cap
spend:

    value_at_replacement = WAR × WAR_DOLLAR_RATE   ($M per WAR on the open market)
    contract_efficiency  = value_at_replacement / cap_hit

    efficiency > 1.0  → player is paid less than their WAR implies (value contract)
    efficiency < 1.0  → player is overpaid relative to WAR
    efficiency = None → cap_hit not available

Market rate (WAR_DOLLAR_RATE) is calibrated from NHL free-agent contracts:
≈ $1.2M USD per WAR (2024 estimate).

Input
-----
The model accepts RAPM output directly:
    player_id, season, player_name, team, gp, toi_ev, toi_pp, toi_pk,
    rapm_ev_off, rapm_ev_def, rapm_pp, rapm_pk

Optional columns (joined if available):
    finishing_delta   (from xG Finishing model, Feature 2.1)
    pp_rating         (from Special Teams model, Feature 2.7)
    pk_rating         (from Special Teams model, Feature 2.7)
    cap_hit           (USD millions, from roster data)
    position          (F/D/G — affects replacement level lookup)

Output
------
``war_{season}.parquet`` — one row per player:
    player_id, player_name, team, position, season,
    toi_ev, toi_pp, toi_pk,
    gar_ev, gar_pp, gar_pk, gar_finishing, gar_total,
    war, replacement_level_war,
    cap_hit, war_dollar_value, contract_efficiency,
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

MODEL_VERSION = "war_v1"

GOALS_PER_WIN     = 5.0     # Pythagorean goals/win for typical NHL season
WAR_DOLLAR_RATE   = 1.2     # $M per WAR — open-market free-agent rate (2024)

# Replacement level: marginal NHL player (≈ −2.0 goals/60 below average)
REPLACEMENT_RAPM_EV   = -2.0   # goals/60 below average at EV
REPLACEMENT_RAPM_PP   = -0.5
REPLACEMENT_RAPM_PK   = -0.5

# GAR scaling: RAPM values are goals/60; multiply by (TOI/60) to get total goals
# For finishing: delta is already in goals (seasonal total)
FINISHING_SCALE   = 1.0

# Output schema
WAR_SCHEMA: dict[str, pl.DataType] = {
    "player_id":           pl.Int64,
    "player_name":         pl.Utf8,
    "team":                pl.Utf8,
    "position":            pl.Utf8,
    "season":              pl.Int64,
    "toi_ev":              pl.Float64,
    "toi_pp":              pl.Float64,
    "toi_pk":              pl.Float64,
    "gar_ev":              pl.Float64,
    "gar_pp":              pl.Float64,
    "gar_pk":              pl.Float64,
    "gar_finishing":       pl.Float64,
    "gar_total":           pl.Float64,
    "war":                 pl.Float64,
    "replacement_level_war": pl.Float64,
    "cap_hit":             pl.Float64,
    "war_dollar_value":    pl.Float64,
    "contract_efficiency": pl.Float64,
    "model_version":       pl.Utf8,
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def gar_from_rapm(
    rapm_ev_off: float | None,
    rapm_ev_def: float | None,
    rapm_pp:     float | None,
    rapm_pk:     float | None,
    toi_ev_min:  float,
    toi_pp_min:  float,
    toi_pk_min:  float,
) -> tuple[float, float, float]:
    """Convert RAPM values (goals/60) to total goals above average (GAA).

    Replacement-level adjustment is applied so that the output represents
    goals above replacement (GAR), not just goals above average.

    Args:
        rapm_ev_off:  EV offensive RAPM (goals/60).
        rapm_ev_def:  EV defensive RAPM (goals/60).
        rapm_pp:      PP RAPM (goals/60).
        rapm_pk:      PK RAPM (goals/60).
        toi_ev_min:   Even-strength TOI in minutes.
        toi_pp_min:   Power-play TOI in minutes.
        toi_pk_min:   Penalty-kill TOI in minutes.

    Returns:
        (gar_ev, gar_pp, gar_pk) in goals.
    """
    ev_off = rapm_ev_off if rapm_ev_off is not None else 0.0
    ev_def = rapm_ev_def if rapm_ev_def is not None else 0.0
    pp     = rapm_pp     if rapm_pp is not None else 0.0
    pk     = rapm_pk     if rapm_pk is not None else 0.0

    # Above-average contribution in goals
    gaa_ev = (ev_off + ev_def - REPLACEMENT_RAPM_EV) * (toi_ev_min / 60.0)
    gaa_pp = (pp - REPLACEMENT_RAPM_PP) * (toi_pp_min / 60.0)
    gaa_pk = (pk - REPLACEMENT_RAPM_PK) * (toi_pk_min / 60.0)

    return float(gaa_ev), float(gaa_pp), float(gaa_pk)


def war_from_gar(gar_total: float, goals_per_win: float = GOALS_PER_WIN) -> float:
    """Convert total GAR to WAR.

    Args:
        gar_total:      Total Goals Above Replacement.
        goals_per_win:  Goals required per marginal win.

    Returns:
        Wins Above Replacement (float).
    """
    return float(gar_total / max(goals_per_win, 0.1))


def contract_efficiency(war: float, cap_hit_m: float | None) -> float | None:
    """Compute contract efficiency ratio.

    Args:
        war:        Player's WAR.
        cap_hit_m:  Cap hit in USD millions.  None → returns None.

    Returns:
        WAR-implied value / cap_hit, or None if cap_hit unavailable / zero.
    """
    if cap_hit_m is None or cap_hit_m <= 0:
        return None
    implied_value = war * WAR_DOLLAR_RATE
    return float(implied_value / cap_hit_m)


# ---------------------------------------------------------------------------
# WARModel — Feature 2.25
# ---------------------------------------------------------------------------

class WARModel:
    """WAR + Contract Efficiency Model (Feature 2.25).

    Usage::

        model = WARModel()
        war_df = model.fit(rapm_df)

        # With optional finishing + cap data
        war_df = model.fit(rapm_df, finishing_df=fin_df, cap_df=cap_df)

        # Lookup
        war = model.get_war(player_id=8478402)
        eff = model.get_contract_efficiency(player_id=8478402)
    """

    def __init__(self, goals_per_win: float = GOALS_PER_WIN) -> None:
        self._goals_per_win = goals_per_win
        self._version = MODEL_VERSION
        self._war_store: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit(
        self,
        rapm_df: pl.DataFrame,
        finishing_df: pl.DataFrame | None = None,
        cap_df:       pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute WAR and contract efficiency for all players in rapm_df.

        Args:
            rapm_df:      RAPM output: player_id, season, player_name, team, gp,
                          toi_ev, toi_pp, toi_pk, rapm_ev_off, rapm_ev_def,
                          rapm_pp, rapm_pk.
                          Also accepts toi_ev_min / toi_pp_min / toi_pk_min aliases.
            finishing_df: Optional xG finishing model output — must have player_id,
                          season, finishing_delta columns.
            cap_df:       Optional cap data — must have player_id, season, cap_hit
                          (USD millions).

        Returns:
            DataFrame matching WAR_SCHEMA, one row per player.
        """
        required = {"player_id"}
        missing = required - set(rapm_df.columns)
        if missing:
            raise ValueError(f"rapm_df missing required columns: {missing}")

        if len(rapm_df) == 0:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in WAR_SCHEMA.items()}
            )

        # Build lookup tables from optional inputs
        finishing_map: dict[tuple[int, int], float] = {}
        if finishing_df is not None and len(finishing_df) > 0:
            for row in finishing_df.to_dicts():
                pid = int(row.get("player_id", 0))
                sea = int(row.get("season", 0))
                val = row.get("finishing_delta")
                if pid and val is not None:
                    finishing_map[(pid, sea)] = float(val)

        cap_map: dict[tuple[int, int], float] = {}
        if cap_df is not None and len(cap_df) > 0:
            for row in cap_df.to_dicts():
                pid = int(row.get("player_id", 0))
                sea = int(row.get("season", 0))
                val = row.get("cap_hit")
                if pid and val is not None:
                    cap_map[(pid, sea)] = float(val)

        output: list[dict[str, Any]] = []

        for row in rapm_df.to_dicts():
            pid     = int(row["player_id"])
            season  = int(row.get("season") or 0)

            # TOI — accept both "toi_ev" and "toi_ev_min" column names.
            # RAPM output stores TOI in seconds; convert to minutes for GAR formula.
            # Auto-detect: if toi_ev > 600 it is almost certainly in seconds
            # (600 min of EV TOI in one season is impossible even for a top forward).
            def _toi_to_min(val: float) -> float:
                return val / 60.0 if val > 600.0 else val
            toi_ev  = _toi_to_min(float(row.get("toi_ev") or row.get("toi_ev_min") or 0.0))
            toi_pp  = _toi_to_min(float(row.get("toi_pp") or row.get("toi_pp_min") or 0.0))
            toi_pk  = _toi_to_min(float(row.get("toi_pk") or row.get("toi_pk_min") or 0.0))

            rapm_ev_off = row.get("rapm_ev_off")
            rapm_ev_def = row.get("rapm_ev_def")
            rapm_pp     = row.get("rapm_pp")
            rapm_pk     = row.get("rapm_pk")

            if rapm_ev_off is None and rapm_ev_def is None and rapm_pp is None and rapm_pk is None:
                warnings.warn(
                    f"Player {pid} has no RAPM values; WAR will be 0.",
                    DataMissingWarning,
                    stacklevel=2,
                )

            # GAR components
            gar_ev, gar_pp, gar_pk = gar_from_rapm(
                rapm_ev_off, rapm_ev_def, rapm_pp, rapm_pk,
                toi_ev, toi_pp, toi_pk,
            )

            gar_finishing = finishing_map.get((pid, season), 0.0) * FINISHING_SCALE
            gar_total     = gar_ev + gar_pp + gar_pk + gar_finishing

            war_val       = war_from_gar(gar_total, self._goals_per_win)

            # Replacement level war (what you'd get from a replacement-level player)
            repl_gar = (REPLACEMENT_RAPM_EV * (toi_ev / 60.0)
                        + REPLACEMENT_RAPM_PP * (toi_pp / 60.0)
                        + REPLACEMENT_RAPM_PK * (toi_pk / 60.0))
            repl_war = war_from_gar(repl_gar, self._goals_per_win)

            cap_hit   = cap_map.get((pid, season))
            war_value = war_val * WAR_DOLLAR_RATE
            eff       = contract_efficiency(war_val, cap_hit)

            rec: dict[str, Any] = {
                "player_id":           pid,
                "player_name":         str(row.get("player_name") or ""),
                "team":                str(row.get("team") or ""),
                "position":            str(row.get("position") or ""),
                "season":              season,
                "toi_ev":              toi_ev,
                "toi_pp":              toi_pp,
                "toi_pk":              toi_pk,
                "gar_ev":              gar_ev,
                "gar_pp":              gar_pp,
                "gar_pk":              gar_pk,
                "gar_finishing":       gar_finishing,
                "gar_total":           gar_total,
                "war":                 war_val,
                "replacement_level_war": repl_war,
                "cap_hit":             cap_hit,
                "war_dollar_value":    war_value,
                "contract_efficiency": eff,
                "model_version":       self._version,
            }
            output.append(rec)
            self._war_store[pid] = rec

        if not output:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in WAR_SCHEMA.items()}
            )

        df_out = pl.DataFrame(output)
        for col, dtype in WAR_SCHEMA.items():
            if col not in df_out.columns:
                df_out = df_out.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df_out.select(list(WAR_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_war(self, player_id: int) -> float | None:
        """Return WAR for a player, or None if not found."""
        rec = self._war_store.get(int(player_id))
        return rec["war"] if rec is not None else None

    def get_contract_efficiency(self, player_id: int) -> float | None:
        """Return contract efficiency ratio for a player, or None."""
        rec = self._war_store.get(int(player_id))
        return rec["contract_efficiency"] if rec is not None else None

    def get_gar(self, player_id: int) -> float | None:
        """Return total GAR for a player, or None if not found."""
        rec = self._war_store.get(int(player_id))
        return rec["gar_total"] if rec is not None else None

    def league_war_percentile(self, war: float) -> float | None:
        """Return the percentile rank of a WAR value across all fitted players.

        Args:
            war: WAR value to rank.

        Returns:
            Percentile in [0, 1], or None if no players are fitted.
        """
        if not self._war_store:
            return None
        all_wars = sorted(r["war"] for r in self._war_store.values())
        n = len(all_wars)
        rank = sum(1 for w in all_wars if w <= war)
        return float(rank / n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version":       self._version,
            "goals_per_win": self._goals_per_win,
            "war_store":     self._war_store,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "WARModel":
        payload = joblib.load(Path(path))
        obj = cls(goals_per_win=payload.get("goals_per_win", GOALS_PER_WIN))
        obj._version   = payload.get("version",   MODEL_VERSION)
        obj._war_store = payload.get("war_store",  {})
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_war(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write WAR scores to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"war_{season}.parquet"
    for col, dtype in WAR_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(WAR_SCHEMA.keys())).write_parquet(path)
    return path
