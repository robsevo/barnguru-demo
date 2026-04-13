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
For skaters (Evolving Hockey v1.1 methodology):

    GAR = (rapm_ev_off × 1.6 + rapm_ev_def × 1.4 − repl_ev) × (toi_ev / 60)
          + (rapm_pp × 1.3 − repl_pp) × (toi_pp / 60)
          + (rapm_pk × 1.3 − repl_pk) × (toi_pk / 60)
          + finishing_contribution

    Multipliers (1.6 EV-off, 1.4 EV-def, 1.3 PP, 1.3 PK) correct for ridge
    shrinkage so that individual RAPMs sum to actual team goal differentials.

    Penalty differential uses our own PBP parquets (committed_by_id / drawn_by_id)
    to compute net PIM per player. No external API needed.

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

MODEL_VERSION = "war_v2"

# Pythagorean goals/win.
# Evolving Hockey uses exponent 2.091 with a season-varying value; the 3-year
# average (2017–2020) is ≈ 5.33. We use 5.5 as a conservative fixed constant
# (higher-scoring recent seasons push this up slightly). This replaces the
# previous value of 5.0, which overstated WAR by ~8–10%.
GOALS_PER_WIN     = 5.5

# Open-market WAR dollar rate. At 2023–24 cap levels (~$88M ceiling), community
# analyses and contract modelling place this at $2.0–2.5M/WAR. The previous
# value of $1.2M was calibrated to a 2019 cap environment and understated
# contract values by roughly half.
WAR_DOLLAR_RATE   = 2.2     # $M per WAR — open-market free-agent rate (2023-24)

# Replacement level: computed from data each season (see _compute_replacement_level).
# These fallbacks are used when there are too few players to compute from data.
REPLACEMENT_RAPM_EV_FALLBACK = -2.0   # goals/60 below average at EV
REPLACEMENT_RAPM_PP_FALLBACK = -0.5
REPLACEMENT_RAPM_PK_FALLBACK = -0.5

# Roster depth: number of skater slots before replacement tier.
# 32 teams (since 2021-22 with SEA + UTH) × 20 active skaters = 640.
# Players ranked below this EV-TOI cutoff ARE the replacement tier.
# NOTE: The standard method (Evolving Hockey) uses per-situation thresholds
# (13F+7D at EV, 9-11F+4D on PP, 8-9F+6D on PK). We use a single EV-TOI
# cutoff as a reasonable approximation.
ROSTER_SLOTS_SKATERS = 640

# Team-adjustment multipliers (Evolving Hockey v1.1).
# Ridge regularization shrinks individual RAPM coefficients toward 0, so the
# sum of all players' RAPM does not add up to actual team goal differentials.
# These multipliers correct for that shrinkage so that GAR components are on
# the same scale as real goals. Values from Evolving Hockey's published methodology.
# NOTE: These multipliers were calibrated against actual-goal RAPM. Our RAPM
# uses xG differentials; the xG-to-goal conversion factor is approximately 1.0
# over large samples, but these multipliers should be recalibrated if xG-based
# RAPM systematically under/over-estimates vs. actual goals.
MULTIPLIER_EV_OFF = 1.6
MULTIPLIER_EV_DEF = 1.4
MULTIPLIER_PP     = 1.3
MULTIPLIER_PK     = 1.3

# Minimum EV TOI to compute WAR — below this the RAPM estimate is too noisy.
# ~200 min EV ≈ 25 games at 8 min/game for a 4th liner, or 15 games for a top-6.
MIN_TOI_EV_MINUTES_OUTPUT = 200  # exclude from WAR output
MIN_TOI_EV_MINUTES_REPL   = 100  # for replacement-level calculation

# RAPM sanity clamps — physically, no skater can be ±5 goals/60 EV;
# PP/PK have higher leverage but ±4 is a hard ceiling.
RAPM_EV_CLAMP = 4.0
RAPM_PP_CLAMP = 4.0
RAPM_PK_CLAMP = 4.0

# GAR scaling: RAPM values are goals/60; multiply by (TOI/60) to get total goals
# For finishing: delta is already in goals (seasonal total)
FINISHING_SCALE   = 1.0

# Penalty differential: each net penalty minute is worth ~0.11 goals.
# Derived from average PP conversion rate (~20%) × average PP opportunity length (2 min).
# A player who draws 10 more PIM than they take = +1.1 GAR from penalties.
# Source: Evolving Hockey WAR 1.1 methodology.
GOALS_PER_PIM_DIFF = 0.11

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
    "gar_penalties":       pl.Float64,
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

def _compute_replacement_level(
    rapm_df: pl.DataFrame,
    roster_slots: int = ROSTER_SLOTS_SKATERS,
    min_toi_ev: float = MIN_TOI_EV_MINUTES_REPL,
) -> tuple[float, float, float]:
    """Derive replacement RAPM from the actual player pool for this season(s).

    Method:
      1. Keep only skaters with ≥ min_toi_ev minutes of EV ice time (excludes
         goalies and micro-sample callups from the reference pool).
      2. Sort by EV TOI descending (more ice = more trusted player).
      3. The replacement tier starts at rank ``roster_slots + 1``.  We take the
         mean RAPM of the players just below that cut (ranks roster_slots+1 →
         roster_slots+30) as the replacement level.  Averaging a small band
         rather than using a single player is more stable.
      4. Falls back to the static constant if there aren't enough players.

    Returns:
        (repl_ev, repl_pp, repl_pk) replacement RAPM values (goals/60).
    """
    if rapm_df.is_empty():
        return REPLACEMENT_RAPM_EV_FALLBACK, REPLACEMENT_RAPM_PP_FALLBACK, REPLACEMENT_RAPM_PK_FALLBACK

    # Normalise TOI to minutes (detect unit from column max)
    toi_col = "toi_ev" if "toi_ev" in rapm_df.columns else "toi_ev_min"
    if toi_col not in rapm_df.columns:
        return REPLACEMENT_RAPM_EV_FALLBACK, REPLACEMENT_RAPM_PP_FALLBACK, REPLACEMENT_RAPM_PK_FALLBACK

    max_toi = float(rapm_df[toi_col].drop_nulls().max() or 0.0)
    scale   = 1.0 / 60.0 if max_toi > 1800.0 else 1.0

    df = rapm_df.with_columns(
        (pl.col(toi_col).cast(pl.Float64) * scale).alias("_toi_ev_min")
    )

    # Filter to qualifying skaters
    df = df.filter(pl.col("_toi_ev_min") >= min_toi_ev)

    # If multiple seasons present, sum TOI and average RAPM per player
    if "season" in df.columns and df["season"].n_unique() > 1:
        agg_exprs = [pl.col("_toi_ev_min").sum()]
        for col in ("rapm_ev_off", "rapm_ev_def", "rapm_pp", "rapm_pk"):
            if col in df.columns:
                agg_exprs.append(pl.col(col).mean())
        df = df.group_by("player_id").agg(agg_exprs)

    # Sort by EV TOI descending
    df = df.sort("_toi_ev_min", descending=True)
    n = len(df)

    if n <= roster_slots:
        # Not enough players to define a replacement tier; use fallback
        return REPLACEMENT_RAPM_EV_FALLBACK, REPLACEMENT_RAPM_PP_FALLBACK, REPLACEMENT_RAPM_PK_FALLBACK

    # Replacement band: players just below the roster cut
    band_size = min(30, n - roster_slots)
    repl_band = df.slice(roster_slots, band_size)

    def _mean(col: str) -> float:
        if col not in repl_band.columns:
            return 0.0
        vals = repl_band[col].drop_nulls().cast(pl.Float64)
        return float(vals.mean()) if len(vals) > 0 else 0.0

    repl_ev  = _mean("rapm_ev_off") + _mean("rapm_ev_def")
    # PP/PK replacement levels from the band are unreliable because:
    #   - Many players in the band have ~0 PP/PK time → extreme small-sample RAPM
    #   - The EV-ranked band isn't a good proxy for PP/PK replacement
    # Use fixed fallbacks for PP and PK.
    repl_pp  = REPLACEMENT_RAPM_PP_FALLBACK
    repl_pk  = REPLACEMENT_RAPM_PK_FALLBACK

    return repl_ev, repl_pp, repl_pk


def gar_from_rapm(
    rapm_ev_off: float | None,
    rapm_ev_def: float | None,
    rapm_pp:     float | None,
    rapm_pk:     float | None,
    toi_ev_min:  float,
    toi_pp_min:  float,
    toi_pk_min:  float,
    replacement_ev: float = REPLACEMENT_RAPM_EV_FALLBACK,
    replacement_pp: float = REPLACEMENT_RAPM_PP_FALLBACK,
    replacement_pk: float = REPLACEMENT_RAPM_PK_FALLBACK,
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
    # Clamp RAPM values to physically plausible ranges before computing GAR.
    # Small-sample players can have extreme ridge-regression values.
    def _clamp(v: float | None, limit: float) -> float:
        if v is None:
            return 0.0
        return float(np.clip(v, -limit, limit))

    ev_off = _clamp(rapm_ev_off, RAPM_EV_CLAMP)
    ev_def = _clamp(rapm_ev_def, RAPM_EV_CLAMP)
    pp     = _clamp(rapm_pp, RAPM_PP_CLAMP)
    pk     = _clamp(rapm_pk, RAPM_PK_CLAMP)

    # Apply team-adjustment multipliers before computing above-replacement goals.
    # These correct for ridge shrinkage so that the sum of individual RAPMs
    # matches actual team goal differentials (Evolving Hockey methodology).
    ev_off_adj = ev_off * MULTIPLIER_EV_OFF
    ev_def_adj = ev_def * MULTIPLIER_EV_DEF
    pp_adj     = pp     * MULTIPLIER_PP
    pk_adj     = pk     * MULTIPLIER_PK

    # Above-replacement contribution in goals
    gaa_ev = (ev_off_adj + ev_def_adj - replacement_ev) * (toi_ev_min / 60.0)
    gaa_pp = (pp_adj - replacement_pp) * (toi_pp_min / 60.0)
    gaa_pk = (pk_adj - replacement_pk) * (toi_pk_min / 60.0)

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
        penalty_df:   pl.DataFrame | None = None,
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
            penalty_df:   Optional penalty aggregation — must have player_id, season,
                          pim_taken, pim_drawn. Built from our own PBP parquets.
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

        # Compute data-driven replacement level from actual player pool
        repl_ev, repl_pp, repl_pk = _compute_replacement_level(rapm_df)
        print(
            f"  [war] Replacement level (data-driven): "
            f"EV={repl_ev:+.3f}  PP={repl_pp:+.3f}  PK={repl_pk:+.3f}  "
            f"(fallback: {REPLACEMENT_RAPM_EV_FALLBACK}/60)"
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

        # Penalty differential map: (player_id, season) → net PIM (drawn - taken)
        penalty_map: dict[tuple[int, int], float] = {}
        if penalty_df is not None and len(penalty_df) > 0:
            for row in penalty_df.to_dicts():
                pid  = int(row.get("player_id", 0))
                sea  = int(row.get("season", 0))
                drawn = float(row.get("pim_drawn") or 0.0)
                taken = float(row.get("pim_taken") or 0.0)
                if pid:
                    penalty_map[(pid, sea)] = drawn - taken

        cap_map: dict[tuple[int, int], float] = {}
        if cap_df is not None and len(cap_df) > 0:
            for row in cap_df.to_dicts():
                pid = int(row.get("player_id", 0))
                sea = int(row.get("season", 0))
                val = row.get("cap_hit")
                if pid and val is not None:
                    cap_map[(pid, sea)] = float(val)

        # Detect TOI unit (seconds vs minutes) from the column maximum.
        # We check per-column because PP/PK TOI can be very small (< 600 s) even
        # when stored in seconds — a per-row heuristic would misidentify them.
        # If max(col) > 1800 (30 min expressed in seconds), it's in seconds.
        def _toi_scale(col: str) -> float:
            """Return 1/60 if column is in seconds, else 1.0 (already minutes)."""
            if col not in rapm_df.columns:
                return 1.0
            max_val = float(rapm_df[col].drop_nulls().max() or 0.0)
            return 1.0 / 60.0 if max_val > 1800.0 else 1.0

        ev_scale = _toi_scale("toi_ev") or _toi_scale("toi_ev_min")
        pp_scale = _toi_scale("toi_pp") or _toi_scale("toi_pp_min")
        pk_scale = _toi_scale("toi_pk") or _toi_scale("toi_pk_min")

        output: list[dict[str, Any]] = []

        for row in rapm_df.to_dicts():
            pid     = int(row["player_id"])
            season  = int(row.get("season") or 0)

            toi_ev  = float(row.get("toi_ev") or row.get("toi_ev_min") or 0.0) * ev_scale
            toi_pp  = float(row.get("toi_pp") or row.get("toi_pp_min") or 0.0) * pp_scale
            toi_pk  = float(row.get("toi_pk") or row.get("toi_pk_min") or 0.0) * pk_scale

            # Skip micro-sample players — RAPM is too noisy below the TOI threshold.
            # These players are by definition replacement-level or worse.
            if toi_ev < MIN_TOI_EV_MINUTES_OUTPUT:
                continue

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

            # GAR components (against data-driven replacement level)
            gar_ev, gar_pp, gar_pk = gar_from_rapm(
                rapm_ev_off, rapm_ev_def, rapm_pp, rapm_pk,
                toi_ev, toi_pp, toi_pk,
                replacement_ev=repl_ev,
                replacement_pp=repl_pp,
                replacement_pk=repl_pk,
            )

            gar_finishing = finishing_map.get((pid, season), 0.0) * FINISHING_SCALE
            # Penalty differential: net PIM drawn (positive = good) × 0.11 goals/PIM
            net_pim       = penalty_map.get((pid, season), 0.0)
            gar_penalties = net_pim * GOALS_PER_PIM_DIFF
            gar_total     = gar_ev + gar_pp + gar_pk + gar_finishing + gar_penalties

            war_val       = war_from_gar(gar_total, self._goals_per_win)

            # Replacement level war (what you'd get from a replacement-level player).
            # The replacement RAPM is already in adjusted scale (it's derived from
            # the same adjusted player pool), so no additional multiplier needed here.
            repl_gar = (repl_ev * (toi_ev / 60.0)
                        + repl_pp * (toi_pp / 60.0)
                        + repl_pk * (toi_pk / 60.0))
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
                "gar_penalties":       gar_penalties,
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
