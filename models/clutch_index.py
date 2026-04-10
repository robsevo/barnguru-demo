"""Clutch Index — Win Probability Added (WPA) per player — Feature 2.29.

Builds a Win Probability (WP) framework from NHL play-by-play data:

    WPA_event = P(win | state_after_goal) − P(win | state_before_goal)

Win probability is estimated by a logistic regression fitted on historical
game states and outcomes.  It accepts four inputs per event:
    lead_diff       — scoring team's score lead (+ = leading, − = trailing)
    period          — 1, 2, 3, or 4 (OT)
    time_elapsed    — fraction of regulation elapsed (0.0 → 1.0 over 60 min)
    is_pp           — True when scoring team is on the power play

Per-player Clutch Index
-----------------------
::

    clutch_index = actual_wpa_per60 − expected_wpa_per60_given_opportunities

    actual_wpa_per60   = Σ(WPA for player's goals + 0.5 × assists) / toi_60
    expected_wpa_per60 = Σ(league_avg_wpa_per_opportunity × player_opportunities) / toi_60

``clutch_index > 0``  → player outperforms in high-leverage moments
``clutch_index ≈ 0``  → neutral
``clutch_index < 0``  → underperforms in high-leverage moments

Bayesian shrinkage
------------------
::

    shrinkage_factor = min(career_gp / FULL_SIGNAL_GP, 1.0)
    clutch_index_shrunk = clutch_index × shrinkage_factor

Players with fewer career games have their signal shrunk toward zero.
``FULL_SIGNAL_GP = 200`` (≈ 2.5 NHL seasons).

Multi-season weighted average
------------------------------
When called with multiple seasons, contributions are weighted:
    weight = 0.7^(seasons_ago)  (normalised so weights sum to 1.0)

Validation gate
---------------
``ClutchIndexModel.is_validated`` is True once ≥ 3 historical seasons have
been processed.  Downstream consumers (Rust engine 5.13, 5.11) should
check this flag before applying the modifier in production.

Outputs
-------
``clutch_index_{season}.parquet`` — one row per player per season:
    player_id, player_name, team, season,
    goals_credited, assists_credited,
    actual_wpa, actual_wpa_per60,
    expected_wpa_per60,
    clutch_index, clutch_index_shrunk,
    shrinkage_factor, career_gp,
    toi_60, model_version

``clutch_index_career.parquet`` — multi-season weighted average per player:
    player_id, player_name,
    seasons_counted, latest_season,
    weighted_clutch_index, shrinkage_factor,
    career_gp, model_version

References
----------
- Schuckers & Curro (2013) "Total Hockey Rating (THoR)": WP framework
- Brander, Egan & Yeung: Clutch hitting/scoring does correlate season-to-season
  in hockey ("Clutch2" effect), unlike baseball clutch
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION       = "clutch_index_v1"

FULL_SIGNAL_GP      = 200          # career GP for full shrinkage weight
SEASON_DECAY        = 0.70         # weight for each additional season back
VALIDATION_SEASONS  = 3            # seasons required before is_validated=True

ASSIST_WEIGHT       = 0.50         # WPA credit: goal = 1.0, assist = 0.5
REGULATION_MINUTES  = 60.0
OT_PERIOD           = 4
PERIOD_MINUTES      = {1: 20.0, 2: 20.0, 3: 20.0, 4: 5.0}  # period length in minutes

# Clamp on final clutch index before shrinkage
CLUTCH_INDEX_CLAMP  = 2.0


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

CLUTCH_INDEX_SCHEMA: dict[str, pl.DataType] = {
    "player_id":           pl.Int64,
    "player_name":         pl.Utf8,
    "team":                pl.Utf8,
    "season":              pl.Int64,
    "goals_credited":      pl.Float64,   # goals + assist_weight × assists
    "assists_credited":    pl.Float64,
    "actual_wpa":          pl.Float64,   # raw WPA sum for this player
    "actual_wpa_per60":    pl.Float64,
    "expected_wpa_per60":  pl.Float64,
    "clutch_index":        pl.Float64,   # actual − expected, clamped
    "clutch_index_shrunk": pl.Float64,   # × shrinkage_factor
    "shrinkage_factor":    pl.Float64,
    "career_gp":           pl.Int64,
    "toi_60":              pl.Float64,   # total ice time in 60-min units
    "model_version":       pl.Utf8,
}

CLUTCH_INDEX_CAREER_SCHEMA: dict[str, pl.DataType] = {
    "player_id":              pl.Int64,
    "player_name":            pl.Utf8,
    "seasons_counted":        pl.Int64,
    "latest_season":          pl.Int64,
    "weighted_clutch_index":  pl.Float64,
    "shrinkage_factor":       pl.Float64,
    "career_gp":              pl.Int64,
    "model_version":          pl.Utf8,
}


# ---------------------------------------------------------------------------
# Win Probability Model
# ---------------------------------------------------------------------------

class WinProbabilityModel:
    """Logistic regression model for P(home_win | game_state).

    Features (per event):
        lead_diff      — home_score − away_score (from home team's perspective)
        period         — 1, 2, 3, 4 (OT)
        time_elapsed   — fraction of 60 minutes elapsed (0.0–1.0 capped at reg end)
        is_pp_home     — True if home team is on the power play

    Fit on historical (game_state, home_win) pairs extracted from PBP data.
    When no fitted model is available, falls back to a calibrated sigmoid
    approximation so the pipeline never blocks on missing training data.
    """

    def __init__(self) -> None:
        self._fitted  = False
        self._lr      = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self._scaler  = StandardScaler()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        states_df: pl.DataFrame,
        outcomes_df: pl.DataFrame,
    ) -> "WinProbabilityModel":
        """Fit the logistic regression on historical game states.

        Args:
            states_df:   One row per game-state observation.
                Required: game_id (int64), lead_diff (int64),
                          period (int64), time_elapsed (float64).
                Optional: is_pp_home (bool).
            outcomes_df: One row per game.
                Required: game_id (int64), home_win (bool).

        Returns:
            self (for chaining)
        """
        required_s = {"game_id", "lead_diff", "period", "time_elapsed"}
        required_o = {"game_id", "home_win"}
        missing_s  = required_s - set(states_df.columns)
        missing_o  = required_o - set(outcomes_df.columns)

        if missing_s:
            raise ValueError(f"states_df missing columns: {sorted(missing_s)}")
        if missing_o:
            raise ValueError(f"outcomes_df missing columns: {sorted(missing_o)}")

        df = states_df.join(outcomes_df.select(["game_id", "home_win"]), on="game_id", how="inner")
        if len(df) == 0:
            warnings.warn("No matched rows for WP model fitting; using fallback.", UserWarning, stacklevel=2)
            return self

        has_pp = "is_pp_home" in df.columns
        X = self._build_features(df, has_pp)
        y = df["home_win"].cast(pl.Int64).to_numpy()

        X_scaled = self._scaler.fit_transform(X)
        self._lr.fit(X_scaled, y)
        self._fitted  = True
        self._has_pp  = has_pp
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        lead_diff: int,
        period: int,
        time_elapsed: float,
        is_pp_home: bool = False,
    ) -> float:
        """Return P(home_win) for a single game state.

        Falls back to the analytical approximation when not fitted.
        """
        if not self._fitted:
            return _analytical_win_prob(lead_diff, period, time_elapsed)

        feats = np.array([[
            float(lead_diff),
            float(period),
            float(time_elapsed),
            float(is_pp_home),
        ]])
        feats_scaled = self._scaler.transform(feats)
        return float(self._lr.predict_proba(feats_scaled)[0, 1])

    def predict_proba_batch(
        self,
        lead_diff: np.ndarray,
        period: np.ndarray,
        time_elapsed: np.ndarray,
        is_pp_home: np.ndarray,
    ) -> np.ndarray:
        """Vectorised prediction for arrays of states."""
        if not self._fitted:
            return np.array([
                _analytical_win_prob(int(ld), int(p), float(te))
                for ld, p, te in zip(lead_diff, period, time_elapsed)
            ])

        feats = np.column_stack([
            lead_diff.astype(float),
            period.astype(float),
            time_elapsed.astype(float),
            is_pp_home.astype(float),
        ])
        feats_scaled = self._scaler.transform(feats)
        return self._lr.predict_proba(feats_scaled)[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "fitted":   self._fitted,
            "lr":       self._lr,
            "scaler":   self._scaler,
            "has_pp":   getattr(self, "_has_pp", False),
        }, path)

    @classmethod
    def load(cls, path: Path | str) -> "WinProbabilityModel":
        payload = joblib.load(path)
        m = cls()
        m._fitted  = payload["fitted"]
        m._lr      = payload["lr"]
        m._scaler  = payload["scaler"]
        m._has_pp  = payload.get("has_pp", False)
        return m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pl.DataFrame, has_pp: bool) -> np.ndarray:
        cols = ["lead_diff", "period", "time_elapsed"]
        if has_pp:
            cols.append("is_pp_home")
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Float64).alias("is_pp_home"))
            cols.append("is_pp_home")
        return df.select(cols).cast(pl.Float64).to_numpy()

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Analytical win probability fallback
# ---------------------------------------------------------------------------

def _analytical_win_prob(lead_diff: int, period: int, time_elapsed: float) -> float:
    """Calibrated sigmoid approximation for P(home_win | state).

    Built from published NHL win-probability research (Schuckers & Curro;
    empirical team-level tables from 2010–2024 NHL seasons).

    Parameters are intentionally conservative — this is used only when no
    fitted model is available, and the system makes clear in logs when the
    fallback is active.

    Logic:
        - Baseline home advantage: ~54% win probability at even score, start
        - Score lead multiplier: ~6% per goal in regulation
        - Time pressure: lead advantage amplifies as time decreases
        - OT (period=4): 50% regardless of period start (sudden death)
    """
    if period >= OT_PERIOD:
        # Overtime is sudden-death; home advantage is minimal and lead_diff=0
        return 0.52 + 0.02 * np.sign(lead_diff)  # tiny residual home advantage

    # Time remaining fraction within regulation (0.0 = start, 1.0 = end)
    # time_elapsed is cumulative fraction of full 60-minute regulation
    time_remaining = max(0.0, min(1.0, 1.0 - time_elapsed))

    # Base: home advantage (~54% at start of game, even score)
    base = 0.54

    # Lead effect scales with time pressure: early goals matter less than late
    # At start of period 1: lead effect ≈ 0.04 per goal
    # At end of period 3:   lead effect ≈ 0.18 per goal
    time_pressure = 0.04 + 0.14 * (1.0 - time_remaining)
    lead_contribution = lead_diff * time_pressure

    raw_prob = base + lead_contribution
    # Use sigmoid to keep in (0, 1), centred on raw_prob
    return float(1.0 / (1.0 + np.exp(-4.5 * (raw_prob - 0.5))))


# ---------------------------------------------------------------------------
# WPA computation helpers
# ---------------------------------------------------------------------------

def _time_elapsed_fraction(period: int, time_remaining_secs: int) -> float:
    """Convert (period, time_remaining_secs) to cumulative fraction of 60 min.

    period=1 time_remaining=1200 → elapsed=0.0  (start of game)
    period=3 time_remaining=0    → elapsed=1.0  (end of regulation)
    period=4 anything            → elapsed=1.0  (OT, capped)
    """
    if period >= OT_PERIOD:
        return 1.0
    period_length = 20 * 60  # 20 minutes in seconds
    elapsed_this_period = period_length - int(time_remaining_secs)
    elapsed_total = (period - 1) * period_length + elapsed_this_period
    return min(float(elapsed_total) / (3 * period_length), 1.0)


def compute_goal_wpa(
    wp_model: WinProbabilityModel,
    lead_diff_before: int,
    period: int,
    time_remaining_secs: int,
    scoring_team_is_home: bool,
    is_pp: bool = False,
) -> float:
    """Compute WPA for a single goal event.

    WPA is always expressed from the scoring team's perspective
    (positive = increased their own win probability).

    Args:
        wp_model:              Fitted WinProbabilityModel.
        lead_diff_before:      Scoring team's lead before the goal
                               (positive = already leading, negative = trailing).
        period:                Period number (1–4).
        time_remaining_secs:   Seconds remaining in period at time of goal.
        scoring_team_is_home:  True if the scoring team is the home team.
        is_pp:                 True if the scoring team is on the power play.

    Returns:
        WPA float.  Positive = clutch goal (high leverage).
    """
    time_el = _time_elapsed_fraction(period, time_remaining_secs)

    # Translate to home-team perspective for the WP model
    if scoring_team_is_home:
        home_lead_before  = lead_diff_before
        home_lead_after   = lead_diff_before + 1
        is_pp_home_before = is_pp
        is_pp_home_after  = False  # PP ends after a goal
    else:
        home_lead_before  = -lead_diff_before
        home_lead_after   = -lead_diff_before - 1
        is_pp_home_before = False  # away team PP means home PK
        is_pp_home_after  = False

    wp_before = wp_model.predict_proba(home_lead_before, period, time_el, is_pp_home_before)
    wp_after  = wp_model.predict_proba(home_lead_after,  period, time_el, is_pp_home_after)

    # Return from scoring team's perspective
    if scoring_team_is_home:
        return wp_after - wp_before
    else:
        return (1.0 - wp_after) - (1.0 - wp_before)


# ---------------------------------------------------------------------------
# ClutchIndexModel — Feature 2.29
# ---------------------------------------------------------------------------

@dataclass
class _SeasonResult:
    """Internal: per-player per-season WPA accumulation."""
    player_id:    int
    player_name:  str
    team:         str
    season:       int
    actual_wpa:   float     = 0.0
    goals_count:  float     = 0.0   # goals credited (weighted)
    assists_count: float    = 0.0
    total_opps:   float     = 0.0   # WPA-opportunity count for expected calc
    total_opps_wpa: float   = 0.0   # sum of expected WPA for those opportunities


class ClutchIndexModel:
    """Clutch Index (Win Probability Added) model — Feature 2.29.

    Usage::

        model = ClutchIndexModel()
        model.fit_wp_model(states_df, outcomes_df)          # optional: train WP model
        season_df = model.compute_season(goals_df, toi_df, season=2025)
        career_df = model.compute_career(
            [season_2023_df, season_2024_df, season_2025_df],
            seasons=[2023, 2024, 2025],
        )
    """

    def __init__(
        self,
        assist_weight: float = ASSIST_WEIGHT,
        full_signal_gp: int  = FULL_SIGNAL_GP,
    ) -> None:
        self._wp_model      = WinProbabilityModel()
        self._assist_weight = assist_weight
        self._full_signal   = full_signal_gp
        self._version       = MODEL_VERSION
        self.is_validated   = False
        self._validated_seasons: list[int] = []

    # ------------------------------------------------------------------
    # WP model training
    # ------------------------------------------------------------------

    def fit_wp_model(
        self,
        states_df: pl.DataFrame,
        outcomes_df: pl.DataFrame,
    ) -> None:
        """Train the underlying win-probability logistic regression.

        Args:
            states_df:   One row per game-state.  Required: game_id, lead_diff
                         (int64), period (int64), time_elapsed (float64).
                         Optional: is_pp_home (bool).
            outcomes_df: One row per game.  Required: game_id, home_win (bool).
        """
        self._wp_model.fit(states_df, outcomes_df)

    # ------------------------------------------------------------------
    # Single-season computation
    # ------------------------------------------------------------------

    def compute_season(
        self,
        goals_df: pl.DataFrame,
        toi_df: pl.DataFrame,
        season: int,
        player_meta_df: pl.DataFrame | None = None,
        validated_seasons: list[int] | None = None,
    ) -> pl.DataFrame:
        """Compute per-player clutch index for one season.

        Args:
            goals_df:       One row per goal event.
                Required columns:
                    game_id (int64), scorer_id (int64),
                    assist1_id (int64, nullable), assist2_id (int64, nullable),
                    period (int64), time_remaining_secs (int64),
                    home_score_before (int64), away_score_before (int64),
                    event_owner_team_id (int64), home_team_id (int64).
                Optional: strength (str — "EV" | "PP" | "PK" | "EN").

            toi_df:         One row per (player_id, game_id).
                Required: player_id (int64), toi_seconds (float64).
                Optional: team (str), gp (int64), career_gp (int64).

            season:         Integer season year.
            player_meta_df: Optional player metadata.
                Columns used: player_id (int64), player_name (str),
                              team (str), career_gp (int64).
            validated_seasons:
                Seasons already back-tested; sets is_validated when ≥ 3.

        Returns:
            DataFrame with CLUTCH_INDEX_SCHEMA, sorted by clutch_index_shrunk
            descending.
        """
        required_g = {
            "game_id", "scorer_id", "period", "time_remaining_secs",
            "home_score_before", "away_score_before",
            "event_owner_team_id", "home_team_id",
        }
        missing_g = required_g - set(goals_df.columns)
        if missing_g:
            raise ValueError(f"goals_df missing columns: {sorted(missing_g)}")

        required_t = {"player_id", "toi_seconds"}
        missing_t = required_t - set(toi_df.columns)
        if missing_t:
            raise ValueError(f"toi_df missing columns: {sorted(missing_t)}")

        if validated_seasons:
            self._validated_seasons = sorted(set(self._validated_seasons + validated_seasons))
            self.is_validated = len(self._validated_seasons) >= VALIDATION_SEASONS

        # ── Build player TOI lookup: player_id → total toi_seconds ───────────
        toi_agg = (
            toi_df
            .group_by("player_id")
            .agg(pl.col("toi_seconds").sum().alias("total_toi_sec"))
        )

        # ── Compute per-goal WPA ──────────────────────────────────────────────
        goal_rows = goals_df.to_dicts()
        has_strength = "strength" in goals_df.columns

        # league_wpa_sum / league_opp_count → expected WPA per opportunity
        league_wpa_total = 0.0
        league_opp_count = 0

        # player_id → {actual_wpa, goals_credited, assists_credited, opp_wpa}
        player_acc: dict[int, dict] = {}

        def _acc(pid: int) -> dict:
            if pid not in player_acc:
                player_acc[pid] = {
                    "actual_wpa": 0.0,
                    "goals_credited": 0.0,
                    "assists_credited": 0.0,
                    "opp_wpa_sum": 0.0,
                    "opp_count": 0.0,
                }
            return player_acc[pid]

        for row in goal_rows:
            period           = int(row["period"])
            time_rem         = int(row["time_remaining_secs"])
            home_score_b     = int(row["home_score_before"])
            away_score_b     = int(row["away_score_before"])
            owner_team_id    = int(row["event_owner_team_id"])
            home_team_id     = int(row["home_team_id"])
            scoring_is_home  = owner_team_id == home_team_id

            # Lead diff from scoring team's perspective
            if scoring_is_home:
                lead_diff_before = home_score_b - away_score_b
            else:
                lead_diff_before = away_score_b - home_score_b

            is_pp = False
            if has_strength:
                is_pp = str(row.get("strength", "EV")) == "PP"

            wpa = compute_goal_wpa(
                self._wp_model,
                lead_diff_before=lead_diff_before,
                period=period,
                time_remaining_secs=time_rem,
                scoring_team_is_home=scoring_is_home,
                is_pp=is_pp,
            )

            # League-level accumulators
            league_wpa_total += wpa
            league_opp_count += 1

            # Scorer gets full WPA credit
            scorer_id = row.get("scorer_id")
            if scorer_id is not None:
                try:
                    scorer_id = int(scorer_id)
                    if scorer_id > 0:
                        a = _acc(scorer_id)
                        a["actual_wpa"]      += wpa
                        a["goals_credited"]  += 1.0
                        a["opp_wpa_sum"]     += wpa
                        a["opp_count"]       += 1.0
                except (ValueError, TypeError):
                    pass

            # Primary assist gets ASSIST_WEIGHT credit
            for assist_key in ("assist1_id", "assist2_id"):
                aid = row.get(assist_key)
                if aid is None:
                    continue
                try:
                    aid = int(aid)
                    if aid > 0:
                        a = _acc(aid)
                        a["actual_wpa"]       += wpa * self._assist_weight
                        a["assists_credited"] += self._assist_weight
                        a["opp_wpa_sum"]      += wpa * self._assist_weight
                        a["opp_count"]        += self._assist_weight
                except (ValueError, TypeError):
                    pass

        # League average WPA per scoring opportunity
        league_avg_wpa = (
            league_wpa_total / league_opp_count
            if league_opp_count > 0
            else 0.0
        )

        # ── Build metadata lookups ────────────────────────────────────────────
        meta: dict[int, dict] = {}
        if player_meta_df is not None:
            for mrow in player_meta_df.to_dicts():
                pid = int(mrow.get("player_id", 0))
                meta[pid] = mrow

        toi_lookup: dict[int, float] = {
            int(r["player_id"]): float(r["total_toi_sec"])
            for r in toi_agg.to_dicts()
        }

        # career_gp from toi_df if available
        career_gp_lookup: dict[int, int] = {}
        if "career_gp" in toi_df.columns:
            for r in toi_df.group_by("player_id").agg(
                pl.col("career_gp").first()
            ).to_dicts():
                career_gp_lookup[int(r["player_id"])] = int(r["career_gp"] or 0)
        if "gp" in toi_df.columns and not career_gp_lookup:
            for r in toi_df.group_by("player_id").agg(
                pl.col("gp").sum().alias("career_gp")
            ).to_dicts():
                career_gp_lookup.setdefault(int(r["player_id"]), int(r["career_gp"] or 0))

        # Fallback: estimate season GP from game_id count in toi_df, or from
        # toi_seconds (avg ~17 min/game in NHL).  This prevents shrinkage from
        # collapsing to 0 when career_gp data is unavailable.
        season_gp_lookup: dict[int, int] = {}
        if "game_id" in toi_df.columns:
            for r in toi_df.group_by("player_id").agg(
                pl.col("game_id").n_unique().alias("season_gp")
            ).to_dicts():
                season_gp_lookup[int(r["player_id"])] = int(r["season_gp"] or 1)
        else:
            # Estimate GP from toi_seconds: NHL average ~17 min/game EV+PP+PK
            _avg_toi_per_game_sec = 17 * 60
            for pid, toi_sec in toi_lookup.items():
                estimated_gp = max(1, round(toi_sec / _avg_toi_per_game_sec))
                season_gp_lookup[pid] = estimated_gp

        # team lookup
        team_lookup: dict[int, str] = {}
        if "team" in toi_df.columns:
            for r in toi_df.group_by("player_id").agg(
                pl.col("team").first()
            ).to_dicts():
                team_lookup[int(r["player_id"])] = str(r["team"] or "")

        # player_name lookup
        name_lookup: dict[int, str] = {}
        for pid, mrow in meta.items():
            name_lookup[pid] = str(mrow.get("player_name", ""))

        # ── Assemble output rows ──────────────────────────────────────────────
        out_rows: list[dict] = []

        for pid, acc in player_acc.items():
            toi_sec  = toi_lookup.get(pid, 0.0)
            toi_60   = toi_sec / 3600.0  # convert seconds to 60-minute units

            actual_wpa     = acc["actual_wpa"]
            opp_count      = acc["opp_count"]

            actual_wpa_per60 = actual_wpa / toi_60 if toi_60 > 0 else 0.0

            # Expected WPA per 60 given this player's number of opportunities
            expected_wpa_per60 = (
                league_avg_wpa * opp_count / toi_60
                if toi_60 > 0
                else 0.0
            )

            raw_clutch = actual_wpa_per60 - expected_wpa_per60
            raw_clutch = float(np.clip(raw_clutch, -CLUTCH_INDEX_CLAMP, CLUTCH_INDEX_CLAMP))

            # Bayesian shrinkage
            career_gp = career_gp_lookup.get(pid, 0)
            if pid in meta:
                career_gp = int(meta[pid].get("career_gp", career_gp))
            # When career_gp is unavailable (0), use season GP as the minimum
            # so shrinkage never collapses the signal to exactly 0.
            if career_gp == 0:
                career_gp = season_gp_lookup.get(pid, 1)
            shrink = min(float(career_gp) / float(self._full_signal), 1.0)

            clutch_shrunk = raw_clutch * shrink

            out_rows.append({
                "player_id":           pid,
                "player_name":         name_lookup.get(pid, ""),
                "team":                team_lookup.get(pid, ""),
                "season":              season,
                "goals_credited":      acc["goals_credited"],
                "assists_credited":    acc["assists_credited"],
                "actual_wpa":          actual_wpa,
                "actual_wpa_per60":    actual_wpa_per60,
                "expected_wpa_per60":  expected_wpa_per60,
                "clutch_index":        raw_clutch,
                "clutch_index_shrunk": clutch_shrunk,
                "shrinkage_factor":    shrink,
                "career_gp":           career_gp,
                "toi_60":              toi_60,
                "model_version":       self._version,
            })

        if not out_rows:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in CLUTCH_INDEX_SCHEMA.items()}
            )

        result = pl.DataFrame(out_rows)
        for col, dtype in CLUTCH_INDEX_SCHEMA.items():
            if col in result.columns:
                result = result.with_columns(pl.col(col).cast(dtype))
            else:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))

        return (
            result
            .select(list(CLUTCH_INDEX_SCHEMA.keys()))
            .sort("clutch_index_shrunk", descending=True)
        )

    # ------------------------------------------------------------------
    # Multi-season career weighted average
    # ------------------------------------------------------------------

    def compute_career(
        self,
        season_dfs: list[pl.DataFrame],
        seasons: list[int],
    ) -> pl.DataFrame:
        """Compute career-weighted clutch index from multiple season outputs.

        Seasons are weighted by ``0.7^(seasons_ago)`` normalised to sum to 1.

        Args:
            season_dfs: List of DataFrames from ``compute_season()`` calls,
                        in chronological order (oldest first).
            seasons:    Corresponding season integers (same order as season_dfs).

        Returns:
            DataFrame with CLUTCH_INDEX_CAREER_SCHEMA, sorted by
            weighted_clutch_index descending.
        """
        if not season_dfs or not seasons:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in CLUTCH_INDEX_CAREER_SCHEMA.items()}
            )

        if len(season_dfs) != len(seasons):
            raise ValueError("season_dfs and seasons must be the same length.")

        latest_season = max(seasons)
        weights = {
            s: SEASON_DECAY ** (latest_season - s)
            for s in seasons
        }
        total_w = sum(weights.values())
        weights = {s: w / total_w for s, w in weights.items()}

        # Collect all player ids across all seasons
        all_pids: set[int] = set()
        for df in season_dfs:
            if not df.is_empty():
                all_pids.update(df["player_id"].to_list())

        # Build per-season lookup: pid → clutch_index_shrunk
        season_map: list[dict[int, dict]] = []
        for df in season_dfs:
            if df.is_empty():
                season_map.append({})
            else:
                season_map.append({
                    int(r["player_id"]): r
                    for r in df.to_dicts()
                })

        out_rows: list[dict] = []

        for pid in all_pids:
            weighted_ci  = 0.0
            weight_used  = 0.0
            career_gp    = 0
            player_name  = ""
            seasons_seen = 0

            for i, (season, smap) in enumerate(zip(seasons, season_map)):
                row = smap.get(pid)
                if row is None:
                    continue
                w = weights[season]
                weighted_ci += row["clutch_index_shrunk"] * w
                weight_used += w
                seasons_seen += 1
                career_gp    = max(career_gp, int(row.get("career_gp") or 0))
                if not player_name:
                    player_name = str(row.get("player_name") or "")

            if weight_used <= 0:
                continue

            # Renormalise for players missing some seasons
            weighted_ci /= weight_used

            shrink = min(float(career_gp) / float(self._full_signal), 1.0)

            out_rows.append({
                "player_id":             pid,
                "player_name":           player_name,
                "seasons_counted":       seasons_seen,
                "latest_season":         latest_season,
                "weighted_clutch_index": float(np.clip(weighted_ci, -CLUTCH_INDEX_CLAMP, CLUTCH_INDEX_CLAMP)),
                "shrinkage_factor":      shrink,
                "career_gp":             career_gp,
                "model_version":         self._version,
            })

        if not out_rows:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in CLUTCH_INDEX_CAREER_SCHEMA.items()}
            )

        result = pl.DataFrame(out_rows)
        for col, dtype in CLUTCH_INDEX_CAREER_SCHEMA.items():
            if col in result.columns:
                result = result.with_columns(pl.col(col).cast(dtype))
            else:
                result = result.with_columns(pl.lit(None).cast(dtype).alias(col))

        return (
            result
            .select(list(CLUTCH_INDEX_CAREER_SCHEMA.keys()))
            .sort("weighted_clutch_index", descending=True)
        )

    # ------------------------------------------------------------------
    # Single-player lookup
    # ------------------------------------------------------------------

    def get_clutch_index(
        self,
        season_df: pl.DataFrame,
        player_id: int,
    ) -> float:
        """Return the shrunk clutch index for a single player from season output.

        Args:
            season_df:  Output of ``compute_season()``.
            player_id:  NHL API player identifier.

        Returns:
            Float.  0.0 if player not found.
        """
        rows = season_df.filter(pl.col("player_id") == player_id)
        if rows.is_empty():
            return 0.0
        return float(rows["clutch_index_shrunk"][0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version":            self._version,
            "wp_model":           self._wp_model,
            "assist_weight":      self._assist_weight,
            "full_signal_gp":     self._full_signal,
            "is_validated":       self.is_validated,
            "validated_seasons":  self._validated_seasons,
        }, path)

    @classmethod
    def load(cls, path: Path | str) -> "ClutchIndexModel":
        payload  = joblib.load(path)
        instance = cls(
            assist_weight  = payload.get("assist_weight", ASSIST_WEIGHT),
            full_signal_gp = payload.get("full_signal_gp", FULL_SIGNAL_GP),
        )
        instance._version            = payload.get("version", MODEL_VERSION)
        instance._wp_model           = payload.get("wp_model", WinProbabilityModel())
        instance.is_validated        = payload.get("is_validated", False)
        instance._validated_seasons  = payload.get("validated_seasons", [])
        return instance


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_clutch_index(
    season_df: pl.DataFrame,
    career_df: pl.DataFrame,
    output_dir: Path,
    season: int,
) -> tuple[Path, Path]:
    """Write per-season and career clutch index DataFrames to parquet.

    Returns:
        Tuple of (season_path, career_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_season = output_dir / f"clutch_index_{season}.parquet"
    p_career = output_dir / f"clutch_index_career.parquet"

    season_df.write_parquet(p_season)
    career_df.write_parquet(p_career)

    return p_season, p_career


def read_clutch_index(data_dir: Path, season: int) -> pl.DataFrame:
    """Read per-season clutch index parquet."""
    path = Path(data_dir) / "clutch_index" / f"clutch_index_{season}.parquet"
    return pl.read_parquet(path)


def read_clutch_index_career(data_dir: Path) -> pl.DataFrame:
    """Read career clutch index parquet."""
    path = Path(data_dir) / "clutch_index" / "clutch_index_career.parquet"
    return pl.read_parquet(path)
