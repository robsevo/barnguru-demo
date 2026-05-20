"""Per-Player Behavioral Neural Network — Feature 2.22.

Hierarchical model: shared base network (universal hockey player) +
per-player adaptation layers. Learns from observed player behaviour via
imitation learning (V1); RL fine-tuning is planned for Feature 14.x (V2).

Architecture (V1):
    1. Base action profiles — per-player 7-action tendency vectors built
       from season stats (carry%, positional zones, battle score).
    2. Context adjustment network — small sklearn MLP that maps context
       feature vectors to 7-dim logit deltas.  Trained on synthetic context
       variation derived from domain-knowledge rules:
           • High FI  → more dump-and-chase, less slot shooting
           • Losing   → more perimeter shooting, urgency actions
           • Winning  → more corner cycling, puck protection
           • Elite opp → more dumps, less carry-ins
           • Low speed → more battles, less net-drives
    3. Per-player bias — residual logit vector per player capturing
       idiosyncratic tendencies not explained by context.

    Final output: softmax(base_logits + player_bias + context_Δ)

Actions (7):
    carry_in          controlled zone entry (carry puck in)
    dump              dump-and-chase zone entry
    shoot_slot        shot from slot / net-front (high danger)
    shoot_perimeter   shot from circle / half-wall / point
    battle_corner     puck battle / scrum in corner
    drive_net         drive to net front / screen / tip
    hold_corner       cycle, possess, hold puck in corner

Inputs
------
player_stats_df : Player season stats with carry_entry_pct and optional
                  battle_score (from Feature 2.18 puck battle output).
positional_df   : Optional positional heatmap DataFrame (Feature 2.21 output).
skating_df      : Optional skating baseline DataFrame (Feature 2.20 output).

Context features for predict():
    fi_score         Fatigue index [0, 1]
    period           Game period (1, 2, 3, 4=OT)
    score_diff       Goal differential for this player's team (clipped ±3)
    home             True if player's team is at home
    manpower         Skater differential (+ = PP, − = PK, 0 = EV)
    matchup_quality  Opponent line quality z-score (from Feature 2.3)
    speed_ratio      Current speed / baseline speed (from 2.20; 1.0 = rested)
    carry_ratio      Current carry% / baseline carry% (from 2.20; 1.0 = rested)

Outputs
-------
``behavior_predictions_{season}.parquet`` — one row per player × context:
    player_id, player_name, team, season, fi_score,
    carry_in, dump, shoot_slot, shoot_perimeter,
    battle_corner, drive_net, hold_corner,
    model_version
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "behavior_v1"

ACTION_LABELS: list[str] = [
    "carry_in",
    "dump",
    "shoot_slot",
    "shoot_perimeter",
    "battle_corner",
    "drive_net",
    "hold_corner",
]
N_ACTIONS = len(ACTION_LABELS)

# Context feature names (same order as the feature vector)
CONTEXT_FEATURE_NAMES: list[str] = [
    "fi_score",          # fatigue index [0, 1]
    "period_norm",       # period mapped to [0, 1] via (period − 1) / 3
    "score_diff_norm",   # score_diff clipped ±3 then / 3 → [−1, 1]
    "home_indicator",    # 1 = home, 0 = away
    "manpower_norm",     # skater diff clipped ±2 then / 2 → [−1, 1]
    "matchup_quality",   # opponent quality z-score (clipped ±2)
    "speed_ratio",       # current speed / baseline speed (clipped [0, 2])
    "carry_ratio",       # current carry / baseline carry (clipped [0, 2])
]
N_CONTEXT = len(CONTEXT_FEATURE_NAMES)

# Synthetic training data
N_SYNTHETIC = 5_000       # number of synthetic context vectors to generate
SYNTHETIC_SEED = 42

# Default base action distribution (when no player-specific data available)
_DEFAULT_BASE = np.array([0.22, 0.22, 0.15, 0.18, 0.10, 0.07, 0.06], dtype=np.float64)

# Output schema for bulk prediction
BEHAVIOR_PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "player_id":       pl.Int64,
    "player_name":     pl.Utf8,
    "team":            pl.Utf8,
    "season":          pl.Int64,
    "fi_score":        pl.Float64,
    "carry_in":        pl.Float64,
    "dump":            pl.Float64,
    "shoot_slot":      pl.Float64,
    "shoot_perimeter": pl.Float64,
    "battle_corner":   pl.Float64,
    "drive_net":       pl.Float64,
    "hold_corner":     pl.Float64,
    "model_version":   pl.Utf8,
}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _logits_from_probs(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probability vector to log-odds (inverse softmax approximation)."""
    p = np.clip(probs, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


# ---------------------------------------------------------------------------
# Domain-knowledge context adjustment rules
# ---------------------------------------------------------------------------

def _apply_context_rules(
    base_probs: np.ndarray,
    fi_score: float,
    period_norm: float,
    score_diff_norm: float,
    manpower_norm: float,
    matchup_quality: float,
    speed_ratio: float,
    carry_ratio: float,
) -> np.ndarray:
    """Apply domain-knowledge adjustments to base action logits.

    Returns adjusted probability vector (after softmax).  Used to generate
    synthetic training labels for the context MLP.

    Rules (validated by Bob's domain knowledge):
        - High FI: dump-and-chase↑, shoot_slot↓, carry_in↓, battle_corner↑
        - Losing (score_diff < 0): shoot_perimeter↑, dump↑ (urgency)
        - Winning (score_diff > 0): hold_corner↑, dump↑ (protect lead)
        - OT (period_norm = 1.0): carry_in↑, shoot_slot↑ (must score)
        - PP (manpower > 0): carry_in↑, shoot_slot↑, shoot_perimeter↑
        - PK (manpower < 0): dump↑, battle_corner↑
        - Elite opponent (matchup_quality > 0): dump↑, carry_in↓
        - Low speed ratio: battle_corner↑, dump↑, drive_net↓
        - Low carry ratio: dump↑, carry_in↓
    """
    logits = np.log(np.clip(base_probs, 1e-6, 1.0))
    # Indices for readability
    ci, dp, ss, sp, bc, dn, hc = 0, 1, 2, 3, 4, 5, 6

    # FI effect
    logits[dp] += 0.60 * fi_score
    logits[ci] -= 0.45 * fi_score
    logits[ss] -= 0.30 * fi_score
    logits[bc] += 0.25 * fi_score
    logits[dn] -= 0.20 * fi_score

    # Losing urgency
    deficit = max(0.0, -score_diff_norm)
    logits[sp] += 0.35 * deficit
    logits[dp] += 0.20 * deficit

    # Winning (puck protection)
    lead = max(0.0, score_diff_norm)
    logits[hc] += 0.30 * lead
    logits[dp] += 0.15 * lead

    # Overtime (period_norm == 1.0 → 0.33 for period 4 in [1-4] mapping)
    ot_factor = max(0.0, period_norm - 0.67)  # kicks in for OT
    logits[ci] += 0.40 * ot_factor
    logits[ss] += 0.30 * ot_factor

    # Power play / penalty kill
    pp = max(0.0, manpower_norm)
    pk = max(0.0, -manpower_norm)
    logits[ci] += 0.30 * pp
    logits[ss] += 0.25 * pp
    logits[dp] += 0.50 * pk
    logits[bc] += 0.30 * pk

    # Matchup quality (opponent strength)
    opp = max(0.0, matchup_quality / 2.0)  # normalise 0–1
    logits[dp] += 0.25 * opp
    logits[ci] -= 0.20 * opp

    # Speed degradation
    speed_drop = max(0.0, 1.0 - speed_ratio)
    logits[bc] += 0.20 * speed_drop
    logits[dp] += 0.15 * speed_drop
    logits[dn] -= 0.15 * speed_drop

    # Carry degradation
    carry_drop = max(0.0, 1.0 - carry_ratio)
    logits[dp] += 0.30 * carry_drop
    logits[ci] -= 0.25 * carry_drop

    return _softmax(logits)


# ---------------------------------------------------------------------------
# Build base action profile from player stats
# ---------------------------------------------------------------------------

def _build_base_profile(
    player_row: dict[str, Any],
    positional_row: dict[str, Any] | None,
    skating_row: dict[str, Any] | None = None,
    position: str | None = None,
) -> np.ndarray:
    """Construct a 7-element action probability vector for one player.

    Sources:
        carry_in / dump      ← carry_entry_pct (puck battle model). When NaN
                               (no EDGE data, the common case in the public
                               feeds), fall back to a position-aware prior so
                               centers carry, wingers are middling, and Ds
                               dump. Modulated by skating burst if available.
        shoot_slot           ← off_net_front_pct + off_slot_pct (positional)
        shoot_perimeter      ← off_circle_pct + off_half_wall_pct + off_point_pct
        battle_corner        ← battle_score (normalised via sigmoid)
        drive_net            ← off_net_front_pct (tendency to go to net)
        hold_corner          ← off_corner_pct + carry cycling proxy

    All raw signals are combined into a raw non-negative weight vector then
    normalised to sum to 1 via softmax.
    """
    # Carry entry / dump.
    # When carry_entry_pct is missing (the typical case — EDGE controlled-entry
    # data is unavailable for ~all public-feed skaters), default per position:
    # centers ~62%, wingers ~55%, defensemen ~38%. Then nudge by skating burst
    # so a fast-skating D still carries more than a plodding one. Without this,
    # every player collapses to 0.50 / 0.50 and the entry side has zero signal.
    _cep = player_row.get("carry_entry_pct")
    carry_missing = _cep is None or (isinstance(_cep, float) and np.isnan(_cep))
    if carry_missing:
        pos = (position or "").upper()
        if pos == "C":
            carry_pct = 0.62
        elif pos in ("L", "R", "LW", "RW"):
            carry_pct = 0.55
        elif pos == "D":
            carry_pct = 0.38
        else:
            carry_pct = 0.50

        if skating_row:
            _ms = skating_row.get("baseline_max_speed_kmh")
            if _ms is not None and not (isinstance(_ms, float) and np.isnan(_ms)):
                # League max_speed ≈ N(35.7, 1.3) — see skating_baseline_2025.
                speed_z = (float(_ms) - 35.7) / 1.3
                carry_pct += float(np.clip(speed_z * 0.05, -0.10, 0.12))
    else:
        carry_pct = float(_cep)
    carry_pct = float(np.clip(carry_pct, 0.05, 0.95))

    # Positional
    if positional_row:
        nf   = float(positional_row.get("off_net_front_pct") or 0.0)
        sl   = float(positional_row.get("off_slot_pct")      or 0.0)
        ci_p = float(positional_row.get("off_circle_pct")    or 0.0)
        hw   = float(positional_row.get("off_half_wall_pct") or 0.0)
        pt   = float(positional_row.get("off_point_pct")     or 0.0)
        co   = float(positional_row.get("off_corner_pct")    or 0.0)
    else:
        nf = sl = ci_p = hw = pt = co = 1.0 / 6.0

    # Battle score (may be z-scored, so map via sigmoid) — guard NaN
    _bs = player_row.get("battle_score")
    battle_raw = 0.0 if (_bs is None or (isinstance(_bs, float) and np.isnan(_bs))) else float(_bs)
    battle_sig = 1.0 / (1.0 + np.exp(-battle_raw))  # sigmoid → (0,1)

    raw = np.array([
        carry_pct,                   # carry_in
        1.0 - carry_pct,             # dump
        nf + sl,                     # shoot_slot
        ci_p + hw + pt,              # shoot_perimeter
        battle_sig * 0.8,            # battle_corner (scaled)
        nf * 1.2,                    # drive_net (net-front tendency × boost)
        co + (1.0 - carry_pct) * 0.1,  # hold_corner
    ], dtype=np.float64)

    # Replace any zeros with a small floor to avoid log(0)
    raw = np.maximum(raw, 1e-4)
    return _softmax(np.log(raw))


# ---------------------------------------------------------------------------
# PlayerBehaviorNet — Feature 2.22
# ---------------------------------------------------------------------------

class PlayerBehaviorNet:
    """Per-player behavioral neural network (Feature 2.22, V1 — imitation learning).

    Usage::

        net = PlayerBehaviorNet()
        net.fit(player_stats_df, positional_df, skating_baseline_df)

        probs = net.predict(player_id=8478402, fi_score=0.6, period=3,
                            score_diff=-1, home=True, manpower=0,
                            matchup_quality=0.5, speed_ratio=0.92,
                            carry_ratio=0.88)
        # probs → {'carry_in': 0.17, 'dump': 0.28, 'shoot_slot': 0.13, ...}
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self._hidden       = hidden_layer_sizes
        self._max_iter     = max_iter
        self._random_state = random_state
        self._version      = MODEL_VERSION

        # Set after fit()
        self._base_profiles: dict[int, np.ndarray] = {}      # player_id → 7-d probs
        self._player_bias:   dict[int, np.ndarray] = {}      # player_id → 7-d logit bias
        self._player_meta:   dict[int, dict[str, str]] = {}  # player_id → {name, team}
        self._context_net:   MLPRegressor | None = None
        self._scaler:        StandardScaler | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Context feature encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode_context(
        fi_score: float = 0.0,
        period: int = 2,
        score_diff: int = 0,
        home: bool = True,
        manpower: int = 0,
        matchup_quality: float = 0.0,
        speed_ratio: float = 1.0,
        carry_ratio: float = 1.0,
    ) -> np.ndarray:
        """Encode raw context values into the 8-dim context feature vector."""
        return np.array([
            float(np.clip(fi_score, 0.0, 1.0)),
            (np.clip(period, 1, 4) - 1) / 3.0,
            float(np.clip(score_diff, -3, 3)) / 3.0,
            1.0 if home else 0.0,
            float(np.clip(manpower, -2, 2)) / 2.0,
            float(np.clip(matchup_quality, -2.0, 2.0)),
            float(np.clip(speed_ratio, 0.0, 2.0)),
            float(np.clip(carry_ratio, 0.0, 2.0)),
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def fit(
        self,
        player_stats_df: pl.DataFrame,
        positional_df: pl.DataFrame | None = None,
        skating_baseline_df: pl.DataFrame | None = None,
        cv_observations_df: pl.DataFrame | None = None,
        position_lookup: dict[int, str] | None = None,
    ) -> "PlayerBehaviorNet":
        """Fit the behavioral network from player season stats.

        Args:
            player_stats_df:     Player stats DataFrame with at least
                                 ``player_id`` and ``carry_entry_pct``.
                                 Optional: ``battle_score``, ``player_name``,
                                 ``team``.
            positional_df:       Feature 2.21 output (positional heatmap).
            skating_baseline_df: Feature 2.20 output (skating baselines).
            cv_observations_df:  Feature 16.26 — aggregated in-browser CV
                                 observations. Only passed when the
                                 ``cv_gate.json`` flag is ON; otherwise the
                                 caller leaves this as None. Currently
                                 logged-only — per-player CV features land
                                 once track_id→player_id mapping is trusted.

        Returns:
            self (for chaining).

        Raises:
            ValueError: If ``player_id`` column is missing.
        """
        if cv_observations_df is not None and len(cv_observations_df) > 0:
            # We intentionally don't join into the MLP yet: track_id → NHL
            # player_id mapping requires jersey OCR confidence we haven't
            # validated. Logging only. When the mapping is trusted, this
            # block turns into a lookup built into the feature vector.
            self._cv_feed_seen = int(len(cv_observations_df))
        else:
            self._cv_feed_seen = 0


        if "player_id" not in (player_stats_df.columns
                                if hasattr(player_stats_df, "columns") else []):
            raise ValueError("player_stats_df must contain 'player_id' column.")

        if len(player_stats_df) == 0:
            warnings.warn(
                "player_stats_df is empty — no base profiles built.",
                DataMissingWarning,
                stacklevel=2,
            )
            self._fitted = True
            return self

        # Build lookups for optional supplementary tables
        pos_lookup: dict[int, dict[str, Any]] = {}
        if positional_df is not None and len(positional_df) > 0:
            for row in positional_df.to_dicts():
                pid = row.get("player_id")
                if pid is not None:
                    pos_lookup[int(pid)] = row

        skate_lookup: dict[int, dict[str, Any]] = {}
        if skating_baseline_df is not None and len(skating_baseline_df) > 0:
            for row in skating_baseline_df.to_dicts():
                pid = row.get("player_id")
                if pid is not None:
                    skate_lookup[int(pid)] = row

        # ------------------------------------------------------------------
        # Step 1: Build per-player base action profiles
        # ------------------------------------------------------------------
        for row in player_stats_df.to_dicts():
            pid = row.get("player_id")
            if pid is None:
                continue
            pid = int(pid)
            pos_row = pos_lookup.get(pid)
            skate_row = skate_lookup.get(pid)
            position = (position_lookup or {}).get(pid) or row.get("position")
            base_probs = _build_base_profile(row, pos_row, skate_row, position)
            self._base_profiles[pid] = base_probs
            self._player_meta[pid] = {
                "player_name": str(row.get("player_name") or ""),
                "team":        str(row.get("team") or ""),
            }

        if not self._base_profiles:
            self._fitted = True
            return self

        # ------------------------------------------------------------------
        # Step 2: Generate synthetic training data for context network
        # ------------------------------------------------------------------
        rng = np.random.default_rng(SYNTHETIC_SEED)
        X_train: list[np.ndarray] = []
        y_train: list[np.ndarray] = []

        # Random context samples
        fi_vals     = rng.uniform(0.0, 1.0, N_SYNTHETIC)
        period_vals = rng.integers(1, 5, N_SYNTHETIC)
        sd_vals     = rng.integers(-3, 4, N_SYNTHETIC)
        home_vals   = rng.integers(0, 2, N_SYNTHETIC)
        mp_vals     = rng.integers(-2, 3, N_SYNTHETIC)
        mq_vals     = rng.uniform(-2.0, 2.0, N_SYNTHETIC)
        sp_vals     = rng.uniform(0.5, 1.2, N_SYNTHETIC)
        cr_vals     = rng.uniform(0.5, 1.2, N_SYNTHETIC)

        # Use the average base profile across all players as the "average player"
        # Filter out any profiles that contain NaN (shouldn't happen but guard)
        clean_profiles = [p for p in self._base_profiles.values() if not np.any(np.isnan(p))]
        if not clean_profiles:
            self._fitted = True
            return self
        avg_base = np.mean(clean_profiles, axis=0)

        for i in range(N_SYNTHETIC):
            ctx = self.encode_context(
                fi_score        = float(fi_vals[i]),
                period          = int(period_vals[i]),
                score_diff      = int(sd_vals[i]),
                home            = bool(home_vals[i]),
                manpower        = int(mp_vals[i]),
                matchup_quality = float(mq_vals[i]),
                speed_ratio     = float(sp_vals[i]),
                carry_ratio     = float(cr_vals[i]),
            )
            # Label = domain-knowledge adjusted target probs
            target_probs = _apply_context_rules(
                base_probs      = avg_base.copy(),
                fi_score        = float(fi_vals[i]),
                period_norm     = ctx[1],
                score_diff_norm = ctx[2],
                manpower_norm   = ctx[4],
                matchup_quality = float(mq_vals[i]),
                speed_ratio     = float(sp_vals[i]),
                carry_ratio     = float(cr_vals[i]),
            )
            # Target is the logit delta from the base
            base_logits = np.log(np.clip(avg_base, 1e-6, 1.0))
            target_logits = np.log(np.clip(target_probs, 1e-6, 1.0))
            delta = target_logits - base_logits

            X_train.append(ctx)
            y_train.append(delta)

        X_arr = np.array(X_train, dtype=np.float64)
        y_arr = np.array(y_train, dtype=np.float64)

        # ------------------------------------------------------------------
        # Step 3: Train shared context adjustment MLP
        # ------------------------------------------------------------------
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        self._context_net = MLPRegressor(
            hidden_layer_sizes = self._hidden,
            max_iter           = self._max_iter,
            random_state       = self._random_state,
            early_stopping     = True,
            n_iter_no_change   = 20,
        )
        self._context_net.fit(X_scaled, y_arr)

        # ------------------------------------------------------------------
        # Step 4: Fit per-player bias vectors
        # Bias = difference between player base logits and average base logits
        # ------------------------------------------------------------------
        avg_logits = np.log(np.clip(avg_base, 1e-6, 1.0))
        for pid, base_probs in self._base_profiles.items():
            player_logits = np.log(np.clip(base_probs, 1e-6, 1.0))
            self._player_bias[pid] = player_logits - avg_logits

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        player_id: int,
        *,
        fi_score: float = 0.0,
        period: int = 2,
        score_diff: int = 0,
        home: bool = True,
        manpower: int = 0,
        matchup_quality: float = 0.0,
        speed_ratio: float = 1.0,
        carry_ratio: float = 1.0,
    ) -> dict[str, float]:
        """Return action probabilities for a player in a given context.

        Args:
            player_id:       NHL player identifier.
            fi_score:        Fatigue index [0, 1] (0 = fully rested).
            period:          Game period (1–3; 4 = overtime).
            score_diff:      Team goal differential (negative = trailing).
            home:            True if player is at home.
            manpower:        Skater differential (1 = PP, −1 = PK, 0 = EV).
            matchup_quality: Opponent quality z-score (from Feature 2.3).
            speed_ratio:     Current speed / baseline speed (1.0 = rested).
            carry_ratio:     Current carry% / baseline carry% (1.0 = rested).

        Returns:
            Dict mapping each of the 7 action labels to a probability in
            [0, 1] that sums to 1.0.
        """
        pid = int(player_id)
        ctx = self.encode_context(
            fi_score=fi_score, period=period, score_diff=score_diff,
            home=home, manpower=manpower, matchup_quality=matchup_quality,
            speed_ratio=speed_ratio, carry_ratio=carry_ratio,
        )

        # Base logits — use player profile if known, else league average
        if pid in self._base_profiles:
            base_probs = self._base_profiles[pid]
        else:
            base_probs = _DEFAULT_BASE.copy()
        base_logits = np.log(np.clip(base_probs, 1e-6, 1.0))

        # Per-player bias
        bias = self._player_bias.get(pid, np.zeros(N_ACTIONS, dtype=np.float64))

        # Context adjustment from MLP
        delta = np.zeros(N_ACTIONS, dtype=np.float64)
        if self._fitted and self._context_net is not None and self._scaler is not None:
            ctx_scaled = self._scaler.transform(ctx.reshape(1, -1))
            delta = self._context_net.predict(ctx_scaled)[0]

        combined_logits = base_logits + bias + delta
        probs = _softmax(combined_logits)

        return {label: float(probs[i]) for i, label in enumerate(ACTION_LABELS)}

    def predict_batch(
        self,
        player_ids: list[int],
        contexts: np.ndarray,
    ) -> pl.DataFrame:
        """Vectorised batch prediction.

        Args:
            player_ids: List of N player_id integers.
            contexts:   (N × 8) array of encoded context feature vectors
                        (use encode_context() per row).

        Returns:
            DataFrame with player_id + action probability columns.
        """
        rows = []
        for pid, ctx in zip(player_ids, contexts):
            pid = int(pid)
            base_probs  = self._base_profiles.get(pid, _DEFAULT_BASE.copy())
            base_logits = np.log(np.clip(base_probs, 1e-6, 1.0))
            bias = self._player_bias.get(pid, np.zeros(N_ACTIONS))

            delta = np.zeros(N_ACTIONS, dtype=np.float64)
            if self._fitted and self._context_net is not None and self._scaler is not None:
                ctx_scaled = self._scaler.transform(ctx.reshape(1, -1))
                delta = self._context_net.predict(ctx_scaled)[0]

            probs = _softmax(base_logits + bias + delta)
            meta  = self._player_meta.get(pid, {"player_name": "", "team": ""})
            rows.append({
                "player_id":   pid,
                "player_name": meta["player_name"],
                "team":        meta["team"],
                **{label: float(probs[i]) for i, label in enumerate(ACTION_LABELS)},
            })

        if not rows:
            return pl.DataFrame(schema={
                "player_id": pl.Int64, "player_name": pl.Utf8, "team": pl.Utf8,
                **{a: pl.Float64 for a in ACTION_LABELS},
            })
        return pl.DataFrame(rows)

    def action_labels(self) -> list[str]:
        """Return the list of action labels in model order."""
        return list(ACTION_LABELS)

    def player_ids(self) -> list[int]:
        """Return list of player_ids with fitted base profiles."""
        return list(self._base_profiles.keys())

    # ------------------------------------------------------------------
    # Bulk prediction helper (all players, resting vs. fatigued)
    # ------------------------------------------------------------------

    def predict_season(
        self,
        season: int,
        fi_scores: dict[int, float] | None = None,
    ) -> pl.DataFrame:
        """Predict action distributions for all fitted players.

        Args:
            season:    Season year to attach.
            fi_scores: Optional mapping player_id → fi_score.  If None,
                       uses fi_score = 0.0 (rested baseline) for all.

        Returns:
            DataFrame matching BEHAVIOR_PREDICTION_SCHEMA.
        """
        rows = []
        for pid in self._base_profiles:
            fi = (fi_scores or {}).get(pid, 0.0)
            probs = self.predict(pid, fi_score=fi)
            meta = self._player_meta.get(pid, {"player_name": "", "team": ""})
            rows.append({
                "player_id":   pid,
                "player_name": meta["player_name"],
                "team":        meta["team"],
                "season":      season,
                "fi_score":    fi,
                **probs,
                "model_version": self._version,
            })

        if not rows:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt)
                 for col, dt in BEHAVIOR_PREDICTION_SCHEMA.items()}
            )

        df = pl.DataFrame(rows)
        for col, dtype in BEHAVIOR_PREDICTION_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df.select(list(BEHAVIOR_PREDICTION_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "hidden":         self._hidden,
            "max_iter":       self._max_iter,
            "random_state":   self._random_state,
            "version":        self._version,
            "base_profiles":  self._base_profiles,
            "player_bias":    self._player_bias,
            "player_meta":    self._player_meta,
            "context_net":    self._context_net,
            "scaler":         self._scaler,
            "fitted":         self._fitted,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PlayerBehaviorNet":
        payload = joblib.load(Path(path))
        obj = cls(
            hidden_layer_sizes = payload.get("hidden",       (32, 16)),
            max_iter           = payload.get("max_iter",     500),
            random_state       = payload.get("random_state", 42),
        )
        obj._version       = payload.get("version",       MODEL_VERSION)
        obj._base_profiles = payload.get("base_profiles", {})
        obj._player_bias   = payload.get("player_bias",   {})
        obj._player_meta   = payload.get("player_meta",   {})
        obj._context_net   = payload.get("context_net",   None)
        obj._scaler        = payload.get("scaler",        None)
        obj._fitted        = payload.get("fitted",        False)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_behavior_predictions(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write behavior prediction scores to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"behavior_predictions_{season}.parquet"
    for col, dtype in BEHAVIOR_PREDICTION_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(BEHAVIOR_PREDICTION_SCHEMA.keys())).write_parquet(path)
    return path
