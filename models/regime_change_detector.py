"""Regime Change Detector — Feature 2.14.

Statistical drift detector: if a player shows >1.5σ deviation from their
current posterior mean for 10+ consecutive games, flag a regime change and
recommend partially discounting the prior / accelerating the learning rate.

Bob can also manually override for known events (coaching change, trade,
injury return, contract year motivation).

Regime states:
    "none"       — no regime change detected
    "up_shift"   — sustained positive deviation (new level of play higher)
    "down_shift" — sustained negative deviation (new level of play lower)
    "manual_up"  — Bob-entered upward regime override
    "manual_down"— Bob-entered downward regime override

When a regime is detected:
    - prior sigma is multiplied by REGIME_PRIOR_SIGMA_MULT (default 1.5×)
      to reflect increased uncertainty about the true level
    - recommended obs_sigma divisor = REGIME_OBS_SIGMA_DIVISOR (default 1.5)
      so future observations are up-weighted (faster learning)
    - alert text is generated for Bob's review

Outputs
-------
``regime_alerts_{season}.parquet`` — one row per active/historical alert:
    player_id, player_name, season, triggered_at_game,
    regime_state, consecutive_games, mean_residual,
    sigma_multiplier, obs_sigma_divisor, reason, model_version
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION              = "regime_v1"
REGIME_SIGMA_THRESHOLD     = 1.5    # σ deviations to count toward streak
REGIME_CONSECUTIVE_TRIGGER = 10     # consecutive games needed to trigger
REGIME_WINDOW_SIZE         = 25     # rolling window for residual tracking
REGIME_PRIOR_SIGMA_MULT    = 1.5    # multiply prior sigma by this on trigger
REGIME_OBS_SIGMA_DIVISOR   = 1.5    # divide obs_sigma by this on trigger (up-weight new data)
REGIME_CLEAR_CONSECUTIVE   = 5      # consecutive neutral games to clear the flag


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

REGIME_ALERT_SCHEMA: dict[str, pl.DataType] = {
    "player_id":          pl.Int64,
    "player_name":        pl.Utf8,
    "season":             pl.Int64,
    "triggered_at_game":  pl.Int64,
    "regime_state":       pl.Utf8,
    "consecutive_games":  pl.Int64,
    "mean_residual":      pl.Float64,
    "sigma_multiplier":   pl.Float64,
    "obs_sigma_divisor":  pl.Float64,
    "reason":             pl.Utf8,
    "model_version":      pl.Utf8,
}


# ---------------------------------------------------------------------------
# Per-player regime state
# ---------------------------------------------------------------------------

@dataclass
class PlayerRegimeState:
    """Mutable regime tracking state for one player.

    Attributes:
        player_id:            NHL player ID.
        player_name:          Display name.
        regime_state:         Current regime flag.
        consecutive_above:    Consecutive games with residual > +σ_thr.
        consecutive_below:    Consecutive games with residual < −σ_thr.
        consecutive_neutral:  Consecutive games within ±σ_thr (clears flag).
        triggered_at_game:    Game number when current regime was first triggered.
        residuals:            Rolling window of recent residuals.
        reason:               Human-readable reason string.
        games_processed:      Total games processed for this player.
    """
    player_id:           int
    player_name:         str
    regime_state:        str = "none"
    consecutive_above:   int = 0
    consecutive_below:   int = 0
    consecutive_neutral: int = 0
    triggered_at_game:   int | None = None
    residuals:           deque = field(default_factory=lambda: deque(maxlen=REGIME_WINDOW_SIZE))
    reason:              str = ""
    games_processed:     int = 0

    def is_active(self) -> bool:
        return self.regime_state != "none"

    def mean_residual(self) -> float:
        if not self.residuals:
            return 0.0
        return float(np.mean(list(self.residuals)))


# ---------------------------------------------------------------------------
# RegimeChangeDetector — Feature 2.14
# ---------------------------------------------------------------------------

class RegimeChangeDetector:
    """Statistical regime change detector (Feature 2.14).

    Tracks per-player residuals (observed xGF/60 − posterior_mean) and
    flags regime changes when 10+ consecutive games show >1.5σ drift.

    Usage::

        detector = RegimeChangeDetector()

        # After each game, provide residual = observed − posterior_mean:
        alert = detector.update(player_id=8478402, player_name="Matthews",
                                observation=5.0, posterior_mean=3.2,
                                posterior_sigma=0.4, game_number=15)
        if alert:
            print(f"Regime change: {alert['regime_state']}")

        # Bob manual override:
        detector.manual_override(player_id=8478402, player_name="Matthews",
                                 direction="up", reason="new coach + contract year")
    """

    def __init__(
        self,
        sigma_threshold:      float = REGIME_SIGMA_THRESHOLD,
        consecutive_trigger:  int   = REGIME_CONSECUTIVE_TRIGGER,
        prior_sigma_mult:     float = REGIME_PRIOR_SIGMA_MULT,
        obs_sigma_divisor:    float = REGIME_OBS_SIGMA_DIVISOR,
        clear_consecutive:    int   = REGIME_CLEAR_CONSECUTIVE,
    ) -> None:
        self._sigma_thr       = sigma_threshold
        self._consec_trigger  = consecutive_trigger
        self._prior_sigma_mult = prior_sigma_mult
        self._obs_sigma_div   = obs_sigma_divisor
        self._clear_consec    = clear_consecutive
        self._version         = MODEL_VERSION
        self._states: dict[int, PlayerRegimeState] = {}
        self._alert_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        player_id:       int,
        player_name:     str,
        observation:     float,
        posterior_mean:  float,
        posterior_sigma: float,
        game_number:     int,
        season:          int = 0,
    ) -> dict[str, Any] | None:
        """Process one game observation and check for regime change.

        Args:
            player_id:       Player identifier.
            player_name:     Display name.
            observation:     Observed xGF/60 (or other metric).
            posterior_mean:  Current posterior mean.
            posterior_sigma: Current posterior standard deviation.
            game_number:     Game count for this player this season.
            season:          NHL season start-year (for alert log).

        Returns:
            Alert dict if a new regime change was just triggered, else None.
            Alert contains: player_id, player_name, regime_state,
            consecutive_games, mean_residual, sigma_multiplier,
            obs_sigma_divisor, triggered_at_game, reason.
        """
        state = self._states.get(player_id)
        if state is None:
            state = PlayerRegimeState(player_id=player_id, player_name=player_name)
            self._states[player_id] = state

        state.player_name = player_name
        state.games_processed += 1

        # Skip manual overrides from being overwritten by data
        if state.regime_state in ("manual_up", "manual_down"):
            state.residuals.append(observation - posterior_mean)
            return None

        if posterior_sigma < 1e-9:
            return None

        residual = observation - posterior_mean
        z_score  = residual / posterior_sigma
        state.residuals.append(residual)

        # Classify this game
        if z_score > self._sigma_thr:
            state.consecutive_above += 1
            state.consecutive_below  = 0
            state.consecutive_neutral = 0
        elif z_score < -self._sigma_thr:
            state.consecutive_below += 1
            state.consecutive_above  = 0
            state.consecutive_neutral = 0
        else:
            state.consecutive_above = 0
            state.consecutive_below = 0
            state.consecutive_neutral += 1

        # Clear existing regime if enough neutral games
        if state.is_active() and state.consecutive_neutral >= self._clear_consec:
            if state.regime_state in ("up_shift", "down_shift"):
                state.regime_state        = "none"
                state.triggered_at_game   = None
                state.consecutive_neutral = 0
                state.reason              = ""

        # Trigger new regime
        newly_triggered: dict[str, Any] | None = None

        if state.regime_state == "none":
            if state.consecutive_above >= self._consec_trigger:
                state.regime_state      = "up_shift"
                state.triggered_at_game = game_number
                state.reason            = (
                    f"Statistical: {state.consecutive_above} consecutive games "
                    f">{self._sigma_thr}σ above posterior"
                )
                newly_triggered = self._make_alert(state, season, game_number)
                self._alert_log.append(newly_triggered)

            elif state.consecutive_below >= self._consec_trigger:
                state.regime_state      = "down_shift"
                state.triggered_at_game = game_number
                state.reason            = (
                    f"Statistical: {state.consecutive_below} consecutive games "
                    f">{self._sigma_thr}σ below posterior"
                )
                newly_triggered = self._make_alert(state, season, game_number)
                self._alert_log.append(newly_triggered)

        return newly_triggered

    def _make_alert(
        self, state: PlayerRegimeState, season: int, game_number: int
    ) -> dict[str, Any]:
        consec = max(state.consecutive_above, state.consecutive_below)
        return {
            "player_id":         state.player_id,
            "player_name":       state.player_name,
            "season":            season,
            "triggered_at_game": game_number,
            "regime_state":      state.regime_state,
            "consecutive_games": consec,
            "mean_residual":     state.mean_residual(),
            "sigma_multiplier":  self._prior_sigma_mult,
            "obs_sigma_divisor": self._obs_sigma_div,
            "reason":            state.reason,
            "model_version":     self._version,
        }

    # ------------------------------------------------------------------
    # Manual override (Bob's domain)
    # ------------------------------------------------------------------

    def manual_override(
        self,
        player_id:   int,
        player_name: str,
        direction:   str,
        reason:      str,
        season:      int = 0,
    ) -> dict[str, Any]:
        """Bob manually flags a regime change.

        Args:
            player_id:   Player identifier.
            player_name: Display name.
            direction:   "up" or "down".
            reason:      Human-readable reason (coaching change, trade, etc.).
            season:      Season integer.

        Returns:
            Alert dict.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got '{direction}'")

        state = self._states.get(player_id)
        if state is None:
            state = PlayerRegimeState(player_id=player_id, player_name=player_name)
            self._states[player_id] = state

        state.player_name    = player_name
        state.regime_state   = "manual_up" if direction == "up" else "manual_down"
        state.triggered_at_game = state.games_processed
        state.reason         = f"[MANUAL] {reason}"

        alert = self._make_alert(state, season, state.games_processed)
        self._alert_log.append(alert)
        return alert

    def clear_manual_override(self, player_id: int) -> None:
        """Remove a manual override, returning to data-driven detection."""
        state = self._states.get(player_id)
        if state and state.regime_state in ("manual_up", "manual_down"):
            state.regime_state        = "none"
            state.triggered_at_game   = None
            state.consecutive_above   = 0
            state.consecutive_below   = 0
            state.consecutive_neutral = 0
            state.reason              = ""

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active_regimes(self) -> list[PlayerRegimeState]:
        """Return all players currently in an active regime."""
        return [s for s in self._states.values() if s.is_active()]

    def get_state(self, player_id: int) -> PlayerRegimeState | None:
        return self._states.get(player_id)

    def alerts_dataframe(self, season: int | None = None) -> pl.DataFrame:
        """Return all historical alerts as a DataFrame."""
        logs = self._alert_log
        if season is not None:
            logs = [a for a in logs if a.get("season") == season]
        if not logs:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in REGIME_ALERT_SCHEMA.items()}
            )
        df = pl.DataFrame(logs)
        for col, dtype in REGIME_ALERT_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df.select(list(REGIME_ALERT_SCHEMA.keys()))

    def recommended_sigma_adjustments(
        self, player_id: int
    ) -> tuple[float, float]:
        """Return (prior_sigma_multiplier, obs_sigma_divisor) for a player.

        Returns (1.0, 1.0) if no active regime — no adjustment needed.
        Returns (REGIME_PRIOR_SIGMA_MULT, REGIME_OBS_SIGMA_DIVISOR) if active.
        """
        state = self._states.get(player_id)
        if state is None or not state.is_active():
            return (1.0, 1.0)
        return (self._prior_sigma_mult, self._obs_sigma_div)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "states":            self._states,
            "alert_log":         self._alert_log,
            "sigma_thr":         self._sigma_thr,
            "consec_trigger":    self._consec_trigger,
            "prior_sigma_mult":  self._prior_sigma_mult,
            "obs_sigma_div":     self._obs_sigma_div,
            "clear_consec":      self._clear_consec,
            "version":           self._version,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "RegimeChangeDetector":
        payload = joblib.load(Path(path))
        obj = cls(
            sigma_threshold     = payload.get("sigma_thr",        REGIME_SIGMA_THRESHOLD),
            consecutive_trigger = payload.get("consec_trigger",   REGIME_CONSECUTIVE_TRIGGER),
            prior_sigma_mult    = payload.get("prior_sigma_mult", REGIME_PRIOR_SIGMA_MULT),
            obs_sigma_divisor   = payload.get("obs_sigma_div",    REGIME_OBS_SIGMA_DIVISOR),
            clear_consecutive   = payload.get("clear_consec",     REGIME_CLEAR_CONSECUTIVE),
        )
        obj._states    = payload.get("states",    {})
        obj._alert_log = payload.get("alert_log", [])
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_regime_alerts(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    """Write regime alerts to ``output_dir/regime_alerts_{season}.parquet``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"regime_alerts_{season}.parquet"
    for col, dtype in REGIME_ALERT_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(REGIME_ALERT_SCHEMA.keys())).write_parquet(path)
    return path
