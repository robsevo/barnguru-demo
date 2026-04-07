"""Roster Disruption Index — Feature 2.19.

Team-level metric quantifying how much recent personnel churn has disrupted
a team's system cohesion and communication. High disruption correlates with
reduced line-chemistry, confused defensive assignments, and suboptimal
power-play execution in the near-term.

Model design:
    Each roster move (trade, call-up, IR placement, send-down) in the last
    30 days contributes a ``disruption score`` weighted by two factors:

    1. Event type weight — trades are most disruptive (0.85), because the new
       player must learn a new system from scratch.  IR returns / call-ups are
       moderate (0.50–0.60).  Pure front-office moves are negligible (0.05).

    2. Time decay — exponential with half-life DECAY_HALF_LIFE_DAYS (14 days,
       ~10 games).  A trade two weeks ago contributes half as much as a trade
       today.  Score is effectively zero after ~42 days (~20 games).

    Chaos multiplier: if 3+ moves occurred in the last 72 hours (``roster
    chaos''), the total score is multiplied by CHAOS_MULTIPLIER (1.5).

    Final formula:
        raw = chaos_mult × Σ( event_weight × 0.5^(days_elapsed / 14) )
        disruption_index = clip(raw / MAX_RAW, 0, 1)
        efficiency_modifier = 1.0 − disruption_index × MAX_EFFICIENCY_EFFECT

    MAX_EFFICIENCY_EFFECT = 0.08 (8% team-efficiency drag at maximum disruption)
    MAX_RAW = 5.0 (normalisation constant — 5 simultaneous max-weight moves)

Inputs
------
Transaction DataFrame from Feature 1.14 (TRANSACTION_SCHEMA) with columns:
    date, event_type, team, secondary_team, player_or_executive, description

Outputs
-------
``roster_disruption_{date}.parquet`` — one row per team on that date:
    team, as_of_date, disruption_index, n_moves_30d, n_moves_72h,
    chaos_multiplier, efficiency_modifier, model_version
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from models.rapm_model import DataMissingWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION          = "rdi_v1"
DECAY_HALF_LIFE_DAYS   = 14      # half-life for disruption decay
LOOKBACK_DAYS          = 30      # window of transactions to consider
CHAOS_THRESHOLD_72H    = 3       # moves in 72h to trigger chaos multiplier
CHAOS_MULTIPLIER       = 1.5
MAX_EFFICIENCY_EFFECT  = 0.08    # max team efficiency drag (8%)
MAX_RAW                = 5.0     # normalisation constant

# Event type weights (disruption impact per move)
EVENT_TYPE_WEIGHTS: dict[str, float] = {
    "trade":          0.85,   # new player learning a new system — highest disruption
    "ltir":           0.70,   # losing a key player for extended period
    "activated":      0.55,   # IR return — reintegrating a player into the system
    "waiver_claim":   0.50,   # adding unknown quantity from waivers
    "call_up":        0.40,   # AHL callup — inexperienced player on lineup
    "waiver_release": 0.35,   # losing depth player
    "send_down":      0.30,   # sending down depth player
    "signing":        0.20,   # roster addition, usually planned
    "front_office":   0.05,   # FO change only (2.19 is player roster, not FO)
    "other":          0.15,
}

# Roster-role multipliers applied when role data is available
ROLE_MULTIPLIERS: dict[str, float] = {
    "top6_f":    1.30,   # top-6 forward — highest line-chemistry impact
    "top4_d":    1.20,   # top-4 defenceman
    "starter_g": 1.40,   # starting goalie — massive disruption
    "bottom6_f": 0.85,
    "bottom_pair_d": 0.70,
    "backup_g":  0.50,
}

# Output schema
ROSTER_DISRUPTION_SCHEMA: dict[str, pl.DataType] = {
    "team":               pl.Utf8,
    "as_of_date":         pl.Utf8,
    "disruption_index":   pl.Float64,
    "n_moves_30d":        pl.Int64,
    "n_moves_72h":        pl.Int64,
    "chaos_multiplier":   pl.Float64,
    "efficiency_modifier": pl.Float64,
    "model_version":      pl.Utf8,
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def decay_factor(days_elapsed: float, half_life: float = DECAY_HALF_LIFE_DAYS) -> float:
    """Exponential decay weight. Returns 1.0 at day 0, 0.5 at half_life days.

    Args:
        days_elapsed: Number of days since the event.
        half_life:    Days for weight to halve (default: DECAY_HALF_LIFE_DAYS).

    Returns:
        Weight in [0, 1]. Never negative.
    """
    if days_elapsed < 0:
        days_elapsed = 0.0
    return float(0.5 ** (days_elapsed / max(half_life, 1e-9)))


def event_weight(
    event_type: str,
    player_role: str | None = None,
    custom_weights: dict[str, float] | None = None,
) -> float:
    """Return disruption weight for one event, optionally scaled by roster role.

    Args:
        event_type:     Normalized EventType string (from transaction_parser).
        player_role:    Optional role key from ROLE_MULTIPLIERS.
        custom_weights: Override EVENT_TYPE_WEIGHTS for this call.

    Returns:
        Float weight in [0, 1].
    """
    weights = custom_weights or EVENT_TYPE_WEIGHTS
    base = weights.get(event_type, weights.get("other", 0.15))
    role_mult = ROLE_MULTIPLIERS.get(player_role or "", 1.0)
    return float(np.clip(base * role_mult, 0.0, 1.0))


# ---------------------------------------------------------------------------
# RosterDisruptionModel — Feature 2.19
# ---------------------------------------------------------------------------

class RosterDisruptionModel:
    """Roster disruption index model (Feature 2.19).

    Computes team-level disruption scores from a transaction log.

    Usage::

        model = RosterDisruptionModel()
        rdi = model.compute(transactions_df, team="EDM", as_of_date=date.today())
        print(rdi)  # {"disruption_index": 0.34, "efficiency_modifier": 0.97, ...}

        all_teams_df = model.compute_all(transactions_df, as_of_date=date.today())
    """

    def __init__(
        self,
        decay_half_life:      float = DECAY_HALF_LIFE_DAYS,
        lookback_days:        int   = LOOKBACK_DAYS,
        chaos_threshold_72h:  int   = CHAOS_THRESHOLD_72H,
        chaos_multiplier:     float = CHAOS_MULTIPLIER,
        max_efficiency_effect: float = MAX_EFFICIENCY_EFFECT,
        max_raw:              float = MAX_RAW,
        event_weights:        dict[str, float] | None = None,
    ) -> None:
        self._half_life     = decay_half_life
        self._lookback      = lookback_days
        self._chaos_thr     = chaos_threshold_72h
        self._chaos_mult    = chaos_multiplier
        self._max_effect    = max_efficiency_effect
        self._max_raw       = max_raw
        self._weights       = event_weights or {**EVENT_TYPE_WEIGHTS}
        self._version       = MODEL_VERSION

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        transactions_df: pl.DataFrame,
        team: str,
        as_of_date: date | str | None = None,
        player_roles: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute disruption index for a single team on a given date.

        Args:
            transactions_df: Transaction DataFrame (TRANSACTION_SCHEMA columns).
                             Must have: date, event_type, team, secondary_team.
            team:            Team abbreviation (e.g. "EDM").
            as_of_date:      Date for the calculation (default: today UTC).
            player_roles:    Optional mapping: player_or_executive → role key
                             (from ROLE_MULTIPLIERS) for role-weighted scoring.

        Returns:
            Dict with: team, as_of_date, disruption_index, n_moves_30d,
            n_moves_72h, chaos_multiplier, efficiency_modifier, model_version.
        """
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc).date()
        elif isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        cutoff = as_of_date - timedelta(days=self._lookback)
        cutoff_72h = as_of_date - timedelta(hours=72)

        # Filter to this team's events in lookback window
        team_events = self._filter_team_events(transactions_df, team, cutoff, as_of_date)

        n_moves_30d = len(team_events)
        cutoff_72h_date = cutoff_72h.date() if isinstance(cutoff_72h, datetime) else cutoff_72h
        n_moves_72h = sum(
            1 for ev in team_events
            if date.fromisoformat(ev["date"]) >= cutoff_72h_date
        ) if team_events else 0

        # Chaos multiplier
        chaos_mult = self._chaos_mult if n_moves_72h >= self._chaos_thr else 1.0

        # Compute weighted decay sum
        raw_score = 0.0
        for ev in team_events:
            try:
                ev_date = date.fromisoformat(ev["date"])
            except (ValueError, KeyError):
                continue
            days_elapsed = (as_of_date - ev_date).days
            ev_type = ev.get("event_type", "other")
            player  = ev.get("player_or_executive", "")
            role    = (player_roles or {}).get(player)

            w = event_weight(ev_type, role, self._weights)
            d = decay_factor(days_elapsed, self._half_life)
            raw_score += w * d

        raw_score *= chaos_mult
        disruption_index = float(np.clip(raw_score / max(self._max_raw, 1e-9), 0.0, 1.0))
        efficiency_modifier = 1.0 - disruption_index * self._max_effect

        return {
            "team":               team,
            "as_of_date":         as_of_date.isoformat(),
            "disruption_index":   disruption_index,
            "n_moves_30d":        n_moves_30d,
            "n_moves_72h":        n_moves_72h,
            "chaos_multiplier":   chaos_mult,
            "efficiency_modifier": efficiency_modifier,
            "model_version":      self._version,
        }

    def _filter_team_events(
        self,
        transactions_df: pl.DataFrame,
        team: str,
        cutoff: date,
        as_of_date: date,
    ) -> list[dict[str, Any]]:
        """Return list of event dicts for a team within the lookback window."""
        required = {"date", "event_type", "team"}
        if not required.issubset(set(transactions_df.columns)):
            missing = required - set(transactions_df.columns)
            warnings.warn(
                f"transactions_df missing columns: {sorted(missing)}. "
                "Returning zero disruption.",
                DataMissingWarning,
                stacklevel=3,
            )
            return []

        # Filter: team is primary or secondary acquirer; date in window; skip FO events
        cutoff_str   = cutoff.isoformat()
        as_of_str    = as_of_date.isoformat()

        # Primary team filter
        primary = transactions_df.filter(
            (pl.col("team") == team) &
            (pl.col("date") >= cutoff_str) &
            (pl.col("date") <= as_of_str) &
            (pl.col("event_type") != "front_office")
        )

        # Secondary team (receiving side of trades)
        if "secondary_team" in transactions_df.columns:
            secondary = transactions_df.filter(
                (pl.col("secondary_team") == team) &
                (pl.col("date") >= cutoff_str) &
                (pl.col("date") <= as_of_str) &
                (pl.col("event_type") == "trade")
            )
            combined = pl.concat([primary, secondary], how="diagonal")
        else:
            combined = primary

        return combined.to_dicts()

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_all(
        self,
        transactions_df: pl.DataFrame,
        as_of_date: date | str | None = None,
        teams: list[str] | None = None,
        player_roles: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Compute disruption index for all teams (or specified list).

        Args:
            transactions_df: Full transaction log.
            as_of_date:      Date for computation (default: today UTC).
            teams:           List of team abbreviations. If None, extracted
                             from transactions_df.
            player_roles:    Optional player → role mapping.

        Returns:
            DataFrame matching ROSTER_DISRUPTION_SCHEMA, one row per team.
        """
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc).date()
        elif isinstance(as_of_date, str):
            as_of_date = date.fromisoformat(as_of_date)

        if teams is None:
            if "team" in transactions_df.columns:
                teams = sorted(transactions_df["team"].drop_nulls().unique().to_list())
            else:
                return pl.DataFrame(
                    {col: pl.Series([], dtype=dt) for col, dt in ROSTER_DISRUPTION_SCHEMA.items()}
                )

        rows: list[dict[str, Any]] = []
        for team in teams:
            row = self.compute(transactions_df, team, as_of_date, player_roles)
            rows.append(row)

        if not rows:
            return pl.DataFrame(
                {col: pl.Series([], dtype=dt) for col, dt in ROSTER_DISRUPTION_SCHEMA.items()}
            )

        df = pl.DataFrame(rows)
        for col, dtype in ROSTER_DISRUPTION_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df.select(list(ROSTER_DISRUPTION_SCHEMA.keys()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def efficiency_modifier(self, disruption_index: float) -> float:
        """Convert disruption_index to team efficiency modifier.

        Args:
            disruption_index: Value in [0, 1] from compute().

        Returns:
            Multiplicative efficiency modifier ∈ [1 − MAX_EFFECT, 1.0].
            Multiply player/team ratings by this before game simulation.
        """
        return float(1.0 - np.clip(disruption_index, 0.0, 1.0) * self._max_effect)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "half_life":    self._half_life,
            "lookback":     self._lookback,
            "chaos_thr":    self._chaos_thr,
            "chaos_mult":   self._chaos_mult,
            "max_effect":   self._max_effect,
            "max_raw":      self._max_raw,
            "weights":      self._weights,
            "version":      self._version,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "RosterDisruptionModel":
        payload = joblib.load(Path(path))
        obj = cls(
            decay_half_life       = payload.get("half_life",   DECAY_HALF_LIFE_DAYS),
            lookback_days         = payload.get("lookback",    LOOKBACK_DAYS),
            chaos_threshold_72h   = payload.get("chaos_thr",  CHAOS_THRESHOLD_72H),
            chaos_multiplier      = payload.get("chaos_mult",  CHAOS_MULTIPLIER),
            max_efficiency_effect = payload.get("max_effect",  MAX_EFFICIENCY_EFFECT),
            max_raw               = payload.get("max_raw",     MAX_RAW),
            event_weights         = payload.get("weights",     None),
        )
        obj._version = payload.get("version", MODEL_VERSION)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_roster_disruption(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write disruption scores to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"roster_disruption_{as_of_date}.parquet"
    for col, dtype in ROSTER_DISRUPTION_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(ROSTER_DISRUPTION_SCHEMA.keys())).write_parquet(path)
    return path
