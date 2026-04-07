"""Player availability probability model — P(player plays tonight).

Aggregates signals from three Phase 1 data sources into a single calibrated
probability using a logit-linear rule system (logistic regression with manually-set
Bob-auditable coefficients):

- Injury feed (1.8)         → InjuryStatus (Healthy / DTD / Out)
- Morning skate (1.10)      → SignalType (practicing_full, expected_in, …)
- Press conference (1.12)   → InjurySeverity (no_concern, out_next_game, …)

Each signal is encoded into logit space, weighted by recency × confidence, and
accumulated on top of a calibrated prior. The sum passes through sigmoid → P(plays).

The prior `sigmoid(3.5) ≈ 0.97` reflects the empirical base rate: ~97 % of
scheduled appearances are fulfilled by players not on IR.

A ``fit()`` stub is included for future training on labeled outcomes once the
backtesting infrastructure (Phase 2) is in place.

Usage::

    from models.player_availability import PlayerAvailabilityModel, SignalInput

    model = PlayerAvailabilityModel()

    # Direct signal API
    signals = [
        SignalInput("morning_skate", "practicing_full", confidence=0.95,
                    fetched_at="2026-03-21T10:00:00+00:00"),
        SignalInput("injury_feed", "DTD", confidence=1.0,
                    fetched_at="2026-03-21T08:00:00+00:00"),
    ]
    pred = model.predict("Connor McDavid", "2026-03-21", signals,
                         as_of="2026-03-21T12:00:00+00:00")
    print(pred.p_plays)   # ~0.97

    # DataFrame API (from sync module output)
    pred = model.predict_from_dataframes(
        "Connor McDavid", "2026-03-21",
        injury_df=injury_syncer.get_injuries("2026-03-21"),
        morning_skate_df=ms_syncer.get_signals("2026-03-21"),
        press_conference_df=pc_syncer.get_mentions("2026-03-21"),
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import polars as pl


# ---------------------------------------------------------------------------
# Signal logit encoding table
# ---------------------------------------------------------------------------

_SIGNAL_LOGIT: dict[tuple[str, str], float] = {
    # --- Injury feed (ESPN) ---
    # Prior = sigmoid(3.5) ≈ 0.97; fresh signal weight ≈ 0.944 (1h old, 12h half-life)
    # Targets: Healthy → no change, DTD → P in (0.3, 0.7), Out → P < 0.03
    ("injury_feed", "Healthy"): +1.5,
    ("injury_feed", "DTD"):     -3.5,   # logit ≈ +0.20 → P ≈ 0.55
    ("injury_feed", "Out"):     -9.0,   # logit ≈ -5.0  → P ≈ 0.007

    # --- Morning skate (1.10) ---
    # Targets: expected_in/practicing_full → P > 0.99, expected_out → P < 0.02
    ("morning_skate", "expected_in"):        +3.5,
    ("morning_skate", "practicing_full"):    +2.5,
    ("morning_skate", "practicing_limited"): +0.5,
    ("morning_skate", "game_time_decision"): -1.5,
    ("morning_skate", "line_change"):         0.0,
    ("morning_skate", "not_skating"):        -5.5,
    ("morning_skate", "expected_out"):       -8.5,   # logit ≈ -4.5 → P ≈ 0.011

    # --- Press conference (1.12) ---
    ("press_conference", "no_concern"):      +1.5,
    ("press_conference", "day_to_day"):      -1.5,
    ("press_conference", "out_next_game"):   -8.5,
    ("press_conference", "week_to_week"):    -9.0,
    ("press_conference", "month_to_month"):  -9.0,
    ("press_conference", "season_ending"):   -9.0,   # logit ≈ -5.0 → P ≈ 0.007
    ("press_conference", "return_expected"): -4.0,
    ("press_conference", "unclear"):         -1.5,
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + e^-x)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _parse_dt(iso: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SignalInput:
    """One availability signal from a data source.

    Attributes:
        source: Data source identifier — one of ``"injury_feed"``,
            ``"morning_skate"``, ``"press_conference"``.
        signal_type: Enum value string matching the source's enum
            (e.g. ``"Healthy"``, ``"practicing_full"``, ``"out_next_game"``).
        confidence: Reported confidence in [0.0, 1.0].
        fetched_at: ISO 8601 timestamp of when the signal was captured;
            used for exponential recency decay.
    """

    source: str
    signal_type: str
    confidence: float
    fetched_at: str


@dataclass
class AvailabilityPrediction:
    """Output of the availability model for one player on one date.

    Attributes:
        player_name: Player name as supplied to ``predict()``.
        date: Game date (YYYY-MM-DD).
        p_plays: Probability in [0, 1] that the player plays tonight.
        confidence: Model confidence in [0, 1] based on signal count
            and recency.  0 = prior only (no signals); 1 = strongly anchored.
        signal_count: Number of signals that contributed.
        dominant_signal: ``"source:signal_type"`` label of the signal with the
            highest absolute weighted contribution.  ``None`` if no signals.
        notes: Human-readable summary of the dominant signal and its weight.
    """

    player_name: str
    date: str
    p_plays: float
    confidence: float
    signal_count: int
    dominant_signal: str | None
    notes: str | None


# ---------------------------------------------------------------------------
# Polars schema
# ---------------------------------------------------------------------------

PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "player_name":     pl.Utf8,
    "date":            pl.Utf8,
    "p_plays":         pl.Float64,
    "confidence":      pl.Float64,
    "signal_count":    pl.Int64,
    "dominant_signal": pl.Utf8,
    "notes":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PlayerAvailabilityModel:
    """Logit-linear player availability model with calibrated priors.

    Coefficients are manually set to produce interpretable, Bob-auditable
    probabilities from day one.  The model can be retrained via ``fit()``
    once labeled data from the backtest phase is available.

    Args:
        half_life_hours: Exponential recency decay half-life in hours.
            A signal fetched ``half_life_hours`` ago has 50 % of the weight
            of an equally-confident fresh signal.  Default: 12 hours.
    """

    DEFAULT_PRIOR_LOGIT: float = 3.5      # sigmoid(3.5) ≈ 0.97
    DEFAULT_HALF_LIFE_HOURS: float = 12.0

    def __init__(self, half_life_hours: float = DEFAULT_HALF_LIFE_HOURS) -> None:
        self._half_life_hours = half_life_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        player_name: str,
        date: str,
        signals: list[SignalInput],
        as_of: str | None = None,
    ) -> AvailabilityPrediction:
        """Predict P(player plays tonight) from a list of signals.

        Args:
            player_name: Player name (used only for labelling the output).
            date: Game date in YYYY-MM-DD format.
            signals: All available signals for this player on this date.
            as_of: ISO timestamp used as the recency anchor (defaults to now).
                Pass a fixed timestamp in tests to avoid time-dependent results.

        Returns:
            AvailabilityPrediction.
        """
        anchor = _parse_dt(as_of) if as_of else datetime.now(timezone.utc)
        logit_sum = self.DEFAULT_PRIOR_LOGIT
        total_weight = 0.0
        best_key: str | None = None
        best_abs_contribution = 0.0

        for s in signals:
            score = self.encode_signal(s.source, s.signal_type)
            age_hours = (anchor - _parse_dt(s.fetched_at)).total_seconds() / 3600
            age_hours = max(0.0, age_hours)  # clamp (future fetches → 0 age)
            weight = s.confidence * math.pow(2.0, -age_hours / self._half_life_hours)
            contribution = score * weight

            logit_sum += contribution
            total_weight += weight

            abs_contrib = abs(contribution)
            if abs_contrib > best_abs_contribution:
                best_abs_contribution = abs_contrib
                best_key = f"{s.source}:{s.signal_type}"

        p_plays = _sigmoid(logit_sum)
        confidence_out = min(1.0, total_weight / 3.0)

        notes: str | None = None
        if best_key is not None:
            notes = f"Dominant signal: {best_key} (|contribution|={best_abs_contribution:.2f})"

        return AvailabilityPrediction(
            player_name=player_name,
            date=date,
            p_plays=p_plays,
            confidence=confidence_out,
            signal_count=len(signals),
            dominant_signal=best_key,
            notes=notes,
        )

    def predict_from_dataframes(
        self,
        player_name: str,
        date: str,
        injury_df: pl.DataFrame,
        morning_skate_df: pl.DataFrame,
        press_conference_df: pl.DataFrame,
        as_of: str | None = None,
    ) -> AvailabilityPrediction:
        """Extract signals from sync-module DataFrames and call predict().

        Player name matching is case-insensitive exact match.  A future
        enhancement may add fuzzy matching for name variations (e.g. accents,
        nicknames).

        Args:
            player_name: Player name to look up (case-insensitive).
            date: Game date in YYYY-MM-DD format.
            injury_df: Output of ``InjurySync.get_injuries()``.
            morning_skate_df: Output of ``MorningSkateSyncer.get_signals()``.
            press_conference_df: Output of ``PressConferenceSyncer.get_mentions()``.
            as_of: Recency anchor timestamp (defaults to now).

        Returns:
            AvailabilityPrediction.
        """
        signals: list[SignalInput] = []
        name_lower = player_name.lower()

        # --- Injury feed ---
        if len(injury_df) > 0 and "player_name" in injury_df.columns:
            inj_rows = injury_df.filter(
                pl.col("player_name").str.to_lowercase() == name_lower
            )
            for row in inj_rows.iter_rows(named=True):
                fetched_at = row.get("fetched_at") or (
                    as_of or datetime.now(timezone.utc).isoformat()
                )
                signals.append(
                    SignalInput(
                        source="injury_feed",
                        signal_type=row.get("status", ""),
                        confidence=1.0,
                        fetched_at=fetched_at,
                    )
                )

        # --- Morning skate ---
        if len(morning_skate_df) > 0 and "player_name" in morning_skate_df.columns:
            ms_rows = morning_skate_df.filter(
                pl.col("player_name").str.to_lowercase() == name_lower
            )
            for row in ms_rows.iter_rows(named=True):
                fetched_at = row.get("fetched_at") or (
                    as_of or datetime.now(timezone.utc).isoformat()
                )
                signals.append(
                    SignalInput(
                        source="morning_skate",
                        signal_type=row.get("signal_type", ""),
                        confidence=float(row.get("confidence") or 0.5),
                        fetched_at=fetched_at,
                    )
                )

        # --- Press conference ---
        if len(press_conference_df) > 0 and "player_name" in press_conference_df.columns:
            pc_rows = press_conference_df.filter(
                pl.col("player_name").str.to_lowercase() == name_lower
            )
            for row in pc_rows.iter_rows(named=True):
                fetched_at = row.get("fetched_at") or (
                    as_of or datetime.now(timezone.utc).isoformat()
                )
                signals.append(
                    SignalInput(
                        source="press_conference",
                        signal_type=row.get("severity", ""),
                        confidence=float(row.get("confidence") or 0.5),
                        fetched_at=fetched_at,
                    )
                )

        return self.predict(player_name, date, signals, as_of=as_of)

    @staticmethod
    def encode_signal(source: str, signal_type: str) -> float:
        """Return the logit-space score for a (source, signal_type) pair.

        Unknown combinations return 0.0 (neutral — no evidence either way).
        """
        return _SIGNAL_LOGIT.get((source, signal_type), 0.0)

    @staticmethod
    def to_polars(predictions: list[AvailabilityPrediction]) -> pl.DataFrame:
        """Convert a list of predictions to a flat Polars DataFrame.

        Returns an empty-schema DataFrame if predictions is empty.
        """
        if not predictions:
            return pl.DataFrame(schema=PREDICTION_SCHEMA)

        rows = [
            {
                "player_name": p.player_name,
                "date": p.date,
                "p_plays": p.p_plays,
                "confidence": p.confidence,
                "signal_count": p.signal_count,
                "dominant_signal": p.dominant_signal,
                "notes": p.notes,
            }
            for p in predictions
        ]
        return pl.DataFrame(rows, schema=PREDICTION_SCHEMA)

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        """Train the model on labeled outcome data.

        Not implemented in v1 — the model uses calibrated priors.
        Signature matches ``sklearn.linear_model.LogisticRegression.fit(X, y)``
        to allow future drop-in replacement.

        Raises:
            NotImplementedError: Always in v1.
        """
        raise NotImplementedError(
            "fit() not implemented in v1; calibrated priors are used. "
            "Re-visit after backtesting Phase 2 produces labeled outcomes."
        )
