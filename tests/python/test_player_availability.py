"""Tests for PlayerAvailabilityModel — P(player plays tonight)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from models.player_availability import (
    PREDICTION_SCHEMA,
    AvailabilityPrediction,
    PlayerAvailabilityModel,
    SignalInput,
    _SIGNAL_LOGIT,
    _sigmoid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AS_OF = "2026-03-21T12:00:00+00:00"
_FRESH = "2026-03-21T11:00:00+00:00"   # 1 hour before as_of
_DATE = "2026-03-21"
_PLAYER = "Connor McDavid"


def _signal(
    source: str,
    signal_type: str,
    confidence: float = 1.0,
    fetched_at: str = _FRESH,
) -> SignalInput:
    return SignalInput(
        source=source,
        signal_type=signal_type,
        confidence=confidence,
        fetched_at=fetched_at,
    )


def _model() -> PlayerAvailabilityModel:
    return PlayerAvailabilityModel(half_life_hours=12.0)


# ---------------------------------------------------------------------------
# 1 — no signals → prior only
# ---------------------------------------------------------------------------


def test_predict_no_signals() -> None:
    pred = _model().predict(_PLAYER, _DATE, [], as_of=_AS_OF)
    assert pred.signal_count == 0
    assert pred.p_plays > 0.95   # sigmoid(3.5) ≈ 0.97
    assert pred.dominant_signal is None
    assert pred.confidence == 0.0


# ---------------------------------------------------------------------------
# 2 — practicing_full → near certain
# ---------------------------------------------------------------------------


def test_predict_practicing_full() -> None:
    pred = _model().predict(
        _PLAYER, _DATE,
        [_signal("morning_skate", "practicing_full")],
        as_of=_AS_OF,
    )
    assert pred.p_plays > 0.99
    assert pred.signal_count == 1


# ---------------------------------------------------------------------------
# 3 — expected_out → near zero
# ---------------------------------------------------------------------------


def test_predict_expected_out() -> None:
    pred = _model().predict(
        _PLAYER, _DATE,
        [_signal("morning_skate", "expected_out")],
        as_of=_AS_OF,
    )
    assert pred.p_plays < 0.02


# ---------------------------------------------------------------------------
# 4 — injury Out → near zero
# ---------------------------------------------------------------------------


def test_predict_out_injury() -> None:
    pred = _model().predict(
        _PLAYER, _DATE,
        [_signal("injury_feed", "Out")],
        as_of=_AS_OF,
    )
    assert pred.p_plays < 0.03


# ---------------------------------------------------------------------------
# 5 — DTD only → uncertain (0.3–0.7)
# ---------------------------------------------------------------------------


def test_predict_dtd_only() -> None:
    pred = _model().predict(
        _PLAYER, _DATE,
        [_signal("injury_feed", "DTD")],
        as_of=_AS_OF,
    )
    assert 0.3 < pred.p_plays < 0.7


# ---------------------------------------------------------------------------
# 6 — conflicting signals: morning skate wins
# ---------------------------------------------------------------------------


def test_predict_conflicting_morning_skate_wins() -> None:
    signals = [
        _signal("injury_feed", "DTD"),
        _signal("morning_skate", "practicing_full"),
    ]
    pred = _model().predict(_PLAYER, _DATE, signals, as_of=_AS_OF)
    assert pred.p_plays > 0.85


# ---------------------------------------------------------------------------
# 7 — recency decay: fresh > stale
# ---------------------------------------------------------------------------


def test_recency_decay() -> None:
    model = PlayerAvailabilityModel(half_life_hours=12.0)
    fresh = _signal("morning_skate", "not_skating", fetched_at="2026-03-21T11:00:00+00:00")
    stale = _signal("morning_skate", "not_skating", fetched_at="2026-03-19T11:00:00+00:00")  # 48h old

    pred_fresh = model.predict(_PLAYER, _DATE, [fresh], as_of=_AS_OF)
    pred_stale = model.predict(_PLAYER, _DATE, [stale], as_of=_AS_OF)

    # Both should lower P below prior, but fresh signal lowers it more
    assert pred_fresh.p_plays < pred_stale.p_plays


# ---------------------------------------------------------------------------
# 8 — confidence weighting
# ---------------------------------------------------------------------------


def test_confidence_weighting() -> None:
    model = _model()
    high_conf = _signal("morning_skate", "not_skating", confidence=1.0)
    low_conf = _signal("morning_skate", "not_skating", confidence=0.2)

    pred_high = model.predict(_PLAYER, _DATE, [high_conf], as_of=_AS_OF)
    pred_low = model.predict(_PLAYER, _DATE, [low_conf], as_of=_AS_OF)

    # High confidence drives P lower than low confidence for a negative signal
    assert pred_high.p_plays < pred_low.p_plays


# ---------------------------------------------------------------------------
# 9 — P bounded [0, 1]
# ---------------------------------------------------------------------------


def test_p_plays_bounded() -> None:
    model = _model()
    combos = [
        [],
        [_signal("injury_feed", "Healthy")],
        [_signal("injury_feed", "Out")],
        [_signal("morning_skate", "expected_in"), _signal("morning_skate", "practicing_full")],
        [_signal("press_conference", "season_ending"), _signal("injury_feed", "Out")],
    ]
    for signals in combos:
        pred = model.predict(_PLAYER, _DATE, signals, as_of=_AS_OF)
        assert 0.0 <= pred.p_plays <= 1.0, f"p_plays={pred.p_plays} out of bounds for {signals}"


# ---------------------------------------------------------------------------
# 10 — all known signal types return float
# ---------------------------------------------------------------------------


def test_encode_signal_all_known_types() -> None:
    for (source, signal_type), expected in _SIGNAL_LOGIT.items():
        result = PlayerAvailabilityModel.encode_signal(source, signal_type)
        assert result == expected, f"encode({source!r}, {signal_type!r}) → {result}, expected {expected}"


# ---------------------------------------------------------------------------
# 11 — unknown signal → 0.0 neutral
# ---------------------------------------------------------------------------


def test_encode_signal_unknown_neutral() -> None:
    assert PlayerAvailabilityModel.encode_signal("mystery_source", "unknown_type") == 0.0
    assert PlayerAvailabilityModel.encode_signal("injury_feed", "NonExistent") == 0.0


# ---------------------------------------------------------------------------
# 12 — season_ending → near zero
# ---------------------------------------------------------------------------


def test_season_ending_near_zero() -> None:
    pred = _model().predict(
        _PLAYER, _DATE,
        [_signal("press_conference", "season_ending")],
        as_of=_AS_OF,
    )
    assert pred.p_plays < 0.03


# ---------------------------------------------------------------------------
# 13 — expected_in → near one
# ---------------------------------------------------------------------------


def test_expected_in_near_one() -> None:
    pred = _model().predict(
        _PLAYER, _DATE,
        [_signal("morning_skate", "expected_in")],
        as_of=_AS_OF,
    )
    assert pred.p_plays > 0.97


# ---------------------------------------------------------------------------
# 14 — dominant signal identified
# ---------------------------------------------------------------------------


def test_dominant_signal_identified() -> None:
    signals = [
        _signal("injury_feed", "DTD"),           # score=-0.5, weak
        _signal("morning_skate", "expected_out"), # score=-4.0, strong
    ]
    pred = _model().predict(_PLAYER, _DATE, signals, as_of=_AS_OF)
    assert pred.dominant_signal == "morning_skate:expected_out"


# ---------------------------------------------------------------------------
# 15 — confidence increases with more signals
# ---------------------------------------------------------------------------


def test_confidence_increases_with_signals() -> None:
    model = _model()
    pred_zero = model.predict(_PLAYER, _DATE, [], as_of=_AS_OF)
    pred_three = model.predict(
        _PLAYER, _DATE,
        [
            _signal("morning_skate", "practicing_full"),
            _signal("injury_feed", "Healthy"),
            _signal("press_conference", "no_concern"),
        ],
        as_of=_AS_OF,
    )
    assert pred_zero.confidence == 0.0
    assert pred_three.confidence > 0.9


# ---------------------------------------------------------------------------
# 16 — to_polars schema
# ---------------------------------------------------------------------------


def test_to_polars_schema() -> None:
    pred = _model().predict(_PLAYER, _DATE, [_signal("injury_feed", "Healthy")], as_of=_AS_OF)
    df = PlayerAvailabilityModel.to_polars([pred])
    assert len(df) == 1
    for col in PREDICTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 17 — to_polars empty
# ---------------------------------------------------------------------------


def test_to_polars_empty() -> None:
    df = PlayerAvailabilityModel.to_polars([])
    assert len(df) == 0
    for col in PREDICTION_SCHEMA:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 18 — predict_from_dataframes: morning skate DF
# ---------------------------------------------------------------------------


def test_predict_from_dataframes_morning_skate() -> None:
    ms_df = pl.DataFrame(
        {
            "player_name": ["Connor McDavid"],
            "signal_type": ["practicing_full"],
            "confidence": [0.9],
            "fetched_at": [_FRESH],
            "date": [_DATE],
            "team_abbrev": ["EDM"],
        }
    )
    empty = pl.DataFrame(schema={"player_name": pl.Utf8})
    pred = _model().predict_from_dataframes(
        _PLAYER, _DATE,
        injury_df=empty,
        morning_skate_df=ms_df,
        press_conference_df=empty,
        as_of=_AS_OF,
    )
    assert pred.signal_count == 1
    assert pred.p_plays > 0.99


# ---------------------------------------------------------------------------
# 19 — predict_from_dataframes: all empty → prior
# ---------------------------------------------------------------------------


def test_predict_from_dataframes_empty() -> None:
    empty = pl.DataFrame()
    pred = _model().predict_from_dataframes(
        _PLAYER, _DATE,
        injury_df=empty,
        morning_skate_df=empty,
        press_conference_df=empty,
        as_of=_AS_OF,
    )
    assert pred.signal_count == 0
    assert pred.p_plays > 0.95


# ---------------------------------------------------------------------------
# 20 — predict_from_dataframes: case-insensitive match
# ---------------------------------------------------------------------------


def test_predict_from_dataframes_case_insensitive() -> None:
    ms_df = pl.DataFrame(
        {
            "player_name": ["connor mcdavid"],   # lower-case in DF
            "signal_type": ["expected_out"],
            "confidence": [1.0],
            "fetched_at": [_FRESH],
        }
    )
    empty = pl.DataFrame()
    pred = _model().predict_from_dataframes(
        "Connor McDavid",   # title-case lookup
        _DATE,
        injury_df=empty,
        morning_skate_df=ms_df,
        press_conference_df=empty,
        as_of=_AS_OF,
    )
    assert pred.signal_count == 1
    assert pred.p_plays < 0.02


# ---------------------------------------------------------------------------
# 21 — fit() raises NotImplementedError
# ---------------------------------------------------------------------------


def test_fit_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="fit()"):
        _model().fit([], [])
