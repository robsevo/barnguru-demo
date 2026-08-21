"""WAR pipeline: the arithmetic, the sign convention, and the replacement guard.

These are the places where a silent change flips a leaderboard rather than
raising, which is why they are the ones pinned here.
"""

import polars as pl
import pytest

from models.war_model import (
    GOALS_PER_WIN,
    RAPM_EV_CLAMP,
    REPLACEMENT_RAPM_EV_FALLBACK,
    REPLACEMENT_RAPM_PK_FALLBACK,
    REPLACEMENT_RAPM_PP_FALLBACK,
    WAR_DOLLAR_RATE,
    _compute_replacement_level,
    contract_efficiency,
    gar_from_rapm,
    war_from_gar,
)


def test_war_from_gar_divides_by_the_documented_goals_per_win():
    assert war_from_gar(2 * GOALS_PER_WIN) == pytest.approx(2.0)
    assert war_from_gar(0.0) == 0.0


def test_war_from_gar_does_not_divide_by_zero():
    # The floor is max(goals_per_win, 0.1); a zero rate must not raise.
    assert war_from_gar(1.0, goals_per_win=0.0) == pytest.approx(10.0)


def test_contract_efficiency_is_none_without_a_usable_cap_hit():
    assert contract_efficiency(3.0, None) is None
    assert contract_efficiency(3.0, 0.0) is None
    assert contract_efficiency(3.0, -1.0) is None


def test_contract_efficiency_is_implied_value_over_cap_hit():
    assert contract_efficiency(2.0, 4.4) == pytest.approx(2.0 * WAR_DOLLAR_RATE / 4.4)


def test_good_defence_is_negative_rapm_and_must_increase_gar():
    """rapm_ev_def is expected goals ALLOWED: negative is good.

    gar_from_rapm negates it. If that negation is ever dropped, every defensive
    contribution inverts and the leaderboard ranks backwards while still
    looking plausible — so it is asserted directly.
    """
    kw = dict(rapm_ev_off=0.0, rapm_pp=0.0, rapm_pk=0.0,
              toi_ev_min=1000.0, toi_pp_min=0.0, toi_pk_min=0.0)
    good = gar_from_rapm(rapm_ev_def=-0.5, **kw)[0]
    bad  = gar_from_rapm(rapm_ev_def=+0.5, **kw)[0]
    assert good > bad


def test_extreme_rapm_is_clamped_before_it_becomes_goals():
    """Small-sample ridge output can be absurd; the clamp is what keeps a
    call-up with nine minutes of ice from topping the board."""
    kw = dict(rapm_ev_def=0.0, rapm_pp=0.0, rapm_pk=0.0,
              toi_ev_min=1000.0, toi_pp_min=0.0, toi_pk_min=0.0)
    at_limit = gar_from_rapm(rapm_ev_off=RAPM_EV_CLAMP, **kw)[0]
    absurd   = gar_from_rapm(rapm_ev_off=1e6, **kw)[0]
    assert absurd == pytest.approx(at_limit)


def test_missing_rapm_components_count_as_zero_not_as_a_crash():
    gar = gar_from_rapm(None, None, None, None, 500.0, 100.0, 100.0)
    assert len(gar) == 3
    assert all(isinstance(v, float) for v in gar)


def test_replacement_level_falls_back_on_an_empty_pool():
    assert _compute_replacement_level(pl.DataFrame()) == (
        REPLACEMENT_RAPM_EV_FALLBACK,
        REPLACEMENT_RAPM_PP_FALLBACK,
        REPLACEMENT_RAPM_PK_FALLBACK,
    )


def test_replacement_level_falls_back_when_the_pool_is_thinner_than_a_league():
    """Fewer players than roster slots means there is no replacement tier to
    average, so the calibrated constant has to stand in."""
    df = pl.DataFrame({
        "player_id":   list(range(50)),
        "toi_ev":      [600.0] * 50,
        "rapm_ev_off": [0.1] * 50,
        "rapm_ev_def": [-0.1] * 50,
        "rapm_pp":     [0.0] * 50,
        "rapm_pk":     [0.0] * 50,
    })
    assert _compute_replacement_level(df)[0] == REPLACEMENT_RAPM_EV_FALLBACK


def test_replacement_level_without_a_toi_column_falls_back():
    df = pl.DataFrame({"player_id": [1, 2], "rapm_ev_off": [0.1, 0.2]})
    assert _compute_replacement_level(df) == (
        REPLACEMENT_RAPM_EV_FALLBACK,
        REPLACEMENT_RAPM_PP_FALLBACK,
        REPLACEMENT_RAPM_PK_FALLBACK,
    )
