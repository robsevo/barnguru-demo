"""Expected-goals feature engineering.

The matrix and its column names are built by two separate functions. Nothing in
the code forces them to agree, so a feature added to one and not the other
mislabels every coefficient downstream without raising. That is the contract
this file exists to hold.
"""

import numpy as np
import polars as pl
import pytest

from models.xg_model import (
    _HIGH_DISTANCE,
    _MED_DISTANCE,
    _SLOT_Y,
    _danger_zone,
    build_features,
    feature_names,
)


def _shots(n: int = 4) -> pl.DataFrame:
    return pl.DataFrame({
        "shot_angle":          [10.0, 25.0, 40.0, 5.0][:n],
        "arena_adj_distance":  [12.0, 30.0, 55.0, 8.0][:n],
        "shot_type":           ["WRIST", "SLAP", "SNAP", "TIP"][:n],
        "arena_adj_y":         [3.0, 10.0, 30.0, 0.0][:n],
        "last_event_team":     ["TOR", "MTL", "TOR", "MTL"][:n],
        "shooting_team":       ["TOR", "TOR", "TOR", "TOR"][:n],
        "home_skaters":        [5, 5, 4, 5][:n],
        "away_skaters":        [5, 4, 5, 5][:n],
        "period":              [1, 2, 3, 3][:n],
        "shooter_hand":        ["L", "R", "L", "R"][:n],
    })


def test_feature_matrix_width_matches_the_declared_names():
    X, _ = build_features(_shots())
    assert X.shape[1] == len(feature_names())


def test_feature_matrix_has_one_row_per_shot_and_no_nan():
    X, _ = build_features(_shots())
    assert X.shape[0] == 4
    assert not np.isnan(X).any()


def test_target_is_absent_until_the_frame_carries_one():
    _, y = build_features(_shots())
    assert y is None

    with_goals = _shots().with_columns(pl.Series("is_goal", [1, 0, 0, 1]))
    _, y = build_features(with_goals)
    assert y is not None
    assert y.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_missing_columns_raise_and_name_what_is_missing():
    df = _shots().drop("shot_angle")
    with pytest.raises(ValueError, match="shot_angle"):
        build_features(df)


def test_danger_zone_needs_both_proximity_and_the_slot():
    distance = np.array([_HIGH_DISTANCE - 1, _HIGH_DISTANCE - 1,
                         _MED_DISTANCE - 1, _MED_DISTANCE + 1], dtype=np.float32)
    y        = np.array([0.0, _SLOT_Y + 5, 0.0, 0.0], dtype=np.float32)
    #                    in slot, close  ->  HIGH
    #                    wide of the slot ->  demoted to MED
    #                    mid range        ->  MED
    #                    beyond mid       ->  LOW
    assert _danger_zone(distance, y).tolist() == [2, 1, 1, 0]
