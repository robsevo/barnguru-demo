"""Playoff Fatigue — Feature 3.23.

The composite FI (3.17) treats every playoff game as just another game in the
weighted-sum stack of Phase 3 signals. That under-counts what a Cup run
actually costs a player: best-of-7 series with no scheduled off-days, 2-3
day travel turnarounds between cities, escalating physical play, and a
compressed schedule that compounds over months.

Feature 2.23 (`playoff_delta`) captures performance *shrinkage* in playoffs
but is not a fatigue model. Feature 3.15 (`prior_playoff_load`) captures
last spring's playoff games as a season-start penalty that decays away — also
not a current-run fatigue model.

This module fills the gap. It produces a per-player-per-playoff-game additive
penalty that lives alongside `playoff_load_penalty` in `composite_fi.py`:

    fi = clamp(base_fi + rust_load + playoff_load_penalty
                       + playoff_fatigue_score, 0.0, 1.0)

When ``game_type != 3`` the penalty is ``0.0`` — no impact on regular-season
FI.

Form
----
Four named, auditable sub-signals summed (clamped at ``MAX_SCORE``):

    series_intensity      = SERIES_INTENSITY_PER_GAME × games_in_current_series
    cumulative_playoff_gp = CUMULATIVE_GP_PER_GAME × max(0,
                                cumulative_playoff_gp - CUMULATIVE_GP_THRESHOLD)
    series_compression    = max(0, SERIES_COMPRESSION_TARGET_REST - rest_days_in_series)
                            × SERIES_COMPRESSION_PER_DAY
    cross_series_travel   = TRAVEL_MULTIPLIER × (miles_load + tz_load)

    playoff_fatigue_score = clamp(sum, 0.0, MAX_SCORE)

Defaults (`PENALTY_*` constants):

    SERIES_INTENSITY_PER_GAME       = 0.005     # 7 games → 0.035
    CUMULATIVE_GP_THRESHOLD         = 7         # first-round baseline
    CUMULATIVE_GP_PER_GAME          = 0.003     # 14 GP → 0.021; Cup Final 25 GP → 0.054
    SERIES_COMPRESSION_TARGET_REST  = 2
    SERIES_COMPRESSION_PER_DAY      = 0.01      # 1-day-rest game → 0.010
    TRAVEL_MULTIPLIER               = 0.04      # max 0.08 when miles_load + tz_load saturate at 2.0
    MAX_SCORE                       = 0.20      # cap

Each input load (miles_load, tz_load) is normalized to [0, 1] upstream
(by the driver), so a fully-saturated travel signal contributes 0.04 × 2.0
= 0.08. A worst-case Cup-Finals Game 7 starter ends up near the cap:
0.035 (intensity) + 0.054 (cumulative) + 0.010 (compression) + ~0.06
(travel) ≈ 0.16 typical; 0.20 is reserved for outlier cross-country
playoff travel.

Inputs
------
``signals_df`` (Polars) — one row per (player_id, game_id, game_date)::

    player_id                            Int64
    game_id                              Int64
    game_date                            Utf8   "YYYY-MM-DD"
    game_type                            Int64  must equal 3 for non-zero score
    games_in_current_series              Int64  1..7
    cumulative_playoff_gp_this_spring    Int64  running total this spring
    rest_days_inside_series              Int64  typically 1-2; -1 if first game of series
    miles_load                           Float64  normalized travel miles in [0,1]
    tz_load                              Float64  normalized TZ load in [0,1]

All optional columns default to 0 / no-effect when absent or null.

``as_of_date`` is written through to every row.

Outputs
-------
One row per (player_id, game_id) with PLAYOFF_FATIGUE_SCHEMA:

    player_id                Int64
    game_id                  Int64
    game_date                Utf8
    as_of_date               Utf8
    series_intensity         Float64
    cumulative_playoff_gp    Float64
    series_compression       Float64
    cross_series_travel      Float64
    playoff_fatigue_score    Float64   in [0.0, MAX_SCORE]
    component_breakdown      Utf8      JSON map (sub-signal → contribution)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "playoff_fatigue_v1"

REQUIRED_COLS = ("player_id", "game_id", "game_date")

PLAYOFF_FATIGUE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":              pl.Int64,
    "game_id":                pl.Int64,
    "game_date":              pl.Utf8,
    "as_of_date":             pl.Utf8,
    "series_intensity":       pl.Float64,
    "cumulative_playoff_gp":  pl.Float64,
    "series_compression":     pl.Float64,
    "cross_series_travel":    pl.Float64,
    "playoff_fatigue_score":  pl.Float64,
    "component_breakdown":    pl.Utf8,
}


SERIES_INTENSITY_PER_GAME      = 0.005
CUMULATIVE_GP_THRESHOLD        = 7
CUMULATIVE_GP_PER_GAME         = 0.003
SERIES_COMPRESSION_TARGET_REST = 2
SERIES_COMPRESSION_PER_DAY     = 0.01
TRAVEL_MULTIPLIER              = 0.04
MAX_SCORE                      = 0.20

PLAYOFF_GAME_TYPE = 3


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in PLAYOFF_FATIGUE_SCHEMA.items()}
    )


def _validate(signals_df: pl.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in signals_df.columns]
    if missing:
        warnings.warn(
            f"signals_df missing columns: {missing}. Required: "
            f"{list(REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _f(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default: int = 0) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


class PlayoffFatigueModel:
    """Current-playoff-run fatigue penalty (Feature 3.23)."""

    def __init__(
        self,
        series_intensity_per_game:      float = SERIES_INTENSITY_PER_GAME,
        cumulative_gp_threshold:        int   = CUMULATIVE_GP_THRESHOLD,
        cumulative_gp_per_game:         float = CUMULATIVE_GP_PER_GAME,
        series_compression_target_rest: int   = SERIES_COMPRESSION_TARGET_REST,
        series_compression_per_day:     float = SERIES_COMPRESSION_PER_DAY,
        travel_multiplier:              float = TRAVEL_MULTIPLIER,
        max_score:                      float = MAX_SCORE,
    ) -> None:
        for name, val in (
            ("series_intensity_per_game", series_intensity_per_game),
            ("cumulative_gp_per_game", cumulative_gp_per_game),
            ("series_compression_per_day", series_compression_per_day),
            ("travel_multiplier", travel_multiplier),
            ("max_score", max_score),
        ):
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0")
        self._intensity_per_game  = float(series_intensity_per_game)
        self._cum_gp_threshold    = int(cumulative_gp_threshold)
        self._cum_gp_per_game     = float(cumulative_gp_per_game)
        self._compress_target     = int(series_compression_target_rest)
        self._compress_per_day    = float(series_compression_per_day)
        self._travel_multiplier   = float(travel_multiplier)
        self._max_score           = float(max_score)
        self._version             = MODEL_VERSION

    @property
    def version(self) -> str:
        return self._version

    def compute_row(self, r: dict) -> tuple[float, dict[str, float]]:
        """Score one row. Returns (score, components dict)."""
        if _i(r.get("game_type"), default=PLAYOFF_GAME_TYPE) != PLAYOFF_GAME_TYPE:
            zero = {
                "series_intensity":      0.0,
                "cumulative_playoff_gp": 0.0,
                "series_compression":    0.0,
                "cross_series_travel":   0.0,
            }
            return 0.0, zero

        games_in_series = max(0, _i(r.get("games_in_current_series")))
        cum_gp          = max(0, _i(r.get("cumulative_playoff_gp_this_spring")))
        rest_in_series  = _i(r.get("rest_days_inside_series"), default=self._compress_target)
        miles_load      = max(0.0, _f(r.get("miles_load")))
        tz_load         = max(0.0, _f(r.get("tz_load")))

        intensity = self._intensity_per_game * games_in_series
        cumulative = self._cum_gp_per_game * max(0, cum_gp - self._cum_gp_threshold)
        # First game of a series has rest_days = -1 (the gap from prior round);
        # only inside-series compression counts here.
        if rest_in_series < 0:
            compression = 0.0
        else:
            compression = self._compress_per_day * max(
                0, self._compress_target - rest_in_series
            )
        travel = self._travel_multiplier * (miles_load + tz_load)

        components = {
            "series_intensity":      float(intensity),
            "cumulative_playoff_gp": float(cumulative),
            "series_compression":    float(compression),
            "cross_series_travel":   float(travel),
        }
        total = _clamp(sum(components.values()), 0.0, self._max_score)
        return total, components

    def compute(self, signals_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        """Return one row per (player_id, game_id) with playoff fatigue score."""
        if not _validate(signals_df):
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(signals_df) == 0:
            return _empty_output()

        out_rows: list[dict] = []
        for r in signals_df.to_dicts():
            pid_raw = r.get("player_id")
            gd_raw  = r.get("game_date")
            gid_raw = r.get("game_id")
            if pid_raw is None or gd_raw is None or gid_raw is None:
                continue
            try:
                pid = int(pid_raw)
                gid = int(gid_raw)
            except (TypeError, ValueError):
                continue
            try:
                _parse_date(str(gd_raw))
            except (TypeError, ValueError):
                continue

            score, comps = self.compute_row(r)

            out_rows.append({
                "player_id":             pid,
                "game_id":               gid,
                "game_date":             str(gd_raw),
                "as_of_date":            as_of_date,
                "series_intensity":      comps["series_intensity"],
                "cumulative_playoff_gp": comps["cumulative_playoff_gp"],
                "series_compression":    comps["series_compression"],
                "cross_series_travel":   comps["cross_series_travel"],
                "playoff_fatigue_score": float(score),
                "component_breakdown":   json.dumps(
                    {k: round(v, 6) for k, v in comps.items()},
                    sort_keys=True,
                ),
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=PLAYOFF_FATIGUE_SCHEMA)


def write_playoff_fatigue(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write playoff-fatigue DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"playoff_fatigue_{as_of_date}.parquet"
    for col, dtype in PLAYOFF_FATIGUE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(PLAYOFF_FATIGUE_SCHEMA.keys())).write_parquet(path)
    return path
