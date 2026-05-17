"""FI → rating multiplier — Feature 3.18.

The composite Fatigue Index (3.17) is a number from 0.0 (fully rested)
to 1.0 (cooked). The Rust simulation engine doesn't speak fatigue; it
speaks *ratings*. This module translates one into the other.

Form
----
The default mapping is a simple linear de-rating::

    multiplier(fi) = 1.0 − MAX_DEGRADATION × fi

with ``MAX_DEGRADATION = 0.10`` by default. So:

    FI = 0.00 → mult = 1.00   (no penalty)
    FI = 0.25 → mult = 0.975  (1.0 − 0.10 × 0.25)
    FI = 0.50 → mult = 0.95
    FI = 0.75 → mult = 0.925
    FI = 1.00 → mult = 0.90   (10% derating, the calibrated worst case)

Why 10%? Hockey performance is robust — even tired NHL skaters skate.
A 10% peak hit on shot-generation rate is roughly consistent with the
observed late-road-trip xGF/60 deltas in MoneyPuck data. The exact
slope is calibratable; the *floor* (0.90 by default) keeps the engine
from ever zeroing out an active player.

This module is intentionally tiny: it takes a per-player FI score (or
a frame of them) and emits the multiplier the engine consumes. No
business logic, no per-position branching, no per-channel weighting.
Those decisions belong in 3.17 (which channels go into FI) and in the
Rust engine (which rating gets multiplied).

Inputs
------
``fi_df`` (Polars) — one row per (player, game) with at minimum::

    player_id      Int64
    game_date      Utf8       "YYYY-MM-DD"
    fatigue_index  Float64    in [0.0, 1.0]

Optional pass-through columns: ``game_id``, ``as_of_date``.

Outputs
-------
One row per (player, game) with FI_MULTIPLIER_SCHEMA::

    player_id         Int64
    game_id           Int64
    game_date         Utf8
    as_of_date        Utf8
    fatigue_index     Float64
    rating_multiplier Float64   in [1.0 − max_degradation, 1.0]

Calibration knobs (constructor):
- ``max_degradation`` default 0.10  — the floor is (1.0 − this).
- ``slope_floor``     default 0.0   — minimum FI value below which the
                                     multiplier returns to 1.0 (gives a
                                     small "free zone" if desired).
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "fi_rating_multiplier_v1"

FI_MULTIPLIER_REQUIRED_COLS = ("player_id", "game_date", "fatigue_index")

FI_MULTIPLIER_SCHEMA: dict[str, pl.DataType] = {
    "player_id":         pl.Int64,
    "game_id":           pl.Int64,
    "game_date":         pl.Utf8,
    "as_of_date":        pl.Utf8,
    "fatigue_index":     pl.Float64,
    "rating_multiplier": pl.Float64,
}


MAX_DEGRADATION_DEFAULT = 0.10
SLOPE_FLOOR_DEFAULT     = 0.0


# ---------------------------------------------------------------------------
# Public formula
# ---------------------------------------------------------------------------

def rating_multiplier(
    fi: float,
    max_degradation: float = MAX_DEGRADATION_DEFAULT,
    slope_floor:     float = SLOPE_FLOOR_DEFAULT,
) -> float:
    """Map fatigue ``fi ∈ [0, 1]`` → multiplier ``∈ [1 − max_degradation, 1]``.

    Below ``slope_floor`` the multiplier is exactly 1.0 (no fatigue penalty
    in the "free zone"). At and above ``slope_floor`` the multiplier
    interpolates linearly down to ``1.0 − max_degradation`` at FI=1.0.
    """
    if not (0.0 <= max_degradation < 1.0):
        raise ValueError("max_degradation must be in [0, 1)")
    if not (0.0 <= slope_floor < 1.0):
        raise ValueError("slope_floor must be in [0, 1)")

    if not math.isfinite(fi):
        return 1.0
    f = max(0.0, min(1.0, float(fi)))
    if f <= slope_floor:
        return 1.0
    # Linear from (slope_floor, 1.0) → (1.0, 1 − max_degradation)
    span = max(1e-9, 1.0 - slope_floor)
    return float(1.0 - max_degradation * (f - slope_floor) / span)


def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in FI_MULTIPLIER_SCHEMA.items()}
    )


def _validate(fi_df: pl.DataFrame) -> bool:
    missing = [c for c in FI_MULTIPLIER_REQUIRED_COLS if c not in fi_df.columns]
    if missing:
        warnings.warn(
            f"fi_df missing columns: {missing}. Required: "
            f"{list(FI_MULTIPLIER_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# FIRatingMultiplier
# ---------------------------------------------------------------------------

class FIRatingMultiplier:
    """Per-player-game FI → rating multiplier (Feature 3.18)."""

    def __init__(
        self,
        max_degradation: float = MAX_DEGRADATION_DEFAULT,
        slope_floor:     float = SLOPE_FLOOR_DEFAULT,
    ) -> None:
        if not (0.0 <= max_degradation < 1.0):
            raise ValueError("max_degradation must be in [0, 1)")
        if not (0.0 <= slope_floor < 1.0):
            raise ValueError("slope_floor must be in [0, 1)")
        self._max_deg = float(max_degradation)
        self._floor   = float(slope_floor)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    @property
    def max_degradation(self) -> float:
        return self._max_deg

    @property
    def slope_floor(self) -> float:
        return self._floor

    @property
    def multiplier_floor(self) -> float:
        return 1.0 - self._max_deg

    # ------------------------------------------------------------------
    def for_fi(self, fi: float) -> float:
        """Single-value multiplier evaluator using this object's parameters."""
        return rating_multiplier(fi, self._max_deg, self._floor)

    def compute(self, fi_df: pl.DataFrame, as_of_date: str | None = None) -> pl.DataFrame:
        """Return one row per (player, game) with the rating multiplier."""
        if not _validate(fi_df):
            return _empty_output()

        if as_of_date is not None:
            try:
                _parse_date(as_of_date)
            except (TypeError, ValueError):
                raise ValueError(
                    f"as_of_date must be YYYY-MM-DD or None, got {as_of_date!r}"
                )
        if len(fi_df) == 0:
            return _empty_output()

        out_rows: list[dict] = []
        for r in fi_df.to_dicts():
            pid_raw = r.get("player_id")
            gd_raw  = r.get("game_date")
            fi_raw  = r.get("fatigue_index")
            if pid_raw is None or gd_raw is None or fi_raw is None:
                continue
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                continue
            try:
                _parse_date(str(gd_raw))
            except (TypeError, ValueError):
                continue
            try:
                fi = float(fi_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fi):
                continue

            mult = self.for_fi(fi)

            gid_raw = r.get("game_id")
            try:
                gid = int(gid_raw) if gid_raw is not None else 0
            except (TypeError, ValueError):
                gid = 0

            # as_of_date pass-through (row → caller → today's default).
            row_as_of = r.get("as_of_date") if "as_of_date" in r else None
            stamp = as_of_date if as_of_date is not None else (
                str(row_as_of) if row_as_of else str(gd_raw)
            )

            out_rows.append({
                "player_id":         pid,
                "game_id":           gid,
                "game_date":         str(gd_raw),
                "as_of_date":        stamp,
                "fatigue_index":     float(max(0.0, min(1.0, fi))),
                "rating_multiplier": float(mult),
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=FI_MULTIPLIER_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_fi_multiplier(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write FI-multiplier DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fi_rating_multiplier_{as_of_date}.parquet"
    for col, dtype in FI_MULTIPLIER_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(FI_MULTIPLIER_SCHEMA.keys())).write_parquet(path)
    return path
