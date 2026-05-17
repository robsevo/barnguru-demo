"""Age-recovery coefficient — Feature 3.12.

Per-player static coefficient capturing how *quickly* a player recovers
from fatigue load. Younger players bounce back overnight; the late-30s
defenceman on a four-in-six road trip is carrying every minute of every
shift into the next building. This coefficient does NOT directly degrade
performance — that is the composite FI's job (3.17). It controls how
fast the *fatigue inputs* (3.7 TOI load, 3.8 ST load, 3.9 hits/blocks,
3.10 OT load, 3.11 fights) bleed off between games.

Form
----
``recovery_coefficient(age) =
    max(RECOVERY_FLOOR,
        exp(−RECOVERY_DECAY_K × max(0, age − RECOVERY_PEAK_AGE)))``

- Below ``RECOVERY_PEAK_AGE`` (default 30): coefficient is 1.0
  (full recovery).
- After 30: exponential decay with rate ``RECOVERY_DECAY_K`` per year.
- Hard floor ``RECOVERY_FLOOR`` (default 0.50) — even at 40+, a player
  still recovers; the floor prevents the model from sending them to
  zero, which would imply infinite carryover load.

Inputs
------
``roster_df`` (Polars) — one row per player with::

    player_id    Int64
    birth_date   Utf8      "YYYY-MM-DD" (preferred — exact age)
    age          Float64   optional fallback if birth_date is missing

If both ``birth_date`` and ``age`` are present, ``birth_date`` wins.
``as_of_date`` (YYYY-MM-DD) is the date at which age is computed.

Outputs
-------
One row per player with AGE_RECOVERY_SCHEMA::

    player_id              — Int64
    as_of_date             — Utf8
    age_years              — Float64   computed age at as_of_date
    recovery_coefficient   — Float64   in [RECOVERY_FLOOR, 1.0]

Calibration knobs (constructor):
- ``peak_age``       default 30.0  — the inflection point
- ``decay_k``        default 0.05  — per-year exponential decay constant
- ``recovery_floor`` default 0.50  — minimum coefficient

Reference table at defaults::

    age 25 → 1.00      age 33 → 0.86      age 38 → 0.67
    age 28 → 1.00      age 35 → 0.78      age 40 → 0.61
    age 30 → 1.00      age 36 → 0.74      age 42 → 0.55
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "age_recovery_v1"

AGE_RECOVERY_REQUIRED_COLS = ("player_id",)   # birth_date/age handled flexibly

AGE_RECOVERY_SCHEMA: dict[str, pl.DataType] = {
    "player_id":             pl.Int64,
    "as_of_date":            pl.Utf8,
    "age_years":             pl.Float64,
    "recovery_coefficient":  pl.Float64,
}


RECOVERY_PEAK_AGE = 30.0
RECOVERY_DECAY_K  = 0.05
RECOVERY_FLOOR    = 0.50


# ---------------------------------------------------------------------------
# Public formula
# ---------------------------------------------------------------------------

def recovery_coefficient(
    age_years: float,
    peak_age: float = RECOVERY_PEAK_AGE,
    decay_k: float  = RECOVERY_DECAY_K,
    floor:   float  = RECOVERY_FLOOR,
) -> float:
    """Recovery coefficient for a player of ``age_years``.

    ``1.0`` at or below ``peak_age``; exponential decay above; clamped to
    ``floor`` below. Returns a finite float in ``[floor, 1.0]``.
    """
    if not math.isfinite(age_years):
        return floor
    excess = max(0.0, float(age_years) - float(peak_age))
    raw    = math.exp(-float(decay_k) * excess)
    return max(float(floor), float(raw))


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _age_from_birth(birth: str, as_of: date) -> float | None:
    try:
        b = _parse_date(birth)
    except (TypeError, ValueError):
        return None
    delta_days = (as_of - b).days
    if delta_days < 0:
        return None
    return delta_days / 365.25


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in AGE_RECOVERY_SCHEMA.items()}
    )


def _validate(roster_df: pl.DataFrame) -> bool:
    missing = [c for c in AGE_RECOVERY_REQUIRED_COLS if c not in roster_df.columns]
    if missing:
        warnings.warn(
            f"roster_df missing columns: {missing}. Required: "
            f"{list(AGE_RECOVERY_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    if "birth_date" not in roster_df.columns and "age" not in roster_df.columns:
        warnings.warn(
            "roster_df needs at least one of 'birth_date' or 'age'.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# AgeRecoveryCalculator
# ---------------------------------------------------------------------------

class AgeRecoveryCalculator:
    """Age → recovery coefficient (Feature 3.12). Stateless transformer."""

    def __init__(
        self,
        peak_age: float = RECOVERY_PEAK_AGE,
        decay_k:  float = RECOVERY_DECAY_K,
        recovery_floor: float = RECOVERY_FLOOR,
    ) -> None:
        if peak_age <= 0:
            raise ValueError("peak_age must be > 0")
        if decay_k < 0:
            raise ValueError("decay_k must be >= 0")
        if not (0.0 <= recovery_floor <= 1.0):
            raise ValueError("recovery_floor must be in [0, 1]")
        self._peak  = float(peak_age)
        self._k     = float(decay_k)
        self._floor = float(recovery_floor)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def coefficient(self, age_years: float) -> float:
        """Single-value evaluator using this calculator's parameters."""
        return recovery_coefficient(age_years, self._peak, self._k, self._floor)

    def compute(self, roster_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        """Return one row per player with recovery_coefficient @ ``as_of_date``."""
        if not _validate(roster_df):
            return _empty_output()
        if len(roster_df) == 0:
            return _empty_output()

        as_of = _parse_date(as_of_date)

        rows = roster_df.to_dicts()
        seen: set[int] = set()
        out_rows: list[dict] = []
        for r in rows:
            pid_raw = r.get("player_id")
            if pid_raw is None:
                continue
            pid = int(pid_raw)
            if pid in seen:
                continue          # dedupe; first occurrence wins
            seen.add(pid)

            age_val: float | None = None
            birth = r.get("birth_date") if "birth_date" in r else None
            if birth not in (None, ""):
                age_val = _age_from_birth(str(birth), as_of)
            if age_val is None and r.get("age") is not None:
                try:
                    age_val = float(r["age"])
                except (TypeError, ValueError):
                    age_val = None
            if age_val is None:
                continue          # skip players with no usable age

            out_rows.append({
                "player_id":            pid,
                "as_of_date":           as_of_date,
                "age_years":            float(age_val),
                "recovery_coefficient": self.coefficient(age_val),
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=AGE_RECOVERY_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_age_recovery(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write age-recovery DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"age_recovery_{as_of_date}.parquet"
    for col, dtype in AGE_RECOVERY_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(AGE_RECOVERY_SCHEMA.keys())).write_parquet(path)
    return path
