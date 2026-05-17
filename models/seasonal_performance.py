"""Seasonal Performance Factor — Feature 3.22.

A per-(player, game) motivational modifier driven by *where in the
season* the game falls. The crowd-market prices an 82-game schedule
evenly; reality does not. November is full hands-on-the-wheel hockey;
January is the dog days where teams already mathematically locked into
playoff positioning go through the motions; April is the playoff push
where teams fighting for a wild-card spot raise their game two full
notches and old legs find one more gear.

Output
------
``seasonal_motivation_factor`` lives in ``[MIN_FACTOR, MAX_FACTOR]``
(default ``[-0.05, +0.03]``) in *performance-delta* convention:

    +ve → player likely outperforms baseline this month
    −ve → player likely underperforms

Composite FI (3.17) is a drag score (higher = worse), so a downstream
consumer that wants to add the seasonal factor into the FI sum should
negate it: ``fi_load = -seasonal_motivation_factor``.

Drivers
-------
Four inputs interact linearly with month-of-season effects:

    month_of_season       1–12  (calendar month)
    playoff_prob          ∈ [0, 1] — boosts April push, dampens dog days
    age                   skater age — older players slog harder mid-season
    games_played_ytd      season GP through this game — fatigue compounder

Formula::

    factor = base_month_effect[month]
           + α_playoff * (playoff_prob - 0.5) * playoff_month_gate
           + α_age     * (age - AGE_PIVOT) / 10 * dog_days_gate
           + α_gp      * (gp - GP_PIVOT) / GP_PIVOT * dog_days_gate
    factor = clamp(factor, MIN_FACTOR, MAX_FACTOR)

``dog_days_gate`` is 1.0 for Jan/Feb (the mid-season slog), 0.0 elsewhere.
``playoff_month_gate`` is 1.0 for Mar/Apr (the push), 0.0 elsewhere.

The four base-month values and three α multipliers are the model's
*fittable* parameters. Bob's eyeball calibration is encoded as
``DEFAULT_BASE_MONTH`` and ``DEFAULT_ALPHAS`` and can be re-fit from
real residuals via ``fit()``.

Inputs to ``predict()``
-----------------------
``inputs_df`` (Polars) — one row per (player, game)::

    player_id         Int64
    game_id           Int64    (optional, default 0)
    game_date         Utf8     "YYYY-MM-DD"
    playoff_prob      Float64  (optional, default 0.5)
    age               Float64  (optional, default AGE_PIVOT)
    games_played_ytd  Int64    (optional, default GP_PIVOT)

Outputs
-------
``SEASONAL_PERFORMANCE_SCHEMA`` — one row per input row.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "seasonal_performance_v1"

# Spec range: [-0.05, +0.03]
MIN_FACTOR: float = -0.05
MAX_FACTOR: float = +0.03

# Centered pivots — what the model treats as the "default" player-game.
AGE_PIVOT: float = 27.0    # skater median career age
GP_PIVOT:  int   = 41      # half a full schedule

# Calendar months that count as "dog days" vs. "playoff push".
DOG_DAYS_MONTHS:      tuple[int, ...] = (1, 2)        # Jan, Feb
PLAYOFF_PUSH_MONTHS:  tuple[int, ...] = (3, 4)        # Mar, Apr

# ---------------------------------------------------------------------------
# Calibrated defaults — Bob's eyeball, refittable from data.
# ---------------------------------------------------------------------------

# Base month effect: signed performance delta in performance-delta
# convention (positive = boost). Off-season months are 0.
DEFAULT_BASE_MONTH: dict[int, float] = {
    1:  -0.040,   # January — dog days mid-season slog
    2:  -0.020,   # February — fatigue still high
    3:  +0.005,   # March — playoff picture sharpening
    4:  +0.025,   # April — playoff push / season-end stretch
    5:  +0.020,   # May — playoffs
    6:  +0.015,   # June — playoffs
    7:   0.0,     # July — off-season
    8:   0.0,     # August — off-season
    9:  -0.005,   # September — preseason ramp
    10: -0.025,   # October — early-season rust
    11: -0.005,   # November — system settled
    12: +0.005,   # December — hits stride
}

# Alpha multipliers on the three modulators.
DEFAULT_ALPHAS: dict[str, float] = {
    "playoff":  +0.030,   # max +0.015 when playoff_prob = 1.0 in Mar/Apr
    "age":      -0.010,   # 10y above pivot → −0.010 added in dog days
    "gp":       -0.010,   # full schedule played → −0.010 in dog days
}


SEASONAL_PERFORMANCE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":                      pl.Int64,
    "game_id":                        pl.Int64,
    "game_date":                      pl.Utf8,
    "as_of_date":                     pl.Utf8,
    "month_of_season":                pl.Int64,
    "playoff_prob":                   pl.Float64,
    "age":                            pl.Float64,
    "games_played_ytd":               pl.Int64,
    "base_month_effect":              pl.Float64,
    "playoff_adj":                    pl.Float64,
    "age_adj":                        pl.Float64,
    "gp_adj":                         pl.Float64,
    "seasonal_motivation_factor":     pl.Float64,
    "model_version":                  pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in SEASONAL_PERFORMANCE_SCHEMA.items()}
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# SeasonalPerformanceFactor
# ---------------------------------------------------------------------------

class SeasonalPerformanceFactor:
    """Per-(player, game) seasonal motivational modifier (Feature 3.22)."""

    def __init__(
        self,
        base_month: dict[int, float] | None = None,
        alphas:     dict[str, float] | None = None,
        min_factor: float = MIN_FACTOR,
        max_factor: float = MAX_FACTOR,
        age_pivot:  float = AGE_PIVOT,
        gp_pivot:   int   = GP_PIVOT,
    ) -> None:
        if min_factor >= max_factor:
            raise ValueError("min_factor must be < max_factor")
        if age_pivot <= 0:
            raise ValueError("age_pivot must be > 0")
        if gp_pivot <= 0:
            raise ValueError("gp_pivot must be > 0")

        bm = dict(DEFAULT_BASE_MONTH) if base_month is None else dict(base_month)
        # Validate keys are 1..12.
        for month in bm:
            if not isinstance(month, int) or not (1 <= month <= 12):
                raise ValueError(f"base_month key must be int in [1, 12], got {month!r}")
        for m in range(1, 13):
            bm.setdefault(m, 0.0)

        al = dict(DEFAULT_ALPHAS) if alphas is None else dict(alphas)
        for key in ("playoff", "age", "gp"):
            al.setdefault(key, DEFAULT_ALPHAS[key])

        self._base_month = {int(k): float(v) for k, v in bm.items()}
        self._alphas     = {k: float(v) for k, v in al.items()}
        self._min        = float(min_factor)
        self._max        = float(max_factor)
        self._age_pivot  = float(age_pivot)
        self._gp_pivot   = int(gp_pivot)
        self._version    = MODEL_VERSION

    # ------------------------------------------------------------------
    @property
    def base_month(self) -> dict[int, float]:
        return dict(self._base_month)

    @property
    def alphas(self) -> dict[str, float]:
        return dict(self._alphas)

    # ------------------------------------------------------------------
    def compute_row(
        self,
        month:        int,
        playoff_prob: float,
        age:          float,
        games_played: int,
    ) -> dict:
        base = self._base_month.get(int(month), 0.0)

        dog_gate    = 1.0 if int(month) in DOG_DAYS_MONTHS     else 0.0
        push_gate   = 1.0 if int(month) in PLAYOFF_PUSH_MONTHS else 0.0

        playoff_adj = self._alphas["playoff"] * (playoff_prob - 0.5) * push_gate
        age_adj     = self._alphas["age"]     * (age - self._age_pivot) / 10.0 * dog_gate
        gp_adj      = (
            self._alphas["gp"]
            * (games_played - self._gp_pivot) / self._gp_pivot
            * dog_gate
        )

        raw    = base + playoff_adj + age_adj + gp_adj
        factor = _clamp(raw, self._min, self._max)
        return {
            "base_month_effect":         float(base),
            "playoff_adj":               float(playoff_adj),
            "age_adj":                   float(age_adj),
            "gp_adj":                    float(gp_adj),
            "seasonal_motivation_factor": float(factor),
        }

    # ------------------------------------------------------------------
    def predict(
        self,
        inputs_df:  pl.DataFrame,
        as_of_date: str,
    ) -> pl.DataFrame:
        required = ("player_id", "game_date")
        missing  = [c for c in required if c not in inputs_df.columns]
        if missing:
            warnings.warn(
                f"inputs_df missing columns: {missing}. Required: "
                f"{list(required)}.",
                DataMissingWarning, stacklevel=2,
            )
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(inputs_df) == 0:
            return _empty_output()

        rows = inputs_df.to_dicts()
        out_rows: list[dict] = []
        for r in rows:
            pid_raw = r.get("player_id")
            gd_raw  = r.get("game_date")
            if pid_raw is None or gd_raw is None:
                continue
            try:
                pid = int(pid_raw)
                dt  = _parse_date(str(gd_raw))
            except (TypeError, ValueError):
                continue
            month = dt.month

            try:
                pp = float(r.get("playoff_prob", 0.5))
            except (TypeError, ValueError):
                pp = 0.5
            if not math.isfinite(pp):
                pp = 0.5
            pp = max(0.0, min(1.0, pp))

            try:
                age = float(r.get("age", self._age_pivot))
            except (TypeError, ValueError):
                age = self._age_pivot
            if not math.isfinite(age):
                age = self._age_pivot

            try:
                gp = int(r.get("games_played_ytd", self._gp_pivot))
            except (TypeError, ValueError):
                gp = self._gp_pivot

            metrics = self.compute_row(month, pp, age, gp)

            gid_raw = r.get("game_id")
            try:
                gid = int(gid_raw) if gid_raw is not None else 0
            except (TypeError, ValueError):
                gid = 0

            out_rows.append({
                "player_id":         pid,
                "game_id":           gid,
                "game_date":         str(gd_raw),
                "as_of_date":        as_of_date,
                "month_of_season":   int(month),
                "playoff_prob":      float(pp),
                "age":               float(age),
                "games_played_ytd":  int(gp),
                "model_version":     self._version,
                **metrics,
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=SEASONAL_PERFORMANCE_SCHEMA)

    # ------------------------------------------------------------------
    def fit(self, training_df: pl.DataFrame) -> "SeasonalPerformanceFactor":
        """Re-fit base month effects from observed residuals.

        Expected columns: ``month_of_season`` (int, 1–12) and ``residual``
        (float, observed minus baseline). Each month's base effect is set
        to the mean residual for that month, clamped to the output range.
        Alphas are kept at their default values — Bob can override the
        full alpha set via the constructor when richer training data is
        available.
        """
        required = {"month_of_season", "residual"}
        missing  = required - set(training_df.columns)
        if missing:
            warnings.warn(
                f"training_df missing columns: {sorted(missing)}. "
                "Defaults retained.",
                DataMissingWarning, stacklevel=2,
            )
            return self

        if len(training_df) == 0:
            warnings.warn(
                "training_df is empty. Defaults retained.",
                DataMissingWarning, stacklevel=2,
            )
            return self

        per_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
        for r in training_df.to_dicts():
            try:
                m  = int(r["month_of_season"])
                rv = float(r["residual"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (1 <= m <= 12) or not math.isfinite(rv):
                continue
            per_month[m].append(rv)

        new_base = dict(self._base_month)
        for m, vals in per_month.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            new_base[m] = _clamp(mean, self._min, self._max)
        self._base_month = new_base
        return self

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version":     self._version,
            "base_month":  self._base_month,
            "alphas":      self._alphas,
            "min":         self._min,
            "max":         self._max,
            "age_pivot":   self._age_pivot,
            "gp_pivot":    self._gp_pivot,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "SeasonalPerformanceFactor":
        payload = joblib.load(Path(path))
        obj = cls(
            base_month = payload.get("base_month"),
            alphas     = payload.get("alphas"),
            min_factor = payload.get("min", MIN_FACTOR),
            max_factor = payload.get("max", MAX_FACTOR),
            age_pivot  = payload.get("age_pivot", AGE_PIVOT),
            gp_pivot   = payload.get("gp_pivot",  GP_PIVOT),
        )
        obj._version = payload.get("version", MODEL_VERSION)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_seasonal_performance(
    df: pl.DataFrame, output_dir: Path, as_of_date: str,
) -> Path:
    """Write seasonal-performance DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"seasonal_performance_{as_of_date}.parquet"
    for col, dtype in SEASONAL_PERFORMANCE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(SEASONAL_PERFORMANCE_SCHEMA.keys())).write_parquet(path)
    return path
