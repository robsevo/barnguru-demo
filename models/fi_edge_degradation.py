"""FI → EDGE metric degradation model — Feature 3.21.

Given a player's composite Fatigue Index (FI, 3.17) and their EDGE
skating baseline (2.20), predict the *expected drop* in each EDGE
metric for that game. This is the calibration layer that lets the
per-player behavioral net (2.22) condition on "what does this player
look like at FI = 0.7 vs. their rested ceiling?" — instead of guessing,
the net consumes regressed deltas grounded in B2B / road-trip /
high-load games.

Model form
----------
For each EDGE metric ``m`` we fit a *bounded* linear regression on the
historical paired observations::

    observed_delta_m = α_m  +  β_m × FI

where:

    observed_delta_m = (metric_value_in_game − baseline_m) / baseline_m

i.e. the *relative* drop from rested baseline. Negative values denote
degradation (slower skating, less distance, less carry-in). FI ≥ 0 by
construction; the slope ``β_m`` should be negative for "load" metrics
(speed, distance, carry%, burst count) — that is the falsifiable check.

The model is intentionally **a per-metric simple linear regression**,
not a multivariate model — Bob has to be able to audit *exactly* how
the FI score moves each output.

Output range guard
------------------
Even at FI = 1.0 the predicted relative delta is clipped to
``[MIN_DELTA, MAX_DELTA]`` (default ``[-0.40, +0.05]``). A 40 % drop is
already extreme; we never want a runaway negative or a positive that
would imply a tired player skates *faster*.

Inputs
------
``training_df`` (Polars) — one row per (player, game) with paired
``FI`` and observed EDGE deltas relative to baseline::

    fi                         Float64    composite FI (3.17) for this game
    delta_speed_vs_baseline    Float64    (observed / baseline) − 1
    delta_distance_vs_baseline Float64
    delta_carry_vs_baseline    Float64
    delta_burst_vs_baseline    Float64

At least ``MIN_OBS`` rows are required (default 50). Below the threshold
the model raises a ``DataMissingWarning`` and falls back to the
``DEFAULT_COEFFS`` constants — these are conservative literature priors
that downstream consumers can use without crashing.

Outputs
-------
``predict(fi_df)`` consumes a frame with columns ``(player_id, game_id,
game_date, fi)`` and returns one row per input with the predicted
deltas + an aggregate ``predicted_load_factor`` (mean across the four
metrics, also clipped). Schema is ``FI_EDGE_DEGRADATION_SCHEMA``.
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


MODEL_VERSION = "fi_edge_degradation_v1"

# Metric names — the four EDGE deltas the behavioral net consumes.
METRICS: tuple[str, ...] = (
    "speed_vs_baseline",
    "distance_vs_baseline",
    "carry_vs_baseline",
    "burst_vs_baseline",
)

# Sample-size floor below which we refuse to fit and fall back to priors.
MIN_OBS = 50

# Output clamp range. Negative values dominate (tired players lose
# output); a small positive ceiling accommodates noise / over-prediction.
MIN_DELTA: float = -0.40
MAX_DELTA: float = +0.05

# Conservative literature-style priors (used when not enough data to fit).
# All slopes are negative — higher FI → lower output. Intercept is small
# positive to absorb non-fatigue noise floor.
DEFAULT_COEFFS: dict[str, tuple[float, float]] = {
    "speed_vs_baseline":    (0.01, -0.12),   # α, β
    "distance_vs_baseline": (0.01, -0.18),
    "carry_vs_baseline":    (0.00, -0.20),
    "burst_vs_baseline":    (0.00, -0.30),
}


FI_EDGE_DEGRADATION_SCHEMA: dict[str, pl.DataType] = {
    "player_id":               pl.Int64,
    "game_id":                 pl.Int64,
    "game_date":               pl.Utf8,
    "as_of_date":              pl.Utf8,
    "fi":                      pl.Float64,
    "speed_vs_baseline":       pl.Float64,
    "distance_vs_baseline":    pl.Float64,
    "carry_vs_baseline":       pl.Float64,
    "burst_vs_baseline":       pl.Float64,
    "predicted_load_factor":   pl.Float64,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in FI_EDGE_DEGRADATION_SCHEMA.items()}
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return max(lo, min(hi, v))


def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    """Simple least-squares fit. Returns (α, β) where ``y ≈ α + β x``.

    Falls back to (mean(y), 0) when ``x`` has zero variance.
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den <= 0.0:
        return float(y_mean), 0.0
    beta  = num / den
    alpha = y_mean - beta * x_mean
    return float(alpha), float(beta)


# ---------------------------------------------------------------------------
# FIEdgeDegradationModel
# ---------------------------------------------------------------------------

class FIEdgeDegradationModel:
    """Per-EDGE-metric linear regression on FI (Feature 3.21).

    The model is **per-metric independent** — fitting one regression per
    EDGE channel keeps every coefficient auditable. Coefficients are
    stored as ``self.coeffs[metric] = (alpha, beta)``.
    """

    def __init__(
        self,
        min_obs:   int   = MIN_OBS,
        min_delta: float = MIN_DELTA,
        max_delta: float = MAX_DELTA,
    ) -> None:
        if min_obs < 2:
            raise ValueError("min_obs must be >= 2")
        if not math.isfinite(min_delta) or not math.isfinite(max_delta):
            raise ValueError("min_delta / max_delta must be finite")
        if min_delta >= max_delta:
            raise ValueError("min_delta must be < max_delta")

        self._min_obs   = int(min_obs)
        self._min_delta = float(min_delta)
        self._max_delta = float(max_delta)
        self._version   = MODEL_VERSION
        self.coeffs: dict[str, tuple[float, float]] = dict(DEFAULT_COEFFS)
        self.is_fitted: bool = False
        self.n_train: int = 0

    # ------------------------------------------------------------------
    @property
    def metrics(self) -> tuple[str, ...]:
        return METRICS

    # ------------------------------------------------------------------
    def fit(self, training_df: pl.DataFrame) -> "FIEdgeDegradationModel":
        """Fit one regression per EDGE metric on observed deltas.

        Falls back to ``DEFAULT_COEFFS`` (with a warning) when
        ``len(training_df) < self._min_obs``.
        """
        required = {"fi", *(f"delta_{m}" for m in METRICS)}
        missing  = required - set(training_df.columns)
        if missing:
            warnings.warn(
                f"training_df missing columns: {sorted(missing)}. "
                "Falling back to default coefficients.",
                DataMissingWarning, stacklevel=2,
            )
            self.coeffs = dict(DEFAULT_COEFFS)
            self.is_fitted = False
            self.n_train   = 0
            return self

        if len(training_df) < self._min_obs:
            warnings.warn(
                f"Only {len(training_df)} training rows; need ≥ {self._min_obs}. "
                "Falling back to default coefficients.",
                DataMissingWarning, stacklevel=2,
            )
            self.coeffs = dict(DEFAULT_COEFFS)
            self.is_fitted = False
            self.n_train   = len(training_df)
            return self

        new_coeffs: dict[str, tuple[float, float]] = {}
        # Pull as python lists once, then run four OLS fits.
        rows = training_df.to_dicts()
        fi_vals: list[float] = []
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for r in rows:
            try:
                f = float(r["fi"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f):
                continue
            try:
                deltas = {
                    m: float(r[f"delta_{m}"])
                    for m in METRICS
                }
            except (TypeError, ValueError, KeyError):
                continue
            if not all(math.isfinite(v) for v in deltas.values()):
                continue
            fi_vals.append(f)
            for m, v in deltas.items():
                per_metric[m].append(v)

        if len(fi_vals) < self._min_obs:
            warnings.warn(
                f"After cleaning, only {len(fi_vals)} usable rows; need ≥ "
                f"{self._min_obs}. Falling back to default coefficients.",
                DataMissingWarning, stacklevel=2,
            )
            self.coeffs = dict(DEFAULT_COEFFS)
            self.is_fitted = False
            self.n_train   = len(fi_vals)
            return self

        for m in METRICS:
            new_coeffs[m] = _ols(fi_vals, per_metric[m])
        self.coeffs    = new_coeffs
        self.is_fitted = True
        self.n_train   = len(fi_vals)
        return self

    # ------------------------------------------------------------------
    def predict_one(self, fi: float) -> dict[str, float]:
        """Return ``{metric_name: predicted_delta}`` for a single FI value."""
        try:
            f = float(fi)
        except (TypeError, ValueError):
            f = 0.0
        if not math.isfinite(f):
            f = 0.0
        f = max(0.0, min(1.0, f))   # FI must live in [0, 1]
        out: dict[str, float] = {}
        for m, (alpha, beta) in self.coeffs.items():
            raw = alpha + beta * f
            out[m] = _clamp(raw, self._min_delta, self._max_delta)
        return out

    # ------------------------------------------------------------------
    def predict(
        self,
        fi_df:       pl.DataFrame,
        as_of_date:  str,
    ) -> pl.DataFrame:
        """Predict per-EDGE-metric deltas for every (player, game) row."""
        required = ("player_id", "game_date", "fi")
        missing  = [c for c in required if c not in fi_df.columns]
        if missing:
            warnings.warn(
                f"fi_df missing columns: {missing}. Required: {list(required)}.",
                DataMissingWarning, stacklevel=2,
            )
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(fi_df) == 0:
            return _empty_output()

        rows = fi_df.to_dicts()
        out_rows: list[dict] = []
        for r in rows:
            pid_raw = r.get("player_id")
            gd_raw  = r.get("game_date")
            fi_raw  = r.get("fi")
            if pid_raw is None or gd_raw is None or fi_raw is None:
                continue
            try:
                pid = int(pid_raw)
                _parse_date(str(gd_raw))
                f   = float(fi_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f):
                continue

            deltas = self.predict_one(f)
            load_factor = _clamp(
                sum(deltas.values()) / len(deltas),
                self._min_delta, self._max_delta,
            )

            gid = r.get("game_id")
            try:
                gid_i = int(gid) if gid is not None else 0
            except (TypeError, ValueError):
                gid_i = 0

            out_rows.append({
                "player_id":              pid,
                "game_id":                gid_i,
                "game_date":              str(gd_raw),
                "as_of_date":             as_of_date,
                "fi":                     float(max(0.0, min(1.0, f))),
                **deltas,
                "predicted_load_factor":  float(load_factor),
                "model_version":          self._version,
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=FI_EDGE_DEGRADATION_SCHEMA)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version":    self._version,
            "coeffs":     self.coeffs,
            "is_fitted":  self.is_fitted,
            "n_train":    self.n_train,
            "min_obs":    self._min_obs,
            "min_delta":  self._min_delta,
            "max_delta":  self._max_delta,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "FIEdgeDegradationModel":
        payload = joblib.load(Path(path))
        obj = cls(
            min_obs   = payload.get("min_obs",   MIN_OBS),
            min_delta = payload.get("min_delta", MIN_DELTA),
            max_delta = payload.get("max_delta", MAX_DELTA),
        )
        obj._version   = payload.get("version", MODEL_VERSION)
        obj.coeffs     = payload.get("coeffs",   dict(DEFAULT_COEFFS))
        obj.is_fitted  = payload.get("is_fitted", False)
        obj.n_train    = payload.get("n_train",   0)
        return obj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_fi_edge_degradation(
    df: pl.DataFrame, output_dir: Path, as_of_date: str,
) -> Path:
    """Write FI→EDGE prediction DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fi_edge_degradation_{as_of_date}.parquet"
    for col, dtype in FI_EDGE_DEGRADATION_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(FI_EDGE_DEGRADATION_SCHEMA.keys())).write_parquet(path)
    return path
