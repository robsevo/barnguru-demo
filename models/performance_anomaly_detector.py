"""Performance anomaly detector — Feature 3.19.

Statistical process control (SPC) over per-player game-score-style
performance metrics. For each player we maintain a rolling 20-game
baseline (mean + std) and emit two complementary signals:

1. **Z-score**: ``(metric − baseline_mean) / baseline_std`` — flags
   *one-game* anomalies. ``z ≤ ANOMALY_Z_THRESHOLD`` (default −2.0)
   triggers ``is_z_anomaly``.

2. **CUSUM**: a one-sided cumulative-sum chart on the *negative* side.
   ``S_t = max(0, S_{t−1} + (baseline_mean − metric_t − K))`` where
   ``K`` is a slack constant in std-units (default 0.5σ). The chart
   fires when ``S_t > H`` (default 4σ × baseline_std). CUSUM catches
   *sustained* small dips that the Z-score test would miss.

A player is flagged as **anomalous** (``is_anomaly = True``) when either
test fires AND the player is not currently on IR. The IR filter is
critical: a player on long-term IR is "anomalous" by construction; the
purpose of this feature is to spot *hidden* injuries — players whose
performance has cratered without an injury report. That is exactly the
signal Bob trusts most: numbers fall first, the body finds out later.

Algorithm — sliding 20-game prior window
----------------------------------------
- The baseline for game *t* uses the *prior* up to 20 games (exclusive
  of game *t* itself). Players need at least ``MIN_BASELINE_GAMES``
  (default 5) prior games to be evaluated; below that the z-score is
  set to ``0.0`` and is_anomaly = False.
- ``baseline_std`` uses the unbiased sample std (N−1). When std is
  exactly 0, z is forced to 0 (no division by zero) and CUSUM continues
  on whatever absolute deviation exists. This is rare in practice — NHL
  game-scores have variance.

Inputs
------
``perf_df`` (Polars) — one row per (player, game)::

    player_id    Int64
    game_id      Int64
    game_date    Utf8        "YYYY-MM-DD"
    metric       Float64     game-score / xGF/60 / any single-game perf number

Optional column:

    is_on_ir     Boolean     True if the player is on IR for this game
                             (default False if column missing)

Outputs
-------
One row per (player, game) with PERFORMANCE_ANOMALY_SCHEMA::

    player_id              Int64
    game_id                Int64
    game_date              Utf8
    as_of_date             Utf8
    metric                 Float64    pass-through
    baseline_mean          Float64    rolling-prior mean
    baseline_std           Float64    rolling-prior std
    baseline_n             Int64      games in the rolling prior window
    z_score                Float64
    cusum                  Float64    one-sided negative CUSUM
    is_z_anomaly           Boolean    z ≤ ANOMALY_Z_THRESHOLD
    is_cusum_anomaly       Boolean    cusum > ANOMALY_CUSUM_H × baseline_std
    consecutive_below_n    Int64      consecutive games with z ≤ −1.0
    is_on_ir               Boolean
    is_anomaly             Boolean    (z or CUSUM) AND not is_on_ir

Calibration knobs (constructor):
- ``window_games``        default 20  — rolling baseline window
- ``min_baseline_games``  default 5
- ``z_threshold``         default −2.0  (note: negative)
- ``cusum_k_sigma``       default 0.5   — slack in std units
- ``cusum_h_sigma``       default 4.0   — alarm threshold in std units
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "performance_anomaly_detector_v1"

PERF_REQUIRED_COLS = ("player_id", "game_id", "game_date", "metric")

PERFORMANCE_ANOMALY_SCHEMA: dict[str, pl.DataType] = {
    "player_id":           pl.Int64,
    "game_id":             pl.Int64,
    "game_date":           pl.Utf8,
    "as_of_date":          pl.Utf8,
    "metric":              pl.Float64,
    "baseline_mean":       pl.Float64,
    "baseline_std":        pl.Float64,
    "baseline_n":          pl.Int64,
    "z_score":             pl.Float64,
    "cusum":               pl.Float64,
    "is_z_anomaly":        pl.Boolean,
    "is_cusum_anomaly":    pl.Boolean,
    "consecutive_below_n": pl.Int64,
    "is_on_ir":            pl.Boolean,
    "is_anomaly":          pl.Boolean,
}


WINDOW_GAMES         = 20
MIN_BASELINE_GAMES   = 5
ANOMALY_Z_THRESHOLD  = -2.0    # z at or below this fires the one-shot alarm
CUSUM_K_SIGMA        = 0.5     # slack
CUSUM_H_SIGMA        = 4.0     # alarm threshold
CONSECUTIVE_Z        = -1.0    # threshold counted in consecutive_below_n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in PERFORMANCE_ANOMALY_SCHEMA.items()}
    )


def _validate(perf_df: pl.DataFrame) -> bool:
    missing = [c for c in PERF_REQUIRED_COLS if c not in perf_df.columns]
    if missing:
        warnings.warn(
            f"perf_df missing columns: {missing}. Required: "
            f"{list(PERF_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _sample_std(values: list[float]) -> float:
    """Unbiased sample standard deviation. ``0.0`` when n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    sq_sum = sum((v - mean) * (v - mean) for v in values)
    return math.sqrt(sq_sum / (n - 1))


# ---------------------------------------------------------------------------
# PerformanceAnomalyDetector
# ---------------------------------------------------------------------------

class PerformanceAnomalyDetector:
    """SPC-style performance anomaly detector (Feature 3.19)."""

    def __init__(
        self,
        window_games:        int   = WINDOW_GAMES,
        min_baseline_games:  int   = MIN_BASELINE_GAMES,
        z_threshold:         float = ANOMALY_Z_THRESHOLD,
        cusum_k_sigma:       float = CUSUM_K_SIGMA,
        cusum_h_sigma:       float = CUSUM_H_SIGMA,
        consecutive_z:       float = CONSECUTIVE_Z,
    ) -> None:
        if window_games < 2:
            raise ValueError("window_games must be >= 2")
        if min_baseline_games < 2:
            raise ValueError("min_baseline_games must be >= 2")
        if min_baseline_games > window_games:
            raise ValueError("min_baseline_games must be <= window_games")
        if z_threshold >= 0.0:
            raise ValueError("z_threshold must be < 0 (this is a *low* alarm)")
        if cusum_k_sigma < 0.0:
            raise ValueError("cusum_k_sigma must be >= 0")
        if cusum_h_sigma <= 0.0:
            raise ValueError("cusum_h_sigma must be > 0")
        if consecutive_z >= 0.0:
            raise ValueError("consecutive_z must be < 0")

        self._window    = int(window_games)
        self._min       = int(min_baseline_games)
        self._z_thresh  = float(z_threshold)
        self._k         = float(cusum_k_sigma)
        self._h         = float(cusum_h_sigma)
        self._c_z       = float(consecutive_z)
        self._version   = MODEL_VERSION

    # ------------------------------------------------------------------
    def _process_player(
        self,
        rows: list[dict],
        as_of_date: str,
    ) -> list[dict]:
        """Walk one player's chronologically-sorted games."""
        out: list[dict] = []
        prior: list[float] = []
        cusum = 0.0
        consecutive = 0

        for r in rows:
            metric = float(r["metric"])
            on_ir  = bool(r.get("is_on_ir") or False)

            # Baseline from the prior window (exclusive of this game).
            window = prior[-self._window:]
            base_n = len(window)
            base_mean = (sum(window) / base_n) if base_n > 0 else 0.0
            base_std  = _sample_std(window) if base_n >= 2 else 0.0

            if base_n >= self._min and base_std > 0.0:
                z = (metric - base_mean) / base_std
            else:
                z = 0.0

            # CUSUM update: one-sided on the negative side. Slack in std
            # units → multiply by current baseline_std (or 1 if 0 so the
            # chart still moves on absolute deviations).
            slack_units = self._k * (base_std if base_std > 0.0 else 1.0)
            cusum_drift = base_mean - metric - slack_units
            cusum = max(0.0, cusum + cusum_drift)

            cusum_alarm = (base_n >= self._min) and (
                cusum > self._h * (base_std if base_std > 0.0 else 1.0)
            )
            z_alarm = (base_n >= self._min) and (z <= self._z_thresh)

            if base_n >= self._min and z <= self._c_z:
                consecutive += 1
            else:
                consecutive = 0

            is_anomaly = (z_alarm or cusum_alarm) and not on_ir

            out.append({
                "player_id":           int(r["player_id"]),
                "game_id":             int(r.get("game_id") or 0),
                "game_date":           str(r["game_date"]),
                "as_of_date":          as_of_date,
                "metric":              float(metric),
                "baseline_mean":       float(base_mean),
                "baseline_std":        float(base_std),
                "baseline_n":          int(base_n),
                "z_score":             float(z),
                "cusum":               float(cusum),
                "is_z_anomaly":        bool(z_alarm),
                "is_cusum_anomaly":    bool(cusum_alarm),
                "consecutive_below_n": int(consecutive),
                "is_on_ir":            bool(on_ir),
                "is_anomaly":          bool(is_anomaly),
            })

            prior.append(metric)

        return out

    # ------------------------------------------------------------------
    def compute(self, perf_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        """Return one row per (player, game) with z + CUSUM + alarm flags."""
        if not _validate(perf_df):
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(perf_df) == 0:
            return _empty_output()

        # Clean + sort.
        clean_rows: list[dict] = []
        for r in perf_df.to_dicts():
            pid_raw = r.get("player_id")
            gd_raw  = r.get("game_date")
            metric  = r.get("metric")
            if pid_raw is None or gd_raw is None or metric is None:
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
                m = float(metric)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(m):
                continue
            clean_rows.append({
                "player_id": pid,
                "game_id":   r.get("game_id"),
                "game_date": str(gd_raw),
                "metric":    m,
                "is_on_ir":  bool(r.get("is_on_ir") or False),
            })

        if not clean_rows:
            return _empty_output()

        clean_rows.sort(key=lambda r: (r["player_id"], r["game_date"],
                                       int(r["game_id"] or 0)))

        out_rows: list[dict] = []
        block: list[dict] = []
        current_pid: int | None = None
        for r in clean_rows:
            if current_pid is None or r["player_id"] == current_pid:
                block.append(r)
                current_pid = r["player_id"]
            else:
                out_rows.extend(self._process_player(block, as_of_date))
                block = [r]
                current_pid = r["player_id"]
        if block:
            out_rows.extend(self._process_player(block, as_of_date))

        return pl.DataFrame(out_rows, schema=PERFORMANCE_ANOMALY_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_performance_anomaly(
    df: pl.DataFrame, output_dir: Path, as_of_date: str,
) -> Path:
    """Write performance-anomaly DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"performance_anomaly_{as_of_date}.parquet"
    for col, dtype in PERFORMANCE_ANOMALY_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(PERFORMANCE_ANOMALY_SCHEMA.keys())).write_parquet(path)
    return path
