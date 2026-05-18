"""Confidence Rating Multiplier — Feature 17.25.

Maps the signed composite confidence index (Phase 17) into engine knobs that
the Rust simulator (or downstream analysis) applies to per-decision
probabilities. Analog of ``fi_rating_multiplier`` for fatigue.

Three knobs, each calibrated so a fully-confident player (±1.0) swings ~5-6%
in one direction — on par with the hot_hand engine weight of ~3% as a sanity
ceiling. The slopes are below the rust engine wiring lands; for now the
Python side ships these so the dashboard can show "expected shoot bias",
"expected risk bias", etc.

    shoot_bias    = 1.0 + 0.06 × c     # ±6% on shoot-vs-pass probability
    risk_bias     = 1.0 + 0.05 × c     # ±5% on pinch / aggressive forecheck
    turnover_bias = 1.0 + 0.04 × c     # +4% turnover prob when confident

All three scale in the SAME direction with confidence. Internally consistent
because confidence drives aggressive play: a confident player shoots more,
pinches more, and turns the puck over more (because the aggressive plays
they're attempting are inherently riskier). A passive/unconfident player
dumps it in — fewer turnovers, but also fewer chances created.

Inputs
------
``conf_df`` (Polars) — output of ``composite_confidence.compute()``, with at
minimum::

    player_id              Int64
    game_id                Int64
    game_date              Utf8
    as_of_date             Utf8
    confidence_index       Float64    in [-1.0, +1.0]

Outputs
-------
One row per (player_id, game_id) with CONFIDENCE_MULTIPLIER_SCHEMA.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


MODEL_VERSION = "confidence_multiplier_v1"

SHOOT_BIAS_SLOPE    = 0.06
RISK_BIAS_SLOPE     = 0.05
TURNOVER_BIAS_SLOPE = 0.04

CONFIDENCE_MULTIPLIER_SCHEMA: dict[str, pl.DataType] = {
    "player_id":         pl.Int64,
    "game_id":           pl.Int64,
    "game_date":         pl.Utf8,
    "as_of_date":        pl.Utf8,
    "confidence_index":  pl.Float64,
    "shoot_bias":        pl.Float64,
    "risk_bias":         pl.Float64,
    "turnover_bias":     pl.Float64,
}


def _f(x, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except (TypeError, ValueError):
        return default


def shoot_bias(c: float) -> float:
    return 1.0 + SHOOT_BIAS_SLOPE * _f(c)


def risk_bias(c: float) -> float:
    return 1.0 + RISK_BIAS_SLOPE * _f(c)


def turnover_bias(c: float) -> float:
    return 1.0 + TURNOVER_BIAS_SLOPE * _f(c)


class ConfidenceRatingMultiplier:
    """Translate composite confidence → engine knob multipliers."""

    def __init__(self) -> None:
        self._version = MODEL_VERSION

    @property
    def version(self) -> str:
        return self._version

    def compute(self, conf_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        if "confidence_index" not in conf_df.columns:
            return pl.DataFrame(
                {c: pl.Series([], dtype=t) for c, t in CONFIDENCE_MULTIPLIER_SCHEMA.items()}
            )
        if len(conf_df) == 0:
            return pl.DataFrame(
                {c: pl.Series([], dtype=t) for c, t in CONFIDENCE_MULTIPLIER_SCHEMA.items()}
            )

        out_rows: list[dict] = []
        for r in conf_df.to_dicts():
            c = _f(r.get("confidence_index"))
            out_rows.append({
                "player_id":        int(r.get("player_id") or 0),
                "game_id":          int(r.get("game_id") or 0),
                "game_date":        str(r.get("game_date") or ""),
                "as_of_date":       as_of_date,
                "confidence_index": float(c),
                "shoot_bias":       float(shoot_bias(c)),
                "risk_bias":        float(risk_bias(c)),
                "turnover_bias":    float(turnover_bias(c)),
            })
        return pl.DataFrame(out_rows, schema=CONFIDENCE_MULTIPLIER_SCHEMA)


def write_confidence_multiplier(
    df: pl.DataFrame, output_dir: Path, as_of_date: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confidence_multiplier_{as_of_date}.parquet"
    for col, dtype in CONFIDENCE_MULTIPLIER_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(CONFIDENCE_MULTIPLIER_SCHEMA.keys())).write_parquet(path)
    return path
