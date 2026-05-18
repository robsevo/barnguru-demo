"""Composite Confidence Index — Phase 17.

Phase 3 (composite_fi) measures physical fatigue — what degrades a player's
*execution* (skating speed, finishing under load). Phase 17 (this module)
measures confidence — what biases their *decision-making* (shoot-vs-pass,
pinch-vs-retreat, aggressive forecheck-vs-cycle). They are independent
mechanisms. A confident-but-exhausted player and an unconfident-but-fresh
player should not look the same to the simulator — they don't here.

Form
----
For each per-player-per-game row, normalize each signal into a signed
contribution in approximately ``[-1, +1]`` (each signal documents its own
sign convention), multiply by its named weight, and sum::

    player_score = Σ w_i × normalized_player_signal_i      # 17.1 – 17.15
    team_score   = Σ w_j × normalized_team_signal_j        # 17.16 – 17.22
    confidence_index = clamp(player_score × 0.7
                           + team_score × 0.3, -1.0, +1.0)

Signed because confidence has direction: +0.7 → home-run pass and aggressive
forecheck; −0.7 → dump-and-chase, low-risk. Range matches the engine
multiplier knobs in ``confidence_rating_multiplier``.

Default weights are auditable by Bob — every coefficient has a name, every
contribution lands in the ``component_breakdown`` JSON.

Inputs
------
``signals_df`` (Polars) — one row per (player_id, game_id, game_date). All
signal columns are optional; missing columns contribute 0 to the composite.
That keeps the model resilient to early Phase 17 rollout where some signals
ship before others.

    player_id                Int64
    game_id                  Int64
    game_date                Utf8         "YYYY-MM-DD"
    is_goalie                Boolean      affects which signals apply
    # Player signals (17.1 – 17.15) — signed, normalized to ~[-1, +1]
    hot_hand_signal          Float64
    ewma_form_signal         Float64
    toi_trust_trend          Float64
    role_usage_delta         Float64
    healthy_scratch_flag     Float64
    point_drought_z          Float64
    linemate_form_drag       Float64
    bounceback_index         Float64
    active_injury_drag       Float64
    targeting_pressure       Float64
    media_sentiment          Float64
    home_away_split          Float64
    trade_rumor_pressure     Float64
    contract_pressure        Float64
    referee_bias             Float64
    # Team signals (17.16 – 17.22) — signed, normalized
    team_streak              Float64
    score_adj_corsi_trend    Float64
    special_teams_trend      Float64
    coach_challenge_rate     Float64
    comeback_quality         Float64
    goalie_confidence        Float64
    team_injury_context      Float64

Outputs
-------
One row per (player_id, game_id) with COMPOSITE_CONFIDENCE_SCHEMA:

    player_id              Int64
    game_id                Int64
    game_date              Utf8
    as_of_date             Utf8
    confidence_index       Float64   in [-1.0, +1.0]
    player_score           Float64   pre-blend player component sum
    team_score             Float64   pre-blend team component sum
    component_breakdown    Utf8      JSON of (signal → contribution)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "composite_confidence_v1"

REQUIRED_COLS = ("player_id", "game_date")

COMPOSITE_CONFIDENCE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":           pl.Int64,
    "game_id":             pl.Int64,
    "game_date":           pl.Utf8,
    "as_of_date":          pl.Utf8,
    "confidence_index":    pl.Float64,
    "player_score":        pl.Float64,
    "team_score":          pl.Float64,
    "component_breakdown": pl.Utf8,
}


# Player-side weights (sum to 1.0). Signs come from the signal values
# themselves — these are positive magnitudes that scale the signed input.
DEFAULT_PLAYER_WEIGHTS: dict[str, float] = {
    "hot_hand_signal":      0.13,
    "ewma_form_signal":     0.08,
    "toi_trust_trend":      0.10,
    "role_usage_delta":     0.09,
    "healthy_scratch_flag": 0.07,
    "point_drought_z":      0.05,
    "linemate_form_drag":   0.05,
    "bounceback_index":     0.04,
    "active_injury_drag":   0.08,
    "targeting_pressure":   0.06,
    "media_sentiment":      0.07,
    "home_away_split":      0.05,
    "trade_rumor_pressure": 0.06,
    "contract_pressure":    0.05,
    "referee_bias":         0.02,
}

# Team-side weights (sum to 1.0)
DEFAULT_TEAM_WEIGHTS: dict[str, float] = {
    "team_streak":            0.15,
    "score_adj_corsi_trend":  0.20,
    "special_teams_trend":    0.15,
    "coach_challenge_rate":   0.05,
    "comeback_quality":       0.15,
    "goalie_confidence":      0.15,
    "team_injury_context":    0.15,
}

# Team blend into the player-level composite (per Bob's 70/30 lock-in)
DEFAULT_TEAM_BLEND = 0.30


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in COMPOSITE_CONFIDENCE_SCHEMA.items()}
    )


def _validate(signals_df: pl.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in signals_df.columns]
    if missing:
        warnings.warn(
            f"signals_df missing columns: {missing}. Required: {list(REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _f(x) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
        if v != v:  # NaN check
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


class CompositeConfidenceIndex:
    """Phase 17 — composite confidence index per (player, game)."""

    def __init__(
        self,
        player_weights: dict[str, float] | None = None,
        team_weights:   dict[str, float] | None = None,
        team_blend:     float = DEFAULT_TEAM_BLEND,
    ) -> None:
        if not 0.0 <= team_blend <= 1.0:
            raise ValueError(f"team_blend must be in [0, 1], got {team_blend!r}")
        self._player_weights = dict(player_weights or DEFAULT_PLAYER_WEIGHTS)
        self._team_weights   = dict(team_weights   or DEFAULT_TEAM_WEIGHTS)
        self._team_blend     = float(team_blend)
        self._version        = MODEL_VERSION

    @property
    def version(self) -> str:
        return self._version

    @property
    def player_weights(self) -> dict[str, float]:
        return dict(self._player_weights)

    @property
    def team_weights(self) -> dict[str, float]:
        return dict(self._team_weights)

    @property
    def team_blend(self) -> float:
        return self._team_blend

    def _row_components(self, r: dict) -> dict[str, float]:
        comps: dict[str, float] = {}
        for name, w in self._player_weights.items():
            comps[name] = w * _f(r.get(name))
        for name, w in self._team_weights.items():
            # Tag team contributions explicitly so the breakdown stays readable.
            comps[f"team:{name}"] = w * _f(r.get(name))
        return comps

    def compute_row(self, r: dict) -> tuple[float, float, float, dict[str, float]]:
        comps = self._row_components(r)
        player_score = sum(v for k, v in comps.items() if not k.startswith("team:"))
        team_score   = sum(v for k, v in comps.items() if k.startswith("team:"))
        blended = player_score * (1.0 - self._team_blend) + team_score * self._team_blend
        confidence = _clamp(blended, -1.0, +1.0)
        return confidence, player_score, team_score, comps

    def compute(self, signals_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
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
            if pid_raw is None or gd_raw is None:
                continue
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                continue
            try:
                _parse_date(str(gd_raw))
            except (TypeError, ValueError):
                continue

            confidence, p_score, t_score, comps = self.compute_row(r)
            game_id = r.get("game_id")
            try:
                gid = int(game_id) if game_id is not None else 0
            except (TypeError, ValueError):
                gid = 0

            out_rows.append({
                "player_id":           pid,
                "game_id":             gid,
                "game_date":           str(gd_raw),
                "as_of_date":          as_of_date,
                "confidence_index":    float(confidence),
                "player_score":        float(p_score),
                "team_score":          float(t_score),
                "component_breakdown": json.dumps(
                    {k: round(v, 6) for k, v in comps.items() if abs(v) > 1e-9},
                    sort_keys=True,
                ),
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=COMPOSITE_CONFIDENCE_SCHEMA)


def write_composite_confidence(
    df: pl.DataFrame, output_dir: Path, as_of_date: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"composite_confidence_{as_of_date}.parquet"
    for col, dtype in COMPOSITE_CONFIDENCE_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(COMPOSITE_CONFIDENCE_SCHEMA.keys())).write_parquet(path)
    return path
