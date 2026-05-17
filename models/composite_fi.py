"""Composite Fatigue Index — Feature 3.17.

The composite FI is the *one number* fed into the rating multiplier
(3.18) and ultimately into the Rust simulation engine. It collapses all
of Phase 3's fatigue signals into a single ``[0.0, 1.0]`` score per
player per game.

Design constraint: **the weights must be auditable by Bob**. No black
box. A linear-regression-style weighted sum, with every coefficient
named and printable. When Bob sees a fatigue value of 0.72 he must be
able to point at the row and say "this is the road trip + the back-to-
back + the OT load." That's the contract.

Form
----
For each per-player-per-game row, normalize each signal into a
``[0.0, 1.0]`` *load contribution*, multiply by its named weight, and
sum::

    raw_load = Σ_i  weight_i × normalized_signal_i

Then apply the two per-player multipliers:

    base_fi  = clamp(raw_load × concussion_multiplier /
                     max(recovery_floor, recovery_coefficient),
                     0.0, 1.0)

And add the always-present absolute penalties (rust + playoff load):

    fi = clamp(base_fi + rust_load + playoff_load_penalty, 0.0, 1.0)

where ``rust_load = 1.0 − rust_factor`` (so a fully-recovered player
contributes 0).

The default weights sum to ``1.0`` after each normalized signal saturates
its own segment, so a player simultaneously maxing every channel ends up
at FI ≈ 1.0 before any multiplier or penalty even applies.

Signals and normalization
-------------------------
Each signal column is mapped to a value in ``[0.0, 1.0]`` via a saturating
linear map ``min(1, max(0, x / scale))`` (or a binary 0/1 for flags).

    Signal name              Source feature   Scale to 1.0 at
    ───────────────────────  ───────────────  ────────────────
    sched_density_load       3.1              b2b + 3-in-4 + 5g/7d
    travel_load              3.3              2000 miles in last 7d
    tz_load                  3.4              3 zones in 48h
    circadian_load           3.5              3.0 hours misalignment
    altitude_load            3.6              full altitude penalty value
    toi_load                 3.7              toi_spike_z ≥ 2.0
    st_load                  3.8              st_intensity_score ≥ 1.0
    contact_load             3.9              contact_load_score ≥ 1.0
    ot_load                  3.10             1 OT in last 7d (ot_equiv_toi)
    fight_load               3.11             1 fight in last 14d
    strain_load              3.16             team_strain_score (already 0..1)

Inputs
------
``signals_df`` (Polars) — one row per (player_id, game_id, game_date)::

    player_id                       Int64
    game_id                         Int64        (optional but written through)
    game_date                       Utf8         "YYYY-MM-DD"
    is_b2b                          Boolean      (3.1)
    is_3_in_4                       Boolean      (3.1)
    games_last_7d                   Int64        (3.1)
    miles_last_7d                   Float64      (3.3)  optional
    tz_load_48h                     Float64      (3.4)  optional
    circadian_misalignment_hours    Float64      (3.5)  optional
    altitude_penalty                Float64      (3.6)  optional
    toi_spike_z                     Float64      (3.7)
    st_intensity_score              Float64      (3.8)  optional
    contact_load_score              Float64      (3.9)  optional
    ot_equiv_toi_secs               Float64      (3.10) optional
    fight_score                     Float64      (3.11) optional
    team_strain_score               Float64      (3.16) optional
    recovery_coefficient            Float64      (3.12) ; default 1.0
    fatigue_sensitivity_multiplier  Float64      (3.14) ; default 1.0
    rust_factor                     Float64      (3.13) ; default 1.0
    playoff_load_penalty            Float64      (3.15) ; default 0.0

All optional columns default to a *no-effect* value when absent — the
weighted sum simply omits them. Required columns are ``player_id`` and
``game_date``; everything else is optional but the more present, the
better the score.

``as_of_date`` is written through to every row.

Outputs
-------
One row per (player, game) with COMPOSITE_FI_SCHEMA:

    player_id              Int64
    game_id                Int64
    game_date              Utf8
    as_of_date             Utf8
    fatigue_index          Float64   in [0.0, 1.0]
    raw_load               Float64   pre-multiplier weighted sum
    rust_load              Float64   1.0 − rust_factor
    playoff_load_penalty   Float64   carried through (additive)
    concussion_multiplier  Float64   carried through (applied)
    recovery_coefficient   Float64   carried through (applied)
    component_breakdown    Utf8      JSON of (signal → contribution)

The component breakdown is a JSON string mapping signal name → raw
contribution (``weight × normalized``). Bob can sort, average, and tally
it from the parquet without re-running the model.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "composite_fi_v1"

FI_REQUIRED_COLS = ("player_id", "game_date")

COMPOSITE_FI_SCHEMA: dict[str, pl.DataType] = {
    "player_id":              pl.Int64,
    "game_id":                pl.Int64,
    "game_date":              pl.Utf8,
    "as_of_date":             pl.Utf8,
    "fatigue_index":          pl.Float64,
    "raw_load":               pl.Float64,
    "rust_load":              pl.Float64,
    "playoff_load_penalty":   pl.Float64,
    "concussion_multiplier":  pl.Float64,
    "recovery_coefficient":   pl.Float64,
    "component_breakdown":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Default weights and per-signal saturation scales.
# Auditable by name; sums to ~1.0 when every channel is saturated.
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "sched_density_load": 0.18,
    "travel_load":        0.10,
    "tz_load":            0.07,
    "circadian_load":     0.05,
    "altitude_load":      0.04,
    "toi_load":           0.16,
    "st_load":            0.08,
    "contact_load":       0.10,
    "ot_load":            0.07,
    "fight_load":         0.05,
    "strain_load":        0.10,
}

# Per-signal saturation scales (signal value at which normalized = 1.0).
SCALE_MILES_7D       = 2000.0   # 3.3
SCALE_TZ_48H         = 3.0      # 3.4
SCALE_CIRCADIAN_HRS  = 3.0      # 3.5
SCALE_ALTITUDE       = 1.0      # 3.6 (already normalized in source)
SCALE_TOI_SPIKE_Z    = 2.0      # 3.7
SCALE_ST_INTENSITY   = 1.0      # 3.8
SCALE_CONTACT        = 1.0      # 3.9
SCALE_OT_EQUIV_SECS  = 420.0    # 3.10 (one OT ≈ 7 min equivalent TOI)
SCALE_FIGHT          = 1.0      # 3.11
SCALE_STRAIN         = 1.0      # 3.16 (already in [0, 1])
RECOVERY_FLOOR_DEFAULT = 0.50

# Schedule density components map to a single ``sched_density_load`` signal:
#   is_b2b           → 0.50
#   is_3_in_4        → 0.30
#   games_last_7d ≥5 → linear to 0.20 in [3, 5]
SCHED_B2B_WEIGHT       = 0.50
SCHED_3_IN_4_WEIGHT    = 0.30
SCHED_DENSE_WEIGHT     = 0.20
SCHED_DENSE_FLOOR      = 3
SCHED_DENSE_CEILING    = 5


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_linear(value: float | None, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return _clamp01(v / scale)


def _normalize_sched_density(
    is_b2b: bool | None,
    is_3_in_4: bool | None,
    games_last_7d: int | float | None,
) -> float:
    b2b   = SCHED_B2B_WEIGHT if bool(is_b2b) else 0.0
    three = SCHED_3_IN_4_WEIGHT if bool(is_3_in_4) else 0.0
    dense = 0.0
    if games_last_7d is not None:
        try:
            g = float(games_last_7d)
        except (TypeError, ValueError):
            g = 0.0
        if g >= SCHED_DENSE_CEILING:
            dense = SCHED_DENSE_WEIGHT
        elif g > SCHED_DENSE_FLOOR:
            span = max(1, SCHED_DENSE_CEILING - SCHED_DENSE_FLOOR)
            dense = SCHED_DENSE_WEIGHT * (g - SCHED_DENSE_FLOOR) / span
    return _clamp01(b2b + three + dense)


def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in COMPOSITE_FI_SCHEMA.items()}
    )


def _validate(signals_df: pl.DataFrame) -> bool:
    missing = [c for c in FI_REQUIRED_COLS if c not in signals_df.columns]
    if missing:
        warnings.warn(
            f"signals_df missing columns: {missing}. Required: "
            f"{list(FI_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# CompositeFatigueIndex
# ---------------------------------------------------------------------------

class CompositeFatigueIndex:
    """Weighted-sum composite FI (Feature 3.17). Stateless transformer."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        recovery_floor: float = RECOVERY_FLOOR_DEFAULT,
    ) -> None:
        w = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
        for name in w:
            if name not in DEFAULT_WEIGHTS:
                raise ValueError(f"Unknown weight key: {name!r}")
        for name, v in w.items():
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"weight[{name}] must be in [0, 1], got {v}")
        # Fill missing channels from defaults so callers can override a subset.
        for name, default_v in DEFAULT_WEIGHTS.items():
            w.setdefault(name, default_v)

        if not (0.0 < recovery_floor <= 1.0):
            raise ValueError("recovery_floor must be in (0, 1]")

        self._weights        = {k: float(w[k]) for k in DEFAULT_WEIGHTS}
        self._recovery_floor = float(recovery_floor)
        self._version        = MODEL_VERSION

    # ------------------------------------------------------------------
    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def total_weight(self) -> float:
        return sum(self._weights.values())

    # ------------------------------------------------------------------
    def _row_components(self, r: dict) -> dict[str, float]:
        """Return raw weighted contributions per signal for one input row."""
        n_sched = _normalize_sched_density(
            r.get("is_b2b"), r.get("is_3_in_4"), r.get("games_last_7d")
        )
        n_travel  = _normalize_linear(r.get("miles_last_7d"), SCALE_MILES_7D)
        n_tz      = _normalize_linear(r.get("tz_load_48h"), SCALE_TZ_48H)
        n_circ    = _normalize_linear(
            r.get("circadian_misalignment_hours"), SCALE_CIRCADIAN_HRS
        )
        n_alt     = _normalize_linear(r.get("altitude_penalty"), SCALE_ALTITUDE)
        n_toi     = _normalize_linear(r.get("toi_spike_z"), SCALE_TOI_SPIKE_Z)
        n_st      = _normalize_linear(r.get("st_intensity_score"), SCALE_ST_INTENSITY)
        n_contact = _normalize_linear(r.get("contact_load_score"), SCALE_CONTACT)
        n_ot      = _normalize_linear(r.get("ot_equiv_toi_secs"), SCALE_OT_EQUIV_SECS)
        n_fight   = _normalize_linear(r.get("fight_score"), SCALE_FIGHT)
        n_strain  = _normalize_linear(r.get("team_strain_score"), SCALE_STRAIN)

        w = self._weights
        return {
            "sched_density_load": w["sched_density_load"] * n_sched,
            "travel_load":        w["travel_load"]        * n_travel,
            "tz_load":            w["tz_load"]            * n_tz,
            "circadian_load":     w["circadian_load"]     * n_circ,
            "altitude_load":      w["altitude_load"]      * n_alt,
            "toi_load":           w["toi_load"]           * n_toi,
            "st_load":            w["st_load"]            * n_st,
            "contact_load":       w["contact_load"]       * n_contact,
            "ot_load":            w["ot_load"]            * n_ot,
            "fight_load":         w["fight_load"]         * n_fight,
            "strain_load":        w["strain_load"]        * n_strain,
        }

    # ------------------------------------------------------------------
    def compute_row(self, r: dict) -> tuple[float, dict[str, float]]:
        """Compute FI for a single dict-row. Returns (fi, components)."""
        comps    = self._row_components(r)
        raw_load = sum(comps.values())

        recov = r.get("recovery_coefficient")
        try:
            recov_v = float(recov) if recov is not None else 1.0
        except (TypeError, ValueError):
            recov_v = 1.0
        recov_v = max(self._recovery_floor, min(1.0, recov_v))

        concussion = r.get("fatigue_sensitivity_multiplier")
        try:
            conc_v = float(concussion) if concussion is not None else 1.0
        except (TypeError, ValueError):
            conc_v = 1.0
        if conc_v < 1.0:
            conc_v = 1.0

        rust = r.get("rust_factor")
        try:
            rust_v = float(rust) if rust is not None else 1.0
        except (TypeError, ValueError):
            rust_v = 1.0
        rust_v = max(0.0, min(1.0, rust_v))
        rust_load = 1.0 - rust_v

        playoff = r.get("playoff_load_penalty")
        try:
            playoff_v = float(playoff) if playoff is not None else 0.0
        except (TypeError, ValueError):
            playoff_v = 0.0
        playoff_v = max(0.0, playoff_v)

        base = raw_load * conc_v / recov_v
        fi   = _clamp01(base + rust_load + playoff_v)
        return fi, comps

    # ------------------------------------------------------------------
    def compute(self, signals_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        """Return one row per (player, game) with the composite FI."""
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

            fi, comps = self.compute_row(r)

            # Pull-throughs for the output row.
            recov = r.get("recovery_coefficient")
            try:
                recov_v = float(recov) if recov is not None else 1.0
            except (TypeError, ValueError):
                recov_v = 1.0
            concussion = r.get("fatigue_sensitivity_multiplier")
            try:
                conc_v = float(concussion) if concussion is not None else 1.0
            except (TypeError, ValueError):
                conc_v = 1.0
            rust = r.get("rust_factor")
            try:
                rust_v = float(rust) if rust is not None else 1.0
            except (TypeError, ValueError):
                rust_v = 1.0
            rust_v = max(0.0, min(1.0, rust_v))
            playoff = r.get("playoff_load_penalty")
            try:
                playoff_v = float(playoff) if playoff is not None else 0.0
            except (TypeError, ValueError):
                playoff_v = 0.0

            game_id = r.get("game_id")
            try:
                gid = int(game_id) if game_id is not None else 0
            except (TypeError, ValueError):
                gid = 0

            out_rows.append({
                "player_id":              pid,
                "game_id":                gid,
                "game_date":              str(gd_raw),
                "as_of_date":             as_of_date,
                "fatigue_index":          float(fi),
                "raw_load":               float(sum(comps.values())),
                "rust_load":              float(max(0.0, 1.0 - rust_v)),
                "playoff_load_penalty":   float(max(0.0, playoff_v)),
                "concussion_multiplier":  float(max(1.0, conc_v)),
                "recovery_coefficient":   float(recov_v),
                "component_breakdown":    json.dumps(
                    {k: round(v, 6) for k, v in comps.items()},
                    sort_keys=True,
                ),
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=COMPOSITE_FI_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_composite_fi(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write composite-FI DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"composite_fi_{as_of_date}.parquet"
    for col, dtype in COMPOSITE_FI_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(COMPOSITE_FI_SCHEMA.keys())).write_parquet(path)
    return path
