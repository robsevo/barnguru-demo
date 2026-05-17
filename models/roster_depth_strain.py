"""Roster depth strain — Feature 3.16.

Per-team / per-player metric capturing how much *extra ice time* the
healthy roster has to absorb because teammates are unavailable (IR /
Out). When a top-six forward goes down, his minutes don't vanish — they
get redistributed among the remaining forwards, who play above their
baseline TOI. That excess load accumulates into the composite FI (3.17)
as a roster-strain signal distinct from the player's own schedule load.

Form
----
For each team we compute:

    ir_minutes_secs     = Σ baseline_toi_secs over players with is_on_ir = True
    healthy_skaters     = count of skaters with is_on_ir = False
    extra_per_healthy   = ir_minutes_secs / max(1, healthy_skaters)
    team_strain_score   = clamp(extra_per_healthy / FULL_STRAIN_SECS, 0, 1)

Each healthy player then gets:

    player_strain_secs  = extra_per_healthy
    player_strain_score = team_strain_score

Players on IR have ``player_strain_score = 0.0`` (they don't carry load
because they aren't playing). The score is *team-level*: every healthy
player on a stretched team gets the same strain value. Position-level
weighting (a 4th-line forward absorbing a top-line guy's minutes is more
strained than the reverse) is deferred — v1 keeps the math auditable.

Inputs
------
``roster_df`` (Polars) — one row per skater on the active org list::

    player_id          Int64
    team               Utf8     3-letter abbreviation
    position           Utf8     "F" | "D" | "C" | "LW" | "RW" | "G"
    is_on_ir           Boolean  True if currently IR / Out / LTIR
    baseline_toi_secs  Int64    expected TOI per game when healthy

Goalies (position starts with "G") are excluded from the strain math —
goalie minute redistribution is governed by goalie_fatigue (2.6), not by
skater depth.

``as_of_date`` (str, ``"YYYY-MM-DD"``) — snapshot date, written through.

Outputs
-------
One row per *skater* with ROSTER_DEPTH_STRAIN_SCHEMA::

    player_id              — Int64
    as_of_date             — Utf8
    team                   — Utf8
    is_on_ir               — Boolean
    ir_skater_count        — Int64    number of teammates on IR
    healthy_skater_count   — Int64    number of healthy teammates incl. self
    ir_minutes_secs        — Float64  total IR baseline minutes (secs)
    extra_per_healthy_secs — Float64  redistributed seconds per healthy skater
    team_strain_score      — Float64  team-level strain in [0, 1]
    player_strain_score    — Float64  0.0 if on IR, team_strain_score otherwise

Calibration knobs (constructor):
- ``full_strain_secs`` default 180 — three full minutes of extra TOI per
  healthy skater = a fully stretched team (≈ 5+ top-six absences).
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "roster_depth_strain_v1"

ROSTER_REQUIRED_COLS = (
    "player_id",
    "team",
    "position",
    "is_on_ir",
    "baseline_toi_secs",
)

ROSTER_DEPTH_STRAIN_SCHEMA: dict[str, pl.DataType] = {
    "player_id":              pl.Int64,
    "as_of_date":             pl.Utf8,
    "team":                   pl.Utf8,
    "is_on_ir":               pl.Boolean,
    "ir_skater_count":        pl.Int64,
    "healthy_skater_count":   pl.Int64,
    "ir_minutes_secs":        pl.Float64,
    "extra_per_healthy_secs": pl.Float64,
    "team_strain_score":      pl.Float64,
    "player_strain_score":    pl.Float64,
}


FULL_STRAIN_SECS = 180.0   # 3 extra minutes per healthy skater = full strain


# ---------------------------------------------------------------------------
# Public formula
# ---------------------------------------------------------------------------

def strain_score(
    extra_per_healthy_secs: float,
    full_strain_secs: float = FULL_STRAIN_SECS,
) -> float:
    """Clamp ``extra_per_healthy_secs / full_strain_secs`` to ``[0, 1]``."""
    if not (full_strain_secs > 0):
        raise ValueError("full_strain_secs must be > 0")
    if extra_per_healthy_secs <= 0:
        return 0.0
    raw = float(extra_per_healthy_secs) / float(full_strain_secs)
    return min(1.0, max(0.0, raw))


def _parse_date(s: str) -> None:
    datetime.strptime(s, "%Y-%m-%d")


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in ROSTER_DEPTH_STRAIN_SCHEMA.items()}
    )


def _validate(roster_df: pl.DataFrame) -> bool:
    missing = [c for c in ROSTER_REQUIRED_COLS if c not in roster_df.columns]
    if missing:
        warnings.warn(
            f"roster_df missing columns: {missing}. Required: "
            f"{list(ROSTER_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _is_goalie(position: str | None) -> bool:
    if position is None:
        return False
    return str(position).upper().startswith("G")


# ---------------------------------------------------------------------------
# RosterDepthStrain
# ---------------------------------------------------------------------------

class RosterDepthStrain:
    """Per-team IR minute redistribution → per-player strain (Feature 3.16)."""

    def __init__(
        self,
        full_strain_secs: float = FULL_STRAIN_SECS,
    ) -> None:
        if full_strain_secs <= 0:
            raise ValueError("full_strain_secs must be > 0")
        self._full    = float(full_strain_secs)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(self, roster_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
        """Return one row per *skater* with the team-strain redistribution."""
        if not _validate(roster_df):
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(roster_df) == 0:
            return _empty_output()

        # First pass: collect per-team skater inventory.
        teams: dict[str, dict] = {}
        rows = roster_df.to_dicts()
        seen_players: set[tuple[str, int]] = set()
        skater_rows: list[dict] = []

        for r in rows:
            pid_raw = r.get("player_id")
            team    = r.get("team")
            pos     = r.get("position")
            if pid_raw is None or team is None:
                continue
            if _is_goalie(pos):
                continue
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                continue

            key = (str(team), pid)
            if key in seen_players:
                continue
            seen_players.add(key)

            try:
                base_toi = float(r.get("baseline_toi_secs") or 0.0)
            except (TypeError, ValueError):
                base_toi = 0.0
            if base_toi < 0.0:
                base_toi = 0.0

            on_ir = bool(r.get("is_on_ir"))

            bucket = teams.setdefault(str(team), {"ir_secs": 0.0, "ir": 0, "healthy": 0})
            if on_ir:
                bucket["ir_secs"] += base_toi
                bucket["ir"]      += 1
            else:
                bucket["healthy"] += 1

            skater_rows.append({
                "player_id":         pid,
                "team":              str(team),
                "is_on_ir":          on_ir,
                "baseline_toi_secs": base_toi,
            })

        if not skater_rows:
            return _empty_output()

        # Second pass: emit per-player rows with team-level strain attached.
        out_rows: list[dict] = []
        for s in skater_rows:
            bucket = teams[s["team"]]
            healthy_n = int(bucket["healthy"])
            ir_n      = int(bucket["ir"])
            ir_secs   = float(bucket["ir_secs"])

            if healthy_n > 0:
                extra = ir_secs / healthy_n
            else:
                # Pathological: every skater on IR → maximum strain.
                extra = ir_secs

            team_score   = strain_score(extra, self._full)
            player_score = 0.0 if s["is_on_ir"] else team_score

            out_rows.append({
                "player_id":              s["player_id"],
                "as_of_date":             as_of_date,
                "team":                   s["team"],
                "is_on_ir":               s["is_on_ir"],
                "ir_skater_count":        ir_n,
                "healthy_skater_count":   healthy_n,
                "ir_minutes_secs":        ir_secs,
                "extra_per_healthy_secs": float(extra),
                "team_strain_score":      float(team_score),
                "player_strain_score":    float(player_score),
            })

        return pl.DataFrame(out_rows, schema=ROSTER_DEPTH_STRAIN_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_roster_depth_strain(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write roster-depth-strain DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"roster_depth_strain_{as_of_date}.parquet"
    for col, dtype in ROSTER_DEPTH_STRAIN_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(ROSTER_DEPTH_STRAIN_SCHEMA.keys())).write_parquet(path)
    return path
