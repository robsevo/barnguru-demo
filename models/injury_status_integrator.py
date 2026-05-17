"""Injury status integrator — Feature 3.13.

Per-player current availability snapshot. Takes a *history* of injury-feed
observations (one row per (player, observed_date) the player appeared on
the report) plus the player's actual games played, and resolves three
things as of a single ``as_of_date``:

1. **Current status** — ``HEALTHY`` / ``DTD`` / ``OUT``, taken from the most
   recent injury observation on or before ``as_of_date``. A player who has
   never appeared on the feed is implicitly ``HEALTHY`` (ESPN only lists
   non-healthy players).

2. **Availability probability** — heuristic mapping from status to
   ``P(plays tonight)``. This is a *static* table here — the calibrated
   logistic model is Feature 1.13; this module is the input layer to the
   composite FI (3.17), not a replacement for 1.13.

3. **Return-from-injury rust factor** — when a player was ``OUT`` and has
   since become available, count the games they have played on or after
   the return-transition date. Games 1–``RUST_RECOVERY_GAMES`` post-return
   get a linear ramp from ``RUST_FLOOR`` (game 1) back to 1.0 (game
   ``RUST_RECOVERY_GAMES``). After that the rust is gone.

Inputs
------
``injury_history_df`` (Polars) — long-form observation log, one row per
(player_id, observed_date)::

    player_id      Int64
    observed_date  Utf8       ("YYYY-MM-DD") — date of the feed snapshot
    status         Utf8       "Healthy" | "DTD" | "Out"  (InjuryStatus.value)

Construction tip: collect each daily injuries parquet, dedupe per (player,
date), and stack. Players known to be healthy on a given day need not be
included — their absence is treated as healthy on dates between two known
status rows surrounding ``as_of_date``.

``player_games_df`` (Polars) — one row per game the player has actually
played::

    player_id   Int64
    game_date   Utf8       ("YYYY-MM-DD")

Used only to count games played since the last OUT→non-OUT transition.
Empty is fine; a player who has not yet played any games since returning
has ``games_since_return = 0`` (return game has not happened yet).

``as_of_date`` (str, ``"YYYY-MM-DD"``) — the snapshot date.

Outputs
-------
One row per player with INJURY_STATUS_SCHEMA::

    player_id                — Int64
    as_of_date               — Utf8
    status                   — Utf8     "Healthy" | "DTD" | "Out"
    availability_probability — Float64  in [0.0, 1.0]
    return_date              — Utf8     date of last OUT→non-OUT transition;
                                        empty string if no such transition
    games_since_return       — Int64    games on/after return_date and
                                        on/before as_of_date; ``-1`` when
                                        no return transition exists
    rust_factor              — Float64  in [RUST_FLOOR, 1.0]; ``0.0`` if
                                        currently OUT

Calibration knobs (constructor):
- ``availability_table``   default ``{Healthy: 1.0, DTD: 0.50, Out: 0.0}``
- ``rust_recovery_games``  default 7   — games to full recovery
- ``rust_floor``           default 0.85 — performance at the return game
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "injury_status_integrator_v1"

STATUS_HEALTHY = "Healthy"
STATUS_DTD     = "DTD"
STATUS_OUT     = "Out"
STATUS_VALID   = frozenset({STATUS_HEALTHY, STATUS_DTD, STATUS_OUT})

HISTORY_REQUIRED_COLS = ("player_id", "observed_date", "status")
GAMES_REQUIRED_COLS   = ("player_id", "game_date")

INJURY_STATUS_SCHEMA: dict[str, pl.DataType] = {
    "player_id":                pl.Int64,
    "as_of_date":               pl.Utf8,
    "status":                   pl.Utf8,
    "availability_probability": pl.Float64,
    "return_date":              pl.Utf8,
    "games_since_return":       pl.Int64,
    "rust_factor":              pl.Float64,
}


DEFAULT_AVAILABILITY: dict[str, float] = {
    STATUS_HEALTHY: 1.0,
    STATUS_DTD:     0.50,
    STATUS_OUT:     0.0,
}

RUST_RECOVERY_GAMES = 7      # games to full recovery after a return
RUST_FLOOR          = 0.85   # performance multiplier at the very first game back
NO_RETURN_SENTINEL  = -1     # games_since_return when no return transition exists


# ---------------------------------------------------------------------------
# Public formulas
# ---------------------------------------------------------------------------

def rust_factor(
    games_since_return: int,
    recovery_games: int = RUST_RECOVERY_GAMES,
    floor:          float = RUST_FLOOR,
) -> float:
    """Linear ramp from ``floor`` at game 1 to 1.0 at ``recovery_games``.

    - ``games_since_return <= 0``  → 1.0 (no recent return = no rust)
    - ``games_since_return == 1``  → ``floor``
    - ``games_since_return >= recovery_games`` → 1.0
    """
    g = int(games_since_return)
    if g <= 0:
        return 1.0
    if g >= recovery_games:
        return 1.0
    span = max(1, recovery_games - 1)
    return float(floor) + (1.0 - float(floor)) * (g - 1) / span


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in INJURY_STATUS_SCHEMA.items()}
    )


def _validate_history(df: pl.DataFrame) -> bool:
    missing = [c for c in HISTORY_REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(
            f"injury_history_df missing columns: {missing}. Required: "
            f"{list(HISTORY_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _validate_games(df: pl.DataFrame) -> bool:
    missing = [c for c in GAMES_REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(
            f"player_games_df missing columns: {missing}. Required: "
            f"{list(GAMES_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# InjuryStatusIntegrator
# ---------------------------------------------------------------------------

class InjuryStatusIntegrator:
    """Injury status + return-from-injury rust factor (Feature 3.13)."""

    def __init__(
        self,
        availability_table: dict[str, float] | None = None,
        rust_recovery_games: int = RUST_RECOVERY_GAMES,
        rust_floor: float = RUST_FLOOR,
    ) -> None:
        if rust_recovery_games < 1:
            raise ValueError("rust_recovery_games must be >= 1")
        if not (0.0 <= rust_floor <= 1.0):
            raise ValueError("rust_floor must be in [0, 1]")

        table = dict(DEFAULT_AVAILABILITY) if availability_table is None else dict(availability_table)
        for s in STATUS_VALID:
            if s not in table:
                table[s] = DEFAULT_AVAILABILITY[s]
            v = table[s]
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"availability_table[{s}] must be in [0, 1]")
            table[s] = float(v)

        self._avail   = table
        self._recover = int(rust_recovery_games)
        self._floor   = float(rust_floor)
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(
        self,
        injury_history_df: pl.DataFrame,
        player_games_df:   pl.DataFrame,
        as_of_date:        str,
    ) -> pl.DataFrame:
        """Resolve per-player status, availability, and rust factor at
        ``as_of_date``.

        A player appearing in either input frame contributes a row to the
        output. A player who has never been on the injury report is
        treated as ``HEALTHY`` with ``games_since_return = -1``.
        """
        if not _validate_history(injury_history_df):
            return _empty_output()
        if len(player_games_df) > 0 and not _validate_games(player_games_df):
            return _empty_output()

        try:
            as_of = _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        # Constrain observations to on/before as_of, valid statuses only.
        history_rows = injury_history_df.to_dicts()
        per_player_history: dict[int, list[tuple[date, str]]] = {}
        for r in history_rows:
            pid_raw = r.get("player_id")
            obs_raw = r.get("observed_date")
            status  = r.get("status")
            if pid_raw is None or obs_raw is None or status is None:
                continue
            if status not in STATUS_VALID:
                warnings.warn(
                    f"Unknown status {status!r} for player {pid_raw}; skipping row.",
                    DataMissingWarning,
                    stacklevel=2,
                )
                continue
            try:
                d = _parse_date(str(obs_raw))
            except (TypeError, ValueError):
                continue
            if d > as_of:
                continue
            per_player_history.setdefault(int(pid_raw), []).append((d, str(status)))

        per_player_games: dict[int, list[date]] = {}
        if len(player_games_df) > 0:
            for r in player_games_df.to_dicts():
                pid_raw = r.get("player_id")
                gd_raw  = r.get("game_date")
                if pid_raw is None or gd_raw is None:
                    continue
                try:
                    gd = _parse_date(str(gd_raw))
                except (TypeError, ValueError):
                    continue
                if gd > as_of:
                    continue
                per_player_games.setdefault(int(pid_raw), []).append(gd)

        all_players = set(per_player_history) | set(per_player_games)
        if not all_players:
            return _empty_output()

        out_rows: list[dict] = []
        for pid in sorted(all_players):
            history = sorted(per_player_history.get(pid, []), key=lambda t: t[0])
            games   = sorted(per_player_games.get(pid, []))

            # Current status = most recent observation; default HEALTHY.
            status = history[-1][1] if history else STATUS_HEALTHY

            # Last OUT→non-OUT transition prior to/at as_of_date.
            return_dt: date | None = None
            prev_status: str | None = None
            for obs_date, obs_status in history:
                if prev_status == STATUS_OUT and obs_status != STATUS_OUT:
                    return_dt = obs_date
                prev_status = obs_status

            # Games since return
            if return_dt is None:
                games_since = NO_RETURN_SENTINEL
            else:
                games_since = sum(1 for gd in games if gd >= return_dt)

            # Rust factor: 0.0 if currently OUT, else ramp by games_since
            if status == STATUS_OUT:
                rust = 0.0
            elif games_since <= 0:
                rust = 1.0
            else:
                rust = rust_factor(games_since, self._recover, self._floor)

            out_rows.append({
                "player_id":                int(pid),
                "as_of_date":               as_of_date,
                "status":                   status,
                "availability_probability": float(self._avail[status]),
                "return_date":              return_dt.isoformat() if return_dt else "",
                "games_since_return":       int(games_since),
                "rust_factor":              float(rust),
            })

        return pl.DataFrame(out_rows, schema=INJURY_STATUS_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_injury_status(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write injury-status DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"injury_status_{as_of_date}.parquet"
    for col, dtype in INJURY_STATUS_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(INJURY_STATUS_SCHEMA.keys())).write_parquet(path)
    return path
