"""Trade Integration Model — Feature 3.20.

When a player is traded mid-season, their performance bends around a
new team / new system / new linemates / new home time-zone for the first
~15–20 games. This is a *separate* signal from the chip-on-shoulder
former-team boost (2.27). 2.27 fires only when the player meets the old
team; 3.20 fires every game until the player is "integrated".

Direction is bidirectional and driven by the *fit delta* between teams:

    fit_delta = new_team_fit_score  (4.12)
              − old_team_fit_score  (4.12)

    fit_delta > 0  → the new team is a *better* fit  → small boost
                     (FI should drop slightly)
    fit_delta < 0  → the new team is a *worse* fit  → drag
                     (FI should rise — player is adapting to a system
                     that does not suit their archetype)
    fit_delta ≈ 0  → neutral; only the travel/transition penalty applies

Because Phase 4 (the coaching style and roster-fit models that produce
those scores) is not yet built, the model accepts a ``fit_scores`` lookup
that downstream callers can leave empty during V1. When a fit score is
missing for either team, the model falls back to ``DEFAULT_FIT_SCORE``
(0.5), so ``fit_delta`` collapses to 0 and the output is travel-only.

Output range
------------
``integration_factor`` is an *additive* modifier on the composite FI
(3.17), conceptually living in roughly ``[−0.05, +0.10]``:

    +ve → drag        (raise FI: player struggling to integrate)
    −ve → boost       (lower FI: player slotted in cleanly)

The 90% credible interval is intentionally wide — every trade is a
unique data point, the underlying signal is noisy, and we'd rather show
Bob honest uncertainty than false precision.

Decay
-----
Effect decays linearly to zero over ``DECAY_GAMES_BY_POSITION``:

    LW / RW          15 games   (wingers integrate fastest)
    C                20 games   (face-off + system reads)
    D                20 games   (pair chemistry + DZ structure)
    G / unknown      18 games   (fallback)

Inputs
------
``trades_df`` (Polars) — one row per (player, trade) event::

    player_id        Int64
    player_name      Utf8    (optional)
    trade_date       Utf8    "YYYY-MM-DD"
    new_team         Utf8    3-letter
    old_team         Utf8    3-letter
    position         Utf8    (optional) C / L / R / D / G / LW / RW

``observations_df`` (Polars) — one row per (player_id, game_date) the
caller wants a modifier for. ``game_id`` is passed through when present::

    player_id    Int64
    game_id      Int64       (optional)
    game_date    Utf8        "YYYY-MM-DD"

``fit_scores`` (dict) — optional lookup of ``(player_id, team) → score``
where ``score ∈ [0, 1]`` is the 4.12 roster-fit score. Missing entries
default to ``DEFAULT_FIT_SCORE``.

Outputs
-------
``TRADE_INTEGRATION_SCHEMA`` — one row per (player, game_date) with the
modifier and metadata::

    player_id              Int64
    game_id                Int64
    game_date              Utf8
    as_of_date             Utf8
    trade_date             Utf8
    new_team               Utf8
    old_team               Utf8
    position               Utf8
    games_since_trade      Int64
    decay_factor           Float64    in [0, 1]
    fit_delta              Float64    in [-1, +1]
    travel_penalty         Float64    in [0, TRAVEL_PENALTY_MAX]
    integration_factor     Float64    additive FI modifier
    ci_low                 Float64    integration_factor − ci_half
    ci_high                Float64    integration_factor + ci_half
    model_version          Utf8
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "trade_integration_v1"

TRADE_REQUIRED_COLS       = ("player_id", "trade_date", "new_team", "old_team")
OBSERVATION_REQUIRED_COLS = ("player_id", "game_date")

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Max absolute |fit_delta| effect, fully un-decayed. fit_delta ∈ [-1, +1]
# means the bidirectional modifier swings within [-FIT_DELTA_MAX, +FIT_DELTA_MAX].
FIT_DELTA_MAX           = 0.10

# Travel / transition penalty: present for the first TRAVEL_PENALTY_GAMES
# games regardless of fit_delta. Decays linearly to 0 across that window.
TRAVEL_PENALTY_MAX      = 0.02
TRAVEL_PENALTY_GAMES    = 5

# Wide CI by design — single-event signal, sparse data, lots of unknowns.
CI_HALF_WIDTH           = 0.04

# Default fit score when 4.12 has not produced a value yet.
DEFAULT_FIT_SCORE       = 0.5

# Per-position decay window (games until the effect reaches zero).
DECAY_GAMES_BY_POSITION: dict[str, int] = {
    "LW": 15,
    "RW": 15,
    "L":  15,
    "R":  15,
    "C":  20,
    "D":  20,
    "G":  18,
}
DEFAULT_DECAY_GAMES     = 18


TRADE_INTEGRATION_SCHEMA: dict[str, pl.DataType] = {
    "player_id":           pl.Int64,
    "game_id":             pl.Int64,
    "game_date":           pl.Utf8,
    "as_of_date":          pl.Utf8,
    "trade_date":          pl.Utf8,
    "new_team":            pl.Utf8,
    "old_team":            pl.Utf8,
    "position":            pl.Utf8,
    "games_since_trade":   pl.Int64,
    "decay_factor":        pl.Float64,
    "fit_delta":           pl.Float64,
    "travel_penalty":      pl.Float64,
    "integration_factor":  pl.Float64,
    "ci_low":              pl.Float64,
    "ci_high":             pl.Float64,
    "model_version":       pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _days_between(a: str, b: str) -> int:
    """Calendar days from ``a`` → ``b``. Returns negative when ``b < a``."""
    da = _parse_date(a)
    db = _parse_date(b)
    return (db - da).days


def _normalize_position(pos: str | None) -> str:
    if not pos:
        return ""
    p = str(pos).strip().upper()
    # NHL API returns bare "L" / "R" for wings; treat as LW / RW.
    if p == "L":
        return "LW"
    if p == "R":
        return "RW"
    return p


def _decay_games(position: str | None) -> int:
    return DECAY_GAMES_BY_POSITION.get(_normalize_position(position),
                                       DEFAULT_DECAY_GAMES)


def _decay_factor(games_since: int, decay_games: int) -> float:
    if games_since < 0:
        return 0.0
    if decay_games <= 0:
        return 0.0
    if games_since >= decay_games:
        return 0.0
    return 1.0 - games_since / decay_games


def _travel_penalty(games_since: int) -> float:
    if games_since < 0:
        return 0.0
    if games_since >= TRAVEL_PENALTY_GAMES:
        return 0.0
    remaining = (TRAVEL_PENALTY_GAMES - games_since) / TRAVEL_PENALTY_GAMES
    return TRAVEL_PENALTY_MAX * remaining


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(lo, min(hi, value))


def _fit_delta(
    fit_scores: dict[tuple[int, str], float] | None,
    player_id:  int,
    new_team:   str,
    old_team:   str,
) -> float:
    if not fit_scores:
        return 0.0
    new_score = fit_scores.get((player_id, new_team), DEFAULT_FIT_SCORE)
    old_score = fit_scores.get((player_id, old_team), DEFAULT_FIT_SCORE)
    delta = float(new_score) - float(old_score)
    return _clamp(delta, -1.0, 1.0)


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in TRADE_INTEGRATION_SCHEMA.items()}
    )


def _validate_trades(trades_df: pl.DataFrame) -> bool:
    missing = [c for c in TRADE_REQUIRED_COLS if c not in trades_df.columns]
    if missing:
        warnings.warn(
            f"trades_df missing columns: {missing}. Required: "
            f"{list(TRADE_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


def _validate_observations(obs_df: pl.DataFrame) -> bool:
    missing = [c for c in OBSERVATION_REQUIRED_COLS if c not in obs_df.columns]
    if missing:
        warnings.warn(
            f"observations_df missing columns: {missing}. Required: "
            f"{list(OBSERVATION_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Trade integration record
# ---------------------------------------------------------------------------

class _TradeRecord:
    """One trade event in a player's history."""
    __slots__ = ("trade_date", "new_team", "old_team", "position")

    def __init__(self, trade_date: str, new_team: str,
                 old_team: str, position: str) -> None:
        self.trade_date = trade_date
        self.new_team   = new_team
        self.old_team   = old_team
        self.position   = position


# ---------------------------------------------------------------------------
# TradeIntegrationModel
# ---------------------------------------------------------------------------

class TradeIntegrationModel:
    """Per-(player, game) trade integration modifier (Feature 3.20)."""

    def __init__(
        self,
        fit_delta_max:       float = FIT_DELTA_MAX,
        travel_penalty_max:  float = TRAVEL_PENALTY_MAX,
        travel_penalty_games: int  = TRAVEL_PENALTY_GAMES,
        ci_half_width:       float = CI_HALF_WIDTH,
    ) -> None:
        if fit_delta_max < 0:
            raise ValueError("fit_delta_max must be >= 0")
        if travel_penalty_max < 0:
            raise ValueError("travel_penalty_max must be >= 0")
        if travel_penalty_games < 0:
            raise ValueError("travel_penalty_games must be >= 0")
        if ci_half_width < 0:
            raise ValueError("ci_half_width must be >= 0")

        self._fit_delta_max  = float(fit_delta_max)
        self._travel_max     = float(travel_penalty_max)
        self._travel_games   = int(travel_penalty_games)
        self._ci_half        = float(ci_half_width)
        self._version        = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute_row(
        self,
        trade: _TradeRecord,
        games_since: int,
        fit_delta: float,
    ) -> dict:
        decay_games = _decay_games(trade.position)
        decay = _decay_factor(games_since, decay_games)

        # Bidirectional fit term: positive fit_delta → boost (negative).
        fit_term = -self._fit_delta_max * fit_delta

        travel = (
            self._travel_max
            * max(0.0, (self._travel_games - games_since)) / self._travel_games
            if self._travel_games > 0 else 0.0
        )
        travel = max(0.0, travel)

        raw = (fit_term + travel) * decay
        return {
            "games_since_trade":   int(games_since),
            "decay_factor":        float(decay),
            "fit_delta":           float(fit_delta),
            "travel_penalty":      float(travel),
            "integration_factor":  float(raw),
            "ci_low":              float(raw - self._ci_half),
            "ci_high":             float(raw + self._ci_half),
        }

    # ------------------------------------------------------------------
    def compute(
        self,
        trades_df:        pl.DataFrame,
        observations_df:  pl.DataFrame,
        as_of_date:       str,
        fit_scores:       dict[tuple[int, str], float] | None = None,
        games_played_lookup: dict[tuple[int, str], int] | None = None,
    ) -> pl.DataFrame:
        """Return one row per (player_id, game_date) observation.

        Parameters
        ----------
        games_played_lookup:
            Optional ``{(player_id, "YYYY-MM-DD"): cumulative_games_played}``
            mapping. When present, ``games_since_trade`` is computed as
            ``GP_at(game_date) − GP_at(trade_date)`` for the player. When
            absent, the model falls back to a 2-games-per-week elapsed-time
            proxy (calendar days since trade × ``GAMES_PER_DAY``).
        """
        if not _validate_trades(trades_df):
            return _empty_output()
        if not _validate_observations(observations_df):
            return _empty_output()

        try:
            _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(trades_df) == 0 or len(observations_df) == 0:
            return _empty_output()

        # Build per-player list of trades sorted by date.
        trades_by_player: dict[int, list[_TradeRecord]] = {}
        for r in trades_df.to_dicts():
            pid_raw = r.get("player_id")
            td_raw  = r.get("trade_date")
            nt_raw  = r.get("new_team")
            ot_raw  = r.get("old_team")
            if pid_raw is None or td_raw is None or nt_raw is None or ot_raw is None:
                continue
            try:
                pid = int(pid_raw)
                _parse_date(str(td_raw))
            except (TypeError, ValueError):
                continue
            rec = _TradeRecord(
                trade_date = str(td_raw),
                new_team   = str(nt_raw).strip().upper(),
                old_team   = str(ot_raw).strip().upper(),
                position   = _normalize_position(r.get("position")),
            )
            trades_by_player.setdefault(pid, []).append(rec)

        for lst in trades_by_player.values():
            lst.sort(key=lambda x: x.trade_date)

        out_rows: list[dict] = []
        for o in observations_df.to_dicts():
            pid_raw = o.get("player_id")
            gd_raw  = o.get("game_date")
            if pid_raw is None or gd_raw is None:
                continue
            try:
                pid = int(pid_raw)
                _parse_date(str(gd_raw))
            except (TypeError, ValueError):
                continue
            game_date = str(gd_raw)

            game_id = o.get("game_id")
            try:
                gid = int(game_id) if game_id is not None else 0
            except (TypeError, ValueError):
                gid = 0

            history = trades_by_player.get(pid, [])
            # Most recent trade on or before this game date.
            active: _TradeRecord | None = None
            for rec in reversed(history):
                if rec.trade_date <= game_date:
                    active = rec
                    break
            if active is None:
                continue   # no relevant trade for this player

            games_since = _games_since(
                pid, active.trade_date, game_date, games_played_lookup
            )
            decay_games = _decay_games(active.position)
            if games_since >= decay_games:
                continue   # fully integrated

            fit_delta = _fit_delta(fit_scores, pid, active.new_team, active.old_team)
            metrics = self.compute_row(active, games_since, fit_delta)

            out_rows.append({
                "player_id":          pid,
                "game_id":            gid,
                "game_date":          game_date,
                "as_of_date":         as_of_date,
                "trade_date":         active.trade_date,
                "new_team":           active.new_team,
                "old_team":           active.old_team,
                "position":           active.position,
                "model_version":      self._version,
                **metrics,
            })

        if not out_rows:
            return _empty_output()
        return pl.DataFrame(out_rows, schema=TRADE_INTEGRATION_SCHEMA)


# ---------------------------------------------------------------------------
# Per-day games-played proxy (when caller has no GP lookup)
# ---------------------------------------------------------------------------

# An NHL team plays ~3.5 games every 7 days, but a *player* may sit a
# game. A 2-games-per-week proxy underestimates slightly, which keeps
# us conservative (effect lingers a hair longer than maybe true) — that
# is the safer default for a modifier this uncertain.
GAMES_PER_DAY = 2.0 / 7.0


def _games_since(
    player_id:  int,
    trade_date: str,
    game_date:  str,
    gp_lookup:  dict[tuple[int, str], int] | None,
) -> int:
    if gp_lookup:
        gp_now  = gp_lookup.get((player_id, game_date))
        gp_then = gp_lookup.get((player_id, trade_date))
        if gp_now is not None and gp_then is not None:
            return max(0, int(gp_now) - int(gp_then))
    days = _days_between(trade_date, game_date)
    if days < 0:
        return -1
    return max(0, int(round(days * GAMES_PER_DAY)))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_trade_integration(
    df: pl.DataFrame, output_dir: Path, as_of_date: str,
) -> Path:
    """Write trade-integration DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"trade_integration_{as_of_date}.parquet"
    for col, dtype in TRADE_INTEGRATION_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(TRADE_INTEGRATION_SCHEMA.keys())).write_parquet(path)
    return path
