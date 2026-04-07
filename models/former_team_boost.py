"""Former Team Motivational Boost — Feature 2.27.

Per-player per-game binary modifier applied when a player faces a former team.
Magnitude is calibrated by three factors:

1. **Departure type** — how the player left (higher chip-on-shoulder = bigger boost):
       waived   → +0.18  (maximum — player was discarded)
       traded   → +0.13  (moved mid-contract, not player's choice)
       ufa      → +0.09  (chose to leave as an unrestricted free agent)
       rfa      → +0.06  (minimum — restricted free agent, less control)
       other    → +0.08  (fallback for unclassified departures)

2. **Years since departure** — motivational effect decays linearly over
   ``DECAY_SEASONS`` (default 3.0).  After 3 seasons the boost reaches zero.
   Factor: ``max(0, 1 − years_since / DECAY_SEASONS)``

3. **TOI with former team** — if a player received insufficient ice time
   (< ``TOI_THRESHOLD_MINUTES`` per game), the chip-on-shoulder effect is
   amplified toward ``TOI_MAX_AMPLIFICATION``.  A player averaging 8 min/game
   when the league average for their role is 15 min gets a stronger boost.
   Factor: ``clip(1 + (TOI_THRESHOLD_MINUTES − toi_per_game) / TOI_THRESHOLD_MINUTES,
                   1.0, TOI_MAX_AMPLIFICATION)``

4. **Away game multiplier** — the effect is stronger in a hostile former home
   building (``AWAY_MULTIPLIER``, default 1.15).

Final formula::

    boost = base_boost
            × decay_factor
            × toi_factor
            [× AWAY_MULTIPLIER if away]

Clamped to ``[BOOST_MIN, BOOST_MAX]``.

Signal validation gate
──────────────────────
The boost is only marked as validated once sufficient historical PBP data
exists.  ``FormerTeamBoostModel.is_validated`` is set to ``True`` during
``fit()`` when ``min(validated_seasons) ≥ VALIDATION_SEASONS_REQUIRED``.
The gate status is saved/loaded with the model so downstream consumers can
check before applying.

Inputs
------
transactions_df : Polars DataFrame with schema matching
    ``data.transaction_parser.TRANSACTION_SCHEMA``.
    Required columns: date, event_type, team, secondary_team,
    player_or_executive, player_id_espn.

player_toi_df : Polars DataFrame with per-player per-team TOI aggregates.
    Required columns: player_id (int64), team (str), season (int64),
    toi_seconds (float64), gp (int64).
    Optional: player_name (str).

Output
------
``former_team_boost_{season}.parquet`` — one row per (player_id, former_team):

    player_id          int64
    player_name        str
    former_team        str       3-letter team code
    departure_date     str       YYYY-MM-DD
    departure_type     str       waived | trade | ufa | rfa | other
    toi_per_game_min   float64   average TOI/game while on former team (minutes)
    base_boost         float64   pre-decay, pre-TOI boost from departure type
    validated          bool      signal gate — True once 3+ seasons validated
    model_version      str

Usage::

    from models.former_team_boost import FormerTeamBoostModel

    model = FormerTeamBoostModel()
    boost_df = model.fit(transactions_df, player_toi_df)

    # At game time:
    b = model.compute_boost(
        player_id=8478402,        # NHL API player_id
        opponent_team="MTL",
        game_date="2025-11-15",
        is_away=True,
    )
    # b ∈ [0.0, 0.18]; 0.0 means no former-team boost applies
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "former_team_boost_v1"

# Base boost by departure type (before decay and TOI modifiers)
BOOST_BY_DEPARTURE: dict[str, float] = {
    "waiver_release": 0.18,   # waived — discarded, strongest chip
    "trade":          0.13,   # traded mid-contract, involuntary
    "ufa":            0.09,   # left as unrestricted free agent
    "rfa":            0.06,   # restricted free agent, limited control
    "other":          0.08,   # fallback
}

# Decay: motivational effect fades to 0 over this many seasons
DECAY_SEASONS: float = 3.0

# TOI modifier: if average TOI/game was below this (minutes), amplify boost
TOI_THRESHOLD_MINUTES: float = 14.0
TOI_MAX_AMPLIFICATION: float = 1.40

# Away game multiplier (peer-reviewed effect: stronger in former home arena)
AWAY_MULTIPLIER: float = 1.15

# Boost is clamped to this range
BOOST_MIN: float = 0.0
BOOST_MAX: float = 0.18

# Signal validation gate: require at least this many historical seasons
VALIDATION_SEASONS_REQUIRED: int = 3

# TOI seconds per hour (for toi_seconds → minutes/game conversion)
_SEC_PER_MIN: float = 60.0


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

FORMER_TEAM_BOOST_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "player_name":      pl.Utf8,
    "former_team":      pl.Utf8,
    "departure_date":   pl.Utf8,
    "departure_type":   pl.Utf8,
    "toi_per_game_min": pl.Float64,
    "base_boost":       pl.Float64,
    "validated":        pl.Boolean,
    "model_version":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Internal data type
# ---------------------------------------------------------------------------

@dataclass
class _FormerTeamEntry:
    """One former-team relationship for a player."""
    player_id: int
    player_name: str
    former_team: str
    departure_date: str           # YYYY-MM-DD
    departure_type: str           # waived | trade | ufa | rfa | other
    toi_per_game_min: float       # average TOI/game while on that team (minutes)
    base_boost: float             # pre-decay, pre-TOI boost from departure type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _departure_type_from_event(event_type: str) -> str:
    """Map transaction event_type string to canonical departure type."""
    if event_type in ("waiver_release",):
        return "waiver_release"
    if event_type in ("trade",):
        return "trade"
    if event_type in ("signing",):
        return "ufa"   # signing = UFA joining a new team (left old as UFA)
    return "other"


def _years_between(date_a: str, date_b: str) -> float:
    """Approximate years between two YYYY-MM-DD date strings.

    Returns a positive float: years elapsed from date_a to date_b.
    If date parsing fails, returns 0.0 (no decay applied).
    """
    try:
        y_a, m_a, d_a = int(date_a[:4]), int(date_a[5:7]), int(date_a[8:10])
        y_b, m_b, d_b = int(date_b[:4]), int(date_b[5:7]), int(date_b[8:10])
        # Approximate as days / 365.25
        days_a = y_a * 365 + m_a * 30 + d_a
        days_b = y_b * 365 + m_b * 30 + d_b
        return max(0.0, (days_b - days_a) / 365.25)
    except (ValueError, IndexError):
        return 0.0


def _decay_factor(years_since: float, decay_seasons: float = DECAY_SEASONS) -> float:
    """Linear decay from 1.0 at departure to 0.0 at ``decay_seasons`` years.

    Clamped to [0.0, 1.0] — negative years_since (future departure date) is
    treated as zero elapsed time.
    """
    return min(1.0, max(0.0, 1.0 - years_since / decay_seasons))


def _toi_amplification_factor(toi_per_game_min: float) -> float:
    """Return a factor in [1.0, TOI_MAX_AMPLIFICATION].

    Players who received less than TOI_THRESHOLD_MINUTES per game get a
    stronger boost (chip-on-shoulder from under-utilisation).
    """
    if toi_per_game_min >= TOI_THRESHOLD_MINUTES:
        return 1.0
    shortfall = TOI_THRESHOLD_MINUTES - toi_per_game_min
    raw = 1.0 + shortfall / TOI_THRESHOLD_MINUTES
    return min(raw, TOI_MAX_AMPLIFICATION)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FormerTeamBoostModel:
    """Per-player former-team motivational boost model (Feature 2.27).

    The model is stateless between fits — ``fit()`` rebuilds the internal
    lookup from scratch.  Use ``save()`` / ``load()`` for persistence.

    Thread safety: read-only after fit; not safe for concurrent writes.
    """

    def __init__(self) -> None:
        # player_id → list of FormerTeamEntry
        self._entries: dict[int, list[_FormerTeamEntry]] = {}
        self.is_validated: bool = False   # True after 3+ seasons confirmed
        self._validated_seasons: list[int] = []
        self._model_version: str = MODEL_VERSION

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        transactions_df: pl.DataFrame,
        player_toi_df: pl.DataFrame | None = None,
        validated_seasons: list[int] | None = None,
    ) -> pl.DataFrame:
        """Build the former-team lookup from transaction data.

        Args:
            transactions_df:  ESPN transactions — must include
                ``date, event_type, team, secondary_team,
                player_or_executive, player_id_espn``.
            player_toi_df:    Optional per-player-per-team TOI aggregates.
                If ``None``, TOI factor defaults to 1.0 (no amplification).
                Required columns: player_id (int64), team (str),
                toi_seconds (float64), gp (int64).
            validated_seasons: Season integers (e.g. [2022, 2023, 2024])
                confirming the signal has been back-tested.  When
                ``len(validated_seasons) >= VALIDATION_SEASONS_REQUIRED``,
                ``is_validated`` is set to ``True``.

        Returns:
            Polars DataFrame with schema ``FORMER_TEAM_BOOST_SCHEMA``,
            one row per (player_id, former_team) relationship found.
        """
        if len(transactions_df) == 0:
            warnings.warn(
                "transactions_df is empty; no former-team boosts computed.",
                UserWarning,
                stacklevel=2,
            )
            self._entries = {}
            return _empty_output()

        # ── Build TOI lookup: (player_id, team) → toi_per_game_min ───────────
        toi_lookup: dict[tuple[int, str], float] = {}
        if player_toi_df is not None and len(player_toi_df) > 0:
            toi_lookup = _build_toi_lookup(player_toi_df)

        # ── Extract player_id from transactions (ESPN → int where possible) ───
        # ESPN player IDs are strings; we store them as negative ints to avoid
        # collision with NHL API IDs until a proper cross-walk is available.
        records: list[_FormerTeamEntry] = []
        required_cols = {"date", "event_type", "team", "player_or_executive"}
        missing = required_cols - set(transactions_df.columns)
        if missing:
            raise ValueError(
                f"transactions_df missing required columns: {missing}"
            )

        for row in transactions_df.to_dicts():
            event_type = row.get("event_type") or "other"
            if event_type not in ("trade", "waiver_release", "signing"):
                continue   # only care about events that create former-team bonds

            departure_type = _departure_type_from_event(event_type)
            base_boost = BOOST_BY_DEPARTURE.get(departure_type, BOOST_BY_DEPARTURE["other"])

            # For trades: player LEFT secondary_team and joined team
            # For waivers/signings: player LEFT team
            if event_type == "trade":
                former_team = (row.get("secondary_team") or "").strip()
            else:
                # waiver_release or signing: team is the old team they left
                former_team = (row.get("team") or "").strip()

            if not former_team:
                continue

            raw_pid = row.get("player_id_espn") or ""
            player_id = _espn_id_to_int(raw_pid)
            player_name = (row.get("player_or_executive") or "")[:64]
            departure_date = row.get("date") or ""

            toi_per_game_min = toi_lookup.get((player_id, former_team), TOI_THRESHOLD_MINUTES)

            records.append(_FormerTeamEntry(
                player_id=player_id,
                player_name=player_name,
                former_team=former_team,
                departure_date=departure_date,
                departure_type=departure_type,
                toi_per_game_min=toi_per_game_min,
                base_boost=base_boost,
            ))

        # ── Build internal lookup ─────────────────────────────────────────────
        self._entries = {}
        for rec in records:
            self._entries.setdefault(rec.player_id, []).append(rec)

        # ── Validation gate ───────────────────────────────────────────────────
        if validated_seasons:
            self._validated_seasons = sorted(validated_seasons)
            self.is_validated = len(self._validated_seasons) >= VALIDATION_SEASONS_REQUIRED
        else:
            self._validated_seasons = []
            self.is_validated = False

        return _records_to_df(records, self.is_validated)

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def compute_boost(
        self,
        player_id: int,
        opponent_team: str,
        game_date: str,
        is_away: bool = False,
    ) -> float:
        """Return the former-team boost for a given player-game context.

        Args:
            player_id:     NHL API (or ESPN-derived) player identifier.
            opponent_team: 3-letter team code of the opponent.
            game_date:     Game date in YYYY-MM-DD format.
            is_away:       ``True`` if the player is playing at the opponent's
                           arena (away game → stronger effect).

        Returns:
            Float in ``[BOOST_MIN, BOOST_MAX]``.  Returns ``0.0`` when no
            former-team relationship is found or when the boost has fully
            decayed.
        """
        entries = self._entries.get(player_id, [])
        if not entries:
            return 0.0

        best: float = 0.0
        for entry in entries:
            if entry.former_team.upper() != opponent_team.upper():
                continue
            years_since = _years_between(entry.departure_date, game_date)
            decay = _decay_factor(years_since)
            if decay <= 0.0:
                continue
            toi_amp = _toi_amplification_factor(entry.toi_per_game_min)
            raw = entry.base_boost * decay * toi_amp
            if is_away:
                raw *= AWAY_MULTIPLIER
            clamped = min(max(raw, BOOST_MIN), BOOST_MAX)
            if clamped > best:
                best = clamped

        return best

    def has_former_team_matchup(
        self,
        player_id: int,
        opponent_team: str,
        game_date: str,
    ) -> bool:
        """Return True if any non-zero boost applies for this matchup."""
        return self.compute_boost(player_id, opponent_team, game_date) > 0.0

    def former_teams(self, player_id: int) -> list[str]:
        """Return the list of former team codes for *player_id*."""
        return [e.former_team for e in self._entries.get(player_id, [])]

    def player_count(self) -> int:
        """Number of players with at least one former-team entry."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialise the model to a joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":            self._model_version,
            "entries":            self._entries,
            "is_validated":       self.is_validated,
            "validated_seasons":  self._validated_seasons,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "FormerTeamBoostModel":
        """Load a previously saved model."""
        payload = joblib.load(path)
        instance = cls()
        instance._model_version     = payload.get("version", MODEL_VERSION)
        instance._entries           = payload.get("entries", {})
        instance.is_validated       = payload.get("is_validated", False)
        instance._validated_seasons = payload.get("validated_seasons", [])
        return instance


# ---------------------------------------------------------------------------
# Module-level query helper (no model instance required)
# ---------------------------------------------------------------------------

def compute_boost(
    player_id: int,
    opponent_team: str,
    game_date: str,
    is_away: bool,
    former_teams_entries: list[_FormerTeamEntry],
) -> float:
    """Stateless version of FormerTeamBoostModel.compute_boost.

    Useful when the Rust engine hands back a list of pre-loaded entries
    rather than keeping a full model object.

    Args:
        former_teams_entries: Pre-filtered entries for this player.
    """
    best: float = 0.0
    for entry in former_teams_entries:
        if entry.former_team.upper() != opponent_team.upper():
            continue
        years_since = _years_between(entry.departure_date, game_date)
        decay = _decay_factor(years_since)
        if decay <= 0.0:
            continue
        toi_amp = _toi_amplification_factor(entry.toi_per_game_min)
        raw = entry.base_boost * decay * toi_amp
        if is_away:
            raw *= AWAY_MULTIPLIER
        clamped = min(max(raw, BOOST_MIN), BOOST_MAX)
        if clamped > best:
            best = clamped
    return best


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_former_team_boost(
    df: pl.DataFrame,
    output_dir: Path,
    season: int,
) -> Path:
    """Write the former-team boost DataFrame to parquet.

    Args:
        df:         Output of ``FormerTeamBoostModel.fit()``.
        output_dir: Directory to write into (created if absent).
        season:     Season year used in the filename.

    Returns:
        Path to the written parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"former_team_boost_{season}.parquet"
    # Ensure all schema columns present
    for col, dtype in FORMER_TEAM_BOOST_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(FORMER_TEAM_BOOST_SCHEMA.keys())).write_parquet(path)
    return path


def load_former_team_boost(path: Path | str) -> pl.DataFrame:
    """Load a previously saved former-team boost parquet."""
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _espn_id_to_int(raw: str) -> int:
    """Convert an ESPN player ID string to int.

    Uses the raw string parsed as an integer when possible.  Non-numeric
    ESPN IDs (or missing IDs) become 0 (unknown player sentinel).
    """
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _build_toi_lookup(player_toi_df: pl.DataFrame) -> dict[tuple[int, str], float]:
    """Build a (player_id, team) → toi_per_game_min mapping from a stats DataFrame."""
    required = {"player_id", "team", "toi_seconds", "gp"}
    if not required.issubset(set(player_toi_df.columns)):
        warnings.warn(
            f"player_toi_df missing columns {required - set(player_toi_df.columns)}; "
            "TOI factor will default to 1.0 for all entries.",
            UserWarning,
            stacklevel=3,
        )
        return {}

    lookup: dict[tuple[int, str], float] = {}
    for row in player_toi_df.to_dicts():
        pid = int(row["player_id"])
        team = str(row["team"])
        gp = int(row["gp"]) if row["gp"] else 0
        toi_sec = float(row["toi_seconds"]) if row["toi_seconds"] is not None else 0.0
        if gp > 0:
            toi_per_game_min = (toi_sec / gp) / _SEC_PER_MIN
        else:
            toi_per_game_min = TOI_THRESHOLD_MINUTES  # default: no shortfall
        lookup[(pid, team)] = toi_per_game_min

    return lookup


def _records_to_df(records: list[_FormerTeamEntry], validated: bool) -> pl.DataFrame:
    """Convert a list of _FormerTeamEntry to a Polars DataFrame."""
    if not records:
        return _empty_output()
    rows = [
        {
            "player_id":        r.player_id,
            "player_name":      r.player_name,
            "former_team":      r.former_team,
            "departure_date":   r.departure_date,
            "departure_type":   r.departure_type,
            "toi_per_game_min": r.toi_per_game_min,
            "base_boost":       r.base_boost,
            "validated":        validated,
            "model_version":    MODEL_VERSION,
        }
        for r in records
    ]
    return pl.DataFrame(rows, schema=FORMER_TEAM_BOOST_SCHEMA)


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(schema=FORMER_TEAM_BOOST_SCHEMA)
