"""Concussion history flag — Feature 3.14.

Per-player binary + count summary of *prior* documented concussions.
Players with one or more career concussions get an elevated **fatigue
sensitivity multiplier**: the same FI inputs (3.7–3.12) translate to a
slightly larger composite FI score (3.17) for these players, because
post-concussion physiology degrades faster under sustained load. The
multiplier is applied at the FI composition stage, not directly to
ratings — keeping the per-feature signals interpretable.

Episode detection
-----------------
The injury-feed history is *daily snapshots*, so a single concussion
will appear on N consecutive reports. We collapse consecutive concussion
observations into a single episode using ``EPISODE_GAP_DAYS``: a gap of
strictly more than that many days between two flagged observations
starts a new episode.

A row is considered "concussion-flagged" when any of
``injury_type`` / ``injury_detail`` / ``status_raw`` contains one of
``CONCUSSION_KEYWORDS`` (case-insensitive).

Multiplier form
---------------
``concussion_multiplier(count) =
    min(MAX_MULTIPLIER, 1.0 + PER_EPISODE_INCREMENT × count)``

At defaults (PER_EPISODE_INCREMENT=0.05, MAX_MULTIPLIER=1.25):

    0 episodes → 1.00
    1 episode  → 1.05
    2 episodes → 1.10
    3 episodes → 1.15
    4 episodes → 1.20
    5+         → 1.25  (hard cap — beyond this the bigger story is
                       career-ending risk, not a fatigue multiplier)

Inputs
------
``injury_history_df`` (Polars) — long-form, one row per (player, date the
player appeared on a non-healthy injury feed)::

    player_id      Int64
    observed_date  Utf8       ("YYYY-MM-DD")
    injury_type    Utf8       e.g. "Concussion", "Head Injury" (nullable)
    injury_detail  Utf8       free-text from ESPN (nullable)
    status_raw     Utf8       e.g. "Concussion Protocol" (nullable)

Any of ``injury_type``, ``injury_detail``, ``status_raw`` can be missing
column-wise; we use whichever is present. At least one must be present
or the input is rejected with a DataMissingWarning.

``as_of_date`` (str, ``"YYYY-MM-DD"``) — the snapshot date.

Outputs
-------
One row per player with CONCUSSION_HISTORY_SCHEMA::

    player_id                   — Int64
    as_of_date                  — Utf8
    prior_concussion_count      — Int64    distinct episodes ≤ as_of
    has_prior_concussion        — Boolean  count >= 1
    last_concussion_date        — Utf8     latest episode start; "" if none
    days_since_last_concussion  — Int64    -1 if none
    fatigue_sensitivity_multiplier — Float64  in [1.0, MAX_MULTIPLIER]

Calibration knobs (constructor):
- ``per_episode_increment`` default 0.05
- ``max_multiplier``        default 1.25
- ``episode_gap_days``      default 14  — gap to start a new episode
- ``keywords``              default {"concussion", "head"}
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path

import polars as pl

from models.rapm_model import DataMissingWarning


MODEL_VERSION = "concussion_history_v1"

CONCUSSION_KEYWORDS: frozenset[str] = frozenset({"concussion", "head"})
EPISODE_GAP_DAYS         = 14
PER_EPISODE_INCREMENT    = 0.05
MAX_MULTIPLIER           = 1.25
NO_CONCUSSION_SENTINEL   = -1

HISTORY_REQUIRED_COLS = ("player_id", "observed_date")
HISTORY_TEXT_COLS     = ("injury_type", "injury_detail", "status_raw")

CONCUSSION_HISTORY_SCHEMA: dict[str, pl.DataType] = {
    "player_id":                      pl.Int64,
    "as_of_date":                     pl.Utf8,
    "prior_concussion_count":         pl.Int64,
    "has_prior_concussion":           pl.Boolean,
    "last_concussion_date":           pl.Utf8,
    "days_since_last_concussion":     pl.Int64,
    "fatigue_sensitivity_multiplier": pl.Float64,
}


# ---------------------------------------------------------------------------
# Public formulas
# ---------------------------------------------------------------------------

def concussion_multiplier(
    count: int,
    per_episode: float = PER_EPISODE_INCREMENT,
    max_mult:    float = MAX_MULTIPLIER,
) -> float:
    """``min(max_mult, 1.0 + per_episode × count)``."""
    c = max(0, int(count))
    return min(float(max_mult), 1.0 + float(per_episode) * c)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _matches_concussion(row: dict, keywords: frozenset[str]) -> bool:
    for col in HISTORY_TEXT_COLS:
        val = row.get(col)
        if val is None:
            continue
        lowered = str(val).lower()
        if any(k in lowered for k in keywords):
            return True
    return False


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        {col: pl.Series([], dtype=dt) for col, dt in CONCUSSION_HISTORY_SCHEMA.items()}
    )


def _validate(df: pl.DataFrame) -> bool:
    missing = [c for c in HISTORY_REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(
            f"injury_history_df missing columns: {missing}. Required: "
            f"{list(HISTORY_REQUIRED_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    if not any(c in df.columns for c in HISTORY_TEXT_COLS):
        warnings.warn(
            f"injury_history_df needs at least one of {list(HISTORY_TEXT_COLS)}.",
            DataMissingWarning,
            stacklevel=3,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# ConcussionHistoryFlag
# ---------------------------------------------------------------------------

class ConcussionHistoryFlag:
    """Concussion history → fatigue sensitivity multiplier (Feature 3.14)."""

    def __init__(
        self,
        per_episode_increment: float = PER_EPISODE_INCREMENT,
        max_multiplier:        float = MAX_MULTIPLIER,
        episode_gap_days:      int   = EPISODE_GAP_DAYS,
        keywords:              frozenset[str] | None = None,
    ) -> None:
        if per_episode_increment < 0.0:
            raise ValueError("per_episode_increment must be >= 0")
        if max_multiplier < 1.0:
            raise ValueError("max_multiplier must be >= 1.0")
        if episode_gap_days < 1:
            raise ValueError("episode_gap_days must be >= 1")

        self._per_episode = float(per_episode_increment)
        self._max_mult    = float(max_multiplier)
        self._gap         = int(episode_gap_days)
        self._keywords    = CONCUSSION_KEYWORDS if keywords is None else frozenset(
            k.lower() for k in keywords
        )
        if not self._keywords:
            raise ValueError("keywords must be non-empty")
        self._version = MODEL_VERSION

    # ------------------------------------------------------------------
    def compute(
        self,
        injury_history_df: pl.DataFrame,
        as_of_date:        str,
    ) -> pl.DataFrame:
        """Return one row per player observed in the history with the
        concussion summary as of ``as_of_date``."""
        if not _validate(injury_history_df):
            return _empty_output()

        try:
            as_of = _parse_date(as_of_date)
        except (TypeError, ValueError):
            raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")

        if len(injury_history_df) == 0:
            return _empty_output()

        per_player_dates: dict[int, list[date]] = {}
        for r in injury_history_df.to_dicts():
            pid_raw = r.get("player_id")
            obs_raw = r.get("observed_date")
            if pid_raw is None or obs_raw is None:
                continue
            try:
                d = _parse_date(str(obs_raw))
            except (TypeError, ValueError):
                continue
            if d > as_of:
                continue
            if not _matches_concussion(r, self._keywords):
                continue
            per_player_dates.setdefault(int(pid_raw), []).append(d)

        if not per_player_dates:
            return _empty_output()

        out_rows: list[dict] = []
        for pid in sorted(per_player_dates):
            dates_sorted = sorted(per_player_dates[pid])

            # Collapse consecutive obs into episodes: a new episode starts
            # whenever the gap between consecutive observations exceeds
            # ``episode_gap_days``. The first observation always opens an
            # episode.
            episode_starts: list[date] = [dates_sorted[0]]
            prev = dates_sorted[0]
            for d in dates_sorted[1:]:
                if (d - prev).days > self._gap:
                    episode_starts.append(d)
                prev = d

            count = len(episode_starts)
            last  = max(dates_sorted)
            days_since = (as_of - last).days

            out_rows.append({
                "player_id":                      int(pid),
                "as_of_date":                     as_of_date,
                "prior_concussion_count":         int(count),
                "has_prior_concussion":           True,
                "last_concussion_date":           last.isoformat(),
                "days_since_last_concussion":     int(days_since),
                "fatigue_sensitivity_multiplier": concussion_multiplier(
                    count, self._per_episode, self._max_mult
                ),
            })

        return pl.DataFrame(out_rows, schema=CONCUSSION_HISTORY_SCHEMA)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_concussion_history(df: pl.DataFrame, output_dir: Path, as_of_date: str) -> Path:
    """Write concussion-history DataFrame to parquet under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"concussion_history_{as_of_date}.parquet"
    for col, dtype in CONCUSSION_HISTORY_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df.select(list(CONCUSSION_HISTORY_SCHEMA.keys())).write_parquet(path)
    return path
