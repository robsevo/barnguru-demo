"""Coaching Staff Change Detector — Feature 4.13.

Scans the transactions feed for mid-season coaching staff changes (head
coach firings/hirings, PP/PK coordinator changes, goalie coach changes)
and flags them so the regime change pipeline (2.14) can accelerate
Bayesian updates on the affected model components.

Data sources
------------
- ``data/transactions/transactions_*.parquet`` — the ESPN-sourced
  transaction feed from Feature 1.14.  ``event_type='front_office'``
  rows with descriptions containing coaching keywords.
- ``data/coaches.json`` — the static head-coach roster; used to cross-
  reference the affected team.

What it produces
----------------
One row per detected staff change event:

- date, team, change_type (head_coach | coordinator | goalie_coach | unknown)
- person_out, person_in (when extractable from description)
- description (raw text)
- regime_change_trigger (bool — always True for detected events)
- decay_games (default 20 for HC, 15 for coordinator, 10 for goalie coach)

Output: ``staff_changes/staff_changes_{season}.parquet``

V1 caveat
---------
Coordinator-level changes are rarely reported in the ESPN transactions
feed — the detector catches head-coach firings reliably but coordinator
swaps often appear only in beat-writer reports.  The column is present
so v2 (WhizBrain or beat-writer scraper) can populate it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "staff_change_detector_v1"

# ESPN → NHL canonical abbreviation mapping.
_ESPN_TO_NHL: dict[str, str] = {
    "NJ": "NJD", "TB": "TBL", "LA": "LAK", "SJ": "SJS",
    "UTA": "UTH", "ARI": "UTH", "WAS": "WSH",
}

DECAY_GAMES: dict[str, int] = {
    "head_coach":   20,
    "coordinator":  15,
    "goalie_coach": 10,
    "unknown":      15,
}

_HC_RE = re.compile(
    r"\b(fired|dismissed|relieved|terminated|parted ways with)\b.*\b(head coach|coach)\b"
    r"|"
    r"\b(named|hired|appointed|promoted)\b.*\b(head coach|interim.*coach)\b",
    re.IGNORECASE,
)

_COORD_RE = re.compile(
    r"\b(pp|pk|power.?play|penalty.?kill)\s*(coordinator|coach)\b",
    re.IGNORECASE,
)

_GOALIE_COACH_RE = re.compile(
    r"\bgoalie\s*(coach|coordinator)\b"
    r"|"
    r"\bgoaltending\s*(coach|coordinator|consultant)\b",
    re.IGNORECASE,
)


class DataMissingWarning(UserWarning):
    """Raised when staff change data is absent or insufficient."""


STAFF_CHANGE_SCHEMA: dict[str, pl.DataType] = {
    "date":                    pl.Utf8,
    "team":                    pl.Utf8,
    "change_type":             pl.Utf8,
    "person_out":              pl.Utf8,
    "person_in":               pl.Utf8,
    "description":             pl.Utf8,
    "regime_change_trigger":   pl.Boolean,
    "decay_games":             pl.Int64,
    "season":                  pl.Int64,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_change(desc: str) -> str:
    """Classify a front_office transaction description into change_type."""
    if _GOALIE_COACH_RE.search(desc):
        return "goalie_coach"
    if _COORD_RE.search(desc):
        return "coordinator"
    if _HC_RE.search(desc):
        return "head_coach"
    if re.search(r"(?i)\bcoach\b", desc):
        return "head_coach"
    return "unknown"


def _extract_person_out(desc: str) -> str:
    """Best-effort extraction of the person removed."""
    # "Fired head coach Patrick Roy" → Patrick Roy
    m = re.search(
        r"(?i)(?:fired|dismissed|relieved|terminated|parted ways with)\s+"
        r"(?:head\s+)?(?:coach|coordinator)\s+(.+?)(?:\.|,|$)",
        desc,
    )
    return m.group(1).strip() if m else ""


def _extract_person_in(desc: str) -> str:
    """Best-effort extraction of the person hired."""
    m = re.search(
        r"(?i)(?:named|hired|appointed|promoted)\s+(.+?)\s+(?:as\s+)?(?:head\s+)?(?:coach|coordinator|interim)",
        desc,
    )
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_staff_changes(
    transactions_df: pl.DataFrame,
    season:          int,
) -> pl.DataFrame:
    """Scan transactions for coaching staff changes.

    Args:
        transactions_df: combined transactions DataFrame (all dates).
        season:          NHL season start year (for output tagging).

    Returns:
        DataFrame matching STAFF_CHANGE_SCHEMA — one row per detected event.
    """
    if transactions_df.is_empty():
        warnings.warn(
            "detect_staff_changes: empty transactions frame — no changes detectable.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=STAFF_CHANGE_SCHEMA)

    required = {"date", "event_type", "team", "description"}
    missing = required - set(transactions_df.columns)
    if missing:
        raise ValueError(f"transactions_df missing columns: {sorted(missing)}")

    # Filter to front_office events with coaching keywords
    coaching_kw = transactions_df.filter(
        (pl.col("event_type") == "front_office")
        & pl.col("description").str.contains("(?i)coach|coordinator")
    )

    if coaching_kw.is_empty():
        return pl.DataFrame(schema=STAFF_CHANGE_SCHEMA)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()  # dedup (date, team, desc)

    for r in coaching_kw.iter_rows(named=True):
        date = str(r.get("date") or "")
        raw_team = str(r.get("team") or "").upper()
        team = _ESPN_TO_NHL.get(raw_team, raw_team)
        desc = str(r.get("description") or "")
        key = (date, team, desc)
        if key in seen:
            continue
        seen.add(key)

        change_type = _classify_change(desc)
        rows.append({
            "date":                  date,
            "team":                  team,
            "change_type":           change_type,
            "person_out":            _extract_person_out(desc),
            "person_in":             _extract_person_in(desc),
            "description":           desc,
            "regime_change_trigger": True,
            "decay_games":           DECAY_GAMES.get(change_type, 15),
            "season":                int(season),
            "model_version":         MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=STAFF_CHANGE_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in STAFF_CHANGE_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(STAFF_CHANGE_SCHEMA.keys())).sort("date")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_staff_changes(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"staff_changes_{season}.parquet"
    df.write_parquet(path)
    return path


def read_staff_changes(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "staff_changes"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"staff_changes_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("staff_changes_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
