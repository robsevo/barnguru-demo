"""Front Office Regime Change Detector — Feature 4.14.

Separate from coaching staff (4.13).  Front-office changes (GM, AGM,
President of Hockey Ops) affect scouting philosophy, development
culture, and analytics investment on a *season* timescale — much slower
decay than a coaching change.

Canonical validation: BUF Dec 15–21, 2025 (Adams → Kekalainen + Bergevin
+ Flynn).

Data sources
------------
- ``data/transactions/transactions_*.parquet`` — ``event_type='front_office'``
  rows with descriptions containing GM / president keywords.

What it produces
----------------
One row per detected FO change event:

- date, team, fo_role (gm | agm | president_hockey_ops | other_exec)
- person_out, person_in
- description
- regime_change_trigger (bool)
- decay_games (50 for GM/Pres, 30 for AGM, 20 for other)

Output: ``fo_regime_changes/fo_regime_changes_{season}.parquet``

Triggers the regime change pipeline (2.14, 15.4) with **slow decay**
so that the model doesn't overreact game-by-game but does credit the
gradual shift in team philosophy.
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

MODEL_VERSION = "fo_regime_detector_v1"

_ESPN_TO_NHL: dict[str, str] = {
    "NJ": "NJD", "TB": "TBL", "LA": "LAK", "SJ": "SJS",
    "UTA": "UTH", "ARI": "UTH", "WAS": "WSH",
}

DECAY_GAMES: dict[str, int] = {
    "gm":                   50,
    "president_hockey_ops": 50,
    "agm":                  30,
    "other_exec":           20,
}

_GM_RE = re.compile(
    r"(?i)\b(general\s+manager|GM)\b"
)
_PRES_RE = re.compile(
    r"(?i)\b(president\s+of\s+hockey\s+op|president.*hockey|hockey\s+ops)\b"
)
_AGM_RE = re.compile(
    r"(?i)\b(assistant\s+general\s+manager|AGM|asst\.?\s+GM)\b"
)
_EXEC_RE = re.compile(
    r"(?i)\b(vice\s+president|VP|director\s+of\s+(?:player\s+)?personnel"
    r"|director\s+of\s+scouting|chief\s+scout)\b"
)


class DataMissingWarning(UserWarning):
    """Raised when FO regime data is absent or insufficient."""


FO_REGIME_SCHEMA: dict[str, pl.DataType] = {
    "date":                    pl.Utf8,
    "team":                    pl.Utf8,
    "fo_role":                 pl.Utf8,
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


def _classify_fo_role(desc: str) -> str | None:
    """Classify a front_office description into an FO role.

    Returns None if the description doesn't match any FO keyword (it may
    be a coaching change which is handled by 4.13).
    """
    if _PRES_RE.search(desc):
        return "president_hockey_ops"
    if _AGM_RE.search(desc):
        return "agm"
    if _GM_RE.search(desc):
        return "gm"
    if _EXEC_RE.search(desc):
        return "other_exec"
    return None


def _extract_person_out(desc: str) -> str:
    m = re.search(
        r"(?i)(?:fired|dismissed|relieved|terminated|parted ways with)\s+"
        r"(?:general\s+manager|GM|president.*?|AGM|assistant.*?GM)\s+(.+?)(?:\.|,|$)",
        desc,
    )
    return m.group(1).strip() if m else ""


def _extract_person_in(desc: str) -> str:
    m = re.search(
        r"(?i)(?:named|hired|appointed|promoted)\s+(.+?)\s+(?:as\s+)?"
        r"(?:general\s+manager|GM|president|AGM|assistant.*?GM)",
        desc,
    )
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_fo_regime_changes(
    transactions_df: pl.DataFrame,
    season:          int,
) -> pl.DataFrame:
    """Scan transactions for front-office regime changes.

    Args:
        transactions_df: combined transactions DataFrame.
        season:          NHL season start year.

    Returns:
        DataFrame matching FO_REGIME_SCHEMA.
    """
    if transactions_df.is_empty():
        warnings.warn(
            "detect_fo_regime_changes: empty transactions — no FO changes detectable.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=FO_REGIME_SCHEMA)

    required = {"date", "event_type", "team", "description"}
    missing = required - set(transactions_df.columns)
    if missing:
        raise ValueError(f"transactions_df missing columns: {sorted(missing)}")

    fo_events = transactions_df.filter(pl.col("event_type") == "front_office")

    if fo_events.is_empty():
        return pl.DataFrame(schema=FO_REGIME_SCHEMA)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for r in fo_events.iter_rows(named=True):
        date = str(r.get("date") or "")
        raw_team = str(r.get("team") or "").upper()
        team = _ESPN_TO_NHL.get(raw_team, raw_team)
        desc = str(r.get("description") or "")
        key = (date, team, desc)
        if key in seen:
            continue
        seen.add(key)

        fo_role = _classify_fo_role(desc)
        if fo_role is None:
            continue

        rows.append({
            "date":                  date,
            "team":                  team,
            "fo_role":               fo_role,
            "person_out":            _extract_person_out(desc),
            "person_in":             _extract_person_in(desc),
            "description":           desc,
            "regime_change_trigger": True,
            "decay_games":           DECAY_GAMES.get(fo_role, 20),
            "season":                int(season),
            "model_version":         MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=FO_REGIME_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in FO_REGIME_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(FO_REGIME_SCHEMA.keys())).sort("date")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_fo_regime_changes(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fo_regime_changes_{season}.parquet"
    df.write_parquet(path)
    return path


def read_fo_regime_changes(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "fo_regime_changes"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"fo_regime_changes_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("fo_regime_changes_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
