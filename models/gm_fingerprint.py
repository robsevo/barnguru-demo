"""Per-GM Behavioral Fingerprint — Feature 4.18.

Hierarchical model: shared prior dominates because GMs make only ~5–15
significant decisions per season.  V1 uses a heuristic archetype
classifier based on observable signals from the transactions feed and
standings context.

What it produces
----------------
For each team's GM (derived from the most recent front_office hire in
transactions, or manual entry in a future ``data/gms.json``):

- **gm_name** — best-effort extraction from transactions; empty if
  not yet detected.
- **action_archetype** — one of:
    ``stand_pat | add_rental | sell_veteran | rebuild | package_deal``
  The archetype with the highest probability.
- **archetype_probs** — dict of {archetype: probability} for all 5.
- **deadline_aggression** ∈ [0, 1] — proxy from the team's buyer/seller
  gap and number of recent transactions.
- **analytics_orientation** ∈ [0, 1] — placeholder (0.5) until the
  analytics-staff ingester lands.

V1 approach
-----------
1. Pull buyer/seller classification (4.15) for standings context.
2. Count recent transactions (call_up, send_down, signing, trade) in
   the last 30 days as an activity proxy.
3. Map (classification, activity_level) to archetype probabilities:
   - buyer + high activity → add_rental (0.5)
   - buyer + low activity → stand_pat (0.5)
   - seller + high activity → sell_veteran (0.5)
   - seller + low activity → rebuild (0.4)
   - neutral → stand_pat (0.4)
4. Deadline aggression = buyer_seller confidence × activity_level.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "gm_fingerprint_v1"

ARCHETYPES = ["stand_pat", "add_rental", "sell_veteran", "rebuild", "package_deal"]

# ESPN → NHL canonical abbreviation mapping.
_ESPN_TO_NHL: dict[str, str] = {
    "NJ": "NJD", "TB": "TBL", "LA": "LAK", "SJ": "SJS",
    "UTA": "UTH", "ARI": "UTH", "WAS": "WSH",
}


class DataMissingWarning(UserWarning):
    pass


GM_FINGERPRINT_SCHEMA: dict[str, pl.DataType] = {
    "team":                   pl.Utf8,
    "season":                 pl.Int64,
    "gm_name":                pl.Utf8,
    "action_archetype":       pl.Utf8,
    "prob_stand_pat":         pl.Float64,
    "prob_add_rental":        pl.Float64,
    "prob_sell_veteran":      pl.Float64,
    "prob_rebuild":           pl.Float64,
    "prob_package_deal":      pl.Float64,
    "deadline_aggression":    pl.Float64,
    "analytics_orientation":  pl.Float64,
    "recent_tx_count":        pl.Int64,
    "model_version":          pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recent_tx_counts(transactions_df: pl.DataFrame) -> dict[str, int]:
    """Count non-front-office transactions per team in the last 30 days."""
    if transactions_df.is_empty():
        return {}
    non_fo = transactions_df.filter(pl.col("event_type") != "front_office")
    if non_fo.is_empty():
        return {}
    counts: dict[str, int] = {}
    for r in non_fo.group_by("team").agg(pl.len().alias("cnt")).iter_rows(named=True):
        raw = str(r["team"]).upper()
        team = _ESPN_TO_NHL.get(raw, raw)
        counts[team] = counts.get(team, 0) + int(r["cnt"])
    return counts


def _archetype_probs(classification: str, activity: float) -> dict[str, float]:
    """Map (classification, activity_level) to archetype probabilities."""
    if classification == "buyer":
        if activity >= 0.5:
            return {"stand_pat": 0.15, "add_rental": 0.50, "sell_veteran": 0.0,
                    "rebuild": 0.0, "package_deal": 0.35}
        return {"stand_pat": 0.50, "add_rental": 0.30, "sell_veteran": 0.0,
                "rebuild": 0.0, "package_deal": 0.20}
    if classification == "seller":
        if activity >= 0.5:
            return {"stand_pat": 0.05, "add_rental": 0.0, "sell_veteran": 0.50,
                    "rebuild": 0.30, "package_deal": 0.15}
        return {"stand_pat": 0.10, "add_rental": 0.0, "sell_veteran": 0.30,
                "rebuild": 0.45, "package_deal": 0.15}
    # Neutral
    return {"stand_pat": 0.40, "add_rental": 0.20, "sell_veteran": 0.10,
            "rebuild": 0.15, "package_deal": 0.15}


def _extract_gm_name(transactions_df: pl.DataFrame, team: str) -> str:
    """Best-effort GM name extraction from transactions."""
    if transactions_df.is_empty():
        return ""
    fo = transactions_df.filter(
        (pl.col("event_type") == "front_office")
    )
    if fo.is_empty():
        return ""
    # Look for GM hires for this team
    import re
    for r in fo.sort("date", descending=True).iter_rows(named=True):
        raw_team = str(r.get("team") or "").upper()
        norm_team = _ESPN_TO_NHL.get(raw_team, raw_team)
        if norm_team != team:
            continue
        desc = str(r.get("description") or "")
        m = re.search(
            r"(?i)(?:named|hired|appointed)\s+(.+?)\s+(?:as\s+)?(?:general\s+manager|GM)",
            desc,
        )
        if m:
            return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_gm_fingerprint(
    buyer_seller_df:  pl.DataFrame,
    transactions_df:  pl.DataFrame,
    season:           int,
) -> pl.DataFrame:
    """Build per-team GM behavioral fingerprint.

    Args:
        buyer_seller_df:   output of compute_buyer_seller (4.15).
        transactions_df:   combined transactions frame.
        season:            NHL season start year.
    """
    if buyer_seller_df.is_empty():
        warnings.warn(
            "compute_gm_fingerprint: empty buyer_seller frame.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=GM_FINGERPRINT_SCHEMA)

    tx_counts = _recent_tx_counts(transactions_df)
    max_tx    = max(tx_counts.values()) if tx_counts else 1

    rows: list[dict[str, Any]] = []
    for r in buyer_seller_df.iter_rows(named=True):
        team = str(r["team"])
        cls  = str(r.get("classification") or "neutral")
        conf = float(r.get("confidence") or 0.0)

        tx_count = tx_counts.get(team, 0)
        activity = tx_count / max_tx if max_tx > 0 else 0.0

        probs = _archetype_probs(cls, activity)
        top_arch = max(probs, key=probs.get)  # type: ignore[arg-type]
        deadline_agg = round(conf * activity, 4)

        gm_name = _extract_gm_name(transactions_df, team)

        rows.append({
            "team":                  team,
            "season":                int(season),
            "gm_name":               gm_name,
            "action_archetype":      top_arch,
            "prob_stand_pat":        probs["stand_pat"],
            "prob_add_rental":       probs["add_rental"],
            "prob_sell_veteran":     probs["sell_veteran"],
            "prob_rebuild":          probs["rebuild"],
            "prob_package_deal":     probs["package_deal"],
            "deadline_aggression":   deadline_agg,
            "analytics_orientation": 0.5,
            "recent_tx_count":       tx_count,
            "model_version":         MODEL_VERSION,
        })

    df = pl.DataFrame(rows)
    for col, dtype in GM_FINGERPRINT_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(GM_FINGERPRINT_SCHEMA.keys())).sort("deadline_aggression", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_gm_fingerprint(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"gm_fingerprint_{season}.parquet"
    df.write_parquet(path)
    return path


def read_gm_fingerprint(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "gm_fingerprint"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"gm_fingerprint_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("gm_fingerprint_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
