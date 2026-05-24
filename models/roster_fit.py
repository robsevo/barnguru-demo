"""Roster Fit Score — Feature 4.12.

Match coach style vector (4.11) against roster archetype composition
(2.11) → single fit score ∈ [0, 1] per team.

Canonical example: Roy's high defensive-zone-structure system with the
Islanders' heavy *offensive-D* archetype on the back end = low fit =
performance drag.  This model surfaces that mismatch as a number.

Approach
--------
For each style dimension, the model carries a curated list of archetypes
that *support* that style — e.g. "dz_structure" is supported by the
"Defensive Anchor" and "Elite Two-Way" archetypes.  For each team:

1. Compute archetype share: fraction of team minutes by archetype.  The
   share for each archetype = sum of EV TOI of every player assigned to
   that archetype / total team EV TOI.
2. For each style dim, compute ``support_score`` = sum of shares of
   archetypes flagged as "supporting" that style.
3. ``fit_score`` = Σ_dim style_rank[dim] × support_score[dim] /
                    Σ_dim style_rank[dim]
   i.e. weighted average of support, weighted by the *strength of the
   style choice*.  A coach with a balanced (~0.5 across the board) style
   vector won't be punished by mismatches; an extreme coach is heavily
   penalized when their archetypes don't back the system.

Output
------
One row per team for the snapshot season.  Columns:

- team, season
- archetype_shares     — list of (archetype, share) tuples, sorted desc.
                         (Stored as two parallel lists for parquet-friendliness.)
- archetype_top        — the top archetype
- fit_score            — [0, 1]
- mismatch_dim         — the style dimension with the weakest support
- mismatch_support     — that dimension's support score
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "roster_fit_v1"


class DataMissingWarning(UserWarning):
    """Raised when fit inputs are absent or insufficient."""


# Mapping style dimension → list of supporting archetypes.
# Built from PLAN.md feature descriptions + the archetype names that
# actually appear in ``archetype_assignments_*.parquet``.  Keep the list
# auditable: the radar is supposed to be Bob-readable.
STYLE_TO_ARCHETYPES: dict[str, list[str]] = {
    "forecheck_aggression":  ["Strong Two-Way", "Elite Two-Way", "Depth Forward"],
    "dz_structure":          ["Defensive Anchor", "Elite Two-Way"],
    "pace":                  ["Elite Scorer", "Elite Two-Way"],
    "physicality":           ["Defensive Anchor", "Depth Forward"],
    "oz_structure":          ["Elite Scorer", "PP Specialist", "Elite Two-Way"],
    "nz_tendency":           ["Elite Two-Way", "Elite Scorer"],
    "line_match":            ["Strong Two-Way", "Middle-Six Forward"],
    "st_aggression":         ["PP Specialist", "Elite Scorer", "Elite Two-Way"],
}


ROSTER_FIT_SCHEMA: dict[str, pl.DataType] = {
    "team":                 pl.Utf8,
    "season":               pl.Int64,
    "n_skaters":            pl.Int64,
    "archetype_top":        pl.Utf8,
    "archetypes":           pl.List(pl.Utf8),
    "archetype_shares":     pl.List(pl.Float64),
    "fit_score":            pl.Float64,
    "mismatch_dim":         pl.Utf8,
    "mismatch_support":     pl.Float64,
    "model_version":        pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _archetype_shares(
    archetype_df: pl.DataFrame,
    rapm_df:      pl.DataFrame,
    team_lookup:  dict[int, str],
) -> dict[str, dict[str, float]]:
    """Compute per-team archetype share of EV TOI.

    Returns ``{team_abbrev: {archetype: share}}`` where shares sum to ≤ 1.0.

    Players can have multiple archetype rows (per-season multiple
    assignments); we use the assignment with smallest distance per
    (player_id, season) pair.
    """
    required_a = {"player_id", "archetype", "distance", "season"}
    required_r = {"player_id", "team", "toi_ev"}
    if not required_a.issubset(set(archetype_df.columns)):
        raise ValueError(f"archetype_df missing: {sorted(required_a - set(archetype_df.columns))}")
    if not required_r.issubset(set(rapm_df.columns)):
        raise ValueError(f"rapm_df missing: {sorted(required_r - set(rapm_df.columns))}")

    # Closest archetype per (player_id, season)
    primary = (
        archetype_df
        .sort(["player_id", "season", "distance"])
        .group_by(["player_id", "season"], maintain_order=True)
        .agg(pl.col("archetype").first().alias("archetype"))
    )

    # Map RAPM team_id (string) → abbrev
    def _abbrev_for(s: Any) -> str | None:
        try:
            return team_lookup.get(int(s))
        except (TypeError, ValueError):
            return None

    joined = (
        rapm_df.select(["player_id", "team", "toi_ev"])
        .with_columns(pl.col("toi_ev").cast(pl.Float64).fill_null(0.0))
        .join(primary, on="player_id", how="left")
    )

    shares: dict[str, dict[str, float]] = {}
    team_totals: dict[str, float] = {}

    for r in joined.iter_rows(named=True):
        abbrev = _abbrev_for(r["team"])
        if abbrev is None:
            continue
        arch = r.get("archetype") or "Unassigned"
        toi  = float(r.get("toi_ev") or 0.0)
        if toi <= 0:
            continue
        shares.setdefault(abbrev, {}).setdefault(arch, 0.0)
        shares[abbrev][arch] += toi
        team_totals[abbrev] = team_totals.get(abbrev, 0.0) + toi

    # Normalize to fractions
    norm: dict[str, dict[str, float]] = {}
    for team, archmap in shares.items():
        total = team_totals.get(team, 0.0) or 1.0
        norm[team] = {a: v / total for a, v in archmap.items()}
    return norm


def _fit_for_team(
    style_row:        dict[str, float],
    archetype_share:  dict[str, float],
) -> tuple[float, str, float]:
    """Compute fit_score + weakest mismatch dimension for one team.

    ``style_row``: dict of {dim_name: rank ∈ [0, 1]}.  NaN treated as 0.5
    (league-average style → neutral weight).
    """
    weighted_sum = 0.0
    weight_total = 0.0
    weakest_dim  = ""
    weakest_supp = 1.0

    for dim, supporting in STYLE_TO_ARCHETYPES.items():
        weight = style_row.get(dim, 0.5)
        if weight != weight:    # NaN
            weight = 0.5
        # Support = sum of shares of supporting archetypes
        support = sum(archetype_share.get(a, 0.0) for a in supporting)
        weighted_sum  += weight * support
        weight_total  += weight
        if support < weakest_supp:
            weakest_supp = support
            weakest_dim  = dim

    fit = (weighted_sum / weight_total) if weight_total > 0 else 0.5
    # Clamp to [0, 1] just in case.
    fit = max(0.0, min(1.0, fit))
    return fit, weakest_dim, weakest_supp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_roster_fit(
    style_df:     pl.DataFrame,
    archetype_df: pl.DataFrame,
    rapm_df:      pl.DataFrame,
    team_lookup:  dict[int, str],
    season:       int,
) -> pl.DataFrame:
    """Build per-team roster-fit rows."""
    if style_df.is_empty():
        warnings.warn(
            "compute_roster_fit: style_df empty — no fit rows produced.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=ROSTER_FIT_SCHEMA)
    if archetype_df.is_empty() or rapm_df.is_empty():
        warnings.warn(
            "compute_roster_fit: archetype/RAPM frame empty — fit scores will fall back to 0.5.",
            DataMissingWarning,
            stacklevel=2,
        )

    shares = _archetype_shares(archetype_df, rapm_df, team_lookup) if not (
        archetype_df.is_empty() or rapm_df.is_empty()
    ) else {}

    rows: list[dict[str, Any]] = []
    for r in style_df.iter_rows(named=True):
        team = r["team"]
        style_row = {
            "forecheck_aggression": float(r.get("forecheck_aggression_rank") or 0.5),
            "dz_structure":         float(r.get("dz_structure_rank") or 0.5),
            "pace":                 float(r.get("pace_rank") or 0.5),
            "physicality":          float(r.get("physicality_rank") or 0.5),
            "oz_structure":         float(r.get("oz_structure_rank") or 0.5),
            "nz_tendency":          float(r.get("nz_tendency_rank")
                                          if r.get("nz_tendency_rank") is not None and
                                             r.get("nz_tendency_rank") == r.get("nz_tendency_rank")
                                          else 0.5),
            "line_match":           float(r.get("line_match_rank") or 0.5),
            "st_aggression":        float(r.get("st_aggression_rank") or 0.5),
        }
        share = shares.get(team, {})
        # Sort top archetypes
        sorted_arch = sorted(share.items(), key=lambda kv: kv[1], reverse=True)
        archetypes  = [a for a, _ in sorted_arch]
        arch_shares = [round(s, 4) for _, s in sorted_arch]
        top_arch    = archetypes[0] if archetypes else ""
        n_skaters   = len(archetypes)

        fit, mismatch_dim, mismatch_supp = _fit_for_team(style_row, share) if share else (
            0.5, "", 0.0
        )

        rows.append({
            "team":              team,
            "season":            int(season),
            "n_skaters":         n_skaters,
            "archetype_top":     top_arch,
            "archetypes":        archetypes,
            "archetype_shares":  arch_shares,
            "fit_score":         round(fit, 4),
            "mismatch_dim":      mismatch_dim,
            "mismatch_support":  round(mismatch_supp, 4),
            "model_version":     MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=ROSTER_FIT_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in ROSTER_FIT_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(ROSTER_FIT_SCHEMA.keys())).sort("fit_score", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_roster_fit(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"roster_fit_{season}.parquet"
    df.write_parquet(path)
    return path


def read_roster_fit(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "roster_fit"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"roster_fit_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("roster_fit_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
