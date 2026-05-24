"""Per-Coach Decision Network — Feature 4.17.

V1: imitation learning from the static tendency models (4.1–4.6).
Rather than building a full neural network (deferred to v2 when we have
per-shift PBP context), v1 consolidates every coach's Phase 4 outputs
into a **single unified decision profile** — one row per coach with
normalized probability estimates for the 7 decision types.

Architecture intent (v2)
------------------------
Hierarchical net: shared base (league-average coaching decisions) +
per-coach adaptation layers fine-tuned on career PBP.  Inputs at
simulation time: ``[game_state, period, time_remaining, own_line_FI_avg,
PP/PK_situation, recent_momentum]``.  Outputs: decision probs
``[call_timeout, hold, pull_goalie, keep_goalie, match_line, send_4th,
challenge]``.  Sample-size constraint respected: ~246 decisions/type/
coach/season → shared prior dominates for sparse coaches.

V1 outputs
----------
For each coach we compute:

- **timeout_aggression** ∈ [0, 1] — from timeout_usage rate data (4.4).
  Higher = calls more timeouts per game.
- **pull_aggression** ∈ [0, 1] — from goalie_pull mean seconds remaining
  (4.5).  Higher = pulls earlier.
- **line_shelter_score** ∈ [0, 1] — from line_deployment cohesion (4.1).
  Higher = more concentrated top-line minutes.
- **st_first_unit_lean** ∈ [0, 1] — from st_deployment PP1 share (4.3).
  Higher = leans harder on first PP unit.
- **penalty_discipline** ∈ [0, 1] — from penalty_tendency penalties/game
  (4.6).  Higher = more disciplined (fewer penalties).
- **matching_intensity** ∈ [0, 1] — from line_matching home top-line
  concentration (4.2).  Higher = more active line matching.
- **overall_aggression** — mean of the above 6 dimensions.

All dimensions are league-percentile-ranked ∈ [0, 1] within the
snapshot season so they share a common scale.

Output: ``coach_decision_net/coach_decision_net_{season}.parquet``
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "coach_decision_net_v1"


class DataMissingWarning(UserWarning):
    pass


COACH_DECISION_SCHEMA: dict[str, pl.DataType] = {
    "coach_name":           pl.Utf8,
    "team":                 pl.Utf8,
    "season":               pl.Int64,
    "timeout_aggression":   pl.Float64,
    "pull_aggression":      pl.Float64,
    "line_shelter_score":   pl.Float64,
    "st_first_unit_lean":   pl.Float64,
    "penalty_discipline":   pl.Float64,
    "matching_intensity":   pl.Float64,
    "overall_aggression":   pl.Float64,
    "model_version":        pl.Utf8,
}

DECISION_DIMENSIONS: list[str] = [
    "timeout_aggression",
    "pull_aggression",
    "line_shelter_score",
    "st_first_unit_lean",
    "penalty_discipline",
    "matching_intensity",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile_rank(values: list[float]) -> list[float]:
    n_valid = sum(1 for v in values if v == v)
    if n_valid == 0:
        return [0.5] * len(values)
    valid = sorted([v for v in values if v == v])
    ranks: list[float] = []
    for v in values:
        if v != v:
            ranks.append(0.5)
            continue
        lo = next(i for i, x in enumerate(valid) if x >= v)
        hi = len(valid) - 1 - next(i for i, x in enumerate(reversed(valid)) if x <= v)
        avg_rank = (lo + hi) / 2.0
        ranks.append(avg_rank / max(1, n_valid - 1) if n_valid > 1 else 0.5)
    return ranks


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_coach_decision_net(
    coaches:        list[dict],
    timeout_df:     pl.DataFrame,
    goalie_pull_df: pl.DataFrame,
    line_deploy_df: pl.DataFrame,
    st_deploy_df:   pl.DataFrame,
    penalty_df:     pl.DataFrame,
    line_match_df:  pl.DataFrame,
    season:         int,
) -> pl.DataFrame:
    """Build unified per-coach decision profiles from Phase 4 outputs.

    Args:
        coaches:         list of dicts from data/coaches.json.
        timeout_df:      timeout_usage_{season}.parquet.
        goalie_pull_df:  goalie_pull_{season}.parquet.
        line_deploy_df:  line_deployment_{season}.parquet.
        st_deploy_df:    st_deployment_{season}.parquet.
        penalty_df:      penalty_tendency_{season}.parquet.
        line_match_df:   line_matching_{season}.parquet.
        season:          NHL season start year.
    """
    if not coaches:
        warnings.warn("compute_coach_decision_net: empty coaches list.",
                       DataMissingWarning, stacklevel=2)
        return pl.DataFrame(schema=COACH_DECISION_SCHEMA)

    # Pre-compute per-team raw signals from the Phase 4 parquets.
    # 1. Timeout aggression — total timeouts / GP
    timeout_rate: dict[str, float] = {}
    if not timeout_df.is_empty() and "team" in timeout_df.columns:
        for team, grp in timeout_df.group_by("team"):
            t = team[0] if isinstance(team, tuple) else team
            n_to = int(grp["n_timeouts"].sum() or 0) if "n_timeouts" in grp.columns else 0
            n_gp = int(grp["n_games"].max() or 1) if "n_games" in grp.columns else 1
            timeout_rate[t] = n_to / max(1, n_gp)

    # 2. Pull aggression — mean pull time (higher seconds = earlier = more aggressive)
    pull_mean_secs: dict[str, float] = {}
    if not goalie_pull_df.is_empty() and "team" in goalie_pull_df.columns:
        for team, grp in goalie_pull_df.group_by("team"):
            t = team[0] if isinstance(team, tuple) else team
            vals = [_safe_float(v) for v in grp["mean_pull_time_secs"].to_list() if v is not None]
            pull_mean_secs[t] = sum(vals) / len(vals) if vals else 0.0

    # 3. Line shelter — concentration = inverse entropy of F-line shares
    line_shelter: dict[str, float] = {}
    if not line_deploy_df.is_empty() and "team" in line_deploy_df.columns:
        f_lines = line_deploy_df.filter(pl.col("line_type") == "F")
        for team, grp in f_lines.group_by("team"):
            t = team[0] if isinstance(team, tuple) else team
            shares = [_safe_float(v) for v in grp["share_of_team_toi"].to_list() if v is not None and _safe_float(v) > 0]
            if not shares:
                line_shelter[t] = 0.5
                continue
            s = sum(shares) or 1.0
            probs = [p / s for p in shares]
            ent = -sum(p * math.log(p) for p in probs if p > 0)
            n = len(probs)
            max_ent = math.log(n) if n > 1 else 1.0
            line_shelter[t] = 1.0 - (ent / max_ent if max_ent > 0 else 0.0)

    # 4. ST first-unit lean — PP1 share of PP TOI
    st_lean: dict[str, float] = {}
    if not st_deploy_df.is_empty() and "team" in st_deploy_df.columns:
        pp_only = st_deploy_df.filter(pl.col("unit_type").is_in(["PP1", "PP2"]))
        for team, grp in pp_only.group_by("team"):
            t = team[0] if isinstance(team, tuple) else team
            pp1 = grp.filter(pl.col("unit_type") == "PP1")
            total_pp_toi = _safe_float(grp["team_st_toi"].first(), 1.0)
            pp1_toi = _safe_float(pp1["unit_toi_secs"].first(), 0.0) if not pp1.is_empty() else 0.0
            st_lean[t] = pp1_toi / total_pp_toi if total_pp_toi > 0 else 0.0

    # 5. Penalty discipline — inverse of penalties/game (fewer = more disciplined)
    pen_rate: dict[str, float] = {}
    if not penalty_df.is_empty() and "team" in penalty_df.columns:
        for r in penalty_df.iter_rows(named=True):
            pen_rate[r["team"]] = _safe_float(r.get("penalties_taken_per_game"))

    # 6. Matching intensity — home top-line share from line_matching
    match_intensity: dict[str, float] = {}
    if not line_match_df.is_empty() and "team" in line_match_df.columns:
        from models.line_matching import team_matchup_profile
        for team_abbrev in set(c.get("team", "") for c in coaches):
            if not team_abbrev:
                continue
            try:
                prof = team_matchup_profile(line_match_df, team_abbrev, line_type="F")
                if not prof.is_empty():
                    home_prof = prof.filter(pl.col("venue") == "home")
                    if not home_prof.is_empty():
                        own1_opp1 = home_prof.filter(
                            (pl.col("own_line_rank") == 1) & (pl.col("opp_line_rank") == 1)
                        )
                        if not own1_opp1.is_empty():
                            match_intensity[team_abbrev] = _safe_float(
                                own1_opp1["weighted_share"].first()
                            )
            except Exception:
                pass

    # Build per-coach raw values, then percentile-rank.
    raw_rows: list[dict[str, Any]] = []
    for coach in coaches:
        cname = (coach.get("name") or "").strip()
        team  = (coach.get("team") or "").strip().upper()
        if not cname or not team:
            continue

        raw_rows.append({
            "coach_name":           cname,
            "team":                 team,
            "season":               int(season),
            "timeout_aggression":   timeout_rate.get(team, 0.0),
            "pull_aggression":      pull_mean_secs.get(team, 0.0),
            "line_shelter_score":   line_shelter.get(team, 0.5),
            "st_first_unit_lean":   st_lean.get(team, 0.0),
            # Invert: lower penalty rate = higher discipline.
            "penalty_discipline":   -(pen_rate.get(team, 0.0)),
            "matching_intensity":   match_intensity.get(team, 0.0),
        })

    if not raw_rows:
        return pl.DataFrame(schema=COACH_DECISION_SCHEMA)

    # Percentile-rank each dimension across all coaches.
    for dim in DECISION_DIMENSIONS:
        vals = [r[dim] for r in raw_rows]
        ranks = _percentile_rank(vals)
        for i, rank in enumerate(ranks):
            raw_rows[i][dim] = round(rank, 4)

    # Overall aggression = mean of the 6 dimensions.
    for r in raw_rows:
        r["overall_aggression"] = round(
            sum(r[d] for d in DECISION_DIMENSIONS) / len(DECISION_DIMENSIONS), 4
        )
        r["model_version"] = MODEL_VERSION

    df = pl.DataFrame(raw_rows)
    for col, dtype in COACH_DECISION_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(COACH_DECISION_SCHEMA.keys())).sort("overall_aggression", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_coach_decision_net(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"coach_decision_net_{season}.parquet"
    df.write_parquet(path)
    return path


def read_coach_decision_net(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "coach_decision_net"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"coach_decision_net_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("coach_decision_net_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
