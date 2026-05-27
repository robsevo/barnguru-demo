"""Venue Atmosphere / Building Scare Factor — Feature 4.19.

Per-arena modifier stacked on standard home advantage, calibrated from
PBP data.  Visiting teams systematically underperform in certain
buildings; this model quantifies how much.

Components (per home-team building, per season)
-----------------------------------------------
1. **visiting_sv_delta**  — visiting goalie save% minus league-avg away SV%.
   Negative = goalies play worse here than at a typical away building.
2. **visiting_fow_delta** — visiting team faceoff win% minus 50%.
   Negative = visiting players lose more faceoffs here.
3. **ref_pp_delta**       — home PP opportunities/game minus away PP opps/game
   in this building.  Positive = refs favor the home team.
4. **visiting_xgf_delta** — visiting team xGF/60 minus their season-avg
   away xGF/60 when playing in this building.  Negative = offense
   suppressed.

**scare_factor (v3)** = weighted composite of five components:
    ``0.30 × z(crowd_intensity)
     + 0.20 × z(−visiting_xgf_delta_resid)
     + 0.20 × z(−visiting_sv_delta_resid)
     + 0.15 × z(−visiting_fow_delta_resid)
     + 0.15 × z(ref_pp_delta_resid)``

All components z-scored across the 32 buildings.  Higher scare_factor =
scarier building for visiting teams.

v2 — team-strength normalization
--------------------------------
The v1 deltas conflated venue effect with home-team quality.  v2
residualises each delta against the home team's overall goal
differential per game before z-scoring, so scare_factor reflects the
part of the delta that is NOT explained by the home team being better.

v3 — crowd / market reputation prior
------------------------------------
v2 still missed the obvious: hardcore hockey markets (MTL, MSG, TD
Garden, Bridgestone) are objectively more terrifying than non-hockey
markets (SJS, UTH, SEA).  v3 adds ``crowd_intensity`` from
``data/venue_reputation.json`` — editorial prior on a roughly z-scored
scale capturing Original Six religion, Canadian-market intensity,
known-loud barns, and the inverse for Sun Belt / recent expansion.
crowd_intensity carries the largest single weight (30%) because the
measured outcome deltas don't fully capture crowd impact on visitor
psyche.

Output: ``venue_atmosphere/venue_atmosphere_{season}.parquet``
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import polars as pl


MODEL_VERSION = "venue_atmosphere_v3"

_REPUTATION_PATH = Path(__file__).resolve().parents[1] / "data" / "venue_reputation.json"


def _load_crowd_intensity() -> dict[str, float]:
    """Load per-team crowd intensity priors from data/venue_reputation.json.

    Returns ``{team_abbrev: crowd_intensity}``.  Missing file or malformed
    entries fall back to 0.0 (league-average prior) with a loud warning.
    """
    if not _REPUTATION_PATH.exists():
        warnings.warn(
            f"venue_reputation.json not found at {_REPUTATION_PATH} — crowd prior disabled.",
            DataMissingWarning,
            stacklevel=2,
        )
        return {}
    try:
        blob = json.loads(_REPUTATION_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        warnings.warn(
            f"failed to parse venue_reputation.json: {e!r} — crowd prior disabled.",
            DataMissingWarning,
            stacklevel=2,
        )
        return {}
    out: dict[str, float] = {}
    for team, entry in (blob.get("venues") or {}).items():
        ci = entry.get("crowd_intensity")
        if isinstance(ci, (int, float)):
            out[team] = float(ci)
    return out


class DataMissingWarning(UserWarning):
    pass


VENUE_ATMOSPHERE_SCHEMA: dict[str, pl.DataType] = {
    "team":                 pl.Utf8,
    "season":               pl.Int64,
    "home_gp":              pl.Int64,
    "visiting_sv_delta":    pl.Float64,
    "visiting_fow_delta":   pl.Float64,
    "ref_pp_delta":         pl.Float64,
    "visiting_xgf_delta":   pl.Float64,
    "crowd_intensity":      pl.Float64,
    "scare_factor":         pl.Float64,
    "scare_rank":           pl.Float64,
    "model_version":        pl.Utf8,
}


def _z_scores(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / max(1, n)
    std  = var ** 0.5
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def _residualize(values: list[float], covariate: list[float]) -> list[float]:
    """Remove the linear effect of ``covariate`` from ``values`` via OLS.

    Returns the residuals (observed - predicted) where the prediction is a
    single-variable linear fit ``y = a·x + b``.  With <3 valid samples we
    return ``values`` unchanged.  Used to strip home-team-strength bias
    from each venue delta before composing scare_factor.
    """
    n = len(values)
    if n != len(covariate) or n < 3:
        return list(values)
    sx = sum(covariate)
    sy = sum(values)
    sxx = sum(x * x for x in covariate)
    sxy = sum(x * y for x, y in zip(covariate, values))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return list(values)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return [v - (slope * x + intercept) for v, x in zip(values, covariate)]


def _percentile_rank(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [0.5] * n
    valid = sorted(values)
    ranks: list[float] = []
    for v in values:
        lo = next(i for i, x in enumerate(valid) if x >= v)
        hi = n - 1 - next(i for i, x in enumerate(reversed(valid)) if x <= v)
        ranks.append((lo + hi) / 2.0 / max(1, n - 1))
    return ranks


def compute_venue_atmosphere(
    pbp_df:       pl.DataFrame,
    shots_df:     pl.DataFrame,
    team_lookup:  dict[int, str],
    season:       int,
) -> pl.DataFrame:
    """Build per-home-team venue atmosphere rows.

    Args:
        pbp_df:       raw PBP with faceoff/penalty/shot events.
        shots_df:     MoneyPuck shots with x_goal.
        team_lookup:  team_id → abbrev.
        season:       NHL season start year.
    """
    if pbp_df.is_empty():
        warnings.warn("compute_venue_atmosphere: empty PBP.", DataMissingWarning, stacklevel=2)
        return pl.DataFrame(schema=VENUE_ATMOSPHERE_SCHEMA)

    pbp_df = pbp_df.with_columns([
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("event_owner_team_id").cast(pl.Int64),
    ])

    # --- 1. Visiting goalie save% per building ---
    shot_events = pbp_df.filter(
        ((pl.col("event_type") == "shot") & (pl.col("shot_result") == "on_goal"))
        | (pl.col("event_type") == "goal")
    ).with_columns(
        pl.when(pl.col("event_owner_team_id") == pl.col("home_team_id"))
          .then(pl.col("away_team_id"))
          .otherwise(pl.col("home_team_id"))
          .alias("defending_team_id")
    )

    # Shots by home team on visiting goalie
    home_shots_on_visitor = shot_events.filter(
        pl.col("event_owner_team_id") == pl.col("home_team_id")
    )

    visitor_sv: dict[str, tuple[int, int]] = {}  # home_team → (shots_against_visitor, goals_against_visitor)
    for r in home_shots_on_visitor.group_by("home_team_id").agg([
        pl.len().alias("sa"),
        (pl.col("event_type") == "goal").sum().cast(pl.Int64).alias("ga"),
    ]).iter_rows(named=True):
        tid = int(r["home_team_id"])
        abbrev = team_lookup.get(tid)
        if abbrev:
            visitor_sv[abbrev] = (int(r["sa"]), int(r["ga"]))

    # League avg away save% (all visiting goalies across all buildings)
    total_sa = sum(v[0] for v in visitor_sv.values())
    total_ga = sum(v[1] for v in visitor_sv.values())
    league_away_sv = 1.0 - (total_ga / total_sa) if total_sa > 0 else 0.900

    # --- 2. Visiting FOW% per building ---
    faceoffs = pbp_df.filter(
        (pl.col("event_type") == "faceoff")
        & pl.col("winning_player_id").is_not_null()
    )

    # Determine if faceoff winner is home or away
    # winning_player_id maps to event_owner_team_id in our PBP schema
    # (the team that won the faceoff)
    fo_home_wins: dict[str, int] = {}
    fo_total: dict[str, int] = {}
    for r in faceoffs.group_by("home_team_id").agg([
        pl.len().alias("total"),
        (pl.col("event_owner_team_id") == pl.col("home_team_id")).sum().cast(pl.Int64).alias("home_wins"),
    ]).iter_rows(named=True):
        tid = int(r["home_team_id"])
        abbrev = team_lookup.get(tid)
        if abbrev:
            fo_total[abbrev] = int(r["total"])
            fo_home_wins[abbrev] = int(r["home_wins"])

    # --- 3. Ref PP differential per building ---
    penalties = pbp_df.filter(
        (pl.col("event_type") == "penalty")
        & pl.col("penalty_type").is_in(["MIN", "MAJ"])
        & pl.col("event_owner_team_id").is_not_null()
    )

    pp_home: dict[str, int] = {}
    pp_away: dict[str, int] = {}
    home_gp_map: dict[str, int] = {}

    games_per_home = (
        pbp_df.select(["game_id", "home_team_id"]).unique()
        .group_by("home_team_id").agg(pl.len().alias("gp"))
    )
    for r in games_per_home.iter_rows(named=True):
        tid = int(r["home_team_id"])
        abbrev = team_lookup.get(tid)
        if abbrev:
            home_gp_map[abbrev] = int(r["gp"])

    for r in penalties.group_by(["home_team_id", "event_owner_team_id"]).agg(
        pl.len().alias("cnt")
    ).iter_rows(named=True):
        home_tid = int(r["home_team_id"])
        offender = int(r["event_owner_team_id"])
        abbrev = team_lookup.get(home_tid)
        if not abbrev:
            continue
        cnt = int(r["cnt"])
        if offender != home_tid:
            # Away team took penalty → home team gets PP
            pp_home[abbrev] = pp_home.get(abbrev, 0) + cnt
        else:
            pp_away[abbrev] = pp_away.get(abbrev, 0) + cnt

    # --- 4. Team strength proxy — overall goal differential per game.
    #        Used as a covariate to residualise venue deltas (v2).
    #        Computed across BOTH home and away games so it isn't confounded
    #        with home-ice effects we're trying to isolate.
    team_gf:  dict[str, int] = {}
    team_ga:  dict[str, int] = {}
    team_gp:  dict[str, int] = {}
    goals_only = pbp_df.filter(pl.col("event_type") == "goal").select(
        ["home_team_id", "away_team_id", "event_owner_team_id"]
    )
    for r in goals_only.iter_rows(named=True):
        h = int(r["home_team_id"]); a = int(r["away_team_id"]); o = int(r["event_owner_team_id"])
        h_ab = team_lookup.get(h); a_ab = team_lookup.get(a)
        if not h_ab or not a_ab:
            continue
        if o == h:
            team_gf[h_ab] = team_gf.get(h_ab, 0) + 1
            team_ga[a_ab] = team_ga.get(a_ab, 0) + 1
        elif o == a:
            team_gf[a_ab] = team_gf.get(a_ab, 0) + 1
            team_ga[h_ab] = team_ga.get(h_ab, 0) + 1
    games_per_team = (
        pbp_df.select(["game_id", "home_team_id", "away_team_id"]).unique()
    )
    for r in games_per_team.iter_rows(named=True):
        h_ab = team_lookup.get(int(r["home_team_id"]))
        a_ab = team_lookup.get(int(r["away_team_id"]))
        if h_ab: team_gp[h_ab] = team_gp.get(h_ab, 0) + 1
        if a_ab: team_gp[a_ab] = team_gp.get(a_ab, 0) + 1

    # --- 5. Visiting xGF delta per building (from MoneyPuck shots) ---
    visiting_xgf: dict[str, float] = {}
    visiting_shots_count: dict[str, int] = {}
    if not shots_df.is_empty() and {"home_team", "away_team", "shooting_team", "x_goal"}.issubset(shots_df.columns):
        # Away team shooting at this building
        away_shots = shots_df.filter(pl.col("shooting_team") == pl.col("away_team"))
        for r in away_shots.group_by("home_team").agg([
            pl.col("x_goal").sum().alias("xgf"),
            pl.len().alias("cnt"),
        ]).iter_rows(named=True):
            home = str(r["home_team"])
            visiting_xgf[home] = float(r["xgf"])
            visiting_shots_count[home] = int(r["cnt"])

    # --- Build rows ---
    all_teams = sorted(set(home_gp_map.keys()))
    rows: list[dict[str, Any]] = []
    for team in all_teams:
        gp = home_gp_map.get(team, 0)
        if gp == 0:
            continue

        # 1. Visiting SV% delta
        sa, ga = visitor_sv.get(team, (0, 0))
        local_sv = 1.0 - (ga / sa) if sa > 0 else league_away_sv
        sv_delta = local_sv - league_away_sv

        # 2. Visiting FOW% delta
        ft = fo_total.get(team, 0)
        fhw = fo_home_wins.get(team, 0)
        visiting_fow = 1.0 - (fhw / ft) if ft > 0 else 0.5
        fow_delta = visiting_fow - 0.5

        # 3. Ref PP differential per game
        h_pp = pp_home.get(team, 0) / gp if gp else 0
        a_pp = pp_away.get(team, 0) / gp if gp else 0
        pp_delta = h_pp - a_pp

        # 4. Visiting xGF/60 delta (vs league avg away xGF/60)
        v_xgf = visiting_xgf.get(team, 0.0)
        v_cnt = visiting_shots_count.get(team, 0)
        xgf_per_shot = v_xgf / v_cnt if v_cnt > 0 else 0.0
        # League avg visiting xGF/shot
        total_v_xgf = sum(visiting_xgf.values())
        total_v_cnt = sum(visiting_shots_count.values())
        lg_xgf_per_shot = total_v_xgf / total_v_cnt if total_v_cnt > 0 else xgf_per_shot
        xgf_delta = xgf_per_shot - lg_xgf_per_shot

        rows.append({
            "team":               team,
            "season":             int(season),
            "home_gp":            gp,
            "visiting_sv_delta":  round(sv_delta, 5),
            "visiting_fow_delta": round(fow_delta, 4),
            "ref_pp_delta":       round(pp_delta, 4),
            "visiting_xgf_delta": round(xgf_delta, 5),
            "crowd_intensity":    0.0,  # filled in below
        })

    if not rows:
        return pl.DataFrame(schema=VENUE_ATMOSPHERE_SCHEMA)

    # v2 normalization — residualise each delta against the home team's
    # goal differential per game so scare_factor isolates the venue-only
    # signal. League-average team strength (≈0) is used when a team has
    # no GP recorded (shouldn't happen but safe).
    strength = [
        (team_gf.get(r["team"], 0) - team_ga.get(r["team"], 0)) / max(1, team_gp.get(r["team"], 0))
        for r in rows
    ]
    sv_resid  = _residualize([-r["visiting_sv_delta"]  for r in rows], strength)
    fow_resid = _residualize([-r["visiting_fow_delta"] for r in rows], strength)
    pp_resid  = _residualize([ r["ref_pp_delta"]       for r in rows], strength)
    xgf_resid = _residualize([-r["visiting_xgf_delta"] for r in rows], strength)

    # v3 — crowd intensity prior (Original Six + Canadian + known-loud
    # barns score high; Sun Belt + recent expansion score low). Pulled
    # from data/venue_reputation.json so the editorial layer is checked-in
    # and reviewable.
    crowd_lut = _load_crowd_intensity()
    for r in rows:
        r["crowd_intensity"] = round(float(crowd_lut.get(r["team"], 0.0)), 3)
    crowd_z = _z_scores([r["crowd_intensity"] for r in rows])

    # Compute scare_factor from z-scored residuals + crowd prior.
    # Weights tuned so the editorial crowd component dominates (30%) and
    # the four outcome residuals share the remaining 70%.
    z_sv  = _z_scores(sv_resid)
    z_fow = _z_scores(fow_resid)
    z_pp  = _z_scores(pp_resid)
    z_xgf = _z_scores(xgf_resid)

    for i, r in enumerate(rows):
        r["scare_factor"] = round(
            0.30 * crowd_z[i]
            + 0.20 * z_xgf[i]
            + 0.20 * z_sv[i]
            + 0.15 * z_fow[i]
            + 0.15 * z_pp[i],
            4,
        )

    scare_vals = [r["scare_factor"] for r in rows]
    scare_ranks = _percentile_rank(scare_vals)
    for i, r in enumerate(rows):
        r["scare_rank"] = round(scare_ranks[i], 4)
        r["model_version"] = MODEL_VERSION

    df = pl.DataFrame(rows)
    for col, dtype in VENUE_ATMOSPHERE_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(VENUE_ATMOSPHERE_SCHEMA.keys())).sort("scare_factor", descending=True)


def write_venue_atmosphere(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"venue_atmosphere_{season}.parquet"
    df.write_parquet(path)
    return path


def read_venue_atmosphere(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "venue_atmosphere"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"venue_atmosphere_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("venue_atmosphere_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
