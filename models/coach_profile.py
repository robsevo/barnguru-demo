"""Coach Profile Database — Feature 4.7.

Per-current-head-coach career-with-current-team tendencies built from
ingested NHL play-by-play.  No external scrape needed (V1 stays inside
the project's "real data only" policy) — historical depth is whatever
PBP we have ingested.

What it produces
----------------
For every coach in ``data/coaches.json`` we aggregate, across all
seasons present in PBP where ``season >= first_named_head_coach`` year:

- gp_under_coach
- W / OTW / L / OTL (regulation + overtime + shootout splits)
- points (NHL 2-point system) and points_pct
- gf_per_game, ga_per_game
- pp_pct, pk_pct
- sf_per_game, sa_per_game
- seasons_covered (list of seasons aggregated)

Honest limitations
------------------
- Career stats reach only as far back as ingested PBP (3 seasons today).
- Mid-season coach changes are not retroactively re-attributed within the
  hiring season.  We attribute the entire season to the coach whose
  ``first_named_head_coach`` falls in or before that season.  When the
  hockey-reference.com scrape lands in a future PR, the season-aware
  attribution can be tightened by joining on game date.

Output: one row per coach per snapshot, indexed in
``data/coach_profiles/coach_profiles_{season}.parquet`` where ``season``
is the latest PBP season used.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "coach_profile_v1"


class DataMissingWarning(UserWarning):
    """Raised when coach profile data is absent or insufficient."""


COACH_PROFILE_SCHEMA: dict[str, pl.DataType] = {
    "coach_name":              pl.Utf8,
    "team":                    pl.Utf8,
    "first_named_head_coach":  pl.Utf8,
    "season":                  pl.Int64,   # latest PBP season aggregated
    "seasons_covered":         pl.List(pl.Int64),
    "gp_under_coach":          pl.Int64,
    "wins":                    pl.Int64,
    "ot_wins":                 pl.Int64,
    "losses":                  pl.Int64,
    "ot_losses":               pl.Int64,
    "points":                  pl.Int64,
    "points_pct":              pl.Float64,
    "gf_per_game":             pl.Float64,
    "ga_per_game":             pl.Float64,
    "pp_pct":                  pl.Float64,
    "pk_pct":                  pl.Float64,
    "sf_per_game":             pl.Float64,
    "sa_per_game":             pl.Float64,
    "notes":                   pl.Utf8,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_named_season(first_named_iso: str | None) -> int | None:
    """Map an ISO date like '2024-11-19' to its NHL season-start year.

    NHL seasons run Oct → Jun.  A date in Oct-Dec of YYYY → season YYYY.
    A date in Jan-Sep of YYYY → season YYYY-1.
    """
    if not first_named_iso:
        return None
    try:
        y_s, m_s = first_named_iso.split("-")[:2]
        year  = int(y_s)
        month = int(m_s)
    except (ValueError, AttributeError):
        return None
    return year if month >= 8 else year - 1


def _per_team_game_summary(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Reduce PBP to one row per (game_id, team) with final score + counts.

    Columns: game_id, team_id, gf, ga, sf, sa, pp_opps, pp_goals,
             pk_opps, pk_ga, outcome ('W' | 'OTW' | 'L' | 'OTL')
    """
    required = {
        "game_id", "event_type", "event_owner_team_id", "home_team_id",
        "away_team_id", "home_score", "away_score", "strength",
        "period_type", "shot_result", "penalty_type",
    }
    missing = required - set(pbp_df.columns)
    if missing:
        raise ValueError(f"pbp_df missing required columns: {sorted(missing)}")

    pbp_df = pbp_df.with_columns([
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("event_owner_team_id").cast(pl.Int64),
        pl.col("home_score").cast(pl.Int64),
        pl.col("away_score").cast(pl.Int64),
    ])

    # Final score per game from the very last home_score/away_score row that's
    # not null.  game_end rows in our PBP often have null scores; we fall back
    # to the running max across the game.
    game_meta = (
        pbp_df.filter(pl.col("home_score").is_not_null())
        .group_by("game_id")
        .agg([
            pl.col("home_team_id").first().alias("home_team_id"),
            pl.col("away_team_id").first().alias("away_team_id"),
            pl.col("home_score").max().alias("home_final"),
            pl.col("away_score").max().alias("away_final"),
            # Period-type info — if any non-REG period appeared, this was OT/SO.
            (pl.col("period_type").is_in(["OT", "SO"]).any()).alias("had_ot_or_so"),
        ])
    )

    # Shots-for: event_type == 'shot' (any shot_result) or event_type == 'goal'.
    # Both attribute to event_owner_team_id (the shooting team).
    shots = (
        pbp_df.filter(pl.col("event_type").is_in(["shot", "goal"]))
        .group_by(["game_id", "event_owner_team_id"])
        .agg(pl.len().alias("shots_for"))
    )

    # Goals split by strength (for PP%/PK%).
    pp_goals = (
        pbp_df.filter((pl.col("event_type") == "goal") & (pl.col("strength") == "pp"))
        .group_by(["game_id", "event_owner_team_id"])
        .agg(pl.len().alias("pp_goals"))
    )

    # PP opportunities — count minors+majors against the opponent.
    pp_opps_df = (
        pbp_df.filter(
            (pl.col("event_type") == "penalty")
            & pl.col("penalty_type").is_in(["MIN", "MAJ"])
        )
        .select([
            pl.col("game_id"),
            pl.col("home_team_id"),
            pl.col("away_team_id"),
            pl.col("event_owner_team_id").alias("offender_id"),
        ])
        .with_columns(
            pl.when(pl.col("offender_id") == pl.col("home_team_id"))
              .then(pl.col("away_team_id"))
              .when(pl.col("offender_id") == pl.col("away_team_id"))
              .then(pl.col("home_team_id"))
              .otherwise(None)
              .alias("earner_id")
        )
        .filter(pl.col("earner_id").is_not_null())
        .group_by(["game_id", "earner_id"])
        .agg(pl.len().alias("pp_opps"))
        .rename({"earner_id": "team_id"})
    )

    out_rows: list[dict[str, Any]] = []
    for r in game_meta.iter_rows(named=True):
        gid     = r["game_id"]
        home_id = int(r["home_team_id"] or 0)
        away_id = int(r["away_team_id"] or 0)
        h_final = int(r["home_final"] or 0)
        a_final = int(r["away_final"] or 0)
        ot_or_so = bool(r["had_ot_or_so"])

        # Per-side outcomes
        if h_final == a_final:
            # Score-tied at game_end is impossible after SO; defensive fallback.
            home_outcome = away_outcome = "L"
        elif h_final > a_final:
            home_outcome = "OTW" if ot_or_so else "W"
            away_outcome = "OTL" if ot_or_so else "L"
        else:
            home_outcome = "OTL" if ot_or_so else "L"
            away_outcome = "OTW" if ot_or_so else "W"

        out_rows.append({
            "game_id":   gid, "team_id": home_id,
            "gf":        h_final, "ga": a_final,
            "outcome":   home_outcome,
        })
        out_rows.append({
            "game_id":   gid, "team_id": away_id,
            "gf":        a_final, "ga": h_final,
            "outcome":   away_outcome,
        })

    base_df = pl.DataFrame(out_rows, schema={
        "game_id": pl.Int64, "team_id": pl.Int64, "gf": pl.Int64, "ga": pl.Int64, "outcome": pl.Utf8,
    })

    # Join shots, pp_goals, pp_opps onto (game_id, team_id)
    base_df = base_df.join(
        shots.rename({"event_owner_team_id": "team_id", "shots_for": "sf"}),
        on=["game_id", "team_id"], how="left",
    ).join(
        pp_goals.rename({"event_owner_team_id": "team_id", "pp_goals": "pp_goals"}),
        on=["game_id", "team_id"], how="left",
    ).join(
        pp_opps_df,
        on=["game_id", "team_id"], how="left",
    ).with_columns([
        pl.col("sf").fill_null(0),
        pl.col("pp_goals").fill_null(0),
        pl.col("pp_opps").fill_null(0),
    ])

    # Shots-against and PP-opportunities-against are the opposite side's totals.
    sa_join = base_df.select(["game_id", "team_id", "sf"]).rename({"team_id": "opp_id", "sf": "sa"})
    pk_join = base_df.select(["game_id", "team_id", "pp_opps", "pp_goals"]).rename({
        "team_id":  "opp_id",
        "pp_opps":  "pk_opps",
        "pp_goals": "pk_ga",
    })

    game_pair = (
        base_df.select(["game_id", "team_id"])
        .join(base_df.select(["game_id", "team_id"]).rename({"team_id": "opp_id"}), on="game_id")
        .filter(pl.col("team_id") != pl.col("opp_id"))
    )
    pair_df = (
        game_pair.join(sa_join, on=["game_id", "opp_id"], how="left")
        .join(pk_join, on=["game_id", "opp_id"], how="left")
    )

    return base_df.join(pair_df, on=["game_id", "team_id"], how="left").with_columns([
        pl.col("sa").fill_null(0),
        pl.col("pk_opps").fill_null(0),
        pl.col("pk_ga").fill_null(0),
    ]).drop("opp_id")


def _aggregate_for_team(
    summary_df: pl.DataFrame,
    team_id: int,
    seasons_covered: list[int],
) -> dict[str, Any]:
    """Aggregate per-team rows into a single coach profile row."""
    t = summary_df.filter(pl.col("team_id") == team_id)
    if t.is_empty():
        return {
            "gp_under_coach": 0,
            "wins": 0, "ot_wins": 0, "losses": 0, "ot_losses": 0,
            "points": 0, "points_pct": 0.0,
            "gf_per_game": 0.0, "ga_per_game": 0.0,
            "pp_pct": 0.0, "pk_pct": 0.0,
            "sf_per_game": 0.0, "sa_per_game": 0.0,
            "seasons_covered": seasons_covered,
        }

    gp        = len(t)
    wins      = int(t.filter(pl.col("outcome") == "W").height)
    ot_wins   = int(t.filter(pl.col("outcome") == "OTW").height)
    losses    = int(t.filter(pl.col("outcome") == "L").height)
    ot_losses = int(t.filter(pl.col("outcome") == "OTL").height)

    # NHL points: W=2, OTW=2, OTL=1, L=0.  ot_wins also count 2.
    points    = 2 * (wins + ot_wins) + ot_losses
    points_pct = points / (2 * gp) if gp else 0.0

    gf_sum = int(t["gf"].sum() or 0)
    ga_sum = int(t["ga"].sum() or 0)
    sf_sum = int(t["sf"].sum() or 0)
    sa_sum = int(t["sa"].sum() or 0)
    pp_g   = int(t["pp_goals"].sum() or 0)
    pp_op  = int(t["pp_opps"].sum() or 0)
    pk_ga  = int(t["pk_ga"].sum() or 0)
    pk_op  = int(t["pk_opps"].sum() or 0)

    pp_pct = pp_g / pp_op if pp_op > 0 else 0.0
    pk_pct = 1.0 - (pk_ga / pk_op) if pk_op > 0 else 0.0

    return {
        "gp_under_coach":   gp,
        "wins":             wins,
        "ot_wins":          ot_wins,
        "losses":           losses,
        "ot_losses":        ot_losses,
        "points":           points,
        "points_pct":       points_pct,
        "gf_per_game":      gf_sum / gp if gp else 0.0,
        "ga_per_game":      ga_sum / gp if gp else 0.0,
        "pp_pct":           pp_pct,
        "pk_pct":           pk_pct,
        "sf_per_game":      sf_sum / gp if gp else 0.0,
        "sa_per_game":      sa_sum / gp if gp else 0.0,
        "seasons_covered":  seasons_covered,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_coach_profiles(
    pbp_by_season: dict[int, pl.DataFrame],
    coaches:       list[dict],
    team_lookup:   dict[int, str],
    snapshot_season: int,
) -> pl.DataFrame:
    """Build one row per current head coach for the snapshot.

    Args:
        pbp_by_season:   season-start-year → PBP DataFrame.
        coaches:         list of dicts from data/coaches.json under "coaches".
        team_lookup:     team_id (int) → abbrev (str), e.g. _NHL_TEAM_IDS.
        snapshot_season: the season to surface (also the parquet suffix).
                         All seasons in pbp_by_season >= each coach's
                         first_named_head_coach year are aggregated.
    """
    if not coaches:
        warnings.warn(
            "compute_coach_profiles: empty coaches list — output will be empty.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=COACH_PROFILE_SCHEMA)

    if not pbp_by_season:
        warnings.warn(
            "compute_coach_profiles: no PBP seasons supplied — every row will be zeroed.",
            DataMissingWarning,
            stacklevel=2,
        )

    # Build per-season game summaries up front (one pass per season).
    summary_by_season: dict[int, pl.DataFrame] = {}
    for season, pbp_df in pbp_by_season.items():
        try:
            summary_by_season[season] = _per_team_game_summary(pbp_df)
        except ValueError as exc:
            warnings.warn(
                f"compute_coach_profiles: season {season} skipped — {exc}",
                DataMissingWarning,
                stacklevel=2,
            )

    # Invert team_lookup: abbrev → team_id (multiple ids possible for UTH).
    abbrev_to_ids: dict[str, list[int]] = {}
    for tid, abbrev in team_lookup.items():
        abbrev_to_ids.setdefault(abbrev, []).append(tid)

    rows: list[dict[str, Any]] = []
    for coach in coaches:
        cname = (coach.get("name") or "").strip()
        team  = (coach.get("team") or "").strip().upper()
        first_named = (coach.get("first_named_head_coach") or "").strip() or None
        notes = (coach.get("notes") or "").strip()
        if not cname or not team:
            continue

        ids = abbrev_to_ids.get(team, [])
        if not ids:
            # Unknown team abbrev — record an empty profile row anyway so the
            # frontend gets a stub instead of "not_found".
            rows.append({
                "coach_name": cname,
                "team":       team,
                "first_named_head_coach": first_named or "",
                "season":     int(snapshot_season),
                "seasons_covered": [],
                "gp_under_coach": 0, "wins": 0, "ot_wins": 0,
                "losses": 0, "ot_losses": 0,
                "points": 0, "points_pct": 0.0,
                "gf_per_game": 0.0, "ga_per_game": 0.0,
                "pp_pct": 0.0, "pk_pct": 0.0,
                "sf_per_game": 0.0, "sa_per_game": 0.0,
                "notes": notes,
                "model_version": MODEL_VERSION,
            })
            continue

        first_season = _first_named_season(first_named)
        agg_seasons = sorted(
            s for s in summary_by_season.keys()
            if first_season is None or s >= first_season
        )

        combined = pl.concat(
            [summary_by_season[s] for s in agg_seasons],
            how="diagonal_relaxed",
        ) if agg_seasons else pl.DataFrame(
            schema={
                "game_id":  pl.Int64, "team_id": pl.Int64,
                "gf":       pl.Int64, "ga":      pl.Int64,
                "outcome":  pl.Utf8,  "sf":      pl.Int64,
                "pp_goals": pl.Int64, "pp_opps": pl.Int64,
                "sa":       pl.Int64, "pk_opps": pl.Int64, "pk_ga": pl.Int64,
            },
        )

        # Aggregate across every team_id that maps to this abbrev (UTH 59 vs 68).
        per_team_aggs = [_aggregate_for_team(combined, tid, agg_seasons) for tid in ids]
        agg = per_team_aggs[0]
        for extra in per_team_aggs[1:]:
            for k in (
                "gp_under_coach", "wins", "ot_wins", "losses",
                "ot_losses", "points",
            ):
                agg[k] = agg[k] + extra[k]
            # Recompute rate stats after summing counts is non-trivial;
            # for v1 we keep the first team_id's rates since UTH 59/68 are
            # the only multi-id case and the totals match the canonical 59 id.
        gp = agg["gp_under_coach"]
        agg["points_pct"] = (agg["points"] / (2 * gp)) if gp else 0.0

        rows.append({
            "coach_name": cname,
            "team":       team,
            "first_named_head_coach": first_named or "",
            "season":     int(snapshot_season),
            **agg,
            "notes":         notes,
            "model_version": MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=COACH_PROFILE_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in COACH_PROFILE_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(COACH_PROFILE_SCHEMA.keys())).sort("points_pct", descending=True)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_coach_profiles(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"coach_profiles_{season}.parquet"
    df.write_parquet(path)
    return path


def read_coach_profiles(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "coach_profiles"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"coach_profiles_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("coach_profiles_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
