"""Line-Matching Model — Feature 4.2.

Predicts defensive counter-deployment: when an opponent's top line is on
the ice, which of your own lines does the coach put out against them?

This captures the **last-change advantage** (home coach gets the final
on-ice line decision) plus per-coach style — defensive coaches like
Trotz / Cooper run hard matchups against opposing L1; offensive coaches
like McLellan lean on rolling four lines.

Approach (V1 — interpretable)
-----------------------------
1. Resolve each team's assembled lines from Feature 4.1's output
   (forward L1–L4 + D-pair D1–D3).
2. For each 5v5 stint where both teams had 5 skaters on ice, identify the
   "majority" line on each side: the line with the most of its assigned
   players currently on the ice (≥ 2 of 3 for forwards, ≥ 1 of 2 for D).
3. Aggregate stint duration by (home_team, away_team, home_line_rank,
   away_line_rank) for both line_type=F and line_type=D.
4. Normalize to per-team **matchup share**: for each (team, opponent,
   own_line_rank), the share of opponent's L1 TOI absorbed by each of
   own L1–L4.  Mirror metric for D-pairs.
5. Compute a **last-change advantage delta**: how often does the home
   team's L1 face the road team's L1 vs. the same matchup with venues
   reversed?  Positive → home heavily matches.

Output: per-coach (team-season) matchup profile.  Drives the Rust line
change model (5.10) and per-coach decision net (4.17).

Limitations
-----------
- "Majority on ice" heuristic — when a coach uses split-line shuffling
  (mid-shift swap), the dominant line is still tagged.  V2 could weight
  by player-share rather than majority.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION    = "line_matching_v1"
MIN_FWD_OVERLAP  = 2    # at least 2 of 3 forwards present to count as "L_k on ice"
MIN_DEF_OVERLAP  = 1    # at least 1 of 2 defensemen present to count as "D_k on ice"
MIN_PAIR_SECS    = 30.0 # require ≥ 30s aggregated TOI in a matchup pair to keep row


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

LINE_MATCHING_SCHEMA: dict[str, pl.DataType] = {
    "team":              pl.Utf8,    # focal team (the "own" side)
    "opponent":          pl.Utf8,    # opposing team
    "season":            pl.Int64,
    "line_type":         pl.Utf8,    # "F" or "D"
    "own_line_rank":     pl.Int64,
    "opp_line_rank":     pl.Int64,
    "venue":             pl.Utf8,    # "home" or "away"
    "matchup_toi_secs":  pl.Float64,
    "share_of_opp_line": pl.Float64, # P(this line out | opp line on ice)
    "model_version":     pl.Utf8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lines_for_team(
    line_dep: pl.DataFrame,
    team: str,
    line_type: str,
    season: int,
) -> dict[int, set[int]]:
    """Return {line_rank → set(player_ids)} from line deployment output."""
    df = line_dep.filter(
        (pl.col("team") == team)
        & (pl.col("line_type") == line_type)
        & (pl.col("season") == season)
    )
    out: dict[int, set[int]] = {}
    for row in df.iter_rows(named=True):
        members = {int(row["player_1"])}
        if row["player_2"] is not None:
            members.add(int(row["player_2"]))
        if row["player_3"] is not None:
            members.add(int(row["player_3"]))
        out[int(row["line_rank"])] = members
    return out


def _majority_line(
    on_ice: set[int],
    lines:  dict[int, set[int]],
    min_overlap: int,
) -> int | None:
    """Return the line_rank with the most members present on ice
    (ties broken by lower rank, i.e. higher-priority line)."""
    best_rank: int | None = None
    best_overlap = 0
    for rank in sorted(lines.keys()):  # lower rank wins ties
        n = len(on_ice & lines[rank])
        if n > best_overlap:
            best_overlap = n
            best_rank    = rank
    if best_overlap < min_overlap:
        return None
    return best_rank


# ---------------------------------------------------------------------------
# Stint pairing
# ---------------------------------------------------------------------------


def _paired_stints(
    shifts_df: pl.DataFrame,
    pbp_df:    pl.DataFrame,
    team_lookup: dict[int, str],
) -> pl.DataFrame:
    """Build per-game stints listing on-ice player_ids for BOTH home and
    away teams simultaneously (only 5v5 EV intervals).

    Returns columns:
        game_id, home_team_abbrev, away_team_abbrev,
        duration_secs, home_pids (List[Int64]), away_pids (List[Int64]).
    """
    required = {"game_id", "player_id", "team_id",
                "game_seconds_start", "game_seconds_end"}
    if not required.issubset(shifts_df.columns):
        return _empty_paired()
    if "game_id" not in pbp_df.columns:
        return _empty_paired()

    shifts = shifts_df.with_columns([
        pl.col("game_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64),
        pl.col("player_id").cast(pl.Int64),
        pl.col("game_seconds_start").cast(pl.Float64),
        pl.col("game_seconds_end").cast(pl.Float64),
    ])
    pbp = pbp_df.with_columns(pl.col("game_id").cast(pl.Int64))
    known_ids: set[int] | None = set(team_lookup.keys()) if team_lookup else None

    rows: list[dict] = []
    game_ids = shifts["game_id"].unique().to_list()

    for gid in game_ids:
        g = shifts.filter(pl.col("game_id") == gid)
        if g.is_empty():
            continue
        pg = pbp.filter(pl.col("game_id") == gid)
        home_id, away_id = _home_away_ids(pg)
        if home_id is None or away_id is None:
            continue
        if known_ids is not None and (home_id not in known_ids or away_id not in known_ids):
            continue

        # Build skater-count intervals from pbp
        ev_ranges = _ev_5v5_intervals(pg)
        if ev_ranges is None or ev_ranges.size == 0:
            continue

        # Drop goalie shift rows so on-ice counts represent skaters only
        goalies = _goalies_in_game(pg)
        if goalies:
            g = g.filter(~pl.col("player_id").is_in(list(goalies)))

        # Get all shift breakpoints
        starts = g["game_seconds_start"].to_numpy()
        ends   = g["game_seconds_end"].to_numpy()
        pids   = g["player_id"].to_numpy()
        tids   = g["team_id"].to_numpy()

        breakpoints = np.unique(np.concatenate([starts, ends]))
        if len(breakpoints) < 2:
            continue

        home_abbrev = team_lookup.get(home_id, str(home_id))
        away_abbrev = team_lookup.get(away_id, str(away_id))

        for i in range(len(breakpoints) - 1):
            t0, t1 = float(breakpoints[i]), float(breakpoints[i + 1])
            dur = t1 - t0
            if dur < 1.0:
                continue
            if not _interval_overlaps(t0, t1, ev_ranges):
                continue

            on_mask = (starts <= t0) & (ends >= t1)
            on_pids = pids[on_mask]
            on_tids = tids[on_mask]
            home_set = on_pids[on_tids == home_id].tolist()
            away_set = on_pids[on_tids == away_id].tolist()
            if len(home_set) != 5 or len(away_set) != 5:
                continue

            rows.append({
                "game_id":       int(gid),
                "home_team":     home_abbrev,
                "away_team":     away_abbrev,
                "duration_secs": dur,
                "home_pids":     [int(p) for p in home_set],
                "away_pids":     [int(p) for p in away_set],
            })

    if not rows:
        return _empty_paired()

    return pl.DataFrame(rows).with_columns([
        pl.col("home_pids").cast(pl.List(pl.Int64)),
        pl.col("away_pids").cast(pl.List(pl.Int64)),
    ])


def _empty_paired() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "game_id":       pl.Int64,
        "home_team":     pl.Utf8,
        "away_team":     pl.Utf8,
        "duration_secs": pl.Float64,
        "home_pids":     pl.List(pl.Int64),
        "away_pids":     pl.List(pl.Int64),
    })


def _home_away_ids(pbp_g: pl.DataFrame) -> tuple[int | None, int | None]:
    if pbp_g.is_empty():
        return None, None
    if not {"home_team_id", "away_team_id"}.issubset(pbp_g.columns):
        return None, None
    h = pbp_g["home_team_id"].drop_nulls()
    a = pbp_g["away_team_id"].drop_nulls()
    if h.is_empty() or a.is_empty():
        return None, None
    return int(h[0]), int(a[0])


def _goalies_in_game(pbp_g: pl.DataFrame) -> set[int]:
    out: set[int] = set()
    if "shot_goalie_id" in pbp_g.columns:
        vals = pbp_g["shot_goalie_id"].drop_nulls().unique().to_list()
        for v in vals:
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                pass
    return out


# ---- copied from line_deployment to avoid circular dep on private helpers ---


def _ev_5v5_intervals(pbp_g: pl.DataFrame) -> np.ndarray | None:
    needed = {"period", "time_in_period_secs", "home_skaters", "away_skaters"}
    if not needed.issubset(pbp_g.columns):
        return None
    if pbp_g.is_empty():
        return None
    df = pbp_g.select([
        "period", "time_in_period_secs", "home_skaters", "away_skaters",
    ]).drop_nulls()
    if df.is_empty():
        return None
    df = df.with_columns(
        ((pl.col("period") - 1) * 1200.0 + pl.col("time_in_period_secs").cast(pl.Float64)).alias("gs")
    ).sort("gs")
    gs = df["gs"].to_numpy()
    hs = df["home_skaters"].to_numpy()
    aw = df["away_skaters"].to_numpy()
    intervals: list[tuple[float, float]] = []
    cur_start: float | None = None
    for i in range(len(gs)):
        ev = (hs[i] == 5 and aw[i] == 5)
        if ev:
            if cur_start is None:
                cur_start = float(gs[i])
        else:
            if cur_start is not None:
                intervals.append((cur_start, float(gs[i])))
                cur_start = None
    if cur_start is not None:
        intervals.append((cur_start, float("inf")))
    if not intervals:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(intervals, dtype=np.float64)


def _interval_overlaps(t0: float, t1: float, ranges: np.ndarray) -> bool:
    if ranges.size == 0:
        return False
    return bool(((ranges[:, 0] < t1) & (ranges[:, 1] > t0)).any())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_line_matching(
    shifts_df:   pl.DataFrame,
    pbp_df:      pl.DataFrame,
    line_dep_df: pl.DataFrame,
    season:      int,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Compute per-(team, opponent, line_rank pair) matchup TOI shares.

    Args:
        shifts_df:    shift records for the season.
        pbp_df:       play-by-play data for the season.
        line_dep_df:  Line deployment output (Feature 4.1) for the same season.
        season:       e.g. 2025.
        team_lookup:  team_id (Int64) → abbrev (str) mapping required so we
                      can match into line_dep_df by team abbreviation.

    Returns:
        Polars DataFrame matching LINE_MATCHING_SCHEMA.  Empty if inputs
        are insufficient.
    """
    if team_lookup is None:
        team_lookup = {}

    if shifts_df.is_empty() or line_dep_df.is_empty():
        return pl.DataFrame(schema=LINE_MATCHING_SCHEMA)

    paired = _paired_stints(shifts_df, pbp_df, team_lookup)
    if paired.is_empty():
        return pl.DataFrame(schema=LINE_MATCHING_SCHEMA)

    # Precompute line dicts per team
    line_dep_season = line_dep_df.filter(pl.col("season") == season)
    teams_in_dep = line_dep_season["team"].unique().to_list()
    f_lines: dict[str, dict[int, set[int]]] = {
        t: _lines_for_team(line_dep_season, t, "F", season) for t in teams_in_dep
    }
    d_pairs: dict[str, dict[int, set[int]]] = {
        t: _lines_for_team(line_dep_season, t, "D", season) for t in teams_in_dep
    }

    # Aggregator: (focal_team, opponent, line_type, focal_rank, opp_rank, venue)
    agg: dict[tuple[str, str, str, int, int, str], float] = defaultdict(float)

    for row in paired.iter_rows(named=True):
        h = row["home_team"]
        a = row["away_team"]
        dur = float(row["duration_secs"])
        home_set = set(row["home_pids"])
        away_set = set(row["away_pids"])

        # Identify majority lines on each side
        h_f = f_lines.get(h, {})
        a_f = f_lines.get(a, {})
        h_d = d_pairs.get(h, {})
        a_d = d_pairs.get(a, {})

        h_fline = _majority_line(home_set, h_f, MIN_FWD_OVERLAP)
        a_fline = _majority_line(away_set, a_f, MIN_FWD_OVERLAP)
        h_dpair = _majority_line(home_set, h_d, MIN_DEF_OVERLAP)
        a_dpair = _majority_line(away_set, a_d, MIN_DEF_OVERLAP)

        # Record one row per side per matchup pair
        if h_fline is not None and a_fline is not None:
            agg[(h, a, "F", h_fline, a_fline, "home")] += dur
            agg[(a, h, "F", a_fline, h_fline, "away")] += dur
        if h_dpair is not None and a_dpair is not None:
            agg[(h, a, "D", h_dpair, a_dpair, "home")] += dur
            agg[(a, h, "D", a_dpair, h_dpair, "away")] += dur

    if not agg:
        return pl.DataFrame(schema=LINE_MATCHING_SCHEMA)

    # Build DF
    rows: list[dict] = []
    for (team, opp, ltype, own_rank, opp_rank, venue), toi in agg.items():
        if toi < MIN_PAIR_SECS:
            continue
        rows.append({
            "team":              team,
            "opponent":          opp,
            "season":            int(season),
            "line_type":         ltype,
            "own_line_rank":     int(own_rank),
            "opp_line_rank":     int(opp_rank),
            "venue":             venue,
            "matchup_toi_secs":  toi,
            "share_of_opp_line": np.nan,  # filled below
            "model_version":     MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=LINE_MATCHING_SCHEMA)

    df = pl.DataFrame(rows)

    # Compute share within (team, opponent, line_type, opp_line_rank, venue):
    # i.e. of total TOI when opponent's L_k was on ice, what fraction
    # was each of our lines deployed?
    df = df.with_columns(
        (pl.col("matchup_toi_secs")
         / pl.col("matchup_toi_secs").sum().over(
             ["team", "opponent", "line_type", "opp_line_rank", "venue"]
         )).alias("share_of_opp_line")
    )

    for col, dtype in LINE_MATCHING_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    df = df.select(list(LINE_MATCHING_SCHEMA.keys()))
    return df.sort(
        ["team", "opponent", "line_type", "opp_line_rank", "own_line_rank"]
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (consumed by the simulator + downstream features)
# ---------------------------------------------------------------------------


def team_matchup_profile(
    matching_df: pl.DataFrame,
    team: str,
    line_type: str = "F",
) -> pl.DataFrame:
    """Reduce per-opponent matching to a per-team average profile.

    For each (own_line_rank, opp_line_rank, venue) cell, compute the
    league-wide ``weighted_share`` as:

        Σ_opp  matchup_toi_secs(own, opp_rank, opp)
        ──────────────────────────────────────────────
        Σ_opp Σ_own'  matchup_toi_secs(own', opp_rank, opp)

    That denominator is the same for every ``own_line_rank`` row at a
    given (opp_rank, venue), so the four-row column at (any opp_rank,
    venue) sums to 1.0.  This is the version of the profile downstream
    consumers (Rust line-change sim, dashboard) expect.
    """
    df = matching_df.filter(
        (pl.col("team") == team) & (pl.col("line_type") == line_type)
    )
    out_schema = {
        "own_line_rank":   pl.Int64,
        "opp_line_rank":   pl.Int64,
        "venue":           pl.Utf8,
        "weighted_share":  pl.Float64,
        "total_toi_secs":  pl.Float64,
    }
    if df.is_empty():
        return pl.DataFrame(schema=out_schema)

    # Numerator: TOI at the (own_rank, opp_rank, venue) cell, summed across opponents
    num = (
        df.group_by(["own_line_rank", "opp_line_rank", "venue"])
        .agg(pl.col("matchup_toi_secs").sum().alias("cell_toi"))
    )
    # Denominator: total TOI when the opp_rank is on ice (any own_rank) — joined back
    den = (
        df.group_by(["opp_line_rank", "venue"])
        .agg(pl.col("matchup_toi_secs").sum().alias("opp_rank_toi"))
    )
    out = (
        num.join(den, on=["opp_line_rank", "venue"], how="left")
        .with_columns(
            (pl.col("cell_toi") / pl.col("opp_rank_toi").clip(lower_bound=1e-9))
              .alias("weighted_share")
        )
        .rename({"cell_toi": "total_toi_secs"})
        .drop("opp_rank_toi")
        .sort(["venue", "opp_line_rank", "own_line_rank"])
    )
    for col, dt in out_schema.items():
        if col in out.columns:
            out = out.with_columns(pl.col(col).cast(dt))
    return out.select(list(out_schema.keys()))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_line_matching(
    df:         pl.DataFrame,
    output_dir: Path,
    season:     int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"line_matching_{season}.parquet"
    df.write_parquet(path)
    return path


def read_line_matching(
    data_dir: Path,
    season:   int | None = None,
) -> pl.DataFrame | None:
    d = Path(data_dir) / "line_matching"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"line_matching_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("line_matching_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
