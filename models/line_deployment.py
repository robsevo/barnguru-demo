"""Line Deployment Forecaster — Feature 4.1.

Predicts which lines a coach will deploy for an upcoming game and how many
minutes each line will play.

Approach (V1 — interpretable, no neural nets per Principle 2)
-------------------------------------------------------------
1. Aggregate **5v5 even-strength shifts** per team across the season.
2. For each team, compute pairwise on-ice co-occurrence TOI for every
   skater pair on that team (forwards vs forwards, defense vs defense).
3. Pull a position lookup from shots data (`player_position` ∈ C/L/R/D).
4. Greedy assemble lines from highest aggregate EV TOI down:
   - Forwards: top-12 by EV TOI → 4 lines of 3 (anchored by highest-TOI
     remaining forward, joined by their 2 strongest co-occurring F partners).
   - Defense: top-6 by EV TOI → 3 pairs of 2 (anchored by highest-TOI
     remaining defender, joined by their strongest co-occurring D partner).
5. Projected minutes per line/pair = average per-game TOI of the assembled
   unit (mean of stints where ALL 3F or 2D were on ice together).

Output: one row per (team, season, line_rank, line_type) — drives the
lineup builder (6.1), minutes allocation (6.2), and line-combination
predictor (6.3).

Limitations
-----------
- Coach identity baked into team-season (V1). Per-coach disambiguation
  (mid-season fires) handled by Feature 4.13's regime detector.
- Position lookup falls back to "F" for any skater absent from shots data.
  Defensemen with no shots in the season would be misclassified; in
  practice this is rare (every NHL D takes at least one shot per season).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION = "line_deployment_v1"

MIN_TEAM_GAMES        = 5      # team must have ≥ this many games to forecast
MIN_LINE_COOCCUR_SECS = 60.0   # co-occurrence pairs must share ≥ 60s EV TOI
N_FORWARDS_PER_LINE   = 3
N_DEFENSE_PER_PAIR    = 2
N_FORWARD_LINES       = 4
N_DEFENSE_PAIRS       = 3


# ---------------------------------------------------------------------------
# Output schema — one row per (team, season, line_rank, line_type)
# ---------------------------------------------------------------------------

LINE_DEPLOYMENT_SCHEMA: dict[str, pl.DataType] = {
    "team":                    pl.Utf8,
    "season":                  pl.Int64,
    "line_type":               pl.Utf8,   # "F" or "D"
    "line_rank":               pl.Int64,  # 1 → 4 for F, 1 → 3 for D
    "player_1":                pl.Int64,
    "player_2":                pl.Int64,
    "player_3":                pl.Int64,  # null for D pairs
    # Cumulative seconds the trio (or pair) was on ice together.  This is
    # the cohesion signal — high when the coach keeps a unit intact, low
    # when it gets shuffled mid-game.
    "chemistry_toi_secs":      pl.Float64,
    # Trio's average per-game time on ice together (chemistry_toi / gp).
    "trio_toi_per_game":       pl.Float64,
    # **Line minutes allocation** — the average per-game EV TOI of the
    # line's members, regardless of who they were paired with at any
    # given moment.  This is what PLAN 4.1 calls "minutes allocation":
    # the projected playing time of the unit's skaters tomorrow.
    "line_toi_per_game":       pl.Float64,
    # How locked-in the unit is: chemistry_toi / least-played-member's
    # EV TOI.  Capped at 1.0.  0 → trio never overlaps, 1 → every shift
    # the bottom member plays is alongside the other two.
    "cohesion_pct":            pl.Float64,
    "share_of_team_toi":       pl.Float64,    # 0–1 — share of team EV TOI together
    "team_gp":                 pl.Int64,
    "model_version":           pl.Utf8,
}


# ---------------------------------------------------------------------------
# Shift-overlap stint expansion (5v5 only)
# ---------------------------------------------------------------------------


def _team_stints_5v5(
    shifts_df: pl.DataFrame,
    pbp_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build per-team 5v5 stints: each row is one interval where the team
    had exactly 5 skaters on ice, with the list of player_ids on ice.

    Args:
        shifts_df: shift records (game_id, player_id, team_id,
                   game_seconds_start, game_seconds_end).
        pbp_df:    play-by-play events (used to mask out non-EV intervals
                   via home_skaters / away_skaters column).

    Returns:
        Polars DataFrame with columns:
            game_id (Int64), team_id (Int64),
            t_start (Float64), t_end (Float64),
            duration_secs (Float64), player_ids (List[Int64]).
        Empty DataFrame if inputs lack required columns.
    """
    required = {"game_id", "player_id", "team_id",
                "game_seconds_start", "game_seconds_end"}
    if not required.issubset(shifts_df.columns):
        return _empty_team_stints()

    # Cast keys to Int64 for stable joining
    shifts = shifts_df.with_columns([
        pl.col("game_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64),
        pl.col("player_id").cast(pl.Int64),
        pl.col("game_seconds_start").cast(pl.Float64),
        pl.col("game_seconds_end").cast(pl.Float64),
    ])

    if "game_id" in pbp_df.columns:
        pbp = pbp_df.with_columns(pl.col("game_id").cast(pl.Int64))
    else:
        pbp = pbp_df

    out_rows: list[dict] = []
    game_ids = shifts["game_id"].unique().to_list()

    for gid in game_ids:
        g = shifts.filter(pl.col("game_id") == gid)
        if g.is_empty():
            continue

        pg = pbp.filter(pl.col("game_id") == gid)
        # Skater-count intervals from pbp (5v5 windows where applicable)
        ev_intervals = _ev_5v5_intervals(pg)

        # Goalies in this game — drop their shift rows so on-ice counts
        # reflect skaters only.  NHL shift records include the goalie
        # alongside skaters; pbp exposes the active goalie via `home_goalie`
        # and `away_goalie` ID columns.
        goalies = _goalies_in_game(pg)
        if goalies:
            g = g.filter(~pl.col("player_id").is_in(list(goalies)))

        # Per-team breakpoints
        for tid in g["team_id"].unique().to_list():
            tg = g.filter(pl.col("team_id") == tid)
            starts = tg["game_seconds_start"].to_numpy()
            ends   = tg["game_seconds_end"].to_numpy()
            pids   = tg["player_id"].to_numpy()

            breakpoints = np.unique(np.concatenate([starts, ends]))
            if len(breakpoints) < 2:
                continue

            for i in range(len(breakpoints) - 1):
                t0, t1 = float(breakpoints[i]), float(breakpoints[i + 1])
                dur = t1 - t0
                if dur < 1.0:
                    continue

                on_ice_mask = (starts <= t0) & (ends >= t1)
                on_ice_pids = pids[on_ice_mask].tolist()

                # Only count 5-skater intervals (excludes goalie which
                # shouldn't be in skater shifts anyway, but defensive guard).
                if len(on_ice_pids) != 5:
                    continue

                # Intersect with 5v5 EV windows from pbp (skip if no overlap)
                if ev_intervals is not None and not _interval_overlaps(t0, t1, ev_intervals):
                    continue

                out_rows.append({
                    "game_id":       int(gid),
                    "team_id":       int(tid),
                    "t_start":       t0,
                    "t_end":         t1,
                    "duration_secs": dur,
                    "player_ids":    on_ice_pids,
                })

    if not out_rows:
        return _empty_team_stints()

    return pl.DataFrame(out_rows).with_columns(
        pl.col("player_ids").cast(pl.List(pl.Int64))
    )


def _empty_team_stints() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "game_id":       pl.Int64,
        "team_id":       pl.Int64,
        "t_start":       pl.Float64,
        "t_end":         pl.Float64,
        "duration_secs": pl.Float64,
        "player_ids":    pl.List(pl.Int64),
    })


def _ev_5v5_intervals(pbp_g: pl.DataFrame) -> np.ndarray | None:
    """Return a sorted (N, 2) array of [start_secs, end_secs] windows
    during which the game was 5v5 even strength, derived from
    home_skaters / away_skaters changes in pbp events for one game.

    Returns None if pbp data is missing required columns — caller will
    treat as "no filtering" (count all shifts as EV).
    """
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

    # Convert (period, time_in_period_secs) → game_seconds
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
    # If the last observed state is EV, extend the interval to +inf — pbp
    # only marks state changes, so the absence of a subsequent event means
    # the state continued through the rest of the game.
    if cur_start is not None:
        intervals.append((cur_start, float("inf")))

    if not intervals:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(intervals, dtype=np.float64)


def _interval_overlaps(t0: float, t1: float, ranges: np.ndarray) -> bool:
    """True if [t0, t1) overlaps any [r0, r1) in `ranges`."""
    if ranges.size == 0:
        return False
    return bool(((ranges[:, 0] < t1) & (ranges[:, 1] > t0)).any())


def _goalies_in_game(pbp_g: pl.DataFrame) -> set[int]:
    """Return the set of goalie player_ids active in a game.

    Uses `shot_goalie_id` (the goalie facing each shot) — the
    `home_goalie` / `away_goalie` booleans only flag presence and don't
    carry the player_id.
    """
    out: set[int] = set()
    if "shot_goalie_id" in pbp_g.columns:
        vals = pbp_g["shot_goalie_id"].drop_nulls().unique().to_list()
        for v in vals:
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# Position lookup
# ---------------------------------------------------------------------------


def build_position_lookup(shots_df: pl.DataFrame) -> dict[int, str]:
    """Map player_id → 'F' or 'D' using shots' `player_position`.

    'C', 'L', 'R' → 'F'; 'D' → 'D'.  Players absent from shots get no entry.
    """
    if shots_df.is_empty() or "shooter_id" not in shots_df.columns:
        return {}
    if "player_position" not in shots_df.columns:
        return {}

    df = (
        shots_df
        .select(["shooter_id", "player_position"])
        .drop_nulls()
        .filter(pl.col("player_position").is_in(["C", "L", "R", "D"]))
        .group_by("shooter_id", "player_position")
        .len()
    )
    if df.is_empty():
        return {}

    # Most-frequent position per shooter
    df = df.sort(["shooter_id", "len"], descending=[False, True])
    df = df.unique(subset=["shooter_id"], keep="first").select(["shooter_id", "player_position"])

    lookup: dict[int, str] = {}
    for pid, pos in df.iter_rows():
        lookup[int(pid)] = "F" if pos in ("C", "L", "R") else "D"
    return lookup


# ---------------------------------------------------------------------------
# Pairwise co-occurrence TOI
# ---------------------------------------------------------------------------


def _toi_aggregates(
    team_stints: pl.DataFrame,
    team_id: int,
    pos_lookup: dict[int, str],
) -> tuple[dict[int, float], dict[frozenset[int], float], dict[frozenset[int], float], list[int]]:
    """Compute per-team aggregate TOI + pairwise TOI + line-trio TOI.

    Returns:
        (player_toi, pair_toi, trio_toi, game_ids)
        - player_toi: pid → seconds on ice
        - pair_toi:   frozenset({p1, p2}) → cumulative seconds
        - trio_toi:   frozenset({p1, p2, p3}) → cumulative seconds for
                      forward trios (or empty if none)
        - game_ids:   list of distinct game_ids this team played
    """
    player_toi:  dict[int, float] = defaultdict(float)
    pair_toi:    dict[frozenset[int], float] = defaultdict(float)
    trio_toi:    dict[frozenset[int], float] = defaultdict(float)

    ts = team_stints.filter(pl.col("team_id") == team_id)
    if ts.is_empty():
        return player_toi, pair_toi, trio_toi, []

    game_ids = ts["game_id"].unique().to_list()

    for row in ts.iter_rows(named=True):
        pids = row["player_ids"]
        dur  = float(row["duration_secs"])
        if not pids:
            continue

        # Per-player TOI
        for p in pids:
            player_toi[int(p)] += dur

        # Pairwise
        for a, b in combinations(sorted(int(p) for p in pids), 2):
            pair_toi[frozenset({a, b})] += dur

        # Forward trios only (skip if mixed positions)
        forwards = [int(p) for p in pids if pos_lookup.get(int(p)) == "F"]
        if len(forwards) >= 3:
            for trio in combinations(sorted(forwards), 3):
                trio_toi[frozenset(trio)] += dur

    return player_toi, pair_toi, trio_toi, game_ids


# ---------------------------------------------------------------------------
# Greedy line / pair assembly
# ---------------------------------------------------------------------------


def _assemble_forward_lines(
    player_toi: dict[int, float],
    pair_toi:   dict[frozenset[int], float],
    trio_toi:   dict[frozenset[int], float],
    pos_lookup: dict[int, str],
) -> list[tuple[list[int], float]]:
    """Greedy assemble up to N_FORWARD_LINES forward lines from
    top-TOI forwards.

    Returns a list of (player_ids_in_line, chemistry_toi_secs) tuples,
    one per line, ordered by line rank (1 → 4).
    """
    # Forwards only, top-N by TOI
    forwards_sorted = sorted(
        (pid for pid in player_toi if pos_lookup.get(pid) == "F"),
        key=lambda p: -player_toi[p],
    )
    pool: list[int] = forwards_sorted[: N_FORWARD_LINES * N_FORWARDS_PER_LINE]
    if len(pool) < N_FORWARDS_PER_LINE:
        return []

    pool_set: set[int] = set(pool)
    lines: list[tuple[list[int], float]] = []

    while len(lines) < N_FORWARD_LINES and len(pool_set) >= N_FORWARDS_PER_LINE:
        # Anchor = highest TOI remaining
        anchor = max(pool_set, key=lambda p: player_toi[p])

        # Score every candidate trio containing the anchor.  Primary key
        # is the trio's direct co-occurrence TOI; secondary is the sum of
        # all 3 pairwise TOIs.  Tuple ordering guarantees a trio with real
        # 3-way chemistry always beats one that only has pair-chemistry.
        best_trio: frozenset[int] | None = None
        best_key: tuple[float, float] = (-1.0, -1.0)
        for other_pair in combinations(pool_set - {anchor}, 2):
            trio = frozenset({anchor, *other_pair})
            p1, p2 = other_pair
            primary   = trio_toi.get(trio, 0.0)
            secondary = (
                pair_toi.get(frozenset({anchor, p1}), 0.0)
                + pair_toi.get(frozenset({anchor, p2}), 0.0)
                + pair_toi.get(frozenset({p1, p2}), 0.0)
            )
            key = (primary, secondary)
            if key > best_key:
                best_key  = key
                best_trio = trio

        if best_trio is None or best_key[1] < MIN_LINE_COOCCUR_SECS:
            break

        line_pids = sorted(best_trio, key=lambda p: -player_toi[p])
        lines.append((line_pids, float(trio_toi.get(best_trio, 0.0))))
        pool_set -= best_trio

    return lines


def _assemble_defense_pairs(
    player_toi: dict[int, float],
    pair_toi:   dict[frozenset[int], float],
    pos_lookup: dict[int, str],
) -> list[tuple[list[int], float]]:
    """Greedy assemble up to N_DEFENSE_PAIRS defense pairs from top-TOI D."""
    defense_sorted = sorted(
        (pid for pid in player_toi if pos_lookup.get(pid) == "D"),
        key=lambda p: -player_toi[p],
    )
    pool_set: set[int] = set(defense_sorted[: N_DEFENSE_PAIRS * N_DEFENSE_PER_PAIR])
    if len(pool_set) < N_DEFENSE_PER_PAIR:
        return []

    pairs: list[tuple[list[int], float]] = []
    while len(pairs) < N_DEFENSE_PAIRS and len(pool_set) >= N_DEFENSE_PER_PAIR:
        anchor = max(pool_set, key=lambda p: player_toi[p])
        best_partner: int | None = None
        best_score   = -1.0
        for cand in pool_set - {anchor}:
            score = pair_toi.get(frozenset({anchor, cand}), 0.0)
            if score > best_score:
                best_score   = score
                best_partner = cand
        if best_partner is None:
            break
        pair_pids = sorted([anchor, best_partner], key=lambda p: -player_toi[p])
        pairs.append((pair_pids, float(best_score)))
        pool_set -= {anchor, best_partner}

    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_line_deployment(
    shifts_df: pl.DataFrame,
    pbp_df:    pl.DataFrame,
    shots_df:  pl.DataFrame,
    season:    int,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Compute per-team forecasted line deployment for `season`.

    Args:
        shifts_df:    raw shift data for the season.
        pbp_df:       play-by-play data (used to mask non-5v5 intervals).
        shots_df:     MoneyPuck shots (used for player position lookup).
        season:       e.g. 2025.
        team_lookup:  optional team_id (Int64) → abbrev (str) mapping.
                      If None, team column is the numeric team_id stringified.

    Returns:
        Polars DataFrame matching LINE_DEPLOYMENT_SCHEMA, ordered by
        (team, line_type, line_rank).  Empty if inputs are insufficient.
    """
    if shifts_df.is_empty():
        return pl.DataFrame(schema=LINE_DEPLOYMENT_SCHEMA)

    pos_lookup = build_position_lookup(shots_df)
    if not pos_lookup:
        warnings.warn(
            "compute_line_deployment: empty position lookup from shots — "
            "forwards/defense classification will be skipped.",
            stacklevel=2,
        )

    team_stints = _team_stints_5v5(shifts_df, pbp_df)
    if team_stints.is_empty():
        return pl.DataFrame(schema=LINE_DEPLOYMENT_SCHEMA)

    teams = team_stints["team_id"].unique().to_list()
    # Filter exhibition team_ids (e.g. 4 Nations Face-Off) when a lookup is
    # provided — keeps the team list aligned with the 32 NHL clubs.
    if team_lookup is not None:
        known = set(team_lookup.keys())
        teams = [t for t in teams if int(t) in known]

    out_rows: list[dict] = []

    for tid in teams:
        player_toi, pair_toi, trio_toi, game_ids = _toi_aggregates(
            team_stints, int(tid), pos_lookup
        )
        if len(game_ids) < MIN_TEAM_GAMES:
            continue

        team_ev_toi = sum(player_toi.values()) / 5.0  # 5 skaters per stint
        if team_ev_toi <= 0:
            continue

        team_abbrev = (team_lookup or {}).get(int(tid), str(int(tid)))

        n_gp = max(len(game_ids), 1)

        # Forward lines
        f_lines = _assemble_forward_lines(player_toi, pair_toi, trio_toi, pos_lookup)
        for rank, (pids, chem) in enumerate(f_lines, start=1):
            trio_per_game = chem / n_gp
            # Real minutes allocation = average per-game EV TOI of the 3 members
            member_tois = [player_toi.get(int(p), 0.0) for p in pids]
            line_toi_pg = (sum(member_tois) / N_FORWARDS_PER_LINE) / n_gp
            # Cohesion = chemistry / least-played member's TOI (upper bound:
            # the trio can never be on ice longer than the least-played member is)
            min_member_toi = min(member_tois) if member_tois else 0.0
            cohesion = chem / max(min_member_toi, 1e-9) if min_member_toi > 0 else 0.0
            share = chem / max(team_ev_toi, 1e-9)
            out_rows.append({
                "team":                   team_abbrev,
                "season":                 int(season),
                "line_type":              "F",
                "line_rank":              rank,
                "player_1":               int(pids[0]),
                "player_2":               int(pids[1]) if len(pids) > 1 else None,
                "player_3":               int(pids[2]) if len(pids) > 2 else None,
                "chemistry_toi_secs":     chem,
                "trio_toi_per_game":      trio_per_game,
                "line_toi_per_game":      line_toi_pg,
                "cohesion_pct":           min(1.0, max(0.0, cohesion)),
                "share_of_team_toi":      share,
                "team_gp":                len(game_ids),
                "model_version":          MODEL_VERSION,
            })

        # Defense pairs
        d_pairs = _assemble_defense_pairs(player_toi, pair_toi, pos_lookup)
        for rank, (pids, chem) in enumerate(d_pairs, start=1):
            trio_per_game = chem / n_gp
            member_tois = [player_toi.get(int(p), 0.0) for p in pids]
            line_toi_pg = (sum(member_tois) / N_DEFENSE_PER_PAIR) / n_gp
            min_member_toi = min(member_tois) if member_tois else 0.0
            cohesion = chem / max(min_member_toi, 1e-9) if min_member_toi > 0 else 0.0
            share = chem / max(team_ev_toi, 1e-9)
            out_rows.append({
                "team":                   team_abbrev,
                "season":                 int(season),
                "line_type":              "D",
                "line_rank":              rank,
                "player_1":               int(pids[0]),
                "player_2":               int(pids[1]) if len(pids) > 1 else None,
                "player_3":               None,
                "chemistry_toi_secs":     chem,
                "trio_toi_per_game":      trio_per_game,
                "line_toi_per_game":      line_toi_pg,
                "cohesion_pct":           min(1.0, max(0.0, cohesion)),
                "share_of_team_toi":      share,
                "team_gp":                len(game_ids),
                "model_version":          MODEL_VERSION,
            })

    if not out_rows:
        return pl.DataFrame(schema=LINE_DEPLOYMENT_SCHEMA)

    out = pl.DataFrame(out_rows)
    for col, dtype in LINE_DEPLOYMENT_SCHEMA.items():
        if col in out.columns:
            out = out.with_columns(pl.col(col).cast(dtype))
    out = out.select(list(LINE_DEPLOYMENT_SCHEMA.keys()))
    return out.sort(["team", "line_type", "line_rank"])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_line_deployment(
    df:         pl.DataFrame,
    output_dir: Path,
    season:     int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"line_deployment_{season}.parquet"
    df.write_parquet(path)
    return path


def read_line_deployment(
    data_dir: Path,
    season:   int | None = None,
) -> pl.DataFrame | None:
    d = Path(data_dir) / "line_deployment"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"line_deployment_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("line_deployment_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
