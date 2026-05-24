"""Special Teams Deployment Model — Feature 4.3.

Predicts coach personnel choices on power play and penalty kill, and the
share of time the first unit absorbs vs. the second unit.

Distinct from Feature 2.7 (PP/PK rating per player).  This model is
about **who goes on the ice**, not how good they are while there:
- PP1 vs PP2 personnel sets
- PK1 vs PK2 personnel sets
- First-unit share of total special-teams TOI (a coaching tendency
  signal — defensive coaches often roll PP2 more; offensive coaches
  ride PP1 deep into 2-minute kills).

Approach (V1)
-------------
1. From play-by-play, derive PP-from-home and PP-from-away windows by
   tracking `home_skaters` / `away_skaters` per event.  A window is one
   contiguous interval where one side has fewer skaters than the other.
2. For each PP window, find shifts of the team-on-PP that overlap the
   window.  Sum overlap-duration per player.
3. For each team, compute total PP TOI per player.  Top 5 by TOI =
   PP1 candidates; next 5 = PP2 candidates.  Within each candidate
   group, pick the most-common 5-player set across PP windows for
   stability.
4. Repeat the entire pipeline for PK (team-with-fewer-skaters side).
5. First-unit share = PP1 TOI / (PP1 + PP2 TOI), similarly for PK.

Output: one row per (team, season, unit) with personnel + share.

Limitations
-----------
- 4-on-3 OT power plays excluded for V1 (situations where both sides
  have ≤4 skaters but it's not a penalty kill).  Only "true" man
  advantage windows are captured.
- Goalie not included in unit personnel (skaters only).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_VERSION  = "st_deployment_v1"

PP_UNIT_SIZE   = 5  # 5-on-4 PP — 5 skaters on offense (typically 4F + 1D)
PK_UNIT_SIZE   = 4  # 4-on-5 PK — 4 skaters on defense (typically 2F + 2D)
MIN_UNIT_SECS  = 60.0  # require ≥ 60s of cumulative TOI to crystallize a unit


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

ST_DEPLOYMENT_SCHEMA: dict[str, pl.DataType] = {
    "team":             pl.Utf8,
    "season":           pl.Int64,
    "unit_type":        pl.Utf8,           # "PP1" | "PP2" | "PK1" | "PK2"
    "personnel":        pl.List(pl.Int64), # player_ids in the unit
    "unit_toi_secs":    pl.Float64,
    "share_of_st_toi":  pl.Float64,        # 0–1, share of team PP or PK TOI
    "team_st_toi":      pl.Float64,        # total PP or PK TOI for the team
    "team_st_gp":       pl.Int64,          # games where team had ≥ 1s on this ST type
    "model_version":    pl.Utf8,
}


# ---------------------------------------------------------------------------
# Goalie filtering
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PBP → power-play / penalty-kill windows
# ---------------------------------------------------------------------------


def _st_windows(pbp_g: pl.DataFrame) -> list[tuple[float, float, str]]:
    """Return [(t_start, t_end, owner), ...] windows for one game.

    `owner` ∈ {"home_pp", "away_pp"} — identifies which side has the
    man advantage in that window.  Windows are contiguous intervals
    where the skater counts indicate a PP/PK state (one side has more
    skaters than the other, both at least 3, neither pulled-goalie >5).
    """
    needed = {"period", "time_in_period_secs", "home_skaters", "away_skaters"}
    if not needed.issubset(pbp_g.columns):
        return []
    if pbp_g.is_empty():
        return []

    df = pbp_g.select([
        "period", "time_in_period_secs", "home_skaters", "away_skaters",
    ]).drop_nulls()
    if df.is_empty():
        return []

    df = df.with_columns(
        ((pl.col("period") - 1) * 1200.0 + pl.col("time_in_period_secs").cast(pl.Float64)).alias("gs")
    ).sort("gs")

    gs = df["gs"].to_numpy()
    hs = df["home_skaters"].to_numpy()
    aw = df["away_skaters"].to_numpy()

    def _label(h: int, a: int) -> str | None:
        # Skip empty-net / pulled-goalie situations (skater counts ≥ 6)
        if h > 5 or a > 5:
            return None
        # Need both sides at ≥ 3 (standard PP/PK frame, ignores 5v3 weirdness)
        if h < 3 or a < 3:
            return None
        if h > a:
            return "home_pp"
        if a > h:
            return "away_pp"
        return None

    windows: list[tuple[float, float, str]] = []
    cur_label: str | None = None
    cur_start: float | None = None
    for i in range(len(gs)):
        lab = _label(int(hs[i]), int(aw[i]))
        if lab != cur_label:
            if cur_label is not None and cur_start is not None:
                windows.append((cur_start, float(gs[i]), cur_label))
            cur_label = lab
            cur_start = float(gs[i]) if lab is not None else None
    if cur_label is not None and cur_start is not None:
        windows.append((cur_start, float(gs[-1]) + 1.0, cur_label))
    return windows


# ---------------------------------------------------------------------------
# Shift-window overlap
# ---------------------------------------------------------------------------


def _overlap_secs(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _attribute_unit_toi(
    shifts_team: pl.DataFrame,
    windows:     list[tuple[float, float]],
) -> tuple[dict[int, float], list[tuple[float, frozenset[int]]]]:
    """Return per-player overlap TOI and per-window personnel sets.

    Args:
        shifts_team: shifts for ONE team in ONE game (skaters).
        windows:     list of (t_start, t_end) for the ST windows on THIS team.

    Returns:
        (player_toi, snapshots) where snapshots is a list of
        (window_duration, frozenset(player_ids_on_ice_during_window)).
    """
    player_toi:  dict[int, float] = defaultdict(float)
    snapshots:   list[tuple[float, frozenset[int]]] = []

    if shifts_team.is_empty() or not windows:
        return player_toi, snapshots

    starts = shifts_team["game_seconds_start"].to_numpy().astype(np.float64)
    ends   = shifts_team["game_seconds_end"].to_numpy().astype(np.float64)
    pids   = shifts_team["player_id"].to_numpy()

    for w0, w1 in windows:
        # Identify the "modal" personnel set during this window — the set of
        # players present for at least 50% of its duration.  This captures the
        # actual unit rather than transient swaps mid-PP.
        wdur = w1 - w0
        if wdur <= 0:
            continue

        # Vector of overlap per player (deduplicated by player_id)
        per_player: dict[int, float] = defaultdict(float)
        for s, e, p in zip(starts, ends, pids):
            ov = _overlap_secs(float(s), float(e), w0, w1)
            if ov > 0:
                per_player[int(p)] += ov

        # Accumulate season-wide TOI
        for pid, secs in per_player.items():
            player_toi[pid] += secs

        # Modal personnel: players present ≥ 50% of the window
        modal = frozenset(pid for pid, secs in per_player.items() if secs >= 0.5 * wdur)
        if modal:
            snapshots.append((wdur, modal))

    return player_toi, snapshots


# ---------------------------------------------------------------------------
# Unit extraction
# ---------------------------------------------------------------------------


def _extract_units(
    snapshots:  list[tuple[float, frozenset[int]]],
    player_toi: dict[int, float],
    unit_size:  int,
) -> tuple[tuple[frozenset[int] | None, float], tuple[frozenset[int] | None, float]]:
    """Identify the two most-used `unit_size`-player sets.

    Strategy:
      - Bucket each snapshot by which subset of size `unit_size` it
        contains (use the top-`unit_size` overlap members of the modal
        set if it's larger).
      - Count cumulative window-duration per candidate set.
      - U1 = highest cumulative; U2 = highest-cumulative disjoint set,
        i.e. set with NO player overlap with U1 (so it's the "other
        five" — typical PP2 / PK2 personnel).

    Returns:
        ((U1_set, U1_secs), (U2_set, U2_secs)).  Either pair may be
        (None, 0.0) when no qualifying unit exists.
    """
    if not snapshots:
        return (None, 0.0), (None, 0.0)

    # Reduce each snapshot to a canonical `unit_size` subset:
    # take the top `unit_size` players from the snapshot by season TOI.
    bucket: Counter[frozenset[int]] = Counter()
    for wdur, modal in snapshots:
        if len(modal) < unit_size:
            continue
        if len(modal) == unit_size:
            sub = modal
        else:
            # Choose by season-wide TOI
            ranked = sorted(modal, key=lambda p: -player_toi[p])
            sub = frozenset(ranked[:unit_size])
        bucket[sub] += wdur

    if not bucket:
        return (None, 0.0), (None, 0.0)

    # U1: most-used
    ordered = bucket.most_common()
    u1_set, u1_secs = ordered[0]
    if u1_secs < MIN_UNIT_SECS:
        return (None, 0.0), (None, 0.0)

    # U2: highest-TOI set disjoint from U1.  If none disjoint, take 2nd most-used.
    u2_set: frozenset[int] | None = None
    u2_secs = 0.0
    for cand_set, cand_secs in ordered[1:]:
        if cand_set.isdisjoint(u1_set):
            u2_set  = cand_set
            u2_secs = float(cand_secs)
            break
    if u2_set is None and len(ordered) > 1:
        u2_set, _ = ordered[1]
        u2_secs   = float(ordered[1][1])

    if u2_secs < MIN_UNIT_SECS:
        u2_set  = None
        u2_secs = 0.0

    return (u1_set, float(u1_secs)), (u2_set, u2_secs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_st_deployment(
    shifts_df: pl.DataFrame,
    pbp_df:    pl.DataFrame,
    season:    int,
    team_lookup: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Compute per-team PP1/PP2/PK1/PK2 personnel + first-unit share.

    Args:
        shifts_df:    raw shift data for the season.
        pbp_df:       play-by-play for the season.
        season:       e.g. 2025.
        team_lookup:  optional team_id (Int64) → abbrev mapping.

    Returns:
        Polars DataFrame matching ST_DEPLOYMENT_SCHEMA.
    """
    required = {"game_id", "player_id", "team_id",
                "game_seconds_start", "game_seconds_end"}
    if not required.issubset(shifts_df.columns):
        return pl.DataFrame(schema=ST_DEPLOYMENT_SCHEMA)
    if "game_id" not in pbp_df.columns:
        return pl.DataFrame(schema=ST_DEPLOYMENT_SCHEMA)

    shifts = shifts_df.with_columns([
        pl.col("game_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64),
        pl.col("player_id").cast(pl.Int64),
        pl.col("game_seconds_start").cast(pl.Float64),
        pl.col("game_seconds_end").cast(pl.Float64),
    ])
    pbp = pbp_df.with_columns(pl.col("game_id").cast(pl.Int64))

    # Per-team accumulators
    pp_player_toi: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    pk_player_toi: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    pp_snapshots:  dict[int, list[tuple[float, frozenset[int]]]] = defaultdict(list)
    pk_snapshots:  dict[int, list[tuple[float, frozenset[int]]]] = defaultdict(list)
    pp_games:      dict[int, set[int]] = defaultdict(set)
    pk_games:      dict[int, set[int]] = defaultdict(set)

    game_ids = shifts["game_id"].unique().to_list()

    for gid in game_ids:
        pg = pbp.filter(pl.col("game_id") == gid)
        if pg.is_empty():
            continue

        # Determine home/away team_ids
        if not {"home_team_id", "away_team_id"}.issubset(pg.columns):
            continue
        ht = pg["home_team_id"].drop_nulls()
        at = pg["away_team_id"].drop_nulls()
        if ht.is_empty() or at.is_empty():
            continue
        home_id = int(ht[0])
        away_id = int(at[0])

        windows = _st_windows(pg)
        if not windows:
            continue

        # Split windows by which team is on PP
        home_pp = [(w0, w1) for (w0, w1, lab) in windows if lab == "home_pp"]
        away_pp = [(w0, w1) for (w0, w1, lab) in windows if lab == "away_pp"]

        g_shifts = shifts.filter(pl.col("game_id") == gid)
        goalies = _goalies_in_game(pg)
        if goalies:
            g_shifts = g_shifts.filter(~pl.col("player_id").is_in(list(goalies)))
        home_shifts = g_shifts.filter(pl.col("team_id") == home_id)
        away_shifts = g_shifts.filter(pl.col("team_id") == away_id)

        # Home on PP: home_shifts vs home_pp windows
        if home_pp:
            ptoi, snaps = _attribute_unit_toi(home_shifts, home_pp)
            for pid, secs in ptoi.items():
                pp_player_toi[home_id][pid] += secs
            pp_snapshots[home_id].extend(snaps)
            if ptoi:
                pp_games[home_id].add(int(gid))

            # Away simultaneously on PK
            kptoi, ksnaps = _attribute_unit_toi(away_shifts, home_pp)
            for pid, secs in kptoi.items():
                pk_player_toi[away_id][pid] += secs
            pk_snapshots[away_id].extend(ksnaps)
            if kptoi:
                pk_games[away_id].add(int(gid))

        if away_pp:
            ptoi, snaps = _attribute_unit_toi(away_shifts, away_pp)
            for pid, secs in ptoi.items():
                pp_player_toi[away_id][pid] += secs
            pp_snapshots[away_id].extend(snaps)
            if ptoi:
                pp_games[away_id].add(int(gid))

            kptoi, ksnaps = _attribute_unit_toi(home_shifts, away_pp)
            for pid, secs in kptoi.items():
                pk_player_toi[home_id][pid] += secs
            pk_snapshots[home_id].extend(ksnaps)
            if kptoi:
                pk_games[home_id].add(int(gid))

    team_lookup = team_lookup or {}
    rows: list[dict] = []

    all_team_ids = set(pp_player_toi.keys()) | set(pk_player_toi.keys())
    if team_lookup:
        known = set(team_lookup.keys())
        all_team_ids = {tid for tid in all_team_ids if tid in known}
    for tid in sorted(all_team_ids):
        abbrev = team_lookup.get(int(tid), str(int(tid)))

        # Power play units
        (u1, u1_secs), (u2, u2_secs) = _extract_units(
            pp_snapshots[tid], pp_player_toi[tid], PP_UNIT_SIZE
        )
        total_pp = sum(pp_player_toi[tid].values()) / max(PP_UNIT_SIZE, 1)
        if u1 is not None:
            rows.append({
                "team":             abbrev,
                "season":           int(season),
                "unit_type":        "PP1",
                "personnel":        sorted(int(p) for p in u1),
                "unit_toi_secs":    u1_secs,
                "share_of_st_toi":  (u1_secs / total_pp) if total_pp > 0 else 0.0,
                "team_st_toi":      total_pp,
                "team_st_gp":       len(pp_games[tid]),
                "model_version":    MODEL_VERSION,
            })
        if u2 is not None:
            rows.append({
                "team":             abbrev,
                "season":           int(season),
                "unit_type":        "PP2",
                "personnel":        sorted(int(p) for p in u2),
                "unit_toi_secs":    u2_secs,
                "share_of_st_toi":  (u2_secs / total_pp) if total_pp > 0 else 0.0,
                "team_st_toi":      total_pp,
                "team_st_gp":       len(pp_games[tid]),
                "model_version":    MODEL_VERSION,
            })

        # Penalty kill units
        (k1, k1_secs), (k2, k2_secs) = _extract_units(
            pk_snapshots[tid], pk_player_toi[tid], PK_UNIT_SIZE
        )
        total_pk = sum(pk_player_toi[tid].values()) / max(PK_UNIT_SIZE, 1)
        if k1 is not None:
            rows.append({
                "team":             abbrev,
                "season":           int(season),
                "unit_type":        "PK1",
                "personnel":        sorted(int(p) for p in k1),
                "unit_toi_secs":    k1_secs,
                "share_of_st_toi":  (k1_secs / total_pk) if total_pk > 0 else 0.0,
                "team_st_toi":      total_pk,
                "team_st_gp":       len(pk_games[tid]),
                "model_version":    MODEL_VERSION,
            })
        if k2 is not None:
            rows.append({
                "team":             abbrev,
                "season":           int(season),
                "unit_type":        "PK2",
                "personnel":        sorted(int(p) for p in k2),
                "unit_toi_secs":    k2_secs,
                "share_of_st_toi":  (k2_secs / total_pk) if total_pk > 0 else 0.0,
                "team_st_toi":      total_pk,
                "team_st_gp":       len(pk_games[tid]),
                "model_version":    MODEL_VERSION,
            })

    if not rows:
        return pl.DataFrame(schema=ST_DEPLOYMENT_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in ST_DEPLOYMENT_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    df = df.select(list(ST_DEPLOYMENT_SCHEMA.keys()))
    return df.sort(["team", "unit_type"])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_st_deployment(
    df:         pl.DataFrame,
    output_dir: Path,
    season:     int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"st_deployment_{season}.parquet"
    df.write_parquet(path)
    return path


def read_st_deployment(
    data_dir: Path,
    season:   int | None = None,
) -> pl.DataFrame | None:
    d = Path(data_dir) / "st_deployment"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"st_deployment_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("st_deployment_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
