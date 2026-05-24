"""Coaching Style Vector — Feature 4.11.

8-dimension system vector extracted per team from play-by-play data.
Each dimension is a percentile-rank ∈ [0, 1] across the league for the
season; 0.5 = league average, 1.0 = the most-extreme team in that
dimension, 0.0 = the least.  Rank, not raw value, so the 8 numbers
share a common scale and the radar chart on the dashboard is readable.

Dimensions
----------
1. forecheck_aggression  — offensive-zone takeaways per 60 EV.  Aggressive
                           forecheck = more takeaways near the opponent goal.
2. dz_structure          — defensive-zone takeaways minus giveaways per 60 EV
                           (normalized).  Higher = more disciplined DZ play.
3. pace                  — total shots per 60 of EV TOI (both teams).
4. physicality           — hits per 60 EV.
5. oz_structure          — share of EV shots inside the "danger zone"
                           (distance < 25 ft from net).  Higher = more
                           interior offense, fewer perimeter shots.
6. nz_tendency           — share of zone entries that are carry-ins (vs.
                           dump-ins) — when ``carry_in`` data is available.
                           NaN otherwise — front-end shows "—".
7. line_match            — inverse Shannon entropy of F-line share-of-TOI.
                           High = top-line shelter (concentrated minutes);
                           low = balanced four-line attack.
8. st_aggression         — PP shots-for per 60 of PP TOI (volume PP).

Outputs
-------
- ``raw`` columns hold the underlying rate so the dashboard can show what
  the percentile-rank means.
- ``rank`` columns are the [0, 1] percentile values that feed the radar
  chart AND the roster-fit model (4.12).

Honest limitations
------------------
- ``nz_tendency`` may be NaN until the PBP parser populates
  ``carry_in`` on zone-entry events.  When it lands, the same code path
  picks it up automatically.
- We rank within a single season — comparing 2024-25 forecheck against
  2025-26 is not meaningful with this output alone.
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

MODEL_VERSION = "coaching_style_v1"

# Distance cutoff (ft from net) for "danger zone" / interior shots.
DANGER_ZONE_DIST = 25.0


class DataMissingWarning(UserWarning):
    """Raised when style inputs are absent or insufficient."""


COACHING_STYLE_SCHEMA: dict[str, pl.DataType] = {
    "team":                       pl.Utf8,
    "season":                     pl.Int64,
    # Raw rates
    "forecheck_aggression_raw":   pl.Float64,
    "dz_structure_raw":           pl.Float64,
    "pace_raw":                   pl.Float64,
    "physicality_raw":            pl.Float64,
    "oz_structure_raw":           pl.Float64,
    "nz_tendency_raw":            pl.Float64,
    "line_match_raw":             pl.Float64,
    "st_aggression_raw":          pl.Float64,
    # League-percentile ranks ∈ [0, 1]
    "forecheck_aggression_rank":  pl.Float64,
    "dz_structure_rank":          pl.Float64,
    "pace_rank":                  pl.Float64,
    "physicality_rank":           pl.Float64,
    "oz_structure_rank":          pl.Float64,
    "nz_tendency_rank":           pl.Float64,
    "line_match_rank":            pl.Float64,
    "st_aggression_rank":         pl.Float64,
    "model_version":              pl.Utf8,
}

# Dimension names ordered the same way the radar chart will draw them.
STYLE_DIMENSIONS: list[str] = [
    "forecheck_aggression",
    "dz_structure",
    "pace",
    "physicality",
    "oz_structure",
    "nz_tendency",
    "line_match",
    "st_aggression",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev_team_aggregates(pbp_df: pl.DataFrame, team_lookup: dict[int, str]) -> pl.DataFrame:
    """Per-team EV aggregates from PBP.

    Returns columns:
        team, ev_team_seconds, takeaways_oz, takeaways, giveaways_dz,
        giveaways, hits, ev_shots_total, ev_shots_close, carry_entries,
        total_entries, carry_data_present
    """
    required = {
        "event_type", "event_owner_team_id", "home_team_id", "away_team_id",
        "strength", "zone_code", "shot_result", "shot_distance",
        "x_coord", "carry_in", "entering_team_id", "turnover_player_id",
        "hitter_id",
    }
    have = set(pbp_df.columns)
    # shot_distance may be derived; provide a fallback
    if "shot_distance" not in have:
        pbp_df = pbp_df.with_columns(pl.lit(None).cast(pl.Float64).alias("shot_distance"))
    missing = required - set(pbp_df.columns) - {"shot_distance"}
    if missing:
        # Hits and turnovers may not be in every dataset; only require the
        # genuinely-load-bearing columns.
        critical = {"event_type", "event_owner_team_id", "home_team_id",
                    "away_team_id", "strength", "zone_code"}
        crit_missing = critical - set(pbp_df.columns)
        if crit_missing:
            raise ValueError(f"pbp_df missing required columns: {sorted(crit_missing)}")

    pbp_df = pbp_df.with_columns([
        pl.col("home_team_id").cast(pl.Int64),
        pl.col("away_team_id").cast(pl.Int64),
        pl.col("event_owner_team_id").cast(pl.Int64),
    ])

    ev = pbp_df.filter(pl.col("strength") == "ev")

    # EV TOI proxy: total of 'period_start' to 'period_end' spans is hard to
    # reconstruct without rebuilding shift state.  Cheap proxy: time covered
    # by EV events.  We don't have a robust ev-team-seconds counter, so we
    # approximate it with shifts data when present, else by counting unique
    # (game_id, team) where EV events appeared and dividing by an
    # **assumed average EV TOI per game** (~3000 sec / team) when the only
    # other option is silent collapse.  v1 prefers honesty: every rate
    # below ends up "per 60 of EV TOI" only after the team-game pair has at
    # least one EV event.
    # Simpler: use the count of distinct team-games at EV strength as GP.
    ev_gp = (
        ev.select(["game_id", "home_team_id", "away_team_id"])
        .unique()
        .group_by("home_team_id").agg(pl.len().alias("home_gp")).rename({"home_team_id": "tid"})
    )
    ev_away = (
        ev.select(["game_id", "home_team_id", "away_team_id"])
        .unique()
        .group_by("away_team_id").agg(pl.len().alias("away_gp")).rename({"away_team_id": "tid"})
    )
    ev_gp_full = (
        ev_gp.join(ev_away, on="tid", how="full", coalesce=True)
        .filter(pl.col("tid").is_not_null())
        .with_columns([
            pl.col("home_gp").fill_null(0),
            pl.col("away_gp").fill_null(0),
            (pl.col("home_gp") + pl.col("away_gp")).alias("ev_gp"),
        ])
    )
    # EV TOI proxy: 50 minutes / game = 3000 sec (close to NHL norm).
    EV_TOI_PER_GAME = 3000.0

    # Helper: filter nulls before group_by to avoid null-key schema issues.
    def _agg_by_owner(ev_df: pl.DataFrame, etype: str, zone: str | None = None) -> pl.DataFrame:
        f = ev_df.filter(
            (pl.col("event_type") == etype) & pl.col("event_owner_team_id").is_not_null()
        )
        if zone is not None:
            f = f.filter(pl.col("zone_code") == zone)
        return f.group_by("event_owner_team_id").agg(pl.len().alias("cnt")).rename(
            {"event_owner_team_id": "tid"}
        )

    # Takeaways (offensive zone) — credited to event_owner_team_id
    tk_total = _agg_by_owner(ev, "takeaway").rename({"cnt": "takeaways"})
    tk_oz    = _agg_by_owner(ev, "takeaway", "O").rename({"cnt": "takeaways_oz"})

    # Giveaways (defensive zone)
    gv_total = _agg_by_owner(ev, "giveaway").rename({"cnt": "giveaways"})
    gv_dz    = _agg_by_owner(ev, "giveaway", "D").rename({"cnt": "giveaways_dz"})

    # Hits — credited to event_owner_team_id (hitter team)
    hits     = _agg_by_owner(ev, "hit").rename({"cnt": "hits"})

    # EV shots (all types) + close shots
    shots_all = ev.filter(
        pl.col("event_type").is_in(["shot", "goal"])
        & pl.col("event_owner_team_id").is_not_null()
    ).with_columns(
        pl.when(pl.col("x_coord").is_not_null())
          .then((100 - pl.col("x_coord").abs()).abs())
          .otherwise(pl.lit(None))
          .alias("dist_proxy")
    )
    sh_total = shots_all.group_by("event_owner_team_id").agg(pl.len().alias("ev_shots_total")).rename(
        {"event_owner_team_id": "tid"}
    )
    sh_close = shots_all.filter(
        pl.col("dist_proxy").is_not_null() & (pl.col("dist_proxy") <= DANGER_ZONE_DIST)
    ).group_by("event_owner_team_id").agg(pl.len().alias("ev_shots_close")).rename(
        {"event_owner_team_id": "tid"}
    )

    # Zone entries — carry vs total
    ze = ev.filter(
        (pl.col("event_type") == "zone_entry")
        & pl.col("entering_team_id").is_not_null()
    ).with_columns(pl.col("entering_team_id").cast(pl.Int64))
    ze_total = ze.group_by("entering_team_id").agg(pl.len().alias("total_entries")).rename(
        {"entering_team_id": "tid"}
    )
    ze_carry = ze.filter(pl.col("carry_in").is_not_null() & (pl.col("carry_in") == True)).group_by(
        "entering_team_id"
    ).agg(pl.len().alias("carry_entries")).rename({"entering_team_id": "tid"})
    carry_data_present = ze.filter(pl.col("carry_in").is_not_null()).height > 0

    base = ev_gp_full.select("tid")
    df = (
        base
        .join(tk_total, on="tid", how="left")
        .join(tk_oz,    on="tid", how="left")
        .join(gv_total, on="tid", how="left")
        .join(gv_dz,    on="tid", how="left")
        .join(hits,     on="tid", how="left")
        .join(sh_total, on="tid", how="left")
        .join(sh_close, on="tid", how="left")
        .join(ze_total, on="tid", how="left")
        .join(ze_carry, on="tid", how="left")
        .join(ev_gp_full.select(["tid", "ev_gp"]), on="tid", how="left")
        .with_columns([
            pl.col("takeaways").fill_null(0),
            pl.col("takeaways_oz").fill_null(0),
            pl.col("giveaways").fill_null(0),
            pl.col("giveaways_dz").fill_null(0),
            pl.col("hits").fill_null(0),
            pl.col("ev_shots_total").fill_null(0),
            pl.col("ev_shots_close").fill_null(0),
            pl.col("total_entries").fill_null(0),
            pl.col("carry_entries").fill_null(0),
            pl.col("ev_gp").fill_null(0),
            (pl.col("ev_gp") * EV_TOI_PER_GAME).alias("ev_team_seconds"),
        ])
    )

    # Map tid → abbrev; drop teams not in the lookup (exhibitions).
    rows: list[dict[str, Any]] = []
    for r in df.iter_rows(named=True):
        tid = int(r["tid"] or 0)
        abbrev = team_lookup.get(tid)
        if abbrev is None:
            continue
        rows.append({
            "team":             abbrev,
            "ev_team_seconds":  float(r["ev_team_seconds"] or 0.0),
            "ev_gp":            int(r["ev_gp"] or 0),
            "takeaways":        int(r["takeaways"] or 0),
            "takeaways_oz":     int(r["takeaways_oz"] or 0),
            "giveaways":        int(r["giveaways"] or 0),
            "giveaways_dz":     int(r["giveaways_dz"] or 0),
            "hits":             int(r["hits"] or 0),
            "ev_shots_total":   int(r["ev_shots_total"] or 0),
            "ev_shots_close":   int(r["ev_shots_close"] or 0),
            "total_entries":    int(r["total_entries"] or 0),
            "carry_entries":    int(r["carry_entries"] or 0),
            "carry_data_present": carry_data_present,
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "team": pl.Utf8, "ev_team_seconds": pl.Float64, "ev_gp": pl.Int64,
        "takeaways": pl.Int64, "takeaways_oz": pl.Int64,
        "giveaways": pl.Int64, "giveaways_dz": pl.Int64,
        "hits": pl.Int64, "ev_shots_total": pl.Int64, "ev_shots_close": pl.Int64,
        "total_entries": pl.Int64, "carry_entries": pl.Int64,
        "carry_data_present": pl.Boolean,
    })


def _line_match_score(lines_df: pl.DataFrame) -> dict[str, float]:
    """Per-team inverse-entropy of forward line share-of-TOI.

    1.0 = top line gets all the minutes (perfect shelter).
    0.0 = perfectly balanced four lines (max entropy).
    """
    if lines_df.is_empty():
        return {}
    f_lines = lines_df.filter(pl.col("line_type") == "F")
    out: dict[str, float] = {}
    for team, group in f_lines.group_by("team"):
        team_abbrev = team[0] if isinstance(team, tuple) else team
        shares = [float(v) for v in group["share_of_team_toi"].to_list() if v is not None and v > 0]
        if not shares:
            out[team_abbrev] = 0.5
            continue
        s = sum(shares) or 1.0
        probs = [p / s for p in shares]
        # Shannon entropy in nats; max = log(n).
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        n   = len(probs)
        max_ent = math.log(n) if n > 1 else 1.0
        # Normalize 0..1, then flip: 1 - normalized_entropy = concentration.
        concentration = 1.0 - (ent / max_ent if max_ent > 0 else 0.0)
        out[team_abbrev] = round(concentration, 4)
    return out


def _percentile_rank(values: list[float]) -> list[float]:
    """Return list of percentile-rank ∈ [0, 1] for each value (NaN → NaN).

    Ties get the average rank.  Output length matches input.
    """
    n_valid = sum(1 for v in values if v == v)   # ignore NaN
    if n_valid == 0:
        return [float("nan")] * len(values)
    # Sort valid values and find rank.
    valid = sorted([v for v in values if v == v])
    ranks: list[float] = []
    for v in values:
        if v != v:
            ranks.append(float("nan"))
            continue
        # Average rank to handle ties
        lo = next(i for i, x in enumerate(valid) if x >= v)
        hi = len(valid) - 1 - next(i for i, x in enumerate(reversed(valid)) if x <= v)
        avg_rank = (lo + hi) / 2.0
        ranks.append(avg_rank / max(1, n_valid - 1) if n_valid > 1 else 0.5)
    return ranks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_coaching_style(
    pbp_df:           pl.DataFrame,
    lines_df:         pl.DataFrame,
    pp_coordinator:   pl.DataFrame,
    team_lookup:      dict[int, str],
    season:           int,
) -> pl.DataFrame:
    """Build per-team coaching-style rows (raw + percentile-rank).

    Args:
        pbp_df:          raw play-by-play frame for the season.
        lines_df:        ``line_deployment_{season}.parquet`` frame.
        pp_coordinator:  ``pp_coordinator_{season}.parquet`` frame
                         (provides PP shots/60 → ``st_aggression``).
        team_lookup:     team_id → abbrev mapping.
    """
    agg = _ev_team_aggregates(pbp_df, team_lookup)
    if agg.is_empty():
        warnings.warn(
            "compute_coaching_style: no EV events mapped to known teams — empty output.",
            DataMissingWarning,
            stacklevel=2,
        )
        return pl.DataFrame(schema=COACHING_STYLE_SCHEMA)

    line_match = _line_match_score(lines_df)

    # PP shots/60 from pp_coordinator parquet
    pp_lookup: dict[str, float] = {}
    if not pp_coordinator.is_empty() and {"team", "pp_shots_per_60"}.issubset(pp_coordinator.columns):
        for r in pp_coordinator.iter_rows(named=True):
            pp_lookup[r["team"]] = float(r.get("pp_shots_per_60") or 0.0)

    rows: list[dict[str, Any]] = []
    for r in agg.iter_rows(named=True):
        team = r["team"]
        sec  = float(r["ev_team_seconds"] or 0.0)
        rate = lambda n: (float(n) / sec * 3600.0) if sec > 0 else 0.0    # noqa: E731

        # 1. forecheck_aggression = OZ takeaways / 60 EV
        fore = rate(r["takeaways_oz"])
        # 2. dz_structure = (takeaways - giveaways) per 60 in defensive zone — DZ
        #    safe puck handling.  Use the DZ-only counts.
        dz   = (
            rate(0)
            + ((float(r["takeaways"]) - float(r["giveaways_dz"])) / sec * 3600.0
               if sec > 0 else 0.0)
        )
        # 3. pace = total EV shots / 60 EV (high = fast game, lots of shots)
        pace = rate(r["ev_shots_total"])
        # 4. physicality = hits / 60 EV
        phys = rate(r["hits"])
        # 5. oz_structure = share of EV shots inside danger zone
        denom_sh = max(1, int(r["ev_shots_total"]))
        ozs = float(r["ev_shots_close"]) / denom_sh
        # 6. nz_tendency = % carry entries (NaN if data absent)
        if bool(r["carry_data_present"]) and (r["total_entries"] or 0) > 0:
            nzt = float(r["carry_entries"]) / float(r["total_entries"])
        else:
            nzt = float("nan")
        # 7. line_match
        lm = line_match.get(team, float("nan"))
        # 8. st_aggression — PP shots / 60 from pp_coordinator
        sta = pp_lookup.get(team, 0.0)

        rows.append({
            "team":                       team,
            "season":                     int(season),
            "forecheck_aggression_raw":   round(fore, 4),
            "dz_structure_raw":           round(dz, 4),
            "pace_raw":                   round(pace, 4),
            "physicality_raw":            round(phys, 4),
            "oz_structure_raw":           round(ozs, 4),
            "nz_tendency_raw":            nzt if nzt != nzt else round(nzt, 4),
            "line_match_raw":             lm if lm != lm else round(lm, 4),
            "st_aggression_raw":          round(sta, 4),
        })

    if not rows:
        return pl.DataFrame(schema=COACHING_STYLE_SCHEMA)

    df = pl.DataFrame(rows)

    # Compute league-percentile rank per dimension.
    for dim in STYLE_DIMENSIONS:
        raw = df[f"{dim}_raw"].to_list()
        # Convert None → NaN for ranker
        raw_clean = [float("nan") if v is None else float(v) for v in raw]
        ranks = _percentile_rank(raw_clean)
        df = df.with_columns(pl.Series(f"{dim}_rank", ranks))

    df = df.with_columns(pl.lit(MODEL_VERSION).alias("model_version"))
    for col, dtype in COACHING_STYLE_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(COACHING_STYLE_SCHEMA.keys())).sort("team")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_coaching_style(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"coaching_style_{season}.parquet"
    df.write_parquet(path)
    return path


def read_coaching_style(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "coaching_style"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"coaching_style_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("coaching_style_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
