"""Per-Player Day-of-Week Signature + Broadcast Context — Feature 4.21.

Layer 1: Per-player z-score on each day of the week — does this player
consistently outperform their own average on a specific day?
``player_dow_signature[player_id][day] ∈ [-2.0, +2.0]``

Layer 2: Game-level broadcast context flags (rivalry matchup, etc.).

Data flow
---------
1. Build ``game_id → date → day_of_week`` mapping from the NHL schedule
   API (cached to ``data/game_dates/game_dates_{season}.parquet``).
2. From PBP goals: per-player goals scored per day-of-week.
3. Compare vs. player's overall goals/game rate → z-score.
4. Bayesian shrinkage: players with small sample on a given day pull
   toward zero.  ``shrunk = raw_z × min(1, gp_on_day / MIN_GP)``.

Output: ``dow_signature/dow_signature_{season}.parquet``

V1 caveat
---------
Without game dates in the ingested PBP, the script fetches the
schedule from the NHL API at training time.  If the API is unreachable,
it falls back to ``game_dates_{season}.parquet`` if previously cached.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl


MODEL_VERSION = "dow_signature_v1"

MIN_GP       = 20
MIN_GP_DAY   = 5
DOW_NAMES    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

RIVALRY_PAIRS = [
    ("CGY", "EDM"),   # Battle of Alberta
    ("TOR", "MTL"),
    ("PIT", "PHI"),
    ("BOS", "BUF"),
    ("NYR", "NJD"),
    ("CHI", "DET"),
    ("WSH", "PIT"),
    ("COL", "MIN"),
    ("LAK", "ANA"),
    ("VAN", "EDM"),
]


class DataMissingWarning(UserWarning):
    pass


DOW_SIGNATURE_SCHEMA: dict[str, pl.DataType] = {
    "player_id":        pl.Int64,
    "player_name":      pl.Utf8,
    "season":           pl.Int64,
    "career_gp":        pl.Int64,
    "goals_total":      pl.Int64,
    "best_day":         pl.Utf8,
    "best_day_zscore":  pl.Float64,
    "worst_day":        pl.Utf8,
    "worst_day_zscore": pl.Float64,
    "mon_z":            pl.Float64,
    "tue_z":            pl.Float64,
    "wed_z":            pl.Float64,
    "thu_z":            pl.Float64,
    "fri_z":            pl.Float64,
    "sat_z":            pl.Float64,
    "sun_z":            pl.Float64,
    "model_version":    pl.Utf8,
}

BROADCAST_CONTEXT_SCHEMA: dict[str, pl.DataType] = {
    "game_id":            pl.Int64,
    "home_team":          pl.Utf8,
    "away_team":          pl.Utf8,
    "is_rivalry":         pl.Boolean,
    "rivalry_name":       pl.Utf8,
    "day_of_week":        pl.Utf8,
    "is_saturday":        pl.Boolean,
    "season":             pl.Int64,
    "model_version":      pl.Utf8,
}


# ---------------------------------------------------------------------------
# Game date helper
# ---------------------------------------------------------------------------


def fetch_game_dates(season: int, cache_dir: Path) -> pl.DataFrame:
    """Fetch game_id → date mapping from NHL API, with file cache.

    Returns DataFrame with columns: game_id (Int64), game_date (Utf8),
    day_of_week (Int64: 0=Mon..6=Sun), day_name (Utf8).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"game_dates_{season}.parquet"

    if cache_path.exists():
        return pl.read_parquet(cache_path)

    try:
        import httpx
        from datetime import datetime

        url = f"https://api-web.nhle.com/v1/schedule/{season}-10-01"
        dates_data: list[dict] = []

        # Fetch the full season schedule week by week
        current = datetime(season, 10, 1)
        end = datetime(season + 1, 7, 1)
        seen_ids: set[int] = set()

        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                resp = httpx.get(f"https://api-web.nhle.com/v1/schedule/{date_str}", timeout=10)
                if resp.status_code != 200:
                    current = current.replace(day=current.day + 7) if current.day <= 24 else current.replace(month=current.month + 1, day=1)
                    continue
                data = resp.json()
                for week in data.get("gameWeek", []):
                    gdate = week.get("date", "")
                    for game in week.get("games", []):
                        gid = game.get("id")
                        if gid and gid not in seen_ids:
                            seen_ids.add(gid)
                            dates_data.append({"game_id": int(gid), "game_date": gdate})
            except Exception:
                pass
            # Jump forward 7 days
            from datetime import timedelta
            current += timedelta(days=7)

        if not dates_data:
            warnings.warn(f"fetch_game_dates: NHL API returned no games for {season}.",
                          DataMissingWarning, stacklevel=2)
            return pl.DataFrame(schema={"game_id": pl.Int64, "game_date": pl.Utf8,
                                        "day_of_week": pl.Int64, "day_name": pl.Utf8})

        df = pl.DataFrame(dates_data)
        # Derive day_of_week from game_date string
        df = df.with_columns([
            pl.col("game_date").str.to_date("%Y-%m-%d").dt.weekday().alias("day_of_week"),
            pl.col("game_date").str.to_date("%Y-%m-%d").dt.strftime("%a").alias("day_name"),
        ])
        df.write_parquet(cache_path)
        return df

    except ImportError:
        warnings.warn("fetch_game_dates: httpx not available.", DataMissingWarning, stacklevel=2)
        return pl.DataFrame(schema={"game_id": pl.Int64, "game_date": pl.Utf8,
                                    "day_of_week": pl.Int64, "day_name": pl.Utf8})


# ---------------------------------------------------------------------------
# DOW Signature
# ---------------------------------------------------------------------------


def compute_dow_signature(
    pbp_df:       pl.DataFrame,
    game_dates:   pl.DataFrame,
    season:       int,
    min_gp:       int = MIN_GP,
    min_gp_day:   int = MIN_GP_DAY,
) -> pl.DataFrame:
    """Compute per-player day-of-week goal-scoring z-scores.

    Args:
        pbp_df:      raw PBP with goal events (scorer_id, game_id).
        game_dates:  game_id → day_of_week mapping.
        season:      NHL season start year.
        min_gp:      minimum career GP to include a player.
        min_gp_day:  minimum GP on a specific day for full-weight z-score.
    """
    if pbp_df.is_empty() or game_dates.is_empty():
        warnings.warn("compute_dow_signature: empty PBP or game_dates.",
                       DataMissingWarning, stacklevel=2)
        return pl.DataFrame(schema=DOW_SIGNATURE_SCHEMA)

    goals = pbp_df.filter(
        (pl.col("event_type") == "goal")
        & pl.col("scorer_id").is_not_null()
    ).select(["game_id", "scorer_id"])

    # Join game dates
    goals = goals.join(game_dates.select(["game_id", "day_of_week"]), on="game_id", how="left")
    goals = goals.filter(pl.col("day_of_week").is_not_null())

    if goals.is_empty():
        return pl.DataFrame(schema=DOW_SIGNATURE_SCHEMA)

    # All game appearances per player (from PBP — any event where they're referenced)
    all_scorers = goals["scorer_id"].unique().to_list()

    # Per-player total goals + GP (from distinct game_ids in goals)
    player_games = (
        pbp_df.filter(
            pl.col("event_type").is_in(["shot", "goal", "faceoff", "hit"])
            & pl.col("event_owner_team_id").is_not_null()
        )
        .select(["game_id", "event_owner_team_id"])
        .unique()
    )

    # Goals per (player, day_of_week)
    goals_by_dow = (
        goals.group_by(["scorer_id", "day_of_week"])
        .agg(pl.len().alias("goals_on_day"))
    )

    # Games per (player, day_of_week) — count distinct game_ids
    games_with_dates = (
        pbp_df.filter(pl.col("scorer_id").is_not_null())
        .select(["game_id", "scorer_id"]).unique()
        .join(game_dates.select(["game_id", "day_of_week"]), on="game_id", how="left")
        .filter(pl.col("day_of_week").is_not_null())
        .group_by(["scorer_id", "day_of_week"])
        .agg(pl.col("game_id").n_unique().alias("gp_on_day"))
    )

    # Total per player
    player_totals = (
        goals.group_by("scorer_id").agg([
            pl.len().alias("goals_total"),
            pl.col("game_id").n_unique().alias("career_gp"),
        ])
        .filter(pl.col("career_gp") >= min_gp)
    )

    if player_totals.is_empty():
        return pl.DataFrame(schema=DOW_SIGNATURE_SCHEMA)

    # Build player name lookup
    name_cols = [c for c in pbp_df.columns if "scorer_id" in c]
    # We don't have scorer_name in PBP; use shots parquet
    name_map: dict[int, str] = {}

    rows: list[dict[str, Any]] = []
    for pr in player_totals.iter_rows(named=True):
        pid = int(pr["scorer_id"])
        total_goals = int(pr["goals_total"])
        career_gp = int(pr["career_gp"])
        avg_gpg = total_goals / career_gp if career_gp > 0 else 0.0

        player_dow_goals = goals_by_dow.filter(pl.col("scorer_id") == pid)
        player_dow_games = games_with_dates.filter(pl.col("scorer_id") == pid)

        z_scores: dict[str, float] = {}
        for dow in range(1, 8):  # Polars weekday: 1=Mon..7=Sun
            dow_goals_row = player_dow_goals.filter(pl.col("day_of_week") == dow)
            dow_games_row = player_dow_games.filter(pl.col("day_of_week") == dow)

            g_on_day = int(dow_goals_row["goals_on_day"].sum() or 0) if not dow_goals_row.is_empty() else 0
            gp_day = int(dow_games_row["gp_on_day"].sum() or 0) if not dow_games_row.is_empty() else 0

            if gp_day == 0:
                z_scores[DOW_NAMES[dow - 1]] = 0.0
                continue

            observed_gpg = g_on_day / gp_day
            # Z-score: (observed - expected) / sqrt(expected_var / n)
            # Poisson approximation: var ≈ mean → se = sqrt(avg_gpg / gp_day)
            se = (avg_gpg / gp_day) ** 0.5 if avg_gpg > 0 and gp_day > 0 else 1.0
            raw_z = (observed_gpg - avg_gpg) / se if se > 0 else 0.0
            # Bayesian shrinkage
            shrink = min(1.0, gp_day / min_gp_day)
            z = max(-2.0, min(2.0, raw_z * shrink))
            z_scores[DOW_NAMES[dow - 1]] = round(z, 4)

        best_day = max(z_scores, key=z_scores.get)  # type: ignore
        worst_day = min(z_scores, key=z_scores.get)  # type: ignore

        rows.append({
            "player_id":       pid,
            "player_name":     name_map.get(pid, ""),
            "season":          int(season),
            "career_gp":       career_gp,
            "goals_total":     total_goals,
            "best_day":        best_day,
            "best_day_zscore": z_scores[best_day],
            "worst_day":       worst_day,
            "worst_day_zscore": z_scores[worst_day],
            **{f"{d.lower()}_z": z_scores.get(d, 0.0) for d in DOW_NAMES},
            "model_version":   MODEL_VERSION,
        })

    if not rows:
        return pl.DataFrame(schema=DOW_SIGNATURE_SCHEMA)

    df = pl.DataFrame(rows)
    for col, dtype in DOW_SIGNATURE_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(DOW_SIGNATURE_SCHEMA.keys())).sort("best_day_zscore", descending=True)


# ---------------------------------------------------------------------------
# Broadcast Context (Layer 2)
# ---------------------------------------------------------------------------


def compute_broadcast_context(
    game_dates:   pl.DataFrame,
    team_lookup:  dict[int, str],
    pbp_df:       pl.DataFrame,
    season:       int,
) -> pl.DataFrame:
    """Tag each game with rivalry flag + day-of-week context."""
    if game_dates.is_empty() or pbp_df.is_empty():
        return pl.DataFrame(schema=BROADCAST_CONTEXT_SCHEMA)

    game_teams = (
        pbp_df.select(["game_id", "home_team_id", "away_team_id"]).unique()
        .with_columns([
            pl.col("home_team_id").cast(pl.Int64),
            pl.col("away_team_id").cast(pl.Int64),
        ])
    )

    joined = game_teams.join(game_dates, on="game_id", how="left")

    rivalry_set = set()
    for a, b in RIVALRY_PAIRS:
        rivalry_set.add((a, b))
        rivalry_set.add((b, a))

    rows: list[dict[str, Any]] = []
    for r in joined.iter_rows(named=True):
        gid = int(r["game_id"])
        home = team_lookup.get(int(r["home_team_id"] or 0), "")
        away = team_lookup.get(int(r["away_team_id"] or 0), "")
        dow  = int(r.get("day_of_week") or 0)
        day_name = DOW_NAMES[dow - 1] if 1 <= dow <= 7 else ""

        is_rivalry = (home, away) in rivalry_set
        rivalry_name = ""
        if is_rivalry:
            for a, b in RIVALRY_PAIRS:
                if (home == a and away == b) or (home == b and away == a):
                    rivalry_name = f"{a} vs {b}"
                    break

        rows.append({
            "game_id":       gid,
            "home_team":     home,
            "away_team":     away,
            "is_rivalry":    is_rivalry,
            "rivalry_name":  rivalry_name,
            "day_of_week":   day_name,
            "is_saturday":   dow == 6,
            "season":        int(season),
            "model_version": MODEL_VERSION,
        })

    df = pl.DataFrame(rows)
    for col, dtype in BROADCAST_CONTEXT_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(BROADCAST_CONTEXT_SCHEMA.keys()))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_dow_signature(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"dow_signature_{season}.parquet"
    df.write_parquet(path)
    return path


def write_broadcast_context(df: pl.DataFrame, output_dir: Path, season: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"broadcast_context_{season}.parquet"
    df.write_parquet(path)
    return path


def read_dow_signature(data_dir: Path, season: int | None = None) -> pl.DataFrame | None:
    d = Path(data_dir) / "dow_signature"
    if not d.exists():
        return None
    if season is not None:
        p = d / f"dow_signature_{season}.parquet"
        return pl.read_parquet(p) if p.exists() else None
    candidates = sorted(d.glob("dow_signature_*.parquet"))
    return pl.read_parquet(candidates[-1]) if candidates else None
