"""Phase 17 Confidence Sub-signals — Features 17.1 through 17.22.

Each function takes pre-loaded raw inputs and returns a Polars DataFrame
with the per-(player, game) or per-(team, game) signal score in [-1, +1].
Missing data returns an empty frame — the composite gracefully treats
absent signals as zero contribution.

Why one file rather than 22:
    - The signals are mostly thin reads/transforms of existing parquets.
    - Keeping them together makes the composite glue script readable.
    - Bob can audit all weights and signals in two adjacent files
      (``composite_confidence.py`` for the weights, this file for the
      signal definitions).

Sign conventions:
    All signals are signed in [-1, +1]. Positive = confidence boost,
    negative = confidence dent. Each function documents its own normalization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


# Common shape: every signal returns a frame with these keys plus its own
# signal column. The composite joiner uses (player_id, game_id) for player
# signals and (team, game_id) for team signals (then broadcasts team values
# to all players on that team).
PLAYER_SIGNAL_BASE = ("player_id", "game_id", "game_date")
TEAM_SIGNAL_BASE   = ("team",      "game_id", "game_date")


def _clamp_series(s: pl.Series, lo: float, hi: float) -> pl.Series:
    return s.clip(lo, hi)


def _empty(cols: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame({k: pl.Series([], dtype=t) for k, t in cols.items()})


# ───────────────────────────────────────────────────────────────────────────
# 17.1 — Hot Hand signal (REUSE 2.28)
# Range: scaled to [-1, +1] from existing hot_hand_score in [-2, +2]
# ───────────────────────────────────────────────────────────────────────────

def hot_hand_signal(hot_hand_df: pl.DataFrame) -> pl.DataFrame:
    if len(hot_hand_df) == 0 or "hot_hand_score" not in hot_hand_df.columns:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "hot_hand_signal": pl.Float64})
    cols = hot_hand_df.columns
    select_exprs = [
        pl.col("player_id"),
        (pl.col("hot_hand_score").clip(-2.0, 2.0) / 2.0).alias("hot_hand_signal"),
    ]
    if "game_id" in cols:
        select_exprs.append(pl.col("game_id"))
    else:
        select_exprs.append(pl.lit(0).cast(pl.Int64).alias("game_id"))
    if "game_date" in cols:
        select_exprs.append(pl.col("game_date"))
    else:
        select_exprs.append(pl.lit("").alias("game_date"))
    return hot_hand_df.select(select_exprs)


# ───────────────────────────────────────────────────────────────────────────
# 17.2 — EWMA Form signal (REUSE 2.13)
# Range: scaled by season std to [-1, +1]
# ───────────────────────────────────────────────────────────────────────────

def ewma_form_signal(ewma_df: pl.DataFrame) -> pl.DataFrame:
    if len(ewma_df) == 0:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "ewma_form_signal": pl.Float64})
    # EWMA parquets typically expose ``ewma_form`` or ``form_z``.
    col = None
    for cand in ("form_z", "ewma_form_z", "form_score", "ewma_form"):
        if cand in ewma_df.columns:
            col = cand
            break
    if col is None:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "ewma_form_signal": pl.Float64})
    return ewma_df.select([
        pl.col("player_id"),
        pl.col("game_id") if "game_id" in ewma_df.columns else pl.lit(0).cast(pl.Int64).alias("game_id"),
        pl.col("game_date") if "game_date" in ewma_df.columns else pl.lit("").alias("game_date"),
        (pl.col(col).cast(pl.Float64).clip(-2.0, 2.0) / 2.0).alias("ewma_form_signal"),
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.3 — TOI Trust Trend (5-game TOI z vs 30-game baseline)
# Built from toi_load parquet's per-player rolling z-score
# ───────────────────────────────────────────────────────────────────────────

def toi_trust_trend(toi_df: pl.DataFrame) -> pl.DataFrame:
    if len(toi_df) == 0 or "toi_spike_z" not in toi_df.columns:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "toi_trust_trend": pl.Float64})
    # TOI spike z > 0 means MORE ice time than baseline = trust climbing.
    return toi_df.select([
        pl.col("player_id"),
        pl.col("game_id"),
        pl.col("game_date") if "game_date" in toi_df.columns else pl.lit("").alias("game_date"),
        (pl.col("toi_spike_z").cast(pl.Float64).clip(-2.5, 2.5) / 2.5).alias("toi_trust_trend"),
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.4 — Role / Linemate-Quality Delta
# Built from matchup parquet's QoT (quality of teammates) per game
# Positive QoT delta = promoted to better line
# ───────────────────────────────────────────────────────────────────────────

def role_usage_delta(matchup_df: pl.DataFrame) -> pl.DataFrame:
    cols = matchup_df.columns if len(matchup_df) > 0 else []
    qot_col = next((c for c in ("qot_xgf60", "qot", "teammate_xgf60") if c in cols), None)
    if qot_col is None:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "role_usage_delta": pl.Float64})
    # Rolling 5-game vs 30-game delta per player, normalized.
    df = matchup_df.sort(["player_id", "game_date"])
    df = df.with_columns([
        pl.col(qot_col).rolling_mean(window_size=5, min_periods=2).over("player_id").alias("_q5"),
        pl.col(qot_col).rolling_mean(window_size=30, min_periods=10).over("player_id").alias("_q30"),
    ])
    df = df.with_columns(
        ((pl.col("_q5") - pl.col("_q30")) / pl.col("_q30").abs().clip(0.05, None))
            .clip(-1.0, 1.0)
            .alias("role_usage_delta")
    )
    return df.select([
        "player_id",
        pl.col("game_id") if "game_id" in df.columns else pl.lit(0).cast(pl.Int64).alias("game_id"),
        "game_date",
        "role_usage_delta",
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.5 — Healthy Scratch Flag
# Binary: -1 if player has no PBP appearances in N of last M games AND no
# concurrent injury record. 0 otherwise.
# Conservative: returns empty if either roster or injury data missing.
# ───────────────────────────────────────────────────────────────────────────

def healthy_scratch_flag(
    appearances_df: pl.DataFrame,
    injury_df:      pl.DataFrame,
    rosters_df:     pl.DataFrame,
    as_of_date:     str,
    lookback_games: int = 5,
    miss_threshold: int = 2,
) -> pl.DataFrame:
    """One row per active roster player; signal = -1 if scratched, else 0."""
    if len(rosters_df) == 0:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "healthy_scratch_flag": pl.Float64})
    if "player_id" not in rosters_df.columns:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "healthy_scratch_flag": pl.Float64})

    injured = set()
    if len(injury_df) > 0 and "player_id" in injury_df.columns and "status" in injury_df.columns:
        injured = set(
            injury_df.filter(pl.col("status").is_in(["OUT", "IR", "DAY_TO_DAY", "PROBABLE"]))
                     ["player_id"].cast(pl.Int64).to_list()
        )

    # Count missed games per player by joining appearances against the
    # latest ``lookback_games`` games per team (best-effort).
    # v1 simplification: missed = no appearance in any of the last
    # ``lookback_games`` distinct game dates for the player's team.
    if len(appearances_df) == 0:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "healthy_scratch_flag": pl.Float64})

    # Count distinct games per player (chronological), keep only the last
    # ``lookback_games`` worth — anyone not appearing is candidate-scratched.
    recent = (
        appearances_df.select(["player_id", "game_id", "game_date"])
                      .unique()
                      .sort("game_date", descending=True)
                      .head(lookback_games * 30)   # generous slice
    )
    appeared_pids = set(recent["player_id"].cast(pl.Int64).to_list())

    out_rows: list[dict] = []
    for pid in rosters_df["player_id"].cast(pl.Int64).to_list():
        if pid in injured:
            continue
        if pid not in appeared_pids:
            out_rows.append({
                "player_id": int(pid),
                "game_id": 0,
                "game_date": as_of_date,
                "healthy_scratch_flag": -0.8,
            })
    if not out_rows:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "healthy_scratch_flag": pl.Float64})
    return pl.DataFrame(out_rows)


# ───────────────────────────────────────────────────────────────────────────
# 17.6 — Point Drought (signed: -1 = long drought, 0 = baseline)
# ───────────────────────────────────────────────────────────────────────────

def point_drought_z(scoring_df: pl.DataFrame) -> pl.DataFrame:
    """Games since last point, normalized by expected drought given xG rate."""
    if len(scoring_df) == 0 or not {
        "player_id", "game_id", "game_date", "games_since_last_point", "expected_drought",
    }.issubset(scoring_df.columns):
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "point_drought_z": pl.Float64})
    return scoring_df.select([
        "player_id", "game_id", "game_date",
        ((-(pl.col("games_since_last_point") - pl.col("expected_drought"))
          / pl.col("expected_drought").clip(1.0, None))
            .clip(-1.0, 1.0))
            .alias("point_drought_z"),
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.7 — Linemate Form Drag
# Avg of current linemates' EWMA form. Cold linemates → negative signal.
# Stub v1: returns empty until linemate-pair table is wired in.
# ───────────────────────────────────────────────────────────────────────────

def linemate_form_drag(chemistry_df: pl.DataFrame, ewma_df: pl.DataFrame) -> pl.DataFrame:
    if len(chemistry_df) == 0 or len(ewma_df) == 0:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "linemate_form_drag": pl.Float64})
    # v1 placeholder — requires per-game linemate-IDs which chemistry_df may
    # not expose directly. Treat as no-effect when the join can't be built.
    return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                   "game_date": pl.Utf8, "linemate_form_drag": pl.Float64})


# ───────────────────────────────────────────────────────────────────────────
# 17.8 — Bounceback Index (deferred to v2 per plan)
# ───────────────────────────────────────────────────────────────────────────

def bounceback_index() -> pl.DataFrame:
    """Deferred to v2. Returns empty frame so composite treats as zero."""
    return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                   "game_date": pl.Utf8, "bounceback_index": pl.Float64})


# ───────────────────────────────────────────────────────────────────────────
# 17.9 — Active Injury Drag
# Negative signal for DTD / fresh-off-IR / concussion-history overlap.
# ───────────────────────────────────────────────────────────────────────────

def active_injury_drag(
    injury_df:     pl.DataFrame,
    concussion_df: pl.DataFrame,
) -> pl.DataFrame:
    if len(injury_df) == 0 or not {"player_id", "status"}.issubset(injury_df.columns):
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "active_injury_drag": pl.Float64})
    severity_map = {
        "DAY_TO_DAY": -0.15,
        "PROBABLE":   -0.05,
        "QUESTIONABLE": -0.10,
        "OUT":        -0.30,
        "IR":         -0.30,
        "LTIR":       -0.30,
    }
    df = injury_df.select(["player_id", "status"]).with_columns(
        pl.col("status").map_elements(
            lambda s: severity_map.get(str(s).upper(), 0.0),
            return_dtype=pl.Float64,
        ).alias("active_injury_drag")
    )

    # Concussion-history overlap deepens the drag (multiplier 1.5x).
    if len(concussion_df) > 0 and "player_id" in concussion_df.columns:
        # If parquet has a flag column, weight; otherwise just presence.
        flag_col = next(
            (c for c in ("has_concussion_history", "concussion_count", "n_concussions")
             if c in concussion_df.columns),
            None,
        )
        if flag_col:
            cdf = concussion_df.select(
                ["player_id",
                 (pl.col(flag_col).cast(pl.Float64) > 0).cast(pl.Float64).alias("_conc")]
            )
            df = df.join(cdf, on="player_id", how="left").with_columns(
                (pl.col("active_injury_drag") * (1.0 + 0.5 * pl.col("_conc").fill_null(0.0)))
                    .clip(-1.0, 0.0)
                    .alias("active_injury_drag")
            ).drop("_conc")

    return df.with_columns([
        pl.lit(0).cast(pl.Int64).alias("game_id"),
        pl.lit("").alias("game_date"),
    ]).select(["player_id", "game_id", "game_date", "active_injury_drag"])


# ───────────────────────────────────────────────────────────────────────────
# 17.10 — Targeting Pressure (hits received differential per TOI)
# Negative signal for high receive-differential.
# ───────────────────────────────────────────────────────────────────────────

def targeting_pressure(contact_df: pl.DataFrame) -> pl.DataFrame:
    cols = contact_df.columns if len(contact_df) > 0 else []
    # Look for explicit hits-received column; fall back to contact_load_score
    if "hits_received_5g" in cols and "hits_delivered_5g" in cols:
        return contact_df.select([
            "player_id",
            pl.col("game_id") if "game_id" in cols else pl.lit(0).cast(pl.Int64).alias("game_id"),
            pl.col("game_date") if "game_date" in cols else pl.lit("").alias("game_date"),
            (((pl.col("hits_received_5g") - pl.col("hits_delivered_5g"))
              .cast(pl.Float64) / 10.0)
                .clip(-1.0, 1.0) * -1.0
            ).alias("targeting_pressure"),
        ])
    if "contact_load_score" in cols:
        # High contact load alone — treat as proxy for being a target.
        return contact_df.select([
            "player_id",
            pl.col("game_id") if "game_id" in cols else pl.lit(0).cast(pl.Int64).alias("game_id"),
            pl.col("game_date") if "game_date" in cols else pl.lit("").alias("game_date"),
            (-(pl.col("contact_load_score").cast(pl.Float64)).clip(0.0, 1.0))
                .alias("targeting_pressure"),
        ])
    return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                   "game_date": pl.Utf8, "targeting_pressure": pl.Float64})


# ───────────────────────────────────────────────────────────────────────────
# 17.11 — Media Sentiment (whiz_brain extension)
# ───────────────────────────────────────────────────────────────────────────

def media_sentiment(whiz_observations: list[dict]) -> pl.DataFrame:
    """Aggregate WhizFeed observations with category='media' per player."""
    if not whiz_observations:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "media_sentiment": pl.Float64})
    rows: list[dict] = []
    for obs in whiz_observations:
        if obs.get("category") != "media":
            continue
        try:
            pid = int(obs.get("player_id") or 0)
        except (TypeError, ValueError):
            continue
        if pid == 0:
            continue
        delta = float(obs.get("sim_delta", {}).get("confidence_delta", 0.0))
        rows.append({
            "player_id":       pid,
            "game_id":         0,
            "game_date":       str(obs.get("game_date") or ""),
            "media_sentiment": max(-0.5, min(0.5, delta)),
        })
    if not rows:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "media_sentiment": pl.Float64})
    df = pl.DataFrame(rows)
    # If multiple observations per player, take the mean.
    return (
        df.group_by("player_id")
          .agg([
              pl.col("media_sentiment").mean().alias("media_sentiment"),
              pl.col("game_date").max().alias("game_date"),
              pl.col("game_id").max().alias("game_id"),
          ])
    )


# ───────────────────────────────────────────────────────────────────────────
# 17.12 — Home / Away Split
# Player's home-vs-road EWMA delta, applied as boost on tonight's venue.
# ───────────────────────────────────────────────────────────────────────────

def home_away_split(
    ewma_df:    pl.DataFrame,
    schedule_df: pl.DataFrame,
    rosters_df:  pl.DataFrame,
    as_of_date:  str,
) -> pl.DataFrame:
    if (len(ewma_df) == 0 or len(schedule_df) == 0 or len(rosters_df) == 0
            or "is_home" not in ewma_df.columns):
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "home_away_split": pl.Float64})
    form_col = next(
        (c for c in ("form_score", "ewma_form", "form_z")
         if c in ewma_df.columns),
        None,
    )
    if form_col is None:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "home_away_split": pl.Float64})
    home_mean = (
        ewma_df.filter(pl.col("is_home"))
               .group_by("player_id")
               .agg(pl.col(form_col).mean().alias("_h"))
    )
    away_mean = (
        ewma_df.filter(~pl.col("is_home"))
               .group_by("player_id")
               .agg(pl.col(form_col).mean().alias("_a"))
    )
    split = home_mean.join(away_mean, on="player_id", how="outer").with_columns(
        ((pl.col("_h").fill_null(0.0) - pl.col("_a").fill_null(0.0))
            .clip(-1.0, 1.0))
            .alias("home_away_split")
    ).select(["player_id", "home_away_split"])
    # Apply only when player is at home tonight (positive sign for the
    # boost; negative if their home boost is actually negative).
    # The schedule context check is left to the composite stage; here we
    # emit the raw split — direction applies to either venue.
    return split.with_columns([
        pl.lit(0).cast(pl.Int64).alias("game_id"),
        pl.lit(as_of_date).alias("game_date"),
    ]).select(["player_id", "game_id", "game_date", "home_away_split"])


# ───────────────────────────────────────────────────────────────────────────
# 17.13 — Trade Rumor Pressure (whiz_brain category='rumor')
# Default direction: negative dent.
# ───────────────────────────────────────────────────────────────────────────

def trade_rumor_pressure(whiz_observations: list[dict]) -> pl.DataFrame:
    if not whiz_observations:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "trade_rumor_pressure": pl.Float64})
    rows: list[dict] = []
    for obs in whiz_observations:
        if obs.get("category") != "rumor":
            continue
        try:
            pid = int(obs.get("player_id") or 0)
        except (TypeError, ValueError):
            continue
        if pid == 0:
            continue
        # confidence_delta overrides default direction; default = -0.3 dent.
        delta = float(obs.get("sim_delta", {}).get("confidence_delta", -0.3))
        rows.append({
            "player_id":            pid,
            "game_id":              0,
            "game_date":            str(obs.get("game_date") or ""),
            "trade_rumor_pressure": max(-1.0, min(1.0, delta)),
        })
    if not rows:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "trade_rumor_pressure": pl.Float64})
    return pl.DataFrame(rows).group_by("player_id").agg([
        pl.col("trade_rumor_pressure").mean().alias("trade_rumor_pressure"),
        pl.col("game_date").max().alias("game_date"),
        pl.col("game_id").max().alias("game_id"),
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.14 — Contract Pressure
# Requires data/contract_sync.py to land. Returns empty until then.
# ───────────────────────────────────────────────────────────────────────────

def contract_pressure(contracts_df: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
    if len(contracts_df) == 0 or "contract_status" not in contracts_df.columns:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "contract_pressure": pl.Float64})
    # walk_year = UFA push (+); extension_stalled = dent (-); other = 0
    pressure_map = {
        "WALK_YEAR_UFA":     +0.10,
        "WALK_YEAR_RFA":     +0.05,
        "EXTENSION_STALLED": -0.10,
        "LAME_DUCK":         -0.10,
        "PROTECTED_NMC":      0.00,
        "PROTECTED_NTC":      0.00,
        "MID_CONTRACT":       0.00,
    }
    df = contracts_df.select(["player_id", "contract_status"]).with_columns(
        pl.col("contract_status").map_elements(
            lambda s: pressure_map.get(str(s).upper(), 0.0),
            return_dtype=pl.Float64,
        ).alias("contract_pressure")
    )
    return df.with_columns([
        pl.lit(0).cast(pl.Int64).alias("game_id"),
        pl.lit(as_of_date).alias("game_date"),
    ]).select(["player_id", "game_id", "game_date", "contract_pressure"])


# ───────────────────────────────────────────────────────────────────────────
# 17.15 — Referee Bias
# Requires data/officials_sync.py + historical PIM-per-ref tables. Empty stub.
# ───────────────────────────────────────────────────────────────────────────

def referee_bias(officials_df: pl.DataFrame, pim_history_df: pl.DataFrame) -> pl.DataFrame:
    if len(officials_df) == 0 or len(pim_history_df) == 0:
        return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "referee_bias": pl.Float64})
    # v1 placeholder — needs sample-size shrinkage that the historical table
    # may not yet provide. Returns empty until both feeds exist.
    return _empty({"player_id": pl.Int64, "game_id": pl.Int64,
                   "game_date": pl.Utf8, "referee_bias": pl.Float64})


# ═══════════════════════════════════════════════════════════════════════════
# TEAM SIGNALS — 17.16 through 4.22
# Each returns per (team, game_date) signal; the composite broadcasts these
# onto every player on that team's roster for the corresponding game.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# 17.16 — Team Streak (signed)
# Positive games_since_last_loss (positive streak); negative
# games_since_last_win (losing streak).
# ───────────────────────────────────────────────────────────────────────────

def team_streak(schedule_df: pl.DataFrame, results_df: pl.DataFrame) -> pl.DataFrame:
    if len(results_df) == 0 or not {
        "team", "game_date", "result",   # result in {"W", "L", "OTL", "SOL"}
    }.issubset(results_df.columns):
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "team_streak": pl.Float64})
    df = results_df.sort(["team", "game_date"])
    rows = df.to_dicts()
    by_team: dict[str, list[dict]] = {}
    for r in rows:
        by_team.setdefault(r["team"], []).append(r)

    out: list[dict] = []
    for team, games in by_team.items():
        streak = 0
        for g in games:
            res = str(g.get("result") or "").upper()
            if res == "W":
                streak = streak + 1 if streak >= 0 else 1
            elif res in ("L", "OTL", "SOL"):
                streak = streak - 1 if streak <= 0 else -1
            normalized = max(-1.0, min(1.0, streak / 8.0))   # 8+ games saturates
            out.append({
                "team":        team,
                "game_id":     int(g.get("game_id") or 0),
                "game_date":   g["game_date"],
                "team_streak": float(normalized),
            })
    return pl.DataFrame(out) if out else _empty({
        "team": pl.Utf8, "game_id": pl.Int64,
        "game_date": pl.Utf8, "team_streak": pl.Float64,
    })


# ───────────────────────────────────────────────────────────────────────────
# 17.17 — Score-Adjusted Corsi Trend (10-game linear trend)
# ───────────────────────────────────────────────────────────────────────────

def score_adj_corsi_trend(team_df: pl.DataFrame) -> pl.DataFrame:
    cols = team_df.columns if len(team_df) > 0 else []
    cf_col = next((c for c in ("score_adj_cf_pct", "cf_pct_score_adj", "cf_pct") if c in cols), None)
    if cf_col is None:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "score_adj_corsi_trend": pl.Float64})
    df = team_df.sort(["team", "game_date"])
    df = df.with_columns([
        pl.col(cf_col).rolling_mean(window_size=5, min_periods=2).over("team").alias("_c5"),
        pl.col(cf_col).rolling_mean(window_size=20, min_periods=8).over("team").alias("_c20"),
    ])
    df = df.with_columns(
        ((pl.col("_c5") - pl.col("_c20")) * 5.0).clip(-1.0, 1.0)
            .alias("score_adj_corsi_trend")
    )
    return df.select([
        "team",
        pl.col("game_id") if "game_id" in df.columns else pl.lit(0).cast(pl.Int64).alias("game_id"),
        "game_date", "score_adj_corsi_trend",
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.18 — Special Teams Trend (PP% + PK% rolling vs baseline)
# ───────────────────────────────────────────────────────────────────────────

def special_teams_trend(team_df: pl.DataFrame) -> pl.DataFrame:
    cols = team_df.columns if len(team_df) > 0 else []
    if not ({"team", "game_date"}.issubset(cols)):
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "special_teams_trend": pl.Float64})
    pp = next((c for c in ("pp_pct", "pp_efficiency") if c in cols), None)
    pk = next((c for c in ("pk_pct", "pk_efficiency") if c in cols), None)
    if not pp and not pk:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "special_teams_trend": pl.Float64})
    df = team_df.sort(["team", "game_date"])
    exprs = []
    if pp:
        exprs.append(
            (pl.col(pp).rolling_mean(window_size=5, min_periods=2).over("team")
             - pl.col(pp).mean().over("team"))
        )
    if pk:
        exprs.append(
            (pl.col(pk).rolling_mean(window_size=5, min_periods=2).over("team")
             - pl.col(pk).mean().over("team"))
        )
    df = df.with_columns(
        (sum(exprs) / len(exprs) * 4.0).clip(-1.0, 1.0)
            .alias("special_teams_trend")
    )
    return df.select([
        "team",
        pl.col("game_id") if "game_id" in df.columns else pl.lit(0).cast(pl.Int64).alias("game_id"),
        "game_date", "special_teams_trend",
    ])


# ───────────────────────────────────────────────────────────────────────────
# 17.19 — Coach Challenge Rate (last-10 games challenge count)
# ───────────────────────────────────────────────────────────────────────────

def coach_challenge_rate(pbp_df: pl.DataFrame) -> pl.DataFrame:
    if len(pbp_df) == 0 or not {"event_type", "event_owner_team_id", "game_id"}.issubset(pbp_df.columns):
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "coach_challenge_rate": pl.Float64})
    # NHL event_type for coach challenges varies; common names:
    challenges = pbp_df.filter(
        pl.col("event_type").is_in(["CHALLENGE", "COACH_CHALLENGE", "CHL"])
    )
    if len(challenges) == 0:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "coach_challenge_rate": pl.Float64})
    by_team_game = (
        challenges.group_by(["event_owner_team_id", "game_id"])
                  .agg(pl.len().alias("n_challenges"))
                  .rename({"event_owner_team_id": "team_id"})
    )
    return by_team_game.with_columns([
        pl.col("team_id").cast(pl.Utf8).alias("team"),
        pl.lit("").alias("game_date"),
        (pl.col("n_challenges").cast(pl.Float64).clip(0.0, 3.0) / 3.0)
            .alias("coach_challenge_rate"),
    ]).select(["team", "game_id", "game_date", "coach_challenge_rate"])


# ───────────────────────────────────────────────────────────────────────────
# 17.20 — Comeback Quality (late-game shot rate when trailing)
# ───────────────────────────────────────────────────────────────────────────

def comeback_quality(pbp_df: pl.DataFrame) -> pl.DataFrame:
    needed = {"event_type", "period", "event_owner_team_id", "game_id",
              "home_team_score", "away_team_score"}
    if len(pbp_df) == 0 or not needed.issubset(pbp_df.columns):
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "comeback_quality": pl.Float64})
    third = pbp_df.filter(pl.col("period") == 3)
    if len(third) == 0:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "comeback_quality": pl.Float64})
    # Shot attempts when trailing by 1-2
    shots_when_trailing = (
        third.filter(pl.col("event_type").is_in(["SHOT", "GOAL", "MISS", "BLOCK", "SHOT_ATTEMPT"]))
             .with_columns(
                 (pl.col("home_team_score") - pl.col("away_team_score")).alias("_diff")
             )
             .filter(pl.col("_diff").abs().is_between(1, 2))
    )
    by_team_game = (
        shots_when_trailing.group_by(["event_owner_team_id", "game_id"])
                           .agg(pl.len().alias("comeback_shots"))
                           .rename({"event_owner_team_id": "team_id"})
    )
    # Normalize: 12+ shots saturates positive
    return by_team_game.with_columns([
        pl.col("team_id").cast(pl.Utf8).alias("team"),
        pl.lit("").alias("game_date"),
        ((pl.col("comeback_shots").cast(pl.Float64).clip(0.0, 12.0) / 12.0))
            .alias("comeback_quality"),
    ]).select(["team", "game_id", "game_date", "comeback_quality"])


# ───────────────────────────────────────────────────────────────────────────
# 17.21 — Goalie Confidence (GSAx trend per starter)
# ───────────────────────────────────────────────────────────────────────────

def goalie_confidence(goalie_stats_df: pl.DataFrame) -> pl.DataFrame:
    if len(goalie_stats_df) == 0 or "gsax" not in goalie_stats_df.columns:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "goalie_confidence": pl.Float64})
    # Aggregate per (team, season) — proxy for the team's #1 goalie's GSAx
    by_team = (
        goalie_stats_df.filter(pl.col("situation") == "all")
                       if "situation" in goalie_stats_df.columns
                       else goalie_stats_df
    )
    if "team" not in by_team.columns:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "goalie_confidence": pl.Float64})
    grouped = (
        by_team.group_by("team")
               .agg(pl.col("gsax").sum().alias("_gsax_season"))
    )
    # Normalize: ±20 GSAx over a season saturates
    return grouped.with_columns([
        (pl.col("_gsax_season").cast(pl.Float64) / 20.0).clip(-1.0, 1.0)
            .alias("goalie_confidence"),
        pl.lit(0).cast(pl.Int64).alias("game_id"),
        pl.lit("").alias("game_date"),
    ]).select(["team", "game_id", "game_date", "goalie_confidence"])


# ───────────────────────────────────────────────────────────────────────────
# 17.22 — Team Injury Context (Roster Disruption Index)
# ───────────────────────────────────────────────────────────────────────────

def team_injury_context(roster_disruption_df: pl.DataFrame) -> pl.DataFrame:
    if len(roster_disruption_df) == 0:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "team_injury_context": pl.Float64})
    cols = roster_disruption_df.columns
    rdi_col = next((c for c in ("rdi", "roster_disruption_index", "disruption_score") if c in cols), None)
    team_col = next((c for c in ("team", "team_abbrev") if c in cols), None)
    if rdi_col is None or team_col is None:
        return _empty({"team": pl.Utf8, "game_id": pl.Int64,
                       "game_date": pl.Utf8, "team_injury_context": pl.Float64})
    return roster_disruption_df.select([
        pl.col(team_col).alias("team"),
        pl.col("game_id") if "game_id" in cols else pl.lit(0).cast(pl.Int64).alias("game_id"),
        pl.col("game_date") if "game_date" in cols else pl.lit("").alias("game_date"),
        # Higher RDI → more disruption → more confidence drag (negative)
        (-(pl.col(rdi_col).cast(pl.Float64)).clip(0.0, 1.0))
            .alias("team_injury_context"),
    ])
