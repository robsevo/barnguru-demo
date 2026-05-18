"""compute_composite_confidence — Phase 17 driver script.

Loads every available Phase 17 signal input from disk (gracefully missing →
empty signal), calls each signal function in ``models.confidence_signals``,
joins them onto the (player_id, game_id) spine, and runs the composite
weighted-sum model.

Also computes the confidence rating multiplier (17.25) and writes both
parquets in one pass — same nightly call surfaces both to the backend.

Usage::

    uv run python scripts/gretzky.py composite-confidence
    uv run python scripts/gretzky.py composite-confidence -- --date 2026-05-17 --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import polars as pl

from data.data_store import DataStore
from data.schedule_sync import latest_schedule_parquet
from models.composite_confidence import (
    CompositeConfidenceIndex,
    write_composite_confidence,
)
from models.confidence_rating_multiplier import (
    ConfidenceRatingMultiplier,
    write_confidence_multiplier,
)
from models import confidence_signals as sigs


DEFAULT_DATA_DIR = Path(
    os.environ.get("GRETZKY_DATA_DIR", str(Path.home() / ".gretzky" / "data"))
)
COMPOSITE_CONFIDENCE_SUBDIR = "composite_confidence"
CONFIDENCE_MULT_SUBDIR      = "confidence_multiplier"


def _latest(dirpath: Path, prefix: str) -> Path | None:
    if not dirpath.exists():
        return None
    files = sorted(dirpath.glob(f"{prefix}_*.parquet"))
    return files[-1] if files else None


def _read_or_empty(dirpath: Path, prefix: str) -> pl.DataFrame:
    p = _latest(dirpath, prefix)
    return pl.read_parquet(p) if p else pl.DataFrame()


def _read_whiz_observations(data_dir: Path) -> list[dict]:
    path = data_dir.parents[0] / "whiz_brain" / "observations.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _build_player_signals(args, as_of_iso: str) -> pl.DataFrame:
    """Compute every player-side signal and stitch them onto a (pid, gid) spine."""
    d = args.data_dir

    # ── Load inputs (every read is gracefully empty if missing) ──
    # Hot hand (2.28) — prefer per-game parquet; fall back to summary.
    hot_hand_dir = d / "hot_hand"
    hh_per_game = sorted(hot_hand_dir.glob("hot_hand_2*.parquet")) if hot_hand_dir.exists() else []
    hh_summary  = sorted(hot_hand_dir.glob("hot_hand_summary_*.parquet")) if hot_hand_dir.exists() else []
    # Filter out summary files from per-game glob (glob matches both)
    hh_per_game = [p for p in hh_per_game if "summary" not in p.name]
    if hh_per_game:
        hot_hand_df = pl.read_parquet(hh_per_game[-1])
    elif hh_summary:
        hot_hand_df = pl.read_parquet(hh_summary[-1])
    else:
        hot_hand_df = pl.DataFrame()

    # EWMA (2.13) — under ewma/
    ewma_files = sorted((d / "ewma").glob("*.parquet")) if (d / "ewma").exists() else []
    ewma_df = pl.read_parquet(ewma_files[-1]) if ewma_files else pl.DataFrame()

    # TOI load (3.7)
    toi_df = _read_or_empty(d / "toi_load", "toi_load")

    # Matchup (2.3) — QoT
    matchup_files = sorted((d / "matchup").glob("*.parquet")) if (d / "matchup").exists() else []
    matchup_df = pl.read_parquet(matchup_files[-1]) if matchup_files else pl.DataFrame()

    # Injury status
    injury_df = _read_or_empty(d / "injuries", "injuries")

    # Rosters
    rosters_path = d / "rosters_latest.parquet"
    rosters_df = pl.read_parquet(rosters_path) if rosters_path.exists() else pl.DataFrame()

    # Concussion history (3.14)
    conc_df = _read_or_empty(d / "concussion_history", "concussion_history") \
        if (d / "concussion_history").exists() else pl.DataFrame()

    # Physical contact load (3.9)
    contact_df = _read_or_empty(d / "physical_contact_load", "physical_contact_load")

    # Chemistry (2.4) — linemate pair table
    chem_files = sorted((d / "chemistry").glob("*.parquet")) if (d / "chemistry").exists() else []
    chemistry_df = pl.read_parquet(chem_files[-1]) if chem_files else pl.DataFrame()

    # Whiz observations (manual override channel)
    whiz_obs = _read_whiz_observations(d)

    # Contract status (17.14) — empty until data/contract_sync.py lands
    contracts_df = _read_or_empty(d / "contracts", "contracts") \
        if (d / "contracts").exists() else pl.DataFrame()

    # Officials (17.15) — empty until data/officials_sync.py lands
    officials_df = _read_or_empty(d / "officials", "officials") \
        if (d / "officials").exists() else pl.DataFrame()
    pim_history_df = _read_or_empty(d / "officials", "pim_history") \
        if (d / "officials").exists() else pl.DataFrame()

    # Schedule
    sched_dir  = d / "schedule"
    sched_path = latest_schedule_parquet(sched_dir)
    schedule_df = pl.read_parquet(sched_path) if sched_path else pl.DataFrame()

    # PBP appearances (for healthy-scratch detection)
    with DataStore(d) as store:
        pbp = store.pbp()
    if len(pbp) > 0 and "shooter_id" in pbp.columns:
        appearances = pbp.select([
            pl.col("game_id"),
            pl.col("shooter_id").alias("player_id"),
        ]).filter(pl.col("player_id").is_not_null()).unique()
        if "game_date" in pbp.columns:
            dates = pbp.select(["game_id", "game_date"]).unique()
            appearances = appearances.join(dates, on="game_id", how="left")
        else:
            appearances = appearances.with_columns(pl.lit("").alias("game_date"))
    else:
        appearances = pl.DataFrame(schema={
            "player_id": pl.Int64, "game_id": pl.Int64, "game_date": pl.Utf8,
        })

    # ── Compute each player signal ──
    print("[confidence] Computing player signals…")
    s_hot      = sigs.hot_hand_signal(hot_hand_df)
    s_ewma     = sigs.ewma_form_signal(ewma_df)
    s_toi      = sigs.toi_trust_trend(toi_df)
    s_role     = sigs.role_usage_delta(matchup_df)
    s_scratch  = sigs.healthy_scratch_flag(appearances, injury_df, rosters_df, as_of_iso)
    s_drought  = sigs.point_drought_z(pl.DataFrame())   # 17.6 needs xG-conditioned drought parquet (v2)
    s_linemate = sigs.linemate_form_drag(chemistry_df, ewma_df)
    s_bounce   = sigs.bounceback_index()
    s_injury   = sigs.active_injury_drag(injury_df, conc_df)
    s_target   = sigs.targeting_pressure(contact_df)
    s_media    = sigs.media_sentiment(whiz_obs)
    s_homeaway = sigs.home_away_split(ewma_df, schedule_df, rosters_df, as_of_iso)
    s_rumor    = sigs.trade_rumor_pressure(whiz_obs)
    s_contract = sigs.contract_pressure(contracts_df, as_of_iso)
    s_ref      = sigs.referee_bias(officials_df, pim_history_df)

    # ── Build spine: every active roster player gets a row ──
    if len(rosters_df) > 0 and "player_id" in rosters_df.columns:
        spine = rosters_df.select([
            pl.col("player_id").cast(pl.Int64),
        ]).unique().with_columns([
            pl.lit(0).cast(pl.Int64).alias("game_id"),
            pl.lit(as_of_iso).alias("game_date"),
        ])
    else:
        # Fall back to union of every signal's player_ids
        spine = pl.concat([
            df.select(["player_id"]) for df in [s_hot, s_ewma, s_toi, s_role, s_scratch,
                                                s_injury, s_target, s_media, s_homeaway,
                                                s_rumor, s_contract]
            if "player_id" in df.columns and len(df) > 0
        ], how="vertical").unique().with_columns([
            pl.lit(0).cast(pl.Int64).alias("game_id"),
            pl.lit(as_of_iso).alias("game_date"),
        ]) if any(
            "player_id" in df.columns and len(df) > 0
            for df in [s_hot, s_ewma, s_toi, s_role, s_scratch, s_injury,
                       s_target, s_media, s_homeaway, s_rumor, s_contract]
        ) else pl.DataFrame()

    if len(spine) == 0:
        print("[confidence] No active players to score — empty spine.")
        return pl.DataFrame()

    # Left-join every signal (taking the most recent value per player).
    def _join_signal(base: pl.DataFrame, sig_df: pl.DataFrame, col: str) -> pl.DataFrame:
        if len(sig_df) == 0 or col not in sig_df.columns or "player_id" not in sig_df.columns:
            return base.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        # Reduce sig_df to one row per player_id (most recent game_date wins).
        if "game_date" in sig_df.columns:
            sig_df = sig_df.sort("game_date", descending=True).group_by("player_id").head(1)
        else:
            sig_df = sig_df.unique(subset=["player_id"])
        return base.join(
            sig_df.select(["player_id", col]),
            on="player_id", how="left",
        ).with_columns(pl.col(col).fill_null(0.0))

    spine = _join_signal(spine, s_hot,      "hot_hand_signal")
    spine = _join_signal(spine, s_ewma,     "ewma_form_signal")
    spine = _join_signal(spine, s_toi,      "toi_trust_trend")
    spine = _join_signal(spine, s_role,     "role_usage_delta")
    spine = _join_signal(spine, s_scratch,  "healthy_scratch_flag")
    spine = _join_signal(spine, s_drought,  "point_drought_z")
    spine = _join_signal(spine, s_linemate, "linemate_form_drag")
    spine = _join_signal(spine, s_bounce,   "bounceback_index")
    spine = _join_signal(spine, s_injury,   "active_injury_drag")
    spine = _join_signal(spine, s_target,   "targeting_pressure")
    spine = _join_signal(spine, s_media,    "media_sentiment")
    spine = _join_signal(spine, s_homeaway, "home_away_split")
    spine = _join_signal(spine, s_rumor,    "trade_rumor_pressure")
    spine = _join_signal(spine, s_contract, "contract_pressure")
    spine = _join_signal(spine, s_ref,      "referee_bias")

    return spine


def _broadcast_team_signals(
    player_spine: pl.DataFrame,
    args,
    as_of_iso: str,
) -> pl.DataFrame:
    """Compute team-side signals, then broadcast onto the player spine via roster."""
    d = args.data_dir

    # Inputs
    results_df = _read_or_empty(d / "results", "results") \
        if (d / "results").exists() else pl.DataFrame()
    schedule_path = latest_schedule_parquet(d / "schedule")
    schedule_df = pl.read_parquet(schedule_path) if schedule_path else pl.DataFrame()
    team_df = pl.DataFrame()    # team CSVs — empty in current setup; signal returns empty
    goalie_stats_files = sorted((d / "goalie_stats").glob("goalie_stats_*.parquet")) \
        if (d / "goalie_stats").exists() else []
    goalie_stats_df = pl.read_parquet(goalie_stats_files[-1]) if goalie_stats_files else pl.DataFrame()
    roster_disruption = _read_or_empty(d / "roster_disruption", "roster_disruption")

    with DataStore(d) as store:
        pbp = store.pbp()

    print("[confidence] Computing team signals…")
    t_streak    = sigs.team_streak(schedule_df, results_df)
    t_corsi     = sigs.score_adj_corsi_trend(team_df)
    t_st        = sigs.special_teams_trend(team_df)
    t_challenge = sigs.coach_challenge_rate(pbp)
    t_comeback  = sigs.comeback_quality(pbp)
    t_goalie    = sigs.goalie_confidence(goalie_stats_df)
    t_injury    = sigs.team_injury_context(roster_disruption)

    # Reduce each team signal to one row per team (most recent value)
    def _team_reduce(df: pl.DataFrame, col: str) -> pl.DataFrame:
        if len(df) == 0 or col not in df.columns or "team" not in df.columns:
            return pl.DataFrame({"team": [], col: []},
                                schema={"team": pl.Utf8, col: pl.Float64})
        if "game_date" in df.columns:
            df = df.sort("game_date", descending=True).group_by("team").head(1)
        else:
            df = df.unique(subset=["team"])
        return df.select(["team", col])

    # Build the team universe first (union of every signal's team list),
    # then left-join each signal so we never collide on `team_right`.
    team_frames = [
        _team_reduce(t_streak,    "team_streak"),
        _team_reduce(t_corsi,     "score_adj_corsi_trend"),
        _team_reduce(t_st,        "special_teams_trend"),
        _team_reduce(t_challenge, "coach_challenge_rate"),
        _team_reduce(t_comeback,  "comeback_quality"),
        _team_reduce(t_goalie,    "goalie_confidence"),
        _team_reduce(t_injury,    "team_injury_context"),
    ]
    team_universe = pl.concat(
        [f.select(["team"]) for f in team_frames if len(f) > 0 and "team" in f.columns],
        how="vertical",
    ).unique() if any(len(f) > 0 for f in team_frames) else pl.DataFrame(
        {"team": []}, schema={"team": pl.Utf8}
    )
    team_summary = team_universe
    for f in team_frames:
        if len(f) > 0 and "team" in f.columns:
            team_summary = team_summary.join(f, on="team", how="left")
    for col in ["team_streak", "score_adj_corsi_trend", "special_teams_trend",
                "coach_challenge_rate", "comeback_quality", "goalie_confidence",
                "team_injury_context"]:
        if col not in team_summary.columns:
            team_summary = team_summary.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        team_summary = team_summary.with_columns(pl.col(col).fill_null(0.0))

    # Broadcast: player_id → team_abbrev. Multiple sources, picked in this order:
    #   1. rosters_latest.parquet (if present)
    #   2. data/raw/roster_*.json (the canonical source the dashboard already uses
    #      in `_build_player_meta`)
    #   3. xg_finishing parquet (shooter_id + team) as a final skater fallback
    #   4. goalie_stats parquet (player_id + team) for goalies
    # On Bob's prod machine (1) doesn't exist; (2) does. Without this fallback the
    # team-broadcast silently no-ops and every team_score is 0.0.
    pid_to_team: dict[int, str] = {}

    rosters_path = d / "rosters_latest.parquet"
    if rosters_path.exists():
        try:
            rdf = pl.read_parquet(rosters_path)
            if {"player_id", "team_abbrev"}.issubset(rdf.columns):
                for r in rdf.select(["player_id", "team_abbrev"]).unique().to_dicts():
                    pid_to_team[int(r["player_id"])] = str(r["team_abbrev"])
        except Exception:
            pass

    # raw/roster_*.json — dashboard's canonical source
    raw_dir = d / "raw"
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("roster_*.json")):
            try:
                data = json.loads(f.read_text())
                team_code = data.get("team_code", "")
                for p in data.get("profiles", []):
                    pid_raw = p.get("player_id")
                    if not pid_raw or not team_code:
                        continue
                    pid_i = int(pid_raw)
                    pid_to_team.setdefault(pid_i, team_code)
            except Exception:
                continue

    # xg_finishing fallback (catches recently-traded / off-roster shooters)
    xg_dir = d / "xg_finishing"
    if xg_dir.exists():
        xg_files = sorted(xg_dir.glob("xg_finishing_*.parquet"))
        if xg_files:
            try:
                xg = pl.read_parquet(xg_files[-1])
                if {"shooter_id", "team"}.issubset(xg.columns):
                    for r in (
                        xg.select(["shooter_id", "team"])
                          .drop_nulls(subset=["shooter_id"])
                          .unique(subset=["shooter_id"])
                          .to_dicts()
                    ):
                        pid_to_team.setdefault(int(r["shooter_id"]), str(r.get("team") or ""))
            except Exception:
                pass

    # goalie_stats fallback
    goalie_dir = d / "goalie_stats"
    if goalie_dir.exists():
        g_files = sorted(goalie_dir.glob("goalie_stats_*.parquet"))
        if g_files:
            try:
                gs = pl.read_parquet(g_files[-1])
                if {"player_id", "team"}.issubset(gs.columns):
                    for r in (
                        gs.select(["player_id", "team"])
                          .unique(subset=["player_id"])
                          .to_dicts()
                    ):
                        pid_to_team.setdefault(int(r["player_id"]), str(r.get("team") or ""))
            except Exception:
                pass

    if pid_to_team:
        roster_df = pl.DataFrame(
            [{"player_id": k, "team_abbrev": v} for k, v in pid_to_team.items() if v],
            schema={"player_id": pl.Int64, "team_abbrev": pl.Utf8},
        )
        spine_with_team = player_spine.join(roster_df, on="player_id", how="left")
        spine_with_team = spine_with_team.join(
            team_summary.rename({"team": "team_abbrev"}),
            on="team_abbrev", how="left",
        )
        for col in ["team_streak", "score_adj_corsi_trend", "special_teams_trend",
                    "coach_challenge_rate", "comeback_quality", "goalie_confidence",
                    "team_injury_context"]:
            if col not in spine_with_team.columns:
                spine_with_team = spine_with_team.with_columns(
                    pl.lit(0.0).cast(pl.Float64).alias(col)
                )
            spine_with_team = spine_with_team.with_columns(pl.col(col).fill_null(0.0))
        if "team_abbrev" in spine_with_team.columns:
            spine_with_team = spine_with_team.drop("team_abbrev")
        n_mapped = (
            spine_with_team.filter(
                (pl.col("team_streak").abs()
                 + pl.col("goalie_confidence").abs()
                 + pl.col("team_injury_context").abs()) > 1e-9
            ).height
        )
        print(f"  Player→team mapping: {len(pid_to_team):,} ids; "
              f"{n_mapped:,} players received non-zero team signal")
        return spine_with_team

    # No roster source at all — team signals contribute zero (true no-op)
    print("  WARN: no roster source found; team_score will be 0.0 for every player")
    for col in ["team_streak", "score_adj_corsi_trend", "special_teams_trend",
                "coach_challenge_rate", "comeback_quality", "goalie_confidence",
                "team_injury_context"]:
        player_spine = player_spine.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
    return player_spine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Phase 17 composite Confidence Index + rating multiplier."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--date", type=str, default=None,
                        help="as-of date. Default: today (UTC).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    as_of = args.date or datetime.now(timezone.utc).date().isoformat()

    out_conf_dir  = args.data_dir / COMPOSITE_CONFIDENCE_SUBDIR
    out_conf_path = out_conf_dir / f"composite_confidence_{as_of}.parquet"
    out_mult_dir  = args.data_dir / CONFIDENCE_MULT_SUBDIR
    out_mult_path = out_mult_dir / f"confidence_multiplier_{as_of}.parquet"

    if out_conf_path.exists() and out_mult_path.exists() and not args.force:
        print(f"[composite-confidence] Outputs already exist:")
        print(f"  {out_conf_path}")
        print(f"  {out_mult_path}")
        print("  Use --force to overwrite.")
        sys.exit(0)

    print(f"[composite-confidence] Building inputs as of {as_of}…")
    player_signals = _build_player_signals(args, as_of)
    if len(player_signals) == 0:
        print("[composite-confidence] No signals available; emitting empty parquets.")
        empty_conf = pl.DataFrame(schema={
            "player_id": pl.Int64, "game_id": pl.Int64,
            "game_date": pl.Utf8,
        })
        write_composite_confidence(empty_conf, out_conf_dir, as_of)
        write_confidence_multiplier(empty_conf, out_mult_dir, as_of)
        return

    signals = _broadcast_team_signals(player_signals, args, as_of)
    print(f"  {len(signals):,} player rows with signals stitched.")

    model = CompositeConfidenceIndex()
    print(f"  Player weights (sum={sum(model.player_weights.values()):.3f})")
    print(f"  Team weights   (sum={sum(model.team_weights.values()):.3f})")
    print(f"  Team blend:    {model.team_blend:.2f}")

    result = model.compute(signals, as_of_date=as_of)
    n = len(result)
    print(f"  {n:,} composite-confidence rows produced.")

    if n > 0:
        top = result.sort("confidence_index", descending=True).head(8)
        bot = result.sort("confidence_index", descending=False).head(8)
        print("\n  Top 8 most-confident players:")
        for r in top.to_dicts():
            print(f"    {r['player_id']:>10}  c={r['confidence_index']:>+.3f}  "
                  f"p={r['player_score']:>+.3f}  t={r['team_score']:>+.3f}")
        print("\n  Bottom 8 least-confident players:")
        for r in bot.to_dicts():
            print(f"    {r['player_id']:>10}  c={r['confidence_index']:>+.3f}  "
                  f"p={r['player_score']:>+.3f}  t={r['team_score']:>+.3f}")

    write_composite_confidence(result, out_conf_dir, as_of)
    print(f"\n[composite-confidence] Written: {out_conf_path}")

    mult = ConfidenceRatingMultiplier().compute(result, as_of_date=as_of)
    write_confidence_multiplier(mult, out_mult_dir, as_of)
    print(f"[confidence-multiplier] Written: {out_mult_path}")


if __name__ == "__main__":
    main()
