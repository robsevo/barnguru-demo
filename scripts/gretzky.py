#!/usr/bin/env python3
"""GRTZKY — Central command dispatcher.

One place to run every script in the project.  Adding a new script takes
one line in the COMMANDS list below.

Usage::

    uv run python scripts/gretzky.py               # list all commands
    uv run python scripts/gretzky.py <command>     # run a command
    uv run python scripts/gretzky.py <command> -- <args>   # pass args through
    uv run python scripts/gretzky.py phase1        # run every phase1 command
    uv run python scripts/gretzky.py phase2        # run every phase2 command
    uv run python scripts/gretzky.py all           # run everything in order

Examples::

    uv run python scripts/gretzky.py sync
    uv run python scripts/gretzky.py train-xg -- --seasons 2025
    uv run python scripts/gretzky.py train-rapm -- --force
    uv run python scripts/gretzky.py all
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Make repo root importable regardless of cwd
_REPO = Path(__file__).parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------
# To add a new script: append one Cmd() here. That's all.
# Fields: name, phase, description, module (importable path), fn (function name)

@dataclass
class Cmd:
    name:        str
    phase:       str   # "phase1", "phase2", … used for group runs
    description: str
    module:      str   # e.g. "scripts.train_xg_model"
    fn:          str = "main"
    # Fixed args prepended to sys.argv before user-supplied passthrough.
    # Lets us register e.g. train-xg-playoffs as a preset of train-xg.
    args:        list[str] = field(default_factory=list)


COMMANDS: list[Cmd] = [
    # ── Phase 1 — Data Pipeline ───────────────────────────────────────────
    Cmd("sync",       "phase1", "Sync all live data feeds (injuries, transactions, EDGE, goalie stats)",
        "scripts.run_phase1_sync"),
    Cmd("sync-rosters", "phase1",
        "Sync current NHL rosters for all 32 teams → DataStore.roster()",
        "scripts.sync_rosters"),
    Cmd("ingest",     "phase1", "Ingest NHL historical PBP + shift data (required for train-rapm)",
        "scripts.run_historical_ingest"),
    Cmd("extract-iptv", "phase1",
        "Extract NHL channels from paid IPTV M3U playlists → iptv_paid_channels.json",
        "scripts.extract_iptv"),
    Cmd("check-iptv", "phase1",
        "Test each saved IPTV channel URL and write iptv_status.json with working/broken status",
        "scripts.check_iptv"),

    # ── Phase 2 — Player Rating Models ───────────────────────────────────
    Cmd("train-xg",   "phase2", "Train xG Finishing model (MoneyPuck shots)",
        "scripts.train_xg_model"),
    Cmd("train-xg-playoffs", "phase2",
        "Train xG Finishing — playoffs variant (writes xg_finishing_{year}_playoffs.parquet)",
        "scripts.train_xg_model", args=["--season-type", "playoffs"]),
    Cmd("train-rapm",    "phase2", "Train RAPM model with daisy-chain priors",
        "scripts.train_rapm_model"),
    Cmd("train-rapm-playoffs", "phase2",
        "Train RAPM — playoffs variant (writes rapm_{year}_playoffs.parquet)",
        "scripts.train_rapm_model", args=["--season-type", "playoffs"]),
    Cmd("train-matchup", "phase2", "Train Matchup Efficiency Matrix — QoT, QoC, pair xGF% (Feature 2.3)",
        "scripts.train_matchup_model"),
    Cmd("train-chemistry", "phase2", "Train Line Chemistry Model — linemate pair xGF% (Feature 2.4)",
        "scripts.train_chemistry_model"),
    Cmd("train-goalie",   "phase2", "Train Goalie Model + Goalie Fatigue sub-model (Features 2.5 + 2.6)",
        "scripts.train_goalie_model"),
    Cmd("train-special-teams", "phase2", "Compute per-player PP/PK special teams ratings (Feature 2.7)",
        "scripts.train_special_teams"),
    Cmd("train-special-teams-playoffs", "phase2",
        "Special Teams — playoffs variant",
        "scripts.train_special_teams", args=["--season-type", "playoffs"]),
    Cmd("train-bayesian", "phase2", "Train Bayesian Player Rating system (Features 2.9 + 2.10)",
        "scripts.train_bayesian_ratings"),
    Cmd("cluster-archetypes", "phase2", "Fit player archetype clusters K-means (Feature 2.11)",
        "scripts.train_archetype_model"),
    Cmd("inseason-bayes",  "phase2", "Run in-season Bayesian blend pipeline (Feature 2.12)",
        "scripts.run_inseason_bayesian"),
    Cmd("compute-ewma",   "phase2", "Compute EWMA form metrics per player (Feature 2.13)",
        "scripts.compute_ewma_form"),
    Cmd("compute-ewma-playoffs", "phase2",
        "Compute EWMA form — playoffs variant",
        "scripts.compute_ewma_form", args=["--season-type", "playoffs"]),
    Cmd("detect-regime",  "phase2", "Detect regime changes in player performance (Feature 2.14)",
        "scripts.run_regime_detector"),
    Cmd("normalize-sos",  "phase2", "Compute Strength-of-Schedule weights per player-game (Feature 2.15)",
        "scripts.normalize_sos"),
    Cmd("snapshot",       "phase2", "Save full GRTZKY system state snapshot (Feature 2.16)",
        "scripts.snapshot_gretzky"),
    Cmd("retrain",        "phase2", "Weekly warm-start incremental xG/Goalie retraining (Feature 2.17)",
        "scripts.retrain_incremental"),
    Cmd("puck-battles",   "phase2", "Compute puck battle / scrum proxy ratings (Feature 2.18)",
        "scripts.compute_puck_battles"),
    Cmd("roster-disruption", "phase2", "Compute Roster Disruption Index per team (Feature 2.19)",
        "scripts.compute_roster_disruption"),
    Cmd("skating-baseline",  "phase2", "Compute Player Skating Baseline profiles from EDGE (Feature 2.20)",
        "scripts.compute_skating_baseline"),
    Cmd("compute-heatmap",   "phase2", "Compute Player Positional Heat Map from shot data (Feature 2.21)",
        "scripts.compute_positional_heatmap"),
    Cmd("train-behavior",    "phase2", "Train Per-Player Behavioral Neural Network (Feature 2.22)",
        "scripts.train_behavior_net"),
    Cmd("rate-def",      "phase2", "Compute Composite Defensive Rating (CDR)",
        "scripts.compute_defensive_rating"),
    Cmd("rate-def-playoffs", "phase2", "CDR — playoffs variant",
        "scripts.compute_defensive_rating", args=["--season-type", "playoffs"]),
    Cmd("playoff-delta", "phase2", "Compute Playoff Performance Delta with Bayesian shrinkage (Feature 2.23)",
        "scripts.compute_playoff_delta"),
    Cmd("train-nhle",    "phase2", "Train NHLe/Prospect Projection Model — league translations (Feature 2.24)",
        "scripts.train_nhle_model"),
    Cmd("compute-war",   "phase2", "Compute WAR + Contract Efficiency ratings (Feature 2.25)",
        "scripts.compute_war"),
    Cmd("compute-war-playoffs", "phase2",
        "WAR — playoffs variant (reads playoff RAPM/xG, writes war_{year}_playoffs.parquet)",
        "scripts.compute_war", args=["--season-type", "playoffs"]),
    Cmd("former-team-boost", "phase2",
        "Compute Former Team Motivational Boost per player (Feature 2.27)",
        "scripts.compute_former_team_boost"),
    Cmd("hot-hand", "phase2",
        "Compute 5-game burst hot-hand signal per player (Feature 2.28)",
        "scripts.compute_hot_hand"),
    Cmd("hot-hand-playoffs", "phase2",
        "Compute hot-hand signal — playoffs variant",
        "scripts.compute_hot_hand", args=["--season-type", "playoffs"]),
    Cmd("clutch-index", "phase2",
        "Compute Clutch Index (Win Probability Added) per player (Feature 2.29)",
        "scripts.compute_clutch_index"),
    Cmd("clutch-index-playoffs", "phase2",
        "Clutch Index — playoffs variant",
        "scripts.compute_clutch_index", args=["--season-type", "playoffs"]),

    # ── Phase 3 — Fatigue Engine ─────────────────────────────────────────
    Cmd("schedule-density", "phase3",
        "Compute schedule density per team-game — B2B / 3-in-4 / rest days (Feature 3.1)",
        "scripts.compute_schedule_density"),
    Cmd("road-trips", "phase3",
        "Annotate team-games with road-trip id + game-within-trip (Feature 3.2)",
        "scripts.compute_road_trips"),
    Cmd("travel-distance", "phase3",
        "Compute travel mileage and rolling 7-day miles per team-game (Feature 3.3)",
        "scripts.compute_travel_distance"),
    Cmd("tz-crossing", "phase3",
        "Compute time-zone crossings + 48h rolling load per team-game (Feature 3.4)",
        "scripts.compute_time_zone_crossing"),
    Cmd("circadian", "phase3",
        "Compute body-clock misalignment + prime-window deviation per team-game (Feature 3.5)",
        "scripts.compute_circadian_alignment"),
    Cmd("altitude", "phase3",
        "Compute altitude aerobic-penalty per visiting team-game (Feature 3.6)",
        "scripts.compute_altitude_adjustment"),
    Cmd("toi-load", "phase3",
        "Compute rolling 5-game TOI load + spike detector per player (Feature 3.7)",
        "scripts.compute_toi_load"),
    Cmd("st-load", "phase3",
        "Compute rolling 5-game PP/PK accumulation + intensity score per player (Feature 3.8)",
        "scripts.compute_special_teams_load"),
    Cmd("contact-load", "phase3",
        "Compute rolling 5-game hits-taken + blocked-shots load per player (Feature 3.9)",
        "scripts.compute_physical_contact_load"),
    Cmd("ot-fatigue", "phase3",
        "Compute rolling 7-day OT fatigue + equivalent-TOI penalty per player (Feature 3.10)",
        "scripts.compute_overtime_fatigue"),
    Cmd("fight-fatigue", "phase3",
        "Compute rolling 14-day fighting-major adrenal-toll score per player (Feature 3.11)",
        "scripts.compute_fight_fatigue"),
    Cmd("age-recovery", "phase3",
        "Compute per-player exponential age-recovery coefficient (Feature 3.12)",
        "scripts.compute_age_recovery"),
    Cmd("injury-status", "phase3",
        "Compute per-player injury status + return-from-injury rust factor (Feature 3.13)",
        "scripts.compute_injury_status"),
    Cmd("concussion-history", "phase3",
        "Compute per-player prior-concussion fatigue sensitivity multiplier (Feature 3.14)",
        "scripts.compute_concussion_history"),
    Cmd("prior-playoff-load", "phase3",
        "Compute per-player prior-playoff-load season-start penalty (Feature 3.15)",
        "scripts.compute_prior_playoff_load"),
    Cmd("roster-depth-strain", "phase3",
        "Compute per-skater roster-depth strain from IR-minute redistribution (Feature 3.16)",
        "scripts.compute_roster_depth_strain"),
    Cmd("playoff-fatigue", "phase3",
        "Compute per-player-per-game current-playoff-run fatigue (Feature 3.23)",
        "scripts.compute_playoff_fatigue"),
    Cmd("goalie-fi", "phase3",
        "Compute daily per-goalie Fatigue Index snapshot (Feature 3.24)",
        "scripts.compute_goalie_fi"),
    Cmd("composite-fi", "phase3",
        "Compute composite Fatigue Index per (player, game) (Feature 3.17)",
        "scripts.compute_composite_fi"),
    Cmd("fi-multiplier", "phase3",
        "Compute FI → rating multiplier per (player, game) (Feature 3.18)",
        "scripts.compute_fi_multiplier"),
    Cmd("performance-anomaly", "phase3",
        "Compute per-(player, game) z-score + CUSUM performance anomalies (Feature 3.19)",
        "scripts.compute_performance_anomaly"),
    Cmd("trade-integration", "phase3",
        "Compute per-(player, game) trade-integration modifier (Feature 3.20)",
        "scripts.compute_trade_integration"),
    Cmd("fi-edge-degradation", "phase3",
        "Compute FI → EDGE metric degradation per player (Feature 3.21)",
        "scripts.compute_fi_edge_degradation"),
    Cmd("seasonal-performance", "phase3",
        "Compute seasonal motivation factor per (player, game) (Feature 3.22)",
        "scripts.compute_seasonal_performance"),
    Cmd("team-results", "phase3",
        "Derive per-team-game W/L/OTL results from PBP final scores (feeds Phase 17 team_streak)",
        "scripts.compute_team_results"),

    # ── Phase 17 — Confidence Layer ───────────────────────────────────────
    Cmd("composite-confidence", "phase17",
        "Compute Phase 17 composite Confidence Index + rating multiplier (Features 17.1–17.25)",
        "scripts.compute_composite_confidence"),

    # ── Phase 16 — CV Tracking Engine ────────────────────────────────────
    # Removed temporarily; will be reintroduced once stable. Static formation
    # rink in the meantime; nothing in Phases 1–2 depends on CV outputs.
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 55


def _phase_label(phase: str) -> str:
    labels = {
        "phase1":  "PHASE 1  — Data Pipeline",
        "phase2":  "PHASE 2  — Player Rating Models",
        "phase3":  "PHASE 3  — Fatigue Engine",
        "phase4":  "PHASE 4  — Coaching Tendency Models",
        "phase5":  "PHASE 5  — Rust Simulation Engine",
        "phase17": "PHASE 17 — Confidence Layer",
    }
    return labels.get(phase, phase.upper())


def _print_commands() -> None:
    print(f"\nGRTZKY — Command Dispatcher")
    print(_SEP)

    # Group by phase, preserving COMMANDS order
    seen: dict[str, list[Cmd]] = {}
    for cmd in COMMANDS:
        seen.setdefault(cmd.phase, []).append(cmd)

    for phase, cmds in seen.items():
        print(f"\n  {_phase_label(phase)}")
        for c in cmds:
            print(f"    {c.name:<18s}  {c.description}")

    # Group commands
    phases = sorted(seen.keys())
    print(f"\n  GROUPS")
    for p in phases:
        print(f"    {p:<18s}  Run all {p} commands in sequence")
    print(f"    {'all':<18s}  Run every command in sequence")

    print(f"\nUsage:")
    print(f"  uv run python scripts/gretzky.py <command> [-- <args>]")
    print(_SEP)
    print()


def _resolve(name: str) -> Callable | None:
    """Import module and return the callable, or None if not found."""
    for cmd in COMMANDS:
        if cmd.name == name:
            mod = importlib.import_module(cmd.module)
            return getattr(mod, cmd.fn)
    return None


def _run_cmd(cmd: Cmd, passthrough: list[str]) -> int:
    """Import and call cmd.fn, patching sys.argv with passthrough args."""
    print(f"\n{'─'*55}")
    print(f"  gretzky › {cmd.name}")
    print(f"{'─'*55}")

    mod = importlib.import_module(cmd.module)
    fn  = getattr(mod, cmd.fn)

    # Patch sys.argv so argparse inside the script sees the right args.
    # cmd.args (registry preset) come first, then user-supplied passthrough.
    old_argv = sys.argv[:]
    sys.argv = [cmd.module] + list(cmd.args) + passthrough
    try:
        fn()
        return 0
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    # No args → print command list
    if not args or args[0] in ("list", "--list", "-l"):
        _print_commands()
        return

    # Split on "--" to separate dispatcher args from passthrough args
    if "--" in args:
        sep_idx   = args.index("--")
        cmd_args  = args[:sep_idx]
        passthrough = args[sep_idx + 1:]
    else:
        cmd_args    = args
        passthrough = []

    verb = cmd_args[0]

    # ── Group: all ──────────────────────────────────────────────────────
    if verb == "all":
        failed = []
        for cmd in COMMANDS:
            rc = _run_cmd(cmd, [])
            if rc != 0:
                failed.append(cmd.name)
        if failed:
            print(f"\n[gretzky] Some commands failed: {', '.join(failed)}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[gretzky] All commands completed.")
        return

    # ── Group: phaseN ───────────────────────────────────────────────────
    if verb.startswith("phase"):
        phase_cmds = [c for c in COMMANDS if c.phase == verb]
        if not phase_cmds:
            print(f"[gretzky] No commands registered for '{verb}'.", file=sys.stderr)
            sys.exit(1)
        failed = []
        for cmd in phase_cmds:
            rc = _run_cmd(cmd, passthrough)
            if rc != 0:
                failed.append(cmd.name)
        if failed:
            print(f"\n[gretzky] Some {verb} commands failed: {', '.join(failed)}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[gretzky] {verb} complete.")
        return

    # ── Single command ───────────────────────────────────────────────────
    cmd_match = next((c for c in COMMANDS if c.name == verb), None)
    if cmd_match is None:
        print(f"[gretzky] Unknown command: '{verb}'", file=sys.stderr)
        print(f"  Run without arguments to see available commands.", file=sys.stderr)
        sys.exit(1)

    rc = _run_cmd(cmd_match, passthrough)
    sys.exit(rc)


if __name__ == "__main__":
    main()
