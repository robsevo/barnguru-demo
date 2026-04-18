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
from dataclasses import dataclass
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


COMMANDS: list[Cmd] = [
    # ── Phase 1 — Data Pipeline ───────────────────────────────────────────
    Cmd("sync",       "phase1", "Sync all live data feeds (injuries, transactions, EDGE, goalie stats)",
        "scripts.run_phase1_sync"),
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
    Cmd("train-rapm",    "phase2", "Train RAPM model with daisy-chain priors",
        "scripts.train_rapm_model"),
    Cmd("train-matchup", "phase2", "Train Matchup Efficiency Matrix — QoT, QoC, pair xGF% (Feature 2.3)",
        "scripts.train_matchup_model"),
    Cmd("train-chemistry", "phase2", "Train Line Chemistry Model — linemate pair xGF% (Feature 2.4)",
        "scripts.train_chemistry_model"),
    Cmd("train-goalie",   "phase2", "Train Goalie Model + Goalie Fatigue sub-model (Features 2.5 + 2.6)",
        "scripts.train_goalie_model"),
    Cmd("train-special-teams", "phase2", "Compute per-player PP/PK special teams ratings (Feature 2.7)",
        "scripts.train_special_teams"),
    Cmd("train-bayesian", "phase2", "Train Bayesian Player Rating system (Features 2.9 + 2.10)",
        "scripts.train_bayesian_ratings"),
    Cmd("cluster-archetypes", "phase2", "Fit player archetype clusters K-means (Feature 2.11)",
        "scripts.train_archetype_model"),
    Cmd("inseason-bayes",  "phase2", "Run in-season Bayesian blend pipeline (Feature 2.12)",
        "scripts.run_inseason_bayesian"),
    Cmd("compute-ewma",   "phase2", "Compute EWMA form metrics per player (Feature 2.13)",
        "scripts.compute_ewma_form"),
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
    Cmd("playoff-delta", "phase2", "Compute Playoff Performance Delta with Bayesian shrinkage (Feature 2.23)",
        "scripts.compute_playoff_delta"),
    Cmd("train-nhle",    "phase2", "Train NHLe/Prospect Projection Model — league translations (Feature 2.24)",
        "scripts.train_nhle_model"),
    Cmd("compute-war",   "phase2", "Compute WAR + Contract Efficiency ratings (Feature 2.25)",
        "scripts.compute_war"),
    Cmd("former-team-boost", "phase2",
        "Compute Former Team Motivational Boost per player (Feature 2.27)",
        "scripts.compute_former_team_boost"),
    Cmd("hot-hand", "phase2",
        "Compute 5-game burst hot-hand signal per player (Feature 2.28)",
        "scripts.compute_hot_hand"),
    Cmd("clutch-index", "phase2",
        "Compute Clutch Index (Win Probability Added) per player (Feature 2.29)",
        "scripts.compute_clutch_index"),

    # ── Phase 3+ — add here as features are built ─────────────────────────

    # ── Phase 16 — CV Tracking Engine ────────────────────────────────────
    Cmd("build-cv-dataset",  "phase16",
        "Build CV training data pipeline: download → extract → MOT→YOLO → augment (Feature 16.1)",
        "scripts.build_cv_dataset"),
    Cmd("train-cv-detector", "phase16",
        "Fine-tune YOLO player+puck detector on 16.1 data; gate mAP@0.5 ≥ 0.82 (Feature 16.2)",
        "scripts.train_cv_detector"),
    Cmd("retrain-cv", "phase16",
        "Weekly warm-start YOLO retrain using cached puck pseudo-labels; rejects regressions",
        "scripts.retrain_cv_detector"),
    Cmd("gen-jersey-data",   "phase16",
        "Generate synthetic jersey-number training crops (~50K samples) (Feature 16.3)",
        "scripts.generate_jersey_data"),
    Cmd("train-jersey-ocr",  "phase16",
        "Train jersey OCR CNN (numbers 1–99); gate top-1 ≥ 85% (Feature 16.3)",
        "scripts.train_jersey_ocr"),
    Cmd("build-color-lut",   "phase16",
        "Build HSV team-colour LUT for all 32 NHL teams (Feature 16.4)",
        "scripts.build_team_color_lut"),
    Cmd("train-rink-kp",     "phase16",
        "Train rink keypoint detector (heatmap regression, 7 landmarks) (Feature 16.5)",
        "scripts.train_rink_keypoints"),
    Cmd("retrain-xg-tracking", "phase16",
        "Retrain xG model with CV tracking features (screen_count, defender_dist, goalie_displacement); gate AUC (Feature 16.10)",
        "scripts.retrain_xg_tracking"),
    Cmd("aggregate-cv-obs",  "phase16",
        "Reduce in-browser CV observation NDJSON into per-track summaries (Feature 16 bridge)",
        "scripts.aggregate_cv_observations"),
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
        "phase16": "PHASE 16 — CV Tracking Engine (parallel track)",
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

    # Patch sys.argv so argparse inside the script sees the right args
    old_argv = sys.argv[:]
    sys.argv = [cmd.module] + passthrough
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
