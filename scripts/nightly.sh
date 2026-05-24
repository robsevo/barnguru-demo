#!/usr/bin/env bash
# nightly.sh — incremental data + model refresh
# Runs after the last game of the day finishes (~11:30 PM ET = 04:30 UTC)
#
# What this does:
#   1. Pull latest code
#   2. Sync live feeds (injuries, EDGE, transactions, goalie stats)
#   3. Ingest any new game PBP + shifts (manifest skips already-done games)
#   4. Rebuild RAPM with fresh game data
#   5. Recompute WAR + EWMA from updated RAPM
#
# Logs → ~/.gretzky/logs/nightly_YYYY-MM-DD.log

set -euo pipefail

REPO="/home/user/programs/grtzky-main"
LOG_DIR="$HOME/.gretzky/logs"
DATE=$(date -u +%Y-%m-%d)
LOG="$LOG_DIR/nightly_${DATE}.log"

mkdir -p "$LOG_DIR"

echo "========================================" | tee -a "$LOG"
echo "  GRTZKY nightly — $(date -u '+%Y-%m-%d %H:%M UTC')" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

cd "$REPO"

# 1. Pull latest code
echo "" | tee -a "$LOG"
echo "[1/5] git pull" | tee -a "$LOG"
git pull --ff-only origin main 2>&1 | tee -a "$LOG"

# 2. Sync live feeds
echo "" | tee -a "$LOG"
echo "[2/5] Phase 1 sync (injuries, EDGE, transactions, goalie stats)" | tee -a "$LOG"
uv run python scripts/run_phase1_sync.py 2>&1 | tee -a "$LOG"

# 3. Ingest new game PBP + shifts (manifest skips completed games)
echo "" | tee -a "$LOG"
echo "[3/5] Incremental PBP ingest" | tee -a "$LOG"
uv run python scripts/gretzky.py ingest 2>&1 | tee -a "$LOG"

# 4. Rebuild RAPM with fresh data
echo "" | tee -a "$LOG"
echo "[4/5] Train RAPM" | tee -a "$LOG"
uv run python scripts/gretzky.py train-rapm -- --force 2>&1 | tee -a "$LOG"

# 5. WAR + EWMA
echo "" | tee -a "$LOG"
echo "[5/5] WAR + EWMA" | tee -a "$LOG"
uv run python scripts/gretzky.py compute-war -- --force 2>&1 | tee -a "$LOG"
uv run python scripts/gretzky.py compute-ewma -- --force 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Done — $(date -u '+%H:%M UTC')" | tee -a "$LOG"

# Keep last 30 days of logs
find "$LOG_DIR" -name "nightly_*.log" -mtime +30 -delete
