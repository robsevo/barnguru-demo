.PHONY: test test-rust test-python test-integration build-engine clean dev gretzky epg-ship vod-ship pool-ship sync-accounts sync-relay-token refresh-all

## Run all tests (Rust unit/integration + Python smoke + Python integration)
test: test-rust build-engine test-python test-integration

## Rust: unit tests + integration tests
test-rust:
	cargo test --manifest-path engine/Cargo.toml --all-features

## Build the PyO3 extension into the local venv (required before test-integration)
build-engine:
	uv run maturin develop --manifest-path engine/Cargo.toml

## Python: smoke tests
test-python:
	uv run pytest tests/python/ -v

## Python: integration tests (requires build-engine first)
test-integration:
	uv run pytest tests/integration/ -v

## Start FastAPI (:8000) + Next.js (:3000) dev servers with hot-reload
dev: build-engine
	uv run uvicorn dashboard.api.main:app --reload --port 8000 &
	cd dashboard/frontend && npm run dev

## Run GRETZKY dispatcher (e.g. make gretzky CMD=sync  or  make gretzky CMD=all)
gretzky:
	uv run python scripts/gretzky.py $(CMD)

## Build the merged cable EPG locally + ship it to the box (daily; offloads the
## OOM-prone on-box XMLTV build). Pass flags via ARGS, e.g. make epg-ship ARGS=--dry-run
epg-ship:
	uv run python scripts/gretzky.py epg-ship -- $(ARGS)

## Build the VOD stream index locally + ship it to the box (offloads the
## account-scaling get_vod_streams fan-out that OOM'd the 2GB box). Ships RAW urls;
## the box relay-wraps on load. e.g. make vod-ship ARGS=--dry-run
vod-ship:
	uv run python scripts/vod_ship.py $(ARGS)

## Build the live IPTV channel pool locally + ship it to the box (offloads the
## account-scaling m3u_plus parse — an upstream host ~554k lines — that OOM'd the 2GB box).
## Ships RAW urls; the box relay-wraps on load. e.g. make pool-ship ARGS=--dry-run
pool-ship:
	uv run python scripts/pool_ship.py $(ARGS)

## Pull the relay's base URL + token onto this PC (-> ~/.config/grtzky/relay.env,
## mode 600). Lets the ships verify sources THROUGH the relay's residential IP —
## the accurate check, incl. the curated panels a direct probe can't test.
sync-relay-token:
	uv run python scripts/sync_relay_token.py

## Pull the box's CURRENT upstream accounts onto this PC so the *-ship builds run from
## fresh accounts (the local copy goes stale). Dispatches push-accounts.yml.
sync-accounts:
	uv run python scripts/sync_accounts.py

## One command to refresh everything the box serves: sync fresh accounts, then build
## + ship the EPG, VOD index, and live pool. Run periodically (or wire to a timer).
refresh-all: sync-accounts epg-ship vod-ship pool-ship

## Clean build artifacts
clean:
	cargo clean --manifest-path engine/Cargo.toml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
