# Targets for the public demo build. Everything here runs against what is in
# this repository — the upstream project's shipping and sync targets are not
# reproduced, because their dispatcher is not part of this build and a target
# that cannot run is worse than an absent one.

.PHONY: help test data dev build-engine test-engine clean

## List these targets
help:
	@awk 'BEGIN {FS = ":"} \
	     /^## / {desc = substr($$0, 4); next} \
	     /^[a-z][a-z-]*:/ && desc {printf "  \033[1m%-13s\033[0m %s\n", $$1, desc; desc = ""}' \
	     $(MAKEFILE_LIST)

## Model and API tests — needs no generated data
test:
	uv run pytest -q

## Rebuild the dataset: real NHL rosters, then the models over them
data:
	uv run python scripts/fetch_nhl.py
	uv run python scripts/make_demo_data.py

## Start FastAPI on :8000 and Next.js on :3000, both with hot reload
dev:
	uv run uvicorn dashboard.api.main:app --reload --port 8000 &
	cd dashboard/frontend && npm run dev

# maturin is not a declared dependency of this build — nothing above needs the
# engine — so it is fetched for the duration of the run rather than installed.
## Build the optional Rust possession simulator into the venv
build-engine:
	uv run --with maturin maturin develop --manifest-path engine/Cargo.toml --release

## Rust unit + integration tests (needs a cargo toolchain)
test-engine:
	cargo test --manifest-path engine/Cargo.toml --all-features

## Remove build artifacts, caches, and the generated dataset
clean:
	cargo clean --manifest-path engine/Cargo.toml 2>/dev/null || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/
