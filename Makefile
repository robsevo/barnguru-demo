.PHONY: test test-rust test-python test-integration build-engine clean dev gretzky

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

## Clean build artifacts
clean:
	cargo clean --manifest-path engine/Cargo.toml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
