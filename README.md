# GRETZKY — NHL Hockey Simulation Engine

Possession-level NHL game simulator inspired by the Ewing model.

## Quick Start

### Rust engine
```bash
cargo test
```

### Python environment
```bash
uv sync --dev
uv run pytest tests/python/ -v
```

### Build PyO3 extension
```bash
uv run maturin develop --manifest-path engine/Cargo.toml
python -c "import gretzky_engine; print(gretzky_engine.gretzky_version())"
```
