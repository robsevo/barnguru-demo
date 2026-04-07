import pytest

gretzky_engine = pytest.importorskip(
    "gretzky_engine",
    reason="Run: uv run maturin develop --manifest-path engine/pyproject.toml",
)


def test_imports():
    assert gretzky_engine is not None


def test_version():
    v = gretzky_engine.gretzky_version()
    assert isinstance(v, str) and v == "0.1.0"
