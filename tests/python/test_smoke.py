import gretzky


def test_gretzky_imports():
    assert gretzky is not None


def test_gretzky_version():
    assert isinstance(gretzky.__version__, str) and gretzky.__version__


def test_dashboard_api_imports():
    from dashboard.api.main import app
    assert app is not None


def test_data_imports():
    import data
    assert data is not None


def test_models_imports():
    import models
    assert models is not None


def test_backtest_imports():
    import backtest
    assert backtest is not None
