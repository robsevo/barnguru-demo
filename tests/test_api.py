"""The dashboard API.

Every assertion here holds whether or not the demo dataset has been generated —
CI runs this step even when the NHL fetch fails, and a test that only passes on
a populated machine is a test that goes quietly red for the wrong reason.
"""

from fastapi.testclient import TestClient

from dashboard.api.main import _leader_rows, app

client = TestClient(app)


def _rows() -> list[dict]:
    return [
        {"player_id": 1, "name": "Nathan MacKinnon", "team": "COL", "war": 3.1},
        {"player_id": 2, "name": "Cale Makar",       "team": "COL", "war": 5.4},
        {"player_id": 3, "name": "Connor McDavid",   "team": "EDM", "war": 4.2},
    ]


def test_leader_rows_rank_by_value_descending():
    out = _leader_rows(_rows(), "war", limit=3)
    assert [r["name"] for r in out] == ["Cale Makar", "Connor McDavid", "Nathan MacKinnon"]
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_leader_rows_can_rank_ascending_for_allowed_goals_metrics():
    out = _leader_rows(_rows(), "war", limit=3, ascending=True)
    assert [r["name"] for r in out] == ["Nathan MacKinnon", "Connor McDavid", "Cale Makar"]


def test_leader_rows_respects_the_limit():
    assert len(_leader_rows(_rows(), "war", limit=2)) == 2


def test_leader_rows_splits_a_name_into_first_and_last():
    top = _leader_rows(_rows(), "war", limit=1)[0]
    assert (top["first_name"], top["last_name"]) == ("Cale", "Makar")


def test_leader_rows_defaults_a_missing_metric_to_zero_rather_than_raising():
    out = _leader_rows(_rows(), "not_a_metric", limit=3)
    assert all(r["value"] == 0 for r in out)


def test_health_reports_what_has_been_built():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ok", "seasons", "players", "models"}
    assert isinstance(body["seasons"], list)
    assert set(body["models"]) <= {"rapm", "war"}


def test_an_unknown_leaderboard_metric_is_never_a_500():
    """Three legitimate answers, depending on what has been generated:

      400  the season is built and that metric is not one of its columns
      200  the model has not been run — reported as ``built: false``
      503  no demo data at all, answered with the command that creates it

    An unhandled 500 is the one outcome that is a defect, so the assertion is
    written as a whitelist rather than a single expected code.
    """
    r = client.get("/leaders/not_a_metric")
    assert r.status_code in (200, 400, 503)
    if r.status_code == 200:
        assert r.json()["built"] is False
    if r.status_code == 503:
        assert "make_demo_data" in r.json()["detail"]
