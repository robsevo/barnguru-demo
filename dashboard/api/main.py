"""GRTZKY demo API.

Serves the statistics dashboard from the generated dataset in `data/`.

WHAT THIS IS NOT
----------------
The upstream service is a 16,000-line module with 96 endpoints, roughly a third
of which exist to resolve and proxy live video. None of that is here. This build
is a statistics site, so the API is a statistics API: read the generated data,
shape it for the page asking, return it.

Everything is read-only and computed from files on disk. There is no database, no
scheduler, no credential of any kind, and no outbound network call — which is why
a fresh clone works offline and a reviewer can run it without accounts.

    uv run uvicorn dashboard.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"

app = FastAPI(
    title="GRTZKY demo API",
    description="Hockey statistics and model output over a generated league.",
    version="0.1.0",
)

# The dashboard is served from a different port in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Loading
#
# Everything is cached: the files ship with the build and cannot change under a
# running server, so re-reading them per request would be pure I/O for an
# identical answer. `lru_cache` also means a missing file is diagnosed once
# rather than on every request.
# ---------------------------------------------------------------------------

def _seasons() -> list[int]:
    return sorted(int(p.name) for p in DEMO_DIR.glob("[0-9]" * 4) if p.is_dir())


def LATEST() -> int:
    seasons = _seasons()
    if not seasons:
        raise HTTPException(
            503,
            "No demo data. Generate it with: uv run python scripts/make_demo_data.py",
        )
    return seasons[-1]


@lru_cache(maxsize=None)
def league() -> dict:
    path = DEMO_DIR / "league.json"
    if not path.exists():
        raise HTTPException(503, "data/demo/league.json missing — run scripts/make_demo_data.py")
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def season_file(season: int, name: str) -> list[dict]:
    path = DEMO_DIR / str(season) / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"no {name} for season {season}")
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def model_frame(model: str, season: int) -> pl.DataFrame | None:
    """Read a model's parquet output, or None when that model has not been run.

    Returning None rather than raising is deliberate: a page that shows RAPM
    alongside counting stats should still render the counting stats when the
    model output is absent. The endpoints below surface that as `built: false`.
    """
    path = DATA_DIR / model / f"{model}_{season}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


@lru_cache(maxsize=None)
def skill_index() -> dict[int, float]:
    """Latent production rating the models were asked to recover.

    Written by scripts/make_demo_data.py. Absent until the models have been run,
    which is fine — the profile simply omits the comparison panel.
    """
    path = DEMO_DIR / "skill.json"
    if not path.exists():
        return {}
    return {int(k): float(v) for k, v in json.loads(path.read_text()).items()}


@lru_cache(maxsize=None)
def players_index() -> dict[int, dict]:
    return {p["player_id"]: p for p in league()["players"]}


@lru_cache(maxsize=None)
def teams_index() -> dict[str, dict]:
    return {t["abbrev"]: t for t in league()["teams"]}


# ---------------------------------------------------------------------------
# Shaping helpers
# ---------------------------------------------------------------------------

def _leader_rows(rows: list[dict], value_field: str, limit: int,
                 ascending: bool = False) -> list[dict]:
    """Shape rows into the `Leader` records the stats page renders."""
    ordered = sorted(rows, key=lambda r: r.get(value_field, 0), reverse=not ascending)
    out = []
    for i, r in enumerate(ordered[:limit], 1):
        first, _, last = r["name"].partition(" ")
        out.append({
            "rank": i,
            "id": r["player_id"],
            "name": r["name"],
            "first_name": first,
            "last_name": last,
            "team": r["team"],
            "position": r.get("position", "F"),
            "number": r.get("number", 0),
            "headshot": "",          # generated players have no photograph
            "value": r.get(value_field, 0),
        })
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    seasons = _seasons()
    return {
        "ok": bool(seasons),
        "seasons": seasons,
        "players": len(league()["players"]) if seasons else 0,
        "models": [m for m in ("rapm", "war") if model_frame(m, seasons[-1]) is not None]
        if seasons else [],
    }


@app.get("/league")
def get_league() -> dict:
    lg = league()
    return {"teams": lg["teams"], "coaches": lg.get("coaches", []),
            "seasons": _seasons()}


@app.get("/standings")
def standings(season: int | None = None) -> dict:
    return {"season": season or LATEST(),
            "standings": season_file(season or LATEST(), "standings")}


@app.get("/skater-stats")
def skater_stats(season: int | None = None, limit: int = 10) -> dict:
    """League leaders per category, in the shape the stats page expects."""
    rows = season_file(season or LATEST(), "skaters")
    cats = ["goals", "assists", "points", "plusMinus", "penaltyMinutes",
            "powerPlayGoals", "gameWinningGoals", "shots", "shootingPctg"]
    out = {c: _leader_rows(rows, c, limit) for c in cats}
    out["toi"] = _leader_rows(rows, "toiPerGame", limit)
    return out


@app.get("/goalie-stats")
def goalie_stats(season: int | None = None, limit: int = 10) -> dict:
    rows = season_file(season or LATEST(), "goalies")
    return {
        "wins": _leader_rows(rows, "wins", limit),
        "savePctg": _leader_rows(rows, "savePctg", limit),
        # Lower is better for GAA — sorting it descending would put the worst
        # goalie in the league at the top of a leaderboard.
        "goalsAgainstAverage": _leader_rows(rows, "goalsAgainstAverage", limit, ascending=True),
        "shutouts": _leader_rows(rows, "shutouts", limit),
    }


# The demo season has no playoffs. Returning an empty payload keeps the tabs
# present and honest rather than 404-ing a route the UI knows about.
@app.get("/skater-stats-playoffs")
@app.get("/goalie-stats-playoffs")
def playoffs_stub() -> dict:
    return {}


# The demo season has no playoff round and does not track rookie eligibility.
# These exist so the pages that ask for them get a well-formed empty answer
# rather than a 404 in every console — the shape is real, the list is empty.
@app.get("/playoff-bracket")
def playoff_bracket() -> dict:
    return {"rounds": [], "built": False,
            "note": "the generated season is regular-season only"}


@app.get("/calder-race")
def calder_race() -> dict:
    return {"players": [], "built": False,
            "note": "rookie eligibility is not modelled in the generated league"}


@app.get("/players")
def list_players(
    season: int | None = None,
    team: str | None = None,
    position: str | None = None,
    limit: int = Query(500, le=2000),
) -> dict:
    """Season lines joined to model output, one row per skater."""
    season = season or LATEST()
    rows = season_file(season, "skaters")
    rapm = model_frame("rapm", season)
    war = model_frame("war", season)

    rapm_by_id = {r["player_id"]: r for r in rapm.iter_rows(named=True)} if rapm is not None else {}
    war_by_id = {r["player_id"]: r for r in war.iter_rows(named=True)} if war is not None else {}

    out = []
    for r in rows:
        if team and r["team"] != team:
            continue
        if position and r["position"] != position:
            continue
        m = rapm_by_id.get(r["player_id"], {})
        w = war_by_id.get(r["player_id"], {})
        out.append({
            **r,
            "rapm_ev_off": m.get("rapm_ev_off"),
            "rapm_ev_def": m.get("rapm_ev_def"),
            "xgf_60": m.get("xgf_60"),
            "xga_60": m.get("xga_60"),
            "war": w.get("war"),
            "gar_total": w.get("gar_total"),
        })
    out.sort(key=lambda r: r["points"], reverse=True)
    return {"season": season, "count": len(out), "players": out[:limit],
            "models_built": {"rapm": rapm is not None, "war": war is not None}}


@app.get("/players/{player_id}")
def get_player(player_id: int, season: int | None = None) -> dict:
    season = season or LATEST()
    meta = players_index().get(player_id)
    if meta is None:
        raise HTTPException(404, f"no player {player_id}")

    line = next((r for r in season_file(season, "skaters") if r["player_id"] == player_id), None)
    if line is None:
        line = next((r for r in season_file(season, "goalies") if r["player_id"] == player_id), None)

    history = []
    for s in _seasons():
        rapm = model_frame("rapm", s)
        war = model_frame("war", s)
        entry: dict = {"season": s}
        if rapm is not None:
            row = rapm.filter(pl.col("player_id") == player_id)
            if len(row):
                entry.update(row.to_dicts()[0])
        if war is not None:
            row = war.filter(pl.col("player_id") == player_id)
            if len(row):
                entry["war"] = row.to_dicts()[0]["war"]
                entry["gar_total"] = row.to_dicts()[0]["gar_total"]
        if len(entry) > 1:
            history.append(entry)

    player = {**meta, "production_rating": skill_index().get(player_id)}
    return {"player": player, "season": season, "stats": line, "history": history,
            "team": teams_index().get(meta["team"])}


@app.get("/leaders/{metric}")
def leaders(metric: str, season: int | None = None, limit: int = 20) -> dict:
    """Model leaderboards: rapm_ev_off | rapm_ev_def | rapm_pp | rapm_pk | war | gar_total."""
    season = season or LATEST()
    model = "war" if metric in {"war", "gar_total"} else "rapm"
    df = model_frame(model, season)
    if df is None:
        return {"metric": metric, "built": False, "players": []}
    if metric not in df.columns:
        raise HTTPException(400, f"unknown metric {metric!r}; have {sorted(df.columns)}")

    # Defensive RAPM is measured as expected goals ALLOWED, so a good defender
    # has a negative coefficient. Sorting descending would rank the leaderboard
    # exactly backwards — the kind of bug that looks like a modelling failure.
    ascending = metric in {"rapm_ev_def", "rapm_pk"}
    ranked = df.sort(metric, descending=not ascending).head(limit)
    return {"metric": metric, "season": season, "built": True,
            "ascending": ascending, "players": ranked.to_dicts()}


@app.get("/teams/{abbrev}")
def get_team(abbrev: str, season: int | None = None) -> dict:
    season = season or LATEST()
    team = teams_index().get(abbrev.upper())
    if team is None:
        raise HTTPException(404, f"no team {abbrev}")
    roster = [r for r in season_file(season, "skaters") if r["team"] == abbrev.upper()]
    goalies = [r for r in season_file(season, "goalies") if r["team"] == abbrev.upper()]
    standing = next((r for r in season_file(season, "standings")
                     if r["team"] == abbrev.upper()), None)
    coach = next((c for c in league().get("coaches", [])
                  if c["team"] == abbrev.upper()), None)
    return {"team": team, "season": season, "standing": standing,
            "coach": coach, "skaters": roster, "goalies": goalies}


@app.get("/coaches")
def list_coaches() -> dict:
    return {"coaches": league().get("coaches", [])}


@app.get("/coaches/{name}")
def get_coach(name: str) -> dict:
    wanted = name.replace("-", " ").lower()
    coach = next((c for c in league().get("coaches", [])
                  if c["name"].lower() == wanted), None)
    if coach is None:
        raise HTTPException(404, f"no coach {name}")
    return {"coach": coach, "team": teams_index().get(coach["team"])}
