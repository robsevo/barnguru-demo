#!/usr/bin/env python3
"""Fetch a season of real NHL data from the league's public API.

WHAT THIS TOUCHES
-----------------
Two public, unauthenticated endpoints:

    api.nhle.com/stats/rest/en/{skater,goalie}/summary   season totals
    api-web.nhle.com/v1/standings/now                    standings

No key, no account, no scraping — these are the same JSON endpoints the league's
own site reads. What comes back is factual: who played, for whom, and what they
did. Nothing is redistributed by this repository; the data is fetched on your
machine and written to `data/`, which is git-ignored.

    uv run python scripts/fetch_nhl.py --season 20242025

WHY IT CACHES
-------------
So that everything downstream — the API, the dashboard, the models — runs
offline afterwards, and so a rate limit or an outage on a Tuesday does not stop
the project from starting.

A NOTE ON THE USER-AGENT
------------------------
`api.nhle.com` rejects some clients. `curl/*` is accepted and is what this sends;
if a future change makes requests start failing with 403, that is the first thing
to look at, not your network.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

STATS_BASE = "https://api.nhle.com/stats/rest/en"
WEB_BASE = "https://api-web.nhle.com/v1"
UA = "curl/8.5.0"
PAGE = 100


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=30.0,
        follow_redirects=True,   # api-web issues 307s
    )


def fetch_summary(client: httpx.Client, kind: str, season: int) -> list[dict]:
    """Page through a season summary. `kind` is "skater" or "goalie"."""
    expr = urllib.parse.quote(f"seasonId={season} and gameTypeId=2")
    out: list[dict] = []
    start = 0
    while True:
        url = (f"{STATS_BASE}/{kind}/summary?isAggregate=false&limit={PAGE}"
               f"&start={start}&cayenneExp={expr}")
        r = client.get(url)
        r.raise_for_status()
        body = r.json()
        rows = body.get("data", [])
        out.extend(rows)
        total = body.get("total", len(out))
        start += PAGE
        if start >= total or not rows:
            break
        time.sleep(0.2)   # be a considerate client
    return out


def fetch_standings(client: httpx.Client) -> list[dict]:
    r = client.get(f"{WEB_BASE}/standings/now")
    r.raise_for_status()
    return r.json().get("standings", [])


def normalise_skaters(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        gp = r.get("gamesPlayed") or 0
        if not gp:
            continue
        out.append({
            "player_id": r["playerId"],
            "name": r.get("skaterFullName") or r.get("lastName", ""),
            # A traded player's `teamAbbrevs` is "BOS,TOR"; the last club is the
            # current one, which is what a roster page should show.
            "team": (r.get("teamAbbrevs") or "").split(",")[-1].strip(),
            "position": r.get("positionCode", "F"),
            "number": 0,                      # not in the summary feed
            "gp": gp,
            "goals": r.get("goals", 0),
            "assists": r.get("assists", 0),
            "points": r.get("points", 0),
            "plusMinus": r.get("plusMinus", 0),
            "penaltyMinutes": r.get("penaltyMinutes", 0),
            "powerPlayGoals": r.get("ppGoals", 0),
            "gameWinningGoals": r.get("gameWinningGoals", 0),
            "shots": r.get("shots", 0),
            "shootingPctg": round(r.get("shootingPct") or 0.0, 4),
            # The feed gives seconds per game; the dashboard shows minutes.
            "toiPerGame": round((r.get("timeOnIcePerGame") or 0.0) / 60.0, 2),
        })
    return out


def normalise_goalies(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        gp = r.get("gamesPlayed") or 0
        if not gp:
            continue
        out.append({
            "player_id": r["playerId"],
            "name": r.get("goalieFullName") or r.get("lastName", ""),
            "team": (r.get("teamAbbrevs") or "").split(",")[-1].strip(),
            "position": "G",
            "number": 0,
            "gp": gp,
            "wins": r.get("wins", 0),
            "losses": r.get("losses", 0),
            "shutouts": r.get("shutouts", 0),
            "goalsAgainstAverage": round(r.get("goalsAgainstAverage") or 0.0, 3),
            "savePctg": round(r.get("savePct") or 0.0, 4),
        })
    return out


def normalise_standings(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        abbrev = (r.get("teamAbbrev") or {}).get("default", "")
        name = (r.get("teamName") or {}).get("default", abbrev)
        out.append({
            "team": abbrev,
            "team_name": name,
            "division": r.get("divisionName", ""),
            "conference": r.get("conferenceName", ""),
            "gp": r.get("gamesPlayed", 0),
            "w": r.get("wins", 0),
            "l": r.get("losses", 0),
            "otl": r.get("otLosses", 0),
            "pts": r.get("points", 0),
            "gf": r.get("goalFor", 0),
            "ga": r.get("goalAgainst", 0),
            "diff": r.get("goalDifferential", 0),
            "pts_pct": round(r.get("pointPctg") or 0.0, 4),
            "div_rank": r.get("divisionSequence", 0),
            "conf_rank": r.get("conferenceSequence", 0),
            "league_rank": r.get("leagueSequence", 0),
            "wildcard_rank": r.get("wildcardSequence", 0),
        })
    return sorted(out, key=lambda x: x["league_rank"] or 99)


def teams_from_standings(standings: list[dict]) -> list[dict]:
    return [
        {"abbrev": s["team"], "name": s["team_name"],
         "conference": s["conference"], "division": s["division"]}
        for s in standings
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=20242025,
                    help="NHL season id, e.g. 20242025")
    args = ap.parse_args()

    year = args.season % 10000          # 20242025 -> 2025
    season_dir = DATA_DIR / "demo" / str(year)
    season_dir.mkdir(parents=True, exist_ok=True)

    with _client() as client:
        print(f"fetching {args.season} …")
        skaters = normalise_skaters(fetch_summary(client, "skater", args.season))
        print(f"  skaters:   {len(skaters)}")
        goalies = normalise_goalies(fetch_summary(client, "goalie", args.season))
        print(f"  goalies:   {len(goalies)}")
        standings = normalise_standings(fetch_standings(client))
        print(f"  standings: {len(standings)} clubs")

    (season_dir / "skaters.json").write_text(json.dumps(skaters, indent=2) + "\n")
    (season_dir / "goalies.json").write_text(json.dumps(goalies, indent=2) + "\n")
    (season_dir / "standings.json").write_text(json.dumps(standings, indent=2) + "\n")

    league = {
        "teams": teams_from_standings(standings),
        "players": skaters + goalies,
        "coaches": [],
        "source": "NHL public API",
        "season": year,
    }
    (DATA_DIR / "demo").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "demo" / "league.json").write_text(json.dumps(league, indent=2) + "\n")

    print(f"wrote data/demo/{year}/ and data/demo/league.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
