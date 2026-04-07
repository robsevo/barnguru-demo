import asyncio
import os
import sys
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ── path bootstrap so data.* and models.* are importable ──────────────────────
_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

app = FastAPI(title="GRTZKY API", version="0.1.0")

# ---------------------------------------------------------------------------
# Simple in-memory TTL cache for expensive NHL API endpoints
# ---------------------------------------------------------------------------
import time as _time

_CACHE: dict[str, tuple[float, object, float]] = {}
_CACHE_TTL = 300.0  # 5 minutes


def _cache_get(key: str) -> object | None:
    entry = _CACHE.get(key)
    if entry:
        ttl = entry[2] if len(entry) > 2 else _CACHE_TTL
        if (_time.monotonic() - entry[0]) < ttl:
            return entry[1]
    return None


def _cache_set(key: str, value: object, ttl: float | None = None) -> None:
    _CACHE[key] = (_time.monotonic(), value, ttl if ttl is not None else _CACHE_TTL)


# ---------------------------------------------------------------------------
# Dev banner state — toggled by rob from the dev dashboard
# ---------------------------------------------------------------------------

_dev_banner: dict = {
    "active":  False,
    "message": "Dev is actively making changes — pages/streams may reload or show errors for a bit.",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],  # POST required for CV start/stop (Feature 16.7)
    allow_headers=["*"],
)

# ── data directory config ──────────────────────────────────────────────────────
# GRETZKY_DATA_DIR env-var overrides the default runtime data location.
# Sub-dirs per module: injuries/, morning_skate/, press_conference/, transactions/, edge/
_DEFAULT_DATA_DIR = Path.home() / ".gretzky" / "data"
_GRETZKY_DATA_DIR = Path(os.environ.get("GRETZKY_DATA_DIR", str(_DEFAULT_DATA_DIR)))

# Legacy: kept for /status backward compat (points at code tree, not runtime data)
DATA_DIR = Path(__file__).parents[2] / "data"


def _module_dir(subdir: str) -> Path:
    return _GRETZKY_DATA_DIR / subdir


_TEAM_KEYWORDS: dict[str, list[str]] = {
    "ANA": ["anaheim"], "ARI": ["arizona"], "BOS": ["boston"], "BUF": ["buffalo"],
    "CGY": ["calgary"], "CAR": ["carolina"], "CHI": ["chicago"], "COL": ["colorado"],
    "CBJ": ["columbus"], "DAL": ["dallas"], "DET": ["detroit"], "EDM": ["edmonton"],
    "FLA": ["florida", "panthers"], "LAK": ["los angeles", "la kings"],
    "MIN": ["minnesota"], "MTL": ["montreal", "montr"],
    "NSH": ["nashville"], "NJD": ["new jersey"], "NYI": ["islanders"],
    "NYR": ["rangers"], "OTT": ["ottawa"], "PHI": ["philadelphia"],
    "PIT": ["pittsburgh"], "SEA": ["seattle"], "SJS": ["san jose"],
    "STL": ["st. louis", "st louis"], "TBL": ["tampa bay"], "TOR": ["toronto"],
    "UTA": ["utah"], "UTH": ["utah"], "VAN": ["vancouver"], "VGK": ["vegas", "golden knights"],
    "WSH": ["washington"], "WPG": ["winnipeg"],
}


def _shots_name_position_map() -> dict[str, str]:
    """Return {shooter_name: player_position} from the most recent shots parquet."""
    import polars as pl
    shots_dir = _GRETZKY_DATA_DIR / "shots"
    files = sorted(shots_dir.glob("*.parquet")) if shots_dir.exists() else []
    if not files:
        return {}
    try:
        df = pl.read_parquet(files[-1], columns=["shooter_name", "player_position"])
        return {
            row["shooter_name"]: row["player_position"]
            for row in df.drop_nulls(subset=["shooter_name", "player_position"])
                         .unique(subset=["shooter_name"])
                         .to_dicts()
            if row["player_position"]
        }
    except Exception:
        return {}


# ── manifest helpers ───────────────────────────────────────────────────────────

def _latest_dates_entry(
    manifest: dict,
    count_key: str = "record_count",
) -> tuple[str, int, str | None]:
    """Return (status, count, synced_at) for the most meaningful dates-keyed entry.

    Prefers the most-recent "ok" entry so that a single empty day at the end
    of a range doesn't mask days with real data.  Falls back to the overall
    most-recent entry if no "ok" entry exists.
    """
    dates: dict = manifest.get("dates", {})
    if not dates:
        return "not_run", 0, None

    sorted_keys = sorted(dates.keys())

    # Prefer most-recent "ok" entry
    ok_keys = [k for k in sorted_keys if dates[k].get("status") == "ok"]
    if ok_keys:
        best = ok_keys[-1]
        entry = dates[best]
        return "ok", entry.get(count_key, 0), entry.get("synced_at")

    # No "ok" entry — return most recent
    latest = sorted_keys[-1]
    entry = dates[latest]
    return (
        entry.get("status", "unknown"),
        entry.get(count_key, 0),
        entry.get("synced_at"),
    )


# ── Existing endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/status")
async def status() -> dict:
    parquet_files = list(DATA_DIR.rglob("*.parquet"))
    data_files = [
        {
            "name": f.name,
            "size_mb": round(f.stat().st_size / 1_048_576, 2),
            "path": str(f.relative_to(DATA_DIR)),
        }
        for f in sorted(parquet_files)
    ]

    try:
        from gretzky_engine import gretzky_version

        engine_version = gretzky_version()
        engine_ok = True
    except ImportError:
        engine_version = None
        engine_ok = False

    return {
        "engine": {"ok": engine_ok, "version": engine_version},
        "data": {"file_count": len(parquet_files), "files": data_files},
        "phases": {
            "phase_0_scaffolding": True,
            "phase_1_data_pipeline": len(parquet_files) > 0,
        },
    }


# ── Dev banner endpoints ───────────────────────────────────────────────────────


@app.get("/dev/banner")
async def get_dev_banner() -> dict:
    """Returns the current dev banner state. Public endpoint."""
    return _dev_banner


@app.post("/dev/banner")
async def set_dev_banner(payload: dict) -> dict:
    """Set dev banner active/message. No server-side auth — frontend restricts to rob."""
    _dev_banner["active"]  = bool(payload.get("active", False))
    _dev_banner["message"] = str(payload.get("message", _dev_banner["message"]))
    return _dev_banner


# ── Phase 1 endpoints ──────────────────────────────────────────────────────────


@app.get("/phase1/modules")
async def phase1_modules() -> dict:
    """Per-module manifest status for the Phase 1 data pipeline.

    Reads each sync module's manifest.json from the runtime data directory
    (GRETZKY_DATA_DIR / <subdir>) and returns status, record count, and last
    sync timestamp.  Returns ``"not_run"`` for modules with no manifest yet.
    """
    from data.edge_sync import EdgeSync
    from data.injury_sync import InjurySync
    from data.morning_skate_sync import MorningSkateSyncer
    from data.moneypuck_goalie_sync import GoalieStatsSyncer
    from data.press_conference_sync import PressConferenceSyncer
    from data.transaction_sync import TransactionSyncer

    modules: list[dict] = []

    # Injuries
    inj_manifest = InjurySync(_module_dir("injuries")).get_manifest()
    s, c, t = _latest_dates_entry(inj_manifest, "record_count")
    modules.append({"name": "injuries", "status": s, "record_count": c, "synced_at": t})

    # Morning skate
    ms_manifest = MorningSkateSyncer(_module_dir("morning_skate")).get_manifest()
    s, c, t = _latest_dates_entry(ms_manifest, "signal_count")
    modules.append({"name": "morning_skate", "status": s, "record_count": c, "synced_at": t})

    # Press conference
    pc_manifest = PressConferenceSyncer(_module_dir("press_conference")).get_manifest()
    s, c, t = _latest_dates_entry(pc_manifest, "mention_count")
    modules.append({"name": "press_conference", "status": s, "record_count": c, "synced_at": t})

    # Transactions — sum event_count across all synced dates, show most-recent sync time
    tx_manifest = TransactionSyncer(_module_dir("transactions")).get_manifest()
    tx_dates: dict = tx_manifest.get("dates", {})
    if not tx_dates:
        modules.append({"name": "transactions", "status": "not_run", "record_count": 0, "synced_at": None})
    else:
        ok_dates = {k: v for k, v in tx_dates.items() if v.get("status") == "ok"}
        total_events = sum(v.get("event_count", 0) for v in ok_dates.values())
        latest_tx = sorted(tx_dates.keys())[-1]
        tx_status = "ok" if ok_dates else tx_dates[latest_tx].get("status", "unknown")
        tx_synced = tx_dates[latest_tx].get("synced_at")
        modules.append({"name": "transactions", "status": tx_status, "record_count": total_events, "synced_at": tx_synced})

    # Goalie stats (MoneyPuck)
    goalie_manifest = GoalieStatsSyncer(_module_dir("goalie_stats")).get_manifest()
    goalie_seasons: dict = goalie_manifest.get("seasons", {})
    if not goalie_seasons:
        modules.append({"name": "goalie_stats", "status": "not_run", "record_count": 0, "synced_at": None})
    else:
        latest_gs = sorted(goalie_seasons.keys())[-1]
        gs_entry = goalie_seasons[latest_gs]
        modules.append({
            "name": "goalie_stats",
            "status": gs_entry.get("status", "unknown"),
            "record_count": gs_entry.get("goalie_count", 0),
            "synced_at": gs_entry.get("synced_at"),
        })

    # EDGE (skating + shot speed share one manifest, keyed by season)
    edge_manifest = EdgeSync(_module_dir("edge")).get_manifest()
    seasons: dict = edge_manifest.get("seasons", {})
    if not seasons:
        modules.append({"name": "edge_skating", "status": "not_run", "record_count": 0, "synced_at": None})
        modules.append({"name": "edge_shot_speed", "status": "not_run", "record_count": 0, "synced_at": None})
    else:
        latest_season = sorted(seasons.keys())[-1]
        entry = seasons[latest_season]
        edge_status = entry.get("status", "unknown")
        edge_synced = entry.get("synced_at")
        modules.append({
            "name": "edge_skating",
            "status": edge_status,
            "record_count": entry.get("skating_count", 0),
            "synced_at": edge_synced,
        })
        modules.append({
            "name": "edge_shot_speed",
            "status": edge_status,
            "record_count": entry.get("shot_count", 0),
            "synced_at": edge_synced,
        })

    return {"modules": modules}


@app.get("/phase1/availability")
async def phase1_availability(
    player: str = Query(..., description="Player name (case-insensitive)"),
    date: str | None = Query(None, description="Game date YYYY-MM-DD (default: today)"),
) -> dict:
    """Player availability prediction using Phase 1 signals.

    Reads injury, morning-skate, and press-conference Parquets for the given
    date and runs PlayerAvailabilityModel.predict_from_dataframes().  With no
    synced data the model returns the prior (P ≈ 0.97, confidence = 0.0).
    """
    from data.injury_sync import InjurySync
    from data.morning_skate_sync import MorningSkateSyncer
    from data.press_conference_sync import PressConferenceSyncer
    from models.player_availability import PlayerAvailabilityModel

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    inj_df = InjurySync(_module_dir("injuries")).get_injuries(date)
    ms_df = MorningSkateSyncer(_module_dir("morning_skate")).get_signals(date)
    pc_df = PressConferenceSyncer(_module_dir("press_conference")).get_mentions(date)

    pred = PlayerAvailabilityModel().predict_from_dataframes(
        player_name=player,
        date=date,
        injury_df=inj_df,
        morning_skate_df=ms_df,
        press_conference_df=pc_df,
    )

    return {
        "player_name": pred.player_name,
        "date": pred.date,
        "p_plays": round(pred.p_plays, 4),
        "confidence": round(pred.confidence, 4),
        "signal_count": pred.signal_count,
        "dominant_signal": pred.dominant_signal,
        "notes": pred.notes,
    }


@app.get("/phase1/edge/player")
async def phase1_edge_player(
    name: str = Query(..., description="Player name (case-insensitive)"),
    season: int | None = Query(None, description="Season start-year (default: most recent)"),
) -> dict:
    """EDGE tracking stats for a single player with league percentiles.

    Returns max skating speed, distance per game, zone time breakdown, and
    top shot speed — all with percentile rank vs the full season dataset.
    """
    import polars as pl

    edge_dir = _module_dir("edge")
    sk_files = sorted(edge_dir.glob("edge_skating_*.parquet")) if edge_dir.exists() else []
    sh_files = sorted(edge_dir.glob("edge_shot_speed_*.parquet")) if edge_dir.exists() else []

    if not sk_files:
        return {"not_found": True, "reason": "no_edge_data"}

    # Resolve season
    if season is None:
        # Infer from filename: edge_skating_2024.parquet → 2024
        season = int(sk_files[-1].stem.split("_")[-1])

    sk_path = edge_dir / f"edge_skating_{season}.parquet"
    sh_path = edge_dir / f"edge_shot_speed_{season}.parquet"

    if not sk_path.exists():
        return {"not_found": True, "reason": f"no_data_for_season_{season}"}

    df_sk = pl.read_parquet(sk_path)

    # Case-insensitive name match: exact first, then substring
    name_lower = name.strip().lower()
    mask_exact = pl.col("player_name").str.to_lowercase() == name_lower
    mask_sub   = pl.col("player_name").str.to_lowercase().str.contains(name_lower)
    row = df_sk.filter(mask_exact)
    if row.is_empty():
        row = df_sk.filter(mask_sub)
    if row.is_empty():
        return {"not_found": True, "reason": "player_not_found"}

    r = row.row(0, named=True)

    def _pct(col: str, val: float | None) -> float | None:
        if val is None:
            return None
        col_data = df_sk[col].drop_nulls()
        if col_data.is_empty():
            return None
        return round(float((col_data < val).mean()), 3)

    # Shot speed from separate Parquet
    max_shot: float | None = None
    shot_pct: float | None = None
    if sh_path.exists():
        df_sh = pl.read_parquet(sh_path)
        sh_row = df_sh.filter(pl.col("player_name").str.to_lowercase() == r["player_name"].lower())
        if not sh_row.is_empty():
            max_shot = sh_row["max_shot_speed_mph"][0]
            sh_col = df_sh["max_shot_speed_mph"].drop_nulls()
            if not sh_col.is_empty() and max_shot is not None:
                shot_pct = round(float((sh_col < max_shot).mean()), 3)

    return {
        "not_found": False,
        "player_id": int(r["player_id"]) if r.get("player_id") else None,
        "player_name": r["player_name"],
        "team": r["team"],
        "season": season,
        "games_played": r["games_played"],
        "max_speed_kmh": r["max_speed_kmh"],
        "max_speed_pct": _pct("max_speed_kmh", r["max_speed_kmh"]),
        "distance_per_game_km": round(r["distance_per_game_km"], 3) if r["distance_per_game_km"] else None,
        "distance_pct": _pct("distance_per_game_km", r["distance_per_game_km"]),
        "zone_time_pct_oz": r["zone_time_pct_oz"],
        "zone_time_pct_nz": r["zone_time_pct_nz"],
        "zone_time_pct_dz": r["zone_time_pct_dz"],
        "max_shot_speed_mph": max_shot,
        "max_shot_speed_pct": shot_pct,
    }


@app.get("/phase1/players")
async def phase1_players() -> dict:
    """Known player names for client-side autocomplete in the availability lookup.

    Sources (in order):
    1. EDGE skating Parquet — all skaters in the most recent season (~735 players)
    2. All historical injury Parquets — all players ever injured including goalies
    """
    import polars as pl

    name_team: dict[str, str] = {}  # name -> team (best known)

    # Primary: EDGE skating Parquet — has team column, most current
    edge_dir = _module_dir("edge")
    edge_files = sorted(edge_dir.glob("edge_skating_*.parquet")) if edge_dir.exists() else []
    if edge_files:
        df_edge = pl.read_parquet(edge_files[-1])
        if "player_name" in df_edge.columns:
            cols = ["player_name"] + (["team"] if "team" in df_edge.columns else [])
            for row in df_edge.select(cols).drop_nulls(subset=["player_name"]).unique(subset=["player_name"]).to_dicts():
                name_team[row["player_name"]] = row.get("team") or ""

    # Injury parquets — catches goalies + injured players not in EDGE
    inj_dir = _module_dir("injuries")
    for parquet_file in sorted(inj_dir.glob("injuries_*.parquet")):
        try:
            df = pl.read_parquet(parquet_file)
            if "player_name" in df.columns:
                cols = ["player_name"] + (["team_code"] if "team_code" in df.columns else [])
                for row in df.select(cols).drop_nulls(subset=["player_name"]).unique(subset=["player_name"]).to_dicts():
                    n = row["player_name"]
                    if n not in name_team:
                        name_team[n] = row.get("team_code") or ""
        except Exception:
            pass

    # Goalie stats — healthy starters not in EDGE or injuries
    goalie_names: set[str] = set()
    goalie_dir = _module_dir("goalie_stats")
    for parquet_file in sorted(goalie_dir.glob("goalie_stats_*.parquet")):
        try:
            df = pl.read_parquet(parquet_file)
            if "player_name" in df.columns and "situation" in df.columns:
                cols = ["player_name"] + (["team"] if "team" in df.columns else [])
                for row in (
                    df.filter(pl.col("situation") == "all")
                    .select(cols).drop_nulls(subset=["player_name"]).unique(subset=["player_name"]).to_dicts()
                ):
                    n = row["player_name"]
                    goalie_names.add(n)
                    if n not in name_team:
                        name_team[n] = row.get("team") or ""
        except Exception:
            pass

    pos_map = _shots_name_position_map()
    # injuries parquet has position directly — prefer it over shots-derived position
    inj_dir2 = _module_dir("injuries")
    inj_pos: dict[str, str] = {}
    for parquet_file in sorted(inj_dir2.glob("injuries_*.parquet")):
        try:
            df = pl.read_parquet(parquet_file)
            if "player_name" in df.columns and "position" in df.columns:
                for row in df.select(["player_name", "position"]).drop_nulls(subset=["player_name", "position"]).to_dicts():
                    inj_pos[row["player_name"]] = row["position"]
        except Exception:
            pass

    players = [
        {"name": n, "team": t, "position": inj_pos.get(n) or pos_map.get(n) or ("G" if n in goalie_names else "")}
        for n, t in sorted(name_team.items())
    ]
    return {"players": players}


@app.get("/phase1/goalie")
async def phase1_goalie(
    name: str = Query(..., description="Goalie name (case-insensitive)"),
    season: int | None = Query(None, description="Season start-year (default: most recent)"),
    situation: str = Query("all", description="Situation filter: all | 5on5 | 4on5 | 5on4 | other"),
) -> dict:
    """Goalie advanced stats from MoneyPuck with GSAx, HDSV%, and danger-zone breakdown.

    Returns stats for the specified situation (default: 'all' = aggregate).
    GSAx (Goals Saved Above Expected) = xGoals − goals_against: positive = elite.
    """
    from data.moneypuck_goalie_sync import GoalieStatsSyncer

    syncer = GoalieStatsSyncer(_module_dir("goalie_stats"))
    manifest = syncer.get_manifest()
    seasons_map = manifest.get("seasons", {})

    if not seasons_map:
        return {"not_found": True, "reason": "no_goalie_data"}

    if season is None:
        season = int(sorted(seasons_map.keys())[-1])

    if seasons_map.get(str(season), {}).get("status") != "ok":
        return {"not_found": True, "reason": f"no_data_for_season_{season}"}

    row = syncer.get_goalie(name, season, situation)
    if row.is_empty():
        return {"not_found": True, "reason": "goalie_not_found"}

    r = row.row(0, named=True)
    return {
        "not_found": False,
        "player_id": int(r["player_id"]),
        "player_name": r["player_name"],
        "team": r["team"],
        "season": int(r["season"]),
        "situation": r["situation"],
        "games_played": int(r["games_played"]),
        "shots": int(r["shots"]),
        "saves": int(r["saves"]),
        "goals_against": int(r["goals_against"]),
        "sv_pct": r["sv_pct"],
        "xga": r["xga"],
        "gsax": r["gsax"],
        "hd_shots": r["hd_shots"],
        "hd_saves": r["hd_saves"],
        "hd_goals": r["hd_goals"],
        "hdsv_pct": r["hdsv_pct"],
        "md_shots": r["md_shots"],
        "md_saves": r["md_saves"],
        "md_goals": r["md_goals"],
        "mdsv_pct": r["mdsv_pct"],
        "ld_shots": r["ld_shots"],
        "ld_saves": r["ld_saves"],
        "ld_goals": r["ld_goals"],
        "ldsv_pct": r["ldsv_pct"],
    }


@app.get("/phase1/injuries")
async def phase1_injuries(
    date: str | None = Query(None, description="Date YYYY-MM-DD (default: today)"),
) -> dict:
    """Injury records for a given date."""
    from data.injury_sync import InjurySync

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    df = InjurySync(_module_dir("injuries")).get_injuries(date)
    records = df.to_dicts() if not df.is_empty() else []
    return {"date": date, "count": len(records), "injuries": records}


@app.get("/phase1/recommendations")
async def phase1_recommendations() -> dict:
    """Recommended players to check availability for today.

    Returns DTD (questionable) players first, then Out players sorted by
    return_date_estimate ascending (soonest return first). Includes position
    so the UI can surface goalies distinctly.
    """
    import polars as pl
    from data.injury_sync import InjurySync

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = InjurySync(_module_dir("injuries")).get_injuries(today)

    if df.is_empty():
        return {"date": today, "recommendations": []}

    # Priority: DTD first, then Out — both sorted by return_date_estimate asc
    status_order = {"DTD": 0, "Out": 1, "Probable": 2, "Questionable": 3}
    records = df.to_dicts()
    records.sort(key=lambda r: (
        status_order.get(r.get("status", ""), 9),
        r.get("return_date_estimate") or "9999-99-99",
        r.get("player_name", ""),
    ))

    return {
        "date": today,
        "recommendations": [
            {
                "player_name": r["player_name"],
                "position": r.get("position"),
                "team_code": r.get("team_code"),
                "status": r.get("status"),
                "return_date_estimate": r.get("return_date_estimate"),
            }
            for r in records
        ],
    }


@app.get("/phase1/transactions")
async def phase1_transactions(
    start: str | None = Query(None, description="Start date YYYY-MM-DD (default: 7 days ago)"),
    end: str | None = Query(None, description="End date YYYY-MM-DD (default: today)"),
) -> dict:
    """Transaction events across a date range (default: last 7 days)."""
    from data.transaction_sync import TransactionSyncer

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if end is None:
        end = today
    if start is None:
        start = (_date.fromisoformat(end) - timedelta(days=6)).isoformat()

    df = TransactionSyncer(_module_dir("transactions")).get_events_range(start, end)
    records = df.to_dicts() if not df.is_empty() else []
    return {"start": start, "end": end, "count": len(records), "transactions": records}


# ===========================================================================
# Scoreboard — today's games (live scores, period, time)
# ===========================================================================


@app.get("/scoreboard")
async def scoreboard() -> dict:
    """Today's NHL games with live scores, period, and clock.

    Returns a normalised list — never crashes the frontend.
    game_state values: FUT (scheduled), PRE (warmup), LIVE, CRIT (late close game),
                       FINAL, OFF (post-game wrap-up)
    outcome_type: "OT" | "SO" | null (only set when game_state is FINAL/OFF)
    """
    from data.nhl_client import NHLClient, NHLApiError

    _PERIOD_LABELS = {1: "1ST", 2: "2ND", 3: "3RD", 4: "OT", 5: "SO"}

    try:
        async with NHLClient() as client:
            raw = await client.get_scoreboard()
    except NHLApiError as exc:
        return {"games": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"games": [], "error": f"Unexpected error: {exc}"}

    games_by_date = raw.get("gamesByDate") or []
    if not games_by_date:
        return {"games": []}

    from datetime import date, timedelta
    # Use NHL's focusedDate to avoid UTC midnight / ET timezone mismatch
    today_str          = raw.get("focusedDate") or date.today().isoformat()
    _today             = date.fromisoformat(today_str)
    yesterday_str      = (_today - timedelta(days=1)).isoformat()
    two_days_ago_str   = (_today - timedelta(days=2)).isoformat()
    keep_dates         = {today_str, yesterday_str, two_days_ago_str}

    # Flatten last 2 days + today
    all_games: list[dict] = []
    date_labels: dict = {}
    for bucket in games_by_date:
        date_str = bucket.get("date", "")
        if date_str not in keep_dates:
            continue
        for g in bucket.get("games") or []:
            all_games.append(g)
            if g.get("id"):
                date_labels[g["id"]] = date_str

    normalised = []
    for g in all_games:
        away = g.get("awayTeam") or {}
        home = g.get("homeTeam") or {}
        clock = g.get("clock") or {}
        outcome = g.get("gameOutcome") or {}
        game_state = g.get("gameState", "FUT")

        period = g.get("period") or 0
        period_label = _PERIOD_LABELS.get(period, f"P{period}") if period else None

        outcome_type: str | None = None
        if game_state in ("FINAL", "OFF"):
            lpt = outcome.get("lastPeriodType", "REG")
            if lpt in ("OT", "SO"):
                outcome_type = lpt

        sit_code = (g.get("situation") or {}).get("situationCode", "")
        away_on_pp = False
        home_on_pp = False
        away_skaters = 5
        home_skaters = 5
        if len(sit_code) >= 3 and not bool(clock.get("inIntermission", False)):
            try:
                away_sk = int(sit_code[1])
                home_sk = int(sit_code[2])
                away_on_pp = away_sk > home_sk
                home_on_pp = home_sk > away_sk
                away_skaters = away_sk
                home_skaters = home_sk
            except (ValueError, IndexError):
                pass

        normalised.append({
            "game_id":         g.get("id"),
            "date":            date_labels.get(g.get("id"), ""),
            "game_state":      game_state,
            "away_team":       away.get("abbrev", ""),
            "home_team":       home.get("abbrev", ""),
            "away_score":      away.get("score"),
            "home_score":      home.get("score"),
            "period":          period,
            "period_label":    period_label,
            "time_remaining":  clock.get("timeRemaining"),
            "in_intermission": bool(clock.get("inIntermission", False)),
            "start_time_utc":  g.get("startTimeUTC"),
            "outcome_type":    outcome_type,
            "away_on_pp":      away_on_pp,
            "home_on_pp":      home_on_pp,
            "away_skaters":    away_skaters,
            "home_skaters":    home_skaters,
        })

    return {"games": normalised}


# ===========================================================================
# Schedule — games for any date
# ===========================================================================

@app.get("/schedule")
async def schedule(date: str = "now") -> dict:
    """Games for a given date (YYYY-MM-DD or 'now').

    Returns the same normalised shape as /scoreboard so the frontend
    can reuse the same rendering logic.
    """
    import re as _re
    from data.nhl_client import NHLClient, NHLApiError

    _PERIOD_LABELS = {1: "1ST", 2: "2ND", 3: "3RD", 4: "OT", 5: "SO"}

    if date != "now" and not _re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return {"games": [], "error": "Invalid date format. Use YYYY-MM-DD or 'now'."}

    try:
        async with NHLClient() as client:
            raw, scoreboard_raw = await asyncio.gather(
                client.get_schedule(date),
                client.get_scoreboard(),
                return_exceptions=True,
            )
            if isinstance(raw, Exception):
                raise raw
    except NHLApiError as exc:
        return {"games": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"games": [], "error": f"Unexpected error: {exc}"}

    # Build a live-clock lookup keyed by game_id from the scoreboard
    live_lookup: dict[int, dict] = {}
    if isinstance(scoreboard_raw, dict):
        for bucket in scoreboard_raw.get("gamesByDate") or []:
            for g in bucket.get("games") or []:
                gid = g.get("id")
                if gid and g.get("gameState") in ("LIVE", "CRIT"):
                    live_lookup[gid] = g

    # Resolve the target date string (NHL API gameWeek spans multiple days)
    from datetime import date as _date
    target_date = _date.today().isoformat() if date == "now" else date

    games_by_date = raw.get("gameWeek") or raw.get("gamesByDate") or []
    normalised = []
    for bucket in games_by_date:
        date_str = bucket.get("date", "")
        # Only include games for the requested date
        if date_str and date_str != target_date:
            continue
        for g in bucket.get("games") or []:
            game_id   = g.get("id")
            away      = g.get("awayTeam") or {}
            home      = g.get("homeTeam") or {}
            outcome   = g.get("gameOutcome") or {}
            game_state = g.get("gameState", "FUT")

            # For live games the schedule endpoint omits clock/period —
            # pull them from the scoreboard instead.
            live = live_lookup.get(game_id) if game_id else None
            src  = live if live else g

            clock  = src.get("clock") or {}
            period = src.get("period") or 0
            period_label = _PERIOD_LABELS.get(period, f"P{period}") if period else None

            outcome_type: str | None = None
            if game_state in ("FINAL", "OFF"):
                lpt = outcome.get("lastPeriodType", "REG")
                if lpt in ("OT", "SO"):
                    outcome_type = lpt

            sit_code = (src.get("situation") or {}).get("situationCode", "")
            away_on_pp = False
            home_on_pp = False
            away_skaters = 5
            home_skaters = 5
            if len(sit_code) >= 3 and not bool(clock.get("inIntermission", False)):
                try:
                    away_sk = int(sit_code[1])
                    home_sk = int(sit_code[2])
                    away_on_pp = away_sk > home_sk
                    home_on_pp = home_sk > away_sk
                    away_skaters = away_sk
                    home_skaters = home_sk
                except (ValueError, IndexError):
                    pass

            normalised.append({
                "game_id":         game_id,
                "date":            date_str,
                "game_state":      game_state,
                "away_team":       away.get("abbrev", ""),
                "home_team":       home.get("abbrev", ""),
                "away_score":      away.get("score"),
                "home_score":      home.get("score"),
                "period":          period,
                "period_label":    period_label,
                "time_remaining":  clock.get("timeRemaining"),
                "in_intermission": bool(clock.get("inIntermission", False)),
                "start_time_utc":  g.get("startTimeUTC"),
                "outcome_type":    outcome_type,
                "venue":           g.get("venue", {}).get("default", ""),
                "away_on_pp":      away_on_pp,
                "home_on_pp":      home_on_pp,
                "away_skaters":    away_skaters,
                "home_skaters":    home_skaters,
            })

    return {"games": normalised}


# ===========================================================================
# Goal feed — today's goals, most recent first
# ===========================================================================


@app.get("/goals")
async def goal_feed() -> dict:
    """Today's goals from all started NHL games, sorted most-recent first.

    Fetches the scoreboard to find today's active game IDs, then pulls each
    game's landing page in parallel and extracts summary.scoring.
    Returns [] if no games have started or no goals have been scored.
    """
    import asyncio
    from data.nhl_client import NHLClient, NHLApiError

    _PERIOD_LABELS = {1: "1ST", 2: "2ND", 3: "3RD", 4: "OT", 5: "SO"}

    async def _extract_goals(
        client: "NHLClient",
        game_id: int,
        away: str,
        home: str,
        game_state: str,
        start_epoch: int,
    ) -> list[dict]:
        try:
            landing = await client.get_landing(game_id)
        except Exception:
            return []

        out: list[dict] = []
        scoring = (landing.get("summary") or {}).get("scoring") or []
        for block in scoring:
            pd = block.get("periodDescriptor") or {}
            period_num = pd.get("number", 0)
            period_type = pd.get("periodType", "REG")
            if period_type == "OT":
                period_label = "OT"
            elif period_type == "SO":
                period_label = "SO"
            else:
                period_label = _PERIOD_LABELS.get(period_num, f"P{period_num}")

            for goal in block.get("goals") or []:
                scorer_id = goal.get("playerId")
                first = (goal.get("firstName") or {}).get("default", "")
                last  = (goal.get("lastName")  or {}).get("default", "")
                scorer = f"{first[0]}. {last}" if first else last

                assists: list[str] = []
                for a in goal.get("assists") or []:
                    af = (a.get("firstName") or {}).get("default", "")
                    al = (a.get("lastName")  or {}).get("default", "")
                    assists.append(f"{af[0]}. {al}" if af else al)

                time_str = goal.get("timeInPeriod", "")
                strength = goal.get("strength", "EV")
                if goal.get("goalModifier") == "empty-net":
                    strength = "EN"

                try:
                    m, s = map(int, time_str.split(":"))
                    # Wall-clock sort: game start epoch + real elapsed time.
                    # Each completed period adds 1200s (play) + 1020s (17-min intermission).
                    sort_key = start_epoch + (period_num - 1) * 2220 + m * 60 + s
                except Exception:
                    sort_key = start_epoch + (period_num - 1) * 2220

                out.append({
                    "game_id":     game_id,
                    "away_team":   away,
                    "home_team":   home,
                    "game_state":  game_state,
                    "period":      period_num,
                    "period_label": period_label,
                    "time":        time_str,
                    "team":        (goal.get("teamAbbrev") or {}).get("default", ""),
                    "scorer":      scorer,
                    "scorer_id":   int(scorer_id) if scorer_id else None,
                    "headshot_url": goal.get("headshot"),
                    "assists":     assists,
                    "away_score":  goal.get("awayScore"),
                    "home_score":  goal.get("homeScore"),
                    "strength":    strength,
                    "scored_at":   sort_key,  # UTC epoch approx: game_start + period_offset + time_in_period
                    "_sort":       sort_key,
                })
        return out

    try:
        async with NHLClient() as client:
            raw_sb = await client.get_scoreboard()

            # Use NHL's own focusedDate — avoids UTC midnight / ET timezone mismatch
            today_str = raw_sb.get("focusedDate") or _date.today().isoformat()
            game_info: list[tuple[int, str, str, str, int]] = []  # (id, away, home, state, start_epoch)
            for bucket in (raw_sb.get("gamesByDate") or []):
                if bucket.get("date") != today_str:
                    continue
                for g in bucket.get("games") or []:
                    state = g.get("gameState", "FUT")
                    if state in ("FUT", "PRE"):
                        continue
                    gid = g.get("id")
                    if gid:
                        away = (g.get("awayTeam") or {}).get("abbrev", "")
                        home = (g.get("homeTeam") or {}).get("abbrev", "")
                        try:
                            from datetime import datetime, timezone
                            utc_str = g.get("startTimeUTC") or ""
                            start_epoch = int(datetime.fromisoformat(utc_str.replace("Z", "+00:00")).timestamp())
                        except Exception:
                            start_epoch = 0
                        game_info.append((gid, away, home, state, start_epoch))

            if not game_info:
                return {"goals": []}

            results = await asyncio.gather(
                *[_extract_goals(client, gid, away, home, state, start_epoch)
                  for gid, away, home, state, start_epoch in game_info],
                return_exceptions=True,
            )
    except NHLApiError as exc:
        return {"goals": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"goals": [], "error": f"Unexpected error: {exc}"}

    all_goals: list[dict] = []
    for r in results:
        if isinstance(r, list):
            all_goals.extend(r)

    all_goals.sort(key=lambda g: g["_sort"], reverse=True)
    for g in all_goals:
        del g["_sort"]

    return {"goals": all_goals}


# ===========================================================================
# Standings — current NHL standings
# ===========================================================================


@app.get("/standings")
async def standings() -> dict:
    """Current NHL standings from api-web.nhle.com/v1/standings/now.

    Returns one record per team with division, conference, wildcard, and
    league sequence ranks plus the core stats (GP, W, L, OTL, PTS).
    """
    from data.nhl_client import NHLClient, NHLApiError

    from datetime import date as _d
    today = _d.today().isoformat()

    try:
        async with NHLClient() as client:
            raw = await client._get(client._web, f"/v1/standings/{today}")
    except NHLApiError as exc:
        return {"standings": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"standings": [], "error": f"Unexpected error: {exc}"}

    out: list[dict] = []
    for t in raw.get("standings") or []:
        abbrev = (t.get("teamAbbrev") or {}).get("default", "")
        if not abbrev:
            continue
        out.append({
            "team":              abbrev,
            "team_name":         (t.get("teamCommonName") or {}).get("default", abbrev),
            "division":          t.get("divisionName", ""),
            "conference":        t.get("conferenceName", ""),
            "gp":                t.get("gamesPlayed", 0),
            "w":                 t.get("wins", 0),
            "l":                 t.get("losses", 0),
            "otl":               t.get("otLosses", 0),
            "pts":               t.get("points", 0),
            "div_rank":          t.get("divisionSequence", 0),
            "wildcard_rank":     t.get("wildcardSequence", 0),
            "conf_rank":         t.get("conferenceSequence", 0),
            "league_rank":       t.get("leagueSequence", 0),
            "pts_pct":           round(t.get("pointPctg", 0.0), 3),
            "streak_code":       t.get("streakCode", ""),
            "streak_count":      t.get("streakCount", 0),
            # Extended fields
            "gf":                t.get("goalFor", 0),
            "ga":                t.get("goalAgainst", 0),
            "goal_diff":         t.get("goalDifferential", 0),
            "home_w":            t.get("homeWins", 0),
            "home_l":            t.get("homeLosses", 0),
            "home_otl":          t.get("homeOtLosses", 0),
            "away_w":            t.get("roadWins", 0),
            "away_l":            t.get("roadLosses", 0),
            "away_otl":          t.get("roadOtLosses", 0),
            "l10_w":             t.get("l10Wins", 0),
            "l10_l":             t.get("l10Losses", 0),
            "l10_otl":           t.get("l10OtLosses", 0),
            "so_w":              t.get("shootoutWins", 0),
            "so_l":              t.get("shootoutLosses", 0),
            "clinch_indicator":  t.get("clinchIndicator", ""),
            "regulation_wins":   t.get("regulationWins", 0),
        })

    return {"standings": out}


# ===========================================================================
# Stats Leaders — skaters and goalies
# ===========================================================================


@app.get("/skater-stats")
async def skater_stats() -> dict:
    """NHL skater stats leaders for the current season.

    Uses the NHL stats REST API which supports all categories including
    PIM, PPG, GWG, shots, and shooting %.
    All 10 category requests fire in parallel via asyncio.gather().
    """
    cached = _cache_get("skater-stats")
    if cached is not None:
        return cached  # type: ignore[return-value]

    # (result_key, rest_sort_field, value_field)
    CAT_MAP: list[tuple[str, str, str]] = [
        ("goals",            "goals",            "goals"),
        ("assists",          "assists",           "assists"),
        ("points",           "points",            "points"),
        ("plusMinus",        "plusMinus",         "plusMinus"),
        ("penaltyMinutes",   "penaltyMinutes",    "penaltyMinutes"),
        ("powerPlayGoals",   "ppGoals",           "ppGoals"),
        ("gameWinningGoals", "gameWinningGoals",  "gameWinningGoals"),
        ("toi",              "timeOnIcePerGame",  "timeOnIcePerGame"),
        ("shots",            "shots",             "shots"),
        ("shootingPctg",     "shootingPct",       "shootingPct"),
    ]
    BASE = "https://api.nhle.com/stats/rest/en/skater/summary"
    _UA  = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    # Extra cayenne filters per category to avoid trivial leaders (e.g. 1/1 shots = 100%)
    EXTRA: dict[str, str] = {
        "shootingPctg": " and shots >= 50",
        "toi":          " and gamesPlayed >= 20",
    }

    async def _fetch_cat(client: httpx.AsyncClient, result_key: str, sort_field: str, value_field: str) -> tuple[str, list]:
        try:
            extra = EXTRA.get(result_key, "")
            r = await client.get(
                BASE,
                params={
                    "cayenneExp": f"seasonId=20252026 and gameTypeId=2{extra}",
                    "sort":       sort_field,
                    "dir":        "DESC",
                    "limit":      40,
                },
            )
            if r.status_code != 200:
                return result_key, []
            rows = r.json().get("data", [])
            leaders = []
            for rank, p in enumerate(rows, 1):
                pid  = p.get("playerId", 0)
                team = (p.get("teamAbbrevs") or "").split(",")[0].strip()
                headshot = f"https://assets.nhle.com/mugs/nhl/20252026/{team}/{pid}.png" if team and pid else ""
                full = p.get("skaterFullName", "")
                parts = full.split(" ", 1)
                leaders.append({
                    "rank":       rank,
                    "id":         pid,
                    "name":       full,
                    "first_name": parts[0] if parts else "",
                    "last_name":  parts[1] if len(parts) > 1 else "",
                    "team":       team,
                    "position":   p.get("positionCode", ""),
                    "number":     None,
                    "headshot":   headshot,
                    "value":      p.get(value_field),
                })
            return result_key, leaders
        except Exception:  # noqa: BLE001
            return result_key, []

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": _UA}) as client:
        pairs = await asyncio.gather(*[_fetch_cat(client, rk, sf, vf) for rk, sf, vf in CAT_MAP])

    result = dict(pairs)
    payload = {"categories": result}
    _cache_set("skater-stats", payload)
    return payload


@app.get("/calder-race")
async def calder_race() -> dict:
    """NHL rookie stats leaders — Calder Trophy race.
    All 3 category requests fire in parallel via asyncio.gather().
    """
    cached = _cache_get("calder-race")
    if cached is not None:
        return cached  # type: ignore[return-value]

    CATEGORIES = ["points", "goals", "assists"]
    BASE = "https://api.nhle.com/stats/rest/en/skater/summary"
    _UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    async def _fetch_cat(client: httpx.AsyncClient, cat: str) -> tuple[str, list]:
        try:
            r = await client.get(
                BASE,
                params={
                    "cayenneExp": "seasonId=20252026 and isRookie=1 and gameTypeId=2",
                    "sort":       cat,
                    "dir":        "DESC",
                    "limit":      40,
                },
            )
            if r.status_code != 200:
                return cat, []
            rows = r.json().get("data", [])
            leaders = []
            for rank, p in enumerate(rows, 1):
                pid  = p.get("playerId", 0)
                team = (p.get("teamAbbrevs") or "").split(",")[0].strip()
                headshot = f"https://assets.nhle.com/mugs/nhl/20252026/{team}/{pid}.png" if team and pid else ""
                full = p.get("skaterFullName", "")
                parts = full.split(" ", 1)
                leaders.append({
                    "rank":       rank,
                    "id":         pid,
                    "name":       full,
                    "first_name": parts[0] if parts else "",
                    "last_name":  parts[1] if len(parts) > 1 else "",
                    "team":       team,
                    "position":   p.get("positionCode", ""),
                    "number":     None,
                    "headshot":   headshot,
                    "value":      p.get(cat),
                })
            return cat, leaders
        except Exception:  # noqa: BLE001
            return cat, []

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": _UA}) as client:
        pairs = await asyncio.gather(*[_fetch_cat(client, cat) for cat in CATEGORIES])

    payload = {"categories": dict(pairs)}
    _cache_set("calder-race", payload)
    return payload


@app.get("/goalie-stats")
async def goalie_stats() -> dict:
    """NHL goalie stats leaders for the current season.

    Returns top-25 for: wins, savePctg, goalsAgainstAverage, shutouts.
    All 4 category requests fire in parallel via asyncio.gather().
    """
    cached = _cache_get("goalie-stats")
    if cached is not None:
        return cached  # type: ignore[return-value]

    CATEGORIES = ["wins", "savePctg", "goalsAgainstAverage", "shutouts"]

    async def _fetch_cat(client: httpx.AsyncClient, cat: str) -> tuple[str, list]:
        try:
            r = await client.get(
                f"https://api-web.nhle.com/v1/goalie-stats-leaders/20252026/2"
                f"?categories={cat}&limit=40"
            )
            if r.status_code != 200:
                return cat, []
            raw_list = r.json().get(cat, [])
            leaders = []
            for rank, p in enumerate(raw_list, 1):
                leaders.append({
                    "rank":       rank,
                    "id":         p.get("id"),
                    "name":       f"{(p.get('firstName') or {}).get('default', '')} {(p.get('lastName') or {}).get('default', '')}".strip(),
                    "first_name": (p.get("firstName") or {}).get("default", ""),
                    "last_name":  (p.get("lastName") or {}).get("default", ""),
                    "team":       p.get("teamAbbrev", ""),
                    "position":   "G",
                    "number":     p.get("sweaterNumber"),
                    "headshot":   p.get("headshot", ""),
                    "value":      p.get("value"),
                })
            return cat, leaders
        except Exception:  # noqa: BLE001
            return cat, []

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        pairs = await asyncio.gather(*[_fetch_cat(client, cat) for cat in CATEGORIES])

    payload = {"categories": dict(pairs)}
    _cache_set("goalie-stats", payload)
    return payload


# ===========================================================================
# Phase 2 — Player Rating Models
# ===========================================================================


@app.get("/phase2/models")
async def phase2_models() -> dict:
    """Phase 2 model build status. All statuses reflect real disk state."""
    def _has(subdir: str, glob: str) -> bool:
        d = _GRETZKY_DATA_DIR / subdir
        return bool(list(d.glob(glob))) if d.exists() else False

    models = [
        {"id": "xg_finishing",    "name": "xG Player Finishing",        "ref": "2.1",  "status": "ok" if _has("xg_finishing", "xg_finishing_*.parquet") else "not_built", "desc": "Goals minus expected goals per player; isolates finishing talent from shot volume", "feeds": "WAR · Rust engine"},
        {"id": "rapm",            "name": "RAPM",                        "ref": "2.2",  "status": "ok" if _has("rapm", "rapm_*.parquet") else "not_built", "desc": "Ridge-regularized on-ice goal differential; separates player impact from teammates", "feeds": "WAR · QoT/QoC"},
        {"id": "matchup_matrix",  "name": "Matchup Efficiency Matrix",   "ref": "2.3",  "status": "ok" if _has("matchup", "matchup_model.pkl") else "not_built", "desc": "Predicted xGF% for every player pair at 5v5; powers QoT and QoC leaderboards", "feeds": "Rust engine · Lineup optimizer"},
        {"id": "line_chemistry",  "name": "Line Chemistry",              "ref": "2.4",  "status": "ok" if _has("chemistry", "chemistry_model.pkl") else "not_built", "desc": "Linemate pair xGF% above individual baselines; detects genuine synergy vs. noise", "feeds": "Lineup optimizer"},
        {"id": "goalie_model",    "name": "Goalie Model (XGBoost)",      "ref": "2.5",  "status": "ok" if _has("goalie_ratings", "goalie_model.pkl") else "not_built", "desc": "GSAx and HDSV% ratings from MoneyPuck shot-quality data via gradient boosting", "feeds": "Rust engine · Goalie fatigue"},
        {"id": "goalie_fatigue",  "name": "Goalie Fatigue Sub-model",    "ref": "2.6",  "status": "ok" if _has("goalie_ratings", "goalie_ratings_*.parquet") else "not_built", "desc": "Back-to-back and workload penalty on goalie expected save percentage", "feeds": "Rust engine"},
        {"id": "special_teams",   "name": "Special Teams Ratings",       "ref": "2.7",  "status": "ok" if _has("special_teams", "special_teams_*.parquet") else "not_built", "desc": "Per-player PP and PK xGF/60 isolated from team context; direct WAR input", "feeds": "WAR"},
        {"id": "bayesian_rating", "name": "Bayesian Rating System",      "ref": "2.9",  "status": "ok" if _has("bayes_ratings", "player_ratings_*.parquet") else "not_built", "desc": "Prior + likelihood update using full posterior; shrinks small samples toward league mean", "feeds": "WAR · In-season blend"},
        {"id": "archetypes",      "name": "Player Archetype Clustering", "ref": "2.11", "status": "ok" if _has("archetypes", "archetype_assignments_*.parquet") else "not_built", "desc": "K-means clusters players by role (power forward, shutdown D, sniper, etc.) for lineup logic", "feeds": "Rust engine · Lineup optimizer"},
        {"id": "ewma_form",       "name": "EWMA Form Metric",            "ref": "2.13", "status": "ok" if _has("ewma", "ewma_form_*.parquet") else "not_built", "desc": "Exponentially weighted rolling xGF/60; recent games weighted 3× more than 20-game baseline", "feeds": "In-season blend · Regime detector"},
        {"id": "regime_detector", "name": "Regime Change Detector",      "ref": "2.14", "status": "ok" if _has("regime", "regime_alerts_*.parquet") else "not_built", "desc": "CUSUM-based breakpoint detection on EWMA; flags when a player's true talent has shifted", "feeds": "In-season blend"},
        {"id": "war",             "name": "WAR Unified Rating",          "ref": "2.25", "status": "ok" if _has("war", "war_*.parquet") else "not_built", "desc": "RAPM + finishing + special teams rolled into wins above replacement; drives contract efficiency", "feeds": "Dashboard · Polymarket edge"},
        {"id": "former_team_boost","name": "Former Team Boost",           "ref": "2.27", "status": "ok" if _has("former_team_boost", "former_team_boost_*.parquet") else "not_built", "desc": "Motivational multiplier when a player faces their former team; calibrated by departure type + decay", "feeds": "Rust engine"},
        {"id": "hot_hand",        "name": "Hot Hand Signal (5g burst)",   "ref": "2.28", "status": "ok" if _has("hot_hand", "hot_hand_summary_*.parquet") else "not_built", "desc": "5-game rolling goals-vs-xG z-score; Bayesian-shrunk by career GP. Complements EWMA (2.13)", "feeds": "Rust engine shot generation"},
        {"id": "clutch_index",    "name": "Clutch Index (WPA)",           "ref": "2.29", "status": "ok" if _has("clutch_index", "clutch_index_*.parquet") else "not_built", "desc": "Win probability added per goal/assist vs. expectation; 3-season weighted avg with Bayesian shrinkage", "feeds": "Rust engine late-game · Player card"},
    ]
    built = sum(1 for m in models if m["status"] == "ok")
    return {"models": models, "built": built, "total": len(models)}


@app.get("/phase2/player")
async def phase2_player(
    name: str = Query(..., description="Player name"),
) -> dict:
    """Player rating lookup. Returns real xG Finishing data once model is trained."""
    import polars as pl

    xg_dir = _GRETZKY_DATA_DIR / "xg_finishing"
    parquets = sorted(xg_dir.glob("xg_finishing_*.parquet")) if xg_dir.exists() else []

    if not parquets:
        return {
            "not_found": True,
            "reason": "models_not_built",
            "player_name": name,
            "message": "Run scripts/train_xg_model.py to train the xG model first.",
        }

    df = pl.read_parquet(parquets[-1])
    name_lower = name.strip().lower()
    row = df.filter(pl.col("shooter_name").str.to_lowercase() == name_lower)

    if row.is_empty():
        # Check if this is a goalie — fall back to goalie_stats parquet
        goalie_dir = _module_dir("goalie_stats")
        goalie_parquets = sorted(goalie_dir.glob("goalie_stats_*.parquet")) if goalie_dir.exists() else []
        if goalie_parquets:
            gdf = pl.read_parquet(goalie_parquets[-1])
            grow = gdf.filter(pl.col("player_name").str.to_lowercase() == name_lower)
            if not grow.is_empty():
                # Use all-situations row if available, else first row
                sit_row = grow.filter(pl.col("situation") == "all") if "situation" in grow.columns else grow
                gr = (sit_row if not sit_row.is_empty() else grow).to_dicts()[0]
                return {
                    "is_goalie": True,
                    "not_found": False,
                    "player_name": gr.get("player_name"),
                    "player_id":   int(gr["player_id"]) if gr.get("player_id") is not None else None,
                    "team":        gr.get("team"),
                    "season":      gr.get("season"),
                    "position":    "G",
                    "games_played": gr.get("games_played"),
                    "shots":       gr.get("shots"),
                    "saves":       gr.get("saves"),
                    "goals_against": gr.get("goals_against"),
                    "sv_pct":      gr.get("sv_pct"),
                    "xga":         gr.get("xga"),
                    "gsax":        gr.get("gsax"),
                    "hd_shots":    gr.get("hd_shots"),
                    "hd_saves":    gr.get("hd_saves"),
                    "hd_goals":    gr.get("hd_goals"),
                    "hdsv_pct":    gr.get("hdsv_pct"),
                    "mdsv_pct":    gr.get("mdsv_pct"),
                    "ldsv_pct":    gr.get("ldsv_pct"),
                }
        return {
            "not_found": True,
            "reason": "player_not_in_model",
            "player_name": name,
            "message": f"'{name}' not found in xG model (may have < 20 shots this season).",
        }

    r = row.to_dicts()[0]

    # Try to join CDR
    cdr_val: float | None = None
    try:
        from models.defensive_rating import read_cdr, lookup_player as _lookup_cdr
        cdr_df = read_cdr(_GRETZKY_DATA_DIR, season=r.get("season"))
        if cdr_df is not None:
            cdr_row = _lookup_cdr(cdr_df, r["shooter_name"])
            if not cdr_row.is_empty():
                raw = cdr_row["cdr"][0]
                cdr_val = round(float(raw), 3) if raw is not None else None
    except Exception:
        pass

    # Try to join RAPM
    rapm_ev_off: float | None = None
    rapm_ev_def: float | None = None
    rapm_xga_60: float | None = None
    try:
        rapm_dir = _GRETZKY_DATA_DIR / "rapm"
        season_val = r.get("season")
        rapm_path = rapm_dir / f"rapm_{season_val}.parquet" if season_val else None
        if rapm_path and rapm_path.exists():
            rapm_df = pl.read_parquet(rapm_path)
            shooter_id = r.get("shooter_id")
            if shooter_id is not None:
                rapm_row = rapm_df.filter(pl.col("player_id") == int(shooter_id))
                if not rapm_row.is_empty():
                    rapm_ev_off = round(float(rapm_row["rapm_ev_off"][0]), 3)
                    rapm_ev_def = round(float(rapm_row["rapm_ev_def"][0]), 3)
                    if "xga_60" in rapm_df.columns:
                        rapm_xga_60 = round(float(rapm_row["xga_60"][0]), 2)
    except Exception:
        pass

    # Try to look up position + shoots from shots parquet (ID first, name fallback)
    position: str | None = None
    shoots: str | None = None
    try:
        shots_dir = _GRETZKY_DATA_DIR / "shots"
        shot_files = sorted(shots_dir.glob("*.parquet")) if shots_dir.exists() else []
        if shot_files:
            shots_df = pl.read_parquet(shot_files[-1], columns=["shooter_id", "shooter_name", "player_position", "shooter_hand"])
            match = pl.DataFrame()
            # Try by numeric ID first
            if r.get("shooter_id") is not None:
                match = shots_df.filter(pl.col("shooter_id") == int(r["shooter_id"]))
            # Fall back to name match
            if match.is_empty() and r.get("shooter_name"):
                match = shots_df.filter(pl.col("shooter_name") == r["shooter_name"])
            if not match.is_empty():
                pos_val = match["player_position"][0]
                hand_val = match["shooter_hand"][0]
                if pos_val:
                    position = str(pos_val)
                if hand_val:
                    shoots = str(hand_val)
    except Exception:
        pass

    player_id_val = int(r["shooter_id"]) if r.get("shooter_id") is not None else None
    season_val2   = r.get("season")

    # Try to join WAR
    war_val: float | None = None
    gar_val: float | None = None
    contract_efficiency_val: float | None = None
    try:
        war_dir = _GRETZKY_DATA_DIR / "war"
        war_parquets = sorted(war_dir.glob("war_*.parquet")) if war_dir.exists() else []
        if war_parquets and player_id_val is not None:
            war_df = pl.read_parquet(war_parquets[-1])
            war_row = war_df.filter(pl.col("player_id") == player_id_val)
            if not war_row.is_empty():
                w = war_row.to_dicts()[0]
                war_val = round(float(w["war"]), 2) if w.get("war") is not None else None
                gar_val = round(float(w["gar_total"]), 2) if w.get("gar_total") is not None else None
                ce = w.get("contract_efficiency")
                contract_efficiency_val = round(float(ce), 2) if ce is not None else None
    except Exception:
        pass

    # Try to join archetype
    archetype_id: int | None = None
    archetype_name: str | None = None
    try:
        arch_dir = _GRETZKY_DATA_DIR / "archetypes"
        arch_parquets = sorted(arch_dir.glob("archetype_assignments_*.parquet")) if arch_dir.exists() else []
        if arch_parquets and player_id_val is not None:
            arch_df = pl.read_parquet(arch_parquets[-1])
            arch_row = arch_df.filter(pl.col("player_id") == player_id_val)
            if not arch_row.is_empty():
                a = arch_row.to_dicts()[0]
                archetype_id   = int(a["cluster_id"]) if a.get("cluster_id") is not None else None
                archetype_name = str(a.get("archetype") or a.get("archetype_name") or "") or None
    except Exception:
        pass

    # Try to join EWMA form
    ewma_xgf60:   float | None = None
    ewma_form_flag: str | None = None
    ewma_games:     int | None = None
    try:
        ewma_dir = _GRETZKY_DATA_DIR / "ewma"
        ewma_parquets = sorted(ewma_dir.glob("ewma_form_*.parquet")) if ewma_dir.exists() else []
        if ewma_parquets and player_id_val is not None:
            ewma_df = pl.read_parquet(ewma_parquets[-1])
            ewma_row = ewma_df.filter(pl.col("player_id") == player_id_val)
            if not ewma_row.is_empty():
                e = ewma_row.to_dicts()[0]
                raw_ewma = e.get("ewma_xgf60") or e.get("current_ewma") or e.get("xgf_per60")
                ewma_xgf60    = round(float(raw_ewma), 3) if raw_ewma is not None else None
                ewma_form_flag = str(e["form_flag"]) if e.get("form_flag") else None
                ewma_games    = int(e["games_processed"]) if e.get("games_processed") is not None else None
    except Exception:
        pass

    # Try to join hot-hand score (Feature 2.28)
    hot_hand_score:  float | None = None
    hot_hand_goals5: float | None = None
    hot_hand_xg5:    float | None = None
    try:
        hh_dir = _GRETZKY_DATA_DIR / "hot_hand"
        hh_parquets = sorted(hh_dir.glob("hot_hand_summary_*.parquet")) if hh_dir.exists() else []
        if hh_parquets and player_id_val is not None:
            hh_df = pl.read_parquet(hh_parquets[-1])
            hh_row = hh_df.filter(pl.col("player_id") == player_id_val)
            if not hh_row.is_empty():
                hh = hh_row.to_dicts()[0]
                hot_hand_score  = round(float(hh["hot_hand_score"]), 3) if hh.get("hot_hand_score") is not None else None
                hot_hand_goals5 = round(float(hh["goals_5g"]), 2)       if hh.get("goals_5g")       is not None else None
                hot_hand_xg5    = round(float(hh["xg_5g"]), 2)          if hh.get("xg_5g")          is not None else None
    except Exception:
        pass

    # Try to join clutch index (Feature 2.29)
    clutch_index:       float | None = None
    clutch_wpa_per60:   float | None = None
    try:
        ci_dir = _GRETZKY_DATA_DIR / "clutch_index"
        ci_parquets = sorted(ci_dir.glob("clutch_index_[0-9]*.parquet")) if ci_dir.exists() else []
        if ci_parquets and player_id_val is not None:
            ci_df = pl.read_parquet(ci_parquets[-1])
            ci_row = ci_df.filter(pl.col("player_id") == player_id_val)
            if not ci_row.is_empty():
                ci = ci_row.to_dicts()[0]
                clutch_index     = round(float(ci["clutch_index_shrunk"]), 4) if ci.get("clutch_index_shrunk") is not None else None
                clutch_wpa_per60 = round(float(ci["actual_wpa_per60"]),   4) if ci.get("actual_wpa_per60")   is not None else None
    except Exception:
        pass

    return {
        "player_name":          r["shooter_name"],
        "player_id":            player_id_val,
        "team":                 r.get("team"),
        "season":               season_val2,
        "shots":                r.get("shots"),
        "goals":                r.get("goals"),
        "xg_sum":               round(r["xg_sum"], 3) if r.get("xg_sum") is not None else None,
        "finishing":            round(r["finishing"], 3) if r.get("finishing") is not None else None,
        "finishing_per60":      round(r["finishing_per60"], 3) if r.get("finishing_per60") is not None else None,
        "model_version":        r.get("model_version"),
        "cdr":                  cdr_val,
        "rapm_ev_off":          rapm_ev_off,
        "rapm_ev_def":          rapm_ev_def,
        "rapm_xga_60":          rapm_xga_60,
        "position":             position,
        "shoots":               shoots,
        "war":                  war_val,
        "gar":                  gar_val,
        "contract_efficiency":  contract_efficiency_val,
        "archetype_id":         archetype_id,
        "archetype_name":       archetype_name,
        "ewma_xgf60":           ewma_xgf60,
        "ewma_form_flag":       ewma_form_flag,
        "ewma_games":           ewma_games,
        "hot_hand_score":       hot_hand_score,
        "hot_hand_goals5":      hot_hand_goals5,
        "hot_hand_xg5":         hot_hand_xg5,
        "clutch_index":         clutch_index,
        "clutch_wpa_per60":     clutch_wpa_per60,
    }


def _build_name_lookup() -> dict[int, str]:
    """Build player_id → player_name lookup from all available parquets."""
    import polars as pl
    lookup: dict[int, str] = {}
    # RAPM (most complete for skaters)
    for p in sorted((_GRETZKY_DATA_DIR / "rapm").glob("rapm_*.parquet")):
        try:
            df = pl.read_parquet(p, columns=["player_id", "player_name"])
            for r in df.to_dicts():
                pid, name = r.get("player_id"), r.get("player_name") or ""
                if pid and name and not name.startswith("player_"):
                    lookup[int(pid)] = name
        except Exception:
            pass
    # Goalie stats (covers goalies missing from RAPM)
    goalie_dir = _module_dir("goalie_stats")
    for p in sorted(goalie_dir.glob("goalie_stats_*.parquet")):
        try:
            df = pl.read_parquet(p, columns=["player_id", "player_name"])
            for r in df.to_dicts():
                pid, name = r.get("player_id"), r.get("player_name") or ""
                if pid and name and int(pid) not in lookup:
                    lookup[int(pid)] = name
        except Exception:
            pass
    # Goalie ratings (another goalie source)
    goalie_ratings_dir = _GRETZKY_DATA_DIR / "goalie_ratings"
    for p in sorted(goalie_ratings_dir.glob("goalie_ratings_*.parquet")):
        try:
            df = pl.read_parquet(p, columns=["player_id", "player_name"])
            for r in df.to_dicts():
                pid, name = r.get("player_id"), r.get("player_name") or ""
                if pid and name and int(pid) not in lookup:
                    lookup[int(pid)] = name
        except Exception:
            pass
    return lookup


_FRANCHISE_ABBR: dict[int, str] = {
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT", 6: "BOS",
    7: "BUF", 8: "MTL", 9: "OTT", 10: "TOR", 12: "CAR", 13: "FLA",
    14: "TBL", 15: "WSH", 16: "CHI", 17: "DET", 18: "NSH", 19: "STL",
    20: "CGY", 21: "COL", 22: "EDM", 23: "VAN", 24: "ANA", 25: "DAL",
    26: "LAK", 28: "SJS", 29: "CBJ", 30: "MIN", 52: "WPG", 53: "UTH",
    54: "VGK", 55: "SEA", 59: "UTH", 68: "UTH",
}


def _abbr(val) -> str | None:
    if val is None:
        return None
    try:
        return _FRANCHISE_ABBR.get(int(val), str(val))
    except (ValueError, TypeError):
        return str(val)


@app.get("/phase2/war-leaderboard")
async def phase2_war_leaderboard(limit: int = 10, player_id: int | None = None) -> dict:
    """Top WAR players (min 200 min EV TOI). Optionally includes selected player rank."""
    import polars as pl
    war_dir = _GRETZKY_DATA_DIR / "war"
    parquets = sorted(war_dir.glob("war_*.parquet")) if war_dir.exists() else []
    if not parquets:
        return {"players": [], "built": False}
    try:
        df = pl.read_parquet(parquets[-1])
        if "war" not in df.columns:
            return {"players": [], "built": False}
        name_lut = _build_name_lookup()

        # Filter to qualified players: non-null WAR + minimum EV TOI
        MIN_TOI = 200.0
        ranked = (
            df.filter(pl.col("war").is_not_null())
            .pipe(lambda d: d.filter(pl.col("toi_ev") >= MIN_TOI) if "toi_ev" in d.columns else d)
            .sort("war", descending=True)
            .with_row_index("rank")  # 0-based rank in full qualified list
        )

        def _make_row(r: dict, rank_override: int | None = None) -> dict:
            pid = r.get("player_id")
            raw_name = r.get("player_name") or ""
            if (not raw_name or raw_name.startswith("player_")) and pid:
                raw_name = name_lut.get(int(pid), raw_name)
            return {
                "player_id":   pid,
                "player_name": raw_name or f"id_{pid}",
                "team":        _abbr(r.get("team")),
                "war":         round(float(r["war"]), 2),
                "gar":         round(float(r["gar_total"]), 2) if r.get("gar_total") is not None else None,
                "rank":        (rank_override if rank_override is not None else r.get("rank", 0)) + 1,
            }

        top = ranked.head(limit)
        players = [_make_row(r) for r in top.to_dicts()]

        # Find selected player's rank if not already in top N
        selected = None
        if player_id is not None:
            sel_rows = ranked.filter(pl.col("player_id") == player_id)
            if not sel_rows.is_empty():
                sel = sel_rows.to_dicts()[0]
                sel_rank = sel.get("rank", 0) + 1  # 1-based
                already_shown = any(p["player_id"] == player_id for p in players)
                if not already_shown:
                    selected = _make_row(sel)

        return {"players": players, "selected": selected, "built": True, "min_toi": MIN_TOI}
    except Exception:
        return {"players": [], "selected": None, "built": False}


@app.get("/phase2/ewma-movers")
async def phase2_ewma_movers(limit: int = 6, player_id: int | None = None) -> dict:
    """Top EWMA form movers ranked by delta vs league mean. Includes team from RAPM."""
    import polars as pl
    ewma_dir = _GRETZKY_DATA_DIR / "ewma"
    parquets = sorted(ewma_dir.glob("ewma_form_*.parquet")) if ewma_dir.exists() else []
    if not parquets:
        return {"rising": [], "falling": [], "built": False}
    try:
        df = pl.read_parquet(parquets[-1])
        ewma_col = "ewma_xgf60" if "ewma_xgf60" in df.columns else "current_ewma"
        if ewma_col not in df.columns:
            return {"rising": [], "falling": [], "built": False}

        # Collapse per-game rows → one row per player (latest game), min 20 games
        MIN_GAMES = 20
        if "games_played" in df.columns:
            # ewma_form parquets already have one row per player with games_played
            df = (
                df.filter(pl.col("games_played") >= MIN_GAMES)
                .with_columns(pl.col("games_played").alias("_n_games"))
            )
        elif "game_number" in df.columns or "game_id" in df.columns:
            sort_col = "game_number" if "game_number" in df.columns else "game_id"
            game_counts = df.group_by("player_id").len().rename({"len": "_n_games"})
            df = (
                df.sort(sort_col, descending=True)
                .group_by("player_id")
                .first()
                .join(game_counts, on="player_id")
                .filter(pl.col("_n_games") >= MIN_GAMES)
            )

        # Build player_id → (name, team_abbr) — shots first (covers all players
        # with real team abbr), RAPM as fallback for name only.
        name_team_lut: dict[int, tuple[str, str | None]] = {}

        # 1. Shots parquets: shooter_id → (shooter_name, shooting_team)
        shots_dir = _GRETZKY_DATA_DIR / "shots"
        for sp in sorted(shots_dir.glob("*.parquet")) if shots_dir.exists() else []:
            try:
                sdf = pl.read_parquet(sp, columns=["shooter_id", "shooter_name", "shooting_team"])
                for r in sdf.filter(
                    pl.col("shooter_id").is_not_null() & pl.col("shooter_name").is_not_null()
                ).unique(subset=["shooter_id"]).to_dicts():
                    pid = r.get("shooter_id")
                    name = (r.get("shooter_name") or "").strip()
                    team = r.get("shooting_team")
                    if pid and name:
                        name_team_lut[int(pid)] = (name, team or None)
            except Exception:
                pass

        # 2. RAPM parquets: fill any gaps (name only; team is numeric → _abbr)
        rapm_dir = _GRETZKY_DATA_DIR / "rapm"
        for rp in sorted(rapm_dir.glob("rapm_*.parquet")):
            try:
                rdf = pl.read_parquet(rp, columns=["player_id", "player_name", "team"])
                for r in rdf.to_dicts():
                    pid = r.get("player_id")
                    name = (r.get("player_name") or "").strip()
                    if pid and name and not name.startswith("player_") and int(pid) not in name_team_lut:
                        name_team_lut[int(pid)] = (name, _abbr(r.get("team")))
            except Exception:
                pass

        def _name_team(r: dict) -> tuple[str, str | None]:
            pid = r.get("player_id")
            if pid and int(pid) in name_team_lut:
                return name_team_lut[int(pid)]
            return (f"id_{pid}", None)

        # Compute delta vs league mean (always, regardless of form_flag)
        valid = df.filter(pl.col(ewma_col).is_not_null())
        league_mean = valid[ewma_col].mean() or 0.0
        valid = valid.with_columns(
            (pl.col(ewma_col) - league_mean).alias("_delta")
        )

        def _make_rows(frame: pl.DataFrame, flag: str) -> list[dict]:
            result = []
            for r in frame.to_dicts():
                val = r.get(ewma_col)
                delta = r.get("_delta")
                n_games = r.get("_n_games")
                name, team = _name_team(r)
                result.append({
                    "player_id":   r.get("player_id"),
                    "player_name": name,
                    "team":        team,
                    "ewma_xgf60":  round(float(val), 2) if val is not None else None,
                    "delta":       round(float(delta), 2) if delta is not None else None,
                    "league_mean": round(league_mean, 2),
                    "n_games":     int(n_games) if n_games is not None else None,
                    "form_flag":   flag,
                })
            return result

        rising  = valid.sort("_delta", descending=True).head(limit)
        falling = valid.sort("_delta").head(limit)
        rising_rows  = _make_rows(rising, "rising")
        falling_rows = _make_rows(falling, "falling")

        # Find selected player if not already in top N rising/falling
        selected = None
        if player_id is not None:
            already_shown = (
                any(p.get("player_id") == player_id for p in rising_rows) or
                any(p.get("player_id") == player_id for p in falling_rows)
            )
            if not already_shown:
                sel_ranked = (
                    valid.sort("_delta", descending=True)
                    .with_row_index("_rank")
                    .filter(pl.col("player_id") == player_id)
                )
                if not sel_ranked.is_empty():
                    sel_r = sel_ranked.to_dicts()[0]
                    rank = int(sel_r.get("_rank", 0)) + 1
                    name, team = _name_team(sel_r)
                    val = sel_r.get(ewma_col)
                    delta_val = sel_r.get("_delta")
                    n_games = sel_r.get("_n_games")
                    selected = {
                        "player_id":   player_id,
                        "player_name": name,
                        "team":        team,
                        "ewma_xgf60":  round(float(val), 2) if val is not None else None,
                        "delta":       round(float(delta_val), 2) if delta_val is not None else None,
                        "league_mean": round(league_mean, 2),
                        "n_games":     int(n_games) if n_games is not None else None,
                        "form_flag":   "rising" if (delta_val or 0) >= 0 else "falling",
                        "rank":        rank,
                    }

        return {
            "rising":       rising_rows,
            "falling":      falling_rows,
            "league_mean":  round(league_mean, 2),
            "built":        True,
            "selected":     selected,
        }
    except Exception:
        return {"rising": [], "falling": [], "built": False, "selected": None}


@app.get("/phase2/matchup-explorer")
async def phase2_matchup_explorer(limit: int = 10, player_id: int | None = None) -> dict:
    """QoT/QoC leaders and top player-pair matchup predictions."""
    import polars as pl
    qot_dir = _GRETZKY_DATA_DIR / "qot_qoc"
    mp_dir  = _GRETZKY_DATA_DIR / "matchup_preds"
    qot_parquets = sorted(qot_dir.glob("qot_qoc_*.parquet")) if qot_dir.exists() else []
    mp_parquets  = sorted(mp_dir.glob("matchup_preds_*.parquet")) if mp_dir.exists() else []
    if not qot_parquets and not mp_parquets:
        return {"qot": [], "qoc": [], "top_pairs": [], "built": False}

    MIN_GP = 20  # minimum games played to appear in QoT/QoC leaderboards

    try:
        # QoT / QoC leaders
        qot_leaders: list[dict] = []
        qoc_leaders: list[dict] = []
        if qot_parquets:
            qdf = pl.read_parquet(qot_parquets[-1])
            # Filter to qualified players
            if "gp" in qdf.columns:
                qdf = qdf.filter(pl.col("gp") >= MIN_GP)
            toi_col = "toi_ev" if "toi_ev" in qdf.columns else None
            if "qot" in qdf.columns:
                for r in qdf.sort("qot", descending=True).head(limit).to_dicts():
                    qot_leaders.append({
                        "player_id":   r.get("player_id"),
                        "player_name": r.get("player_name") or f"id_{r.get('player_id')}",
                        "team":        _abbr(r.get("team")),
                        "qot":         round(float(r["qot"]), 3),
                        "qoc":         round(float(r["qoc"]), 3) if r.get("qoc") is not None else None,
                        "gp":          r.get("gp"),
                        "toi_ev":      round(float(r[toi_col]), 1) if toi_col and r.get(toi_col) is not None else None,
                    })
            if "qoc" in qdf.columns:
                for r in qdf.sort("qoc", descending=True).head(limit).to_dicts():
                    qoc_leaders.append({
                        "player_id":   r.get("player_id"),
                        "player_name": r.get("player_name") or f"id_{r.get('player_id')}",
                        "team":        _abbr(r.get("team")),
                        "qot":         round(float(r["qot"]), 3) if r.get("qot") is not None else None,
                        "qoc":         round(float(r["qoc"]), 3),
                        "gp":          r.get("gp"),
                        "toi_ev":      round(float(r[toi_col]), 1) if toi_col and r.get(toi_col) is not None else None,
                    })

        # Top player-pair matchups — highest predicted xGF% vs each other (min 1 game)
        top_pairs: list[dict] = []
        if mp_parquets:
            # Build player_id → (name, team_abbr) from RAPM
            name_team_lut: dict[int, tuple[str, str | None]] = {}
            _rapm_dir = _GRETZKY_DATA_DIR / "rapm"
            for p in sorted(_rapm_dir.glob("rapm_*.parquet")):
                try:
                    rdf = pl.read_parquet(p, columns=["player_id", "player_name", "team"])
                    for row in rdf.to_dicts():
                        pid, nm, tm = row.get("player_id"), row.get("player_name") or "", row.get("team")
                        if pid and nm and not nm.startswith("player_"):
                            name_team_lut[int(pid)] = (nm, _abbr(tm))
                except Exception:
                    pass

            def _player_info(pid) -> dict:
                if pid is None:
                    return {"name": "Unknown", "team": None}
                name, team = name_team_lut.get(int(pid), (f"id_{pid}", None))
                return {"name": name, "team": team}

            mdf = pl.read_parquet(mp_parquets[-1])
            if "model_xgf_pct" in mdf.columns:
                pairs = (
                    mdf.filter(pl.col("games_together").is_not_null() & (pl.col("games_together") >= 1))
                    .sort("model_xgf_pct", descending=True)
                    .head(limit * 3)  # fetch more; filter unknowns below
                )
                for r in pairs.to_dicts():
                    aid, bid = r.get("player_a_id"), r.get("player_b_id")
                    a, b = _player_info(aid), _player_info(bid)
                    # Skip pairs where both names are fallback IDs
                    if a["name"].startswith("id_") and b["name"].startswith("id_"):
                        continue
                    games = r.get("games_together") or 0
                    co_toi = r.get("co_toi_ev")
                    top_pairs.append({
                        "player_a_id":    aid,
                        "player_a_name":  a["name"],
                        "player_a_team":  a["team"],
                        "player_b_id":    bid,
                        "player_b_name":  b["name"],
                        "player_b_team":  b["team"],
                        "xgf_pct":        round(float(r["model_xgf_pct"]), 3),
                        "hist_xgf_pct":   round(float(r["hist_xgf_pct"]), 3) if r.get("hist_xgf_pct") is not None else None,
                        "games":          games,
                        "co_toi_ev":      round(float(co_toi), 1) if co_toi is not None else None,
                        "avg_toi_per_gp": round(float(co_toi) / games, 1) if co_toi and games else None,
                    })
                    if len(top_pairs) >= limit:
                        break

        # ── Selected player QoT/QoC rank + best pairs ────────────────────────
        selected_player: dict | None = None
        selected_pairs: list[dict]   = []

        if player_id is not None and qot_parquets:
            # qdf is the filtered (>= MIN_GP) dataframe from above
            try:
                sel_rows = qdf.filter(pl.col("player_id") == player_id)
                if not sel_rows.is_empty():
                    r = sel_rows.to_dicts()[0]
                    qot_rank: int | None = None
                    qoc_rank: int | None = None
                    if "qot" in qdf.columns:
                        qot_sorted = qdf.sort("qot", descending=True).with_row_index("_r")
                        qrs = qot_sorted.filter(pl.col("player_id") == player_id)
                        if not qrs.is_empty():
                            qot_rank = int(qrs.to_dicts()[0]["_r"]) + 1
                    if "qoc" in qdf.columns:
                        qoc_sorted = qdf.sort("qoc", descending=True).with_row_index("_r")
                        qrs = qoc_sorted.filter(pl.col("player_id") == player_id)
                        if not qrs.is_empty():
                            qoc_rank = int(qrs.to_dicts()[0]["_r"]) + 1
                    selected_player = {
                        "player_id":   player_id,
                        "player_name": r.get("player_name") or f"id_{player_id}",
                        "team":        _abbr(r.get("team")),
                        "qot":         round(float(r["qot"]), 3) if r.get("qot") is not None else None,
                        "qoc":         round(float(r["qoc"]), 3) if r.get("qoc") is not None else None,
                        "qot_rank":    qot_rank,
                        "qoc_rank":    qoc_rank,
                        "gp":          r.get("gp"),
                        "toi_ev":      round(float(r[toi_col]), 1) if toi_col and r.get(toi_col) is not None else None,
                    }
            except Exception:
                pass

        if player_id is not None and mp_parquets:
            try:
                # Use existing name_team_lut and mdf if available
                sp_mdf = pl.read_parquet(mp_parquets[-1])
                if "model_xgf_pct" in sp_mdf.columns:
                    sp_pairs = (
                        sp_mdf.filter(
                            ((pl.col("player_a_id") == player_id) | (pl.col("player_b_id") == player_id)) &
                            pl.col("games_together").is_not_null() &
                            (pl.col("games_together") >= 1)
                        )
                        .sort("model_xgf_pct", descending=True)
                        .head(6)
                    )
                    # Rebuild minimal name lut if top_pairs block didn't run
                    sp_lut: dict[int, tuple[str, str | None]] = {}
                    _rapm_dir2 = _GRETZKY_DATA_DIR / "rapm"
                    for p2 in sorted(_rapm_dir2.glob("rapm_*.parquet")):
                        try:
                            rdf2 = pl.read_parquet(p2, columns=["player_id", "player_name", "team"])
                            for row2 in rdf2.to_dicts():
                                pid2, nm2, tm2 = row2.get("player_id"), row2.get("player_name") or "", row2.get("team")
                                if pid2 and nm2 and not nm2.startswith("player_"):
                                    sp_lut[int(pid2)] = (nm2, _abbr(tm2))
                        except Exception:
                            pass

                    def _sp_info(pid) -> dict:
                        if pid is None:
                            return {"name": "Unknown", "team": None}
                        nm, tm = sp_lut.get(int(pid), (f"id_{pid}", None))
                        return {"name": nm, "team": tm}

                    for r in sp_pairs.to_dicts():
                        aid, bid = r.get("player_a_id"), r.get("player_b_id")
                        other_id = bid if aid == player_id else aid
                        other = _sp_info(other_id)
                        if other["name"].startswith("id_"):
                            continue
                        games = r.get("games_together") or 0
                        co_toi = r.get("co_toi_ev")
                        selected_pairs.append({
                            "player_id":   other_id,
                            "player_name": other["name"],
                            "team":        other["team"],
                            "xgf_pct":     round(float(r["model_xgf_pct"]), 3),
                            "hist_xgf_pct": round(float(r["hist_xgf_pct"]), 3) if r.get("hist_xgf_pct") is not None else None,
                            "games":       games,
                            "co_toi_ev":   round(float(co_toi), 1) if co_toi is not None else None,
                        })
            except Exception:
                pass

        return {
            "qot":             qot_leaders,
            "qoc":             qoc_leaders,
            "top_pairs":       top_pairs,
            "selected_player": selected_player,
            "selected_pairs":  selected_pairs,
            "built":           True,
        }
    except Exception:
        return {"qot": [], "qoc": [], "top_pairs": [], "selected_player": None, "selected_pairs": [], "built": False}


@app.get("/phase2/players")
async def phase2_players() -> dict:
    """Known player names for client-side autocomplete in the Phase 2 rating lookup.

    Sources skaters from xg_finishing parquet and goalies from goalie_stats parquet.
    """
    import polars as pl

    players: dict[str, dict] = {}

    # Skaters from xg_finishing
    xg_dir = _GRETZKY_DATA_DIR / "xg_finishing"
    xg_parquets = sorted(xg_dir.glob("xg_finishing_*.parquet")) if xg_dir.exists() else []
    if xg_parquets:
        df = pl.read_parquet(xg_parquets[-1])
        if "shooter_name" in df.columns:
            cols = ["shooter_name"] + (["team"] if "team" in df.columns else [])
            pos_map = _shots_name_position_map()
            for r in df.select(cols).drop_nulls(subset=["shooter_name"]).unique(subset=["shooter_name"]).to_dicts():
                name = r["shooter_name"]
                players[name] = {"name": name, "team": r.get("team") or "", "position": pos_map.get(name, "")}

    # Goalies from goalie_stats
    goalie_dir = _module_dir("goalie_stats")
    goalie_parquets = sorted(goalie_dir.glob("goalie_stats_*.parquet")) if goalie_dir.exists() else []
    if goalie_parquets:
        gdf = pl.read_parquet(goalie_parquets[-1], columns=["player_name", "team"])
        for r in gdf.drop_nulls(subset=["player_name"]).unique(subset=["player_name"]).to_dicts():
            name = r["player_name"]
            if name not in players:
                players[name] = {"name": name, "team": r.get("team") or "", "position": "G"}

    return {"players": sorted(players.values(), key=lambda p: p["name"])}


# ===========================================================================
# Game Centre — full game data for /game/{game_id}
# ===========================================================================

_GC_PERIOD_LABELS: dict[int, str] = {1: "1ST", 2: "2ND", 3: "3RD", 4: "OT", 5: "SO"}


def _gc_period_label(num: int, period_type: str) -> str:
    if period_type == "OT":
        return "OT"
    if period_type == "SO":
        return "SO"
    return _GC_PERIOD_LABELS.get(num, f"P{num}") if num else ""


def _gc_strength(situation_code: str, event_owner_is_home: bool) -> str:
    """Derive EV/PP/PK/EN from NHL situationCode (4-char string like '1551')."""
    if not situation_code or len(situation_code) < 4:
        return "EV"
    try:
        away_g  = int(situation_code[0])
        away_sk = int(situation_code[1])
        home_sk = int(situation_code[2])
        home_g  = int(situation_code[3])
    except ValueError:
        return "EV"
    if away_g == 0 or home_g == 0:
        return "EN"
    if away_sk == home_sk:
        return "EV"
    if event_owner_is_home:
        return "PP" if home_sk > away_sk else "PK"
    return "PP" if away_sk > home_sk else "PK"


def _gc_player_name(details: dict, key: str, roster: dict[int, str]) -> str:
    pid = details.get(key)
    if pid and pid in roster:
        return roster[pid]
    return ""


def _gc_parse_skater(p: dict, xg_map: dict[str, tuple[float, float]]) -> dict:
    name = (p.get("name") or {}).get("default", "")
    toi_raw = p.get("toi") or p.get("toi", "0:00")
    fow = p.get("faceoffWins", 0) or 0
    fol = p.get("faceoffLosses", 0) or 0
    xg_sum, finishing = xg_map.get(name.lower(), (None, None))
    return {
        "name":         name,
        "number":       p.get("sweaterNumber") or 0,
        "position":     p.get("position") or "",
        "g":            p.get("goals") or 0,
        "a":            p.get("assists") or 0,
        "pts":          p.get("points") or 0,
        "plus_minus":   p.get("plusMinus") or 0,
        "shots":        p.get("shots") or 0,
        "hits":         p.get("hits") or 0,
        "blocks":       p.get("blockedShots") or 0,
        "giveaways":    p.get("giveaways") or 0,
        "takeaways":    p.get("takeaways") or 0,
        "toi":          toi_raw,
        "fow":          fow,
        "fol":          fol,
        "xg_sum":       round(xg_sum, 3) if xg_sum is not None else None,
        "finishing":    round(finishing, 3) if finishing is not None else None,
    }


def _gc_parse_goalie(p: dict) -> dict:
    name = (p.get("name") or {}).get("default", "")
    sa = (p.get("shotsAgainst") or p.get("shots") or 0)
    sv = (p.get("saves") or 0)
    ga = p.get("goalsAgainst") if p.get("goalsAgainst") is not None else (sa - sv)
    sv_pct = p.get("savePctg") or p.get("savePercentage") or (sv / sa if sa > 0 else 0.0)
    return {
        "name":    name,
        "number":  p.get("sweaterNumber") or 0,
        "sa":      sa,
        "sv":      sv,
        "ga":      ga,
        "sv_pct":  round(float(sv_pct), 4),
        "toi":     p.get("toi") or "0:00",
    }


def _line_order_from_shifts(
    shifts_data: list[dict],
    team_id: int,
    pid_positions: dict[int, str],
) -> dict[int, int]:
    """Return {player_id: sort_rank} using shift co-occurrence to find real lines.

    Algorithm:
    1. Convert each shift to (start_sec, end_sec, player_id).
    2. Sweep timeline to count pair co-occurrence seconds.
    3. Greedily assign forward lines by picking the trio with the highest
       combined pair-seconds, then repeat for the remaining forwards.
    4. Assign defensive pairs the same way (pairs of 2 instead of 3).

    Returns a rank dict so rank 0 = Line-1 player 1, rank 11 = Line-4 player 3.
    Players not in shifts get rank 999 (sent to end).
    """
    from itertools import combinations
    from collections import defaultdict

    FWD_CODES = {"C", "L", "R", "LW", "RW"}

    def _mmss(t: str, period: int) -> int:
        try:
            m, s = t.split(":")
            return (period - 1) * 1200 + int(m) * 60 + int(s)
        except Exception:
            return 0

    fwd_ids: set[int] = set()
    def_ids: set[int] = set()
    intervals: list[tuple[int, int, int]] = []

    for sh in shifts_data:
        if sh.get("teamId") != team_id:
            continue
        pid = sh.get("playerId")
        if not pid:
            continue
        pos = pid_positions.get(pid, "")
        if pos in FWD_CODES:
            fwd_ids.add(pid)
        elif pos == "D":
            def_ids.add(pid)
        else:
            continue
        start = _mmss(sh.get("startTime", "0:00"), sh.get("period", 1))
        end   = _mmss(sh.get("endTime",   "0:00"), sh.get("period", 1))
        if end > start:
            intervals.append((start, end, pid))

    # Sweep timeline → pair co-occurrence seconds
    pair_secs: dict[tuple[int, int], int] = defaultdict(int)
    events: list[tuple[int, int, int]] = []
    for start, end, pid in intervals:
        events.append((start,  1, pid))
        events.append((end,   -1, pid))
    events.sort()

    on_ice: set[int] = set()
    prev_t = 0
    for t, ev_type, pid in events:
        if t > prev_t and len(on_ice) >= 2:
            dur = t - prev_t
            for a, b in combinations(sorted(on_ice), 2):
                pair_secs[(a, b)] += dur
        if ev_type == 1:
            on_ice.add(pid)
        else:
            on_ice.discard(pid)
        prev_t = t

    def _greedy_groups(player_ids: list[int], group_size: int) -> list[list[int]]:
        """Greedy: repeatedly pick the group_size players with highest pairwise co-occurrence."""
        remaining = list(player_ids)
        groups: list[list[int]] = []
        while len(remaining) >= group_size:
            best_combo = max(
                combinations(remaining, group_size),
                key=lambda combo: sum(
                    pair_secs.get((min(a, b), max(a, b)), 0)
                    for a, b in combinations(combo, 2)
                ),
                default=tuple(remaining[:group_size]),
            )
            groups.append(list(best_combo))
            for pid in best_combo:
                remaining.remove(pid)
        if remaining:
            groups.append(remaining)
        return groups

    rank: dict[int, int] = {}
    r = 0
    for group in _greedy_groups(sorted(fwd_ids), 3):
        for pid in group:
            rank[pid] = r
            r += 1
    r = 100  # defensemen start at 100
    for group in _greedy_groups(sorted(def_ids), 2):
        for pid in group:
            rank[pid] = r
            r += 1
    return rank


def _gc_team_stats(team_raw: dict) -> dict:
    shots        = (team_raw.get("sog") or team_raw.get("shots") or 0)
    hits         = team_raw.get("hits") or 0
    blocked      = team_raw.get("blockedShots") or team_raw.get("blocked") or 0
    giveaways    = team_raw.get("giveaways") or 0
    takeaways    = team_raw.get("takeaways") or 0
    raw_fow_pct  = team_raw.get("faceoffWinningPctg") or 0.0
    pp_str       = team_raw.get("powerPlay") or team_raw.get("powerPlayConversions") or "0/0"
    pp_parts     = str(pp_str).split("/")
    pp_goals     = int(pp_parts[0]) if pp_parts[0].isdigit() else 0
    pp_opps      = int(pp_parts[1]) if len(pp_parts) > 1 and pp_parts[1].isdigit() else 0
    return {
        "shots":           shots,
        "hits":            hits,
        "blocked_shots":   blocked,
        "giveaways":       giveaways,
        "takeaways":       takeaways,
        "faceoff_pct":     round(float(raw_fow_pct), 4),
        "pp_goals":        pp_goals,
        "pp_opportunities": pp_opps,
        "xg":              None,  # filled later if parquet available
    }


@app.get("/game/{game_id}")
async def game_centre(game_id: int) -> dict:
    """Full game centre data for a single NHL game.

    Fetches the NHL landing page and play-by-play in parallel, then
    assembles scoring, penalties, team stats, per-player stats, lineup,
    event feed, and shot map data. xG finishing enrichment is attempted
    from the local parquet if available; gracefully skipped otherwise.

    game_state: FUT | PRE | LIVE | CRIT | FINAL | OFF
    shots_for_map: all shot events with coordinates normalised so every
                   shot points toward positive-x (right net).
    """
    import asyncio
    from data.nhl_client import NHLClient, NHLApiError

    _SHOT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
    _SKIP_PLAYS = {"faceoff", "stoppage", "period-start", "period-end",
                   "game-end", "delayed-penalty", "icing", "offsides"}

    try:
        async with NHLClient() as client:
            landing_result, pbp_result, boxscore_result, shifts_result = await asyncio.gather(
                client.get_landing(game_id),
                client.get_play_by_play(game_id),
                client.get_boxscore(game_id),
                client.get_shifts(game_id),
                return_exceptions=True,
            )
    except NHLApiError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected error: {exc}"}

    if isinstance(landing_result, Exception):
        return {"error": f"Could not fetch game {game_id}: {landing_result}"}

    landing: dict      = landing_result
    pbp: dict          = pbp_result if not isinstance(pbp_result, Exception) else {}
    boxscore_raw: dict = boxscore_result if not isinstance(boxscore_result, Exception) else {}
    shifts_raw: list   = (shifts_result.get("data") or []) if not isinstance(shifts_result, Exception) else []

    # ── Top-level metadata ────────────────────────────────────────────────────
    away_raw     = landing.get("awayTeam") or {}
    home_raw     = landing.get("homeTeam") or {}
    clock_raw    = landing.get("clock") or {}
    outcome_raw  = landing.get("gameOutcome") or {}
    period_desc  = landing.get("periodDescriptor") or {}

    game_state   = landing.get("gameState", "FUT")
    period_num   = period_desc.get("number") or landing.get("period") or 0
    period_type  = period_desc.get("periodType", "REG")

    away_abbrev  = away_raw.get("abbrev", "")
    home_abbrev  = home_raw.get("abbrev", "")
    away_id      = away_raw.get("id")
    home_id      = home_raw.get("id")

    # ── Scoring summary ───────────────────────────────────────────────────────
    scoring_out: list[dict] = []
    for block in (landing.get("summary") or {}).get("scoring") or []:
        pd        = block.get("periodDescriptor") or {}
        p_num     = pd.get("number", 0)
        p_type    = pd.get("periodType", "REG")
        p_label   = _gc_period_label(p_num, p_type)
        for goal in block.get("goals") or []:
            first  = (goal.get("firstName") or {}).get("default", "")
            last   = (goal.get("lastName")  or {}).get("default", "")
            scorer = f"{first[0]}. {last}" if first else last
            assists: list[str] = []
            for a in goal.get("assists") or []:
                af = (a.get("firstName") or {}).get("default", "")
                al = (a.get("lastName")  or {}).get("default", "")
                assists.append(f"{af[0]}. {al}" if af else al)
            strength = goal.get("strength", "EV")
            if goal.get("goalModifier") == "empty-net":
                strength = "EN"
            scoring_out.append({
                "period":       p_num,
                "period_label": p_label,
                "time":         goal.get("timeInPeriod", ""),
                "team":         (goal.get("teamAbbrev") or {}).get("default", ""),
                "scorer":       scorer,
                "scorer_id":    goal.get("playerId"),
                "assists":      assists,
                "strength":     strength,
                "away_score":   goal.get("awayScore"),
                "home_score":   goal.get("homeScore"),
            })

    # ── Penalties ─────────────────────────────────────────────────────────────
    penalties_out: list[dict] = []
    for block in (landing.get("summary") or {}).get("penalties") or []:
        pd      = block.get("periodDescriptor") or {}
        p_num   = pd.get("number", 0)
        p_type  = pd.get("periodType", "REG")
        p_label = _gc_period_label(p_num, p_type)
        for pen in block.get("penalties") or []:
            first  = (pen.get("committedByPlayer") or {})
            pname  = (first.get("firstName") or {}).get("default", "")
            plast  = (first.get("lastName")  or {}).get("default", "")
            pname_full = f"{pname[0]}. {plast}" if pname else plast
            penalties_out.append({
                "period":       p_num,
                "period_label": p_label,
                "time":         pen.get("timeInPeriod", ""),
                "team":         (pen.get("teamAbbrev") or {}).get("default", ""),
                "player":       pname_full,
                "description":  pen.get("descKey", ""),
                "duration":     pen.get("duration", 0) or 0,
            })

    # ── Team stats ────────────────────────────────────────────────────────────
    team_stats_raw: dict[str, dict] = {}
    # Landing has teamGameStats inside summary; boxscore has it at top level.
    # Use whichever is non-empty (boxscore is more reliably populated mid-game).
    _raw_tgs = (
        (landing.get("summary") or {}).get("teamGameStats")
        or boxscore_raw.get("teamGameStats")
        or []
    )
    for row in _raw_tgs:
        cat    = row.get("category", "")
        a_val  = row.get("awayValue", 0)
        h_val  = row.get("homeValue", 0)
        team_stats_raw.setdefault("away", {})[cat] = a_val
        team_stats_raw.setdefault("home", {})[cat] = h_val

    away_stats = _gc_team_stats(team_stats_raw.get("away") or {})
    home_stats = _gc_team_stats(team_stats_raw.get("home") or {})

    # ── PBP-based stat fallback (fills gaps when API summary is absent/empty) ──
    # Count directly from play events so live games early in a period always
    # have real numbers rather than zeros from an as-yet-unpopulated summary.
    _pbp_away: dict[str, int] = {"shots": 0, "hits": 0, "giveaways": 0, "takeaways": 0}
    _pbp_home: dict[str, int] = {"shots": 0, "hits": 0, "giveaways": 0, "takeaways": 0}
    for _play in (pbp.get("plays") or []):
        _et  = _play.get("typeDescKey", "")
        _oid = (_play.get("details") or {}).get("eventOwnerTeamId")
        _tgt = _pbp_away if _oid == away_id else _pbp_home if _oid == home_id else None
        if _tgt is None:
            continue
        if _et == "shot-on-goal":
            _tgt["shots"]    += 1
        elif _et == "hit":
            _tgt["hits"]     += 1
        elif _et == "giveaway":
            _tgt["giveaways"] += 1
        elif _et == "takeaway":
            _tgt["takeaways"] += 1

    for _side_stats, _pbp in [(away_stats, _pbp_away), (home_stats, _pbp_home)]:
        for _key in ("shots", "hits", "giveaways", "takeaways"):
            if _side_stats[_key] == 0 and _pbp[_key] > 0:
                _side_stats[_key] = _pbp[_key]

    # ── Faceoff % from PBP faceoff events ────────────────────────────────────
    # teamGameStats is empty in modern NHL API responses; count directly from plays.
    _fo_away_wins = _fo_home_wins = _fo_away_total = _fo_home_total = 0
    for _play in (pbp.get("plays") or []):
        if _play.get("typeDescKey") != "faceoff":
            continue
        _det = _play.get("details") or {}
        _winner_team = _det.get("eventOwnerTeamId")
        if _winner_team == away_id:
            _fo_away_wins += 1
        elif _winner_team == home_id:
            _fo_home_wins += 1
        _fo_away_total += 1
        _fo_home_total += 1
    if _fo_away_total > 0:
        away_stats["faceoff_pct"] = round(_fo_away_wins / _fo_away_total * 100, 1)
        home_stats["faceoff_pct"] = round(_fo_home_wins / _fo_home_total * 100, 1)

    # ── PP opportunities from PBP penalty events ─────────────────────────────
    # A penalty against team X gives a PP to the other team.
    _pp_away_opps = _pp_home_opps = 0
    for _play in (pbp.get("plays") or []):
        if _play.get("typeDescKey") != "penalty":
            continue
        _det  = _play.get("details") or {}
        _tc   = _det.get("typeCode", "")
        _dur  = _det.get("duration", 0) or 0
        if _tc in ("BEN", "PS") or _dur < 2:   # bench minor / penalty shot: no PP
            continue
        _pen_team = _det.get("eventOwnerTeamId")
        if _pen_team == away_id:
            _pp_home_opps += 1   # away penalised → home PP
        elif _pen_team == home_id:
            _pp_away_opps += 1   # home penalised → away PP
    if away_stats["pp_opportunities"] == 0:
        away_stats["pp_opportunities"] = _pp_away_opps
    if home_stats["pp_opportunities"] == 0:
        home_stats["pp_opportunities"] = _pp_home_opps

    # ── xG enrichment (best-effort) ───────────────────────────────────────────
    xg_map: dict[str, tuple[float, float]] = {}
    try:
        import polars as pl
        xg_dir = _GRETZKY_DATA_DIR / "xg_finishing"
        xg_files = sorted(xg_dir.glob("xg_finishing_*.parquet")) if xg_dir.exists() else []
        if xg_files:
            cols_needed = ["shooter_name", "xg_sum", "finishing"]
            xg_df = pl.read_parquet(xg_files[-1], columns=cols_needed)
            for r in xg_df.drop_nulls(subset=["shooter_name"]).to_dicts():
                key = (r["shooter_name"] or "").lower()
                xg_map[key] = (r.get("xg_sum") or 0.0, r.get("finishing") or 0.0)
    except Exception:
        pass

    # ── Per-player stats (dedicated boxscore endpoint) ────────────────────────
    # Note: the landing endpoint's "boxscore" key only has team-level stats;
    # playerByGameStats lives exclusively in /v1/gamecenter/{id}/boxscore.
    by_game      = (boxscore_raw.get("playerByGameStats") or {})
    away_game    = by_game.get("awayTeam") or {}
    home_game    = by_game.get("homeTeam") or {}

    away_skaters = [_gc_parse_skater(p, xg_map)
                    for group in ["forwards", "defensemen"]
                    for p in (away_game.get(group) or [])]
    home_skaters = [_gc_parse_skater(p, xg_map)
                    for group in ["forwards", "defensemen"]
                    for p in (home_game.get(group) or [])]
    away_goalies = [_gc_parse_goalie(p) for p in (away_game.get("goalies") or [])]
    home_goalies = [_gc_parse_goalie(p) for p in (home_game.get("goalies") or [])]

    # ── PP goals from playerByGameStats (teamGameStats is empty in modern API) ─
    for _side_stats, _game_team in [(away_stats, away_game), (home_stats, home_game)]:
        if _side_stats["pp_goals"] == 0:
            _ppg = sum(
                p.get("powerPlayGoals", 0) or 0
                for group in ["forwards", "defense", "defensemen"]
                for p in (_game_team.get(group) or [])
            )
            if _ppg > 0:
                _side_stats["pp_goals"] = _ppg

    # Compute per-team xG total from player sums (only if we have xg data)
    if xg_map:
        away_xg_total = sum(
            s["xg_sum"] for s in away_skaters if s["xg_sum"] is not None
        )
        home_xg_total = sum(
            s["xg_sum"] for s in home_skaters if s["xg_sum"] is not None
        )
        away_stats["xg"] = round(away_xg_total, 2) if away_xg_total else None
        home_stats["xg"] = round(home_xg_total, 2) if home_xg_total else None

    # ── Lineup groups (for Rosters tab + rink animation) ─────────────────────
    # Primary: PBP rosterSpots — always populated for any game state, includes
    # all positions (forwards/D/G). Fallback to playerByGameStats if PBP empty.
    # Players are sorted by TOI descending so Line 1 = highest-ice-time forwards.
    _FWD_CODES = {"C", "L", "R", "LW", "RW"}

    _away_spots = [s for s in (pbp.get("rosterSpots") or []) if s.get("teamId") == away_id]
    _home_spots = [s for s in (pbp.get("rosterSpots") or []) if s.get("teamId") == home_id]

    # Build pid → position map from rosterSpots + playerByGameStats for shift ranking
    _pid_pos: dict[int, str] = {}
    for spot in (pbp.get("rosterSpots") or []):
        pid = spot.get("playerId")
        if pid:
            _pid_pos[int(pid)] = spot.get("positionCode", "")
    for _side in [away_game, home_game]:
        for _grp, _pos in [("forwards", ""), ("defense", "D")]:
            for _p in (_side.get(_grp) or []):
                pid = _p.get("playerId")
                if pid:
                    _pid_pos[int(pid)] = _p.get("position") or _pos

    # Compute shift-based line ranks (empty if no shift data yet — e.g. pre-game)
    _away_rank = _line_order_from_shifts(shifts_raw, away_id, _pid_pos) if shifts_raw else {}
    _home_rank = _line_order_from_shifts(shifts_raw, home_id, _pid_pos) if shifts_raw else {}

    def _spots_to_group(spots: list[dict], rank: dict[int, int]) -> dict:
        def _spot_name(s: dict) -> str:
            first = (s.get("firstName") or {}).get("default", "")
            last  = (s.get("lastName")  or {}).get("default", "")
            return f"{first[0]}. {last}" if first else last

        def _sort_key(s: dict) -> int:
            pid = int(s["playerId"]) if s.get("playerId") else 0
            return rank.get(pid, 999)

        fwd  = sorted([s for s in spots if s.get("positionCode") in _FWD_CODES], key=_sort_key)
        dmen = sorted([s for s in spots if s.get("positionCode") == "D"],         key=_sort_key)
        gols = [s for s in spots if s.get("positionCode") == "G"]

        def _row(s: dict, pos: str) -> dict:
            return {
                "name": _spot_name(s),
                "number": s.get("sweaterNumber") or 0,
                "position": pos,
                "player_id": int(s["playerId"]) if s.get("playerId") else None,
                "headshot_url": s.get("headshot") or None,
            }

        return {
            "forwards":   [_row(s, s.get("positionCode", "")) for s in fwd],
            "defensemen": [_row(s, "D")                       for s in dmen],
            "goalies":    [_row(s, "G")                       for s in gols],
        }

    def _lineup_group_fallback(game_team: dict, rank: dict[int, int]) -> dict:
        def _sort_key(p: dict) -> int:
            pid = int(p["playerId"]) if p.get("playerId") else 0
            return rank.get(pid, 999)

        def _fb_row(p: dict, pos: str) -> dict:
            return {
                "name": (p.get("name") or {}).get("default", ""),
                "number": p.get("sweaterNumber") or 0,
                "position": pos,
                "player_id": int(p["playerId"]) if p.get("playerId") else None,
                "headshot_url": p.get("headshot") or None,
            }
        return {
            "forwards":   [_fb_row(p, p.get("position") or "")
                           for p in sorted(game_team.get("forwards") or [], key=_sort_key)],
            "defensemen": [_fb_row(p, "D")
                           for p in sorted(game_team.get("defensemen") or [], key=_sort_key)],
            "goalies":    [_fb_row(p, "G") for p in (game_team.get("goalies") or [])],
        }

    if _away_spots or _home_spots:
        lineup = {
            "away": _spots_to_group(_away_spots, _away_rank),
            "home": _spots_to_group(_home_spots, _home_rank),
        }
    else:
        lineup = {
            "away": _lineup_group_fallback(away_game, _away_rank),
            "home": _lineup_group_fallback(home_game, _home_rank),
        }

    # ── Plays + shot map (from PBP) ───────────────────────────────────────────
    plays_out:    list[dict] = []
    shots_for_map: list[dict] = []

    # Build a minimal player-id → name roster from the boxscore for descriptions
    roster: dict[int, str] = {}
    for side in [away_game, home_game]:
        for group in ["forwards", "defensemen", "goalies"]:
            for p in (side.get(group) or []):
                pid = p.get("playerId")
                if pid:
                    roster[int(pid)] = (p.get("name") or {}).get("default", "")

    # Team-id → abbrev map for play attribution
    team_id_map: dict = {}
    if away_id:
        team_id_map[away_id] = away_abbrev
    if home_id:
        team_id_map[home_id] = home_abbrev

    all_plays = pbp.get("plays") or []
    for play in all_plays:
        ev_type  = play.get("typeDescKey", "")
        pd2      = play.get("periodDescriptor") or {}
        p_num2   = pd2.get("number", 0)
        p_type2  = pd2.get("periodType", "REG")
        time_str = play.get("timeInPeriod", "")
        details  = play.get("details") or {}
        sit_code = play.get("situationCode") or ""

        owner_id     = details.get("eventOwnerTeamId")
        team_abbrev  = team_id_map.get(owner_id, "")
        is_home_ev   = (owner_id == home_id)
        strength     = _gc_strength(sit_code, is_home_ev)

        x_raw = details.get("xCoord")
        y_raw = details.get("yCoord")
        x = float(x_raw) if x_raw is not None else None
        y = float(y_raw) if y_raw is not None else None

        # homeTeamDefendingSide is a per-play field in the NHL PBP API (not game-level).
        # "left"  → home defends left end  → home net at x=-89 → home attacks right (x>0)
        # "right" → home defends right end → home net at x=+89 → home attacks left  (x<0)
        # Our convention: home attacks RIGHT (x>0 = home attacking zone, x<0 = away attacking zone).
        # Flip when home attacks left (htds="right") so that convention holds.
        htds = play.get("homeTeamDefendingSide", "left")
        flip_coords = (htds == "right")
        if flip_coords:
            if x is not None: x = -x
            if y is not None: y = -y

        # ── Shots for map ──────────────────────────────────────────────────
        if ev_type in _SHOT_TYPES and x is not None and y is not None:
            # Normalise: fold so that all shots point toward positive-x (attacking end)
            nx, ny = (x, y) if x >= 0 else (-x, -y)
            # Only plot shots inside the attacking blue line (25 ft from centre).
            # Outliers from the neutral/defensive zone are NHL API data artefacts
            # (e.g. dump-ins labelled as missed-shots) and look wrong on the map.
            if nx < 25:
                continue
            pid_key = "scoringPlayerId" if ev_type == "goal" else "shootingPlayerId"
            shooter = roster.get(details.get(pid_key), "")
            shots_for_map.append({
                "team":       team_abbrev,
                "event_type": ev_type,
                "x":          nx,
                "y":          ny,
                "shooter":    shooter,
                "result":     ev_type.replace("-", " "),
                "period":     p_num2,
                "time":       time_str,
            })

        # ── Plays feed (skip noise) ────────────────────────────────────────
        if ev_type in _SKIP_PLAYS:
            continue

        # Build a human-readable description
        if ev_type == "goal":
            pid    = details.get("scoringPlayerId")
            scorer = roster.get(pid, "")
            desc   = f"GOAL — {scorer}" if scorer else "GOAL"
        elif ev_type == "shot-on-goal":
            pid    = details.get("shootingPlayerId")
            name   = roster.get(pid, "")
            stype  = details.get("shotType", "")
            desc   = f"Shot{f' ({stype})' if stype else ''}{f' — {name}' if name else ''}"
        elif ev_type == "missed-shot":
            pid    = details.get("shootingPlayerId")
            name   = roster.get(pid, "")
            desc   = f"Missed shot{f' — {name}' if name else ''}"
        elif ev_type == "blocked-shot":
            pid    = details.get("blockingPlayerId")
            name   = roster.get(pid, "")
            desc   = f"Blocked{f' by {name}' if name else ''}"
        elif ev_type == "penalty":
            pid    = details.get("committedByPlayerId")
            name   = roster.get(pid, "")
            pdesc  = details.get("descKey", "")
            dur    = details.get("duration", 0)
            desc   = f"Penalty — {pdesc}{f' ({dur} min)' if dur else ''}{f' on {name}' if name else ''}"
        elif ev_type == "hit":
            hitter = roster.get(details.get("hittingPlayerId"), "")
            hittee = roster.get(details.get("hitteePlayerId"), "")
            desc   = f"Hit{f' by {hitter}' if hitter else ''}{f' on {hittee}' if hittee else ''}"
        elif ev_type == "giveaway":
            name   = roster.get(details.get("playerId"), "")
            desc   = f"Giveaway{f' — {name}' if name else ''}"
        elif ev_type == "takeaway":
            name   = roster.get(details.get("playerId"), "")
            desc   = f"Takeaway{f' — {name}' if name else ''}"
        else:
            desc   = ev_type.replace("-", " ").title()

        plays_out.append({
            "event_type": ev_type,
            "period":     p_num2,
            "time":       time_str,
            "team":       team_abbrev,
            "description": desc,
            "x":          x,
            "y":          y,
            "strength":   strength,
            "sit_code":   sit_code,
        })

    # Most-recent 50 plays, latest first
    plays_out = plays_out[-50:][::-1]

    # ── Assemble response ─────────────────────────────────────────────────────
    outcome_type: str | None = None
    if game_state in ("FINAL", "OFF"):
        lpt = outcome_raw.get("lastPeriodType", "REG")
        if lpt in ("OT", "SO"):
            outcome_type = lpt

    return {
        "game_id":       game_id,
        "game_state":    game_state,
        "period":        period_num,
        "period_label":  _gc_period_label(period_num, period_type),
        "clock":         clock_raw.get("timeRemaining", ""),
        "in_intermission": bool(clock_raw.get("inIntermission", False)),
        "outcome_type":  outcome_type,
        "venue":         (landing.get("venue") or {}).get("default", ""),
        "game_date":     landing.get("gameDate", ""),
        "away": {
            "team":  away_abbrev,
            "score": away_raw.get("score", 0),
        },
        "home": {
            "team":  home_abbrev,
            "score": home_raw.get("score", 0),
        },
        "scoring":    scoring_out,
        "penalties":  penalties_out,
        "team_stats": {"away": away_stats, "home": home_stats},
        "skaters":    {"away": away_skaters, "home": home_skaters},
        "goalies":    {"away": away_goalies, "home": home_goalies},
        "lineup":     lineup,
        "plays":      plays_out,
        "shots_for_map": shots_for_map,
    }


# ===========================================================================
# /game/{game_id}/highlights — NHL highlight video IDs (Brightcove)
# ===========================================================================


@app.get("/game/{game_id}/highlights")
async def game_highlights(game_id: int) -> dict:
    """Return highlight data for a completed NHL game.

    Fetches two sources in parallel:
    1. NHL ``/v1/wsc/game-story/{game_id}`` — per-goal Brightcove clip IDs
    2. NHL D3 content API ``forge-dapi.d3.nhle.com/v2/content/en-us/videos``
       filtered by tag ``gameid-{game_id}`` — returns the full game recap
       (tagged ``game-recap``) and condensed game (tagged ``condensed-game``).

    All clips embed via:
        https://players.brightcove.net/6415718365001/EXtG1xJ7H_default/index.html?videoId={id}
    """
    cache_key = f"highlights:{game_id}"
    cached = _cache_get(cache_key)
    if cached:
        return cached  # type: ignore[return-value]

    async def _fetch_story() -> dict:
        async with httpx.AsyncClient(base_url="https://api-web.nhle.com", timeout=10.0) as c:
            r = await c.get(f"/v1/wsc/game-story/{game_id}")
            r.raise_for_status()
            return r.json()

    async def _fetch_d3_videos() -> list[dict]:
        url = (
            "https://forge-dapi.d3.nhle.com/v2/content/en-us/videos"
            f"?tags.slug=gameid-{game_id}&context.slug=nhl&limit=50"
        )
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.json().get("items", [])

    try:
        story, d3_items = await asyncio.gather(_fetch_story(), _fetch_d3_videos())
    except Exception as exc:
        return {"error": str(exc)}

    away_abbrev = (story.get("awayTeam") or {}).get("abbrev", "")
    home_abbrev = (story.get("homeTeam") or {}).get("abbrev", "")
    game_state  = story.get("gameState", "")

    # ── Featured clips (recap + condensed) from D3 ────────────────────────────
    recap_id: str | None     = None
    condensed_id: str | None = None
    for item in d3_items:
        tags   = [t.get("slug", "") for t in (item.get("tags") or [])]
        bc_id  = str((item.get("fields") or {}).get("brightcoveId", "") or "")
        if not bc_id:
            continue
        if "game-recap" in tags:
            recap_id = bc_id
        elif "condensed-game" in tags:
            condensed_id = bc_id

    # ── Per-goal clips from game-story ────────────────────────────────────────
    goals: list[dict] = []
    for period_block in (story.get("summary") or {}).get("scoring") or []:
        pd      = period_block.get("periodDescriptor") or {}
        p_num   = pd.get("number", 0)
        p_type  = pd.get("periodType", "REG")
        p_label = "OT" if p_type == "OT" else "SO" if p_type == "SO" else f"P{p_num}"
        for goal in period_block.get("goals") or []:
            clip_id = goal.get("highlightClip") or goal.get("discreteClip")
            if not clip_id:
                continue
            first  = (goal.get("firstName") or {}).get("default", "") if isinstance(goal.get("firstName"), dict) else goal.get("firstName", "")
            last   = (goal.get("lastName")  or {}).get("default", "") if isinstance(goal.get("lastName"),  dict) else goal.get("lastName",  "")
            team   = (goal.get("teamAbbrev") or {}).get("default", "") if isinstance(goal.get("teamAbbrev"), dict) else goal.get("teamAbbrev", "")
            goals.append({
                "clip_id":   str(clip_id),
                "period":    p_label,
                "time":      goal.get("timeInPeriod", ""),
                "scorer":    f"{first} {last}".strip(),
                "team":      team,
                "strength":  goal.get("strength", "ev").upper(),
                "share_url": goal.get("highlightClipSharingUrl", ""),
            })

    result: dict = {
        "game_id":      game_id,
        "away_team":    away_abbrev,
        "home_team":    home_abbrev,
        "game_state":   game_state,
        "recap_id":     recap_id,
        "condensed_id": condensed_id,
        "goals":        goals,
    }
    # Use a short TTL when no clips are available yet — they typically appear
    # within minutes of the game ending.  Once clips are present, cache for the
    # full 5 minutes (clips don't change after publication).
    has_any = recap_id or condensed_id or bool(goals)
    _cache_set(cache_key, result, ttl=_CACHE_TTL if has_any else 45.0)
    return result


# ===========================================================================
# /clip/resolve/{clip_id} — Brightcove → direct HLS (ad-free)
# ===========================================================================

_BC_ACCOUNT   = "6415718365001"
_BC_PLAYER    = "EXtG1xJ7H_default"
_clip_resolve_cache: dict[str, tuple[float, str | None]] = {}
_CLIP_CACHE_TTL = 3600.0   # 1 hour — Brightcove signed URLs last ~4 h


@app.get("/clip/resolve/{clip_id}")
async def resolve_clip(clip_id: str) -> dict:
    """Return the best direct HLS URL for a Brightcove clip.

    Uses yt-dlp to extract the manifest URL (no ads, no Brightcove iframe).
    Results are cached for 1 hour (Brightcove signed URLs are valid ~4 h).

    Returns ``{"url": "<m3u8>"}`` on success, ``{"url": null, "error": "..."}``
    on failure.
    """
    import time as _t

    cached = _clip_resolve_cache.get(clip_id)
    if cached and (_t.monotonic() - cached[0]) < _CLIP_CACHE_TTL:
        return {"url": cached[1]}

    bc_url = (
        f"https://players.brightcove.net/{_BC_ACCOUNT}/"
        f"{_BC_PLAYER}/index.html?videoId={clip_id}"
    )

    try:
        import yt_dlp as _yt_dlp

        ydl_opts = {
            "quiet":        True,
            "no_warnings":  True,
            "skip_download": True,
            # Prefer HLS video+audio combo; avoid audio-only streams
            "format": "best[ext=mp4][vcodec!=none]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        }

        loop = asyncio.get_event_loop()

        def _extract() -> str | None:
            with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(bc_url, download=False)
            fmts = info.get("formats") or []

            # Use the master HLS manifest_url from any video+audio HLS format.
            # The master playlist contains both video and audio renditions; HLS.js
            # selects them automatically.  Using a specific variant URL loses the
            # audio track (Brightcove separates them in the manifest).
            hls_fmts = [
                f for f in fmts
                if (f.get("protocol") or "").startswith("m3u8")
                and f.get("vcodec") not in (None, "none")
                and f.get("manifest_url")
            ]
            if hls_fmts:
                hls_fmts.sort(key=lambda f: f.get("tbr") or f.get("vbr") or 0, reverse=True)
                return hls_fmts[0]["manifest_url"]   # master playlist → audio included

            # Fallback: any HLS format with a manifest_url
            any_hls = [f for f in fmts if f.get("manifest_url")]
            if any_hls:
                return any_hls[0]["manifest_url"]

            # Fallback: direct mp4
            mp4_fmts = [f for f in fmts if f.get("ext") == "mp4" and f.get("vcodec") not in (None, "none")]
            if mp4_fmts:
                mp4_fmts.sort(key=lambda f: f.get("tbr") or 0, reverse=True)
                return mp4_fmts[0].get("url")
            return info.get("url")

        url = await loop.run_in_executor(None, _extract)
        _clip_resolve_cache[clip_id] = (_t.monotonic(), url)
        return {"url": url}

    except Exception as exc:
        return {"url": None, "error": str(exc)}


# ===========================================================================
# /streams/{game_id} — onhockey.tv stream link scraper
# ===========================================================================

_streams_cache: dict[int, tuple[list, float]] = {}  # game_id -> (streams, timestamp)
_STREAMS_CACHE_TTL = 180  # 3 minutes


@app.get("/streams/{game_id}")
async def game_streams(game_id: int) -> dict:
    """Scrape live stream embed links from onhockey.tv for a given NHL game."""
    import time as _t
    cached = _streams_cache.get(game_id)
    if cached and (_t.monotonic() - cached[1]) < _STREAMS_CACHE_TTL:
        streams = cached[0]
        # Still need team abbrs for response — read from cache metadata
        meta = _streams_cache.get(-game_id)  # negative key stores metadata
        away = meta[0] if meta else "?"
        home = meta[1] if meta else "?"
        return {"streams": streams, "game_id": game_id, "away": away, "home": home, "cached": True}
    import re as _re
    from data.nhl_client import NHLClient

    # 1. Resolve team abbreviations via NHL landing
    try:
        async with NHLClient() as client:
            landing = await client.get_landing(game_id)
        if isinstance(landing, Exception) or not landing:
            return {"streams": [], "error": "Could not fetch game info from NHL API"}
        away_abbr = (landing.get("awayTeam") or {}).get("abbrev", "")
        home_abbr = (landing.get("homeTeam") or {}).get("abbrev", "")
    except Exception as exc:
        return {"streams": [], "error": f"NHL API error: {exc}"}

    away_kw = _TEAM_KEYWORDS.get(away_abbr, [away_abbr.lower()])
    home_kw = _TEAM_KEYWORDS.get(home_abbr, [home_abbr.lower()])

    # 2. Fetch onhockey.tv schedule with browser-like headers
    import httpx as _httpx
    _headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://onhockey.tv/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    html = ""
    try:
        async with _httpx.AsyncClient(timeout=10, follow_redirects=True) as hx:
            resp = await hx.get("https://onhockey.tv/schedule_table.php", headers=_headers)
        if resp.status_code == 200:
            html = resp.text
    except Exception:
        pass  # fall through to backup scrapers

    # 3. Find game row matching both teams (split on <tr class='game')
    rows = _re.split(r"<tr class='game'", html)[1:]
    matched_row = None
    for row in rows:
        row_lower = row.lower()
        if (any(kw in row_lower for kw in away_kw)
                and any(kw in row_lower for kw in home_kw)):
            matched_row = row
            break

    # 4. Walk the row to extract feed labels + stream links
    streams: list[dict] = []
    if matched_row:
        current_feed = "main"
        # Split into tokens: <br>..., <a ...>, </a>, or plain text
        tokens = _re.split(r"(<br[^>]*>|<a [^>]+>|</a>)", matched_row)
        for token in tokens:
            # Detect feed-type section header (plain text after a <br>)
            feed_m = _re.search(
                r"(home\s+feed|away\s+feed|french|russian|spanish|english)[:\s]*",
                token, _re.I
            )
            if feed_m and "<a " not in token:
                current_feed = _re.sub(r"\s+", "_", feed_m.group(1).strip().lower())
                continue
            # Extract stream link
            link_m = _re.match(
                r"<a\s+href='np_stream400\.php\?channel=(//.+?)'\s+target='player_frame'\s+title='([^']*)'",
                token,
            )
            if link_m:
                raw_url = "https:" + link_m.group(1)
                # Only treat as direct CDN if it's an actual m3u8/mpd URL.
                # Most onhockey.tv channel links are embed pages — mark those
                # embed_only so the frontend skips slow yt-dlp extraction.
                is_direct_m3u8 = ".m3u8" in raw_url or ".mpd" in raw_url
                entry: dict = {
                    "url":   raw_url,
                    "feed":  current_feed,
                    "title": link_m.group(2),
                }
                if is_direct_m3u8:
                    entry["_direct"] = True
                else:
                    entry["embed_only"] = True
                streams.append(entry)

    # ── Backup sources — always supplement onhockey.tv up to 30 total ───────
    _MAX_STREAMS = 30
    if len(streams) < _MAX_STREAMS:
        import asyncio as _aio_streams
        backup_results = await _aio_streams.gather(
            _scrape_streameast(away_kw, home_kw, away_abbr, home_abbr),
            _scrape_methstreams(away_kw, home_kw, away_abbr, home_abbr),
            _scrape_weakstreams(away_kw, home_kw, away_abbr, home_abbr),
            _scrape_sportsurge(away_kw, home_kw, away_abbr, home_abbr),
            _scrape_crackstreams(away_kw, home_kw, away_abbr, home_abbr),
            _scrape_buffstreams(away_kw, home_kw, away_abbr, home_abbr),
            return_exceptions=True,
        )
        seen_urls: set[str] = {s["url"] for s in streams}
        for result in backup_results:
            if isinstance(result, list):
                for s in result:
                    if s["url"] not in seen_urls:
                        streams.append(s)
                        seen_urls.add(s["url"])
        streams = streams[:_MAX_STREAMS]

    # ── Priority sort: lower priority number = fewer ads / more reliable ─────
    # 0 = direct CDN m3u8 (onhockey.tv extracts these — already ad-free)
    # 1 = known-clean embeds where static extraction works (apl396, streamfree)
    # 2 = standard backup aggregators (Playwright extracts m3u8)
    # 3 = iframe-only sites / unknown
    def _stream_priority(s: dict) -> int:
        if s.get("_direct"):
            return 0   # onhockey.tv direct CDN — plays via HLS proxy, zero ads
        url = s.get("url", "").lower()
        if url.endswith(".m3u8") or ".m3u8?" in url:
            return 0
        if any(d in url for d in ("apl396.me", "streamfree.app")):
            return 1   # static HTML extraction works, clean
        if any(d in url for d in ("methstreams", "crackstreams")):
            return 2   # Playwright extracts m3u8, minimal ads
        return 3       # general aggregator or iframe fallback

    streams.sort(key=_stream_priority)
    # Strip internal metadata field before sending to client
    for s in streams:
        s.pop("_direct", None)

    # Cache result to avoid re-scraping on every page load
    import time as _t2
    _streams_cache[game_id] = (streams, _t2.monotonic())
    _streams_cache[-game_id] = (away_abbr, home_abbr)  # type: ignore[assignment]

    return {"streams": streams, "game_id": game_id, "away": away_abbr, "home": home_abbr}


# ---------------------------------------------------------------------------
# Backup stream scrapers — all run in parallel, each returns [{url,feed,title}]
# Priority assigned at the call-site based on URL patterns, not source.
# ---------------------------------------------------------------------------

import httpx as _httpx_backup
import re as _re_backup

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


async def _fetch_html(url: str, referer: str = "", timeout: int = 5) -> str | None:
    hdrs = {**_BROWSER_HEADERS}
    if referer:
        hdrs["Referer"] = referer
    try:
        async with _httpx_backup.AsyncClient(timeout=timeout, follow_redirects=True) as hx:
            r = await hx.get(url, headers=hdrs)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def _row_matches(text: str, kw_a: list[str], kw_b: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in kw_a) and any(k in t for k in kw_b)


def _extract_game_links(
    html: str, base: str, away_kw: list[str], home_kw: list[str],
    away_abbr: str, home_abbr: str, feed: str = "main",
    embed_only: bool = False,
) -> list[dict]:
    """Generic anchor-tag link extractor shared by all scrapers."""
    found: list[dict] = []
    for m in _re_backup.finditer(
        r"<a\s[^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        html, _re_backup.S | _re_backup.I
    ):
        href, inner = m.group(1), m.group(2)
        text = _re_backup.sub(r"<[^>]+>", "", inner).strip()
        combined = (href + " " + text).lower()
        if _row_matches(combined, away_kw, home_kw):
            full_url = href if href.startswith("http") else base.rstrip("/") + "/" + href.lstrip("/")
            entry: dict = {
                "url":   full_url,
                "feed":  feed,
                "title": text or f"{away_abbr} @ {home_abbr}",
            }
            if embed_only:
                entry["embed_only"] = True
            found.append(entry)
    return found


async def _scrape_multi(
    candidates: list[tuple[str, str]],   # [(schedule_url, referer), ...]
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    """Try each (url, referer) in order; return first non-empty result.
    All results are tagged embed_only=True — these are aggregator pages that
    embed third-party players, so stream-resolve should skip yt-dlp/Playwright.
    """
    for url, referer in candidates:
        base = "/".join(url.split("/")[:3])  # scheme + host
        html = await _fetch_html(url, referer=referer)
        if not html:
            continue
        links = _extract_game_links(html, base, away_kw, home_kw, away_abbr, home_abbr, embed_only=True)
        if links:
            return links
    return []


async def _scrape_streameast(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    return await _scrape_multi([
        ("https://streameast.live/nhl",        "https://streameast.live/"),
        ("https://streameast.io/nhl",           "https://streameast.io/"),
        ("https://streameast.xyz/nhl",          "https://streameast.xyz/"),
        ("https://the.streameast.app/nhl",      "https://the.streameast.app/"),
        ("https://streameast.app/nhl",          "https://streameast.app/"),
    ], away_kw, home_kw, away_abbr, home_abbr)


async def _scrape_methstreams(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    return await _scrape_multi([
        ("https://methstreams.com/nhl",         "https://methstreams.com/"),
        ("https://methstreams.live/nhl",        "https://methstreams.live/"),
        ("https://www.methstreams.com/nhl",     "https://www.methstreams.com/"),
    ], away_kw, home_kw, away_abbr, home_abbr)


async def _scrape_crackstreams(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    return await _scrape_multi([
        ("https://crackstreams.is/nhl",         "https://crackstreams.is/"),
        ("https://crackstreams.sx/nhl",         "https://crackstreams.sx/"),
        ("https://crackstreams.biz/nhl",        "https://crackstreams.biz/"),
        ("https://crackstreams.com/nhlstream",  "https://crackstreams.com/"),
    ], away_kw, home_kw, away_abbr, home_abbr)


async def _scrape_buffstreams(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    return await _scrape_multi([
        ("https://buffstreams.app/sport/nhl",   "https://buffstreams.app/"),
        ("https://buffstreams.me/sport/nhl",    "https://buffstreams.me/"),
        ("https://buffstreams.sx/sport/nhl",    "https://buffstreams.sx/"),
    ], away_kw, home_kw, away_abbr, home_abbr)


async def _scrape_weakstreams(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    return await _scrape_multi([
        ("https://weakstreams.com/nhl",         "https://weakstreams.com/"),
        ("https://weakstreams.live/nhl",        "https://weakstreams.live/"),
    ], away_kw, home_kw, away_abbr, home_abbr)


async def _scrape_sportsurge(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    return await _scrape_multi([
        ("https://sportsurge.net/",             "https://sportsurge.net/"),
        ("https://sportsurge.net/nhl",          "https://sportsurge.net/"),
        ("https://sportsurge.club/",            "https://sportsurge.club/"),
        ("https://sportsurge.app/",             "https://sportsurge.app/"),
    ], away_kw, home_kw, away_abbr, home_abbr)


# ===========================================================================
# /stream-proxy — transparent HLS proxy (bypasses CDN CORS restrictions)
# The browser fetches /stream-proxy?url=<cdn_url> and this endpoint forwards
# the request server-side, rewrites segment/manifest URLs, and returns the
# content with permissive CORS headers so hls.js can load it.
# ===========================================================================

from urllib.parse import urljoin, quote, urlparse, urlunparse
from fastapi import Request
from fastapi.responses import Response as FastResponse
import re as _re_proxy


def _rewrite_m3u8(content: str, original_url: str, proxy_base: str) -> str:
    """Rewrite .m3u8 and .ts URLs in a manifest to go through our proxy."""
    lines = content.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # Rewrite URI= attributes inside tags (e.g. #EXT-X-MEDIA, #EXT-X-I-FRAME-STREAM-INF)
            def replace_uri(m: _re_proxy.Match) -> str:
                uri = m.group(1)
                abs_uri = urljoin(original_url, uri)
                return f'URI="{proxy_base}{quote(abs_uri, safe="")}"'
            line = _re_proxy.sub(r'URI="([^"]+)"', replace_uri, line)
            out.append(line)
        elif stripped and not stripped.startswith("#"):
            # Segment or sub-manifest URL
            abs_url = urljoin(original_url, stripped)
            out.append(f"{proxy_base}{quote(abs_url, safe='')}")
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# yt-dlp extraction: scans JWPlayer/Video.js configs without a browser
# ---------------------------------------------------------------------------
def _ytdlp_extract_m3u8(embed_url: str) -> str | None:
    import yt_dlp as _yt
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 8,
        "http_headers": {
            "Referer": "https://onhockey.tv/",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    }
    try:
        with _yt.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(embed_url, download=False)
        if not info:
            return None
        formats = info.get("formats") or []
        hls = [f for f in formats
               if f.get("protocol") in ("m3u8", "m3u8_native")
               or f.get("url", "").endswith(".m3u8")]
        if hls:
            best = max(hls, key=lambda f: f.get("tbr") or 0)
            return best.get("url")
        return info.get("url") or info.get("manifest_url")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Playwright extraction: intercepts the real m3u8 network request
# ---------------------------------------------------------------------------
_pw_browser = None
_pw_lock = None


async def _get_pw_browser():
    global _pw_browser, _pw_lock
    import asyncio as _aio
    if _pw_lock is None:
        _pw_lock = _aio.Lock()
    async with _pw_lock:
        if _pw_browser is None or not _pw_browser.is_connected():
            from playwright.async_api import async_playwright
            _pw = await async_playwright().start()
            _pw_browser = await _pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox", "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage", "--disable-gpu",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
    return _pw_browser


async def _playwright_extract_m3u8(embed_url: str) -> str | None:
    import asyncio as _aio
    try:
        browser = await _get_pw_browser()
        ctx = await browser.new_context(
            extra_http_headers={"Referer": "https://onhockey.tv/"},
            # Spoof a real desktop viewport so players don't hide behind mobile layouts
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()

        # Block ads/trackers so the player initialises faster
        await page.route(
            _re_proxy.compile(
                r"(doubleclick|googlesyndication|adservice|amazon-adsystem"
                r"|adnxs|adsafeprotected|moatads|scorecardresearch"
                r"|outbrain|taboola|googletag|pubmatic|openx|rubiconproject"
                r"|criteo|adsystem|ad\.net|popads|trafficjunky)"
            ),
            lambda route, _: route.abort(),
        )

        found: list[str] = []
        done = _aio.Event()

        # Ad CDN patterns — skip these m3u8 requests, they're pre-roll ads
        _AD_PATTERNS = (
            "doubleclick", "2mdn.net", "googlesyndication", "adservice",
            "adnxs", "rubiconproject", "openx", "pubmatic", "criteo",
            "/ads/", "dai.google", "ima3", "videoplayback",
            "adserver", "adsystem", "ad.net",
        )

        def _is_ad(u: str) -> bool:
            ul = u.lower()
            return any(p in ul for p in _AD_PATTERNS)

        def _on_request(req):
            u = req.url
            if (".m3u8" in u or ".mpd" in u) and not _is_ad(u) and not done.is_set():
                found.append(u)
                done.set()

        page.on("request", _on_request)

        try:
            await page.goto(embed_url, wait_until="domcontentloaded", timeout=15_000)
        except Exception:
            pass

        # Short wait — m3u8 may fire immediately after DOM load
        try:
            await _aio.wait_for(done.wait(), timeout=3.0)
        except _aio.TimeoutError:
            pass

        if not found:
            # Many players need a click to start. Try common play-button selectors.
            _play_selectors = [
                "button.jw-icon-display",   # JW Player
                ".vjs-big-play-button",      # Video.js
                ".plyr__control--overlaid",  # Plyr
                "[class*='play'][class*='btn']",
                "[class*='play-btn']",
                "[class*='playBtn']",
                "[aria-label*='play' i]",
                "[title*='play' i]",
                "video",                     # bare <video> element itself
            ]
            for sel in _play_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.click(timeout=1_500, force=True)
                        break
                except Exception:
                    continue

            # Dismiss common consent / overlay popups that block the player
            _dismiss = [
                "[class*='consent'] button",
                "[class*='gdpr'] button",
                "[class*='cookie'] button",
                "[class*='close']",
                "[class*='overlay'] [class*='close']",
            ]
            for sel in _dismiss:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.click(timeout=500, force=True)
                except Exception:
                    continue

            # Wait longer after clicking
            try:
                await _aio.wait_for(done.wait(), timeout=12.0)
            except _aio.TimeoutError:
                pass

        await page.close()
        await ctx.close()
        return found[0] if found else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-domain static HTML extractors (no browser needed)
# ---------------------------------------------------------------------------

def _extract_apl396(html: str, page_url: str) -> str | None:
    """emb.apl396.me — stream URL is in pl.init('//cdn/hls/streamaXXX/index.m3u8?cst=...')"""
    m = _re_proxy.search(r"pl\.init\(['\"]([^'\"]+)['\"]", html)
    if m:
        raw = m.group(1)
        if raw.startswith("//"):
            raw = "https:" + raw
        return raw
    return None


def _extract_streamfree(html: str, page_url: str) -> str | None:
    """streamfree.app — token dict + slug-based URL in page HTML."""
    import json as _json
    m = _re_proxy.search(r'const _0x\s*=\s*(\{[^;]+\})\s*;', html)
    if not m:
        return None
    try:
        tokens = _json.loads(m.group(1))
    except Exception:
        return None
    # Extract slug from URL path /embed/[sport]/[slug]
    slug_m = _re_proxy.search(r'/embed/[^/]+/([^/?#]+)', page_url)
    if not slug_m:
        return None
    slug = slug_m.group(1)
    for quality in ("720p", "1080p", "540p", "480p"):
        p = tokens.get(quality)
        if p and p.get("_t") and p.get("_e") and p.get("_n"):
            return (
                f"https://streamfree.app/live/{slug}{quality}/index.m3u8"
                f"?_t={p['_t']}&_e={p['_e']}&_n={p['_n']}"
            )
    return None


async def _html_extract_m3u8(embed_url: str) -> str | None:
    """Try fast static-HTML extraction for known embed domains."""
    import httpx as _httpx
    _hdr = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://onhockey.tv/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with _httpx.AsyncClient(timeout=8, follow_redirects=True) as hx:
            r = await hx.get(embed_url, headers=_hdr)
        html = r.text
    except Exception:
        return None

    parsed = urlparse(embed_url)
    host = parsed.netloc.lower()

    if "apl396.me" in host:
        return _extract_apl396(html, embed_url)
    if "streamfree.app" in host:
        return _extract_streamfree(html, embed_url)
    return None


# ---------------------------------------------------------------------------
# Ad-stripped embed proxy
# Ad networks used by these embed pages (removed from proxied HTML):
#   - aclib / aclib.runPop — popunder network
#   - acscdn.com / acacdn.com — ad delivery CDNs
#   - The hidden /ad.html iframe appended every 10 minutes
# ---------------------------------------------------------------------------

_AD_STRIP_PATTERNS = [
    # aclib script tag entirely (the src that loads the popunder library)
    _re_proxy.compile(
        r'<script[^>]+src=["\'][^"\']*aclib[^"\']*["\'][^>]*>\s*</script>',
        _re_proxy.DOTALL | _re_proxy.IGNORECASE,
    ),
    # aclib popunder call (inline)
    _re_proxy.compile(r'aclib\.runPop\([^)]*\);?', _re_proxy.DOTALL),
    # ad script tags (acscdn, acacdn, adcash, histats, awistats, etc.)
    _re_proxy.compile(
        r'<script[^>]+src=["\'][^"\']*(?:acscdn|acacdn|adcash|histats|awistats'
        r'|adnimation|bahaymulley|adexchange|trafficjunky|popads|adexchangeclear'
        r'|yuntracking|helpless\.click)[^"\']*["\'][^>]*>.*?</script>',
        _re_proxy.DOTALL | _re_proxy.IGNORECASE,
    ),
    _re_proxy.compile(
        r'<script[^>]+(?:acscdn|acacdn|adcash|histats|awistats|adexchangeclear)[^>]*></script>',
        _re_proxy.IGNORECASE,
    ),
    # Hidden /ad.html iframe injection
    _re_proxy.compile(r"src=['\"][^'\"]*?/ad\.html[^'\"]*['\"]", _re_proxy.IGNORECASE),
    # The setTimeout ad iframe injection code block
    _re_proxy.compile(
        r"\(\(\)\s*=>\s*\{[^}]*insertAdjacentHTML[^}]*/ad\.html[^}]*\}\)\(\);?",
        _re_proxy.DOTALL,
    ),
]

# Domains that use obfuscated JS players — skip yt-dlp / Playwright entirely and
# serve them directly via the ad-stripped embed proxy.  Attempting headless
# extraction on these wastes 30-45 s and always fails due to bot-detection.
_EMBED_ONLY_DOMAINS = {
    "embedsports.top", "embedsports.me", "dlstreams.top", "wikisport.club",
    "vuen.link", "gopst.link", "dabac.link", "zenoz.link",
    "lovecdn.ru", "viewembed.ru", "streams.center", "embedhd.org",
    "yuntracking.co", "helpless.click", "onhockey.tv",
    "sportscentral.io", "sportsonline.to", "720pstream.me",
    # Additional embed-page hosts seen in onhockey.tv channel links
    "lovetier.bz", "cdn-live.tv", "antenasport.org", "livesport.ws",
    "sportlemon.tv", "streambtw.com", "livetv.sx", "livetv.ru",
    "nhl66.ir", "nhltv.is", "nhl.com", "espn.com",
}


def _strip_ads_from_html(html: str) -> str:
    for pat in _AD_STRIP_PATTERNS:
        html = pat.sub("", html)
    return html


@app.get("/stream-embed")
async def stream_embed(url: str, request: Request) -> FastResponse:  # noqa: ARG001
    """
    Serve a wrapper page that embeds the original URL in a sandboxed iframe.

    WHY THE WRAPPER APPROACH
    ────────────────────────
    Proxying the embed HTML and re-serving it from our domain breaks every
    origin-sensitive player: JWPlayer / Video.js / custom players all check
    window.location.hostname and refuse to start when they see the wrong
    domain.  onhockey.tv works because the embed loads at its native origin.

    Instead we return a thin HTML wrapper that:
      • Loads the embed URL directly in a sandboxed <iframe> (correct origin)
      • Omits allow-popups → browser silently drops all window.open() calls
      • Overrides window.open in the outer frame as a second layer
    """
    cors = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Cache-Control": "no-cache",
    }

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return FastResponse(content="Invalid url", status_code=400, headers=cors)
        safe_url = url.replace('"', "%22").replace("'", "%27")
    except Exception:
        return FastResponse(content="Invalid url", status_code=400, headers=cors)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  html,body{{width:100%;height:100%;background:#000;overflow:hidden}}
  iframe{{position:absolute;inset:0;width:100%;height:100%;border:0}}
</style>
<script>
(function(){{
  var _noop = function(){{ return {{ closed:true, focus:function(){{}} }}; }};
  try{{ window.open = _noop; }}catch(e){{}}
  window.addEventListener('blur', function(){{ setTimeout(function(){{ window.focus&&window.focus(); }}, 0); }}, true);
}})();
</script>
</head>
<body>
<iframe
  src="{safe_url}"
  allow="autoplay; fullscreen; picture-in-picture; encrypted-media"
  allowfullscreen
  referrerpolicy="no-referrer-when-downgrade"
></iframe>
</body>
</html>"""

    return FastResponse(content=html, media_type="text/html", headers=cors)


# ---------------------------------------------------------------------------
# Resolution cache — avoids re-running yt-dlp/Playwright on repeat visits
# ---------------------------------------------------------------------------
import time as _time

_resolve_cache: dict[str, tuple[str | None, float]] = {}  # embed_url -> (m3u8_url|None, ts)
_RESOLVE_TTL = 3600  # 1 hour — CDN tokens typically last 1–2 h; short enough to avoid stale caches


async def _resolve_embed(embed_url: str) -> str | None:
    """Return the direct m3u8 URL for an embed page, using cache.

    Returns None when the URL should be served via the embed proxy instead
    (caller checks _EMBED_ONLY_DOMAINS before calling this).
    """
    cached = _resolve_cache.get(embed_url)
    if cached and (_time.monotonic() - cached[1]) < _RESOLVE_TTL:
        return cached[0]

    host = urlparse(embed_url).netloc.lstrip("www.")
    # Known bot-detected domains — skip slow extractors, go straight to embed proxy
    if host in _EMBED_ONLY_DOMAINS:
        return None

    # Layer 0: fast static HTML extraction (apl396.me, streamfree.app)
    result = await _html_extract_m3u8(embed_url)

    # Layer 1: yt-dlp (JWPlayer config detection)
    if not result:
        import asyncio as _aio
        result = await _aio.to_thread(_ytdlp_extract_m3u8, embed_url)

    # Layer 2: Playwright (headless browser, clicks play, intercepts request)
    if not result:
        result = await _playwright_extract_m3u8(embed_url)

    # Only cache successes — failures should be retried on next request
    if result:
        _resolve_cache[embed_url] = (result, _time.monotonic())
    return result


@app.get("/stream-resolve")
async def stream_resolve(url: str) -> dict:
    """Resolve an embed page URL to a direct m3u8 URL (with caching)."""
    # Fast path: already a direct m3u8
    lower = url.split("?")[0].lower()
    if lower.endswith(".m3u8") or lower.endswith(".mpd"):
        return {"type": "m3u8", "url": url}

    # Try all extraction layers
    m3u8 = await _resolve_embed(url)
    if m3u8:
        return {"type": "m3u8", "url": m3u8}

    # All extractors failed — serve as direct embed (sandboxed by frontend)
    return {"type": "embed", "url": url}


@app.get("/stream-proxy")
async def stream_proxy(url: str, request: Request) -> FastResponse:
    """Proxy an HLS stream URL, rewriting internal URLs to also go through proxy."""
    import httpx as _httpx

    # Build the proxy base URL so rewritten segment URLs point back through the
    # same origin the browser used to load the manifest.
    # When running behind Vercel (or any reverse proxy), X-Forwarded-Host carries
    # the public hostname (e.g. gretzky-dashboard.vercel.app).  Segment URLs must
    # use that host + the /api/ prefix so they route back through the Next.js rewrite
    # — otherwise the browser gets cross-origin URLs that CORS blocks.
    forwarded_host = request.headers.get("x-forwarded-host", "")
    if forwarded_host:
        forwarded_proto = request.headers.get("x-forwarded-proto", "https")
        proxy_base = f"{forwarded_proto}://{forwarded_host}/api/stream-proxy?url="
    else:
        base = str(request.base_url).rstrip("/")
        proxy_base = f"{base}/stream-proxy?url="

    _stream_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://onhockey.tv/",
        "Origin": "https://onhockey.tv",
    }

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Cache-Control": "no-cache",
    }

    try:
        async with _httpx.AsyncClient(timeout=15, follow_redirects=True) as hx:
            resp = await hx.get(url, headers=_stream_headers)

        content_type = resp.headers.get("content-type", "")
        body = resp.content

        # If it's an m3u8 manifest, rewrite internal URLs
        is_m3u8 = (
            "mpegurl" in content_type.lower()
            or url.split("?")[0].endswith(".m3u8")
            or body[:7] == b"#EXTM3U"
        )

        if is_m3u8:
            text = body.decode("utf-8", errors="replace")
            text = _rewrite_m3u8(text, url, proxy_base)
            return FastResponse(
                content=text,
                media_type="application/vnd.apple.mpegurl",
                headers=cors_headers,
            )

        # HTML embed page — extract the real m3u8 via three escalating layers
        is_html = "text/html" in content_type.lower() or body[:15].lower().lstrip().startswith(b"<!doctype")
        if is_html:
            import asyncio as _aio

            # Layer 1: yt-dlp (JWPlayer/Video.js config detection, no browser, ~2s)
            real_m3u8 = await _aio.to_thread(_ytdlp_extract_m3u8, url)

            # Layer 2: Playwright network interception (headless browser, ~8s)
            if not real_m3u8:
                real_m3u8 = await _playwright_extract_m3u8(url)

            if real_m3u8:
                if not real_m3u8.startswith("http"):
                    real_m3u8 = urljoin(url, real_m3u8)
                async with _httpx.AsyncClient(timeout=15, follow_redirects=True) as hx2:
                    m3u8_resp = await hx2.get(real_m3u8, headers=_stream_headers)
                m3u8_text = m3u8_resp.content.decode("utf-8", errors="replace")
                m3u8_text = _rewrite_m3u8(m3u8_text, real_m3u8, proxy_base)
                return FastResponse(
                    content=m3u8_text,
                    media_type="application/vnd.apple.mpegurl",
                    headers=cors_headers,
                )
            # Layer 3: iframe fallback — tell the frontend to embed the page directly
            import json as _json
            return FastResponse(
                content=_json.dumps({"fallback_url": url}),
                status_code=422,
                media_type="application/json",
                headers={**cors_headers, "Content-Type": "application/json"},
            )

        # Binary segment (.ts, .aac, etc.) — stream through as-is
        return FastResponse(
            content=body,
            media_type=content_type or "application/octet-stream",
            headers=cors_headers,
        )

    except Exception as exc:
        return FastResponse(
            content=str(exc),
            status_code=502,
            headers=cors_headers,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Feature 16.7 — CV Worker Service
# ══════════════════════════════════════════════════════════════════════════════
#
# Endpoints:
#   POST /api/cv/start/{game_id}      body: CvStartRequest
#   POST /api/cv/stop/{game_id}
#   GET  /api/cv/status
#   GET  /api/cv/positions/{game_id}
#
# The CvWorkerManager singleton lives on app.state.cv and is initialised on
# startup.  Workers run as daemon threads — they are stopped on shutdown.

from pydantic import BaseModel as _BaseModel


class CvStartRequest(_BaseModel):
    """Request body for POST /api/cv/start/{game_id}."""
    hls_url: str
    fps: float = 6.0
    arena: str | None = None   # NHL API arena name; used for homography cache


def _cv_accumulator():
    """Return the shared TrackingAccumulator, initialising lazily (Feature 16.8)."""
    if not hasattr(app.state, "cv_acc"):
        from data.tracking_accumulator import TrackingAccumulator
        app.state.cv_acc = TrackingAccumulator()
    return app.state.cv_acc


def _cv_manager():
    """Return the CvWorkerManager from app state, initialising lazily."""
    if not hasattr(app.state, "cv"):
        from data.cv_tracker import CvWorkerManager
        app.state.cv = CvWorkerManager(accumulator=_cv_accumulator())
    return app.state.cv


@app.on_event("shutdown")
async def _cv_shutdown():
    """Stop all CV workers and close the accumulator cleanly when the server exits."""
    if hasattr(app.state, "cv"):
        app.state.cv.shutdown()
    if hasattr(app.state, "cv_acc"):
        app.state.cv_acc.close()


@app.post("/api/cv/start/{game_id}")
async def cv_start(game_id: int, body: CvStartRequest):
    """Start (or restart) the CV tracking worker for *game_id*.

    The worker launches immediately in a background thread and returns
    ``{"status": "starting", ...}`` — polling ``GET /api/cv/status`` or
    ``GET /api/cv/positions/{game_id}`` for live data.

    Requires ffmpeg on PATH and trained model weights (run Phase 16 training
    scripts first).  Returns 503 with an error message when models are missing.
    """
    try:
        mgr = _cv_manager()
        status = mgr.start(
            game_id=game_id,
            hls_url=body.hls_url,
            fps=body.fps,
            arena=body.arena,
        )
        return status.to_dict()
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/cv/stop/{game_id}")
async def cv_stop(game_id: int):
    """Stop the CV tracking worker for *game_id*.

    Returns ``{"stopped": true}`` if a worker was running, or
    ``{"stopped": false}`` when no worker was found for that game.
    """
    try:
        mgr = _cv_manager()
        stopped = mgr.stop(game_id)
        return {"stopped": stopped, "game_id": game_id}
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/cv/status")
async def cv_status():
    """Return status for all active CV workers.

    Each entry includes ``game_id``, ``state``, ``frames_processed``,
    ``tracks_active``, ``started_at``, ``last_frame_at``, and ``error``.
    """
    try:
        mgr = _cv_manager()
        return {"workers": [s.to_dict() for s in mgr.all_statuses()]}
    except Exception as exc:
        return {"workers": [], "error": str(exc)}


@app.get("/api/cv/positions/{game_id}")
async def cv_positions(game_id: int):
    """Return the latest tracked positions for *game_id*.

    Each element in ``tracks`` has:
      ``track_id``, ``class_name``, ``cx``, ``cy``, ``vx``, ``vy``,
      ``w``, ``h``, ``rink_x``, ``rink_y``,
      ``jersey_number``, ``jersey_locked``, ``team``, ``frame_ts``.

    ``rink_x`` / ``rink_y`` are ``null`` until the rink keypoint detector
    establishes the homography (typically within the first 30 frames).

    Returns ``{"game_id": id, "tracks": [], "error": "no worker"}`` when no
    worker is running for this game.
    """
    try:
        mgr = _cv_manager()
        positions = mgr.get_positions(game_id)
        if positions is None:
            return {"game_id": game_id, "tracks": [], "error": "no worker running"}
        return {
            "game_id": game_id,
            "tracks": [p.to_dict() for p in positions],
            "count": len(positions),
        }
    except Exception as exc:
        return {"game_id": game_id, "tracks": [], "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# Feature 16.8 — Tracking data accumulator
# ══════════════════════════════════════════════════════════════════════════════
#
# Endpoints:
#   GET  /api/cv/tracking/{game_id}     — recent tracking frames for a game
#   GET  /api/cv/shot-labels            — shot_tracking_labels view (training data)
#   GET  /api/cv/shot-labels/{game_id}  — shot labels for one game
#   POST /api/cv/shot-event/{game_id}   — inject a PBP shot event manually
#   GET  /api/cv/tracking-games         — list games with tracking data


class CvShotEventRequest(_BaseModel):
    """Body for POST /api/cv/shot-event/{game_id} — mirrors ShotEvent fields."""
    event_id: int
    period: int = 1
    time_in_period: str = "00:00"
    time_in_period_secs: int = 0
    shooter_id: int = 0
    shooter_team: str = ""
    shot_result: str = "on_goal"   # 'on_goal' | 'missed' | 'blocked'
    is_goal: bool = False
    shot_x: float | None = None
    shot_y: float | None = None
    home_team: str = ""
    away_team: str = ""


@app.get("/api/cv/tracking/{game_id}")
async def cv_tracking_frames(
    game_id: int,
    limit: int = Query(default=5_000, ge=1, le=50_000),
):
    """Return recent tracking frames for *game_id* from the DuckDB accumulator.

    Results are ordered latest-first (frame_seq DESC).  Each row contains the
    full positional record including rink feet and jersey identity.
    """
    try:
        acc = _cv_accumulator()
        df = acc.tracking_frames(game_id=game_id, limit=limit)
        return {
            "game_id": game_id,
            "rows": len(df),
            "frames": acc.frame_count(game_id),
            "data": df.to_dicts(),
        }
    except Exception as exc:
        return {"game_id": game_id, "rows": 0, "frames": 0, "data": [], "error": str(exc)}


@app.get("/cv/tracking/{game_id}/frames")
async def cv_tracking_replay_frames(
    game_id: int,
    from_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=1000),
):
    """Return CV tracking frames in ascending frame_seq order for replay.

    Designed to be fetched in chunks as the client steps through playback.
    Returns lightweight per-player fields only (no full snapshot columns).
    """
    try:
        acc = _cv_accumulator()
        if acc is None:
            return {"game_id": game_id, "frames": [], "next_seq": None, "total_frames": 0}

        # Total distinct frames available for this game
        total_df = acc.query(
            "SELECT COUNT(DISTINCT frame_seq) AS n FROM tracking_frames WHERE game_id = ?",
            [game_id],
        )
        total_frames = int(total_df["n"][0]) if not total_df.is_empty() else 0

        if total_frames == 0:
            return {"game_id": game_id, "frames": [], "next_seq": None, "total_frames": 0}

        # Fetch next chunk — ASC order, from_seq inclusive
        # Max rows: limit frames × 15 players/frame
        df = acc.query(
            """
            SELECT frame_seq, timestamp_ms, track_id, team, class_name,
                   rink_x, rink_y, vx, vy, jersey_number, jersey_locked,
                   cx, cy, w, h
            FROM tracking_frames
            WHERE game_id = ? AND frame_seq >= ?
            ORDER BY frame_seq ASC, track_id ASC
            LIMIT ?
            """,
            [game_id, from_seq, limit * 15],
        )

        if df.is_empty():
            return {"game_id": game_id, "frames": [], "next_seq": None, "total_frames": total_frames}

        # Group by frame_seq, build player lists
        import math as _m
        frames_out: list[dict] = []
        for (fseq,), group in df.group_by(["frame_seq"], maintain_order=True):
            rows = group.to_dicts()
            # Find puck position for proximity computation
            puck_rx_r = next(
                (float(r["rink_x"]) for r in rows if r.get("class_name") == "puck" and r.get("rink_x") is not None),
                None,
            )
            puck_ry_r = next(
                (float(r["rink_y"]) for r in rows if r.get("class_name") == "puck" and r.get("rink_y") is not None),
                None,
            )
            players = []
            for r in rows:
                vx = float(r.get("vx") or 0)
                vy = float(r.get("vy") or 0)
                spd = _m.sqrt(vx * vx + vy * vy)
                cx = float(r.get("cx") or 640)
                cy = float(r.get("cy") or 360)
                w  = float(r.get("w")  or 40)
                h  = float(r.get("h")  or 60)
                rx = float(r.get("rink_x") or 0)
                ry = float(r.get("rink_y") or 0)
                if puck_rx_r is not None and r.get("class_name") != "puck":
                    dx = rx - puck_rx_r; dy = ry - (puck_ry_r or 0.0)
                    dtp_r: float | None = round(_m.sqrt(dx * dx + dy * dy), 1)
                else:
                    dtp_r = None
                players.append({
                    "track_id":      int(r["track_id"]),
                    "team":          r["team"] or "",
                    "class_name":    r["class_name"] or "",
                    "x":             r["rink_x"],
                    "y":             r["rink_y"],
                    "vx":            round(vx, 3),
                    "vy":            round(vy, 3),
                    "speed_kmh":     round(spd * 6.0 * 1.09728, 1),
                    "in_zone":       abs(rx) > 36,
                    "dist_to_puck":  dtp_r,
                    "is_burst":      False,
                    "jersey_number": r["jersey_number"],
                    "jersey_locked": bool(r.get("jersey_locked")),
                    "cx_n":          round(cx / 1280, 4),
                    "cy_n":          round(cy / 720,  4),
                    "w_n":           round(w  / 1280, 4),
                    "h_n":           round(h  / 720,  4),
                })
            frames_out.append({
                "frame_seq":    int(fseq),
                "timestamp_ms": int(rows[0]["timestamp_ms"]) if rows else 0,
                "players":      players,
            })

        next_seq = frames_out[-1]["frame_seq"] + 1 if frames_out else None
        return {
            "game_id":      game_id,
            "frames":       frames_out,
            "next_seq":     next_seq,
            "total_frames": total_frames,
        }
    except Exception as exc:
        return {"game_id": game_id, "frames": [], "next_seq": None, "total_frames": 0, "error": str(exc)}


@app.get("/api/cv/shot-labels")
async def cv_shot_labels_all(
    goals_only: bool = Query(default=False),
    limit: int = Query(default=10_000, ge=1, le=100_000),
):
    """Return rows from the ``shot_tracking_labels`` view (all games).

    This is the auto-labeled training dataset for Feature 16.10.  Each row is
    one player's position at the moment of a NHL PBP shot event, with
    ``is_goal`` as the label.

    Args:
        goals_only: Filter to goal events only.
        limit:      Max rows returned (default 10 000).
    """
    try:
        acc = _cv_accumulator()
        df = acc.shot_labels(goals_only=goals_only, limit=limit)
        return {
            "rows": len(df),
            "shots": acc.shot_count(),
            "data": df.to_dicts(),
        }
    except Exception as exc:
        return {"rows": 0, "shots": 0, "data": [], "error": str(exc)}


@app.get("/api/cv/shot-labels/{game_id}")
async def cv_shot_labels_game(
    game_id: int,
    goals_only: bool = Query(default=False),
    limit: int = Query(default=10_000, ge=1, le=100_000),
):
    """Return ``shot_tracking_labels`` rows for a single *game_id*."""
    try:
        acc = _cv_accumulator()
        df = acc.shot_labels(game_id=game_id, goals_only=goals_only, limit=limit)
        return {
            "game_id": game_id,
            "rows": len(df),
            "shots": acc.shot_count(game_id),
            "data": df.to_dicts(),
        }
    except Exception as exc:
        return {"game_id": game_id, "rows": 0, "shots": 0, "data": [], "error": str(exc)}


@app.post("/api/cv/shot-event/{game_id}")
async def cv_shot_event(game_id: int, body: CvShotEventRequest):
    """Manually inject a NHL PBP shot event and snapshot current tracking positions.

    In production this is called automatically by the PBP watcher.  This
    endpoint exists for testing and for manual injection when the PBP watcher
    is not running.

    Returns the ``snapshot_id`` of the written record.
    """
    try:
        from data.tracking_accumulator import ShotEvent
        event = ShotEvent(
            game_id=game_id,
            event_id=body.event_id,
            period=body.period,
            time_in_period=body.time_in_period,
            time_in_period_secs=body.time_in_period_secs,
            shooter_id=body.shooter_id,
            shooter_team=body.shooter_team,
            shot_result=body.shot_result,
            is_goal=body.is_goal,
            shot_x=body.shot_x,
            shot_y=body.shot_y,
            home_team=body.home_team,
            away_team=body.away_team,
        )
        mgr = _cv_manager()
        snapshot_id = mgr.notify_shot_event(event, game_id=game_id)
        if snapshot_id is None:
            # No accumulator or manager; fall back to writing directly
            acc = _cv_accumulator()
            positions = mgr.get_positions(game_id) or []
            snapshot_id = acc.notify_shot_event(event, positions)
        return {"snapshot_id": snapshot_id, "game_id": game_id}
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/cv/tracking-games")
async def cv_tracking_games():
    """List all game_ids that have tracking data in the accumulator."""
    try:
        acc = _cv_accumulator()
        games = acc.games_with_data()
        return {"games": games, "count": len(games)}
    except Exception as exc:
        return {"games": [], "count": 0, "error": str(exc)}


# ── Feature 16.11 — Homography cache coverage ─────────────────────────────
#
#   GET  /api/cv/homography/cache   — 32-arena calibration coverage + quality
#
# Returns calibration status for all 32 NHL arenas (which are cached, which
# are missing, RMS reprojection error, when each was last calibrated).

@app.get("/api/cv/homography/cache")
async def cv_homography_cache():
    """Return calibration coverage across all 32 NHL arenas.

    Reads the on-disk homography cache (``data/cv_training/homography_cache/``)
    and reports which arenas have cached calibrations and their quality.

    Response shape::

        {
          "total":        32,
          "calibrated":   int,
          "missing":      int,
          "coverage_pct": float,
          "arenas": [
            {
              "team":                  "MTL",
              "arena":                 "Bell Centre",
              "slug":                  "bell_centre",
              "calibrated":            true,
              "calibrated_at":         "2026-04-03T20:00:00Z",
              "game_id":               2024021000,
              "reprojection_error_ft": 1.23,
              "keypoints_used":        6,
              "calibration_count":     4
            },
            ...
          ]
        }
    """
    try:
        from models.cv.homography import get_cache_coverage
        return get_cache_coverage()
    except Exception as exc:
        return {
            "total":        32,
            "calibrated":   0,
            "missing":      32,
            "coverage_pct": 0.0,
            "arenas":       [],
            "error":        str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Feature 16.9 — WebSocket position feed
# ══════════════════════════════════════════════════════════════════════════════
#
# Endpoint:
#   WS  /ws/game/{game_id}/positions      — ~15 fps position stream
#
# Payload shape (JSON per frame)::
#
#   {
#     "game_id": int,
#     "players": [
#       {
#         "track_id": int,
#         "team": str,            # 3-letter NHL code or ""
#         "class_name": str,      # "player_home" | "player_away" |
#                                 # "goalie_home" | "goalie_away" | "referee"
#         "x": float | null,      # rink feet (+100 = home end)
#         "y": float | null,      # rink feet (+42.5 = right boards)
#         "vx": float,            # pixel/frame velocity x (direction indicator)
#         "vy": float,            # pixel/frame velocity y
#         "speed_ms": float,      # pixel/frame speed magnitude
#         "is_burst": bool,       # speed > 1.8× rolling 15-frame baseline
#         "jersey_number": int | null,
#         "jersey_locked": bool
#       }, ...
#     ],
#     "puck": {
#       "x": float | null, "y": float | null,
#       "speed_ms": float, "occluded": bool
#     } | null
#   }
#
# Sends {"players": [], "puck": null} when no CV worker is running.
# Client should reconnect on disconnect (exponential back-off recommended).

import math as _math

from fastapi import WebSocket as _WebSocket, WebSocketDisconnect as _WebSocketDisconnect


@app.websocket("/ws/game/{game_id}/positions")
async def ws_positions(websocket: _WebSocket, game_id: int):
    """Stream live CV-tracked positions for *game_id* at ~15 fps.

    Reads from the running ``CvWorkerManager``; falls back to an empty
    ``{"players": [], "puck": null}`` payload when no worker is active.
    The outer ``try/except`` swallows all errors so the API server never
    crashes on a bad client connection.
    """
    await websocket.accept()

    _HISTORY_LEN = 15           # frames of speed history per track (~1 s at 15 fps)
    _BURST_MULT  = 1.8          # speed multiplier above baseline to trigger burst
    _BURST_MIN   = 0.5          # px/frame: ignore burst when nearly stationary
    _FT_FRAME_TO_MS = 6.0 * 0.3048  # ft/frame @ 6fps → m/s
    speed_history:  dict[int, list[float]] = {}
    prev_speed_ms:  dict[int, float]       = {}   # for acceleration

    try:
        while True:
            # ── Collect positions from CV worker ──────────────────────────────
            try:
                positions = _cv_manager().get_positions(game_id) or []
            except Exception:
                positions = []

            players: list[dict] = []
            puck:    dict | None = None

            # Pass 1: find puck position for proximity computation
            puck_rx: float | None = None
            puck_ry: float | None = None
            for tp in positions:
                if tp.class_name == "puck" and tp.rink_x is not None and tp.rink_y is not None:
                    puck_rx = tp.rink_x
                    puck_ry = tp.rink_y
                    break

            for tp in positions:
                spd = _math.sqrt(tp.vx ** 2 + tp.vy ** 2)

                # Rolling baseline for burst detection
                hist = speed_history.setdefault(tp.track_id, [])
                hist.append(spd)
                if len(hist) > _HISTORY_LEN:
                    hist.pop(0)
                baseline  = sum(hist) / len(hist) if hist else spd
                is_burst  = baseline >= _BURST_MIN and spd >= _BURST_MULT * baseline

                if tp.class_name == "puck":
                    puck = {
                        "x":        tp.rink_x,
                        "y":        tp.rink_y,
                        "speed_ms": round(spd, 2),
                        "occluded": tp.occluded,
                    }
                else:
                    # Feature 16.18: pressure score from observer if running
                    pressure_score: float = 0.0
                    try:
                        obs = getattr(app.state, "observer", None)
                        if obs is not None:
                            game_obs = obs.get(game_id)
                            if game_obs is not None:
                                pressure_score = game_obs._pressure.pressure_score(tp.track_id)
                    except Exception:
                        pass

                    # speed in m/s; acceleration from previous frame
                    speed_ms_val = spd * _FT_FRAME_TO_MS
                    prev_ms = prev_speed_ms.get(tp.track_id, speed_ms_val)
                    # Δv/Δt — WS loop runs at ~15fps so Δt ≈ 1/15 s
                    accel_ms2 = round((speed_ms_val - prev_ms) * 15.0, 2)
                    prev_speed_ms[tp.track_id] = speed_ms_val

                    # speed_kmh: rink-ft/frame × 6fps × 1.09728 (ft/s → km/h)
                    speed_kmh = round(spd * 6.0 * 1.09728, 1)

                    # Rink zone: NHL blue lines are ≈36 ft from centre (64 ft from end boards)
                    rx = tp.rink_x or 0.0
                    in_zone = abs(rx) > 36

                    # Distance to puck (feet)
                    if puck_rx is not None and tp.rink_x is not None and tp.rink_y is not None:
                        dx = tp.rink_x - puck_rx
                        dy = tp.rink_y - (puck_ry or 0.0)
                        dist_to_puck: float | None = round(_math.sqrt(dx * dx + dy * dy), 1)
                    else:
                        dist_to_puck = None

                    players.append({
                        "track_id":      tp.track_id,
                        "team":          tp.team or "",
                        "class_name":    tp.class_name or "",
                        "x":             tp.rink_x,
                        "y":             tp.rink_y,
                        "vx":            round(tp.vx, 3),
                        "vy":            round(tp.vy, 3),
                        "speed_ms":      round(spd, 2),
                        "speed_kmh":     speed_kmh,
                        "accel_ms2":     accel_ms2,
                        "in_zone":       in_zone,
                        "dist_to_puck":  dist_to_puck,
                        "is_burst":      is_burst,
                        "jersey_number": tp.jersey_number,
                        "jersey_locked": tp.jersey_locked,
                        "pressure":      round(pressure_score, 1),
                        # Normalised pixel bbox (0-1, based on 1280×720 pipeline frame)
                        "cx_n":          round(tp.cx / 1280, 4),
                        "cy_n":          round(tp.cy / 720,  4),
                        "w_n":           round(tp.w  / 1280, 4),
                        "h_n":           round(tp.h  / 720,  4),
                    })

            # ── Feature 16.16: possession state ───────────────────────────
            possession: dict | None = None
            try:
                obs = getattr(app.state, "observer", None)
                if obs is not None:
                    game_obs = obs.get(game_id)
                    if game_obs is not None:
                        poss_state = game_obs._possession.current
                        if poss_state.is_possessed():
                            possession = poss_state.to_dict()
            except Exception:
                pass

            await websocket.send_json({
                "game_id":    game_id,
                "players":    players,
                "puck":       puck,
                "possession": possession,
            })
            await asyncio.sleep(1 / 15)   # ~15 fps

    except _WebSocketDisconnect:
        pass
    except Exception:
        # Swallow connection-reset / encoding errors; client will reconnect
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Feature 16.13 — AI Game Observer
# ══════════════════════════════════════════════════════════════════════════════
#
# Endpoints:
#   POST /api/observer/start/{game_id}   body: ObserverStartRequest
#   POST /api/observer/stop/{game_id}
#   GET  /api/observer/status
#   GET  /api/observer/events/{game_id}
#
# The AiObserverManager singleton lives on app.state.observer.
# It shares the CvWorkerManager and WhizFeed from app.state.


class ObserverStartRequest(_BaseModel):
    """Request body for POST /api/observer/start/{game_id}."""
    game_date: str = ""
    away_team: str = ""
    home_team: str = ""


def _observer_manager():
    """Return the AiObserverManager from app state, initialising lazily."""
    if not hasattr(app.state, "observer"):
        from data.cv_ai_observer import AiObserverManager
        from models.whiz_feed import WhizFeed
        feed = WhizFeed()
        app.state.observer = AiObserverManager(
            cv_manager=_cv_manager(),
            feed=feed,
        )
    return app.state.observer


@app.on_event("shutdown")
async def _observer_shutdown():
    """Stop all AI observers cleanly when the server exits."""
    if hasattr(app.state, "observer"):
        app.state.observer.shutdown()


@app.post("/api/observer/start/{game_id}")
async def observer_start(game_id: int, body: ObserverStartRequest):
    """Start (or return existing) AI observer for *game_id*.

    The observer polls the CV worker for live positions and the NHL PBP
    for goals and shots, auto-generating observations into WhizFeed.

    Returns the observer status dict immediately; observation logging
    happens asynchronously in the background thread.
    """
    try:
        mgr = _observer_manager()
        status = mgr.start(
            game_id=game_id,
            game_date=body.game_date,
            away_team=body.away_team,
            home_team=body.home_team,
            accumulator=_cv_accumulator(),  # wire 16.14-16.25 DuckDB writes
        )
        return status
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/observer/stop/{game_id}")
async def observer_stop(game_id: int):
    """Stop the AI observer for *game_id*.

    Returns ``{"stopped": true}`` if an observer was running, or
    ``{"stopped": false}`` when none was found for this game.
    """
    try:
        mgr     = _observer_manager()
        stopped = mgr.stop(game_id)
        return {"stopped": stopped, "game_id": game_id}
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/observer/status")
async def observer_status():
    """Return status for all active AI observers.

    Each entry includes ``game_id``, ``alive``, ``obs_count``,
    ``started_at``, ``away_team``, ``home_team``, and ``game_date``.
    """
    try:
        mgr = _observer_manager()
        return {"observers": mgr.all_statuses()}
    except Exception as exc:
        return {"observers": [], "error": str(exc)}


@app.get("/api/observer/events/{game_id}")
async def observer_events(
    game_id: int,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return recent auto-generated observations for *game_id* from WhizFeed.

    Filters the WhizFeed to observations where ``source == "ai_observer"``
    and ``game_id`` matches, ordered newest-first.
    """
    try:
        from models.whiz_feed import WhizFeed
        feed = WhizFeed()
        obs  = [
            o for o in feed.all_observations()
            if o.get("source") == "ai_observer"
            and str(o.get("game_id", "")) == str(game_id)
        ]
        # Newest first
        obs.sort(key=lambda o: o.get("noted_at", ""), reverse=True)
        return {
            "game_id":    game_id,
            "count":      len(obs),
            "events":     obs[:limit],
        }
    except Exception as exc:
        return {"game_id": game_id, "count": 0, "events": [], "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# Features 16.14–16.25 — CV Analytics query endpoint
# ══════════════════════════════════════════════════════════════════════════════
#
# GET /api/cv/analytics/{game_id}
#
# Returns a summary of all analytics tables for the given game:
#   shift_metrics   — per-player per-shift skating load + live ratios (16.14/15/22)
#   pass_events     — detected pass events (16.17)
#   battle_events   — puck battle outcomes (16.21)
#   goalie_shifts   — goalie crease analytics (16.19)
#   zone_pressure   — zone pressure snapshots (16.20)
#   line_shifts     — on-ice line combinations (16.23)
#   turnover_events — possession losses by zone/cause (16.24)


def _df_to_records(df) -> list[dict]:
    """Convert a Polars DataFrame to a list of JSON-serialisable dicts."""
    return df.to_dicts() if df is not None and len(df) > 0 else []


@app.get("/api/cv/analytics/{game_id}")
async def cv_analytics(game_id: int):
    """Return all CV analytics table summaries for *game_id*.

    Data is only present when the AI Observer has been running for this game
    (POST /api/observer/start/{game_id} was called).  Returns empty lists when
    no data has been recorded yet.
    """
    try:
        acc = _cv_accumulator()
        return {
            "game_id": game_id,
            "shift_metrics":   _df_to_records(acc.shift_metrics(game_id)),
            "pass_events":     _df_to_records(acc.pass_events(game_id)),
            "battle_events":   _df_to_records(acc.battle_events(game_id)),
            "goalie_shifts":   _df_to_records(acc.goalie_shifts(game_id)),
            "zone_pressure":   _df_to_records(acc.zone_pressure(game_id)),
            "line_shifts":     _df_to_records(acc.line_shifts(game_id)),
            "turnover_events": _df_to_records(acc.turnover_events(game_id)),
        }
    except Exception as exc:
        return {
            "game_id": game_id,
            "error":   str(exc),
            "shift_metrics": [], "pass_events": [], "battle_events": [],
            "goalie_shifts": [], "zone_pressure": [], "line_shifts": [],
            "turnover_events": [],
        }


@app.get("/api/cv/analytics/{game_id}/shift_metrics")
async def cv_shift_metrics(game_id: int):
    """Return shift_metrics rows for *game_id* (per-player per-shift load + live ratios)."""
    try:
        acc = _cv_accumulator()
        return {"game_id": game_id, "rows": _df_to_records(acc.shift_metrics(game_id))}
    except Exception as exc:
        return {"game_id": game_id, "rows": [], "error": str(exc)}


@app.get("/api/cv/analytics/{game_id}/battles")
async def cv_battle_events(game_id: int):
    """Return battle_events rows for *game_id* (puck battle outcomes)."""
    try:
        acc = _cv_accumulator()
        return {"game_id": game_id, "rows": _df_to_records(acc.battle_events(game_id))}
    except Exception as exc:
        return {"game_id": game_id, "rows": [], "error": str(exc)}


@app.get("/api/cv/analytics/{game_id}/turnovers")
async def cv_turnover_events(game_id: int):
    """Return turnover_events rows for *game_id* (possession losses by zone/cause)."""
    try:
        acc = _cv_accumulator()
        return {"game_id": game_id, "rows": _df_to_records(acc.turnover_events(game_id))}
    except Exception as exc:
        return {"game_id": game_id, "rows": [], "error": str(exc)}

