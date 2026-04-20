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

# Load IPTV env-var sidecar (nightly workflow writes this from the GH secret
# IPTV_ENV_BLOCK so credentials & relay URLs live outside the systemd unit).
# Parse key=value lines and seed os.environ — shell/systemd vars win over file.
def _load_iptv_env_file() -> None:
    p = Path(__file__).parent / "iptv.env"
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass
_load_iptv_env_file()

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


# ---------------------------------------------------------------------------
# CV-to-player-NN feed gate — rob-only switch (Feature 16.26)
# ---------------------------------------------------------------------------
# Passive CV capture runs for all 7 localhost:3000 users silently. Only when rob
# flips this gate ON does train_behavior_net.py start folding aggregated CV
# observations into the player behavioral net (Feature 2.22). Detector and
# per-arena training happen regardless — they don't touch player identities.
#
# No server-side auth enforced here; Next.js middleware restricts POST to
# rob, mirroring the dev-banner pattern. Training script reads the JSON
# file directly, not via HTTP.


def _cv_gate_path() -> Path:
    return _GRETZKY_DATA_DIR / "cv_gate.json"


def _read_cv_gate() -> dict:
    import json
    p = _cv_gate_path()
    if not p.exists():
        return {"enabled": False, "updated_at": None, "updated_by": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False, "updated_at": None, "updated_by": None}


@app.get("/cv/gate")
async def get_cv_gate() -> dict:
    """Current state of the CV → player-NN feed gate."""
    return _read_cv_gate()


@app.post("/cv/gate")
async def set_cv_gate(payload: dict) -> dict:
    """Flip the gate. Middleware restricts this path to rob."""
    import json
    enabled = bool(payload.get("enabled", False))
    updated_by = str(payload.get("updated_by") or "rob")[:64]
    state = {
        "enabled":    enabled,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": updated_by,
    }
    p = _cv_gate_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, p)
    return state


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
            raw, sched_raw = await asyncio.gather(
                client.get_scoreboard(),
                client.get_schedule("now"),
                return_exceptions=True,
            )
            if isinstance(raw, Exception):
                raise raw
    except NHLApiError as exc:
        return {"games": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"games": [], "error": f"Unexpected error: {exc}"}

    # NHL's /score/now omits seriesStatus entirely. /schedule/now has it for every
    # playoff game going forward — match by game_id first, then fall back to the
    # team-pair (yesterday's FINAL won't be in schedule/now, but tomorrow's game 2
    # in the same series carries the post-game-1 series score).
    series_by_id: dict[int, dict] = {}
    series_by_pair: dict[frozenset, dict] = {}
    if isinstance(sched_raw, dict):
        for bucket in sched_raw.get("gameWeek") or []:
            for g in bucket.get("games") or []:
                ss = g.get("seriesStatus")
                if not ss:
                    continue
                gid = g.get("id")
                if gid:
                    series_by_id[gid] = ss
                away_abbrev = (g.get("awayTeam") or {}).get("abbrev", "")
                home_abbrev = (g.get("homeTeam") or {}).get("abbrev", "")
                if away_abbrev and home_abbrev:
                    series_by_pair.setdefault(frozenset({away_abbrev, home_abbrev}), ss)

    games_by_date = raw.get("gamesByDate") or []
    if not games_by_date:
        return {"games": []}

    from datetime import date, timedelta
    # Use NHL's focusedDate to avoid UTC midnight / ET timezone mismatch
    today_str          = raw.get("focusedDate") or date.today().isoformat()
    _today             = date.fromisoformat(today_str)
    yesterday_str      = (_today - timedelta(days=1)).isoformat()
    two_days_ago_str   = (_today - timedelta(days=2)).isoformat()
    tomorrow_str       = (_today + timedelta(days=1)).isoformat()
    keep_dates         = {today_str, yesterday_str, two_days_ago_str, tomorrow_str}

    # Flatten last 2 days + today + tomorrow
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

        gid = g.get("id")
        away_abbrev = away.get("abbrev", "")
        home_abbrev = home.get("abbrev", "")
        series_status = (
            g.get("seriesStatus")
            or series_by_id.get(gid)
            or (series_by_pair.get(frozenset({away_abbrev, home_abbrev})) if away_abbrev and home_abbrev else None)
        )

        normalised.append({
            "game_id":         gid,
            "date":            date_labels.get(gid, ""),
            "game_state":      game_state,
            "away_team":       away_abbrev,
            "home_team":       home_abbrev,
            "away_score":      away.get("score"),
            "home_score":      home.get("score"),
            "period":          period,
            "period_label":    period_label,
            "time_remaining":  clock.get("timeRemaining"),
            "in_intermission": bool(clock.get("inIntermission", False)),
            "start_time_utc":  g.get("startTimeUTC"),
            "outcome_type":    outcome_type,
            "game_type":       g.get("gameType"),
            "series_status":   series_status,
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
                "game_type":       g.get("gameType"),
                "series_status":   g.get("seriesStatus"),
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

        series_status = landing.get("seriesStatus")
        game_type     = landing.get("gameType")

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
                    "game_type":    game_type,
                    "series_status": series_status,
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

    from datetime import date as _d, timedelta as _td
    today = _d.today()

    # During playoffs / offseason /v1/standings/<today> returns {"standings":[]}.
    # Walk back up to 14 days until we find a date with real records so the
    # frontend keeps rendering end-of-regular-season standings.
    raw: dict = {"standings": []}
    try:
        async with NHLClient() as client:
            for back in range(15):
                date_str = (today - _td(days=back)).isoformat()
                raw = await client._get(client._web, f"/v1/standings/{date_str}")
                if raw.get("standings"):
                    break
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
# Team page — roster stats + cap data
# ===========================================================================

_PUCKPEDIA_SLUGS: dict[str, str] = {
    "ANA": "anaheim-ducks",    "BOS": "boston-bruins",      "BUF": "buffalo-sabres",
    "CGY": "calgary-flames",   "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche","CBJ": "columbus-blue-jackets","DAL": "dallas-stars",
    "DET": "detroit-red-wings","EDM": "edmonton-oilers",    "FLA": "florida-panthers",
    "LAK": "los-angeles-kings","MIN": "minnesota-wild",     "MTL": "montreal-canadiens",
    "NSH": "nashville-predators","NJD": "new-jersey-devils", "NYI": "new-york-islanders",
    "NYR": "new-york-rangers", "OTT": "ottawa-senators",    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins","SEA": "seattle-kraken",   "SJS": "san-jose-sharks",
    "STL": "st-louis-blues",   "TBL": "tampa-bay-lightning","TOR": "toronto-maple-leafs",
    "UTA": "utah-hockey-club", "VAN": "vancouver-canucks",  "VGK": "vegas-golden-knights",
    "WSH": "washington-capitals","WPG": "winnipeg-jets",
}


_TEAM_CODE_REMAP: dict[str, str] = {
    "UTH": "UTA", "LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL",
}

@app.get("/team/{team}/stats")
async def team_stats(team: str) -> dict:
    """Player stats for all skaters and goalies on a team — current season from NHL API."""
    team = _TEAM_CODE_REMAP.get(team.upper(), team.upper())
    _UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    def _name(obj: object) -> str:
        if isinstance(obj, dict):
            return obj.get("default", "") or ""
        return str(obj or "")

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers={"User-Agent": _UA}) as client:
            r = await client.get(f"https://api-web.nhle.com/v1/club-stats/{team}/now")
        if r.status_code != 200:
            return {"team": team, "skaters": [], "goalies": [], "error": f"NHL API {r.status_code}"}
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        return {"team": team, "skaters": [], "goalies": [], "error": str(exc)}

    def _fmt_toi(secs: object) -> str:
        """Convert seconds-per-game float to MM:SS string."""
        try:
            s = int(float(secs or 0))
            return f"{s // 60}:{s % 60:02d}"
        except (TypeError, ValueError):
            return str(secs or "")

    skaters = []
    for p in data.get("skaters") or []:
        ppg = int(p.get("powerPlayGoals") or 0)
        ppa = int(p.get("powerPlayAssists") or 0)
        skaters.append({
            "player_id":    p.get("playerId"),
            "headshot":     p.get("headshot", ""),
            "first_name":   _name(p.get("firstName")),
            "last_name":    _name(p.get("lastName")),
            "position":     p.get("positionCode", ""),
            "jersey":       p.get("sweaterNumber"),
            "gp":           p.get("gamesPlayed", 0),
            "goals":        p.get("goals", 0),
            "assists":      p.get("assists", 0),
            "points":       p.get("points", 0),
            "plus_minus":   p.get("plusMinus", 0),
            "pim":          p.get("penaltyMinutes", 0),
            "pp_points":    ppg + ppa,
            "pp_goals":     ppg,
            "sh_goals":     p.get("shortHandedGoals") or p.get("shorthandedGoals", 0),
            "gwg":          p.get("gameWinningGoals", 0),
            "shots":        p.get("shots", 0),
            "shooting_pct": round(float(p.get("shootingPctg") or 0) * 100, 1),
            "avg_toi":      _fmt_toi(p.get("avgTimeOnIcePerGame") or p.get("avgToi")),
            "faceoff_pct":  round(float(p.get("faceoffWinPctg") or 0) * 100, 1),
        })

    goalies = []
    for g in data.get("goalies") or []:
        goalies.append({
            "player_id":     g.get("playerId"),
            "headshot":      g.get("headshot", ""),
            "first_name":    _name(g.get("firstName")),
            "last_name":     _name(g.get("lastName")),
            "position":      "G",
            "jersey":        g.get("sweaterNumber"),
            "gp":            g.get("gamesPlayed", 0),
            "wins":          g.get("wins", 0),
            "losses":        g.get("losses", 0),
            "ot_losses":     g.get("otLosses", 0),
            "gaa":           round(float(g.get("goalsAgainstAvg") or 0), 2),
            "sv_pct":        round(float(g.get("savePctg") or 0), 3),
            "shutouts":      g.get("shutouts", 0),
            "saves":         g.get("saves", 0),
            "goals_against": g.get("goalsAgainst", 0),
        })

    skaters.sort(key=lambda x: (-x["points"], -x["goals"]))
    return {"team": team, "skaters": skaters, "goalies": goalies}


@app.get("/team/{team}/cap")
async def team_cap(team: str) -> dict:
    """Cap data for a team scraped from PuckPedia __NEXT_DATA__."""
    import json as _json
    import re as _re

    team = _TEAM_CODE_REMAP.get(team.upper(), team.upper())
    slug = _PUCKPEDIA_SLUGS.get(team)
    if not slug:
        return {"team": team, "players": [], "cap_ceiling": 88_000_000, "status": "unknown_team"}

    _UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    try:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True,
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
        ) as client:
            r = await client.get(f"https://puckpedia.com/team/{slug}")
        if r.status_code != 200:
            return {"team": team, "players": [], "cap_ceiling": 88_000_000, "status": f"http_{r.status_code}"}
    except Exception as exc:  # noqa: BLE001
        return {"team": team, "players": [], "cap_ceiling": 88_000_000, "status": "fetch_error", "error": str(exc)}

    match = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, _re.DOTALL)
    if not match:
        return {"team": team, "players": [], "cap_ceiling": 88_000_000, "status": "parse_error"}

    try:
        nd = _json.loads(match.group(1))
    except _json.JSONDecodeError:
        return {"team": team, "players": [], "cap_ceiling": 88_000_000, "status": "json_error"}

    def _fmt_money(val: object) -> int | None:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            try:
                return int(float(val.replace("$", "").replace(",", "").strip()))
            except ValueError:
                return None
        return None

    def _find_list(obj: object, depth: int = 0) -> list | None:
        if depth > 14:
            return None
        if isinstance(obj, list) and len(obj) >= 3:
            s = obj[0] if obj else {}
            if isinstance(s, dict) and any(k in s for k in ("capHit", "cap_hit", "aav", "salary", "name", "lastName", "player")):
                return obj
        if isinstance(obj, dict):
            for v in obj.values():
                r2 = _find_list(v, depth + 1)
                if r2:
                    return r2
        return None

    def _find_summary(obj: object, depth: int = 0) -> dict | None:
        if depth > 10:
            return None
        if isinstance(obj, dict):
            if any(k in obj for k in ("capSpace", "cap_space", "capCeiling", "totalCapHit", "total_cap_hit")):
                return obj  # type: ignore[return-value]
            for v in obj.values():
                r2 = _find_summary(v, depth + 1)
                if r2:
                    return r2
        return None

    raw = _find_list(nd) or []
    players: list[dict] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        player_obj = p.get("player") if isinstance(p.get("player"), dict) else {}
        name = (
            p.get("name")
            or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
            or (player_obj.get("name") or f"{player_obj.get('firstName', '')} {player_obj.get('lastName', '')}".strip())
        )
        if not name or len(name) < 3:
            continue
        cap_raw = p.get("capHit") or p.get("cap_hit") or p.get("aav") or p.get("salary") or player_obj.get("capHit")
        pos = p.get("position") or p.get("pos") or player_obj.get("position") or player_obj.get("pos") or ""
        players.append({
            "name":            str(name),
            "position":        str(pos),
            "age":             p.get("age") or player_obj.get("age"),
            "cap_hit":         _fmt_money(cap_raw),
            "contract_type":   str(p.get("type") or p.get("contractType") or p.get("contract_type") or ""),
            "expiry_year":     p.get("expiryYear") or p.get("expiry_year") or p.get("expiryDate"),
            "years_remaining": p.get("yearsRemaining") or p.get("years_remaining") or p.get("term"),
        })

    cap_ceiling = 88_000_000
    total_hit: int | None = None
    cap_space: int | None = None
    summary = _find_summary(nd)
    if summary:
        cap_ceiling = _fmt_money(summary.get("capCeiling") or summary.get("cap_ceiling")) or cap_ceiling
        total_hit   = _fmt_money(summary.get("totalCapHit") or summary.get("total_cap_hit") or summary.get("capHit"))
        cap_space   = _fmt_money(summary.get("capSpace") or summary.get("cap_space"))

    return {
        "team":        team,
        "players":     players,
        "cap_ceiling": cap_ceiling,
        "total_hit":   total_hit,
        "cap_space":   cap_space,
        "status":      "ok" if players else "empty",
    }


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
                    "limit":      50,
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


@app.get("/skater-stats-playoffs")
async def skater_stats_playoffs() -> dict:
    """NHL skater stats leaders for the current playoffs (gameTypeId=3)."""
    cached = _cache_get("skater-stats-playoffs")
    if cached is not None:
        return cached  # type: ignore[return-value]

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

    # Playoff shot / TOI floors are lower — small series samples.
    EXTRA: dict[str, str] = {
        "shootingPctg": " and shots >= 10",
        "toi":          " and gamesPlayed >= 3",
    }

    async def _fetch_cat(client: httpx.AsyncClient, result_key: str, sort_field: str, value_field: str) -> tuple[str, list]:
        try:
            extra = EXTRA.get(result_key, "")
            r = await client.get(
                BASE,
                params={
                    "cayenneExp": f"seasonId=20252026 and gameTypeId=3{extra}",
                    "sort":       sort_field,
                    "dir":        "DESC",
                    "limit":      50,
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

    payload = {"categories": dict(pairs)}
    _cache_set("skater-stats-playoffs", payload)
    return payload


@app.get("/goalie-stats-playoffs")
async def goalie_stats_playoffs() -> dict:
    """NHL goalie stats leaders for the current playoffs (gameTypeId=3)."""
    cached = _cache_get("goalie-stats-playoffs")
    if cached is not None:
        return cached  # type: ignore[return-value]

    # (result_key, sort_field, value_field)
    CAT_MAP: list[tuple[str, str, str]] = [
        ("wins",                "wins",             "wins"),
        ("savePctg",            "savePct",          "savePct"),
        ("goalsAgainstAverage", "goalsAgainstAverage", "goalsAgainstAverage"),
        ("shutouts",            "shutouts",         "shutouts"),
    ]
    BASE = "https://api.nhle.com/stats/rest/en/goalie/summary"
    _UA  = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    EXTRA: dict[str, str] = {
        "savePctg":            " and gamesPlayed >= 2",
        "goalsAgainstAverage": " and gamesPlayed >= 2",
    }

    async def _fetch_cat(client: httpx.AsyncClient, rk: str, sf: str, vf: str) -> tuple[str, list]:
        try:
            extra = EXTRA.get(rk, "")
            direction = "ASC" if rk == "goalsAgainstAverage" else "DESC"
            r = await client.get(
                BASE,
                params={
                    "cayenneExp": f"seasonId=20252026 and gameTypeId=3{extra}",
                    "sort":       sf,
                    "dir":        direction,
                    "limit":      40,
                },
            )
            if r.status_code != 200:
                return rk, []
            rows = r.json().get("data", [])
            leaders = []
            for rank, p in enumerate(rows, 1):
                pid  = p.get("playerId", 0)
                team = (p.get("teamAbbrevs") or "").split(",")[0].strip()
                headshot = f"https://assets.nhle.com/mugs/nhl/20252026/{team}/{pid}.png" if team and pid else ""
                full = p.get("goalieFullName", "")
                parts = full.split(" ", 1)
                leaders.append({
                    "rank":       rank,
                    "id":         pid,
                    "name":       full,
                    "first_name": parts[0] if parts else "",
                    "last_name":  parts[1] if len(parts) > 1 else "",
                    "team":       team,
                    "position":   "G",
                    "number":     None,
                    "headshot":   headshot,
                    "value":      p.get(vf),
                })
            return rk, leaders
        except Exception:  # noqa: BLE001
            return rk, []

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": _UA}) as client:
        pairs = await asyncio.gather(*[_fetch_cat(client, rk, sf, vf) for rk, sf, vf in CAT_MAP])

    payload = {"categories": dict(pairs)}
    _cache_set("goalie-stats-playoffs", payload)
    return payload


@app.get("/playoff-bracket")
async def playoff_bracket(season: str = "20252026") -> dict:
    """NHL playoff bracket (carousel) for a given season.

    Returns normalized rounds → series with {letter, round, topSeed, bottomSeed,
    wins, neededToWin, winningTeamId, losingTeamId}. Always includes teams when
    NHL API has set the matchup; wins default to 0 before round 1 starts.
    """
    cache_key = f"playoff-bracket:{season}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    _UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers={"User-Agent": _UA}) as client:
            r = await client.get(f"https://api-web.nhle.com/v1/playoff-series/carousel/{season}")
            if r.status_code != 200:
                return {"rounds": [], "currentRound": 0, "error": f"status={r.status_code}"}
            raw = r.json()
    except Exception as exc:  # noqa: BLE001
        return {"rounds": [], "currentRound": 0, "error": str(exc)}

    rounds_out: list[dict] = []
    for rnd in raw.get("rounds") or []:
        series_out: list[dict] = []
        for s in rnd.get("series") or []:
            top = s.get("topSeed")    or {}
            bot = s.get("bottomSeed") or {}
            series_out.append({
                "letter":         s.get("seriesLetter"),
                "round":          s.get("roundNumber"),
                "label":          s.get("seriesLabel"),
                "neededToWin":    s.get("neededToWin", 4),
                "topSeed": {
                    "team":    top.get("abbrev"),
                    "team_id": top.get("id"),
                    "wins":    top.get("wins", 0),
                },
                "bottomSeed": {
                    "team":    bot.get("abbrev"),
                    "team_id": bot.get("id"),
                    "wins":    bot.get("wins", 0),
                },
                "winningTeamId":  s.get("winningTeamId"),
                "losingTeamId":   s.get("losingTeamId"),
            })
        rounds_out.append({
            "round":  rnd.get("roundNumber"),
            "label":  rnd.get("roundLabel"),
            "abbrev": rnd.get("roundAbbrev"),
            "series": series_out,
        })

    payload = {
        "season":       raw.get("seasonId", season),
        "currentRound": raw.get("currentRound", 0),
        "rounds":       rounds_out,
    }
    _cache_set(cache_key, payload)
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

    import unicodedata

    def _norm(s: str) -> str:
        """Lowercase + strip accents (e.g. ý→y, é→e) for fuzzy name matching."""
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()

    df = pl.read_parquet(parquets[-1])
    name_lower = _norm(name)
    # Exact match first (fast path)
    row = df.filter(pl.col("shooter_name").str.to_lowercase() == name.strip().lower())
    # Accent-normalised fallback — handles NHL API names like "Slafkovský" vs parquet "Slafkovsky"
    if row.is_empty():
        row = df.filter(
            pl.col("shooter_name").map_elements(
                lambda n: _norm(n or "") == name_lower, return_dtype=pl.Boolean
            )
        )
    # Truncation fallback — parquet may have a prefix-truncated name (e.g. "Juraj Slafkovsk"
    # instead of "Juraj Slafkovsky"). Try matching any parquet name that is a prefix of the
    # search name (up to 3 chars shorter) so the full/correct name always resolves to the data.
    if row.is_empty():
        row = df.filter(
            pl.col("shooter_name").map_elements(
                lambda n: bool(n and name_lower.startswith(_norm(n)) and 0 < len(name_lower) - len(_norm(n)) <= 3),
                return_dtype=pl.Boolean,
            )
        )

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
    rapm_xgf_60: float | None = None
    rapm_toi_ev: float | None = None
    rapm_goals_p60: float | None = None
    rapm_shots_p60: float | None = None
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
                    if "xgf_60" in rapm_df.columns:
                        rapm_xgf_60 = round(float(rapm_row["xgf_60"][0]), 2)
                    if "toi_ev" in rapm_df.columns:
                        rapm_toi_ev = round(float(rapm_row["toi_ev"][0]), 1)
                    # Derive goals/shots per 60 from raw counts + toi
                    toi_h = (rapm_toi_ev or 0) / 3600
                    if toi_h > 0:
                        if "goals_per60" in rapm_df.columns:
                            rapm_goals_p60 = round(float(rapm_row["goals_per60"][0]), 2)
                        elif r.get("goals") is not None:
                            rapm_goals_p60 = round(float(r["goals"]) / toi_h, 2)
                        if "shots_per60" in rapm_df.columns:
                            rapm_shots_p60 = round(float(rapm_row["shots_per60"][0]), 2)
                        elif r.get("shots") is not None:
                            rapm_shots_p60 = round(float(r["shots"]) / toi_h, 2)
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

    # Try to join Special Teams ratings (Feature 2.7)
    special_teams_pp: float | None = None
    special_teams_pk: float | None = None
    try:
        st_dir = _GRETZKY_DATA_DIR / "special_teams"
        st_parquets = sorted(st_dir.glob("special_teams_*.parquet")) if st_dir.exists() else []
        if st_parquets and player_id_val is not None:
            st_df = pl.read_parquet(st_parquets[-1])
            st_row = st_df.filter(pl.col("player_id") == player_id_val)
            if not st_row.is_empty():
                st = st_row.to_dicts()[0]
                special_teams_pp = round(float(st["pp_rating"]), 3) if st.get("pp_rating") is not None else None
                special_teams_pk = round(float(st["pk_rating"]), 3) if st.get("pk_rating") is not None else None
    except Exception:
        pass

    # Try to join Bayesian player rating (Feature 2.9)
    bayesian_rating:      float | None = None
    bayesian_uncertainty: float | None = None
    try:
        br_dir = _GRETZKY_DATA_DIR / "bayes_ratings"
        br_parquets = sorted(br_dir.glob("player_ratings_*.parquet")) if br_dir.exists() else []
        if br_parquets and player_id_val is not None:
            br_df = pl.read_parquet(br_parquets[-1])
            br_row = br_df.filter(pl.col("player_id") == player_id_val)
            if not br_row.is_empty():
                br = br_row.to_dicts()[0]
                bayesian_rating      = round(float(br["posterior_mean"]),  3) if br.get("posterior_mean")  is not None else None
                bayesian_uncertainty = round(float(br["posterior_sigma"]), 3) if br.get("posterior_sigma") is not None else None
    except Exception:
        pass

    # Try to join Playoff Delta (Feature 2.23)
    playoff_delta: float | None = None
    try:
        pd_dir = _GRETZKY_DATA_DIR / "playoff_delta"
        pd_parquets = sorted(pd_dir.glob("playoff_delta_*.parquet")) if pd_dir.exists() else []
        if pd_parquets and player_id_val is not None:
            pd_df = pl.read_parquet(pd_parquets[-1])
            if not pd_df.is_empty():
                pd_row = pd_df.filter(pl.col("player_id") == player_id_val)
                if not pd_row.is_empty():
                    pd_r = pd_row.to_dicts()[0]
                    reg_xgf = pd_r.get("reg_xgf_per60")
                    po_xgf  = pd_r.get("playoff_xgf_per60")
                    if reg_xgf is not None and po_xgf is not None and float(reg_xgf) > 0:
                        playoff_delta = round(float(po_xgf) - float(reg_xgf), 3)
    except Exception:
        pass

    # Try to join Former Team Boost (Feature 2.27)
    former_team_boost: float | None = None
    former_team:       str | None = None
    try:
        ftb_dir = _GRETZKY_DATA_DIR / "former_team_boost"
        ftb_parquets = sorted(ftb_dir.glob("former_team_boost_*.parquet")) if ftb_dir.exists() else []
        if ftb_parquets and player_id_val is not None:
            ftb_df = pl.read_parquet(ftb_parquets[-1])
            ftb_row = ftb_df.filter(pl.col("player_id") == player_id_val)
            if not ftb_row.is_empty():
                ftb = ftb_row.to_dicts()[0]
                former_team_boost = round(float(ftb["base_boost"]), 3) if ftb.get("base_boost") is not None else None
                former_team       = str(ftb["former_team"]) if ftb.get("former_team") else None
    except Exception:
        pass

    # Try to join puck battles (battle_percentile, hits_per60, blocks_per60)
    battle_score_val:      float | None = None
    battle_percentile_val: float | None = None
    hits_per60_val:        float | None = None
    blocks_per60_val:      float | None = None
    try:
        battles_dir = _GRETZKY_DATA_DIR / "battles"
        battle_parquets = sorted(battles_dir.glob("puck_battle_*.parquet")) if battles_dir.exists() else []
        if battle_parquets and player_id_val is not None:
            b_df = pl.read_parquet(battle_parquets[-1])
            b_row = b_df.filter(pl.col("player_id") == player_id_val) if "player_id" in b_df.columns else pl.DataFrame()
            if not b_row.is_empty():
                b = b_row.to_dicts()[0]
                battle_score_val      = round(float(b["battle_score"]), 3)      if b.get("battle_score")      is not None else None
                battle_percentile_val = round(float(b["battle_percentile"]), 1) if b.get("battle_percentile") is not None else None
                hits_per60_val        = round(float(b["hits_per60"]), 2)        if b.get("hits_per60")        is not None else None
                blocks_per60_val      = round(float(b["blocks_per60"]), 2)      if b.get("blocks_per60")      is not None else None
    except Exception:
        pass

    # Try to join behavioral NN predictions (zone tendencies)
    nn_carry_in_pct_val:        float | None = None
    nn_dump_pct_val:            float | None = None
    nn_shoot_slot_pct_val:      float | None = None
    nn_shoot_perimeter_pct_val: float | None = None
    nn_drive_net_pct_val:       float | None = None
    nn_battle_corner_pct_val:   float | None = None
    nn_hold_corner_pct_val:     float | None = None
    try:
        beh_dir = _GRETZKY_DATA_DIR / "behavior_net"
        beh_parquets = sorted(beh_dir.glob("behavior_predictions_*.parquet")) if beh_dir.exists() else []
        if beh_parquets and player_id_val is not None:
            beh_df = pl.read_parquet(beh_parquets[-1])
            beh_row = beh_df.filter(pl.col("player_id") == player_id_val) if "player_id" in beh_df.columns else pl.DataFrame()
            if not beh_row.is_empty():
                beh = beh_row.to_dicts()[0]
                nn_carry_in_pct_val        = round(float(beh["carry_in"]) * 100, 1)          if beh.get("carry_in")         is not None else None
                nn_dump_pct_val            = round(float(beh["dump"]) * 100, 1)              if beh.get("dump")             is not None else None
                nn_shoot_slot_pct_val      = round(float(beh["shoot_slot"]) * 100, 1)        if beh.get("shoot_slot")       is not None else None
                nn_shoot_perimeter_pct_val = round(float(beh["shoot_perimeter"]) * 100, 1)   if beh.get("shoot_perimeter")  is not None else None
                nn_drive_net_pct_val       = round(float(beh["drive_net"]) * 100, 1)         if beh.get("drive_net")        is not None else None
                nn_battle_corner_pct_val   = round(float(beh["battle_corner"]) * 100, 1)     if beh.get("battle_corner")    is not None else None
                nn_hold_corner_pct_val     = round(float(beh["hold_corner"]) * 100, 1)       if beh.get("hold_corner")      is not None else None
    except Exception:
        pass

    # Try to join skating baseline (zone time splits, speed, distance)
    skating_zone_oz_val:       float | None = None
    skating_zone_dz_val:       float | None = None
    skating_avg_speed_val:     float | None = None
    skating_max_speed_val:     float | None = None
    skating_distance_val:      float | None = None
    skating_games_val:         int   | None = None
    try:
        skate_dir = _GRETZKY_DATA_DIR / "skating_baseline"
        skate_parquets = sorted(skate_dir.glob("skating_baseline_*.parquet")) if skate_dir.exists() else []
        if skate_parquets and player_id_val is not None:
            sk_df = pl.read_parquet(skate_parquets[-1])
            sk_row = sk_df.filter(pl.col("player_id") == player_id_val) if "player_id" in sk_df.columns else pl.DataFrame()
            if not sk_row.is_empty():
                sk = sk_row.to_dicts()[0]
                skating_zone_oz_val   = round(float(sk["baseline_zone_time_pct_oz"]) * 100, 1) if sk.get("baseline_zone_time_pct_oz") is not None else None
                skating_zone_dz_val   = round(float(sk["baseline_zone_time_pct_dz"]) * 100, 1) if sk.get("baseline_zone_time_pct_dz") is not None else None
                skating_avg_speed_val = round(float(sk["baseline_avg_speed_kmh"]), 1)           if sk.get("baseline_avg_speed_kmh")    is not None else None
                skating_max_speed_val = round(float(sk["baseline_max_speed_kmh"]), 1)           if sk.get("baseline_max_speed_kmh")    is not None else None
                skating_distance_val  = round(float(sk["baseline_distance_per_game_km"]), 2)    if sk.get("baseline_distance_per_game_km") is not None else None
                skating_games_val     = int(sk["n_games_total"]) if sk.get("n_games_total") is not None else None
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
        "xgf_per60":            rapm_xgf_60,
        "toi_ev":               round(rapm_toi_ev / 60, 1) if rapm_toi_ev is not None else None,
        "goals_per60":          rapm_goals_p60,
        "shots_per60":          rapm_shots_p60,
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
        "special_teams_pp":     special_teams_pp,
        "special_teams_pk":     special_teams_pk,
        "bayesian_rating":      bayesian_rating,
        "bayesian_uncertainty": bayesian_uncertainty,
        "playoff_delta":        playoff_delta,
        "former_team_boost":    former_team_boost,
        "former_team":          former_team,
        "battle_score":              battle_score_val,
        "battle_percentile":         battle_percentile_val,
        "hits_per60":                hits_per60_val,
        "blocks_per60":              blocks_per60_val,
        "nn_carry_in_pct":           nn_carry_in_pct_val,
        "nn_dump_pct":               nn_dump_pct_val,
        "nn_shoot_slot_pct":         nn_shoot_slot_pct_val,
        "nn_shoot_perimeter_pct":    nn_shoot_perimeter_pct_val,
        "nn_drive_net_pct":          nn_drive_net_pct_val,
        "nn_battle_corner_pct":      nn_battle_corner_pct_val,
        "nn_hold_corner_pct":           nn_hold_corner_pct_val,
        "skating_zone_time_oz_pct":     skating_zone_oz_val,
        "skating_zone_time_dz_pct":     skating_zone_dz_val,
        "skating_avg_speed_kmh":        skating_avg_speed_val,
        "skating_max_speed_kmh":        skating_max_speed_val,
        "skating_distance_per_game_km": skating_distance_val,
        "skating_games_sample":         skating_games_val,
    }


def _build_name_lookup() -> dict[int, str]:
    """Build player_id → player_name lookup from all available parquets."""
    import polars as pl
    lookup: dict[int, str] = {}

    def _ingest(path_glob, only_if_missing: bool = False) -> None:
        for p in sorted(path_glob):
            try:
                df = pl.read_parquet(p, columns=["player_id", "player_name"])
                for r in df.to_dicts():
                    pid, name = r.get("player_id"), r.get("player_name") or ""
                    if not pid or not name or name.startswith("player_"):
                        continue
                    key = int(pid)
                    if only_if_missing and key in lookup:
                        continue
                    lookup[key] = name
            except Exception:
                pass

    # Broadest real-name sources first (zero placeholders) ─────────────────
    _ingest((_GRETZKY_DATA_DIR / "bayes_ratings").glob("*.parquet"))
    _ingest((_GRETZKY_DATA_DIR / "skating_baseline").glob("*.parquet"), only_if_missing=True)
    _ingest((_GRETZKY_DATA_DIR / "war").glob("*.parquet"), only_if_missing=True)
    _ingest((_GRETZKY_DATA_DIR / "special_teams").glob("*.parquet"), only_if_missing=True)

    # RAPM — fills in any remaining skaters (may have some placeholders,
    # which are filtered by _ingest's startswith check)
    _ingest((_GRETZKY_DATA_DIR / "rapm").glob("rapm_*.parquet"), only_if_missing=True)

    # Goalies (missing from skater-only sources)
    _ingest(_module_dir("goalie_stats").glob("goalie_stats_*.parquet"), only_if_missing=True)
    _ingest((_GRETZKY_DATA_DIR / "goalie_ratings").glob("goalie_ratings_*.parquet"), only_if_missing=True)

    # Persistent NHL API cache — resolves any IDs still missing after parquet pass
    _cache_path = _GRETZKY_DATA_DIR / "player_name_cache.json"
    try:
        import json
        cache: dict[str, str] = {}
        if _cache_path.exists():
            cache = json.loads(_cache_path.read_text())
        for k, v in cache.items():
            key = int(k)
            if key not in lookup and v and not v.startswith("player_"):
                lookup[key] = v
    except Exception:
        pass

    return lookup


def _resolve_names_via_nhl_api(player_ids: list[int]) -> None:
    """Fetch real names for unresolved player IDs from the NHL API and persist to cache.

    Call this from async endpoints after noticing placeholder names:
        asyncio.create_task(_resolve_names_async([pid1, pid2]))
    """
    import json, httpx
    cache_path = _GRETZKY_DATA_DIR / "player_name_cache.json"
    try:
        cache: dict[str, str] = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
        updated = False
        for pid in player_ids:
            key = str(pid)
            if key in cache:
                continue
            try:
                r = httpx.get(
                    f"https://api-web.nhle.com/v1/player/{pid}/landing",
                    timeout=5.0, headers={"User-Agent": "GRTZKY/1.0"},
                )
                if r.status_code == 200:
                    d = r.json()
                    first = (d.get("firstName") or {}).get("default", "")
                    last  = (d.get("lastName")  or {}).get("default", "")
                    name  = f"{first} {last}".strip()
                    if name:
                        cache[key] = name
                        updated = True
            except Exception:
                pass
        if updated:
            cache_path.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass


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


# ---------------------------------------------------------------------------
# Phase 2 — Regime Change Alerts (Feature 2.14)
# ---------------------------------------------------------------------------

@app.get("/phase2/regime-alerts")
async def phase2_regime_alerts(limit: int = 10) -> dict:
    """Return recent regime change alerts — players who have statistically shifted up or down."""
    import polars as pl

    regime_dir = _GRETZKY_DATA_DIR / "regime"
    files = sorted(regime_dir.glob("*.parquet")) if regime_dir.exists() else []
    if not files:
        return {"alerts": [], "status": "not_built"}

    try:
        df = pl.read_parquet(files[-1])
        if df.is_empty():
            return {"alerts": [], "status": "empty"}

        # Build name lookup to resolve placeholder names like "player_8470613"
        name_lookup = _build_name_lookup()

        # Build team lookup from RAPM (most complete source)
        team_lookup: dict[int, str] = {}
        try:
            rapm_dir = _GRETZKY_DATA_DIR / "rapm"
            rapm_files = sorted(rapm_dir.glob("rapm_*.parquet"))
            if rapm_files:
                rapm_df = pl.read_parquet(rapm_files[-1], columns=["player_id", "team"])
                for row in rapm_df.drop_nulls().to_dicts():
                    if row.get("player_id") and row.get("team"):
                        abbr = _abbr(row["team"]) or str(row["team"])
                        team_lookup[int(row["player_id"])] = abbr
        except Exception:
            pass

        alerts = []
        rows = (
            df.sort("mean_residual", descending=True)
            .head(limit * 3)
            .to_dicts()
        )

        # Collect any still-unresolved IDs for background NHL API resolution
        _unresolved_ids: list[int] = []

        for r in rows:
            pid = r.get("player_id")
            raw_name = r.get("player_name", "")
            # Resolve placeholder names
            if pid and (not raw_name or raw_name.startswith("player_")):
                raw_name = name_lookup.get(int(pid), raw_name)
            if not raw_name or raw_name.startswith("player_"):
                if pid:
                    _unresolved_ids.append(int(pid))
                continue  # skip unresolvable for now
            direction = "breakout" if r.get("regime_state") == "up_shift" else "crash"
            alerts.append({
                "player_id":     int(pid) if pid else None,
                "player_name":   raw_name,
                "team":          team_lookup.get(int(pid), None) if pid else None,
                "direction":     direction,
                "mean_residual": round(float(r["mean_residual"]), 3) if r.get("mean_residual") is not None else None,
                "games_detected": int(r["consecutive_games"]) if r.get("consecutive_games") is not None else None,
                "reason":        r.get("reason"),
            })
            if len(alerts) >= limit:
                break

        # Fire-and-forget: resolve any still-missing IDs so next call has them
        if _unresolved_ids:
            import threading
            threading.Thread(
                target=_resolve_names_via_nhl_api,
                args=(_unresolved_ids,),
                daemon=True,
            ).start()

        return {"alerts": alerts, "status": "ok"}
    except Exception:
        return {"alerts": [], "status": "error"}


# ---------------------------------------------------------------------------
# Phase 2 — Top Line Pairs (Feature 2.4)
# ---------------------------------------------------------------------------

@app.get("/phase2/top-pairs")
async def phase2_top_pairs(limit: int = 10) -> dict:
    """Return top line pairs by chemistry delta (how much better they are together vs. expected)."""
    import polars as pl

    chem_dir = _GRETZKY_DATA_DIR / "chemistry"
    files = sorted(chem_dir.glob("pair_chemistry_*.parquet")) if chem_dir.exists() else []
    if not files:
        return {"pairs": [], "status": "not_built"}

    try:
        df = pl.read_parquet(files[-1])
        if df.is_empty():
            return {"pairs": [], "status": "empty"}

        name_lookup = _build_name_lookup()

        # Team lookup from RAPM
        team_lookup: dict[int, str] = {}
        try:
            rapm_dir = _GRETZKY_DATA_DIR / "rapm"
            rapm_files = sorted(rapm_dir.glob("rapm_*.parquet"))
            if rapm_files:
                rapm_df = pl.read_parquet(rapm_files[-1], columns=["player_id", "team"])
                for row in rapm_df.drop_nulls().to_dicts():
                    if row.get("player_id") and row.get("team"):
                        abbr = _abbr(row["team"]) or str(row["team"])
                        team_lookup[int(row["player_id"])] = abbr
        except Exception:
            pass

        top = df.sort("chemistry_delta", descending=True).head(limit)
        pairs = []
        for r in top.to_dicts():
            a_id = r.get("player_a_id")
            b_id = r.get("player_b_id")
            a_name = name_lookup.get(int(a_id), f"player_{a_id}") if a_id else None
            b_name = name_lookup.get(int(b_id), f"player_{b_id}") if b_id else None
            if not a_name or a_name.startswith("player_") or not b_name or b_name.startswith("player_"):
                continue
            team = team_lookup.get(int(a_id)) or team_lookup.get(int(b_id)) if a_id else None
            pairs.append({
                "player_a":        a_name,
                "player_b":        b_name,
                "player_a_id":     int(a_id) if a_id else None,
                "player_b_id":     int(b_id) if b_id else None,
                "team":            team,
                "chemistry_delta": round(float(r["chemistry_delta"]), 4) if r.get("chemistry_delta") is not None else None,
                "model_xgf_pct":   round(float(r["model_xgf_pct"]) * 100, 1) if r.get("model_xgf_pct") is not None else None,
                "co_toi_ev":       round(float(r["co_toi_ev"]), 1) if r.get("co_toi_ev") is not None else None,
                "games_together":  int(r["games_together"]) if r.get("games_together") is not None else None,
            })

        return {"pairs": pairs, "status": "ok"}
    except Exception:
        return {"pairs": [], "status": "error"}


# ---------------------------------------------------------------------------
# Phase 2 — Roster Disruption Index (Feature 2.19)
# ---------------------------------------------------------------------------

@app.get("/phase2/roster-disruption")
async def phase2_roster_disruption() -> dict:
    """Return all teams sorted by Roster Disruption Index (how many key roster moves recently)."""
    rdi_dir = _GRETZKY_DATA_DIR / "roster_disruption"
    files = sorted(rdi_dir.glob("*.parquet")) if rdi_dir.exists() else []
    if not files:
        return {"teams": [], "status": "not_built"}

    try:
        import polars as pl
        df = pl.read_parquet(files[-1])
        if df.is_empty():
            return {"teams": [], "status": "empty"}

        teams = []
        for r in df.sort("disruption_index", descending=True).to_dicts():
            teams.append({
                "team":            r.get("team"),
                "disruption_index": round(float(r["disruption_index"]), 3) if r.get("disruption_index") is not None else None,
                "n_moves_30d":     int(r["n_moves_30d"]) if r.get("n_moves_30d") is not None else None,
                "n_moves_72h":     int(r["n_moves_72h"]) if r.get("n_moves_72h") is not None else None,
                "as_of_date":      str(r["as_of_date"]) if r.get("as_of_date") is not None else None,
            })

        return {"teams": teams, "status": "ok"}
    except Exception:
        return {"teams": [], "status": "error"}


# ---------------------------------------------------------------------------
# Cortex leaderboard endpoints — EDGE, Goalie, Clutch, ST, Hot Hand, CDR, RAPM, xGA
# ---------------------------------------------------------------------------


@app.get("/phase2/edge-leaderboard")
async def phase2_edge_leaderboard(
    metric: str = Query("max_speed_kmh", description="max_speed_kmh | avg_speed_kmh | distance_per_game_km | max_shot_speed_mph | avg_shot_speed_mph | hard_shot_count"),
    limit: int = 20,
) -> dict:
    """EDGE skating/shot-speed leaderboard. All metrics are season aggregates."""
    import polars as pl
    edge_dir = _module_dir("edge")
    SHOT_METRICS = {"max_shot_speed_mph", "avg_shot_speed_mph", "hard_shot_count"}
    is_shot = metric in SHOT_METRICS
    pattern = "edge_shot_speed_*.parquet" if is_shot else "edge_skating_*.parquet"
    files = sorted(edge_dir.glob(pattern)) if edge_dir.exists() else []
    if not files:
        return {"players": [], "built": False, "metric": metric}
    try:
        df = pl.read_parquet(files[-1])
        if metric not in df.columns:
            return {"players": [], "built": False, "metric": metric, "reason": "metric_not_found"}
        # Min games played to qualify
        MIN_GP = 10
        df = df.filter(pl.col("games_played") >= MIN_GP).filter(pl.col(metric).is_not_null())
        ranked = df.sort(metric, descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            rows.append({
                "rank":        i + 1,
                "player_id":   r.get("player_id"),
                "player_name": r.get("player_name"),
                "team":        r.get("team"),
                "value":       round(float(r[metric]), 2) if r[metric] is not None else None,
                "games_played":r.get("games_played"),
            })
        return {"players": rows, "built": True, "metric": metric}
    except Exception:
        return {"players": [], "built": False, "metric": metric}


@app.get("/phase2/goalie-leaderboard")
async def phase2_goalie_leaderboard(
    metric: str = Query("gsax", description="gsax | hdsv_pct | sv_pct"),
    limit: int = 20,
) -> dict:
    """Goalie leaderboard from MoneyPuck goalie_stats (all situations, min 10 GP)."""
    import polars as pl
    goalie_dir = _module_dir("goalie_stats")
    files = sorted(goalie_dir.glob("goalie_stats_*.parquet")) if goalie_dir.exists() else []
    if not files:
        return {"players": [], "built": False, "metric": metric}
    try:
        df = pl.read_parquet(files[-1])
        df = df.filter(pl.col("situation") == "all") if "situation" in df.columns else df
        # Compute hdsv_pct if needed
        if "hdsv_pct" not in df.columns and "hd_saves" in df.columns and "hd_shots" in df.columns:
            df = df.with_columns(
                (pl.col("hd_saves") / pl.col("hd_shots").clip(lower_bound=1)).alias("hdsv_pct")
            )
        if "sv_pct" not in df.columns and "saves" in df.columns and "shots" in df.columns:
            df = df.with_columns(
                (pl.col("saves") / pl.col("shots").clip(lower_bound=1)).alias("sv_pct")
            )
        if metric not in df.columns:
            return {"players": [], "built": False, "metric": metric, "reason": "metric_not_found"}
        MIN_GP = 10
        df = df.filter(pl.col("games_played") >= MIN_GP).filter(pl.col(metric).is_not_null())
        ranked = df.sort(metric, descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            val = r.get(metric)
            rows.append({
                "rank":        i + 1,
                "player_id":   r.get("player_id"),
                "player_name": r.get("player_name"),
                "team":        _abbr(r.get("team")) if r.get("team") else r.get("team"),
                "value":       round(float(val), 4) if val is not None else None,
                "games_played":r.get("games_played"),
                "gsax":        round(float(r["gsax"]), 2) if r.get("gsax") is not None else None,
            })
        return {"players": rows, "built": True, "metric": metric}
    except Exception as exc:
        return {"players": [], "built": False, "metric": metric, "error": str(exc)}


@app.get("/phase2/clutch-leaderboard")
async def phase2_clutch_leaderboard(limit: int = 20) -> dict:
    """Clutch Index leaderboard (WPA above expected, Bayesian-shrunk)."""
    import polars as pl
    ci_dir = _GRETZKY_DATA_DIR / "clutch_index"
    files = sorted(ci_dir.glob("clutch_index_2*.parquet")) if ci_dir.exists() else []
    if not files:
        return {"players": [], "built": False}
    try:
        df = pl.read_parquet(files[-1])
        col = "clutch_index_shrunk" if "clutch_index_shrunk" in df.columns else "clutch_index"
        name_lut = _build_name_lookup()

        # Build player_id → team fallback from xg_finishing / RAPM parquets
        team_lut: dict[int, str] = {}
        for src_dir, glob_pat in [
            (_GRETZKY_DATA_DIR / "xg_finishing", "xg_finishing_*.parquet"),
            (_GRETZKY_DATA_DIR / "rapm",         "rapm_*.parquet"),
        ]:
            if src_dir.exists():
                for pf in sorted(src_dir.glob(glob_pat)):
                    try:
                        tdf = pl.read_parquet(pf)
                        team_col = next((c for c in ("team", "shooting_team") if c in tdf.columns), None)
                        id_col   = next((c for c in ("player_id", "shooter_id") if c in tdf.columns), None)
                        if team_col and id_col:
                            for row in tdf.select([id_col, team_col]).unique(subset=[id_col]).to_dicts():
                                pid2 = row.get(id_col)
                                tm   = row.get(team_col)
                                if pid2 and tm and int(pid2) not in team_lut:
                                    team_lut[int(pid2)] = str(tm)
                    except Exception:
                        pass
                break  # use first successful source

        MIN_TOI_HRS = 5.0  # toi_60 column is in hours (e.g. 35h max for top players)
        df = df.filter(pl.col(col).is_not_null())
        if "toi_60" in df.columns:
            df = df.filter(pl.col("toi_60") >= MIN_TOI_HRS)
        ranked = df.sort(col, descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            raw_team = r.get("team") or (team_lut.get(int(pid)) if pid else None)
            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        _abbr(raw_team),
                "value":       round(float(r[col]), 4),
                "wpa_per60":   round(float(r["actual_wpa_per60"]), 4) if r.get("actual_wpa_per60") is not None else None,
            })
        return {"players": rows, "built": True}
    except Exception:
        return {"players": [], "built": False}


@app.get("/phase2/special-teams-leaderboard")
async def phase2_special_teams_leaderboard(
    side: str = Query("pp", description="pp | pk"),
    limit: int = 20,
) -> dict:
    """Power Play or Penalty Kill xGF/60 leaderboard."""
    import polars as pl
    st_dir = _GRETZKY_DATA_DIR / "special_teams"
    files = sorted(st_dir.glob("special_teams_*.parquet")) if st_dir.exists() else []
    if not files:
        return {"players": [], "built": False, "side": side}
    try:
        df = pl.read_parquet(files[-1])
        toi_col  = "toi_pp"  if side == "pp" else "toi_pk"
        xgf_col  = "pp_xgf60" if side == "pp" else "pk_xgf60"
        rapm_col = "rapm_pp"  if side == "pp" else "rapm_pk"
        name_lut = _build_name_lookup()
        MIN_TOI = 50.0  # 50+ PP/PK minutes to qualify
        if toi_col not in df.columns or xgf_col not in df.columns:
            return {"players": [], "built": False, "side": side, "reason": "columns_missing"}
        df = (
            df.filter(pl.col(toi_col) >= MIN_TOI)
              .filter(pl.col(xgf_col).is_not_null())
        )
        ranked = df.sort(xgf_col, descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        _abbr(r.get("team")),
                "xgf60":       round(float(r[xgf_col]), 2),
                "rapm":        round(float(r[rapm_col]), 3) if r.get(rapm_col) is not None else None,
                "toi":         round(float(r[toi_col]), 1),
            })
        return {"players": rows, "built": True, "side": side}
    except Exception:
        return {"players": [], "built": False, "side": side}


@app.get("/phase2/hot-hand-leaderboard")
async def phase2_hot_hand_leaderboard(limit: int = 20) -> dict:
    """Hot Hand Signal leaderboard — 5-game burst goals-vs-xG z-score."""
    import polars as pl
    hh_dir = _GRETZKY_DATA_DIR / "hot_hand"
    files = sorted(hh_dir.glob("hot_hand_summary_*.parquet")) if hh_dir.exists() else []
    if not files:
        return {"players": [], "built": False}
    try:
        df = pl.read_parquet(files[-1])
        name_lut = _build_name_lookup()
        MIN_GP = 5
        if "games_played" in df.columns:
            df = df.filter(pl.col("games_played") >= MIN_GP)
        df = df.filter(pl.col("hot_hand_score").is_not_null())
        ranked = df.sort("hot_hand_score", descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"

            # Try to get team from RAPM parquet (hot_hand doesn't carry team)
            team = None
            rapm_dir = _GRETZKY_DATA_DIR / "rapm"
            rapm_files = sorted(rapm_dir.glob("rapm_*.parquet")) if rapm_dir.exists() else []
            if rapm_files and pid:
                try:
                    rdf = pl.read_parquet(rapm_files[-1], columns=["player_id", "team"])
                    t_rows = rdf.filter(pl.col("player_id") == pid).to_dicts()
                    if t_rows:
                        team = _abbr(t_rows[0].get("team"))
                except Exception:
                    pass

            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        team,
                "value":       round(float(r["hot_hand_score"]), 3),
                "goals_5g":    r.get("goals_5g"),
                "xg_5g":       round(float(r["xg_5g"]), 2) if r.get("xg_5g") is not None else None,
            })
        return {"players": rows, "built": True}
    except Exception:
        return {"players": [], "built": False}


@app.get("/phase2/xg-leaderboard")
async def phase2_xg_leaderboard(limit: int = 20) -> dict:
    """xGF/60 leaderboard from EWMA form model (min 20 games)."""
    import polars as pl
    ewma_dir = _GRETZKY_DATA_DIR / "ewma"
    files = sorted(ewma_dir.glob("ewma_form_*.parquet")) if ewma_dir.exists() else []
    if not files:
        return {"players": [], "built": False}
    try:
        df = pl.read_parquet(files[-1])
        ewma_col = "ewma_xgf60" if "ewma_xgf60" in df.columns else "current_ewma"
        if ewma_col not in df.columns:
            return {"players": [], "built": False}
        name_lut = _build_name_lookup()

        MIN_GP = 20
        if "games_played" in df.columns:
            df = df.filter(pl.col("games_played") >= MIN_GP)
        df = df.filter(pl.col(ewma_col).is_not_null())
        ranked = df.sort(ewma_col, descending=True).head(limit)

        # Build team lookup from RAPM
        team_lut: dict[int, str | None] = {}
        rapm_dir = _GRETZKY_DATA_DIR / "rapm"
        rapm_files = sorted(rapm_dir.glob("rapm_*.parquet")) if rapm_dir.exists() else []
        if rapm_files:
            try:
                rdf = pl.read_parquet(rapm_files[-1], columns=["player_id", "team"])
                for r in rdf.to_dicts():
                    pid2 = r.get("player_id")
                    if pid2:
                        team_lut[int(pid2)] = _abbr(r.get("team"))
            except Exception:
                pass

        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            team = team_lut.get(int(pid)) if pid else None
            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        team,
                "value":       round(float(r[ewma_col]), 2),
                "games_played":r.get("games_played"),
            })
        return {"players": rows, "built": True}
    except Exception:
        return {"players": [], "built": False}


@app.get("/phase2/cdr-leaderboard")
async def phase2_cdr_leaderboard(limit: int = 20) -> dict:
    """Composite Defensive Rating leaderboard (higher = better defender)."""
    import polars as pl
    cdr_dir = _GRETZKY_DATA_DIR / "cdr"
    files = sorted(cdr_dir.glob("cdr_*.parquet")) if cdr_dir.exists() else []
    if not files:
        return {"players": [], "built": False}
    try:
        df = pl.read_parquet(files[-1])
        name_lut = _build_name_lookup()
        MIN_GP = 20
        if "gp" in df.columns:
            df = df.filter(pl.col("gp") >= MIN_GP)
        df = df.filter(pl.col("cdr").is_not_null())
        ranked = df.sort("cdr", descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        _abbr(r.get("team")),
                "value":       round(float(r["cdr"]), 3),
                "xga_60":      round(float(r["xga_60"]), 2) if r.get("xga_60") is not None else None,
                "tk_gv_ratio": round(float(r["tk_gv_ratio"]), 2) if r.get("tk_gv_ratio") is not None else None,
                "gp":          r.get("gp"),
            })
        return {"players": rows, "built": True}
    except Exception:
        return {"players": [], "built": False}


@app.get("/phase2/rapm-leaderboard")
async def phase2_rapm_leaderboard(
    category: str = Query("ev_off", description="ev_off | ev_def | pp | pk"),
    limit: int = 20,
) -> dict:
    """RAPM leaderboard by category (EV Off, EV Def, PP, PK)."""
    import polars as pl
    rapm_dir = _GRETZKY_DATA_DIR / "rapm"
    files = sorted(rapm_dir.glob("rapm_*.parquet")) if rapm_dir.exists() else []
    if not files:
        return {"players": [], "built": False, "category": category}
    _COL = {"ev_off": "rapm_ev_off", "ev_def": "rapm_ev_def", "pp": "rapm_pp", "pk": "rapm_pk"}
    col = _COL.get(category, "rapm_ev_off")
    try:
        df = pl.read_parquet(files[-1])
        if col not in df.columns:
            return {"players": [], "built": False, "category": category, "reason": "column_missing"}
        name_lut = _build_name_lookup()
        MIN_GP = 20
        if "gp" in df.columns:
            df = df.filter(pl.col("gp") >= MIN_GP)
        df = df.filter(pl.col(col).is_not_null())
        ranked = df.sort(col, descending=True).head(limit)
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        _abbr(r.get("team")),
                "value":       round(float(r[col]), 3),
                "gp":          r.get("gp"),
                "toi_ev":      round(float(r["toi_ev"]), 1) if r.get("toi_ev") is not None else None,
            })
        return {"players": rows, "built": True, "category": category}
    except Exception:
        return {"players": [], "built": False, "category": category}


@app.get("/phase2/xga-leaderboard")
async def phase2_xga_leaderboard(limit: int = 20) -> dict:
    """xGA/60 leaderboard — best defenders by lowest on-ice xGA/60 (DZS-adjusted from CDR model, min 20 GP)."""
    import polars as pl
    # Use CDR parquet — it has xga_60_adj (DZS-corrected on-ice xGA/60), which is
    # the real on-ice suppression number. RAPM xga_60 is a marginal ridge-regression
    # differential and is near zero by construction; not useful as a leaderboard.
    cdr_dir = _GRETZKY_DATA_DIR / "cdr"
    files = sorted(cdr_dir.glob("cdr_*.parquet")) if cdr_dir.exists() else []
    if not files:
        return {"players": [], "built": False}
    try:
        df = pl.read_parquet(files[-1])
        col = "xga_60_adj" if "xga_60_adj" in df.columns else "xga_60"
        if col not in df.columns:
            return {"players": [], "built": False, "reason": "xga_60_missing"}
        name_lut = _build_name_lookup()
        MIN_GP = 20
        if "gp" in df.columns:
            df = df.filter(pl.col("gp") >= MIN_GP)
        df = df.filter(pl.col(col).is_not_null()).filter(pl.col(col) > 0)
        # Lower xGA/60 = better defensive suppression; sort ascending
        ranked = df.sort(col, descending=False).head(limit)
        # Build RAPM EV Def lookup for context column
        rapm_lut: dict[int, float | None] = {}
        rapm_dir = _GRETZKY_DATA_DIR / "rapm"
        rapm_files = sorted(rapm_dir.glob("rapm_*.parquet")) if rapm_dir.exists() else []
        if rapm_files:
            try:
                rdf = pl.read_parquet(rapm_files[-1], columns=["player_id", "rapm_ev_def"])
                for r in rdf.to_dicts():
                    pid2 = r.get("player_id")
                    if pid2:
                        rapm_lut[int(pid2)] = r.get("rapm_ev_def")
            except Exception:
                pass
        rows = []
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            rapm_def = rapm_lut.get(int(pid)) if pid else None
            rows.append({
                "rank":       i + 1,
                "player_id":  pid,
                "player_name":name,
                "team":       _abbr(r.get("team")),
                "value":      round(float(r[col]), 2),
                "gp":         r.get("gp"),
                "rapm_ev_def":round(float(rapm_def), 3) if rapm_def is not None else None,
            })
        return {"players": rows, "built": True}
    except Exception as exc:
        return {"players": [], "built": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Player recent game log — NHL API passthrough (last N games)
# ---------------------------------------------------------------------------

@app.get("/player-gamelog/{player_id}")
async def player_gamelog(player_id: int, limit: int = 5) -> dict:
    """Fetch recent game log for a player from the NHL API.
    Returns last `limit` regular-season games with goals, assists, TOI, opponent.
    """
    cache_key = f"gamelog:{player_id}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        async with httpx.AsyncClient(
            base_url="https://api-web.nhle.com",
            timeout=8.0,
            headers={"User-Agent": "grtzky/1.0"},
            follow_redirects=True,
        ) as client:
            resp = await client.get(f"/v1/player/{player_id}/game-log/now")
            resp.raise_for_status()
            data = resp.json()

        games_raw = data.get("gameLog", [])
        # Filter to regular season (gameTypeId 2) only, most recent first
        regular = [g for g in games_raw if g.get("gameTypeId", 2) == 2]
        recent = regular[:limit]

        games = []
        for g in recent:
            toi_raw = g.get("toi", "0:00")
            games.append({
                "game_id":   g.get("gameId"),
                "date":      g.get("gameDate"),
                "opponent":  g.get("opponentAbbrev"),
                "home_road": g.get("homeRoadFlag"),
                "goals":     g.get("goals", 0),
                "assists":   g.get("assists", 0),
                "points":    g.get("points", 0),
                "shots":     g.get("shots", 0),
                "toi":       toi_raw,
                "plus_minus": g.get("plusMinus", 0),
                "pp_goals":  g.get("powerPlayGoals", 0),
                "pp_points": g.get("powerPlayPoints", 0),
            })

        # Aggregate last-N summary
        total_g  = sum(x["goals"]   for x in games)
        total_a  = sum(x["assists"] for x in games)
        total_pts = sum(x["points"] for x in games)
        n = len(games)

        result: dict = {
            "player_id": player_id,
            "games":     games,
            "summary": {
                "n_games":  n,
                "goals":    total_g,
                "assists":  total_a,
                "points":   total_pts,
                "gpg":      round(total_g  / n, 2) if n else 0,
                "apg":      round(total_a  / n, 2) if n else 0,
                "ppg":      round(total_pts / n, 2) if n else 0,
            },
            "status": "ok",
        }
        _cache_set(cache_key, result, ttl=180.0)
        return result

    except Exception as exc:
        return {"player_id": player_id, "games": [], "summary": {}, "status": "error", "error": str(exc)}


@app.get("/player-profile/{player_id}")
async def player_profile(player_id: int) -> dict:
    """Comprehensive player profile — all phase model data + NHL bio in a single call.

    Fetches (all null-safe if parquet missing):
    - Bio from NHL API (name, team, position, age, height, weight, draft)
    - xG Finishing, RAPM, CDR, WAR, Archetype, EWMA, Hot Hand, Clutch
    - Special Teams PP/PK, Bayesian Rating, Playoff Delta, Former Team Boost
    - Skating Baseline (speed, distance, zone times)
    - Puck Battle Rating (battle_score, battle_percentile, physical metrics)
    - Behavioral NN tendencies (carry_in, dump, shoot_slot, net drive, etc.)
    - In-Season Bayesian Blend (mu_blend, CI bands)
    - Recent game log (last 10 games)
    - WAR rank context
    """
    import polars as pl

    cache_key = f"profile:{player_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    result: dict = {"player_id": player_id, "status": "ok"}

    # ── 1. NHL API bio ─────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(
            base_url="https://api-web.nhle.com",
            timeout=8.0,
            headers={"User-Agent": "grtzky/1.0"},
            follow_redirects=True,
        ) as client:
            bio_resp = await client.get(f"/v1/player/{player_id}/landing")
            bio_resp.raise_for_status()
            bio = bio_resp.json()

        result.update({
            "player_name":    bio.get("firstName", {}).get("default", "") + " " + bio.get("lastName", {}).get("default", ""),
            "team":           bio.get("currentTeamAbbrev"),
            "position":       bio.get("position"),
            "jersey_number":  bio.get("sweaterNumber"),
            "height_cm":      bio.get("heightInCentimeters"),
            "weight_kg":      bio.get("weightInKilograms"),
            "shoots_catches": bio.get("shootsCatches"),
            "birth_date":     bio.get("birthDate"),
            "birth_city":     bio.get("birthCity", {}).get("default"),
            "birth_country":  bio.get("birthCountryCode"),
            "draft_year":     bio.get("draftDetails", {}).get("year"),
            "draft_round":    bio.get("draftDetails", {}).get("round"),
            "draft_pick":     bio.get("draftDetails", {}).get("pickInRound"),
            "draft_team":     bio.get("draftDetails", {}).get("teamAbbrev"),
            "headshot":       bio.get("headshot"),
            "hero_image":     bio.get("heroImage"),
            "nhl_games_played": bio.get("careerTotals", {}).get("regularSeason", {}).get("gamesPlayed"),
            "nhl_career_goals": bio.get("careerTotals", {}).get("regularSeason", {}).get("goals"),
            "nhl_career_points": bio.get("careerTotals", {}).get("regularSeason", {}).get("points"),
        })
    except Exception:
        pass  # bio fields remain unset

    # Ensure we at least have a name from our lookup
    if not result.get("player_name", "").strip():
        result["player_name"] = _build_name_lookup().get(player_id, f"Player {player_id}")

    # ── 2. xG Finishing ────────────────────────────────────────────────────────
    xg_dir = _GRETZKY_DATA_DIR / "xg_finishing"
    xg_parquets = sorted(xg_dir.glob("xg_finishing_*.parquet")) if xg_dir.exists() else []
    season_val: int | None = None
    if xg_parquets:
        try:
            df = pl.read_parquet(xg_parquets[-1])
            row = df.filter(pl.col("shooter_id") == player_id) if "shooter_id" in df.columns else pl.DataFrame()
            if not row.is_empty():
                r = row.to_dicts()[0]
                season_val = r.get("season")
                result.update({
                    "season":          season_val,
                    "shots":           r.get("shots"),
                    "goals":           r.get("goals"),
                    "xg_sum":          round(r["xg_sum"], 3) if r.get("xg_sum") is not None else None,
                    "finishing":       round(r["finishing"], 3) if r.get("finishing") is not None else None,
                    "finishing_per60": round(r["finishing_per60"], 3) if r.get("finishing_per60") is not None else None,
                })
                if not result.get("team"):
                    result["team"] = r.get("team")
        except Exception:
            pass

    # Goalie fallback
    is_goalie = False
    if not result.get("shots"):
        goalie_dir = _module_dir("goalie_stats")
        goalie_parquets = sorted(goalie_dir.glob("goalie_stats_*.parquet")) if goalie_dir.exists() else []
        if goalie_parquets:
            try:
                gdf = pl.read_parquet(goalie_parquets[-1])
                grow = gdf.filter(pl.col("player_id") == player_id)
                if not grow.is_empty():
                    is_goalie = True
                    sit_row = grow.filter(pl.col("situation") == "all") if "situation" in grow.columns else grow
                    gr = (sit_row if not sit_row.is_empty() else grow).to_dicts()[0]
                    result.update({
                        "is_goalie":      True,
                        "season":         gr.get("season"),
                        "games_played":   gr.get("games_played"),
                        "shots_against":  gr.get("shots"),
                        "saves":          gr.get("saves"),
                        "goals_against":  gr.get("goals_against"),
                        "sv_pct":         gr.get("sv_pct"),
                        "xga":            gr.get("xga"),
                        "gsax":           gr.get("gsax"),
                        "hd_shots":       gr.get("hd_shots"),
                        "hd_saves":       gr.get("hd_saves"),
                        "hdsv_pct":       gr.get("hdsv_pct"),
                        "mdsv_pct":       gr.get("mdsv_pct"),
                        "ldsv_pct":       gr.get("ldsv_pct"),
                    })
            except Exception:
                pass

    if not is_goalie:
        # ── 3. CDR ─────────────────────────────────────────────────────────────
        try:
            from models.defensive_rating import read_cdr, lookup_player as _lookup_cdr
            cdr_df = read_cdr(_GRETZKY_DATA_DIR, season=season_val)
            if cdr_df is not None:
                name_for_cdr = result.get("player_name", "")
                cdr_row = _lookup_cdr(cdr_df, name_for_cdr)
                if not cdr_row.is_empty():
                    raw = cdr_row["cdr"][0]
                    result["cdr"] = round(float(raw), 3) if raw is not None else None
        except Exception:
            pass

        # ── 4. RAPM ────────────────────────────────────────────────────────────
        try:
            rapm_dir = _GRETZKY_DATA_DIR / "rapm"
            rapm_path = rapm_dir / f"rapm_{season_val}.parquet" if season_val else None
            if rapm_path and rapm_path.exists():
                rapm_df = pl.read_parquet(rapm_path)
                rapm_row = rapm_df.filter(pl.col("player_id") == player_id)
                if not rapm_row.is_empty():
                    toi_ev_val = round(float(rapm_row["toi_ev"][0]) / 60, 1) if "toi_ev" in rapm_df.columns else None
                    shots_p60  = round(float(rapm_row["shots_per60"][0]), 2) if "shots_per60" in rapm_df.columns else None
                    goals_p60  = round(float(rapm_row["goals_per60"][0]), 2) if "goals_per60" in rapm_df.columns else None
                    if "xgf_60" in rapm_df.columns:
                        xgf_p60 = round(float(rapm_row["xgf_60"][0]), 2)
                    elif "xgf_per60" in rapm_df.columns:
                        xgf_p60 = round(float(rapm_row["xgf_per60"][0]), 2)
                    else:
                        xgf_p60 = None

                    # Derive per-60 from raw counts + toi_ev when RAPM parquet lacks them
                    toi_h = (toi_ev_val or 0) / 3600
                    if goals_p60 is None and toi_h > 0:
                        raw_goals = result.get("goals")
                        if raw_goals is not None:
                            goals_p60 = round(float(raw_goals) / toi_h, 2)
                    if shots_p60 is None and toi_h > 0:
                        raw_shots = result.get("shots")
                        if raw_shots is not None:
                            shots_p60 = round(float(raw_shots) / toi_h, 2)

                    result.update({
                        "rapm_ev_off": round(float(rapm_row["rapm_ev_off"][0]), 3),
                        "rapm_ev_def": round(float(rapm_row["rapm_ev_def"][0]), 3),
                        "rapm_xga_60": round(float(rapm_row["xga_60"][0]), 2) if "xga_60" in rapm_df.columns else None,
                        "shots_per60": shots_p60,
                        "goals_per60": goals_p60,
                        "xgf_per60":   xgf_p60,
                        "toi_ev":      toi_ev_val,
                    })
        except Exception:
            pass

        # ── 5. WAR ─────────────────────────────────────────────────────────────
        try:
            war_dir = _GRETZKY_DATA_DIR / "war"
            war_parquets = sorted(war_dir.glob("war_*.parquet")) if war_dir.exists() else []
            if war_parquets:
                war_df = pl.read_parquet(war_parquets[-1])
                war_row = war_df.filter(pl.col("player_id") == player_id)
                if not war_row.is_empty():
                    w = war_row.to_dicts()[0]
                    result.update({
                        "war":                 round(float(w["war"]), 2) if w.get("war") is not None else None,
                        "gar":                 round(float(w["gar_total"]), 2) if w.get("gar_total") is not None else None,
                        "contract_efficiency": round(float(w["contract_efficiency"]), 2) if w.get("contract_efficiency") is not None else None,
                    })
                # Compute WAR rank
                ranked = (
                    war_df.filter(pl.col("war").is_not_null())
                    .pipe(lambda d: d.filter(pl.col("toi_ev") >= 200) if "toi_ev" in d.columns else d)
                    .sort("war", descending=True)
                )
                ids = ranked["player_id"].to_list()
                if player_id in ids:
                    result["war_rank"] = ids.index(player_id) + 1
                    result["war_total_qualified"] = len(ids)
        except Exception:
            pass

        # ── 6. Archetype ───────────────────────────────────────────────────────
        try:
            arch_dir = _GRETZKY_DATA_DIR / "archetypes"
            arch_parquets = sorted(arch_dir.glob("archetype_assignments_*.parquet")) if arch_dir.exists() else []
            if arch_parquets:
                arch_df = pl.read_parquet(arch_parquets[-1])
                arch_row = arch_df.filter(pl.col("player_id") == player_id)
                if not arch_row.is_empty():
                    a = arch_row.to_dicts()[0]
                    result.update({
                        "archetype_id":   int(a["cluster_id"]) if a.get("cluster_id") is not None else None,
                        "archetype_name": str(a.get("archetype") or a.get("archetype_name") or "") or None,
                    })
        except Exception:
            pass

        # ── 7. EWMA ────────────────────────────────────────────────────────────
        try:
            ewma_dir = _GRETZKY_DATA_DIR / "ewma"
            ewma_parquets = sorted(ewma_dir.glob("ewma_form_*.parquet")) if ewma_dir.exists() else []
            if ewma_parquets:
                ewma_df = pl.read_parquet(ewma_parquets[-1])
                ewma_row = ewma_df.filter(pl.col("player_id") == player_id)
                if not ewma_row.is_empty():
                    e = ewma_row.to_dicts()[0]
                    raw_ewma = e.get("ewma_xgf60") or e.get("current_ewma") or e.get("xgf_per60")
                    result.update({
                        "ewma_xgf60":    round(float(raw_ewma), 3) if raw_ewma is not None else None,
                        "ewma_form_flag": str(e["form_flag"]) if e.get("form_flag") else None,
                        "ewma_games":    int(e["games_processed"]) if e.get("games_processed") is not None else None,
                    })
        except Exception:
            pass

        # ── 8. Hot Hand ────────────────────────────────────────────────────────
        try:
            hh_dir = _GRETZKY_DATA_DIR / "hot_hand"
            hh_parquets = sorted(hh_dir.glob("hot_hand_summary_*.parquet")) if hh_dir.exists() else []
            if hh_parquets:
                hh_df = pl.read_parquet(hh_parquets[-1])
                hh_row = hh_df.filter(pl.col("player_id") == player_id)
                if not hh_row.is_empty():
                    hh = hh_row.to_dicts()[0]
                    result.update({
                        "hot_hand_score":  round(float(hh["hot_hand_score"]), 3) if hh.get("hot_hand_score") is not None else None,
                        "hot_hand_goals5": round(float(hh["goals_5g"]), 2)       if hh.get("goals_5g")       is not None else None,
                        "hot_hand_xg5":    round(float(hh["xg_5g"]), 2)          if hh.get("xg_5g")          is not None else None,
                    })
        except Exception:
            pass

        # ── 9. Clutch ──────────────────────────────────────────────────────────
        try:
            ci_dir = _GRETZKY_DATA_DIR / "clutch_index"
            ci_parquets = sorted(ci_dir.glob("clutch_index_[0-9]*.parquet")) if ci_dir.exists() else []
            if ci_parquets:
                ci_df = pl.read_parquet(ci_parquets[-1])
                ci_row = ci_df.filter(pl.col("player_id") == player_id)
                if not ci_row.is_empty():
                    ci = ci_row.to_dicts()[0]
                    result.update({
                        "clutch_index":    round(float(ci["clutch_index_shrunk"]), 4) if ci.get("clutch_index_shrunk") is not None else None,
                        "clutch_wpa_per60": round(float(ci["actual_wpa_per60"]), 4)   if ci.get("actual_wpa_per60")   is not None else None,
                    })
        except Exception:
            pass

        # ── 10. Special Teams ──────────────────────────────────────────────────
        try:
            st_dir = _GRETZKY_DATA_DIR / "special_teams"
            st_parquets = sorted(st_dir.glob("special_teams_*.parquet")) if st_dir.exists() else []
            if st_parquets:
                st_df = pl.read_parquet(st_parquets[-1])
                st_row = st_df.filter(pl.col("player_id") == player_id)
                if not st_row.is_empty():
                    st = st_row.to_dicts()[0]
                    result.update({
                        "special_teams_pp": round(float(st["pp_rating"]), 3) if st.get("pp_rating") is not None else None,
                        "special_teams_pk": round(float(st["pk_rating"]), 3) if st.get("pk_rating") is not None else None,
                    })
        except Exception:
            pass

        # ── 11. Bayesian Rating ────────────────────────────────────────────────
        try:
            br_dir = _GRETZKY_DATA_DIR / "bayes_ratings"
            br_parquets = sorted(br_dir.glob("player_ratings_*.parquet")) if br_dir.exists() else []
            if br_parquets:
                br_df = pl.read_parquet(br_parquets[-1])
                br_row = br_df.filter(pl.col("player_id") == player_id)
                if not br_row.is_empty():
                    br = br_row.to_dicts()[0]
                    result.update({
                        "bayesian_rating":      round(float(br["posterior_mean"]),  3) if br.get("posterior_mean")  is not None else None,
                        "bayesian_uncertainty": round(float(br["posterior_sigma"]), 3) if br.get("posterior_sigma") is not None else None,
                    })
        except Exception:
            pass

        # ── 12. Playoff Delta ──────────────────────────────────────────────────
        try:
            pd_dir = _GRETZKY_DATA_DIR / "playoff_delta"
            pd_parquets = sorted(pd_dir.glob("playoff_delta_*.parquet")) if pd_dir.exists() else []
            if pd_parquets:
                pd_df = pl.read_parquet(pd_parquets[-1])
                pd_row = pd_df.filter(pl.col("player_id") == player_id)
                if not pd_row.is_empty():
                    pd_r = pd_row.to_dicts()[0]
                    reg_xgf = pd_r.get("reg_xgf_per60")
                    po_xgf  = pd_r.get("playoff_xgf_per60")
                    if reg_xgf is not None and po_xgf is not None and float(reg_xgf) > 0:
                        result["playoff_delta"] = round(float(po_xgf) - float(reg_xgf), 3)
        except Exception:
            pass

        # ── 13. Former Team Boost ──────────────────────────────────────────────
        try:
            ftb_dir = _GRETZKY_DATA_DIR / "former_team_boost"
            ftb_parquets = sorted(ftb_dir.glob("former_team_boost_*.parquet")) if ftb_dir.exists() else []
            if ftb_parquets:
                ftb_df = pl.read_parquet(ftb_parquets[-1])
                ftb_row = ftb_df.filter(pl.col("player_id") == player_id)
                if not ftb_row.is_empty():
                    ftb = ftb_row.to_dicts()[0]
                    result.update({
                        "former_team_boost": round(float(ftb["base_boost"]), 3) if ftb.get("base_boost") is not None else None,
                        "former_team":       str(ftb["former_team"]) if ftb.get("former_team") else None,
                    })
        except Exception:
            pass

        # ── 14. Skating Baseline ───────────────────────────────────────────────
        try:
            skate_dir = _GRETZKY_DATA_DIR / "skating_baseline"
            skate_parquets = sorted(skate_dir.glob("skating_baseline_*.parquet")) if skate_dir.exists() else []
            if skate_parquets:
                sk_df = pl.read_parquet(skate_parquets[-1])
                sk_row = sk_df.filter(pl.col("player_id") == player_id)
                if not sk_row.is_empty():
                    sk = sk_row.to_dicts()[0]
                    result.update({
                        "skating_avg_speed_kmh":         round(float(sk["baseline_avg_speed_kmh"]), 1)          if sk.get("baseline_avg_speed_kmh")          is not None else None,
                        "skating_max_speed_kmh":         round(float(sk["baseline_max_speed_kmh"]), 1)          if sk.get("baseline_max_speed_kmh")          is not None else None,
                        "skating_distance_per_game_km":  round(float(sk["baseline_distance_per_game_km"]), 2)   if sk.get("baseline_distance_per_game_km")   is not None else None,
                        "skating_zone_time_oz_pct":      round(float(sk["baseline_zone_time_pct_oz"]) * 100, 1) if sk.get("baseline_zone_time_pct_oz")       is not None else None,
                        "skating_zone_time_dz_pct":      round(float(sk["baseline_zone_time_pct_dz"]) * 100, 1) if sk.get("baseline_zone_time_pct_dz")       is not None else None,
                        "skating_games_sample":          int(sk["n_games_total"])                                if sk.get("n_games_total")                  is not None else None,
                    })
        except Exception:
            pass

        # ── 15. Puck Battles ───────────────────────────────────────────────────
        try:
            battles_dir = _GRETZKY_DATA_DIR / "battles"
            battle_parquets = sorted(battles_dir.glob("puck_battle_*.parquet")) if battles_dir.exists() else []
            if battle_parquets:
                b_df = pl.read_parquet(battle_parquets[-1])
                b_row = b_df.filter(pl.col("player_id") == player_id) if "player_id" in b_df.columns else pl.DataFrame()
                if not b_row.is_empty():
                    b = b_row.to_dicts()[0]
                    result.update({
                        "battle_score":       round(float(b["battle_score"]), 3)       if b.get("battle_score")       is not None else None,
                        "battle_percentile":  round(float(b["battle_percentile"]), 1)  if b.get("battle_percentile")  is not None else None,
                        "hits_per60":         round(float(b["hits_per60"]), 2)         if b.get("hits_per60")         is not None else None,
                        "blocks_per60":       round(float(b["blocks_per60"]), 2)       if b.get("blocks_per60")       is not None else None,
                        "carry_entry_pct":    round(float(b["carry_entry_pct"]) * 100, 1) if b.get("carry_entry_pct") is not None else None,
                        "net_front_pct":      round(float(b["net_front_pct"]) * 100, 1)   if b.get("net_front_pct")  is not None else None,
                    })
        except Exception:
            pass

        # ── 16. Behavioral NN ──────────────────────────────────────────────────
        try:
            beh_dir = _GRETZKY_DATA_DIR / "behavior_net"
            beh_parquets = sorted(beh_dir.glob("behavior_predictions_*.parquet")) if beh_dir.exists() else []
            if beh_parquets:
                beh_df = pl.read_parquet(beh_parquets[-1])
                beh_row = beh_df.filter(pl.col("player_id") == player_id) if "player_id" in beh_df.columns else pl.DataFrame()
                if not beh_row.is_empty():
                    beh = beh_row.to_dicts()[0]
                    result.update({
                        "nn_carry_in_pct":       round(float(beh["carry_in"]) * 100, 1)         if beh.get("carry_in")        is not None else None,
                        "nn_dump_pct":           round(float(beh["dump"]) * 100, 1)             if beh.get("dump")            is not None else None,
                        "nn_shoot_slot_pct":     round(float(beh["shoot_slot"]) * 100, 1)       if beh.get("shoot_slot")      is not None else None,
                        "nn_shoot_perimeter_pct": round(float(beh["shoot_perimeter"]) * 100, 1) if beh.get("shoot_perimeter") is not None else None,
                        "nn_drive_net_pct":      round(float(beh["drive_net"]) * 100, 1)        if beh.get("drive_net")       is not None else None,
                        "nn_battle_corner_pct":  round(float(beh["battle_corner"]) * 100, 1)    if beh.get("battle_corner")   is not None else None,
                        "nn_hold_corner_pct":    round(float(beh["hold_corner"]) * 100, 1)      if beh.get("hold_corner")     is not None else None,
                        "nn_fi_score":           round(float(beh["fi_score"]), 3)               if beh.get("fi_score")        is not None else None,
                    })
        except Exception:
            pass

        # ── 17. In-Season Bayesian Blend ───────────────────────────────────────
        try:
            inseason_dir = _GRETZKY_DATA_DIR / "inseason"
            inseason_parquets = sorted(inseason_dir.glob("inseason_blend_*.parquet")) if inseason_dir.exists() else []
            if inseason_parquets:
                is_df = pl.read_parquet(inseason_parquets[-1])
                is_row = is_df.filter(pl.col("player_id") == player_id)
                if not is_row.is_empty():
                    ib = is_row.to_dicts()[0]
                    result.update({
                        "inseason_mu_blend":   round(float(ib["mu_blend"]), 3)    if ib.get("mu_blend")    is not None else None,
                        "inseason_ci_lower":   round(float(ib["ci_lower_95"]), 3) if ib.get("ci_lower_95") is not None else None,
                        "inseason_ci_upper":   round(float(ib["ci_upper_95"]), 3) if ib.get("ci_upper_95") is not None else None,
                        "inseason_games":      int(ib["games_played"])            if ib.get("games_played") is not None else None,
                        "inseason_blend_weight": round(float(ib["blend_weight"]), 3) if ib.get("blend_weight") is not None else None,
                    })
        except Exception:
            pass

    # ── 18. Game Log (last 10) ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(
            base_url="https://api-web.nhle.com",
            timeout=8.0,
            headers={"User-Agent": "grtzky/1.0"},
            follow_redirects=True,
        ) as client:
            gl_resp = await client.get(f"/v1/player/{player_id}/game-log/now")
            gl_resp.raise_for_status()
            gl_data = gl_resp.json()

        games_raw = gl_data.get("gameLog", [])
        regular = [g for g in games_raw if g.get("gameTypeId", 2) == 2][:10]
        games = []
        for g in regular:
            games.append({
                "game_id":    g.get("gameId"),
                "date":       g.get("gameDate"),
                "opponent":   g.get("opponentAbbrev"),
                "home_road":  g.get("homeRoadFlag"),
                "goals":      g.get("goals", 0),
                "assists":    g.get("assists", 0),
                "points":     g.get("points", 0),
                "shots":      g.get("shots", 0),
                "toi":        g.get("toi", "0:00"),
                "plus_minus": g.get("plusMinus", 0),
                "pp_goals":   g.get("powerPlayGoals", 0),
                "pp_points":  g.get("powerPlayPoints", 0),
            })
        n = len(games)
        total_g = sum(x["goals"] for x in games)
        total_a = sum(x["assists"] for x in games)
        result["game_log"] = {
            "games": games,
            "summary": {
                "n_games": n, "goals": total_g, "assists": total_a, "points": total_g + total_a,
                "gpg": round(total_g / n, 2) if n else 0,
                "apg": round(total_a / n, 2) if n else 0,
                "ppg": round((total_g + total_a) / n, 2) if n else 0,
            },
        }
    except Exception:
        result["game_log"] = {"games": [], "summary": {}}

    # ── 19. Line Pairs (best linemates for this player) ────────────────────────
    if not is_goalie:
        try:
            pair_dir = _GRETZKY_DATA_DIR / "chemistry"
            pair_parquets = sorted(pair_dir.glob("pair_chemistry_*.parquet")) if pair_dir.exists() else []
            name_lut = _build_name_lookup()
            if pair_parquets:
                pc_df = pl.read_parquet(pair_parquets[-1])
                # Collect all rows where player appears on either side, keeping partner_id
                pairs_seen: dict[int, dict] = {}
                for row in pc_df.filter(
                    (pl.col("player_a_id") == player_id) | (pl.col("player_b_id") == player_id)
                ).to_dicts():
                    partner_id = int(row["player_b_id"]) if int(row["player_a_id"]) == player_id else int(row["player_a_id"])
                    delta = row.get("chemistry_delta")
                    if partner_id not in pairs_seen or (delta or 0) > (pairs_seen[partner_id]["chemistry_delta"] or 0):
                        pairs_seen[partner_id] = {
                            "partner_id":      partner_id,
                            "partner_name":    name_lut.get(partner_id, f"Player {partner_id}"),
                            "games_together":  row.get("games_together"),
                            "chemistry_delta": round(float(delta), 3) if delta is not None else None,
                            "model_xgf_pct":   round(float(row["model_xgf_pct"]) * 100, 1) if row.get("model_xgf_pct") is not None else None,
                            "co_toi_ev":       round(float(row["co_toi_ev"]), 0) if row.get("co_toi_ev") is not None else None,
                        }
                result["line_pairs"] = sorted(pairs_seen.values(), key=lambda x: x["chemistry_delta"] or 0, reverse=True)[:5]
        except Exception:
            result["line_pairs"] = []

    _cache_set(cache_key, result, ttl=120.0)
    return result


@app.get("/phase2/players")
async def phase2_players() -> dict:
    """Known player names for client-side autocomplete in the Phase 2 rating lookup.

    Sources skaters from xg_finishing parquet and goalies from goalie_stats parquet.
    """
    import polars as pl

    players: dict[str, dict] = {}

    # Skaters from xg_finishing (include shooter_id as player_id)
    xg_dir = _GRETZKY_DATA_DIR / "xg_finishing"
    xg_parquets = sorted(xg_dir.glob("xg_finishing_*.parquet")) if xg_dir.exists() else []
    if xg_parquets:
        df = pl.read_parquet(xg_parquets[-1])
        if "shooter_name" in df.columns:
            has_id = "shooter_id" in df.columns
            cols = ["shooter_name"] + (["shooter_id"] if has_id else []) + (["team"] if "team" in df.columns else [])
            pos_map = _shots_name_position_map()
            for r in df.select(cols).drop_nulls(subset=["shooter_name"]).unique(subset=["shooter_name"]).to_dicts():
                name = r["shooter_name"]
                pid = int(r["shooter_id"]) if has_id and r.get("shooter_id") is not None else None
                players[name] = {"name": name, "team": r.get("team") or "", "position": pos_map.get(name, ""), "player_id": pid}

    # Goalies from goalie_stats
    goalie_dir = _module_dir("goalie_stats")
    goalie_parquets = sorted(goalie_dir.glob("goalie_stats_*.parquet")) if goalie_dir.exists() else []
    if goalie_parquets:
        has_id = True
        try:
            gdf = pl.read_parquet(goalie_parquets[-1], columns=["player_id", "player_name", "team"])
        except Exception:
            has_id = False
            gdf = pl.read_parquet(goalie_parquets[-1], columns=["player_name", "team"])
        for r in gdf.drop_nulls(subset=["player_name"]).unique(subset=["player_name"]).to_dicts():
            name = r["player_name"]
            if name not in players:
                pid = int(r["player_id"]) if has_id and r.get("player_id") is not None else None
                players[name] = {"name": name, "team": r.get("team") or "", "position": "G", "player_id": pid}

    # Dedup: if one name is a strict prefix of another (truncated data artifact),
    # keep only the longer (complete) name, preserving player_id if available.
    names = list(players.keys())
    to_remove: set[str] = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            short, long_ = (a, b) if len(a) < len(b) else (b, a)
            if long_.startswith(short) and len(long_) - len(short) <= 3:
                # Merge: ensure the longer name has the best player_id
                if players[long_].get("player_id") is None and players[short].get("player_id") is not None:
                    players[long_]["player_id"] = players[short]["player_id"]
                to_remove.add(short)
    for k in to_remove:
        players.pop(k, None)

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
    # NHL API no longer populates teamGameStats. Aggregate from playerByGameStats.
    def _sum_player_stats(side_key: str) -> dict:
        pgs = (boxscore_raw.get("playerByGameStats") or {}).get(side_key, {})
        totals: dict[str, int | float] = {"sog": 0, "hits": 0, "blockedShots": 0,
                                           "giveaways": 0, "takeaways": 0}
        for group in ("forwards", "defense"):
            for p in pgs.get(group) or []:
                for k in totals:
                    totals[k] += p.get(k) or 0
        return totals

    _away_raw = _sum_player_stats("awayTeam")
    _home_raw = _sum_player_stats("homeTeam")

    # sog on awayTeam/homeTeam top-level is authoritative (includes goals)
    _away_raw["sog"] = (boxscore_raw.get("awayTeam") or {}).get("sog") or _away_raw["sog"]
    _home_raw["sog"] = (boxscore_raw.get("homeTeam") or {}).get("sog") or _home_raw["sog"]

    # Also check legacy teamGameStats (populated in older game data)
    team_stats_raw: dict[str, dict] = {}
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

    away_stats = _gc_team_stats(team_stats_raw.get("away") or _away_raw)
    home_stats = _gc_team_stats(team_stats_raw.get("home") or _home_raw)

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
            if nx >= 25:
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
        "venue":           (landing.get("venue") or {}).get("default", ""),
        "game_date":       landing.get("gameDate", ""),
        "start_time_utc":  landing.get("startTimeUTC", ""),
        "game_type":       landing.get("gameType"),
        "series_status":   landing.get("seriesStatus"),
        "away": {
            "team":   away_abbrev,
            "score":  away_raw.get("score", 0),
            "record": away_raw.get("record", ""),
        },
        "home": {
            "team":   home_abbrev,
            "score":  home_raw.get("score", 0),
            "record": home_raw.get("record", ""),
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
    venue       = (story.get("venue") or {}).get("default", "") if isinstance(story.get("venue"), dict) else str(story.get("venue") or "")

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
        "venue":        venue,
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

@app.get("/clip/resolve/{clip_id}")
async def resolve_clip(clip_id: str) -> dict:
    """Return the best direct HLS URL for a Brightcove clip.

    Uses yt-dlp to extract the manifest URL (no ads, no Brightcove iframe).
    Results are cached for 1 hour (Brightcove signed URLs are valid ~4 h).

    Returns ``{"url": "<m3u8>"}`` on success, ``{"url": null, "error": "..."}``
    on failure.
    """
    from dashboard.api.clip_resolver import resolve_brightcove_hls
    try:
        url = await resolve_brightcove_hls(clip_id)
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
            _scrape_pixelsports(away_kw, home_kw, away_abbr, home_abbr),
            _scrape_bunchatv(away_kw, home_kw, away_abbr, home_abbr),
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
    # Annotate priority + strip internal metadata before sending to client
    for s in streams:
        s["priority"] = _stream_priority(s)
        s.pop("_direct", None)

    # Cache result to avoid re-scraping on every page load
    import time as _t2
    _streams_cache[game_id] = (streams, _t2.monotonic())
    _streams_cache[-game_id] = (away_abbr, home_abbr)  # type: ignore[assignment]

    # Pre-warm the resolve cache for the first stream that needs extraction.
    # Fires in the background so the response returns immediately; by the time
    # the frontend calls /stream-resolve (~200ms later), the result is cached.
    import asyncio as _aio_pre
    _pre_candidate = next(
        (s for s in streams
         if not s.get("embed_only")
         and not (s.get("url", "").lower().endswith(".m3u8") or ".m3u8?" in s.get("url", "").lower())),
        None,
    )
    if _pre_candidate:
        _aio_pre.create_task(_resolve_embed(_pre_candidate["url"]))

    return {"streams": streams, "game_id": game_id, "away": away_abbr, "home": home_abbr}


# ===========================================================================
# /game-iptv-streams/{game_id} — broadcast-matched IPTV channels
# Returns only the IPTV channels whose network is actually broadcasting this
# game, grouped by market (home / away / national).  Replaces the 138-channel
# IPTV dump with a short, targeted list per game.
# ===========================================================================

# NHL API uses abbreviated network codes; map to the canonical prefix used in
# our IPTV channel titles (e.g. "TVA Sports HD" starts with "TVA Sports").
_BROADCAST_CODE_MAP: dict[str, str] = {
    # Canadian English — TSN
    "TSN1": "TSN1", "TSN2": "TSN2", "TSN3": "TSN3", "TSN4": "TSN4", "TSN5": "TSN5",
    # Canadian English — Sportsnet
    "SN":  "Sportsnet",
    "SNE": "Sportsnet East",
    "SNO": "Sportsnet Ontario",
    "SNW": "Sportsnet West",
    "SNP": "Sportsnet Pacific",
    # US National
    "ESPN":  "ESPN",
    "ESPN2": "ESPN2",
    "ESPNP": "ESPN+",    # ESPN+ streaming (NHL direct deal)
    "NHLN":  "NHL Network",
    "TNT":   "TNT",
    "TBS":   "TBS",
    "MAX":   "Max",
    "ABC":   "ABC",
    "FS1":   "FS1",
    "FS2":   "FS2",
    # US Regional — general
    "MSG":   "MSG",
    "MSGSN": "MSG+",
    "NESN":  "NESN",
    "NBCSP": "NBC Sports",
    "FS1":   "FS1",
    "FS2":   "FS2",
    # Victory+ (Anaheim Ducks streaming platform)
    "Victory+": "Victory+",
    "VICTORY":  "Victory+",
    # Fanduel Sports (formerly Bally Sports) — NHL regional US networks
    "FDSNNO":  "Fanduel Sports Network North",     # Minnesota Wild
    "FDSNN":   "Fanduel Sports Network North",
    "FDSNNOX": "Fanduel Sports Network North",
    "FDSNWI":  "Fanduel Sports Network Wisconsin", # Chicago Blackhawks
    "FDSNWIX": "Fanduel Sports Network Wisconsin",
    "FDSNDET": "Fanduel Sports Network Detroit",   # Detroit Red Wings
    "FDSNGL":  "Fanduel Sports Network Great Lakes",
    "FDSNFL":  "Fanduel Sports Network Florida",   # Florida Panthers / Tampa Bay
    "FDSNSUN": "Fanduel Sports Network Florida",
    "FDSNOH":  "Fanduel Sports Network Ohio Cleveland",  # Columbus Blue Jackets
    "FDSNIN":  "Fanduel Sports Indiana",           # Indiana / CHI overlap
    "FDSNST":  "Fanduel Sports Network South Tennessee", # Nashville Predators
    "FDSNSO":  "Fanduel Sports Southeast South Carolina", # Carolina Hurricanes
    "FDSNSC":  "Fanduel Sports Southeast South Carolina",
    "FDSNSCA": "Fanduel Sports Network Socal",     # Anaheim Ducks / LA Kings
    "FDSNW":   "Fanduel Sports Network West",      # Utah Hockey Club / others
    "FDSNOK":  "Fanduel Sports Network Oklahoma",
    # Canadian French — RDS / TVA Sports
    "RDS":   "RDS",
    "RDS2":  "RDS2",
    "RDSI":  "RDS Info",
    "TVAS":  "TVA Sports",
    "TVAS2": "TVA Sports 2",
    # Canadian English national (OTA + extra Sportsnet channels)
    "CBC":   "CBC",
    "SN1":   "Sportsnet One",
    "SN360": "Sportsnet 360",
}

_game_iptv_cache: dict[int, tuple[list, float]] = {}
_GAME_IPTV_TTL = 3600  # 1 hour — broadcasts don't change during a game


@app.get("/game-iptv-streams/{game_id}")
async def game_iptv_streams(game_id: int) -> dict:
    """Return IPTV channels matched to this game's actual broadcast networks."""
    import time as _t_iptv
    cached = _game_iptv_cache.get(game_id)
    if cached and (_t_iptv.monotonic() - cached[1]) < _GAME_IPTV_TTL:
        return {"broadcasts": cached[0], "cached": True}

    from data.nhl_client import NHLClient

    # 1. Fetch tvBroadcasts from NHL API
    try:
        async with NHLClient() as client:
            landing = await client.get_landing(game_id)
        if isinstance(landing, Exception) or not landing:
            return {"broadcasts": []}
    except Exception:
        return {"broadcasts": []}

    tv_broadcasts: list[dict] = landing.get("tvBroadcasts") or []
    if not tv_broadcasts:
        return {"broadcasts": []}

    # 2. Get our IPTV channel list (cached 1hr inside iptv_channels())
    try:
        iptv_result = await iptv_channels()
        all_channels: list[dict] = iptv_result.get("channels", []) if isinstance(iptv_result, dict) else []
    except Exception:
        all_channels = []

    # 3. Match each broadcast network to IPTV channels by title tokens.
    # Only match against curated tvpass/upstream channels — NOT raw M3U playlist dumps
    # which contain regional channels like "NBC Sports California" that would show up
    # on games they never broadcast.
    curated_channels = [ch for ch in all_channels if ch.get("source") in ("tvpass", "upstream")]

    # ampztl publishes several feeds per channel: a primary marked with `ƒ`,
    # plus alt/duplicate variants marked `✤`, `✪`, or `☆`. The alt feeds
    # consistently fail to stream even though they advertise as live. Drop
    # them here so the game page never offers a chip that can't play.
    _AMPZTL_DEAD_MARKER = "ƒ"
    curated_channels = [
        ch for ch in curated_channels
        if _AMPZTL_DEAD_MARKER not in (ch.get("title", "") or "")
    ]

    # Strict exact-match filter. The channel title is normalized (strip HD/FHD,
    # upstream source suffix, CA/US/UK country prefix, trailing region markers)
    # and must equal the network's display name. Prevents a single network like
    # ESPN from dragging in hundreds of Spanish/regional variants that merely
    # contain the word ("DEP | ESPN FHD", "ESPN Deportes", "ESPN 2 HD" are all
    # rejected). Uses the same _normalize_ch() that /barncentre-channels uses
    # so both surfaces stay consistent.
    # French-network titles often carry provider suffixes (e.g. "Canal RDS HD
    # (tv14s)") that defeat strict equality. Whitelist a small set of known
    # short French displays so we can apply a token-contains fallback only for
    # them — never for ESPN/Sportsnet, which would drag in hundreds of
    # regional/international variants.
    _SHORT_FR_DISPLAYS = {"rds", "rds2", "rds info", "tva sports", "tva sports 2"}

    def _matches_broadcast(ch_title: str, display: str) -> bool:
        norm = _normalize_ch(ch_title)
        want = display.lower().strip()
        if norm == want:
            return True
        # ESPN+ alternation — providers commonly list it as "espnplus"/"espn plus"
        if want == "espn+" and norm in ("espnplus", "espn plus"):
            return True
        if want in _SHORT_FR_DISPLAYS:
            needed = [t for t in want.split() if len(t) >= 2]
            if needed and all(tok in norm for tok in needed):
                return True
        return False

    result: list[dict] = []
    seen_codes: set[str] = set()
    for b in tv_broadcasts:
        code = (b.get("network") or "").strip()
        market = b.get("market", "N")  # "H" | "A" | "N"
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)

        display = _BROADCAST_CODE_MAP.get(code, code)
        matched = [
            ch for ch in curated_channels
            if _matches_broadcast(ch.get("title", ""), display)
        ]
        # Sort so working URLs land first (tvpass redirect → thetvapp → generic
        # upstream → an upstream host). Same ordering /barncentre-channels uses — keeps
        # TVA Sports from picking an upstream host's dead /play/TOKEN/m3u8 as primary.
        matched = _sort_by_url_priority(matched)
        # Always emit a row — the frontend renders unmatched rows as disabled
        # chips so Bob sees the full broadcast slate, not just what we resolved.
        result.append({
            "network":  display,
            "code":     code,
            "market":   market,
            "channels": matched,
        })

    _game_iptv_cache[game_id] = (result, _t_iptv.monotonic())
    return {"broadcasts": result}


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


async def _scrape_pixelsports(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    """
    Pixelsports.tv live events API — returns direct CDN stream URLs.
    Endpoint: GET https://pixelsport.tv/backend/liveTV/events
    Response: list of event objects with channel.server1URL / server2URL / server3URL
    """
    try:
        async with _httpx_backup.AsyncClient(timeout=8.0, follow_redirects=True) as hx:
            r = await hx.get(
                "https://pixelsport.tv/backend/liveTV/events",
                headers={
                    "User-Agent": _BROWSER_HEADERS["User-Agent"],
                    "Accept": "application/json, */*",
                    "Referer": "https://pixelsport.tv/",
                    "Origin": "https://pixelsport.tv",
                },
            )
        if r.status_code != 200:
            return []
        events = r.json()
        if not isinstance(events, list):
            # some responses are wrapped: {"data": [...]}
            events = events.get("data") or events.get("events") or []
    except Exception:
        return []

    results: list[dict] = []
    seen: set[str] = set()

    for ev in events:
        if not isinstance(ev, dict):
            continue
        # Match name against our team keywords
        match_name: str = (
            ev.get("match_name") or ev.get("title") or ev.get("name") or ""
        ).lower()
        category: str = (
            (ev.get("channel") or {}).get("TVCategory", {}).get("name", "") or
            ev.get("category") or ""
        ).lower()
        combined = match_name + " " + category
        if not _row_matches(combined, away_kw, home_kw):
            continue

        channel: dict = ev.get("channel") or {}
        for field in ("server1URL", "server2URL", "server3URL", "stream_url", "streamURL"):
            url = channel.get(field) or ev.get(field) or ""
            if not url or url in seen:
                continue
            seen.add(url)
            title_raw = ev.get("match_name") or ev.get("title") or f"{away_abbr} @ {home_abbr}"
            entry: dict = {
                "url":   url,
                "feed":  "main",
                "title": str(title_raw),
            }
            # Direct m3u8 — mark as clean source
            if url.endswith(".m3u8") or ".m3u8?" in url:
                entry["_direct"] = True
            else:
                entry["embed_only"] = True
            results.append(entry)

    return results


async def _scrape_bunchatv(
    away_kw: list[str], home_kw: list[str], away_abbr: str, home_abbr: str,
) -> list[dict]:
    """
    bunchatv1.net sports streaming aggregator.
    - Parse homepage for match containers (class: item_streaming)
    - Hit each matching match page with the homepage URL as Referer
    - Regex `"file":"https://..."` from player config to extract m3u8 URLs
    """
    BASE = "https://bunchatv1.net"
    hdrs = {
        **_BROWSER_HEADERS,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    }
    try:
        async with _httpx_backup.AsyncClient(timeout=10.0, follow_redirects=True) as hx:
            r = await hx.get(BASE, headers=hdrs)
        if r.status_code != 200:
            return []
        html = r.text
    except Exception:
        return []

    # Find match links inside item_streaming containers
    # Pattern: <div class="item_streaming ..."><a href="/match/...">...text...</a>
    match_links: list[tuple[str, str]] = []
    for m in _re_backup.finditer(
        r'class=["\'][^"\']*item_streaming[^"\']*["\'][^>]*>.*?<a\s+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html, _re_backup.S | _re_backup.I,
    ):
        href, text = m.group(1), _re_backup.sub(r"<[^>]+>", "", m.group(2)).strip()
        combined = (href + " " + text).lower()
        if _row_matches(combined, away_kw, home_kw):
            full = href if href.startswith("http") else BASE + "/" + href.lstrip("/")
            match_links.append((full, text))

    if not match_links:
        # Fallback: any anchor whose text matches our teams
        for m in _re_backup.finditer(r'<a\s+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, _re_backup.S | _re_backup.I):
            href, text = m.group(1), _re_backup.sub(r"<[^>]+>", "", m.group(2)).strip()
            combined = (href + " " + text).lower()
            if _row_matches(combined, away_kw, home_kw):
                full = href if href.startswith("http") else BASE + "/" + href.lstrip("/")
                match_links.append((full, text))

    results: list[dict] = []
    seen: set[str] = set()

    for page_url, page_title in match_links[:5]:  # cap at 5 match pages
        try:
            async with _httpx_backup.AsyncClient(timeout=8.0, follow_redirects=True) as hx:
                rp = await hx.get(page_url, headers={**hdrs, "Referer": BASE + "/"})
            if rp.status_code != 200:
                continue
            page_html = rp.text
        except Exception:
            continue

        # Extract m3u8/stream URLs from player config
        for fm in _re_backup.finditer(
            r'"file"\s*:\s*"(https?://[^"]+\.m3u8[^"]*)"', page_html, _re_backup.I
        ):
            url = fm.group(1)
            if url not in seen:
                seen.add(url)
                results.append({
                    "url":     url,
                    "feed":    "main",
                    "title":   page_title or f"{away_abbr} @ {home_abbr}",
                    "_direct": True,  # direct m3u8 → priority 0
                })

        # Also catch non-m3u8 stream src patterns
        for fm in _re_backup.finditer(
            r'"(?:file|src|source)"\s*:\s*"(https?://[^"]+)"', page_html, _re_backup.I
        ):
            url = fm.group(1)
            if url in seen or not any(ext in url for ext in (".m3u8", "/live/", "stream")):
                continue
            seen.add(url)
            results.append({
                "url":        url,
                "feed":       "main",
                "title":      page_title or f"{away_abbr} @ {home_abbr}",
                "embed_only": True,
            })

    return results


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
    from urllib.parse import urlparse, urlunparse

    # If the original URL has query params (e.g. ?token=...&expires=...&user_id=...)
    # those must be inherited by relative segment URLs that have no own params.
    # thetvapp.to tokens are IP-locked and applied per-manifest; all segments share them.
    parsed_original = urlparse(original_url)
    inherited_query = parsed_original.query  # may be ""

    def _abs_with_auth(base: str, relative: str) -> str:
        """urljoin that preserves base query params for relative URLs with no own params."""
        abs_url = urljoin(base, relative)
        if inherited_query:
            parsed = urlparse(abs_url)
            if not parsed.query:
                abs_url = urlunparse(parsed._replace(query=inherited_query))
        return abs_url

    lines = content.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # Rewrite URI= attributes inside tags (e.g. #EXT-X-MEDIA, #EXT-X-I-FRAME-STREAM-INF)
            def replace_uri(m: _re_proxy.Match) -> str:
                uri = m.group(1)
                abs_uri = _abs_with_auth(original_url, uri)
                return f'URI="{proxy_base}{quote(abs_uri, safe="")}"'
            line = _re_proxy.sub(r'URI="([^"]+)"', replace_uri, line)
            out.append(line)
        elif stripped and not stripped.startswith("#"):
            # Segment or sub-manifest URL
            abs_url = _abs_with_auth(original_url, stripped)
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
_tvpass_resolve_cache: dict[str, tuple[str, float]] = {}  # tvpass url -> (resolved url, ts)
_TVPASS_RESOLVE_TTL = 300  # 5 min — thetvapp.to tokens last ~1-2h but refresh is cheap


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
    # Fast path: already a direct m3u8. Also catches our own relay endpoints
    # (/hls, /m3u8) — they produce HLS manifests that hls.js can consume
    # directly, so skip the yt-dlp / Playwright extraction layers.
    lower = url.split("?")[0].lower()
    if (
        lower.endswith(".m3u8")
        or lower.endswith("/m3u8")
        or lower.endswith("/hls")
        or lower.endswith(".mpd")
    ):
        return {"type": "m3u8", "url": url}

    # Fast path: tvpass.org/live/* URLs redirect to thetvapp.to.
    # Follow the 302 — if it ends at an m3u8 we're done; otherwise embed the page.
    # Never fall through to yt-dlp/Playwright for these URLs.
    if "tvpass.org/live/" in lower:
        # Check cache first (5-min TTL to avoid re-fetching token on every click)
        _tv_cached = _tvpass_resolve_cache.get(url)
        if _tv_cached and (_time.monotonic() - _tv_cached[1]) < _TVPASS_RESOLVE_TTL:
            _cached_url = _tv_cached[0]
            _ctype = "m3u8" if ".m3u8" in _cached_url else "embed"
            return {"type": _ctype, "url": _cached_url, "cached": True}
        try:
            import httpx as _hx
            async with _hx.AsyncClient(
                follow_redirects=True,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://tvpass.org/",
                    "Origin": "https://tvpass.org",
                },
            ) as _cl:
                r = await _cl.get(url)
                final = str(r.url)
                if ".m3u8" in final:
                    _tvpass_resolve_cache[url] = (final, _time.monotonic())
                    return {"type": "m3u8", "url": final}
        except Exception:
            pass
        # Redirect didn't give us HLS — embed the tvpass page (has built-in player)
        _tvpass_resolve_cache[url] = (url, _time.monotonic())
        return {"type": "embed", "url": url}

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
        # Use x-forwarded-proto if present; fall back to actual request scheme.
        # Defaulting to "https" breaks local dev where Next.js forwards x-forwarded-host
        # but no x-forwarded-proto, causing segment URLs to use https://localhost:3000.
        forwarded_proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        proxy_base = f"{forwarded_proto}://{forwarded_host}/api/stream-proxy?url="
    else:
        base = str(request.base_url).rstrip("/")
        proxy_base = f"{base}/stream-proxy?url="

    # Choose referer based on the upstream CDN to avoid hotlink blocks.
    # thetvapp.to tokens are IP-locked; tvpass.org is the expected referrer
    # for that CDN. Brightcove's boltdns CDN validates against its own player
    # origin — anything else risks a 403 on the manifest XHR.
    if "thetvapp.to" in url:
        _referer = "https://tvpass.org/"
    elif "boltdns.net" in url or "brightcove" in url:
        _referer = "https://players.brightcove.net/"
    else:
        _referer = "https://onhockey.tv/"
    _stream_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": _referer,
        "Origin": _referer.rstrip("/"),
        # Some upstream upstreams route through ngrok (IPTV_LOCAL_PROXY_URL). ngrok's
        # interstitial swallows browser-UA requests with a 200-OK HTML warning
        # page, which _rewrite_m3u8 then treats as a manifest and feeds to hls.js.
        "ngrok-skip-browser-warning": "skip",
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

        # If it's an m3u8 manifest, rewrite internal URLs.
        # Only treat as M3U8 on success — a 403/404 with a .m3u8 URL returns HTML
        # which must not be passed to _rewrite_m3u8 as garbage segment lines.
        is_m3u8 = resp.status_code == 200 and (
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
    Never 500s — if the CV module isn't loaded or no worker exists, returns stopped=false.
    """
    try:
        if not hasattr(app.state, "cv"):
            # No CV manager ever started — nothing to stop
            return {"stopped": False, "game_id": game_id}
        mgr = _cv_manager()
        stopped = mgr.stop(game_id)
        return {"stopped": stopped, "game_id": game_id}
    except Exception:
        return {"stopped": False, "game_id": game_id}


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
# One-shot clip CV — post-game highlight processing
# ══════════════════════════════════════════════════════════════════════════════
#
# Endpoints:
#   POST /api/cv/clip/{clip_id}         — kick off background processing
#   GET  /api/cv/clip/{clip_id}         — job status (processing | ready | error)
#   GET  /api/cv/clip/{clip_id}/frames  — paginated per-frame detections
#
# Jobs run in a FastAPI BackgroundTask thread and persist results as JSON under
# data/cv_clip_cache/<clip_id>.json.  Subsequent POSTs for a ready clip return
# immediately without re-processing.  Status is tracked in a module-level dict
# guarded by a lock.

import threading as _threading_clip
import logging as _logging_clip

_clip_log = _logging_clip.getLogger("cv_clip_api")
_clip_jobs: "dict[str, object]" = {}
_clip_jobs_lock = _threading_clip.Lock()


def _clip_job_get(clip_id: str):
    with _clip_jobs_lock:
        return _clip_jobs.get(clip_id)


def _clip_job_set(clip_id: str, status_obj) -> None:
    with _clip_jobs_lock:
        _clip_jobs[clip_id] = status_obj


async def _run_clip_job(clip_id: str, hls_url: str, game_id: int | None):
    """Background task — resolve HLS (if needed) and run the clip pipeline."""
    from dashboard.api.cv_clip import process_clip
    status = _clip_job_get(clip_id)
    if status is None:
        return
    try:
        # process_clip is synchronous + CPU-heavy — run in a thread so we don't
        # block the event loop.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: process_clip(clip_id=clip_id, hls_url=hls_url, status=status, game_id=game_id),
        )
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        # Trim the traceback to the last ~4 frames so it fits in the JSON
        # status payload without flooding the response.
        tb_short = "\n".join(tb.strip().splitlines()[-8:])
        status.update(
            status="error",
            message=f"{type(exc).__name__}: {exc}\n---\n{tb_short}",
            finished_at=_time.time(),
        )
        _clip_log.error("[cv-clip:%s] job failed: %s\n%s", clip_id, exc, tb)


@app.post("/cv/clip/{clip_id}")
async def cv_clip_start(clip_id: str, game_id: int | None = None):
    """Kick off CV processing for a Brightcove highlight clip.

    Resolves the clip's HLS URL via the shared resolver, then launches a
    background task that runs detect → track → homography end-to-end and writes
    ``data/cv_clip_cache/<clip_id>.json``.

    Response values for ``status``:
      ``ready``       — cache already exists, no work to do
      ``processing``  — job is running (just started, or already in flight)
      ``error``       — resolution failed; see ``message``
    """
    from dashboard.api.cv_clip import ClipJobStatus, load_cache
    from dashboard.api.clip_resolver import resolve_brightcove_hls

    # Fast path — cache already present
    cached = load_cache(clip_id)
    if cached is not None:
        return {
            "clip_id":      clip_id,
            "status":       "ready",
            "total_frames": cached.get("total_frames", 0),
            "duration_s":   cached.get("duration_s", 0.0),
        }

    # Already running?
    existing = _clip_job_get(clip_id)
    if existing is not None and existing.status == "processing":
        return {"clip_id": clip_id, **existing.to_dict()}

    hls_url = await resolve_brightcove_hls(clip_id)
    if not hls_url:
        return {
            "clip_id": clip_id,
            "status":  "error",
            "message": "failed to resolve HLS URL for clip",
        }

    status = ClipJobStatus()
    _clip_job_set(clip_id, status)

    # Launch in background — do not await
    asyncio.create_task(_run_clip_job(clip_id, hls_url, game_id))

    return {"clip_id": clip_id, **status.to_dict()}


@app.get("/cv/clip/{clip_id}")
async def cv_clip_status(clip_id: str):
    """Return the current processing status for *clip_id*.

    ``ready`` indicates the cache file exists and ``GET .../frames`` will work.
    """
    from dashboard.api.cv_clip import load_cache

    cached = load_cache(clip_id)
    if cached is not None:
        return {
            "clip_id":      clip_id,
            "status":       "ready",
            "frames_done":  cached.get("total_frames", 0),
            "total_frames": cached.get("total_frames", 0),
            "duration_s":   cached.get("duration_s", 0.0),
        }

    existing = _clip_job_get(clip_id)
    if existing is None:
        return {"clip_id": clip_id, "status": "unknown"}
    return {"clip_id": clip_id, **existing.to_dict()}


@app.get("/cv/clip/{clip_id}/frames")
async def cv_clip_frames(
    clip_id: str,
    from_seq: int = 0,
    limit:    int = 500,
):
    """Return a paginated slice of per-frame detections.

    Serves from the on-disk cache once processing is done, or from the
    live in-memory buffer while processing is still running — so the UI
    can stream overlays progressively. ``next_from_seq`` is the seq for
    the next page, or ``null`` when done (job is ready *and* the caller
    has drained every frame).
    """
    from fastapi import HTTPException
    from dashboard.api.cv_clip import load_cache, _FPS as CV_FPS

    limit = max(1, min(int(limit), 2000))

    # Completed clips — serve from disk
    cached = load_cache(clip_id)
    if cached is not None:
        all_frames = cached.get("frames", [])
        slc = [f for f in all_frames if int(f.get("frame_seq", -1)) >= int(from_seq)][:limit]
        last_seq = slc[-1]["frame_seq"] if slc else None
        next_from = (int(last_seq) + 1) if (last_seq is not None and
                                             int(last_seq) + 1 < cached.get("total_frames", 0)) else None
        return {
            "clip_id":       clip_id,
            "fps":           cached.get("fps"),
            "total_frames":  cached.get("total_frames", 0),
            "duration_s":    cached.get("duration_s", 0.0),
            "frames":        slc,
            "next_from_seq": next_from,
        }

    # Mid-flight — serve from the running job's in-memory buffer
    existing = _clip_job_get(clip_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="clip not started")
    if existing.status == "error":
        raise HTTPException(status_code=500, detail=existing.message or "processing failed")

    slc = existing.get_frames_slice(int(from_seq), limit)
    d = existing.to_dict()
    is_ready = d.get("status") == "ready"

    # Advance cursor only if we received frames; otherwise let the client
    # retry the same from_seq on the next poll. next_from_seq == null means
    # "we're caught up AND the job is done" — safe to stop polling.
    if slc:
        nxt: int | None = int(slc[-1]["frame_seq"]) + 1
    elif not is_ready:
        nxt = int(from_seq)
    else:
        nxt = None

    return {
        "clip_id":       clip_id,
        "fps":           CV_FPS,
        "total_frames":  d.get("total_frames") or 0,
        "duration_s":    0.0,
        "frames":        slc,
        "next_from_seq": nxt,
    }


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


# ── Feature 16.6 bridge — in-browser CV observations ingestion ───────────
#
# Desktop clients run the YOLO player detector in-browser (useCvLiveReplay)
# and batch ~5s windows of tracked positions + derived events. They POST the
# batch here. We append to NDJSON per-game-per-day for offline aggregation
# into per-player action counts that feed player_behavior_net (Feature 2.22).
#
# This is an append-only ingestion log. No reads from here in real time —
# a nightly aggregator reduces raw observations into training features.


class CvLiveObservationsRequest(_BaseModel):
    """Batch of CV observation bundles from one in-browser client session."""
    game_id:   int
    client_id: str
    bundles:   list[dict]   # flexible shape; frontend CvObservationBundle
    # Arena name from NHL API (venue.default). Optional so older clients keep
    # working; newer ones include it so downstream aggregators can bucket
    # training data by arena without joining to game metadata.
    arena:     str | None = None


@app.post("/api/cv/live/observations")
async def cv_live_observations(body: CvLiveObservationsRequest):
    """Append a batch of in-browser CV observations to NDJSON for offline aggregation.

    Each bundle captures ~1 s of tracked state: scene kind, player count,
    per-track positions + velocities, and detected events (pass/shot, scene
    change). We persist raw — aggregation happens nightly in a separate
    pipeline so the ingestion endpoint stays fast and non-blocking.
    """
    from fastapi import HTTPException
    n = len(body.bundles)
    if n == 0:
        return {"accepted": 0, "game_id": body.game_id}
    if n > 120:
        raise HTTPException(status_code=413, detail="too many bundles in batch")
    if not body.client_id or len(body.client_id) > 128:
        raise HTTPException(status_code=400, detail="invalid client_id")

    import json
    obs_dir = _GRETZKY_DATA_DIR / "cv_live_observations" / str(body.game_id)
    obs_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = obs_dir / f"{today}.ndjson"

    t_wall = datetime.now(timezone.utc).isoformat()
    arena = (body.arena or "").strip()[:128] or None
    lines = [
        json.dumps({
            "t_wall":    t_wall,
            "game_id":   body.game_id,
            "client_id": body.client_id,
            "arena":     arena,
            **b,
        }, separators=(",", ":"))
        for b in body.bundles
    ]

    def _append():
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    await asyncio.to_thread(_append)

    return {
        "accepted": n,
        "game_id":  body.game_id,
    }


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


@app.get("/player-shots/{player_id}")
async def player_shots(player_id: int):
    """Return arena-adjusted shot coordinates for a player (last 2 seasons of MoneyPuck data)."""
    shots_dir = _GRETZKY_DATA_DIR / "shots"
    if not shots_dir.exists():
        return {"shots": [], "count": 0, "status": "no_data"}
    files = sorted(shots_dir.glob("*.parquet"))
    if not files:
        return {"shots": [], "count": 0, "status": "no_data"}
    import polars as pl
    want_cols = {"shooter_id", "arena_adj_x", "arena_adj_y", "x_goal", "is_goal", "shot_type"}
    dfs = []
    for f in files[-2:]:  # last 2 seasons
        try:
            schema = pl.read_parquet_schema(f)
            cols = [c for c in want_cols if c in schema]
            if "shooter_id" not in cols:
                continue
            dfs.append(pl.read_parquet(f, columns=cols))
        except Exception:
            pass
    if not dfs:
        return {"shots": [], "count": 0, "status": "no_data"}
    combined = pl.concat(dfs, how="diagonal_relaxed")
    player_df = combined.filter(pl.col("shooter_id") == player_id)
    shots_out = []
    for r in player_df.to_dicts():
        x = r.get("arena_adj_x")
        y = r.get("arena_adj_y")
        if x is None or y is None:
            continue
        shots_out.append({
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "xg": round(float(r.get("x_goal") or 0), 3),
            "goal": bool(r.get("is_goal", False)),
            "type": str(r.get("shot_type") or ""),
        })
    return {"shots": shots_out, "count": len(shots_out), "status": "ok"}


@app.get("/goalie-shots/{player_id}")
async def goalie_shots(player_id: int):
    """Return arena-adjusted shot coordinates for shots AGAINST a goalie (last 2 seasons)."""
    shots_dir = _GRETZKY_DATA_DIR / "shots"
    if not shots_dir.exists():
        return {"shots": [], "count": 0, "status": "no_data"}
    files = sorted(shots_dir.glob("*.parquet"))
    if not files:
        return {"shots": [], "count": 0, "status": "no_data"}
    import polars as pl
    want_cols = {"goalie_id", "arena_adj_x", "arena_adj_y", "x_goal", "is_goal", "shot_type"}
    dfs = []
    for f in files[-2:]:
        try:
            schema = pl.read_parquet_schema(f)
            cols = [c for c in want_cols if c in schema]
            if "goalie_id" not in cols:
                continue
            dfs.append(pl.read_parquet(f, columns=cols))
        except Exception:
            pass
    if not dfs:
        return {"shots": [], "count": 0, "status": "no_data"}
    combined = pl.concat(dfs, how="diagonal_relaxed")
    player_df = combined.filter(pl.col("goalie_id") == player_id)
    shots_out = []
    for r in player_df.to_dicts():
        x = r.get("arena_adj_x")
        y = r.get("arena_adj_y")
        if x is None or y is None:
            continue
        shots_out.append({
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "xg": round(float(r.get("x_goal") or 0), 3),
            "goal": bool(r.get("is_goal", False)),
            "type": str(r.get("shot_type") or ""),
        })
    return {"shots": shots_out, "count": len(shots_out), "status": "ok"}



@app.get("/goalie-neural-net/{player_id}")
async def goalie_neural_net(player_id: int):
    """Return shot-type + zone save% breakdown for a goalie — last 2 seasons."""
    shots_dir = _GRETZKY_DATA_DIR / "shots"
    if not shots_dir.exists():
        return {"status": "no_data", "shot_types": [], "zones": []}
    files = sorted(shots_dir.glob("*.parquet"))
    if not files:
        return {"status": "no_data", "shot_types": [], "zones": []}
    import polars as pl

    want_cols = {"goalie_id", "arena_adj_x", "arena_adj_y", "is_goal", "shot_type"}
    dfs = []
    for f in files[-2:]:
        try:
            schema = pl.read_parquet_schema(f)
            cols = [c for c in want_cols if c in schema]
            if "goalie_id" not in cols:
                continue
            dfs.append(pl.read_parquet(f, columns=cols))
        except Exception:
            pass
    if not dfs:
        return {"status": "no_data", "shot_types": [], "zones": []}

    combined = pl.concat(dfs, how="diagonal_relaxed")
    df = combined.filter(pl.col("goalie_id") == player_id)
    if len(df) == 0:
        return {"status": "no_shots", "shot_types": [], "zones": []}

    total = len(df)
    goals_allowed = int(df["is_goal"].cast(pl.Int64).sum()) if "is_goal" in df.columns else 0

    type_labels = {
        "WRIST": "Wrist", "SLAP": "Slap", "SNAP": "Snap",
        "TIP": "Tip-In", "DEFL": "Deflection", "BACK": "Backhand", "WRAP": "Wraparound",
    }
    shot_types_out = []
    if "shot_type" in df.columns:
        by_type = (
            df.filter(pl.col("shot_type").is_not_null())
            .group_by("shot_type")
            .agg([
                pl.len().alias("shots"),
                pl.col("is_goal").cast(pl.Int64).sum().alias("goals"),
            ])
            .with_columns(
                (1.0 - pl.col("goals").cast(pl.Float64) / pl.col("shots").cast(pl.Float64)).alias("sv_pct")
            )
            .filter(pl.col("shots") >= 10)
            .sort("shots", descending=True)
        )
        for row in by_type.to_dicts():
            shot_types_out.append({
                "type": type_labels.get(row["shot_type"], row["shot_type"]),
                "shots": int(row["shots"]),
                "goals": int(row["goals"]),
                "sv_pct": round(float(row["sv_pct"]), 4),
            })

    zones_out = []
    if "arena_adj_x" in df.columns and "arena_adj_y" in df.columns:
        df2 = (
            df
            .filter(pl.col("arena_adj_x").is_not_null() & pl.col("arena_adj_y").is_not_null())
            .with_columns([
                ((89.0 - pl.col("arena_adj_x").cast(pl.Float64)) ** 2
                 + pl.col("arena_adj_y").cast(pl.Float64) ** 2).sqrt().alias("_dist"),
                pl.when(pl.col("arena_adj_y").cast(pl.Float64) > 14).then(pl.lit("right"))
                  .when(pl.col("arena_adj_y").cast(pl.Float64) < -14).then(pl.lit("left"))
                  .otherwise(pl.lit("center")).alias("_side"),
            ])
            .with_columns(
                pl.when(pl.col("_dist") < 25).then(pl.lit("close"))
                  .when(pl.col("_dist") < 45).then(pl.lit("mid"))
                  .otherwise(pl.lit("far")).alias("_dist_zone"),
            )
        )
        by_zone = (
            df2.group_by(["_side", "_dist_zone"])
            .agg([
                pl.len().alias("shots"),
                pl.col("is_goal").cast(pl.Int64).sum().alias("goals"),
            ])
            .with_columns(
                (1.0 - pl.col("goals").cast(pl.Float64) / pl.col("shots").cast(pl.Float64)).alias("sv_pct")
            )
            .filter(pl.col("shots") >= 5)
        )
        for row in by_zone.to_dicts():
            zones_out.append({
                "side": str(row["_side"]),
                "dist": str(row["_dist_zone"]),
                "shots": int(row["shots"]),
                "goals": int(row["goals"]),
                "sv_pct": round(float(row["sv_pct"]), 4),
            })

    return {
        "status": "ok",
        "total_shots": total,
        "goals_allowed": goals_allowed,
        "overall_sv_pct": round(1.0 - goals_allowed / total, 4) if total > 0 else None,
        "shot_types": shot_types_out,
        "zones": zones_out,
    }


# ---------------------------------------------------------------------------
# IPTV Test Channels — ad-free NHL broadcast streams for testing
# ---------------------------------------------------------------------------
import re as _re_iptv
import time as _time_iptv

_IPTV_CACHE: dict = {"data": [], "ts": 0.0}
_IPTV_TTL = 3600  # 1 hour

# Static tvpass.org slugs — all confirmed HTTP 302 active
_TVPASS_CHANNELS = [
    ("NHL Network HD",       "NHLNetwork",                 "hd"),
    ("NHL Network SD",       "NHLNetwork",                 "sd"),
    ("ESPN HD",              "ESPN",                       "hd"),
    ("ESPN SD",              "ESPN",                       "sd"),
    ("ESPN2 HD",             "ESPN2",                      "hd"),
    ("ESPN2 SD",             "ESPN2",                      "sd"),
    ("ESPNU HD",             "ESPNU",                      "hd"),
    ("TNT East HD",          "TNTEast",                    "hd"),
    ("TNT East SD",          "TNTEast",                    "sd"),
    ("TBS East HD",          "TBSEast",                    "hd"),
    ("TBS East SD",          "TBSEast",                    "sd"),
    ("FS1 HD",               "FoxSports1",                 "hd"),
    ("FS1 SD",               "FoxSports1",                 "sd"),
    ("FS2 HD",               "FoxSports2",                 "hd"),
    ("MSG HD",               "msg-madison-square-gardens", "hd"),
    ("MSG+ HD",              "msg-plus",                   "hd"),
    ("Sportsnet East HD",    "sportsnet-east",             "hd"),
    ("Sportsnet East SD",    "sportsnet-east",             "sd"),
    ("Sportsnet Ontario HD", "sportsnet-ontario",          "hd"),
    ("Sportsnet West HD",    "sportsnet-west",             "hd"),
    ("Sportsnet Pacific HD", "sportsnet-pacific",          "hd"),
    ("Sportsnet 360 HD",     "sportsnet-360",              "hd"),
    ("Sportsnet One HD",     "sportsnet-one",              "hd"),
    ("TSN1 HD",  "tsn1",  "hd"), ("TSN1 SD",  "tsn1",  "sd"),
    ("TSN2 HD",  "tsn2",  "hd"), ("TSN2 SD",  "tsn2",  "sd"),
    ("TSN3 HD",  "tsn3",  "hd"),
    ("TSN4 HD",  "tsn4",  "hd"),
    ("TSN5 HD",  "tsn5",  "hd"),
    # FanDuel Sports Networks (formerly Bally Sports) — NHL regional US
    ("Fanduel Sports Network Detroit HD",    "fanduel-sports-network-detroit",    "hd"),
    ("Fanduel Sports Network Florida HD",    "fanduel-sports-network-florida",    "hd"),
    ("Fanduel Sports Network North HD",      "fanduel-sports-network-north",      "hd"),
    ("Fanduel Sports Network Wisconsin HD",  "fanduel-sports-network-wisconsin",  "hd"),
    ("Fanduel Sports Network Ohio HD",       "fanduel-sports-network-ohio",       "hd"),
    ("Fanduel Sports Network South HD",      "fanduel-sports-network-south",      "hd"),
    ("Fanduel Sports Network Southeast HD",  "fanduel-sports-network-southeast",  "hd"),
    ("Fanduel Sports Network West HD",       "fanduel-sports-network-west",       "hd"),
    ("Fanduel Sports Network Socal HD",      "fanduel-sports-network-socal",      "hd"),
]

_NHL_KEYWORDS = [
    "espn","tnt","tbs","nhl","sportsnet","tsn","msg","foxsports","fs1","fs2","abc","nbc","tva","rds",
    # US regional sports (Fanduel = rebranded Bally Sports; Victory+ = Ducks)
    "fanduel","bally","victory",
    # Other regional US nets that carry NHL
    "nesn","nbcs","nbcsp","spectrum sportsnet","nhln",
    # CBS Sports family — non-NHL carrier but surfaced in BarnCentre guide
    "cbs",
]

# ---------------------------------------------------------------------------
# upstream Codes IPTV accounts — fetched at startup, cached 1hr alongside tvpass
# Each entry: (label, host, port, username, password)
# ---------------------------------------------------------------------------
_upstream_ACCOUNTS: list[tuple[str, str, int, str, str]] = [
    ("an upstream host",  "an upstream host.ddns.net", 8081, "PNbV7ywsHG",            "u7jmr3xvcM"),
    ("ampztl-a",    "ampztl.xyz",          8080, "arturo",                "YZcm6gw6Ukwt"),
    ("ampztl-b",    "ampztl.xyz",          8080, "webtv1847",             "YsAPRy6Jq8TJ"),
    # Disabled: tv14s CDN tarpits our IP, lunar returns 403. Re-enable only
    # if providers start accepting requests again.
    # ("tv14s",       "tv14s.xyz",           8080, "Serentiy2@ogbtv.com",   "0306@1954"),
    # ("lunar",       "lunar.pm",            8080, "JeffOglesby",           "Marriage101"),
]

# Residential relay — scripts/iptv_relay.py run on a laptop/Pi and exposed via
# ngrok. When IPTV_LOCAL_PROXY_URL is set, every stream URL we hand to the
# browser is rewritten to go through that tunnel (upstream providers block VPS
# IPs; the relay punches through from a residential address). Empty ⇒
# direct-to-provider fallback, same as before this release.
_upstream_HOSTS: frozenset[str] = frozenset(h for _, h, *_ in _upstream_ACCOUNTS)


def _rewrite_iptv_url(url: str) -> str:
    """Route an upstream stream URL through the local residential relay, if configured.

    Playlist URLs (.m3u8) → /m3u8: fetch + rewrite segment URIs back through tunnel.
    Non-playlist URLs     → /hls:  ffmpeg-transmux raw MPEG-TS into a rolling HLS
                                   playlist (needed for ampztl and other TS-only
                                   accounts that hls.js cannot consume directly).
    """
    tunnel = (os.environ.get("IPTV_LOCAL_PROXY_URL") or "").rstrip("/")
    if not tunnel or not url.startswith("http"):
        return url
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return url
    if host.lower() not in _upstream_HOSTS:
        return url
    # an upstream host hard-codes `/play/<token>/m3u8` in its M3U, but that path
    # returns 404 — only the `/ts` variant serves raw MPEG-TS. Rewrite to /ts.
    inner = url
    if host.lower() == "an upstream host.ddns.net":
        pre, sep, _ = inner.partition("?")
        if pre.endswith("/m3u8"):
            inner = pre[:-len("/m3u8")] + "/ts" + (sep + _ if sep else "")
    endpoint = "m3u8" if ".m3u8" in inner.lower().split("?", 1)[0] else "hls"
    token    = os.environ.get("IPTV_RELAY_TOKEN") or ""
    tq       = f"&t={quote(token, safe='')}" if token else ""
    return f"{tunnel}/{endpoint}?u={quote(inner, safe='')}{tq}"


async def _fetch_upstream_channels(
    label: str, host: str, port: int, username: str, password: str,
) -> list[dict]:
    """
    Fetch the M3U playlist from an upstream Codes server and return NHL-relevant
    channels as standardized channel dicts.

    Prefers HLS (output=m3u8) so the /m3u8 relay can passthrough-rewrite the
    playlist cheaply. Accounts that only advertise `ts` fall back to the raw
    MPEG-TS path — the URL rewriter routes those through the relay's /hls
    endpoint (ffmpeg transmux) so hls.js can still consume them.
    """
    base = f"http://{host}:{port}"
    ua   = _BROWSER_HEADERS["User-Agent"]

    output_fmt = "m3u8"
    try:
        async with _httpx_backup.AsyncClient(timeout=10.0, follow_redirects=True) as hx:
            probe = await hx.get(
                f"{base}/player_api.php?username={username}&password={password}",
                headers={"User-Agent": ua},
            )
        if probe.status_code == 200:
            fmts = [str(f).lower() for f in
                    (probe.json().get("user_info") or {}).get("allowed_output_formats") or []]
            if "m3u8" in fmts:
                output_fmt = "m3u8"
            elif "ts" in fmts:
                output_fmt = "ts"
            else:
                return []
    except Exception:
        pass  # probe failure is not fatal — fall through to M3U fetch with m3u8 default

    url = (
        f"{base}/get.php?username={username}&password={password}"
        f"&type=m3u_plus&output={output_fmt}"
    )
    try:
        async with _httpx_backup.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
            r = await hx.get(url, headers={"User-Agent": ua})
        if r.status_code != 200:
            return []
        text = r.text
    except Exception:
        return []

    results: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        if line.startswith("#EXTINF"):
            # Channel name: last comma-separated segment
            name_m = _re_iptv.search(r'tvg-name="([^"]+)"', line)
            ch_name = name_m.group(1) if name_m else line.split(",")[-1].strip()
            stream_url = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if (
                stream_url.startswith("http")
                and any(k in ch_name.lower() for k in _NHL_KEYWORDS)
            ):
                results.append({
                    "title":      f"{ch_name} ({label})",
                    "url":        _rewrite_iptv_url(stream_url),
                    "source":     "upstream",
                    "feed":       "iptv",
                    "priority":   1,
                    "embed_only": False,
                })
            i += 2
        else:
            i += 1
    return results

def _build_m3u_sources() -> list[str]:
    return [
        "https://raw.githubusercontent.com/phosani/tvpass/refs/heads/main/tvpasshd.m3u",
        "https://raw.githubusercontent.com/musicmashupstv-hue/tvpassplaylist/main/tvpassplaylist.m3u8",
        "https://raw.githubusercontent.com/jburg229/iptv-playlist/main/playlist.m3u",
    ]

_M3U_SOURCES = _build_m3u_sources()


@app.get("/iptv-channels")
async def iptv_channels(force: bool = False):
    now = _time_iptv.time()
    if not force and now - _IPTV_CACHE["ts"] < _IPTV_TTL and _IPTV_CACHE["data"]:
        return {"channels": _IPTV_CACHE["data"], "count": len(_IPTV_CACHE["data"]), "cached": True}

    channels = []

    # Source 1: static tvpass.org slugs (always available)
    for name, slug, quality in _TVPASS_CHANNELS:
        channels.append({
            "title": name,
            "url": f"https://tvpass.org/live/{slug}/{quality}",
            "source": "tvpass",
            "feed": "iptv",
            "priority": 1,
            "embed_only": False,
        })

    # Source 2: albinchristo04/tvpass streams.json — tvpass.org redirect URLs with fresh tokens.
    # The JSON has structure {"channels": [...]} where each entry has:
    #   original_url: tvpass.org/live/<slug>/hd  ← use this (generates fresh token at stream time)
    #   stream_url:   thetvapp.to/hls/...?token=... ← stale, expires within hours — do NOT use
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://raw.githubusercontent.com/albinchristo04/tvpass/main/streams.json")
            if r.status_code == 200:
                payload = r.json()
                entries = payload.get("channels", []) if isinstance(payload, dict) else payload
                for entry in entries:
                    name_lower = (entry.get("name") or "").lower()
                    # Prefer original_url (tvpass redirect → fresh token); fall back to stream_url
                    original_url = entry.get("original_url") or ""
                    stream_url   = entry.get("stream_url") or ""
                    use_url = original_url if original_url.startswith("http") else stream_url
                    if (
                        entry.get("status") == "working"
                        and use_url.startswith("http")
                        and any(k in name_lower for k in _NHL_KEYWORDS)
                    ):
                        channels.append({
                            "title": f"{entry['name']} (thetvapp)",
                            "url": use_url,
                            "source": "thetvapp",
                            "feed": "iptv",
                            "priority": 0,
                            "embed_only": False,
                        })
    except Exception:
        pass

    # Source 3: public M3U playlists — parse and filter for NHL channels
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for m3u_url in _M3U_SOURCES:
                try:
                    r = await client.get(m3u_url)
                    if r.status_code != 200:
                        continue
                    lines = r.text.splitlines()
                    i = 0
                    while i < len(lines) - 1:
                        line = lines[i]
                        if line.startswith("#EXTINF"):
                            name_match = _re_iptv.search(r'tvg-name="([^"]+)"', line)
                            display = name_match.group(1) if name_match else line.split(",")[-1].strip()
                            url_line = lines[i + 1].strip()
                            if any(k in display.lower() for k in _NHL_KEYWORDS) and url_line.startswith("http"):
                                channels.append({
                                    "title": f"{display} (playlist)",
                                    "url": url_line,
                                    "source": "m3u_playlist",
                                    "feed": "iptv",
                                    "priority": 1,
                                    "embed_only": False,
                                })
                            i += 2
                        else:
                            i += 1
                except Exception:
                    continue
    except Exception:
        pass

    # Source 4: upstream Codes accounts — fire all in parallel, merge results
    try:
        import asyncio as _aio_xt
        xt_results = await _aio_xt.gather(
            *[_fetch_upstream_channels(lbl, h, p, u, pw) for lbl, h, p, u, pw in _upstream_ACCOUNTS],
            return_exceptions=True,
        )
        for res in xt_results:
            if isinstance(res, list):
                channels.extend(res)
    except Exception:
        pass

    # Deduplicate by exact URL — keeps first occurrence (highest-priority source wins)
    _seen_urls: set[str] = set()
    _deduped: list = []
    for _ch in channels:
        if _ch["url"] not in _seen_urls:
            _seen_urls.add(_ch["url"])
            _deduped.append(_ch)
    channels = _deduped

    _IPTV_CACHE["data"] = channels
    _IPTV_CACHE["ts"] = now
    return {"channels": channels, "count": len(channels), "cached": False}


@app.get("/iptv-channel-status")
async def iptv_channel_status():
    """Return the last check-iptv status report (working/broken per channel with URLs).

    Written by `uv run python scripts/gretzky.py check-iptv`.
    Returns empty result if check-iptv has never been run.
    """
    import json as _json_status
    _status_path = Path(__file__).parent / "iptv_status.json"
    if not _status_path.exists():
        return {"checked_at": None, "summary": {}, "channels": []}
    try:
        return _json_status.loads(_status_path.read_text())
    except Exception:
        return {"checked_at": None, "summary": {}, "channels": []}


@app.get("/iptv-debug")
async def iptv_debug():
    """Dump every channel name each upstream account emits, split by NHL-keyword match.

    Rob-only (Next.js middleware gates /api/iptv-debug). No caching — always
    re-fetches so the matcher can be iterated live. Use to tune the
    game-iptv-streams token matcher when a known broadcast isn't lighting up.
    """
    import asyncio as _aio_dbg

    async def _raw(label: str, host: str, port: int, user: str, pw: str) -> dict:
        import httpx as _hx
        base = f"http://{host}:{port}"
        url = f"{base}/get.php?username={user}&password={pw}&type=m3u_plus&output=m3u8"
        try:
            async with _hx.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
                r = await hx.get(url, headers={"User-Agent": _BROWSER_HEADERS["User-Agent"]})
            if r.status_code != 200:
                return {"label": label, "error": f"HTTP {r.status_code}", "titles": [], "nhl_titles": []}
            text = r.text
        except Exception as e:
            return {"label": label, "error": str(e), "titles": [], "nhl_titles": []}

        titles: list[str] = []
        lines = text.splitlines()
        for line in lines:
            if line.startswith("#EXTINF"):
                m = _re_iptv.search(r'tvg-name="([^"]+)"', line)
                name = m.group(1) if m else line.split(",")[-1].strip()
                if name:
                    titles.append(name)
        nhl = [t for t in titles if any(k in t.lower() for k in _NHL_KEYWORDS)]
        return {"label": label, "host": host, "count": len(titles), "nhl_count": len(nhl),
                "nhl_titles": sorted(set(nhl)), "titles": sorted(set(titles))}

    results = await _aio_dbg.gather(
        *[_raw(lbl, h, p, u, pw) for lbl, h, p, u, pw in _upstream_ACCOUNTS],
        return_exceptions=True,
    )
    accounts = [r if isinstance(r, dict) else {"error": str(r)} for r in results]

    # Expose the relay config so we can verify IPTV_LOCAL_PROXY_URL matches the
    # laptop's current ngrok tunnel without leaking the raw token value.
    tunnel = (os.environ.get("IPTV_LOCAL_PROXY_URL") or "").rstrip("/")
    sample_in  = "http://an upstream host.ddns.net:80/live/foo/bar/123.m3u8"
    sample_out = _rewrite_iptv_url(sample_in)
    return {
        "accounts": accounts,
        "tunnel": {
            "url":        tunnel or None,
            "token_set":  bool(os.environ.get("IPTV_RELAY_TOKEN")),
            "rewrite_in":  sample_in,
            "rewrite_out": sample_out,
            "is_rewritten": sample_out != sample_in,
        },
    }


# ===========================================================================
# /barncentre-channels — curated sports highlight channels for BarnCentre TV
# ===========================================================================

# Channels shown in BarnCentre — all-day sports / highlights content
_BARNCENTRE_CHANNEL_NAMES = [
    # Canadian — TSN
    "TSN1", "TSN2", "TSN3", "TSN4", "TSN5",
    # Canadian — Sportsnet
    "Sportsnet East", "Sportsnet Ontario", "Sportsnet West", "Sportsnet Pacific",
    "Sportsnet 360", "Sportsnet One",
    # US national
    "ESPN", "ESPN2",
    "NHL Network",
    "FS1", "FS2",
    # US regional (big-market)
    "NESN",
    # US sports betting / general sports
    "FanDuel",
    # US regional — Bally Sports (an upstream host carries 30 regional feeds;
    # single card multiplexes them all as backups)
    "Bally Sports",
    # US — CBS Sports (placed right before the French block)
    "CBS Sports", "CBS Sports Network",
    # Canadian French — at the back, grouped by network
    "RDS", "RDS 2", "RDS INFO",
    "TVA Sports", "TVA Sports 2",
]

import re as _re_bc

def _normalize_ch(title: str) -> str:
    """Strip source suffix and HD/SD qualifier for channel name matching.

    Handles three naming styles:
    - tvpass/thetvapp: "Sportsnet East HD"   → "sportsnet east"
    - upstream-style:   "CA (FR) RDS HD (B)"  → "rds"
                      "CA-FR | TVA SPORTS 1" → "tva sports 1"
                      "CA: RDS (FR)"         → "rds"
    - upstream label:   "RDS HD (tv14s)"       → "rds"
    """
    t = title.strip()
    # Strip upstream source labels we append: "(tv14s)", "(an upstream host)", etc.
    t = _re_bc.sub(r"\s*\([a-z0-9._-]{3,30}\)\s*$", "", t, flags=_re_bc.I)
    # Strip playlist/thetvapp suffixes
    t = _re_bc.sub(r"\s*\(thetvapp\)|\s*\(playlist\)", "", t, flags=_re_bc.I)
    # Strip trailing quality flags: HD, SD, FHD, UHD, (B), (R), (E), (D)
    t = _re_bc.sub(r"\s+(?:fhd|uhd|hd|sd)\s*$", "", t, flags=_re_bc.I)
    t = _re_bc.sub(r"\s*\([a-z]\)\s*$", "", t, flags=_re_bc.I)
    # Strip upstream country/language prefixes: "CA (FR)", "CA:", "CA-FR |", "CA "
    t = _re_bc.sub(r"^(?:CA|US|UK)(?:[-_\s]+[A-Z]{2})?\s*(?:\([A-Z]{2}\))?\s*[\|:\-]?\s*", "", t, flags=_re_bc.I)
    # Strip trailing language flag "(FR)" / "(EN)" / "(ES)" / "(DE)" — lunar
    # keeps this after "CA:" prefix consumption.
    t = _re_bc.sub(r"\s*\((?:FR|EN|ES|DE)\)\s*$", "", t, flags=_re_bc.I)
    # Normalize "TVA SPORTS 1" → "TVA SPORTS": tv14s numbers the primary feed.
    t = _re_bc.sub(r"(?i)\btva\s+sports\s+1\s*$", "TVA Sports", t)
    # Strip remaining leading/trailing punctuation
    t = t.strip(" |:-")
    return t.strip().lower()


_SHORT_FR_DISPLAYS = {"rds", "rds2", "rds info", "tva sports", "tva sports 2"}


def _ch_matches(raw_title: str, ch_name: str) -> bool:
    """Return True if an IPTV channel title matches our BarnCentre channel name."""
    norm  = _normalize_ch(raw_title)
    want  = ch_name.lower()
    # Exact match or starts with name + space (e.g. "tsn1" or "tsn1 hd")
    if norm == want or norm.startswith(want + " "):
        return True
    # Also handle "espn+" ↔ "espnplus" / "espn plus"
    if want == "espn+":
        return norm in ("espnplus", "espn plus", "espn+") or norm.startswith("espn+ ") or norm.startswith("espnplus ")
    # French-network fallback: providers label RDS as "Canal RDS HD" etc. which
    # normalises to "canal rds" ≠ "rds". Accept if every token of the display
    # name appears in the normalised title. Narrow to French displays so ESPN /
    # Sportsnet don't over-match.
    if want in _SHORT_FR_DISPLAYS:
        tokens = [t for t in want.split() if len(t) >= 2]
        if tokens and all(tok in norm for tok in tokens):
            return True
    return False


async def _verify_stream_alive(url: str, timeout: float = 10.0) -> bool:
    """Follow redirects and confirm the URL resolves to a reachable HLS stream.

    For tvpass.org/live/* URLs: follow 302 → check final URL is m3u8 + HTTP 200.
    For direct m3u8 URLs: do a lightweight HEAD/GET.
    Returns False on timeout, 404/403, or non-m3u8 destination.

    Relay /hls URLs: don't probe directly — that would spawn ffmpeg and stream
    video just to health-check. Decode the inner upstream TS URL and range-probe
    the first KB instead.

    Note on ngrok: a browser UA triggers ngrok's HTML interstitial (200 OK) even
    when the upstream is 404. Send `ngrok-skip-browser-warning` so the relay
    returns the true upstream status instead of lying.
    """
    import httpx as _hx_v
    from urllib.parse import parse_qs as _parse_qs, urlparse as _urlparse

    # /hls passthrough: probe the decoded TS URL, not the ffmpeg-backed endpoint.
    if "/hls?" in url and "u=" in url:
        try:
            inner = _parse_qs(_urlparse(url).query).get("u", [""])[0]
            if inner:
                inner_url = unquote(inner)
                async with _hx_v.AsyncClient(
                    follow_redirects=True,
                    timeout=timeout,
                    headers={
                        "User-Agent": _BROWSER_HEADERS["User-Agent"],
                        "Range":      "bytes=0-1023",
                    },
                ) as cl:
                    r = await cl.get(inner_url)
                    return r.status_code in (200, 206) and len(r.content) > 0
        except Exception:
            return False
        return False

    try:
        async with _hx_v.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": "Grtzky-StreamVerifier/1.0",
                "ngrok-skip-browser-warning": "skip",
                "Referer":  "https://tvpass.org/",
                "Origin":   "https://tvpass.org",
            },
        ) as cl:
            if url.lower().split("?")[0].endswith(".m3u8"):
                r = await cl.head(url)
                return r.status_code == 200
            r = await cl.get(url)
            final = str(r.url)
            if r.status_code in (404, 403, 410):
                return False
            # tvpass redirect lands on .m3u8 URL — confirmed live
            if ".m3u8" in final.lower() and r.status_code == 200:
                return True
            # Other 2xx destinations (embed pages) — treat as available
            return r.status_code < 400
    except Exception:
        # tvpass is our most reliable source; a transient timeout from the
        # verifier shouldn't flip the chip to "No signal" when the stream
        # actually plays (Bob saw FS2 marked offline while it was live).
        # Be optimistic for tvpass; strict for everything else.
        if "tvpass.org/live/" in url:
            return True
        return False


def _sort_by_url_priority(candidates: list[dict]) -> list[dict]:
    """Sort channel candidates: tvpass.org redirect URLs first (fresh token), raw CDN second.

    an upstream host is last — their /play/TOKEN/m3u8 endpoints 404 on real browsers
    (ngrok warning page masks this as 200); the only working endpoint serves raw
    MPEG-TS which hls.js cannot consume.
    """
    def _prio(m: dict) -> int:
        u = m["url"]
        if "tvpass.org/live" in u:
            return 0
        if "thetvapp.to" in u:
            return 1
        if "an upstream host" in u:
            return 3
        return 2
    return sorted(candidates, key=_prio)


# ESPN broadcast name → BarnCentre channel name
_ESPN_NETWORK_TO_CHANNEL: dict[str, str] = {
    "ESPN":         "ESPN",
    "ESPN2":        "ESPN2",
    "ESPN+":        "ESPN+",
    "ESPNU":        "ESPN",
    "FS1":          "FS1",
    "FS2":          "FS2",
    "NHL Network":  "NHL Network",
    "NHLN":         "NHL Network",
    "TSN":          "TSN1",
    "TSN2":         "TSN2",
    "TSN3":         "TSN3",
    "TSN4":         "TSN4",
    "TSN5":         "TSN5",
    "NESN":         "NESN",
    "FanDuel TV":   "FanDuel",
    "FuboSports":   "FanDuel",
}

# ESPN scoreboard API endpoints to poll for today's events
_ESPN_SCHEDULE_ENDPOINTS = [
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=hockey&league=nhl",
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=basketball&league=nba",
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=baseball&league=mlb",
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=football&league=nfl",
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=soccer&league=usa.1",
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=soccer&league=eng.1",
    "https://site.api.espn.com/apis/v2/scoreboard/header?sport=soccer&league=uefa.champions",
]

_espn_schedule_cache: dict = {"data": None, "ts": 0.0}
_ESPN_SCHEDULE_TTL = 3600  # 1 hour


async def _fetch_espn_schedule() -> dict[str, list[dict]]:
    """Fetch today's ESPN multi-sport schedule; return {channel_name: [programs]}."""
    import time as _t_es
    import httpx as _hx_es
    import asyncio as _aio_es

    now_ts = _t_es.time()
    if _espn_schedule_cache["data"] is not None and now_ts - _espn_schedule_cache["ts"] < _ESPN_SCHEDULE_TTL:
        return _espn_schedule_cache["data"]

    async def _fetch_one(cl, url: str) -> list[dict]:
        try:
            r = await cl.get(url, timeout=8)
            if r.status_code != 200:
                return []
            d = r.json()
            events = []
            for sport_block in d.get("sports", []):
                sport_name = sport_block.get("name", "")
                for league in sport_block.get("leagues", []):
                    league_name = league.get("name", "")
                    for event in league.get("events", []):
                        broadcasts = event.get("broadcasts", [])
                        short_name = event.get("shortName", "")
                        date_utc   = event.get("date", "")
                        state      = event.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")
                        # Map state → our game states
                        if "in_progress" in state.lower() or "progress" in state.lower():
                            game_state = "LIVE"
                        elif "final" in state.lower():
                            game_state = "FINAL"
                        else:
                            game_state = "SCH"
                        for b in broadcasts:
                            net_name = b.get("name", "")
                            ch_name = _ESPN_NETWORK_TO_CHANNEL.get(net_name)
                            if not ch_name:
                                continue
                            # Build a 3-hour window stop time
                            import datetime as _dt_es
                            try:
                                start_dt = _dt_es.datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
                                stop_dt  = start_dt + _dt_es.timedelta(hours=3)
                                stop_utc = stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                            except Exception:
                                stop_utc = ""
                            events.append({
                                "ch_name":   ch_name,
                                "game_id":   None,
                                "title":     f"{short_name} ({league_name})" if league_name else short_name,
                                "desc":      sport_name,
                                "start_utc": date_utc,
                                "stop_utc":  stop_utc,
                                "state":     game_state,
                                "market":    "N",
                            })
            return events
        except Exception:
            return []

    async with _hx_es.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 BarnCentre/1.0"},
        follow_redirects=True,
    ) as cl:
        results = await _aio_es.gather(*[_fetch_one(cl, url) for url in _ESPN_SCHEDULE_ENDPOINTS])

    # Group by channel name
    data: dict[str, list[dict]] = {}
    for event_list in results:
        for ev in event_list:
            ch = ev.pop("ch_name")
            data.setdefault(ch, []).append(ev)

    # Deduplicate by (title, start_utc) within each channel
    for ch in data:
        seen: set[tuple] = set()
        unique = []
        for p in data[ch]:
            key = (p["title"], p["start_utc"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        data[ch] = unique

    _espn_schedule_cache["data"] = data
    _espn_schedule_cache["ts"]   = now_ts
    return data


# ---------------------------------------------------------------------------
# MLB schedule — statsapi.mlb.com returns broadcast network names per game
# ---------------------------------------------------------------------------
_MLB_NETWORK_TO_CHANNEL: dict[str, str] = {
    "ESPN":         "ESPN",
    "ESPN2":        "ESPN2",
    "FS1":          "FS1",
    "FS2":          "FS2",
    "NESN":         "NESN",
    "TSN":          "TSN1",
    "TSN1":         "TSN1",
    "TSN2":         "TSN2",
    "TSN3":         "TSN3",
    "TSN4":         "TSN4",
    "TSN5":         "TSN5",
    "FanDuel Sports Network": "FanDuel",
    "FanDuel":      "FanDuel",
}

_mlb_schedule_cache: dict = {"data": None, "ts": 0.0}


async def _fetch_mlb_schedule() -> dict[str, list[dict]]:
    """Fetch today's MLB schedule with broadcast info; return {channel_name: [programs]}."""
    import time as _t_mlb
    import httpx as _hx_mlb
    import datetime as _dt_mlb

    now_ts = _t_mlb.time()
    if _mlb_schedule_cache["data"] is not None and now_ts - _mlb_schedule_cache["ts"] < _ESPN_SCHEDULE_TTL:
        return _mlb_schedule_cache["data"]

    today = _dt_mlb.date.today().isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=broadcasts(all),game(content(summary)),linescore"
    data: dict[str, list[dict]] = {}
    try:
        async with _hx_mlb.AsyncClient(timeout=8, headers={"User-Agent": "Mozilla/5.0 BarnCentre/1.0"}) as cl:
            r = await cl.get(url)
            if r.status_code != 200:
                return {}
            for date_block in r.json().get("dates", []):
                for game in date_block.get("games", []):
                    away = (game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation") or
                            game.get("teams", {}).get("away", {}).get("team", {}).get("name", ""))
                    home = (game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation") or
                            game.get("teams", {}).get("home", {}).get("team", {}).get("name", ""))
                    start_utc = game.get("gameDate", "")
                    status = game.get("status", {}).get("codedGameState", "S")
                    if status in ("F", "FT", "FR", "FO"):
                        game_state = "FINAL"
                    elif status in ("I", "MA", "MF"):
                        game_state = "LIVE"
                    else:
                        game_state = "SCH"
                    try:
                        start_dt = _dt_mlb.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                        stop_dt  = start_dt + _dt_mlb.timedelta(hours=3, minutes=30)
                        stop_utc = stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        stop_utc = ""
                    for b in game.get("broadcasts", []):
                        if b.get("type", "").upper() != "TV":
                            continue
                        net = b.get("name", "")
                        ch_name = _MLB_NETWORK_TO_CHANNEL.get(net)
                        if not ch_name:
                            continue
                        data.setdefault(ch_name, []).append({
                            "game_id":   None,
                            "title":     f"{away} @ {home} (MLB)",
                            "desc":      "Major League Baseball",
                            "start_utc": start_utc,
                            "stop_utc":  stop_utc,
                            "state":     game_state,
                            "market":    "N",
                        })
    except Exception:
        pass

    # Deduplicate
    for ch in data:
        seen: set[tuple] = set()
        unique = []
        for p in data[ch]:
            key = (p["title"], p["start_utc"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        data[ch] = unique

    _mlb_schedule_cache["data"] = data
    _mlb_schedule_cache["ts"]   = now_ts
    return data


# ---------------------------------------------------------------------------
# NBA schedule — cdn.nba.com static JSON with broadcast info
# ---------------------------------------------------------------------------
_NBA_NETWORK_TO_CHANNEL: dict[str, str] = {
    "ESPN":         "ESPN",
    "ESPN2":        "ESPN2",
    "ABC":          "ESPN",   # ABC games air on ESPN channel in our lineup
    "TNT":          None,
    "TBS":          None,
    "Prime Video":  None,
    "TSN":          "TSN1",
    "TSN1":         "TSN1",
    "TSN2":         "TSN2",
    "TSN3":         "TSN3",
    "TSN4":         "TSN4",
    "TSN5":         "TSN5",
    "NHLN":         "NHL Network",
}

_nba_schedule_cache: dict = {"data": None, "ts": 0.0}


async def _fetch_nba_schedule() -> dict[str, list[dict]]:
    """Fetch today's NBA schedule with broadcast info; return {channel_name: [programs]}."""
    import time as _t_nba
    import httpx as _hx_nba
    import datetime as _dt_nba

    now_ts = _t_nba.time()
    if _nba_schedule_cache["data"] is not None and now_ts - _nba_schedule_cache["ts"] < _ESPN_SCHEDULE_TTL:
        return _nba_schedule_cache["data"]

    today = _dt_nba.date.today().isoformat()
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
    data: dict[str, list[dict]] = {}
    try:
        async with _hx_nba.AsyncClient(
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 BarnCentre/1.0", "Accept": "application/json"},
        ) as cl:
            r = await cl.get(url)
            if r.status_code != 200:
                return {}
            league_schedule = r.json().get("leagueSchedule", {})
            for date_block in league_schedule.get("gameDates", []):
                game_date = date_block.get("gameDate", "")[:10]  # "04/14/2026 00:00:00" → first 10 chars
                # Parse MM/DD/YYYY
                try:
                    gd = _dt_nba.datetime.strptime(game_date, "%m/%d/%Y").date().isoformat()
                except Exception:
                    gd = game_date
                if gd != today:
                    continue
                for game in date_block.get("games", []):
                    away = game.get("awayTeam", {}).get("teamTricode", "")
                    home = game.get("homeTeam", {}).get("teamTricode", "")
                    start_utc = game.get("gameDateTimeUTC", "")
                    status_id = game.get("gameStatusId", 1)
                    if status_id == 3:
                        game_state = "FINAL"
                    elif status_id == 2:
                        game_state = "LIVE"
                    else:
                        game_state = "SCH"
                    try:
                        start_dt = _dt_nba.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                        stop_dt  = start_dt + _dt_nba.timedelta(hours=2, minutes=30)
                        stop_utc = stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        stop_utc = ""
                    broadcasters = game.get("broadcasters", {})
                    all_nets: list[str] = []
                    for key in ("nationalTvBroadcasters", "nationalRadioBroadcasters",
                                "homeTvBroadcasters", "awayTvBroadcasters"):
                        for b in broadcasters.get(key, []):
                            disp = b.get("broadcasterDisplay") or b.get("broadcasterAbbreviation", "")
                            if disp:
                                all_nets.append(disp)
                    for net in all_nets:
                        ch_name = _NBA_NETWORK_TO_CHANNEL.get(net)
                        if not ch_name:
                            continue
                        data.setdefault(ch_name, []).append({
                            "game_id":   None,
                            "title":     f"{away} @ {home} (NBA)",
                            "desc":      "National Basketball Association",
                            "start_utc": start_utc,
                            "stop_utc":  stop_utc,
                            "state":     game_state,
                            "market":    "N",
                        })
    except Exception:
        pass

    # Deduplicate
    for ch in data:
        seen: set[tuple] = set()
        unique = []
        for p in data[ch]:
            key = (p["title"], p["start_utc"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        data[ch] = unique

    _nba_schedule_cache["data"] = data
    _nba_schedule_cache["ts"]   = now_ts
    return data


_barncentre_cache: dict = {"data": None, "ts": 0.0}
_BARNCENTRE_TTL = 3600  # 1 hour


@app.get("/barncentre-channels")
async def barncentre_channels() -> dict:
    """Return curated, verified channel list for BarnCentre with today's NHL programs.

    Each channel is verified live by following the stream URL and checking for a
    reachable HLS manifest. Verification runs in parallel across all channels.
    Cache is valid for 1 hour.
    """
    import time as _t_bc
    import asyncio as _aio_bc
    now_ts = _t_bc.time()
    if _barncentre_cache["data"] is not None and now_ts - _barncentre_cache["ts"] < _BARNCENTRE_TTL:
        return {"channels": _barncentre_cache["data"], "cached": True}

    # ── 1. Full IPTV channel list (shared 1-hr cache) ─────────────────────────
    iptv_result  = await iptv_channels()
    all_channels: list[dict] = iptv_result.get("channels", [])

    # ── 2. Today's NHL schedule → program guide ───────────────────────────────
    import httpx as _hx_bc
    import datetime as _dt_bc
    today    = _dt_bc.date.today().isoformat()
    programs: dict[str, list] = {}
    try:
        async with _hx_bc.AsyncClient(timeout=8) as _cl:
            resp = await _cl.get(f"https://api-web.nhle.com/v1/score/{today}")
            if resp.status_code == 200:
                for game in (resp.json().get("games") or []):
                    gid        = game.get("id")
                    away       = (game.get("awayTeam") or {}).get("abbrev", "")
                    home       = (game.get("homeTeam") or {}).get("abbrev", "")
                    start_utc  = game.get("startTimeUTC", "")
                    game_state = game.get("gameState", "PRE")
                    away_score = (game.get("awayTeam") or {}).get("score")
                    home_score = (game.get("homeTeam") or {}).get("score")
                    for b in (game.get("tvBroadcasts") or []):
                        code    = b.get("network", "")
                        display = _BROADCAST_CODE_MAP.get(code, code)
                        programs.setdefault(display, []).append({
                            "game_id":    gid,
                            "title":      f"{away} @ {home}",
                            "start_utc":  start_utc,
                            "state":      game_state,
                            "market":     b.get("market", "N"),
                            "away_score": away_score,
                            "home_score": home_score,
                        })
    except Exception:
        pass

    # ── 2b. ESPN / MLB / NBA multi-sport schedule data ───────────────────────
    import asyncio as _aio_epg
    epg_data, mlb_data, nba_data = await _aio_epg.gather(
        _fetch_espn_schedule(),
        _fetch_mlb_schedule(),
        _fetch_nba_schedule(),
    )

    # Helper: merge a sport's channel programs, skipping time windows already
    # covered by a higher-priority source (NHL takes priority, then ESPN, MLB, NBA).
    import datetime as _dt_merge

    def _existing_windows(ch_name: str) -> list[tuple[str, str]]:
        windows = []
        for p in programs.get(ch_name, []):
            if p.get("start_utc"):
                try:
                    t = _dt_merge.datetime.fromisoformat(p["start_utc"].replace("Z", "+00:00"))
                    windows.append((
                        (t - _dt_merge.timedelta(hours=1)).isoformat(),
                        (t + _dt_merge.timedelta(hours=4)).isoformat(),
                    ))
                except Exception:
                    pass
        return windows

    def _in_windows(start: str, windows: list[tuple[str, str]]) -> bool:
        if not start or not windows:
            return False
        try:
            ts = _dt_merge.datetime.fromisoformat(start.replace("Z", "+00:00")).isoformat()
            return any(lo <= ts <= hi for lo, hi in windows)
        except Exception:
            return False

    def _merge_sport(sport_data: dict[str, list[dict]]) -> None:
        for ch_name, new_progs in sport_data.items():
            windows = _existing_windows(ch_name)
            non_overlap = [p for p in new_progs if not _in_windows(p.get("start_utc", ""), windows)]
            programs.setdefault(ch_name, []).extend(non_overlap)

    _merge_sport(epg_data)
    _merge_sport(mlb_data)
    _merge_sport(nba_data)

    # Sort each channel's programs chronologically
    for ch_name in programs:
        programs[ch_name].sort(key=lambda p: p.get("start_utc", ""))

    # ── 3. Match IPTV channels to our curated list ────────────────────────────
    # Assign each title to exactly one channel. Longer/more-specific channel
    # names claim first so "CBS Sports Network" and "RDS 2" don't leak into
    # the bare "CBS Sports" / "RDS" candidate pools.
    sorted_names = sorted(_BARNCENTRE_CHANNEL_NAMES, key=lambda n: -len(n))
    assigned: dict[int, str] = {}
    for ch_name in sorted_names:
        for ch in all_channels:
            if id(ch) in assigned:
                continue
            if _ch_matches(ch.get("title", ""), ch_name):
                assigned[id(ch)] = ch_name

    channel_candidates: dict[str, list[dict]] = {}
    for ch in all_channels:
        name = assigned.get(id(ch))
        if not name:
            continue
        channel_candidates.setdefault(name, []).append(ch)
    for name, cands in list(channel_candidates.items()):
        seen: set[str] = set()
        unique: list[dict] = []
        for m in cands:
            if m["url"] not in seen:
                seen.add(m["url"])
                unique.append(m)
        channel_candidates[name] = _sort_by_url_priority(unique)

    # ── 4. Verify each channel's primary stream in parallel ───────────────────
    async def _build_channel(ch_name: str, candidates: list[dict]) -> dict:
        """Check primary (and first backup) for liveness; return channel dict."""
        # Try candidates in priority order — use first live one as primary
        verified_primary: str | None = None
        for cand in candidates[:3]:          # check up to 3 URLs per channel
            alive = await _verify_stream_alive(cand["url"])
            if alive:
                verified_primary = cand["url"]
                break

        # If none verified live, fall back to top-priority URL and mark offline
        primary    = verified_primary or candidates[0]["url"]
        backup_src = [c["url"] for c in candidates if c["url"] != primary][:4]

        return {
            "name":         ch_name,
            "primary_url":  primary,
            "backup_urls":  backup_src,
            "programs":     programs.get(ch_name, []),
            "online":       verified_primary is not None,
        }

    tasks  = [_build_channel(n, c) for n, c in channel_candidates.items()]
    result = list(await _aio_bc.gather(*tasks))

    # Preserve the _BARNCENTRE_CHANNEL_NAMES ordering
    order  = {n: i for i, n in enumerate(_BARNCENTRE_CHANNEL_NAMES)}
    result.sort(key=lambda ch: order.get(ch["name"], 999))

    # ── French-channel override ───────────────────────────────────────────────
    # tv14s and lunar provider accounts actively block our relay IP (tv14s CDN
    # tarpit / lunar 403 Forbidden), so they're excluded. Stream preference is
    # hard-coded per channel from field testing; only the accounts that play
    # are listed here. The first-seen URL per account wins.
    _fr_priority: dict[str, list[str]] = {
        "RDS":          ["an upstream host", "ampztl-a", "ampztl-b"],
        "RDS 2":        ["an upstream host", "ampztl-a", "ampztl-b"],
        "RDS INFO":     ["an upstream host"],
        "TVA Sports":   ["an upstream host"],
        "TVA Sports 2": ["an upstream host", "ampztl-a", "ampztl-b"],
    }
    _acct_re  = _re_bc.compile(r"\(([a-z0-9._-]{3,30})\)\s*$", _re_bc.I)
    _decor_re = _re_bc.compile(r"(?:\s+[^\w\s]+)+\s*$")
    def _is_fr_exact(title: str, ch_name: str) -> bool:
        norm = _normalize_ch(title)
        want = ch_name.lower()
        if norm == want:
            return True
        return _decor_re.sub("", norm).strip() == want

    def _build_fr_channel(ch_name: str, priority: list[str]) -> dict:
        by_acct: dict[str, str] = {}
        for cand in all_channels:
            title = cand.get("title", "")
            if not _is_fr_exact(title, ch_name):
                continue
            m = _acct_re.search(title)
            if not m:
                continue
            acct = m.group(1).lower()
            if acct not in priority or acct in by_acct:
                continue
            url = cand.get("url", "")
            if url:
                by_acct[acct] = url
        ordered_urls = [by_acct[a] for a in priority if a in by_acct]
        return {
            "name":         ch_name,
            "primary_url":  ordered_urls[0] if ordered_urls else "",
            "backup_urls":  ordered_urls[1:],
            "programs":     programs.get(ch_name, []),
            "online":       bool(ordered_urls),
        }

    # Replace whatever the default match produced for the French channels.
    result = [ch for ch in result if ch["name"] not in _fr_priority]
    for ch_name, pri in _fr_priority.items():
        built = _build_fr_channel(ch_name, pri)
        if built["primary_url"]:
            result.append(built)

    # Re-apply the global ordering after the French rebuild.
    result.sort(key=lambda ch: order.get(ch["name"], 999))
    # ──────────────────────────────────────────────────────────────────────────

    _barncentre_cache["data"] = result
    _barncentre_cache["ts"]   = now_ts
    return {"channels": result, "cached": False}
