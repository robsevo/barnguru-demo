import asyncio
import collections.abc
import json
import os
import sys
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

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
    allow_methods=["GET", "POST"],
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


def _context_parquets(subdir: str, prefix: str, context: str) -> list[Path]:
    """Pick the right set of parquets for a season-or-playoff context.

    context="playoffs" → files matching {prefix}*_playoffs.parquet
    anything else      → files matching {prefix}*.parquet, EXCLUDING *_playoffs.parquet

    Returns an empty list if the directory doesn't exist.
    """
    d = _GRETZKY_DATA_DIR / subdir
    if not d.exists():
        return []
    all_files = sorted(d.glob(f"{prefix}*.parquet"))
    if context == "playoffs":
        return [p for p in all_files if p.stem.endswith("_playoffs")]
    return [p for p in all_files if not p.stem.endswith("_playoffs")]


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


_BEHAVIOR_LEAGUE_AVG_CACHE: dict[tuple[str, int], dict[str, dict[str, float]]] = {}


def _behavior_league_avg(beh_path: Path) -> dict[str, dict[str, float]] | None:
    """Per-position league-mean action probabilities (percentages).

    Returns a dict shaped ``{"all": {...}, "forwards": {...}, "defense": {...}}``
    so the UI can render delta-vs-peers with a fair comparison group: a D's
    shoot_perimeter looks dominant against the global mean (forwards dilute
    the average) but reads as average against the D-only mean — which is the
    honest signal. We keep ``all`` for backward compat and the rare case
    where position is unknown.

    Position comes from the shots parquet (``shooter_id → player_position``),
    joined on player_id. Players missing from shots fall into ``all`` only.

    Cached on (path, mtime_ns); recomputed when the behavior parquet is
    overwritten by a fresh training run.
    """
    try:
        mtime = beh_path.stat().st_mtime_ns
    except OSError:
        return None
    key = (str(beh_path), mtime)
    if key in _BEHAVIOR_LEAGUE_AVG_CACHE:
        return _BEHAVIOR_LEAGUE_AVG_CACHE[key]
    try:
        import polars as pl
        cols = ["carry_in", "dump", "shoot_slot", "shoot_perimeter", "drive_net", "battle_corner", "hold_corner"]
        df = pl.read_parquet(beh_path, columns=["player_id"] + cols)
        if df.is_empty():
            return None
        # Join player_position from the most recent shots parquet. shooter_id
        # is the same NHL player_id used in behavior_predictions.
        shots_dir = _GRETZKY_DATA_DIR / "shots"
        shots_files = sorted(shots_dir.glob("*.parquet")) if shots_dir.exists() else []
        pos_df: pl.DataFrame | None = None
        if shots_files:
            try:
                raw = pl.read_parquet(shots_files[-1], columns=["shooter_id", "player_position"])
                pos_df = (raw.filter(pl.col("shooter_id").is_not_null()
                                     & pl.col("player_position").is_not_null())
                              .unique(subset=["shooter_id"])
                              .rename({"shooter_id": "player_id"}))
            except Exception:
                pos_df = None
        joined = df.join(pos_df, on="player_id", how="left") if pos_df is not None else df.with_columns(pl.lit(None).alias("player_position"))

        def _mean(frame: pl.DataFrame) -> dict[str, float]:
            return {c: round(float(frame[c].drop_nulls().mean() or 0.0) * 100, 1) for c in cols}

        out: dict[str, dict[str, float]] = {"all": _mean(joined)}
        # Forwards: C / L / R. Defense: D. (G is filtered by is_goalie upstream.)
        fwd = joined.filter(pl.col("player_position").is_in(["C", "L", "R"]))
        de  = joined.filter(pl.col("player_position") == "D")
        if not fwd.is_empty():
            out["forwards"] = _mean(fwd)
        if not de.is_empty():
            out["defense"] = _mean(de)
    except Exception:
        return None
    _BEHAVIOR_LEAGUE_AVG_CACHE[key] = out
    return out


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

    from datetime import date, timedelta
    _today_local = date.today()
    # /schedule/{date} returns a 7-day gameWeek starting from `date`. To cover
    # ~14 days forward (the ECF/Cup window can span multiple rounds with sparse
    # scheduling), we fetch the current week plus the next two weeks.
    week_anchor_1 = _today_local.isoformat()
    week_anchor_2 = (_today_local + timedelta(days=7)).isoformat()
    week_anchor_3 = (_today_local + timedelta(days=14)).isoformat()
    try:
        async with NHLClient() as client:
            raw, sched_raw, sched_next, sched_next2 = await asyncio.gather(
                client.get_scoreboard(),
                client.get_schedule(week_anchor_1),
                client.get_schedule(week_anchor_2),
                client.get_schedule(week_anchor_3),
                return_exceptions=True,
            )
            if isinstance(raw, Exception):
                raise raw
    except NHLApiError as exc:
        return {"games": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"games": [], "error": f"Unexpected error: {exc}"}

    # Merge the three weekly schedules into a single iterable of gameWeek
    # buckets — duplicate dates will be deduped by game_id below.
    schedule_buckets: list[dict] = []
    for s in (sched_raw, sched_next, sched_next2):
        if isinstance(s, dict):
            schedule_buckets.extend(s.get("gameWeek") or [])

    # NHL's /score/now omits seriesStatus entirely. /schedule has it for every
    # playoff game going forward — match by game_id first, then fall back to the
    # team-pair (yesterday's FINAL won't be in schedule, but tomorrow's game 2
    # in the same series carries the post-game-1 series score).
    series_by_id: dict[int, dict] = {}
    series_by_pair: dict[frozenset, dict] = {}
    for bucket in schedule_buckets:
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
    if not games_by_date and not schedule_buckets:
        return {"games": []}

    # Use NHL's focusedDate to avoid UTC midnight / ET timezone mismatch
    today_str          = raw.get("focusedDate") or _today_local.isoformat()
    _today             = date.fromisoformat(today_str)
    yesterday_str      = (_today - timedelta(days=1)).isoformat()
    two_days_ago_str   = (_today - timedelta(days=2)).isoformat()
    # Look 14 days forward so the ECF / Cup Final window stays populated even
    # when only one game per series is scheduled at a time.
    forward_dates = [(_today + timedelta(days=i)).isoformat() for i in range(1, 15)]
    keep_dates    = {today_str, yesterday_str, two_days_ago_str, *forward_dates}

    all_games: list[dict] = []
    date_labels: dict = {}
    seen_game_ids: set[int] = set()
    for bucket in games_by_date:
        date_str = bucket.get("date", "")
        if date_str not in keep_dates:
            continue
        for g in bucket.get("games") or []:
            all_games.append(g)
            gid = g.get("id")
            if gid:
                date_labels[gid] = date_str
                seen_game_ids.add(gid)

    # Supplement future dates from the merged schedule weeks so the strip can
    # carry up to ~14 days forward even when /score/now only covers ~2 days.
    for bucket in schedule_buckets:
            date_str = bucket.get("date", "")
            if date_str not in keep_dates:
                continue
            for g in bucket.get("games") or []:
                gid = g.get("id")
                if gid and gid in seen_game_ids:
                    continue
                all_games.append(g)
                if gid:
                    date_labels[gid] = date_str
                    seen_game_ids.add(gid)

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
    cached = _cache_get("standings")
    if cached is not None:
        return cached  # type: ignore[return-value]

    from data.nhl_client import NHLClient, NHLApiError

    from datetime import date as _d, timedelta as _td
    today = _d.today()

    # During playoffs / offseason /v1/standings/<today> returns {"standings":[]}.
    # Walk back day-by-day until we find a date with real records so the frontend
    # keeps rendering end-of-regular-season standings. Window must cover the full
    # playoff run + early-offseason gap from the regular-season finale (~75 days
    # SCF Game 7 → April finale; we add slack for an extended offseason view).
    raw: dict = {"standings": []}
    try:
        async with NHLClient() as client:
            for back in range(120):
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

    payload = {"standings": out}
    if out:
        _cache_set("standings", payload, ttl=600.0)
    return payload


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
    context: str = Query("season", description="playoffs | season — selects per-60 aggregates"),
) -> dict:
    """Player rating lookup. Returns real xG Finishing data once model is trained.

    ``context`` is honoured for the metrics that have a per-context aggregate
    available (playoff_delta carries playoff_* and reg_* per-60 splits). For
    metrics that only exist as season-long aggregates (RAPM, WAR, EWMA, hot
    hand, etc.) the season number is returned in both contexts — the field
    ``context_applied`` lets the dashboard label them honestly.
    """
    import polars as pl

    # Context-aware xG finishing source: when ?context=playoffs is requested
    # and a playoff parquet exists, use it directly. Otherwise fall back to
    # the season-aggregate file.
    parquets = _context_parquets("xg_finishing", "xg_finishing_", context)
    xg_source = "playoffs" if context == "playoffs" and parquets else "season"
    # When playoffs requested but no playoff parquet yet, fall back to season so
    # the page still renders (cards will be season-aggregate and labelled).
    if not parquets:
        parquets = _context_parquets("xg_finishing", "xg_finishing_", "season")

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

    # Try to join CDR — pick context-matching parquet directly so we don't
    # rely on the model's read_cdr (which doesn't know about _playoffs files).
    cdr_val: float | None = None
    try:
        cdr_files = _context_parquets("cdr", "cdr_", context) or _context_parquets("cdr", "cdr_", "season")
        if cdr_files:
            cdr_df = pl.read_parquet(cdr_files[-1])
            cdr_row = cdr_df.filter(pl.col("player_name").str.to_lowercase() == r["shooter_name"].lower())
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
        # Pick playoff RAPM parquet first when context=playoffs, fall back to
        # season variant if no playoff data exists for this season yet.
        rapm_path = None
        if season_val:
            if context == "playoffs":
                _po = rapm_dir / f"rapm_{season_val}_playoffs.parquet"
                if _po.exists():
                    rapm_path = _po
            if rapm_path is None:
                _rg = rapm_dir / f"rapm_{season_val}.parquet"
                if _rg.exists():
                    rapm_path = _rg
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
        war_parquets = _context_parquets("war", "war_", context) \
                       or _context_parquets("war", "war_", "season")
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
        ewma_parquets = _context_parquets("ewma", "ewma_form_", context)
        if not ewma_parquets:
            ewma_parquets = _context_parquets("ewma", "ewma_form_", "season")
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
        hh_parquets = _context_parquets("hot_hand", "hot_hand_summary_", context)
        if not hh_parquets:
            hh_parquets = _context_parquets("hot_hand", "hot_hand_summary_", "season")
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
        ci_parquets = _context_parquets("clutch_index", "clutch_index_2", context) \
                      or _context_parquets("clutch_index", "clutch_index_2", "season")
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
        st_parquets = _context_parquets("special_teams", "special_teams_", context) \
                      or _context_parquets("special_teams", "special_teams_", "season")
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
        br_parquets = _context_parquets("bayes_ratings", "player_ratings_", context) \
                      or _context_parquets("bayes_ratings", "player_ratings_", "season")
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
    nn_league_avg_payload:      dict[str, float] | None = None
    nn_league_avg_by_pos_payload: dict[str, dict[str, float]] | None = None
    try:
        beh_dir = _GRETZKY_DATA_DIR / "behavior_net"
        beh_parquets = sorted(beh_dir.glob("behavior_predictions_*.parquet")) if beh_dir.exists() else []
        if beh_parquets and player_id_val is not None:
            beh_path = beh_parquets[-1]
            beh_df = pl.read_parquet(beh_path)
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
            full = _behavior_league_avg(beh_path)
            if full is not None:
                # ``all`` stays in the flat field for back-compat with any
                # client still reading nn_league_avg as a flat dict.
                nn_league_avg_payload = full.get("all")
                nn_league_avg_by_pos_payload = full
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

    # NHL EDGE rankings — top shot speed, top skating speed, total distance.
    # Pulled from the latest edge_skating + edge_shot_speed parquets. We
    # compute the player's own value plus the league percentile + rank so
    # the UI can render "87.5 mph · rank 142/612 · 76th percentile" the way
    # NHL EDGE pages do, without re-walking the parquet on every render.
    edge_top_shot_speed:     float | None = None
    edge_top_shot_rank:      int   | None = None
    edge_top_shot_pop:       int   | None = None
    edge_top_shot_pct:       float | None = None
    edge_hard_shot_count:    int   | None = None
    edge_high_danger_shots:  int   | None = None
    edge_hd_shots_rank:      int   | None = None
    edge_hd_shots_pop:       int   | None = None
    edge_hd_shots_pct:       float | None = None
    edge_top_skate_speed:    float | None = None
    edge_top_skate_rank:     int   | None = None
    edge_top_skate_pop:      int   | None = None
    edge_top_skate_pct:      float | None = None
    edge_total_distance:     float | None = None
    edge_avg_speed_kmh:      float | None = None
    edge_games_played:       int   | None = None
    try:
        edge_dir = _GRETZKY_DATA_DIR / "edge"
        if edge_dir.exists() and player_id_val is not None:
            shot_parquets = sorted(edge_dir.glob("edge_shot_speed_*.parquet"))
            if shot_parquets:
                shot_df = pl.read_parquet(shot_parquets[-1])
                shot_pop = shot_df.filter(pl.col("max_shot_speed_mph").is_not_null()).shape[0]
                shot_row = shot_df.filter(pl.col("player_id") == player_id_val)
                if not shot_row.is_empty():
                    sr = shot_row.to_dicts()[0]
                    mv = sr.get("max_shot_speed_mph")
                    if mv is not None:
                        edge_top_shot_speed = round(float(mv), 1)
                        ranked = shot_df.filter(pl.col("max_shot_speed_mph") > mv).shape[0]
                        edge_top_shot_rank = ranked + 1
                        edge_top_shot_pop  = shot_pop
                        edge_top_shot_pct  = round((shot_pop - edge_top_shot_rank) / max(shot_pop - 1, 1) * 100, 0) if shot_pop > 1 else None
                    hsc = sr.get("hard_shot_count")
                    if hsc is not None:
                        edge_hard_shot_count = int(hsc)
                    hd = sr.get("high_danger_shots")
                    if hd is not None:
                        edge_high_danger_shots = int(hd)
                # High-danger shot rank lives on the same parquet but a
                # different column — compute it in the same pass to save
                # a second read.
                if "high_danger_shots" in shot_df.columns:
                    hd_pop = shot_df.filter(pl.col("high_danger_shots").is_not_null()).shape[0]
                    if edge_high_danger_shots is not None and hd_pop > 0:
                        ranked = shot_df.filter(pl.col("high_danger_shots") > edge_high_danger_shots).shape[0]
                        edge_hd_shots_rank = ranked + 1
                        edge_hd_shots_pop  = hd_pop
                        edge_hd_shots_pct  = round((hd_pop - edge_hd_shots_rank) / max(hd_pop - 1, 1) * 100, 0) if hd_pop > 1 else None
            skate_parquets = sorted(edge_dir.glob("edge_skating_*.parquet"))
            if skate_parquets:
                skate_df = pl.read_parquet(skate_parquets[-1])
                skate_pop = skate_df.filter(pl.col("max_speed_kmh").is_not_null()).shape[0]
                skate_row = skate_df.filter(pl.col("player_id") == player_id_val)
                if not skate_row.is_empty():
                    er = skate_row.to_dicts()[0]
                    mxk = er.get("max_speed_kmh")
                    if mxk is not None:
                        edge_top_skate_speed = round(float(mxk), 2)
                        ranked = skate_df.filter(pl.col("max_speed_kmh") > mxk).shape[0]
                        edge_top_skate_rank = ranked + 1
                        edge_top_skate_pop  = skate_pop
                        edge_top_skate_pct  = round((skate_pop - edge_top_skate_rank) / max(skate_pop - 1, 1) * 100, 0) if skate_pop > 1 else None
                    if er.get("total_distance_km") is not None:
                        edge_total_distance = round(float(er["total_distance_km"]), 1)
                    if er.get("avg_speed_kmh") is not None:
                        edge_avg_speed_kmh = round(float(er["avg_speed_kmh"]), 2)
                    if er.get("games_played") is not None:
                        edge_games_played = int(er["games_played"])
    except Exception:
        pass

    # ── Context-aware per-60 substitution ──────────────────────────────
    # When the caller asks for ``playoffs`` and the playoff_delta parquet
    # has a row for this player, replace the per-60 stats that have a
    # genuine playoff split. The remaining (RAPM, WAR, hot hand, ...)
    # stay season-aggregate — flagged via ``context_applied`` so the
    # dashboard knows what is or isn't context-aware.
    #
    # NOTE: xG Finishing (shots / goals / xg_sum / finishing / finishing_per60)
    # is already context-correct because we picked the right xg_finishing
    # parquet above based on ``xg_source``. So we flag it via context_applied
    # as soon as that was a playoffs read, even if playoff_delta itself fails.
    context_applied = "playoffs" if xg_source == "playoffs" else "season"
    playoff_gp_val:       int   | None = None
    playoff_toi_ev_min:   float | None = None
    playoff_goals_total:  int   | None = None
    if context == "playoffs" and player_id_val is not None:
        try:
            pd_dir2 = _GRETZKY_DATA_DIR / "playoff_delta"
            pd_parquets2 = sorted(pd_dir2.glob("playoff_delta_*.parquet")) if pd_dir2.exists() else []
            if pd_parquets2:
                pd_df2 = pl.read_parquet(pd_parquets2[-1])
                pd_row2 = pd_df2.filter(pl.col("player_id") == player_id_val)
                if not pd_row2.is_empty():
                    pr = pd_row2.to_dicts()[0]
                    po_xgf  = pr.get("playoff_xgf_per60")
                    po_g60  = pr.get("playoff_goals_per60")
                    po_toi  = pr.get("playoff_toi_sec")
                    po_gp   = pr.get("playoff_gp")
                    if po_xgf is not None:
                        rapm_xgf_60 = round(float(po_xgf), 2)
                    if po_g60 is not None:
                        rapm_goals_p60 = round(float(po_g60), 2)
                    if po_toi is not None:
                        rapm_toi_ev = float(po_toi)
                    if po_gp is not None:
                        playoff_gp_val = int(po_gp)
                    if po_toi is not None and po_g60 is not None and float(po_toi) > 0:
                        toi_h = float(po_toi) / 3600.0
                        playoff_goals_total = round(float(po_g60) * toi_h)
                    context_applied = "playoffs"
        except Exception:
            pass

    return {
        "player_name":          r["shooter_name"],
        "player_id":            player_id_val,
        "team":                 r.get("team"),
        "season":               season_val2,
        "context":              context,
        "context_applied":      context_applied,
        "shots":                r.get("shots"),
        # If finishing data already came from the playoff parquet, r["goals"]
        # is playoff goals directly. Only fall back to playoff_delta's derived
        # estimate when finishing was season-aggregate but we still want to
        # surface a playoff goal count.
        "goals":                r.get("goals") if xg_source == "playoffs" else (playoff_goals_total if context_applied == "playoffs" else r.get("goals")),
        "xg_source":            xg_source,
        "playoff_gp":           playoff_gp_val,
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
        "nn_hold_corner_pct":        nn_hold_corner_pct_val,
        "nn_league_avg":             nn_league_avg_payload,
        "nn_league_avg_by_pos":      nn_league_avg_by_pos_payload,
        "skating_zone_time_oz_pct":     skating_zone_oz_val,
        "skating_zone_time_dz_pct":     skating_zone_dz_val,
        "skating_avg_speed_kmh":        skating_avg_speed_val,
        "skating_max_speed_kmh":        skating_max_speed_val,
        "skating_distance_per_game_km": skating_distance_val,
        "skating_games_sample":         skating_games_val,
        # NHL EDGE — top speeds + ranking + distance
        "edge_top_shot_speed_mph":      edge_top_shot_speed,
        "edge_top_shot_speed_rank":     edge_top_shot_rank,
        "edge_top_shot_speed_pop":      edge_top_shot_pop,
        "edge_top_shot_speed_pct":      edge_top_shot_pct,
        "edge_hard_shot_count":         edge_hard_shot_count,
        "edge_high_danger_shots":       edge_high_danger_shots,
        "edge_high_danger_shots_rank":  edge_hd_shots_rank,
        "edge_high_danger_shots_pop":   edge_hd_shots_pop,
        "edge_high_danger_shots_pct":   edge_hd_shots_pct,
        "edge_top_skating_speed_kmh":   edge_top_skate_speed,
        "edge_top_skating_speed_rank":  edge_top_skate_rank,
        "edge_top_skating_speed_pop":   edge_top_skate_pop,
        "edge_top_skating_speed_pct":   edge_top_skate_pct,
        "edge_total_distance_km":       edge_total_distance,
        "edge_avg_speed_kmh":           edge_avg_speed_kmh,
        "edge_games_played":            edge_games_played,
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


def _build_player_meta() -> dict[int, dict]:
    """Build player_id → {team, position, name} from cached roster JSONs.

    Used by dashboard endpoints to enrich leaderboard rows with team
    abbreviation and position so the frontend can render logos + tier
    badges instead of bare names. Falls back to xg_finishing /
    goalie_stats parquets for IDs that aren't on a current roster
    (recently traded / retired but still in season-aggregate data).
    """
    import json as _json
    import polars as pl

    meta: dict[int, dict] = {}

    # Roster JSON is the most accurate "current team" source.
    raw_dir = _GRETZKY_DATA_DIR / "raw"
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("roster_*.json")):
            try:
                data = _json.loads(f.read_text())
                team_code = data.get("team_code", "")
                for p in data.get("profiles", []):
                    pid = p.get("player_id")
                    if not pid:
                        continue
                    pid_i = int(pid)
                    first = p.get("first_name") or ""
                    last  = p.get("last_name") or ""
                    meta[pid_i] = {
                        "team":     team_code,
                        "position": p.get("position") or "",
                        "name":     f"{first} {last}".strip(),
                    }
            except Exception:
                pass

    # Fall back to skater stats parquet for any IDs not on a roster.
    xg_dir = _GRETZKY_DATA_DIR / "xg_finishing"
    xg_files = sorted(xg_dir.glob("xg_finishing_*.parquet")) if xg_dir.exists() else []
    if xg_files:
        try:
            df = pl.read_parquet(xg_files[-1])
            if "shooter_id" in df.columns:
                pos_map = _shots_name_position_map()
                for r in df.select(["shooter_id", "shooter_name", "team"]).drop_nulls(subset=["shooter_id"]).unique(subset=["shooter_id"]).to_dicts():
                    pid = int(r["shooter_id"])
                    if pid in meta:
                        continue
                    meta[pid] = {
                        "team":     r.get("team") or "",
                        "position": pos_map.get(r.get("shooter_name") or "", ""),
                        "name":     r.get("shooter_name") or "",
                    }
        except Exception:
            pass

    # Goalies fallback.
    goalie_dir = _GRETZKY_DATA_DIR / "goalie_stats"
    g_files = sorted(goalie_dir.glob("goalie_stats_*.parquet")) if goalie_dir.exists() else []
    if g_files:
        try:
            gdf = pl.read_parquet(g_files[-1], columns=["player_id", "player_name", "team"])
            for r in gdf.drop_nulls(subset=["player_id"]).unique(subset=["player_id"]).to_dicts():
                pid = int(r["player_id"])
                if pid in meta:
                    continue
                meta[pid] = {
                    "team":     r.get("team") or "",
                    "position": "G",
                    "name":     r.get("player_name") or "",
                }
        except Exception:
            pass

    return meta


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
async def phase2_war_leaderboard(
    limit: int = 10,
    player_id: int | None = None,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Top WAR players (min EV TOI threshold flexes with context)."""
    import polars as pl
    parquets = _context_parquets("war", "war_", context)
    source = "playoffs" if context == "playoffs" and parquets else "season"
    if not parquets:
        parquets = _context_parquets("war", "war_", "season")
    if not parquets:
        return {"players": [], "built": False, "source": source}
    try:
        df = pl.read_parquet(parquets[-1])
        if "war" not in df.columns:
            return {"players": [], "built": False, "source": source}
        name_lut = _build_name_lookup()

        # Playoff TOI caps at ~10% of regular season — drop the floor.
        MIN_TOI = 20.0 if source == "playoffs" else 200.0
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

        return {"players": players, "selected": selected, "built": True, "min_toi": MIN_TOI, "source": source}
    except Exception:
        return {"players": [], "selected": None, "built": False}


@app.get("/phase2/ewma-movers")
async def phase2_ewma_movers(
    limit: int = 6,
    player_id: int | None = None,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Top EWMA form movers ranked by delta vs league mean. Includes team from RAPM."""
    import polars as pl
    parquets = _context_parquets("ewma", "ewma_form_", context)
    source = "playoffs" if context == "playoffs" and parquets else "season"
    if not parquets:
        parquets = _context_parquets("ewma", "ewma_form_", "season")
    if not parquets:
        return {"rising": [], "falling": [], "built": False, "source": source}
    try:
        df = pl.read_parquet(parquets[-1])
        ewma_col = "ewma_xgf60" if "ewma_xgf60" in df.columns else "current_ewma"
        if ewma_col not in df.columns:
            return {"rising": [], "falling": [], "built": False, "source": source}

        # Playoffs cap at ~28 GP per team — drop the floor accordingly.
        MIN_GAMES = 4 if source == "playoffs" else 20
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
            "source":       source,
        }
    except Exception:
        return {"rising": [], "falling": [], "built": False, "selected": None, "source": source}


@app.get("/phase2/matchup-explorer")
async def phase2_matchup_explorer(
    limit: int = 10,
    player_id: int | None = None,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """QoT/QoC leaders and top player-pair matchup predictions."""
    import polars as pl
    qot_parquets = _context_parquets("qot_qoc", "qot_qoc_", context) \
                   or _context_parquets("qot_qoc", "qot_qoc_", "season")
    mp_parquets  = _context_parquets("matchup_preds", "matchup_preds_", context) \
                   or _context_parquets("matchup_preds", "matchup_preds_", "season")
    source = "playoffs" if context == "playoffs" and (
        _context_parquets("qot_qoc", "qot_qoc_", context) or
        _context_parquets("matchup_preds", "matchup_preds_", context)
    ) else "season"
    if not qot_parquets and not mp_parquets:
        return {"qot": [], "qoc": [], "top_pairs": [], "built": False, "source": source}

    # Playoff GP caps at ~28 — drop the floor accordingly.
    MIN_GP = 4 if source == "playoffs" else 20

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
            "source":          source,
        }
    except Exception:
        return {"qot": [], "qoc": [], "top_pairs": [], "selected_player": None, "selected_pairs": [], "built": False, "source": source}


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
async def phase2_top_pairs(
    limit: int = 10,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Return top line pairs by chemistry delta (how much better they are together vs. expected)."""
    import polars as pl

    files = _context_parquets("chemistry", "pair_chemistry_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("chemistry", "pair_chemistry_", "season")
    if not files:
        return {"pairs": [], "status": "not_built", "source": source}

    try:
        df = pl.read_parquet(files[-1])
        if df.is_empty():
            return {"pairs": [], "status": "empty", "source": source}

        name_lookup = _build_name_lookup()

        # Team lookup from RAPM — match the context so playoff rosters surface correctly.
        team_lookup: dict[int, str] = {}
        try:
            rapm_files = _context_parquets("rapm", "rapm_", context) or _context_parquets("rapm", "rapm_", "season")
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

        return {"pairs": pairs, "status": "ok", "source": source}
    except Exception:
        return {"pairs": [], "status": "error", "source": source}


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
async def phase2_clutch_leaderboard(
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Clutch Index leaderboard (WPA above expected, Bayesian-shrunk)."""
    import polars as pl
    files = _context_parquets("clutch_index", "clutch_index_2", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("clutch_index", "clutch_index_2", "season")
    if not files:
        return {"players": [], "built": False, "source": source}
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

        # Playoffs have ~10% of regular-season TOI; drop the floor accordingly.
        MIN_TOI_HRS = 0.5 if source == "playoffs" else 5.0
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
        return {"players": rows, "built": True, "source": source}
    except Exception:
        return {"players": [], "built": False, "source": source}


@app.get("/phase2/special-teams-leaderboard")
async def phase2_special_teams_leaderboard(
    side: str = Query("pp", description="pp | pk"),
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Power Play or Penalty Kill xGF/60 leaderboard."""
    import polars as pl
    files = _context_parquets("special_teams", "special_teams_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("special_teams", "special_teams_", "season")
    if not files:
        return {"players": [], "built": False, "side": side, "source": source}
    try:
        df = pl.read_parquet(files[-1])
        toi_col  = "toi_pp"  if side == "pp" else "toi_pk"
        xgf_col  = "pp_xgf60" if side == "pp" else "pk_xgf60"
        rapm_col = "rapm_pp"  if side == "pp" else "rapm_pk"
        name_lut = _build_name_lookup()
        # Playoff TOI is ~10% of regular season; floor 5 min is enough to weed
        # out one-off PP/PK appearances while keeping real specialists.
        MIN_TOI = 5.0 if source == "playoffs" else 50.0
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
        return {"players": rows, "built": True, "side": side, "source": source}
    except Exception:
        return {"players": [], "built": False, "side": side, "source": source}


@app.get("/phase2/hot-hand-leaderboard")
async def phase2_hot_hand_leaderboard(
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Hot Hand Signal leaderboard — 5-game burst goals-vs-xG z-score."""
    import polars as pl
    files = _context_parquets("hot_hand", "hot_hand_summary_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("hot_hand", "hot_hand_summary_", "season")
    if not files:
        return {"players": [], "built": False, "source": source}
    try:
        df = pl.read_parquet(files[-1])
        name_lut = _build_name_lookup()
        # Playoff burst signal works on a tiny sample (5 games == half a series).
        # Drop the floor so the leaderboard isn't empty.
        MIN_GP = 3 if source == "playoffs" else 5
        if "games_played" in df.columns:
            df = df.filter(pl.col("games_played") >= MIN_GP)
        df = df.filter(pl.col("hot_hand_score").is_not_null())
        ranked = df.sort("hot_hand_score", descending=True).head(limit)
        rows = []
        # Team lookup — use the matching-context RAPM parquet first so the team
        # reflects the playoff roster, not stale regular-season assignments.
        team_lut: dict[int, str | None] = {}
        rapm_files = _context_parquets("rapm", "rapm_", context) or _context_parquets("rapm", "rapm_", "season")
        if rapm_files:
            try:
                rdf = pl.read_parquet(rapm_files[-1], columns=["player_id", "team"])
                for tr in rdf.to_dicts():
                    if tr.get("player_id") is not None:
                        team_lut[int(tr["player_id"])] = _abbr(tr.get("team"))
            except Exception:
                pass
        for i, r in enumerate(ranked.to_dicts()):
            pid  = r.get("player_id")
            name = r.get("player_name") or (name_lut.get(int(pid)) if pid else None) or f"id_{pid}"
            team = team_lut.get(int(pid)) if pid else None
            rows.append({
                "rank":        i + 1,
                "player_id":   pid,
                "player_name": name,
                "team":        team,
                "value":       round(float(r["hot_hand_score"]), 3),
                "goals_5g":    r.get("goals_5g"),
                "xg_5g":       round(float(r["xg_5g"]), 2) if r.get("xg_5g") is not None else None,
            })
        return {"players": rows, "built": True, "source": source}
    except Exception:
        return {"players": [], "built": False, "source": source}


@app.get("/phase2/xg-leaderboard")
async def phase2_xg_leaderboard(
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """xGF/60 leaderboard from EWMA form model (min 20 GP regular / 4 GP playoffs)."""
    import polars as pl
    files = _context_parquets("ewma", "ewma_form_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("ewma", "ewma_form_", "season")
    if not files:
        return {"players": [], "built": False, "source": source}
    try:
        df = pl.read_parquet(files[-1])
        ewma_col = "ewma_xgf60" if "ewma_xgf60" in df.columns else "current_ewma"
        if ewma_col not in df.columns:
            return {"players": [], "built": False, "source": source}
        name_lut = _build_name_lookup()

        MIN_GP = 4 if source == "playoffs" else 20
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
        return {"players": rows, "built": True, "source": source}
    except Exception:
        return {"players": [], "built": False, "source": source}


@app.get("/phase2/cdr-leaderboard")
async def phase2_cdr_leaderboard(
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """Composite Defensive Rating leaderboard (higher = better defender)."""
    import polars as pl
    files = _context_parquets("cdr", "cdr_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("cdr", "cdr_", "season")
    if not files:
        return {"players": [], "built": False, "source": source}
    try:
        df = pl.read_parquet(files[-1])
        name_lut = _build_name_lookup()
        MIN_GP = 4 if source == "playoffs" else 20
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
        return {"players": rows, "built": True, "source": source}
    except Exception:
        return {"players": [], "built": False, "source": source}


@app.get("/phase2/rapm-leaderboard")
async def phase2_rapm_leaderboard(
    category: str = Query("ev_off", description="ev_off | ev_def | pp | pk"),
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """RAPM leaderboard by category (EV Off, EV Def, PP, PK)."""
    import polars as pl
    files = _context_parquets("rapm", "rapm_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("rapm", "rapm_", "season")
    if not files:
        return {"players": [], "built": False, "category": category}
    _COL = {"ev_off": "rapm_ev_off", "ev_def": "rapm_ev_def", "pp": "rapm_pp", "pk": "rapm_pk"}
    col = _COL.get(category, "rapm_ev_off")
    try:
        df = pl.read_parquet(files[-1])
        if col not in df.columns:
            return {"players": [], "built": False, "category": category, "reason": "column_missing"}
        name_lut = _build_name_lookup()
        # Playoffs cap at ~28 GP per team; drop the floor accordingly so the
        # leaderboard isn't empty in May/June.
        MIN_GP = 4 if source == "playoffs" else 20
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
        return {"players": rows, "built": True, "category": category, "source": source}
    except Exception:
        return {"players": [], "built": False, "category": category, "source": source}


@app.get("/phase2/xga-leaderboard")
async def phase2_xga_leaderboard(
    limit: int = 20,
    context: str = Query("season", description="season | playoffs"),
) -> dict:
    """xGA/60 leaderboard — best defenders by lowest on-ice xGA/60 (DZS-adjusted from CDR model)."""
    import polars as pl
    # Use CDR parquet — it has xga_60_adj (DZS-corrected on-ice xGA/60), which is
    # the real on-ice suppression number. RAPM xga_60 is a marginal ridge-regression
    # differential and is near zero by construction; not useful as a leaderboard.
    files = _context_parquets("cdr", "cdr_", context)
    source = "playoffs" if context == "playoffs" and files else "season"
    if not files:
        files = _context_parquets("cdr", "cdr_", "season")
    if not files:
        return {"players": [], "built": False, "source": source}
    try:
        df = pl.read_parquet(files[-1])
        col = "xga_60_adj" if "xga_60_adj" in df.columns else "xga_60"
        if col not in df.columns:
            return {"players": [], "built": False, "reason": "xga_60_missing", "source": source}
        name_lut = _build_name_lookup()
        MIN_GP = 4 if source == "playoffs" else 20
        if "gp" in df.columns:
            df = df.filter(pl.col("gp") >= MIN_GP)
        df = df.filter(pl.col(col).is_not_null()).filter(pl.col(col) > 0)
        # Lower xGA/60 = better defensive suppression; sort ascending
        ranked = df.sort(col, descending=False).head(limit)
        # Build RAPM EV Def lookup for context column (matching context).
        rapm_lut: dict[int, float | None] = {}
        rapm_files = _context_parquets("rapm", "rapm_", context) or _context_parquets("rapm", "rapm_", "season")
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
        return {"players": rows, "built": True, "source": source}
    except Exception as exc:
        return {"players": [], "built": False, "error": str(exc), "source": source}


# ---------------------------------------------------------------------------
# Phase 3 — Fatigue Engine
# ---------------------------------------------------------------------------

_PHASE3_FEATURES: list[dict] = [
    {"id": "schedule_density",    "ref": "3.1",  "subdir": "schedule_density",      "glob": "schedule_density_*.parquet",
     "name": "Schedule Density",            "desc": "B2B / 3-in-4 / rest-day flags per team-game",                          "feeds": "Composite FI (3.17)"},
    {"id": "road_trips",          "ref": "3.2",  "subdir": "road_trips",            "glob": "road_trips_*.parquet",
     "name": "Road Trip Tracker",           "desc": "Trip-id + game-within-trip annotation per team-game",                  "feeds": "Composite FI (3.17)"},
    {"id": "travel_distance",     "ref": "3.3",  "subdir": "travel_distance",       "glob": "travel_distance_*.parquet",
     "name": "Travel Distance",             "desc": "City-pair miles + rolling 7-day total per team-game",                  "feeds": "Composite FI (3.17)"},
    {"id": "time_zone_crossing",  "ref": "3.4",  "subdir": "time_zone_crossing",    "glob": "time_zone_crossing_*.parquet",
     "name": "Time-Zone Crossing",          "desc": "Zones crossed in 48h + east/west direction",                           "feeds": "Composite FI (3.17)"},
    {"id": "circadian_alignment", "ref": "3.5",  "subdir": "circadian_alignment",   "glob": "circadian_alignment_*.parquet",
     "name": "Circadian Alignment",         "desc": "Body-clock vs. game-start misalignment in hours",                      "feeds": "Composite FI (3.17)"},
    {"id": "altitude_adjustment", "ref": "3.6",  "subdir": "altitude_adjustment",   "glob": "altitude_adjustment_*.parquet",
     "name": "Altitude Penalty",            "desc": "Aerobic penalty for visitors at high-altitude arenas",                 "feeds": "Composite FI (3.17)"},
    {"id": "toi_load",            "ref": "3.7",  "subdir": "toi_load",              "glob": "toi_load_*.parquet",
     "name": "TOI Load",                    "desc": "Rolling 5-game TOI average + spike detector per player",               "feeds": "Composite FI (3.17)"},
    {"id": "special_teams_load",  "ref": "3.8",  "subdir": "special_teams_load",    "glob": "special_teams_load_*.parquet",
     "name": "Special-Teams Load",          "desc": "Rolling PP+PK minutes (higher intensity than 5v5)",                    "feeds": "Composite FI (3.17)"},
    {"id": "physical_contact_load","ref": "3.9", "subdir": "physical_contact_load", "glob": "physical_contact_load_*.parquet",
     "name": "Physical Contact Load",       "desc": "Rolling 5-game hits-taken + blocks load per skater",                   "feeds": "Composite FI (3.17)"},
    {"id": "overtime_fatigue",    "ref": "3.10", "subdir": "overtime_fatigue",      "glob": "overtime_fatigue_*.parquet",
     "name": "Overtime Fatigue",            "desc": "OT games in last 7 days + equivalent-TOI penalty",                     "feeds": "Composite FI (3.17)"},
    {"id": "fight_fatigue",       "ref": "3.11", "subdir": "fight_fatigue",         "glob": "fight_fatigue_*.parquet",
     "name": "Fight Fatigue",               "desc": "Rolling fighting-major adrenal-toll score",                            "feeds": "Composite FI (3.17)"},
    {"id": "age_recovery",        "ref": "3.12", "subdir": "age_recovery",          "glob": "age_recovery_*.parquet",
     "name": "Age Recovery Coefficient",    "desc": "Exponential recovery-rate decay function of age",                      "feeds": "Composite FI (3.17)"},
    {"id": "injury_status",       "ref": "3.13", "subdir": "injury_status",         "glob": "injury_status_*.parquet",
     "name": "Injury Status + Rust",        "desc": "DTD/Out flags + return-from-injury rust factor",                       "feeds": "Composite FI (3.17)"},
    {"id": "concussion_history",  "ref": "3.14", "subdir": "concussion_history",    "glob": "concussion_history_*.parquet",
     "name": "Concussion History",          "desc": "Elevated fatigue-sensitivity multiplier for prior concussions",        "feeds": "Composite FI (3.17)"},
    {"id": "prior_playoff_load",  "ref": "3.15", "subdir": "prior_playoff_load",    "glob": "prior_playoff_load_*.parquet",
     "name": "Prior Playoff Load",          "desc": "Deep-run summer-rest deficit → season-start penalty",                  "feeds": "Composite FI (3.17)"},
    {"id": "roster_depth_strain", "ref": "3.16", "subdir": "roster_depth_strain",   "glob": "roster_depth_strain_*.parquet",
     "name": "Roster Depth Strain",         "desc": "IR-driven minute redistribution onto remaining healthy skaters",       "feeds": "Composite FI (3.17)"},
    {"id": "composite_fi",        "ref": "3.17", "subdir": "composite_fi",          "glob": "composite_fi_*.parquet",
     "name": "Composite Fatigue Index",     "desc": "Weighted sum of all signals → FI 0.0–1.0 per (player, game)",          "feeds": "FI multiplier (3.18) · Rust engine"},
    {"id": "fi_rating_multiplier","ref": "3.18", "subdir": "fi_rating_multiplier",  "glob": "fi_rating_multiplier_*.parquet",
     "name": "FI → Rating Multiplier",      "desc": "Scaling factor on player ratings fed into Rust simulation",            "feeds": "Rust engine (Phase 5)"},
    {"id": "performance_anomaly", "ref": "3.19", "subdir": "performance_anomaly",   "glob": "performance_anomaly_*.parquet",
     "name": "Performance Anomaly Detector","desc": "Z-score + CUSUM SPC; flags hidden injuries (≥2σ below baseline)",      "feeds": "Dashboard alert feed"},
    {"id": "trade_integration",   "ref": "3.20", "subdir": "trade_integration",     "glob": "trade_integration_*.parquet",
     "name": "Trade Integration Model",     "desc": "Bidirectional fit-delta modifier; decays ~15–20 games post-trade",     "feeds": "Composite FI (3.17)"},
    {"id": "fi_edge_degradation", "ref": "3.21", "subdir": "fi_edge_degradation",   "glob": "fi_edge_degradation_*.parquet",
     "name": "FI → EDGE Degradation",       "desc": "Predicted drop in EDGE speed/distance/carry/burst vs. baseline",        "feeds": "Behavioral net (2.22)"},
    {"id": "seasonal_performance","ref": "3.22", "subdir": "seasonal_performance",  "glob": "seasonal_performance_*.parquet",
     "name": "Seasonal Performance Factor", "desc": "Month-of-season motivational modifier (Jan slump / Apr push)",         "feeds": "Composite FI (3.17)"},
]


def _filter_by_context(df, context: str):
    """Filter a per-(player, game) DataFrame by game_type.

    ``context``:
      - ``playoffs`` → game_type == 3
      - ``season``   → game_type == 2
      - ``auto``     → playoffs if any game_type=3 rows exist, else season
      - ``all`` / anything else → no filter

    Many Phase 3 parquets carry a ``game_type`` column that is null for every
    row (the column was added recently and back-fills haven't run). In that
    case the canonical NHL ``game_id`` still encodes the type at positions
    4–5 (``02`` = regular, ``03`` = playoffs), so we derive it on the fly
    rather than zapping every row.

    Silently no-op if neither ``game_type`` nor ``game_id`` exists, or if
    ``game_id`` carries no usable encoding (e.g. snapshot parquets where
    every game_id is 0).
    """
    import polars as pl
    if df is None or len(df) == 0:
        return df
    if context not in ("playoffs", "season", "auto"):
        return df

    has_gt  = "game_type" in df.columns
    has_gid = "game_id"   in df.columns

    if has_gt:
        non_null = df.filter(pl.col("game_type").is_not_null())
        if non_null.height > 0:
            if context == "playoffs":
                return df.filter(pl.col("game_type") == 3)
            if context == "season":
                return df.filter(pl.col("game_type") == 2)
            if context == "auto":
                has_playoffs = non_null.filter(pl.col("game_type") == 3).height > 0
                if has_playoffs:
                    return df.filter(pl.col("game_type") == 3)
                return df.filter(pl.col("game_type") == 2)

    if has_gid:
        gt_expr = pl.col("game_id").cast(pl.Utf8).str.slice(4, 2)
        usable  = df.filter(gt_expr.is_in(["02", "03"]))
        if usable.height > 0:
            if context == "playoffs":
                return df.filter(gt_expr == "03")
            if context == "season":
                return df.filter(gt_expr == "02")
            if context == "auto":
                has_playoffs = df.filter(gt_expr == "03").height > 0
                if has_playoffs:
                    return df.filter(gt_expr == "03")
                return df.filter(gt_expr == "02")

    return df


def _phase3_latest(subdir: str, glob: str) -> tuple[Path | None, datetime | None]:
    """Return (path, mtime) of the most-recently modified parquet, or (None, None)."""
    d = _GRETZKY_DATA_DIR / subdir
    if not d.exists():
        return None, None
    paths = sorted(d.glob(glob))
    if not paths:
        return None, None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return latest, datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)


@app.get("/phase3/modules")
async def phase3_modules() -> dict:
    """Per-feature build + data status for the Phase 3 fatigue engine.

    Each entry reports whether the model script has been run against the
    runtime data directory. ``status`` is ``ok`` when at least one
    parquet exists under ``<GRETZKY_DATA_DIR>/<subdir>/`` matching the
    feature's glob, ``not_run`` otherwise.
    """
    import polars as pl

    out: list[dict] = []
    for f in _PHASE3_FEATURES:
        path, mtime = _phase3_latest(f["subdir"], f["glob"])
        if path is None:
            out.append({
                "id":            f["id"],
                "ref":           f["ref"],
                "name":          f["name"],
                "desc":          f["desc"],
                "feeds":         f["feeds"],
                "status":        "not_run",
                "record_count":  0,
                "last_computed": None,
                "parquet":       None,
            })
            continue
        try:
            rc = pl.scan_parquet(path).select(pl.len()).collect().item()
        except Exception:
            rc = 0
        out.append({
            "id":            f["id"],
            "ref":           f["ref"],
            "name":          f["name"],
            "desc":          f["desc"],
            "feeds":         f["feeds"],
            "status":        "ok",
            "record_count":  int(rc),
            "last_computed": mtime.date().isoformat() if mtime else None,
            "parquet":       path.name,
        })

    built = sum(1 for m in out if m["status"] == "ok")
    return {"modules": out, "built": built, "total": len(out)}


@app.get("/phase3/fatigue/top")
async def phase3_fatigue_top(limit: int = 25, context: str = "auto") -> dict:
    """Top-N players by composite FI (3.17) from the latest parquet.

    ``context``:
      - ``playoffs`` → only rows where ``game_type == 3``
      - ``season`` → only rows where ``game_type == 2``
      - ``auto``    → playoffs if any playoff rows exist in the parquet,
                      else season (during NHL postseason this defaults to playoffs)
      - ``all``     → no filter
    """
    import polars as pl

    path, mtime = _phase3_latest("composite_fi", "composite_fi_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    df = _filter_by_context(df, context)

    names = _build_name_lookup()
    meta  = _build_player_meta()
    top = df.sort("fatigue_index", descending=True).head(int(max(1, limit)))
    rows: list[dict] = []
    for r in top.to_dicts():
        pid = int(r.get("player_id") or 0)
        m   = meta.get(pid, {})
        rows.append({
            "player_id":             pid,
            "player_name":           names.get(pid, m.get("name") or f"player_{pid}"),
            "team":                  m.get("team") or "",
            "position":              m.get("position") or "",
            "game_id":               int(r.get("game_id") or 0),
            "game_date":             r.get("game_date"),
            "fatigue_index":         r.get("fatigue_index"),
            "raw_load":              r.get("raw_load"),
            "rust_load":             r.get("rust_load"),
            "playoff_load_penalty":  r.get("playoff_load_penalty"),
            "component_breakdown":   r.get("component_breakdown"),
        })
    return {
        "status": "ok",
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase17/confidence/top")
async def phase17_confidence_top(limit: int = 25) -> dict:
    """Top-N most-confident players from the latest 4.24 parquet."""
    import polars as pl

    path, mtime = _phase3_latest("composite_confidence", "composite_confidence_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    names = _build_name_lookup()
    meta  = _build_player_meta()
    top = df.sort("confidence_index", descending=True).head(int(max(1, limit)))
    rows: list[dict] = []
    for r in top.to_dicts():
        pid = int(r.get("player_id") or 0)
        m   = meta.get(pid, {})
        rows.append({
            "player_id":           pid,
            "player_name":         names.get(pid, m.get("name") or f"player_{pid}"),
            "team":                m.get("team") or "",
            "position":            m.get("position") or "",
            "game_id":             int(r.get("game_id") or 0),
            "game_date":           r.get("game_date"),
            "confidence_index":    r.get("confidence_index"),
            "player_score":        r.get("player_score"),
            "team_score":          r.get("team_score"),
            "component_breakdown": r.get("component_breakdown"),
        })
    return {
        "status": "ok",
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase17/confidence/bottom")
async def phase17_confidence_bottom(limit: int = 25) -> dict:
    """Bottom-N (least-confident) players from the latest 4.24 parquet."""
    import polars as pl

    path, mtime = _phase3_latest("composite_confidence", "composite_confidence_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    names = _build_name_lookup()
    meta  = _build_player_meta()
    bot = df.sort("confidence_index", descending=False).head(int(max(1, limit)))
    rows: list[dict] = []
    for r in bot.to_dicts():
        pid = int(r.get("player_id") or 0)
        m   = meta.get(pid, {})
        rows.append({
            "player_id":           pid,
            "player_name":         names.get(pid, m.get("name") or f"player_{pid}"),
            "team":                m.get("team") or "",
            "position":            m.get("position") or "",
            "game_id":             int(r.get("game_id") or 0),
            "game_date":           r.get("game_date"),
            "confidence_index":    r.get("confidence_index"),
            "player_score":        r.get("player_score"),
            "team_score":          r.get("team_score"),
            "component_breakdown": r.get("component_breakdown"),
        })
    return {
        "status": "ok",
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase17/confidence/player/{player_id}")
async def phase17_confidence_player(player_id: int) -> dict:
    """Single-player confidence detail."""
    import polars as pl

    path, mtime = _phase3_latest("composite_confidence", "composite_confidence_*.parquet")
    if path is None:
        return {"status": "not_run", "row": None, "as_of": None}

    df = pl.read_parquet(path)
    sub = df.filter(pl.col("player_id") == int(player_id))
    if len(sub) == 0:
        return {"status": "no_data", "row": None,
                "as_of": mtime.date().isoformat() if mtime else None}
    latest = sub.sort("game_date", descending=True).head(1).to_dicts()[0]
    names = _build_name_lookup()
    meta  = _build_player_meta().get(int(player_id), {})
    return {
        "status": "ok",
        "row": {
            "player_id":           int(player_id),
            "player_name":         names.get(int(player_id), meta.get("name") or f"player_{player_id}"),
            "team":                meta.get("team") or "",
            "position":            meta.get("position") or "",
            "game_id":             int(latest.get("game_id") or 0),
            "game_date":           latest.get("game_date"),
            "confidence_index":    latest.get("confidence_index"),
            "player_score":        latest.get("player_score"),
            "team_score":          latest.get("team_score"),
            "component_breakdown": latest.get("component_breakdown"),
        },
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase17/confidence/team/{team_abbrev}")
async def phase17_confidence_team(team_abbrev: str) -> dict:
    """Team-confidence rollup: per-player rows for the team's roster."""
    import polars as pl

    path, mtime = _phase3_latest("composite_confidence", "composite_confidence_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [],
                "as_of": mtime.date().isoformat() if mtime else None}

    names = _build_name_lookup()
    meta  = _build_player_meta()
    abbrev = str(team_abbrev).upper()
    pids = [pid for pid, m in meta.items() if str(m.get("team", "")).upper() == abbrev]
    if not pids:
        return {"status": "no_roster", "rows": [],
                "as_of": mtime.date().isoformat() if mtime else None}

    sub = df.filter(pl.col("player_id").is_in([int(p) for p in pids]))
    if len(sub) == 0:
        return {"status": "no_data", "rows": [],
                "as_of": mtime.date().isoformat() if mtime else None}

    latest_per_player = (
        sub.sort("game_date", descending=True)
           .group_by("player_id").head(1)
    )
    rows: list[dict] = []
    team_scores: list[float] = []
    for r in latest_per_player.to_dicts():
        pid = int(r.get("player_id") or 0)
        m = meta.get(pid, {})
        ci = r.get("confidence_index")
        if ci is not None:
            team_scores.append(float(ci))
        rows.append({
            "player_id":        pid,
            "player_name":      names.get(pid, m.get("name") or f"player_{pid}"),
            "position":         m.get("position") or "",
            "confidence_index": ci,
            "player_score":     r.get("player_score"),
            "team_score":       r.get("team_score"),
            "game_date":        r.get("game_date"),
        })
    rows.sort(key=lambda x: (x["confidence_index"] or 0.0), reverse=True)
    return {
        "status": "ok",
        "team":   abbrev,
        "team_avg_confidence": (sum(team_scores) / len(team_scores)) if team_scores else 0.0,
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase3/goalie-fatigue/top")
async def phase3_goalie_fatigue_top(limit: int = 25) -> dict:
    """Top-N goalies by goalie_fi (3.24) from the latest snapshot parquet."""
    import polars as pl

    path, mtime = _phase3_latest("goalie_fatigue", "goalie_fi_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    names = _build_name_lookup()
    meta  = _build_player_meta()
    top = df.sort("goalie_fi", descending=True).head(int(max(1, limit)))
    rows: list[dict] = []
    for r in top.to_dicts():
        gid = int(r.get("goalie_id") or 0)
        m   = meta.get(gid, {})
        rows.append({
            "player_id":           gid,
            "player_name":         names.get(gid, m.get("name") or f"goalie_{gid}"),
            "team":                m.get("team") or "",
            "position":            m.get("position") or "G",
            "game_id":             int(r.get("game_id") or 0),
            "game_date":           r.get("game_date"),
            "goalie_fi":           r.get("goalie_fi"),
            "fatigue_sv_delta":    r.get("fatigue_sv_delta"),
            "is_b2b":              r.get("is_b2b"),
            "rest_days":           r.get("rest_days"),
            "gp_last_7":           r.get("gp_last_7"),
            "shots_faced_last_7":  r.get("shots_faced_last_7"),
            "component_breakdown": r.get("component_breakdown"),
        })
    return {
        "status": "ok",
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/goalies/{goalie_id}/fatigue")
async def goalie_fatigue_detail(goalie_id: int) -> dict:
    """Single-goalie fatigue snapshot from the latest 3.24 parquet."""
    import polars as pl

    path, mtime = _phase3_latest("goalie_fatigue", "goalie_fi_*.parquet")
    if path is None:
        return {"status": "not_run", "row": None, "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "row": None,
                "as_of": mtime.date().isoformat() if mtime else None}

    # Most recent row for this goalie (highest game_date)
    sub = df.filter(pl.col("goalie_id") == int(goalie_id))
    if len(sub) == 0:
        return {"status": "no_data", "row": None,
                "as_of": mtime.date().isoformat() if mtime else None}
    latest = sub.sort("game_date", descending=True).head(1).to_dicts()[0]

    names = _build_name_lookup()
    meta  = _build_player_meta().get(int(goalie_id), {})
    return {
        "status": "ok",
        "row": {
            "player_id":           int(goalie_id),
            "player_name":         names.get(int(goalie_id), meta.get("name") or f"goalie_{goalie_id}"),
            "team":                meta.get("team") or "",
            "position":            meta.get("position") or "G",
            "game_id":             int(latest.get("game_id") or 0),
            "game_date":           latest.get("game_date"),
            "goalie_fi":           latest.get("goalie_fi"),
            "fatigue_sv_delta":    latest.get("fatigue_sv_delta"),
            "is_b2b":              latest.get("is_b2b"),
            "rest_days":           latest.get("rest_days"),
            "gp_last_7":           latest.get("gp_last_7"),
            "shots_faced_last_7":  latest.get("shots_faced_last_7"),
            "road_game_num":       latest.get("road_game_num"),
            "component_breakdown": latest.get("component_breakdown"),
        },
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase3/anomalies")
async def phase3_anomalies(limit: int = 25) -> dict:
    """Active performance anomalies from the latest 3.19 parquet."""
    import polars as pl

    path, mtime = _phase3_latest("performance_anomaly", "performance_anomaly_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    flagged = df.filter(pl.col("is_anomaly") == True) if "is_anomaly" in df.columns else df
    if len(flagged) == 0:
        return {"status": "ok", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    top = flagged.sort("z_score").head(int(max(1, limit)))
    names = _build_name_lookup()
    meta  = _build_player_meta()
    rows: list[dict] = []
    for r in top.to_dicts():
        pid = int(r.get("player_id") or 0)
        m   = meta.get(pid, {})
        rows.append({
            "player_id":            pid,
            "player_name":          names.get(pid, m.get("name") or f"player_{pid}"),
            "team":                 m.get("team") or "",
            "position":             m.get("position") or "",
            "game_id":              int(r.get("game_id") or 0),
            "game_date":            r.get("game_date"),
            "z_score":              r.get("z_score"),
            "cusum":                r.get("cusum"),
            "consecutive_below_n":  r.get("consecutive_below_n"),
            "is_z_anomaly":         r.get("is_z_anomaly"),
            "is_cusum_anomaly":     r.get("is_cusum_anomaly"),
        })
    return {
        "status": "ok",
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase3/seasonal-distribution")
async def phase3_seasonal_distribution() -> dict:
    """Per-month average seasonal-motivation factor (3.22)."""
    import polars as pl

    path, mtime = _phase3_latest("seasonal_performance", "seasonal_performance_*.parquet")
    if path is None:
        return {"status": "not_run", "months": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0 or "month_of_season" not in df.columns:
        return {"status": "empty", "months": [], "as_of": mtime.date().isoformat() if mtime else None}

    grouped = (
        df.group_by("month_of_season")
          .agg([
              pl.col("seasonal_motivation_factor").mean().alias("mean_factor"),
              pl.len().alias("n"),
          ])
          .sort("month_of_season")
    )
    months = [
        {"month": int(r["month_of_season"]),
         "mean_factor": float(r["mean_factor"]) if r["mean_factor"] is not None else 0.0,
         "n": int(r["n"])}
        for r in grouped.to_dicts()
    ]
    return {
        "status": "ok",
        "months": months,
        "as_of":  mtime.date().isoformat() if mtime else None,
    }


@app.get("/phase3/trade-integration")
async def phase3_trade_integration(limit: int = 25) -> dict:
    """Active trade-integration modifiers (3.20), sorted by |factor| descending."""
    import polars as pl

    path, mtime = _phase3_latest("trade_integration", "trade_integration_*.parquet")
    if path is None:
        return {"status": "not_run", "rows": [], "as_of": None}

    df = pl.read_parquet(path)
    if len(df) == 0:
        return {"status": "empty", "rows": [], "as_of": mtime.date().isoformat() if mtime else None}

    sorted_df = (
        df.with_columns(pl.col("integration_factor").abs().alias("_abs"))
          .sort("_abs", descending=True)
          .drop("_abs")
          .head(int(max(1, limit)))
    )
    names = _build_name_lookup()
    meta  = _build_player_meta()
    rows: list[dict] = []
    for r in sorted_df.to_dicts():
        pid = int(r.get("player_id") or 0)
        m   = meta.get(pid, {})
        # Position from trade_integration parquet takes priority — captures
        # the position used at trade time; fall back to roster snapshot.
        pos = r.get("position") or m.get("position") or ""
        rows.append({
            "player_id":          pid,
            "player_name":        names.get(pid, m.get("name") or f"player_{pid}"),
            "team":               m.get("team") or "",
            "game_date":          r.get("game_date"),
            "trade_date":         r.get("trade_date"),
            "new_team":           r.get("new_team"),
            "old_team":           r.get("old_team"),
            "position":           pos,
            "games_since_trade":  r.get("games_since_trade"),
            "decay_factor":       r.get("decay_factor"),
            "fit_delta":          r.get("fit_delta"),
            "integration_factor": r.get("integration_factor"),
            "ci_low":             r.get("ci_low"),
            "ci_high":            r.get("ci_high"),
        })
    return {
        "status": "ok",
        "rows":   rows,
        "as_of":  mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase3/player")
async def phase3_player(
    name: str = Query(..., description="Player name"),
    context: str = Query("auto", description="playoffs | season | auto | all"),
) -> dict:
    """Per-player Phase 3 lookup — composite FI + components + anomaly + multiplier.

    Returns the most-recent (player, game) row for each Phase 3 feature
    that has been computed. Fields are ``null`` when a sub-model has not
    been run yet — the page surfaces that as "not run" rather than
    inventing a default.

    ``context`` controls playoff vs regular-season views for per-game
    parquets (composite_fi, goalie_fatigue, fi_multiplier). Confidence
    (Phase 17) is a snapshot and ignores this param.
    """
    import polars as pl
    import unicodedata, json as _json

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()

    needle = _norm(name)
    names  = _build_name_lookup()

    # Restrict the name → id resolution to player IDs that are actually
    # in the current composite_fi parquet. Two reasons:
    #   1. Old bayes_ratings parquets (2023) carry corrupted player_ids
    #      from an early pipeline bug (e.g. Matthews appeared as both
    #      220478230 in 2023 and 8479318 in 2025). Both keys end up in
    #      _build_name_lookup; matching by name picks whichever inserted
    #      first, which is the broken one. We never want to return an
    #      ID the dashboard can't pull stats for.
    #   2. Even valid IDs from older parquets may belong to retired
    #      players who don't appear in the current FI run.
    fi_path, _ = _phase3_latest("composite_fi", "composite_fi_*.parquet")
    valid_pids: set[int] = set()
    if fi_path is not None:
        try:
            valid_pids = set(
                pl.scan_parquet(fi_path)
                  .select("player_id").unique().collect()["player_id"].to_list()
            )
        except Exception:
            valid_pids = set()

    def _resolve(needle: str) -> tuple[int | None, str | None]:
        # Exact normalized match, scoped to FI'd players first.
        for k, v in names.items():
            if _norm(v) == needle and (not valid_pids or int(k) in valid_pids):
                return int(k), v
        # Same exact match, no scope (handles players present in older
        # parquets only — e.g. Phase 2-only data with no current FI row).
        for k, v in names.items():
            if _norm(v) == needle:
                return int(k), v
        # Prefix fallback (truncated names like "Slafkovsk"), scoped first.
        for k, v in names.items():
            nv = _norm(v)
            if (nv.startswith(needle) or needle.startswith(nv)) and \
               (not valid_pids or int(k) in valid_pids):
                return int(k), v
        for k, v in names.items():
            nv = _norm(v)
            if nv.startswith(needle) or needle.startswith(nv):
                return int(k), v
        return None, None

    pid, matched_name = _resolve(needle)

    if pid is None:
        return {
            "not_found":    True,
            "player_name":  name,
            "reason":       "player_id_not_resolved",
            "message":      "No Phase 1/2 parquet contains this player yet.",
        }

    # Enrich with current team + position so the dashboard can render a
    # logo + position pill in the search-result header.
    meta = _build_player_meta()
    pmeta = meta.get(pid, {}) if pid is not None else {}

    out: dict = {
        "not_found":    False,
        "player_id":    pid,
        "player_name":  matched_name or name,
        "team":         pmeta.get("team") or "",
        "position":     pmeta.get("position") or "",
    }

    # Composite FI (3.17) — newest row for this player.
    fi_path, fi_mtime = _phase3_latest("composite_fi", "composite_fi_*.parquet")
    fi_row: dict | None = None
    fi_context_fallback = False
    if fi_path is not None:
        try:
            fi_all = pl.read_parquet(fi_path).filter(pl.col("player_id") == pid)
            fi_df  = _filter_by_context(fi_all, context)
            if not fi_df.is_empty():
                fi_row = fi_df.sort("game_date", descending=True).head(1).to_dicts()[0]
            elif context == "playoffs" and not fi_all.is_empty():
                # No playoff fatigue computed yet (pipeline gap). Surface the
                # latest regular-season row so the card doesn't vanish, but
                # tag it so the dashboard can label it honestly.
                fi_row = fi_all.sort("game_date", descending=True).head(1).to_dicts()[0]
                fi_context_fallback = True
        except Exception:
            pass
    out["fi"] = {
        "status":                "ok" if fi_row else ("empty" if fi_path else "not_run"),
        "as_of":                 fi_mtime.date().isoformat() if fi_mtime else None,
        "context_fallback":      fi_context_fallback,
        "fatigue_index":         (fi_row or {}).get("fatigue_index"),
        "raw_load":              (fi_row or {}).get("raw_load"),
        "rust_load":             (fi_row or {}).get("rust_load"),
        "playoff_load_penalty":  (fi_row or {}).get("playoff_load_penalty"),
        "game_date":             (fi_row or {}).get("game_date"),
    }
    # Decode the component_breakdown JSON if present.
    cb_raw = (fi_row or {}).get("component_breakdown")
    components: dict | None = None
    if cb_raw:
        try:
            components = _json.loads(cb_raw) if isinstance(cb_raw, str) else dict(cb_raw)
        except Exception:
            components = None
    out["fi"]["component_breakdown"] = components

    # FI → Rating Multiplier (3.18)
    fm_path, fm_mtime = _phase3_latest("fi_rating_multiplier", "fi_rating_multiplier_*.parquet")
    fm_row: dict | None = None
    if fm_path is not None:
        try:
            fmdf = pl.read_parquet(fm_path).filter(pl.col("player_id") == pid)
            fmdf = _filter_by_context(fmdf, context)
            if not fmdf.is_empty():
                fm_row = fmdf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    out["fi_multiplier"] = {
        "status":      "ok" if fm_row else ("empty" if fm_path else "not_run"),
        "as_of":       fm_mtime.date().isoformat() if fm_mtime else None,
        "multiplier":  (fm_row or {}).get("rating_multiplier"),
        "game_date":   (fm_row or {}).get("game_date"),
    }

    # Performance anomaly (3.19)
    pa_path, pa_mtime = _phase3_latest("performance_anomaly", "performance_anomaly_*.parquet")
    pa_row: dict | None = None
    if pa_path is not None:
        try:
            padf = pl.read_parquet(pa_path).filter(pl.col("player_id") == pid)
            if not padf.is_empty():
                pa_row = padf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    out["anomaly"] = {
        "status":               "ok" if pa_row else ("empty" if pa_path else "not_run"),
        "as_of":                pa_mtime.date().isoformat() if pa_mtime else None,
        "z_score":              (pa_row or {}).get("z_score"),
        "cusum":                (pa_row or {}).get("cusum"),
        "consecutive_below_n":  (pa_row or {}).get("consecutive_below_n"),
        "is_anomaly":           (pa_row or {}).get("is_anomaly"),
        "is_on_ir":             (pa_row or {}).get("is_on_ir"),
        "game_date":            (pa_row or {}).get("game_date"),
    }

    # Trade integration (3.20)
    ti_path, ti_mtime = _phase3_latest("trade_integration", "trade_integration_*.parquet")
    ti_row: dict | None = None
    if ti_path is not None:
        try:
            tidf = pl.read_parquet(ti_path).filter(pl.col("player_id") == pid)
            if not tidf.is_empty():
                ti_row = tidf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    out["trade_integration"] = {
        "status":             "ok" if ti_row else ("empty" if ti_path else "not_run"),
        "as_of":              ti_mtime.date().isoformat() if ti_mtime else None,
        "integration_factor": (ti_row or {}).get("integration_factor"),
        "games_since_trade":  (ti_row or {}).get("games_since_trade"),
        "fit_delta":          (ti_row or {}).get("fit_delta"),
        "trade_date":         (ti_row or {}).get("trade_date"),
        "new_team":           (ti_row or {}).get("new_team"),
        "old_team":           (ti_row or {}).get("old_team"),
    }

    # FI → EDGE degradation (3.21)
    fe_path, fe_mtime = _phase3_latest("fi_edge_degradation", "fi_edge_degradation_*.parquet")
    fe_row: dict | None = None
    if fe_path is not None:
        try:
            fedf = pl.read_parquet(fe_path).filter(pl.col("player_id") == pid)
            if not fedf.is_empty():
                fe_row = fedf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    out["edge_degradation"] = {
        "status":                  "ok" if fe_row else ("empty" if fe_path else "not_run"),
        "as_of":                   fe_mtime.date().isoformat() if fe_mtime else None,
        "predicted_load_factor":   (fe_row or {}).get("predicted_load_factor"),
        "speed_vs_baseline":       (fe_row or {}).get("speed_vs_baseline"),
        "distance_vs_baseline":    (fe_row or {}).get("distance_vs_baseline"),
        "carry_vs_baseline":       (fe_row or {}).get("carry_vs_baseline"),
        "burst_vs_baseline":       (fe_row or {}).get("burst_vs_baseline"),
    }

    # Seasonal performance (3.22)
    sp_path, sp_mtime = _phase3_latest("seasonal_performance", "seasonal_performance_*.parquet")
    sp_row: dict | None = None
    if sp_path is not None:
        try:
            spdf = pl.read_parquet(sp_path).filter(pl.col("player_id") == pid)
            if not spdf.is_empty():
                sp_row = spdf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    out["seasonal"] = {
        "status":                      "ok" if sp_row else ("empty" if sp_path else "not_run"),
        "as_of":                       sp_mtime.date().isoformat() if sp_mtime else None,
        "seasonal_motivation_factor":  (sp_row or {}).get("seasonal_motivation_factor"),
        "month_of_season":             (sp_row or {}).get("month_of_season"),
        "base_month_effect":           (sp_row or {}).get("base_month_effect"),
    }

    # Confidence (Phase 17.24) — most-recent confidence_index row
    cf_path, cf_mtime = _phase3_latest("composite_confidence", "composite_confidence_*.parquet")
    cf_row: dict | None = None
    if cf_path is not None:
        try:
            cfdf = pl.read_parquet(cf_path).filter(pl.col("player_id") == pid)
            if not cfdf.is_empty():
                cf_row = cfdf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    cf_comps_raw = (cf_row or {}).get("component_breakdown")
    cf_comps: dict | None = None
    if cf_comps_raw:
        try:
            cf_comps = _json.loads(cf_comps_raw) if isinstance(cf_comps_raw, str) else dict(cf_comps_raw)
        except Exception:
            cf_comps = None
    out["confidence"] = {
        "status":              "ok" if cf_row else ("empty" if cf_path else "not_run"),
        "as_of":               cf_mtime.date().isoformat() if cf_mtime else None,
        "confidence_index":    (cf_row or {}).get("confidence_index"),
        "player_score":        (cf_row or {}).get("player_score"),
        "team_score":          (cf_row or {}).get("team_score"),
        "game_date":           (cf_row or {}).get("game_date"),
        "component_breakdown": cf_comps,
    }

    # Confidence rating multiplier (Phase 17.25)
    cm_path, cm_mtime = _phase3_latest("confidence_multiplier", "confidence_multiplier_*.parquet")
    cm_row: dict | None = None
    if cm_path is not None:
        try:
            cmdf = pl.read_parquet(cm_path).filter(pl.col("player_id") == pid)
            if not cmdf.is_empty():
                cm_row = cmdf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    out["confidence_multiplier"] = {
        "status":          "ok" if cm_row else ("empty" if cm_path else "not_run"),
        "as_of":           cm_mtime.date().isoformat() if cm_mtime else None,
        "shoot_bias":      (cm_row or {}).get("shoot_bias"),
        "risk_bias":       (cm_row or {}).get("risk_bias"),
        "turnover_bias":   (cm_row or {}).get("turnover_bias"),
        "game_date":       (cm_row or {}).get("game_date"),
    }

    # Goalie fatigue (3.24) — goalies only; empty for skaters
    gf_path, gf_mtime = _phase3_latest("goalie_fatigue", "goalie_fi_*.parquet")
    gf_row: dict | None = None
    if gf_path is not None:
        try:
            gfdf = pl.read_parquet(gf_path).filter(pl.col("goalie_id") == pid)
            gfdf = _filter_by_context(gfdf, context)
            if not gfdf.is_empty():
                gf_row = gfdf.sort("game_date", descending=True).head(1).to_dicts()[0]
        except Exception:
            pass
    gf_comps_raw = (gf_row or {}).get("component_breakdown")
    gf_comps: dict | None = None
    if gf_comps_raw:
        try:
            gf_comps = _json.loads(gf_comps_raw) if isinstance(gf_comps_raw, str) else dict(gf_comps_raw)
        except Exception:
            gf_comps = None
    out["goalie_fatigue"] = {
        "status":              "ok" if gf_row else ("empty" if gf_path else "not_run"),
        "as_of":               gf_mtime.date().isoformat() if gf_mtime else None,
        "goalie_fi":           (gf_row or {}).get("goalie_fi"),
        "fatigue_sv_delta":    (gf_row or {}).get("fatigue_sv_delta"),
        "is_b2b":              (gf_row or {}).get("is_b2b"),
        "rest_days":           (gf_row or {}).get("rest_days"),
        "gp_last_7":           (gf_row or {}).get("gp_last_7"),
        "shots_faced_last_7":  (gf_row or {}).get("shots_faced_last_7"),
        "road_game_num":       (gf_row or {}).get("road_game_num"),
        "game_date":           (gf_row or {}).get("game_date"),
        "component_breakdown": gf_comps,
    }

    return out


# ---------------------------------------------------------------------------
# Phase 4 — Coaching Tendency Models
# ---------------------------------------------------------------------------

_PHASE4_FEATURES: list[dict] = [
    {"id": "line_deployment",  "ref": "4.1", "subdir": "line_deployment", "glob": "line_deployment_*.parquet",
     "name": "Line Deployment Forecaster",   "desc": "Predicted F lines + D pairs + minutes allocation per coach", "feeds": "Lineup Forecaster (6.1) · Rust line-change (5.10)"},
    {"id": "line_matching",    "ref": "4.2", "subdir": "line_matching",   "glob": "line_matching_*.parquet",
     "name": "Line-Matching Model",          "desc": "Defensive counter-deployment vs. opponent top lines; last-change Δ", "feeds": "Rust line-change (5.10)"},
    {"id": "st_deployment",    "ref": "4.3", "subdir": "st_deployment",   "glob": "st_deployment_*.parquet",
     "name": "Special Teams Deployment",     "desc": "PP1/PP2 + PK1/PK2 personnel choices and first-unit share", "feeds": "Rust PP/PK sim (5.8, 5.9)"},
    {"id": "timeout_usage",    "ref": "4.4", "subdir": "timeout_usage",   "glob": "timeout_usage_*.parquet",
     "name": "Timeout Usage Model",          "desc": "P(timeout) by period × score state × time remaining per coach", "feeds": "Coach decision net (4.17)"},
    {"id": "goalie_pull",      "ref": "4.5", "subdir": "goalie_pull",     "glob": "goalie_pull_*.parquet",
     "name": "Goalie Pull Timing",           "desc": "P(pull goalie) at each (score deficit, time remaining) per coach", "feeds": "Coach decision net (4.17) · Rust pull sim (5.12)"},
    {"id": "penalty_tendency", "ref": "4.6", "subdir": "penalty_tendency","glob": "penalty_tendency_*.parquet",
     "name": "Penalty Tendency",             "desc": "Per-referee-crew foul call rate + PP opps/game baseline", "feeds": "Rust penalty model (5.7)"},
    {"id": "coach_profiles",   "ref": "4.7", "subdir": "coach_profiles",  "glob": "coach_profiles_*.parquet",
     "name": "Coach Profile Database",       "desc": "Career-with-current-team tendencies for every NHL head coach", "feeds": "Coach decision net (4.17) · /coaches/{name}"},
    {"id": "goalie_coach_curve","ref": "4.8","subdir": "goalie_coach_curve","glob": "goalie_coach_curve_*.parquet",
     "name": "Goalie Coach Model",           "desc": "Team save% trajectory + mid-season change-point detection", "feeds": "Goalie Bayesian update (2.10) · regime detector (2.14)"},
    {"id": "pp_coordinator",   "ref": "4.9", "subdir": "pp_coordinator",  "glob": "pp_coordinator_*.parquet",
     "name": "PP Coordinator Signature",     "desc": "PP system: xG/60, shot quality, carry%, PP1 QB usage", "feeds": "Rust PP simulator (5.8) · /coaches/{name}"},
    {"id": "pk_coordinator",   "ref": "4.10","subdir": "pk_coordinator",  "glob": "pk_coordinator_*.parquet",
     "name": "PK Coordinator Signature",     "desc": "PK system: SV%, xGA/60, SH forecheck pressure", "feeds": "Rust PK simulator (5.9) · /coaches/{name}"},
    {"id": "coaching_style",   "ref": "4.11","subdir": "coaching_style",  "glob": "coaching_style_*.parquet",
     "name": "Coaching Style Vector",        "desc": "8-dim system vector per team (forecheck, DZ, pace, …)", "feeds": "Roster fit (4.12) · Coach decision net (4.17)"},
    {"id": "roster_fit",       "ref": "4.12","subdir": "roster_fit",      "glob": "roster_fit_*.parquet",
     "name": "Roster Fit Score",             "desc": "Coach style × roster archetype composition (Roy / Dobson mismatch)", "feeds": "Team efficiency modifier in game setup (7.1)"},
    {"id": "staff_changes",    "ref": "4.13","subdir": "staff_changes",   "glob": "staff_changes_*.parquet",
     "name": "Staff Change Detector",        "desc": "Mid-season HC/coordinator/goalie-coach changes → regime pipeline", "feeds": "Regime change (2.14) · Bayesian update acceleration"},
    {"id": "fo_regime_changes","ref": "4.14","subdir": "fo_regime_changes","glob": "fo_regime_changes_*.parquet",
     "name": "FO Regime Change Detector",    "desc": "GM/AGM/Pres Hockey Ops changes → slow-decay regime pipeline", "feeds": "Regime change (2.14, 15.4) · team efficiency"},
    {"id": "buyer_seller",     "ref": "4.15","subdir": "buyer_seller",    "glob": "buyer_seller_*.parquet",
     "name": "Buyer/Seller Classifier",      "desc": "Per-team buyer|seller|neutral from standings + season progress", "feeds": "Lineup projector (6.1) · seller motivation (4.16) · roster disruption (2.19)"},
    {"id": "seller_motivation","ref": "4.16","subdir": "seller_motivation","glob": "seller_motivation_*.parquet",
     "name": "Seller Motivation State",      "desc": "Post-deadline efficiency drag for confirmed sellers", "feeds": "Game setup builder (7.1)"},
    {"id": "coach_decision_net","ref": "4.17","subdir": "coach_decision_net","glob": "coach_decision_net_*.parquet",
     "name": "Coach Decision Network",       "desc": "Unified per-coach decision profile from Phase 4 aggregates", "feeds": "Rust sim coach decisions (5.10, 5.12)"},
    {"id": "gm_fingerprint",   "ref": "4.18","subdir": "gm_fingerprint",  "glob": "gm_fingerprint_*.parquet",
     "name": "GM Behavioral Fingerprint",    "desc": "Per-GM action archetype + deadline aggression", "feeds": "Buyer/seller (4.15) · roster disruption (2.19)"},
    {"id": "venue_atmosphere", "ref": "4.19","subdir": "venue_atmosphere", "glob": "venue_atmosphere_*.parquet",
     "name": "Venue Atmosphere / Scare",     "desc": "Per-arena scare factor: visiting SV%/FOW%/PP/xGF deltas", "feeds": "Game setup builder (7.1)"},
    {"id": "playoff_elimination","ref":"4.20","subdir":"playoff_elimination","glob":"playoff_elimination_*.parquet",
     "name": "Playoff Elimination Fatigue",  "desc": "Team motivational regression when playoff_prob < 25%", "feeds": "Game setup builder (7.1) · stacks with FI + seller (4.16)"},
    {"id": "dow_signature",    "ref": "4.21","subdir": "dow_signature",    "glob": "dow_signature_*.parquet",
     "name": "DOW Signature + Broadcast",    "desc": "Per-player day-of-week z-score + rivalry/broadcast context", "feeds": "Rust engine shooting modifier · game setup (7.1)"},
]


def _phase4_latest(subdir: str, glob: str) -> tuple[Path | None, datetime | None]:
    """Return (path, mtime) of the most-recently modified Phase 4 parquet."""
    d = _GRETZKY_DATA_DIR / subdir
    if not d.exists():
        return None, None
    paths = sorted(d.glob(glob))
    if not paths:
        return None, None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return latest, datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)


@app.get("/phase4/modules")
async def phase4_modules() -> dict:
    """Per-feature build + data status for the Phase 4 coaching models."""
    import polars as pl

    out: list[dict] = []
    for f in _PHASE4_FEATURES:
        path, mtime = _phase4_latest(f["subdir"], f["glob"])
        if path is None:
            out.append({
                "id":            f["id"],
                "ref":           f["ref"],
                "name":          f["name"],
                "desc":          f["desc"],
                "feeds":         f["feeds"],
                "status":        "not_run",
                "record_count":  0,
                "last_computed": None,
                "parquet":       None,
            })
            continue
        try:
            rc = pl.scan_parquet(path).select(pl.len()).collect().item()
        except Exception:
            rc = 0
        out.append({
            "id":            f["id"],
            "ref":           f["ref"],
            "name":          f["name"],
            "desc":          f["desc"],
            "feeds":         f["feeds"],
            "status":        "ok",
            "record_count":  int(rc),
            "last_computed": mtime.date().isoformat() if mtime else None,
            "parquet":       path.name,
        })

    built = sum(1 for m in out if m["status"] == "ok")
    return {"modules": out, "built": built, "total": len(out)}


@app.get("/phase4/teams")
async def phase4_teams() -> dict:
    """Return the list of teams with line-deployment data, sorted alphabetically."""
    import polars as pl

    path, mtime = _phase4_latest("line_deployment", "line_deployment_*.parquet")
    if path is None:
        return {"status": "not_run", "teams": [], "as_of": None}
    df = pl.read_parquet(path)
    if df.is_empty():
        return {"status": "empty", "teams": [], "as_of": mtime.date().isoformat() if mtime else None}
    teams = sorted(df["team"].unique().drop_nulls().to_list())
    return {
        "status":  "ok",
        "teams":   teams,
        "as_of":   mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase4/deployment/{team}")
async def phase4_deployment(team: str) -> dict:
    """Return F lines + D pairs + projected minutes for a team."""
    import polars as pl

    path, mtime = _phase4_latest("line_deployment", "line_deployment_*.parquet")
    if path is None:
        return {"status": "not_run", "team": team, "lines": [], "as_of": None}

    df = pl.read_parquet(path).filter(pl.col("team") == team.upper())
    if df.is_empty():
        return {"status": "empty", "team": team, "lines": [], "as_of": mtime.date().isoformat() if mtime else None}

    names = _build_name_lookup()
    rows: list[dict] = []
    for r in df.to_dicts():
        pids: list[int] = []
        for k in ("player_1", "player_2", "player_3"):
            v = r.get(k)
            if v is not None:
                pids.append(int(v))
        rows.append({
            "line_type":              r.get("line_type"),
            "line_rank":              int(r.get("line_rank") or 0),
            "player_ids":             pids,
            "player_names":           [names.get(p, f"player_{p}") for p in pids],
            "chemistry_toi_secs":     r.get("chemistry_toi_secs"),
            "trio_toi_per_game":      r.get("trio_toi_per_game"),
            "line_toi_per_game":      r.get("line_toi_per_game"),
            "cohesion_pct":           r.get("cohesion_pct"),
            "share_of_team_toi":      r.get("share_of_team_toi"),
            "team_gp":                int(r.get("team_gp") or 0),
        })
    rows.sort(key=lambda r: (r["line_type"], r["line_rank"]))
    return {
        "status":  "ok",
        "team":    team.upper(),
        "lines":   rows,
        "as_of":   mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


@app.get("/phase4/matching/{team}")
async def phase4_matching(team: str, line_type: str = "F", venue: str = "all") -> dict:
    """Return matchup profile for a team — for each (own_line, opp_line, venue)
    the weighted share averaged across opponents.

    Args:
        team:      focal team abbrev (e.g. "MTL").
        line_type: "F" or "D".
        venue:     "home", "away", or "all" (combines).
    """
    import polars as pl

    path, mtime = _phase4_latest("line_matching", "line_matching_*.parquet")
    if path is None:
        return {"status": "not_run", "team": team, "rows": [], "as_of": None}

    df = pl.read_parquet(path).filter(
        (pl.col("team") == team.upper())
        & (pl.col("line_type") == line_type.upper())
    )
    if venue in ("home", "away"):
        df = df.filter(pl.col("venue") == venue)
    if df.is_empty():
        return {
            "status": "empty", "team": team, "rows": [],
            "as_of": mtime.date().isoformat() if mtime else None,
        }

    # Weighted-share aggregation across opponents.  Numerator = cell TOI,
    # denominator = total TOI when opp_rank was on the ice (any own_rank).
    # This guarantees that the four-row column at any (opp_rank, venue)
    # sums to 1.0.
    num = (
        df.group_by(["own_line_rank", "opp_line_rank", "venue"])
        .agg(pl.col("matchup_toi_secs").sum().alias("cell_toi"))
    )
    den = (
        df.group_by(["opp_line_rank", "venue"])
        .agg(pl.col("matchup_toi_secs").sum().alias("opp_rank_toi"))
    )
    grouped = (
        num.join(den, on=["opp_line_rank", "venue"], how="left")
        .with_columns(
            (pl.col("cell_toi") / pl.col("opp_rank_toi").clip(lower_bound=1e-9))
              .alias("weighted_share")
        )
        .rename({"cell_toi": "total_toi_secs"})
        .drop("opp_rank_toi")
        .sort(["venue", "opp_line_rank", "own_line_rank"])
    )
    rows = [
        {
            "own_line_rank":   int(r["own_line_rank"]),
            "opp_line_rank":   int(r["opp_line_rank"]),
            "venue":           r["venue"],
            "weighted_share":  float(r["weighted_share"]) if r["weighted_share"] is not None else 0.0,
            "total_toi_secs":  float(r["total_toi_secs"]) if r["total_toi_secs"] is not None else 0.0,
        }
        for r in grouped.to_dicts()
    ]
    return {
        "status":    "ok",
        "team":      team.upper(),
        "line_type": line_type.upper(),
        "venue":     venue,
        "rows":      rows,
        "as_of":     mtime.date().isoformat() if mtime else None,
        "parquet":   path.name,
    }


@app.get("/phase4/st-deployment/{team}")
async def phase4_st_deployment(team: str) -> dict:
    """Return PP1/PP2/PK1/PK2 units + first-unit share for a team."""
    import polars as pl

    path, mtime = _phase4_latest("st_deployment", "st_deployment_*.parquet")
    if path is None:
        return {"status": "not_run", "team": team, "units": [], "as_of": None}

    df = pl.read_parquet(path).filter(pl.col("team") == team.upper())
    if df.is_empty():
        return {
            "status": "empty", "team": team, "units": [],
            "as_of": mtime.date().isoformat() if mtime else None,
        }

    names = _build_name_lookup()
    units: list[dict] = []
    for r in df.to_dicts():
        pids = [int(p) for p in (r.get("personnel") or [])]
        units.append({
            "unit_type":        r.get("unit_type"),
            "player_ids":       pids,
            "player_names":     [names.get(p, f"player_{p}") for p in pids],
            "unit_toi_secs":    r.get("unit_toi_secs"),
            "share_of_st_toi":  r.get("share_of_st_toi"),
            "team_st_toi":      r.get("team_st_toi"),
            "team_st_gp":       int(r.get("team_st_gp") or 0),
        })

    # Canonical PP1, PP2, PK1, PK2 order
    order = {"PP1": 0, "PP2": 1, "PK1": 2, "PK2": 3}
    units.sort(key=lambda u: order.get(u["unit_type"] or "", 9))
    return {
        "status":  "ok",
        "team":    team.upper(),
        "units":   units,
        "as_of":   mtime.date().isoformat() if mtime else None,
        "parquet": path.name,
    }


# ---------------------------------------------------------------------------
# Coach profiles — joins Phase 4 outputs per team
# ---------------------------------------------------------------------------

_COACHES_PATH = Path(__file__).resolve().parents[2] / "data" / "coaches.json"


def _load_coaches() -> list[dict]:
    """Load the static head-coach roster.  Returns a list of dicts with
    at minimum keys ``team``, ``name``.  Returns ``[]`` if the file is
    missing or malformed (with stderr warning)."""
    if not _COACHES_PATH.exists():
        return []
    try:
        with open(_COACHES_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        coaches = blob.get("coaches", [])
        if isinstance(coaches, list):
            return coaches
    except Exception as e:
        print(f"[coaches] failed to load {_COACHES_PATH}: {e}", file=sys.stderr)
    return []


def _coach_slug(name: str) -> str:
    """URL-safe slug for a coach name.  Preserves apostrophes/accents in
    the slug because the frontend uses encodeURIComponent + this lookup
    is exact-match (case-insensitive).
    """
    return (name or "").strip()


@app.get("/coaches")
async def list_coaches() -> dict:
    """Return all known head coaches.  Powers Cortex search + the team
    page coach tile."""
    coaches = _load_coaches()
    out = [
        {
            "name":                   c.get("name") or "",
            "team":                   c.get("team") or "",
            "kind":                   "coach",
            "position":               "HC",  # head coach
            "first_named_head_coach": c.get("first_named_head_coach"),
            "notes":                  c.get("notes") or "",
            "image_url":              c.get("image_url"),
        }
        for c in coaches
        if c.get("name") and c.get("team")
    ]
    return {"coaches": out, "count": len(out)}


@app.get("/coaches/{name}")
async def coach_profile(name: str) -> dict:
    """Return a unified coach profile combining every Phase 4 signal we
    have for the coach's team.

    Output schema:
        - meta: {name, team, first_named_head_coach, notes}
        - line_deployment:  same shape as /phase4/deployment/{team}
        - line_matching_F:  team_matchup_profile rows (F)
        - line_matching_D:  team_matchup_profile rows (D)
        - st_deployment:    PP/PK units
        - goalie_pull:      per-deficit summary rows
        - penalty_tendency: per-team baseline row
        - timeout_usage:    aggregated per-team timeout buckets (often empty
                            until the ingester captures team timeouts)
    """
    import polars as pl

    coaches = _load_coaches()
    target = (name or "").strip().lower()
    match = next(
        (c for c in coaches if (c.get("name") or "").lower() == target),
        None,
    )
    if match is None:
        return {"status": "not_found", "name": name}

    team = (match.get("team") or "").upper()

    out: dict = {
        "status": "ok",
        "meta": {
            "name":                       match.get("name"),
            "team":                       team,
            "first_named_head_coach":     match.get("first_named_head_coach"),
            "notes":                      match.get("notes") or "",
            "image_url":                  match.get("image_url"),
        },
    }

    names_lut = _build_name_lookup()

    # Line deployment (4.1) — flatten same shape as /phase4/deployment/{team}
    ld_path, ld_mtime = _phase4_latest("line_deployment", "line_deployment_*.parquet")
    ld_rows: list[dict] = []
    if ld_path is not None:
        df = pl.read_parquet(ld_path).filter(pl.col("team") == team)
        for r in df.to_dicts():
            pids: list[int] = []
            for k in ("player_1", "player_2", "player_3"):
                v = r.get(k)
                if v is not None:
                    pids.append(int(v))
            ld_rows.append({
                "line_type":              r.get("line_type"),
                "line_rank":              int(r.get("line_rank") or 0),
                "player_ids":             pids,
                "player_names":           [names_lut.get(p, f"player_{p}") for p in pids],
                "chemistry_toi_secs":     r.get("chemistry_toi_secs"),
                "trio_toi_per_game":      r.get("trio_toi_per_game"),
                "line_toi_per_game":      r.get("line_toi_per_game"),
                "cohesion_pct":           r.get("cohesion_pct"),
                "share_of_team_toi":      r.get("share_of_team_toi"),
                "team_gp":                int(r.get("team_gp") or 0),
            })
        ld_rows.sort(key=lambda r: (r["line_type"], r["line_rank"]))
    out["line_deployment"] = {
        "rows":  ld_rows,
        "as_of": ld_mtime.date().isoformat() if ld_mtime else None,
    }

    # Line matching (4.2) — team profile across opponents
    lm_path, lm_mtime = _phase4_latest("line_matching", "line_matching_*.parquet")
    matching_F: list[dict] = []
    matching_D: list[dict] = []
    if lm_path is not None:
        from models.line_matching import team_matchup_profile
        lm_df = pl.read_parquet(lm_path)
        for line_type, bucket in (("F", matching_F), ("D", matching_D)):
            prof = team_matchup_profile(lm_df, team, line_type=line_type)
            for r in prof.to_dicts():
                bucket.append({
                    "own_line_rank":  int(r["own_line_rank"]),
                    "opp_line_rank":  int(r["opp_line_rank"]),
                    "venue":          r["venue"],
                    "weighted_share": float(r["weighted_share"] or 0.0),
                    "total_toi_secs": float(r["total_toi_secs"] or 0.0),
                })
    out["line_matching"] = {
        "F":     matching_F,
        "D":     matching_D,
        "as_of": lm_mtime.date().isoformat() if lm_mtime else None,
    }

    # Special teams (4.3)
    st_path, st_mtime = _phase4_latest("st_deployment", "st_deployment_*.parquet")
    st_units: list[dict] = []
    if st_path is not None:
        df = pl.read_parquet(st_path).filter(pl.col("team") == team)
        for r in df.to_dicts():
            pids = [int(p) for p in (r.get("personnel") or [])]
            st_units.append({
                "unit_type":        r.get("unit_type"),
                "player_ids":       pids,
                "player_names":     [names_lut.get(p, f"player_{p}") for p in pids],
                "unit_toi_secs":    r.get("unit_toi_secs"),
                "share_of_st_toi":  r.get("share_of_st_toi"),
                "team_st_toi":      r.get("team_st_toi"),
                "team_st_gp":       int(r.get("team_st_gp") or 0),
            })
        order = {"PP1": 0, "PP2": 1, "PK1": 2, "PK2": 3}
        st_units.sort(key=lambda u: order.get(u["unit_type"] or "", 9))
    out["st_deployment"] = {
        "units": st_units,
        "as_of": st_mtime.date().isoformat() if st_mtime else None,
    }

    # Goalie pull (4.5)
    gp_path, gp_mtime = _phase4_latest("goalie_pull", "goalie_pull_[0-9]*.parquet")
    pulls: list[dict] = []
    if gp_path is not None:
        df = pl.read_parquet(gp_path).filter(pl.col("team") == team)
        for r in df.to_dicts():
            pulls.append({
                "deficit":               int(r.get("deficit") or 0),
                "n_pulls":               int(r.get("n_pulls") or 0),
                "n_team_games":          int(r.get("n_team_games") or 0),
                "mean_pull_time_secs":   r.get("mean_pull_time_secs"),
                "median_pull_time_secs": r.get("median_pull_time_secs"),
                "earliest_pull_secs":    r.get("earliest_pull_secs"),
            })
        pulls.sort(key=lambda r: r["deficit"])
    out["goalie_pull"] = {
        "rows":  pulls,
        "as_of": gp_mtime.date().isoformat() if gp_mtime else None,
    }

    # Penalty tendency (4.6)
    pt_path, pt_mtime = _phase4_latest("penalty_tendency", "penalty_tendency_*.parquet")
    pt_row: dict | None = None
    league_avg: dict[str, float] = {}
    if pt_path is not None:
        df = pl.read_parquet(pt_path)
        if not df.is_empty():
            league_avg = {
                "penalties_taken_per_game": float(df["penalties_taken_per_game"].mean() or 0.0),
                "pp_opps_per_game":         float(df["pp_opps_per_game"].mean() or 0.0),
                "pim_per_game":             float(df["pim_per_game"].mean() or 0.0),
            }
        team_df = df.filter(pl.col("team") == team)
        if not team_df.is_empty():
            r = team_df.row(0, named=True)
            pt_row = {
                "n_games":                  int(r.get("n_games") or 0),
                "n_penalties_taken":        int(r.get("n_penalties_taken") or 0),
                "n_pp_opportunities":       int(r.get("n_pp_opportunities") or 0),
                "pim_total":                int(r.get("pim_total") or 0),
                "penalties_taken_per_game": float(r.get("penalties_taken_per_game") or 0.0),
                "pp_opps_per_game":         float(r.get("pp_opps_per_game") or 0.0),
                "pim_per_game":             float(r.get("pim_per_game") or 0.0),
                "ref_dim":                  r.get("ref_dim"),
            }
    out["penalty_tendency"] = {
        "row":        pt_row,
        "league_avg": league_avg,
        "as_of":      pt_mtime.date().isoformat() if pt_mtime else None,
    }

    # Timeout usage (4.4) — may be empty until ingester captures timeouts
    to_path, to_mtime = _phase4_latest("timeout_usage", "timeout_usage_*.parquet")
    timeouts: list[dict] = []
    if to_path is not None:
        df = pl.read_parquet(to_path).filter(pl.col("team") == team)
        for r in df.to_dicts():
            timeouts.append({
                "period_bucket": r.get("period_bucket"),
                "score_state":   r.get("score_state"),
                "time_bucket":   r.get("time_bucket"),
                "n_timeouts":    int(r.get("n_timeouts") or 0),
                "n_games":       int(r.get("n_games") or 0),
                "rate_per_game": float(r.get("rate_per_game") or 0.0),
            })
    out["timeout_usage"] = {
        "rows":  timeouts,
        "as_of": to_mtime.date().isoformat() if to_mtime else None,
    }

    # Coach profile (4.7) — match by coach_name
    cp_path, cp_mtime = _phase4_latest("coach_profiles", "coach_profiles_*.parquet")
    coach_profile_row: dict | None = None
    if cp_path is not None:
        df = pl.read_parquet(cp_path).filter(
            pl.col("coach_name").str.to_lowercase() == target
        )
        if not df.is_empty():
            r = df.row(0, named=True)
            coach_profile_row = {
                "coach_name":             r.get("coach_name"),
                "team":                   r.get("team"),
                "first_named_head_coach": r.get("first_named_head_coach"),
                "season":                 int(r.get("season") or 0),
                "seasons_covered":        [int(s) for s in (r.get("seasons_covered") or [])],
                "gp_under_coach":         int(r.get("gp_under_coach") or 0),
                "wins":                   int(r.get("wins") or 0),
                "ot_wins":                int(r.get("ot_wins") or 0),
                "losses":                 int(r.get("losses") or 0),
                "ot_losses":              int(r.get("ot_losses") or 0),
                "points":                 int(r.get("points") or 0),
                "points_pct":             float(r.get("points_pct") or 0.0),
                "gf_per_game":            float(r.get("gf_per_game") or 0.0),
                "ga_per_game":            float(r.get("ga_per_game") or 0.0),
                "pp_pct":                 float(r.get("pp_pct") or 0.0),
                "pk_pct":                 float(r.get("pk_pct") or 0.0),
                "sf_per_game":            float(r.get("sf_per_game") or 0.0),
                "sa_per_game":            float(r.get("sa_per_game") or 0.0),
            }
    out["coach_profile"] = {
        "row":   coach_profile_row,
        "as_of": cp_mtime.date().isoformat() if cp_mtime else None,
    }

    # Goalie coach curve (4.8) — match by team
    gc_path, gc_mtime = _phase4_latest("goalie_coach_curve", "goalie_coach_curve_*.parquet")
    goalie_coach_row: dict | None = None
    if gc_path is not None:
        df = pl.read_parquet(gc_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            def _nz(v):
                try:
                    f = float(v)
                    return f if f == f else None   # NaN → None
                except (TypeError, ValueError):
                    return None
            goalie_coach_row = {
                "team":                  r.get("team"),
                "season":                int(r.get("season") or 0),
                "gp":                    int(r.get("gp") or 0),
                "shots_against":         int(r.get("shots_against") or 0),
                "goals_against":         int(r.get("goals_against") or 0),
                "season_save_pct":       _nz(r.get("season_save_pct")),
                "prior_save_pct":        _nz(r.get("prior_save_pct")),
                "save_pct_delta":        _nz(r.get("save_pct_delta")),
                "early_split_save_pct":  _nz(r.get("early_split_save_pct")),
                "late_split_save_pct":   _nz(r.get("late_split_save_pct")),
                "split_delta":           _nz(r.get("split_delta")),
                "change_point_detected": bool(r.get("change_point_detected") or False),
                "rolling_save_pct":      [float(v) for v in (r.get("rolling_save_pct") or [])],
                "goalie_coach":          r.get("goalie_coach") or "",
            }
    out["goalie_coach"] = {
        "row":   goalie_coach_row,
        "as_of": gc_mtime.date().isoformat() if gc_mtime else None,
    }

    # PP Coordinator (4.9) — match by team
    pc_path, pc_mtime = _phase4_latest("pp_coordinator", "pp_coordinator_*.parquet")
    pp_coord_row: dict | None = None
    if pc_path is not None:
        df = pl.read_parquet(pc_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            def _nz(v):
                try:
                    f = float(v)
                    return f if f == f else None
                except (TypeError, ValueError):
                    return None
            qb_id = int(r.get("pp1_qb_id") or -1)
            pp_coord_row = {
                "team":                 r.get("team"),
                "season":               int(r.get("season") or 0),
                "pp_toi_secs":          float(r.get("pp_toi_secs") or 0.0),
                "pp_team_gp":           int(r.get("pp_team_gp") or 0),
                "pp_shots":             int(r.get("pp_shots") or 0),
                "pp_goals":             int(r.get("pp_goals") or 0),
                "pp_xg_total":          float(r.get("pp_xg_total") or 0.0),
                "pp_shots_per_60":      float(r.get("pp_shots_per_60") or 0.0),
                "pp_xg_per_60":         float(r.get("pp_xg_per_60") or 0.0),
                "pp_goals_per_60":      float(r.get("pp_goals_per_60") or 0.0),
                "pp_xg_per_shot":       float(r.get("pp_xg_per_shot") or 0.0),
                "pp_shot_distance_avg": float(r.get("pp_shot_distance_avg") or 0.0),
                "pp_carry_pct":         _nz(r.get("pp_carry_pct")),
                "pp1_qb_id":            qb_id if qb_id > 0 else None,
                "pp1_qb_name":          r.get("pp1_qb_name") or "",
                "pp1_qb_share":         float(r.get("pp1_qb_share") or 0.0),
                "pp_coordinator":       r.get("pp_coordinator") or "",
            }
    out["pp_coordinator"] = {
        "row":   pp_coord_row,
        "as_of": pc_mtime.date().isoformat() if pc_mtime else None,
    }

    # PK Coordinator (4.10) — match by team
    kc_path, kc_mtime = _phase4_latest("pk_coordinator", "pk_coordinator_*.parquet")
    pk_coord_row: dict | None = None
    if kc_path is not None:
        df = pl.read_parquet(kc_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            pk_coord_row = {
                "team":                  r.get("team"),
                "season":                int(r.get("season") or 0),
                "pk_toi_secs":           float(r.get("pk_toi_secs") or 0.0),
                "pk_team_gp":            int(r.get("pk_team_gp") or 0),
                "pk_sa":                 int(r.get("pk_sa") or 0),
                "pk_ga":                 int(r.get("pk_ga") or 0),
                "pk_xga_total":          float(r.get("pk_xga_total") or 0.0),
                "pk_sa_per_60":          float(r.get("pk_sa_per_60") or 0.0),
                "pk_xga_per_60":         float(r.get("pk_xga_per_60") or 0.0),
                "pk_ga_per_60":          float(r.get("pk_ga_per_60") or 0.0),
                "pk_save_pct":           float(r.get("pk_save_pct") or 0.0),
                "pk_xga_per_shot":       float(r.get("pk_xga_per_shot") or 0.0),
                "pk_shot_distance_avg":  float(r.get("pk_shot_distance_avg") or 0.0),
                "sh_shots_for":          int(r.get("sh_shots_for") or 0),
                "sh_goals_for":          int(r.get("sh_goals_for") or 0),
                "sh_shots_per_60":       float(r.get("sh_shots_per_60") or 0.0),
                "pk1_share":             float(r.get("pk1_share") or 0.0),
                "pk_coordinator":        r.get("pk_coordinator") or "",
            }
    out["pk_coordinator"] = {
        "row":   pk_coord_row,
        "as_of": kc_mtime.date().isoformat() if kc_mtime else None,
    }

    # Coaching Style Vector (4.11) — match by team
    cs_path, cs_mtime = _phase4_latest("coaching_style", "coaching_style_*.parquet")
    coaching_style_row: dict | None = None
    league_avg_style: dict | None = None
    if cs_path is not None:
        df_all = pl.read_parquet(cs_path)
        if not df_all.is_empty():
            # League average raw values for context (NaN → None so JSON stays valid).
            league_avg_style = {}
            for col in df_all.columns:
                if col.endswith("_raw") and df_all[col].dtype.is_float():
                    m = df_all[col].mean()
                    league_avg_style[col] = (
                        float(m) if m is not None and m == m else None
                    )
            df = df_all.filter(pl.col("team") == team)
            if not df.is_empty():
                r = df.row(0, named=True)
                def _nz(v):
                    try:
                        f = float(v)
                        return f if f == f else None
                    except (TypeError, ValueError):
                        return None
                dims = ["forecheck_aggression", "dz_structure", "pace", "physicality",
                        "oz_structure", "nz_tendency", "line_match", "st_aggression"]
                coaching_style_row = {
                    "team": r.get("team"),
                    "season": int(r.get("season") or 0),
                    "dimensions": {
                        d: {
                            "raw":  _nz(r.get(f"{d}_raw")),
                            "rank": _nz(r.get(f"{d}_rank")),
                        }
                        for d in dims
                    },
                }
    out["coaching_style"] = {
        "row":        coaching_style_row,
        "league_avg": league_avg_style,
        "as_of":      cs_mtime.date().isoformat() if cs_mtime else None,
    }

    # Roster Fit (4.12) — match by team
    rf_path, rf_mtime = _phase4_latest("roster_fit", "roster_fit_*.parquet")
    roster_fit_row: dict | None = None
    if rf_path is not None:
        df = pl.read_parquet(rf_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            archs   = list(r.get("archetypes") or [])
            shares  = [float(s) for s in (r.get("archetype_shares") or [])]
            roster_fit_row = {
                "team":              r.get("team"),
                "season":            int(r.get("season") or 0),
                "n_skaters":         int(r.get("n_skaters") or 0),
                "archetype_top":     r.get("archetype_top") or "",
                "archetypes":        archs,
                "archetype_shares":  shares,
                "fit_score":         float(r.get("fit_score") or 0.0),
                "mismatch_dim":      r.get("mismatch_dim") or "",
                "mismatch_support":  float(r.get("mismatch_support") or 0.0),
            }
    out["roster_fit"] = {
        "row":   roster_fit_row,
        "as_of": rf_mtime.date().isoformat() if rf_mtime else None,
    }

    # Staff changes (4.13) — all events for this team
    sc_path, sc_mtime = _phase4_latest("staff_changes", "staff_changes_*.parquet")
    staff_rows: list[dict] = []
    if sc_path is not None:
        df = pl.read_parquet(sc_path).filter(pl.col("team") == team)
        for r in df.to_dicts():
            staff_rows.append({
                "date":          r.get("date") or "",
                "change_type":   r.get("change_type") or "",
                "person_out":    r.get("person_out") or "",
                "person_in":     r.get("person_in") or "",
                "description":   r.get("description") or "",
                "decay_games":   int(r.get("decay_games") or 15),
            })
    out["staff_changes"] = {
        "rows":  staff_rows,
        "as_of": sc_mtime.date().isoformat() if sc_mtime else None,
    }

    # FO regime changes (4.14) — all events for this team
    fo_path, fo_mtime = _phase4_latest("fo_regime_changes", "fo_regime_changes_*.parquet")
    fo_rows: list[dict] = []
    if fo_path is not None:
        df = pl.read_parquet(fo_path).filter(pl.col("team") == team)
        for r in df.to_dicts():
            fo_rows.append({
                "date":          r.get("date") or "",
                "fo_role":       r.get("fo_role") or "",
                "person_out":    r.get("person_out") or "",
                "person_in":     r.get("person_in") or "",
                "description":   r.get("description") or "",
                "decay_games":   int(r.get("decay_games") or 50),
            })
    out["fo_regime_changes"] = {
        "rows":  fo_rows,
        "as_of": fo_mtime.date().isoformat() if fo_mtime else None,
    }

    # Buyer/seller (4.15) — this team's row
    bs_path, bs_mtime = _phase4_latest("buyer_seller", "buyer_seller_*.parquet")
    bs_row: dict | None = None
    if bs_path is not None:
        df = pl.read_parquet(bs_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            bs_row = {
                "team":            r.get("team"),
                "season":          int(r.get("season") or 0),
                "gp":              int(r.get("gp") or 0),
                "points_pct":      float(r.get("points_pct") or 0.0),
                "classification":  r.get("classification") or "neutral",
                "confidence":      float(r.get("confidence") or 0.0),
                "gap":             float(r.get("gap") or 0.0),
                "threshold":       float(r.get("threshold") or 0.0),
            }
    out["buyer_seller"] = {
        "row":   bs_row,
        "as_of": bs_mtime.date().isoformat() if bs_mtime else None,
    }

    # Seller motivation (4.16) — match by team
    sm_path, sm_mtime = _phase4_latest("seller_motivation", "seller_motivation_*.parquet")
    sm_row: dict | None = None
    if sm_path is not None:
        df = pl.read_parquet(sm_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            sm_row = {
                "team":                  r.get("team"),
                "seller_drag":           float(r.get("seller_drag") or 0.0),
                "efficiency_multiplier": float(r.get("efficiency_multiplier") or 1.0),
                "games_since_deadline":  int(r.get("games_since_deadline") or 0),
                "contextual_flag":       r.get("contextual_flag") or "",
            }
    out["seller_motivation"] = {
        "row":   sm_row,
        "as_of": sm_mtime.date().isoformat() if sm_mtime else None,
    }

    # Coach decision net (4.17) — match by coach name
    dn_path, dn_mtime = _phase4_latest("coach_decision_net", "coach_decision_net_*.parquet")
    dn_row: dict | None = None
    if dn_path is not None:
        df = pl.read_parquet(dn_path).filter(
            pl.col("coach_name").str.to_lowercase() == target
        )
        if not df.is_empty():
            r = df.row(0, named=True)
            dn_row = {
                "coach_name":          r.get("coach_name"),
                "team":                r.get("team"),
                "timeout_aggression":  float(r.get("timeout_aggression") or 0.5),
                "pull_aggression":     float(r.get("pull_aggression") or 0.5),
                "line_shelter_score":  float(r.get("line_shelter_score") or 0.5),
                "st_first_unit_lean":  float(r.get("st_first_unit_lean") or 0.5),
                "penalty_discipline":  float(r.get("penalty_discipline") or 0.5),
                "matching_intensity":  float(r.get("matching_intensity") or 0.5),
                "overall_aggression":  float(r.get("overall_aggression") or 0.5),
            }
    out["coach_decision_net"] = {
        "row":   dn_row,
        "as_of": dn_mtime.date().isoformat() if dn_mtime else None,
    }

    # GM fingerprint (4.18) — match by team
    gm_path, gm_mtime = _phase4_latest("gm_fingerprint", "gm_fingerprint_*.parquet")
    gm_row: dict | None = None
    if gm_path is not None:
        df = pl.read_parquet(gm_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            gm_row = {
                "team":                 r.get("team"),
                "gm_name":              r.get("gm_name") or "",
                "action_archetype":     r.get("action_archetype") or "",
                "prob_stand_pat":       float(r.get("prob_stand_pat") or 0.0),
                "prob_add_rental":      float(r.get("prob_add_rental") or 0.0),
                "prob_sell_veteran":    float(r.get("prob_sell_veteran") or 0.0),
                "prob_rebuild":         float(r.get("prob_rebuild") or 0.0),
                "prob_package_deal":    float(r.get("prob_package_deal") or 0.0),
                "deadline_aggression":  float(r.get("deadline_aggression") or 0.0),
                "recent_tx_count":      int(r.get("recent_tx_count") or 0),
            }
    out["gm_fingerprint"] = {
        "row":   gm_row,
        "as_of": gm_mtime.date().isoformat() if gm_mtime else None,
    }

    # Venue atmosphere (4.19) — match by team
    va_path, va_mtime = _phase4_latest("venue_atmosphere", "venue_atmosphere_*.parquet")
    va_row: dict | None = None
    if va_path is not None:
        df = pl.read_parquet(va_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            va_row = {
                "team":               r.get("team"),
                "home_gp":            int(r.get("home_gp") or 0),
                "visiting_sv_delta":  float(r.get("visiting_sv_delta") or 0.0),
                "visiting_fow_delta": float(r.get("visiting_fow_delta") or 0.0),
                "ref_pp_delta":       float(r.get("ref_pp_delta") or 0.0),
                "visiting_xgf_delta": float(r.get("visiting_xgf_delta") or 0.0),
                "crowd_intensity":    float(r.get("crowd_intensity") or 0.0),
                "scare_factor":       float(r.get("scare_factor") or 0.0),
                "scare_rank":         float(r.get("scare_rank") or 0.0),
            }
    out["venue_atmosphere"] = {
        "row":   va_row,
        "as_of": va_mtime.date().isoformat() if va_mtime else None,
    }

    # Playoff elimination (4.20) — match by team
    pe_path, pe_mtime = _phase4_latest("playoff_elimination", "playoff_elimination_*.parquet")
    pe_row: dict | None = None
    if pe_path is not None:
        df = pl.read_parquet(pe_path).filter(pl.col("team") == team)
        if not df.is_empty():
            r = df.row(0, named=True)
            pe_row = {
                "team":                  r.get("team"),
                "playoff_prob":          float(r.get("playoff_prob") or 0.0),
                "elimination_drag":      float(r.get("elimination_drag") or 0.0),
                "efficiency_multiplier": float(r.get("efficiency_multiplier") or 1.0),
                "games_remaining":       int(r.get("games_remaining") or 0),
                "points_pct":            float(r.get("points_pct") or 0.0),
            }
    out["playoff_elimination"] = {
        "row":   pe_row,
        "as_of": pe_mtime.date().isoformat() if pe_mtime else None,
    }

    return out


# ---------------------------------------------------------------------------
# Chart endpoints — feed the Phase 1/2/3 dashboard visualizations
# ---------------------------------------------------------------------------

@app.get("/phase3/fi-histogram")
async def phase3_fi_histogram(bins: int = 20) -> dict:
    """Distribution of fatigue_index across every (player, game) row.

    Returns ``bins`` evenly-spaced buckets over [0, 1] with a count per
    bucket — feeds the dashboard histogram. Pulls a single column out
    of the latest composite_fi parquet (cheap even at 28k rows).
    """
    import polars as pl

    path, mtime = _phase3_latest("composite_fi", "composite_fi_*.parquet")
    if path is None:
        return {"status": "not_run", "bins": [], "as_of": None, "total": 0}
    df = pl.read_parquet(path, columns=["fatigue_index"])
    if len(df) == 0:
        return {"status": "empty", "bins": [], "as_of": mtime.date().isoformat(), "total": 0}

    n_bins = max(4, min(40, int(bins)))
    width = 1.0 / n_bins
    vals  = df["fatigue_index"].drop_nulls().to_list()
    counts = [0] * n_bins
    for v in vals:
        idx = min(n_bins - 1, max(0, int(v / width)))
        counts[idx] += 1
    buckets = [
        {
            "bin_lo": round(i * width, 3),
            "bin_hi": round((i + 1) * width, 3),
            "count":  counts[i],
        }
        for i in range(n_bins)
    ]
    return {
        "status": "ok",
        "bins":   buckets,
        "total":  len(vals),
        "mean":   float(sum(vals) / len(vals)) if vals else 0.0,
        "max":    float(max(vals)) if vals else 0.0,
        "as_of":  mtime.date().isoformat() if mtime else None,
    }


@app.get("/phase3/team-fatigue")
async def phase3_team_fatigue(window_days: int = 21) -> dict:
    """Recent-game fatigue per team — defaults to the last 21 days of
    composite_fi data so teams that stopped playing weeks ago don't
    skew the league ranking.

    Each team's row is computed as the mean fatigue_index of its
    player-games in ``[latest_fi_date - window_days, latest_fi_date]``.
    Teams with zero games in that window are dropped. The window cutoff
    is included in the response so the dashboard can label it.
    """
    import polars as pl

    fi_path, mtime = _phase3_latest("composite_fi", "composite_fi_*.parquet")
    if fi_path is None:
        return {"status": "not_run", "teams": [], "as_of": None}
    fi_df = pl.read_parquet(fi_path, columns=["player_id", "game_date", "fatigue_index"])
    if len(fi_df) == 0:
        return {"status": "empty", "teams": [], "as_of": mtime.date().isoformat()}

    # Build player_id → team_code from cached roster JSON
    import json as _json
    raw_dir = _GRETZKY_DATA_DIR / "raw"
    pid_team: dict[int, str] = {}
    if raw_dir.exists():
        for f in sorted(raw_dir.glob("roster_*.json")):
            try:
                data = _json.loads(f.read_text())
                team = data.get("team_code", "")
                for p in data.get("profiles", []):
                    pid = p.get("player_id")
                    if pid:
                        pid_team[int(pid)] = team
            except Exception:
                pass
    if not pid_team:
        return {"status": "no_roster", "teams": [], "as_of": mtime.date().isoformat()}

    # Anchor the rolling window on the most recent game_date *in the
    # parquet itself*, not today. Today might be off-season (May), when
    # no regular-season games are happening at all.
    latest_in_data = fi_df["game_date"].max()
    window = max(7, min(60, int(window_days)))
    cutoff = (datetime.strptime(latest_in_data, "%Y-%m-%d")
              - timedelta(days=window)).date().isoformat()

    recent = fi_df.filter(pl.col("game_date") >= cutoff)
    if len(recent) == 0:
        return {
            "status": "empty",
            "teams": [],
            "as_of": mtime.date().isoformat() if mtime else None,
            "window_start": cutoff,
            "window_end":   latest_in_data,
        }

    team_df = pl.DataFrame(
        [{"player_id": k, "team": v} for k, v in pid_team.items()],
        schema={"player_id": pl.Int64, "team": pl.Utf8},
    )
    agg = (
        recent.join(team_df, on="player_id", how="inner")
              .group_by("team")
              .agg([
                  pl.col("fatigue_index").mean().alias("mean_fi"),
                  pl.col("fatigue_index").max().alias("max_fi"),
                  pl.col("game_date").max().alias("last_game"),
                  pl.len().alias("rows"),
              ])
              .sort("mean_fi", descending=True)
    )
    return {
        "status": "ok",
        "teams":  [
            {
                "team":      r["team"],
                "mean_fi":   float(r["mean_fi"] or 0.0),
                "max_fi":    float(r["max_fi"]  or 0.0),
                "last_game": r["last_game"],
                "rows":      int(r["rows"]),
            }
            for r in agg.to_dicts()
        ],
        "as_of":        mtime.date().isoformat() if mtime else None,
        "window_start": cutoff,
        "window_end":   latest_in_data,
    }


@app.get("/phase2/war-distribution")
async def phase2_war_distribution(bins: int = 18) -> dict:
    """Histogram of WAR values + summary stats.

    Reads the latest war/war_*.parquet (Feature 2.25) and bins WAR into
    ``bins`` buckets between the min and max in the data. Returns the
    top-10 ranked WAR rows alongside so the chart card can render a
    leaderboard tooltip.
    """
    import polars as pl

    war_dir = _GRETZKY_DATA_DIR / "war"
    if not war_dir.exists():
        return {"status": "not_run", "bins": [], "top": [], "as_of": None}
    parquets = sorted(war_dir.glob("war_*.parquet"))
    if not parquets:
        return {"status": "not_run", "bins": [], "top": [], "as_of": None}

    path = parquets[-1]
    df = pl.read_parquet(path)
    if "war" not in df.columns or len(df) == 0:
        return {"status": "empty", "bins": [], "top": [], "as_of": None}

    vals = df["war"].drop_nulls().to_list()
    if not vals:
        return {"status": "empty", "bins": [], "top": [], "as_of": None}

    lo = float(min(vals))
    hi = float(max(vals))
    if hi <= lo:
        hi = lo + 1e-6
    n_bins = max(4, min(40, int(bins)))
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in vals:
        idx = min(n_bins - 1, max(0, int((v - lo) / width)))
        counts[idx] += 1
    buckets = [
        {
            "bin_lo": round(lo + i * width, 3),
            "bin_hi": round(lo + (i + 1) * width, 3),
            "count":  counts[i],
        }
        for i in range(n_bins)
    ]

    # Top-10 by WAR
    cols = [c for c in ("player_name", "team", "position", "war") if c in df.columns]
    top10 = (
        df.select(cols).sort("war", descending=True).head(10).to_dicts()
        if "war" in df.columns else []
    )

    import datetime
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "status": "ok",
        "bins":   buckets,
        "top":    top10,
        "mean":   float(sum(vals) / len(vals)),
        "max":    hi,
        "min":    lo,
        "total":  len(vals),
        "as_of":  mtime.date().isoformat(),
    }


@app.get("/phase1/freshness")
async def phase1_freshness() -> dict:
    """Per-module sync freshness — last_synced_at + records for each Phase 1 module.

    Reuses /phase1/modules' shape but adds an ``age_hours`` field that
    the frontend renders as a color-coded chip (green < 12h, amber < 36h,
    red ≥ 36h). Used by the Phase 1 page's data-pipeline freshness chart.
    """
    # Delegate to the existing module status function
    mods_resp = await phase1_modules()
    modules = mods_resp.get("modules", [])
    now = datetime.now(timezone.utc)
    out = []
    for m in modules:
        synced = m.get("synced_at")
        age_hours: float | None = None
        if synced:
            try:
                # Accept both Z-suffixed and offset-naive ISO timestamps
                s = synced.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_hours = round((now - dt).total_seconds() / 3600.0, 2)
            except Exception:
                age_hours = None
        out.append({
            "name":         m.get("name"),
            "status":       m.get("status"),
            "record_count": m.get("record_count", 0),
            "synced_at":    synced,
            "age_hours":    age_hours,
        })
    return {"modules": out, "now": now.isoformat()}


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

    # Tag everything we already have as "skater" or "goalie".  We mix
    # coaches into the same search list so Cortex autocomplete surfaces
    # all three kinds in one dropdown.
    for entry in players.values():
        entry.setdefault("kind", "goalie" if entry.get("position") == "G" else "skater")

    # Layer in head coaches (sourced from data/coaches.json).
    for coach in _load_coaches():
        cname = (coach.get("name") or "").strip()
        if not cname:
            continue
        # Don't overwrite a player who happens to share a name; coach
        # nodes are otherwise unique.
        if cname not in players:
            players[cname] = {
                "name":      cname,
                "team":      (coach.get("team") or "").upper(),
                "position":  "HC",       # head coach
                "player_id": None,
                "kind":      "coach",
            }

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

# Recent-success tracker. Frontend POSTs /stream-success?url=<chip_url> the
# moment a stream actually starts playing (HLS manifest parsed or iframe
# wrapper heartbeat received). We rank chips within their priority bucket
# by how many successes they've logged in the last hour, so consistently
# dead backup scrapers sink to the bottom.
import collections as _collections
_stream_success: dict[str, _collections.deque] = _collections.defaultdict(
    lambda: _collections.deque(maxlen=200)
)
_STREAM_SUCCESS_WINDOW_S = 3600


def _record_stream_success(url: str) -> None:
    import time as _t
    _stream_success[url].append(_t.monotonic())


def _stream_recent_success_count(url: str) -> int:
    import time as _t
    cutoff = _t.monotonic() - _STREAM_SUCCESS_WINDOW_S
    dq = _stream_success.get(url)
    if not dq:
        return 0
    while dq and dq[0] < cutoff:
        dq.popleft()
    return len(dq)


@app.get("/streams/{game_id}")
async def game_streams(game_id: int) -> dict:
    """Scrape live stream embed links from onhockey.tv for a given NHL game.

    Disabled for now — onhockey.tv chips were adding noise to the game-page
    stream selector without contributing reliable playback. IPTV (upstream)
    chips come from a separate endpoint and are unaffected. Re-enable by
    removing this short-circuit.
    """
    return {"streams": [], "game_id": game_id, "away": "", "home": "", "disabled": True}
    import time as _t  # noqa: F401  (dead code below; kept for easy revert)
    cached = _streams_cache.get(game_id)
    if cached and (_t.monotonic() - cached[1]) < _STREAMS_CACHE_TTL:
        streams = list(cached[0])
        # Re-sort on every request so success-rate updates from /stream-success
        # bubble newly-validated chips up without waiting for cache expiry.
        streams.sort(key=lambda s: (s.get("priority", 99), -_stream_recent_success_count(s.get("url", ""))))
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
                # Direct .m3u8/.mpd → priority 0, plays via hls.js/stream-proxy.
                # Anything else → embed_only=true → frontend goes straight to
                # IframePlayer (no /stream-resolve round-trip, no Playwright
                # spin). Per Bob: chips were "just loading" on desktop while
                # Playwright tried to extract for 30s and almost always
                # failed; iframe is instant even when it dies (user clicks
                # "try next"). Mobile sandboxes the iframe to block popup ads.
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

    # Annotate priority first so the cache stores it; sort by (priority asc,
    # recent-success desc) so dead backup scrapers sink within their bucket.
    for s in streams:
        s["priority"] = _stream_priority(s)
        s.pop("_direct", None)

    # Pre-flight the priority-3 backup chips. The scrapers happily list
    # template URLs (wikisport.club placeholders, /sorry redirects, geo-blocked
    # cdnlivetv proxies) that 200 with empty player containers — the wrapper
    # iframe heartbeat keeps firing while the user stares at a broken-image
    # icon. We probe each in parallel, drop the dead ones, and only then sort.
    # Cached 10 min so a hot game doesn't re-probe on every browser refresh.
    streams = await _filter_dead_embeds(streams)

    streams.sort(key=lambda s: (s["priority"], -_stream_recent_success_count(s["url"])))

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
    # NHL API sends "truTV" (camelCase) and "HBO MAX" (spaced) during playoffs —
    # both carry the TNT simulcast of Round 1 games. Keys are looked up case-
    # insensitive below, so casing here is cosmetic; the value is what we match
    # IPTV channel titles against.
    "TRUTV":   "truTV",
    "MAX":     "Max",
    "HBO MAX": "Max",
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


_GAME_IPTV_REFRESH_TASKS: dict[int, "asyncio.Task"] = {}


def _kick_game_iptv_refresh(game_id: int) -> None:
    """Spawn one background rebuild of _game_iptv_cache[game_id]; no-op if
    one is already in flight for this game. Used by the stale-while-revalidate
    path so users never wait for the NHL landing fetch + IPTV fan-out once
    the cache has any data for this game."""
    existing = _GAME_IPTV_REFRESH_TASKS.get(game_id)
    if existing is not None and not existing.done():
        return
    try:
        async def _refresh() -> None:
            try:
                fresh = await _build_game_iptv(game_id)
                if fresh is not None:
                    import time as _t_refresh
                    _game_iptv_cache[game_id] = (fresh, _t_refresh.monotonic())
            except Exception:
                pass
            finally:
                _GAME_IPTV_REFRESH_TASKS.pop(game_id, None)

        _GAME_IPTV_REFRESH_TASKS[game_id] = asyncio.create_task(_refresh())
    except RuntimeError:
        pass


@app.get("/game-iptv-streams/{game_id}")
async def game_iptv_streams(game_id: int) -> dict:
    """Return IPTV channels matched to this game's actual broadcast networks."""
    import time as _t_iptv
    cached = _game_iptv_cache.get(game_id)
    if cached:
        age = _t_iptv.monotonic() - cached[1]
        # Cache TTL is 1 hr; relay ffmpeg idle timeout is 30 s, so we always
        # re-warm the top relay /hls chips on every request, not just the
        # cache miss. Cheap because _PROXY_INFLIGHT dedups collisions.
        _kick_chip_warmups(cached[0])
        if age < _GAME_IPTV_TTL:
            return {"broadcasts": cached[0], "cached": True}
        # Stale-while-revalidate: serve last-good slate immediately, refresh
        # in the background. Hides the NHL landing fetch + IPTV fan-out
        # latency from every cache turnover.
        _kick_game_iptv_refresh(game_id)
        return {"broadcasts": cached[0], "cached": True, "stale": True}

    # First request for this game (or first after process start): pay the cost.
    result = await _build_game_iptv(game_id)
    if result is None:
        return {"broadcasts": []}
    _game_iptv_cache[game_id] = (result, _t_iptv.monotonic())
    _kick_chip_warmups(result)
    return {"broadcasts": result}


async def _build_game_iptv(game_id: int) -> list[dict] | None:
    """Build the broadcast slate for a game. Returns the slate list, or None
    on hard failure (the caller skips caching so subsequent requests retry).

    Runs the NHL landing fetch and the IPTV channel pull in parallel via
    asyncio.gather — they're independent, so doing them sequentially used to
    add the smaller of the two latencies on every cold start."""
    from data.nhl_client import NHLClient

    try:
        async with NHLClient() as client:
            landing, iptv_result = await asyncio.gather(
                client.get_landing(game_id),
                iptv_channels(),
                return_exceptions=True,
            )
    except Exception:
        return None

    if isinstance(landing, Exception) or not landing:
        return None

    tv_broadcasts: list[dict] = landing.get("tvBroadcasts") or []
    if not tv_broadcasts:
        return []

    # Playoffs (gameType == 3): national networks have exclusive rights. NHL
    # API still lists the home/away RSN rows (NESN, MSG, FDSN*, Victory+, etc.)
    # because those networks hold historical rights, but they don't actually
    # air the game during playoffs — they run other programming (wrestling,
    # baseball, paid programming). Rendering those as chips sent users to
    # non-hockey content. Keep only national-market (N) broadcasts.
    game_type = int(landing.get("gameType") or 0)
    is_playoffs = game_type == 3
    if is_playoffs:
        tv_broadcasts = [b for b in tv_broadcasts if b.get("market") == "N"]

    home_abbrev = (landing.get("homeTeam") or {}).get("abbrev", "") or ""
    away_abbrev = (landing.get("awayTeam") or {}).get("abbrev", "") or ""

    if isinstance(iptv_result, dict):
        all_channels: list[dict] = iptv_result.get("channels", []) or []
    else:
        all_channels = []

    # Match each broadcast network to IPTV channels by title tokens.
    # Approved sources: thetvapp redirects (resolve to v4.thetvapp.to tokens),
    # and upstream accounts (ampztl + an upstream host + bgdc + an upstream host + kstv).
    # tvpass.org dropped 2026-05-05 — both the static slugs and any
    # GitHub-mirrored tvpass entries are filtered out via _drop_tvpass below.
    # The previous comment noted "tvpass static slugs" as approved; we now
    # rely on thetvapp + upstream alone.
    curated_channels = [ch for ch in all_channels if ch.get("source") in ("thetvapp", "upstream")]
    curated_channels = [ch for ch in curated_channels if not _is_tvpass(ch)]

    # ampztl publishes several feeds per channel marked with different glyphs.
    # Field-tested: `ƒ` and `≋` feeds don't play; `✪` plays fine (often the
    # only working variant). Keep ✪/✤/☆ in the chip list; drop the known-dead
    # markers so the game page never offers a chip that can't play.
    _AMPZTL_DEAD_MARKERS = ("ƒ", "≋")
    curated_channels = [
        ch for ch in curated_channels
        if not any(m in (ch.get("title", "") or "") for m in _AMPZTL_DEAD_MARKERS)
    ]

    # 2026-05-03: an upstream host's TVA Sports / TVA Sports 2 slots auth-redirect
    # successfully but serve 0 bytes (dead upstream channel). Drop those two
    # specific titles so users don't see a chip that returns "manifest not
    # yet available" after 30s. Other an upstream host channels are unaffected.
    def _is_dead_an upstream host_tva(ch: dict) -> bool:
        title = (ch.get("title", "") or "").lower()
        if "(an upstream host)" not in title:
            return False
        return ("tva sports (an upstream host)" in title) or ("tva sports 2 (an upstream host)" in title)
    curated_channels = [ch for ch in curated_channels if not _is_dead_an upstream host_tva(ch)]

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
    _SHORT_FR_DISPLAYS = {"rds", "rds2", "rds info", "tva sports"}

    def _matches_broadcast(ch_title: str, display: str) -> bool:
        norm = _normalize_ch(ch_title)
        want = display.lower().strip()
        if norm == want:
            return True
        # ESPN+ alternation — providers commonly list it as "espnplus"/"espn plus"
        if want == "espn+" and norm in ("espnplus", "espn plus"):
            return True
        if want in _SHORT_FR_DISPLAYS:
            # French titles often have un-stripped prefixes like
            # "Canal TVA Sports HD" → "canal tva sports". Token-contains
            # matches, BUT the trailing digit suffix (if any) must match
            # exactly so "TVA Sports" (primary) and "TVA Sports 2"
            # (secondary) never cross over — during playoffs they carry
            # concurrent but DIFFERENT games. Same applies to RDS / RDS2.
            import re as _re_bc_match
            want_m = _re_bc_match.match(r"^(.*?)\s*(\d*)$", want.strip())
            norm_m = _re_bc_match.match(r"^(.*?)\s*(\d*)$", norm.strip())
            if want_m and norm_m and want_m.group(2) == norm_m.group(2):
                needed = [t for t in want_m.group(1).split() if len(t) >= 2]
                if needed and all(tok in norm for tok in needed):
                    return True
        # Playoffs: Sportsnet simulcasts on the Canadian regional feeds (East,
        # West, Ontario, Pacific, Atlantic) — those carry the same SN broadcast
        # in-market. Sportsnet 360 / Sportsnet One are excluded: even during
        # playoffs they routinely run other sports (NBA, F1, soccer) and
        # surfacing them as a "this game" chip sent users to the wrong content.
        # If a game truly airs on SN1 / SN360, NHL API ships an explicit
        # "SN1" / "SN360" code that gets its own broadcast row.
        if is_playoffs and want == "sportsnet" and norm in (
            "sportsnet",
            "sportsnet east", "sportsnet west",
            "sportsnet ontario", "sportsnet pacific", "sportsnet atlantic",
        ):
            return True
        # US national playoff nets (TNT, TBS) ship as "TNT East HD" / "TNT West
        # HD" on our IPTV providers — timezone variants that simulcast the same
        # game. Allow those directional suffixes on TNT/TBS specifically, so we
        # don't drag in unrelated networks ("TNT Sports" Argentina, "TBS Very
        # Funny" Mexico, etc.). Regular season same behavior is fine — TNT/TBS
        # only carry NHL on Wed/Thu national windows, no regional overlap.
        if want in ("tnt", "tbs") and norm in (f"{want} east", f"{want} west"):
            return True
        # CBC regional outlets ("CBC Montreal", "CBC Toronto", etc.) simulcast
        # the national broadcast — match them to the "CBC" row in regular
        # season only, and only for the city of either team in this game (so
        # an MTL game doesn't surface CBC Toronto and vice versa). Excluded
        # in playoffs because CBC's national rights overlap differently with
        # the regional schedules and the city feeds may carry filler.
        if want == "cbc" and not is_playoffs:
            for team in (home_abbrev, away_abbrev):
                city = _TEAM_CITY_FOR_CBC.get(team)
                if city and norm == f"cbc {city}":
                    return True
        return False

    # NHL API casings: "TNT", "TBS", "SN", "CBC" (upper) but also "truTV"
    # (camelCase) and "HBO MAX" (spaced). Normalize the lookup key so all three
    # resolve through the same map regardless of how NHL types them today.
    _bc_map_ci = {k.upper(): v for k, v in _BROADCAST_CODE_MAP.items()}

    result: list[dict] = []
    seen_codes: set[str] = set()
    for b in tv_broadcasts:
        code = (b.get("network") or "").strip()
        market = b.get("market", "N")  # "H" | "A" | "N"
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)

        display = _bc_map_ci.get(code.upper(), code)
        matched = [
            ch for ch in curated_channels
            if _matches_broadcast(ch.get("title", ""), display)
        ]
        # Sort so working URLs land first (tvpass redirect → thetvapp → generic
        # upstream → an upstream host). Same ordering /barncentre-channels uses — keeps
        # TVA Sports from picking an upstream host's dead /play/TOKEN/m3u8 as primary.
        matched = _sort_by_url_priority(matched)
        # Force `q=720p` re-encode on relay /hls URLs for TSN broadcasts. Their
        # source MPEG-TS has irregular GOPs that produce variable-duration
        # segments under passthrough; hls.js misreads those as live-edge drift
        # and triggers buffer/resync cycles. Game-page parity with BarnCentre.
        if _needs_recode("", code) or _needs_recode("", display):
            matched = [
                {**ch, "url": _apply_recode(ch["url"], force=True)}
                for ch in matched
            ]
        # Always emit a row — the frontend renders unmatched rows as disabled
        # chips so Bob sees the full broadcast slate, not just what we resolved.
        result.append({
            "network":  display,
            "code":     code,
            "market":   market,
            "channels": matched,
        })

    # Per-team dedicated NHL feeds (e.g. "US : NHL MONTREAL CANADIENS"). These
    # are independent of NHL API's tvBroadcasts list — they're a per-team
    # always-on row so users have a fallback when the national/regional slate
    # is thin (especially during playoffs where the home/away market filter
    # strips RSNs). Emitted in away→home order so the away team's feed sits
    # above the home team's feed in the slate, matching how the frontend
    # already orders Home/Away sections.
    for team_abbrev, market in ((away_abbrev, "A"), (home_abbrev, "H")):
        needle = _TEAM_FEED_NEEDLES.get(team_abbrev)
        if not needle:
            continue
        feed_matched = [
            ch for ch in curated_channels
            if needle in (ch.get("title", "") or "").lower()
        ]
        if not feed_matched:
            continue
        feed_matched = _sort_by_url_priority(feed_matched)
        result.append({
            "network":  _TEAM_FEED_LABELS.get(team_abbrev, f"{team_abbrev} Feed"),
            "code":     f"FEED-{team_abbrev}",
            "market":   market,
            "team":     team_abbrev,
            "channels": feed_matched,
        })

    # National-row ordering preference. NHL API ships TNT/Sportsnet/TVA/etc.
    # in roughly broadcast-priority order, but the priority Bob wants in the
    # UI is: Sportsnet first (densest catalog, most reliable feeds), then TNT,
    # then TVA Sports — followed by every other national row in original
    # order. Home/away/team-feed rows are untouched (they're rendered in
    # their own sections by the frontend).
    _NATIONAL_PRIORITY = ("SN", "TNT", "TVAS")
    def _national_sort_key(row: dict) -> tuple[int, int]:
        if row.get("market") != "N":
            return (1, 0)
        code = (row.get("code") or "").upper()
        try:
            return (0, _NATIONAL_PRIORITY.index(code))
        except ValueError:
            return (0, len(_NATIONAL_PRIORITY))
    # Stable sort preserves original order within each priority bucket.
    result.sort(key=_national_sort_key)

    return result


def _kick_chip_warmups(broadcasts: list[dict]) -> None:
    """Fire-and-forget GET against the top relay /hls URL of up to 5 distinct
    networks. The relay's ffmpeg cold-start (~4-7s on libx264) used to be paid
    in full by whoever clicked first; this spawns the session before any user
    click. Cap of 5 per call avoids spawning too many concurrent ffmpegs on
    slate load. _PROXY_INFLIGHT dedups concurrent collisions.

    We warm any chip whose final URL hits the relay's /hls endpoint — that's
    where ffmpeg actually starts. Covers both an upstream host and ts-only ampztl
    variants. /m3u8 passthrough URLs (no ffmpeg) and direct provider URLs
    (no benefit, may anger ratelimits) are skipped.
    """
    try:
        import asyncio as _aio_warm
        warmups: list[str] = []
        seen_networks: set[str] = set()
        for b in broadcasts:
            if len(warmups) >= 5:
                break
            net = b.get("network", "")
            if net in seen_networks:
                continue
            for c in b.get("channels", []):
                u = c.get("url", "")
                if "/hls?" in u and u.startswith("http"):
                    warmups.append(u)
                    seen_networks.add(net)
                    break
        for url in warmups:
            _aio_warm.create_task(_warm_relay_url(url))
    except Exception:
        pass


async def _warm_relay_url(url: str) -> None:
    """Single GET at a relay /hls URL to spin up its ffmpeg session.

    Body is discarded. Any error is swallowed — this is best-effort.
    """
    try:
        client = await _get_proxy_http()
        r = await client.get(
            url,
            headers={
                "User-Agent": "Grtzky-Warmup/1.0",
            },
            timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0),
        )
        _ = r.content
    except Exception:
        return


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
from fastapi.responses import Response as FastResponse, StreamingResponse
import re as _re_proxy


def _rewrite_m3u8(content: str, original_url: str, proxy_base: str) -> str:
    """Rewrite .m3u8 and .ts URLs in a manifest to go through our proxy."""
    from urllib.parse import urlparse, urlunparse

    # Fast-path: if the manifest came from our own relay, its segment URLs
    # already point back at the relay (e.g. localhost:8000/hls-seg/...,
    # localhost:8000/ts?u=...). The relay has CORS wide open
    # (Access-Control-Allow-Origin: *) so the browser can fetch those
    # directly. Rewriting them through /api/stream-proxy adds a wasted
    # browser→Vercel→localhost:8000→relay hop on EVERY 3-second segment
    # fetch (~50-150ms each), for hours of viewing. Skip the rewrite —
    # browser hits the relay direct.
    parsed_original = urlparse(original_url)
    if (parsed_original.hostname or "").lower() == "localhost:8000":
        return content

    # If the original URL has query params (e.g. ?token=...&expires=...&user_id=...)
    # those must be inherited by relative segment URLs that have no own params.
    # thetvapp.to tokens are IP-locked and applied per-manifest; all segments share them.
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
# ---------------------------------------------------------------------------
# Backup-stream embed liveness probe
# ---------------------------------------------------------------------------
# onhockey.tv + the secondary scrapers (streameast, methstreams, etc.) ship
# *template* URLs that always 200 — wikisport.club/court/<n>.php, streams.center
# /embed/ch<n>.php, embedhd.org/source/fetch.php?hd=<n>, cdnlivetv.tv player
# proxies, etc. The HTML page exists, our /stream-embed wrapper renders it,
# the heartbeat fires forever — but the inner CDN is dead, the wrong sport,
# or geo-blocked from our backend, so the user just stares at a broken-image
# icon with no auto-skip. Pre-flighting each embed before returning it as a
# chip is the only way to stop those chips from reaching the UI.
_embed_alive_cache: dict[str, tuple[bool, float]] = {}
_EMBED_ALIVE_TTL = 600  # 10 min — long enough to skip the probe on cache hits,
                        # short enough that providers coming back online surface

# Substrings that, on their own, mean the page is a dead skeleton or a
# wrong-sport landing. Matched after lowercasing the body.
_DEAD_BODY_MARKERS = (
    'window.location.replace("/sorry"',  # wikisport.club, livesport.ws family
    "window.location.replace('/sorry'",
    "this stream is offline",
    "stream not available",
    "stream is not available",
    "no stream found",
    "this content is not available",
    "geo-restricted",
    "not licensed for your region",
    # Generic "page deleted / 404 in 200" markers. These render as a white
    # broken-image-ish page in the iframe — kill them at probe time.
    "content removed",
    "page not found",
    "404 not found",
    "video unavailable",
    "match has not started",
    "stream will be available",
    "no stream is currently available",
    "<h4>content removed",
    "<h1>404",
    "this page has been removed",
    "the requested url was not found",
    "stream will start",
    "stream offline",
)

# Markers that strongly indicate a working video player is present in the body.
# A page is considered alive if it has any of these.
_LIVE_BODY_MARKERS = (
    "<video",
    "jwplayer",
    "clappr",
    "playerinstance",
    "hls.js",
    ".m3u8",
    "video.js",
    "videojs",
    "shaka",
    "dashjs",
)


async def _probe_embed_alive(url: str) -> bool:
    """Return True if the embed URL is plausibly playable.

    Heuristic: GET the page with browser headers + onhockey Referer, then:
      • if any _DEAD_BODY_MARKERS appears → False
      • if body < 1.5 KB and no _LIVE_BODY_MARKERS → False
      • if status is 4xx/5xx → False
    Otherwise → True. Errs on the side of keeping the chip — false positives
    here only mean a working stream gets dropped, which is worse than letting
    one bad chip through (the user can still click "try next").

    Cached for 10 minutes so a chip appearing across 5 scrapers is probed once.
    """
    import time as _tep
    cached = _embed_alive_cache.get(url)
    if cached and (_tep.monotonic() - cached[1]) < _EMBED_ALIVE_TTL:
        return cached[0]

    try:
        async with _httpx_backup.AsyncClient(
            timeout=6,
            follow_redirects=True,
            headers={
                **_BROWSER_HEADERS,
                "Referer": "https://onhockey.tv/",
            },
        ) as hx:
            r = await hx.get(url)
        if r.status_code >= 400:
            _embed_alive_cache[url] = (False, _tep.monotonic())
            return False
        body = r.text
        body_lower = body.lower()
        if any(m in body_lower for m in _DEAD_BODY_MARKERS):
            _embed_alive_cache[url] = (False, _tep.monotonic())
            return False
        # embedsports.top/.me popunder stub: every URL returns the same
        # 1221-byte clappr-loader that calls aclib.runPop and reads stream
        # config from the URL hash. onhockey.tv passes us URLs without a
        # hash, so the stub never resolves to a real stream. Detect by the
        # aclib.runpop tell + absence of any actual stream-source token.
        from urllib.parse import urlparse as _u
        try:
            host = _u(url).netloc.lstrip("www.")
        except Exception:
            host = ""
        if host in ("embedsports.top", "embedsports.me") and "aclib.runpop" in body_lower:
            has_stream_src = any(
                s in body_lower for s in (".m3u8", ".mpd", '"file":"http', "'file':'http", '"source":"http')
            )
            if not has_stream_src:
                _embed_alive_cache[url] = (False, _tep.monotonic())
                return False
        # Tiny bodies are only OK if they contain a player marker — a 1KB page
        # with no <video>/jwplayer/clappr/m3u8 string is a placeholder.
        has_player = any(m in body_lower for m in _LIVE_BODY_MARKERS)
        if len(body) < 1500 and not has_player:
            _embed_alive_cache[url] = (False, _tep.monotonic())
            return False
        # Broken-image-only stub: small-ish body with <img> and no <video>/
        # player markers is the white-screen-with-broken-image-icon page Bob
        # was hitting. Drop it before it reaches the chip list.
        if not has_player and len(body) < 4000 and "<img" in body_lower:
            _embed_alive_cache[url] = (False, _tep.monotonic())
            return False
        _embed_alive_cache[url] = (True, _tep.monotonic())
        return True
    except Exception:
        # Network error talking to the third-party — treat as dead. If it was
        # a transient blip the cache TTL expires in 10 min and we re-probe.
        _embed_alive_cache[url] = (False, _tep.monotonic())
        return False


async def _filter_dead_embeds(streams: list[dict]) -> list[dict]:
    """Drop chips whose embed URL is clearly dead. Direct m3u8 / known-clean
    extractor sources skip the probe — those have other liveness signals."""
    import asyncio as _aio_filt
    # Probe every embed_only chip. Auto-skip is gone on the frontend, so a
    # dead chip just shows a white-screen-with-broken-image until the user
    # taps "try next ›" — better to drop them all up here. Direct m3u8
    # (priority 0) skips the probe; those have hls.js error handling.
    needs_probe = [
        s for s in streams
        if s.get("embed_only")
    ]
    if not needs_probe:
        return streams

    probe_results = await _aio_filt.gather(
        *(_probe_embed_alive(s["url"]) for s in needs_probe),
        return_exceptions=True,
    )
    dead: set[str] = set()
    for s, alive in zip(needs_probe, probe_results, strict=False):
        if isinstance(alive, BaseException) or not alive:
            dead.add(s["url"])
    if not dead:
        return streams
    return [s for s in streams if s["url"] not in dead]


# Hosts where Playwright extraction has no chance of returning an m3u8.
# Trimmed aggressively (Apr 2026): every other host gets a Playwright run via
# /stream-resolve so the user gets a clean m3u8 served through /stream-proxy.
# Iframing those hosts directly was broken for any user with a popup-ad
# blocker (uBlock / AdGuard / Brave / Pi-hole) because the embed pages load
# aclib.runPop / popads, which the blocker cuts and Chrome reports as
# "<host> refused to connect" inside the iframe.
#
# What stays here:
#   - onhockey.tv: sets X-Frame-Options: SAMEORIGIN, can't be iframed; keep
#     for safety so _resolve_embed short-circuits if a stray onhockey URL
#     ever leaks through to a chip URL.
#   - nhl.com / espn.com / nhl66.ir / nhltv.is: auth-walled / DRM, no public
#     m3u8 to extract.
_EMBED_ONLY_DOMAINS = {
    "onhockey.tv",
    "nhl.com", "espn.com", "nhl66.ir", "nhltv.is",
    # embedsports.top / .me are popunder stubs — every URL returns the same
    # 1221-byte clappr-loader page that reads stream config from the URL
    # hash, but onhockey.tv hands us URLs without a hash, so there's
    # nothing for Playwright to extract. We keep them here so the chip is
    # routed through the iframe path → _filter_dead_embeds probes it →
    # _probe_embed_alive's aclib-stub check drops it before it reaches
    # the user. Net effect: these chips disappear from the list.
    "embedsports.top", "embedsports.me",
}


def _is_embed_only_host(url: str) -> bool:
    """True iff the URL's host is in _EMBED_ONLY_DOMAINS.

    Used at chip-stamp time to decide whether to set ``embed_only=true`` on a
    stream entry. When False, the frontend will hit /stream-resolve and the
    Playwright extraction chain gets a chance to find a real m3u8.
    """
    from urllib.parse import urlparse as _u
    try:
        host = _u(url).netloc.lstrip("www.")
    except Exception:
        return False
    return host in _EMBED_ONLY_DOMAINS


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
      • Sandbox blocks popups, top-nav, downloads, pointer-lock — kills every
        ad vector the scraper streams rely on while still letting the player JS
        run (allow-scripts + allow-same-origin).
      • Overrides window.open at the wrapper level as a second layer
      • Watchdog: if the inner frame navigates (ad redirect that escapes
        sandbox on non-sandbox providers), post ``Origin-stream-killed`` so the
        outer frontend skips to the next stream chip.

    The ``mobile`` query flag tightens the sandbox (strips ``allow-presentation``
    which some ad players abuse to trigger fullscreen popups on iOS).
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

    # Parse optional flags. ``mobile=1`` tightens the sandbox; ``priority`` is
    # accepted for forward-compat (frontend already sends it; reserved for
    # per-stream sandbox tuning if specific providers break).
    q = request.query_params
    mobile = q.get("mobile") == "1"

    # Sandbox policy:
    #   Mobile: sandboxed — blocks popup ads, force-fullscreen, and popunders
    #     on iOS/Android where ad vectors are most hostile. Some players
    #     (JWPlayer/Clappr) refuse to initialise under sandbox, but Bob's
    #     call: a dead chip is better than a popup-ad chip on a phone.
    #   Desktop: no sandbox — many RSN / JWPlayer / Video.js / custom players
    #     detect a sandboxed ancestor and refuse to initialise, which broke
    #     local stream playback on localhost:3000. Desktop users get the normal
    #     browser ad surface in exchange for the local feeds actually loading.
    sandbox_html = ' sandbox="allow-scripts allow-same-origin"' if mobile else ""

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

  // Heartbeat: confirms /stream-embed loaded and is alive so the parent can
  // distinguish "wrapper never rendered" (no heartbeat → auto-skip after
  // ~12s) from "wrapper alive, inner iframe playing some content." We can't
  // peek inside cross-origin iframes to verify playback, but a missing
  // heartbeat reliably catches the cases the user actually sees most:
  // backend timeouts, relay tunnel down, or the wrapper getting kicked out
  // by an ad redirect.
  var killed = false;
  function postBeat() {{
    if (killed) return;
    try {{ parent.postMessage({{ type: 'Origin-frame-progress', url: {safe_url!r} }}, '*'); }} catch (_) {{}}
  }}
  postBeat();
  var beatTimer = setInterval(postBeat, 3000);

  // Watchdog: if the inner iframe's location ever changes (ad redirect that
  // escapes sandbox), we lose the video. Tell the outer frontend to skip.
  var frame = document.querySelector('iframe');
  if (frame) {{
    var lastSrc = frame.src;
    frame.addEventListener('load', function() {{
      try {{
        // Cross-origin: reading frame.contentWindow.location throws. Only
        // same-origin (navigated-away) loads reach this branch, which is
        // exactly the ad-redirect case we want to kill.
        var loc = frame.contentWindow.location.href;
        if (loc && loc !== lastSrc && loc !== 'about:blank') {{
          killed = true;
          if (beatTimer) {{ clearInterval(beatTimer); beatTimer = null; }}
          parent.postMessage({{ type: 'Origin-stream-killed', url: {safe_url!r} }}, '*');
        }}
      }} catch (_) {{ /* cross-origin — healthy */ }}
    }});
  }}
}})();
</script>
</head>
<body>
<iframe
  src="{safe_url}"{sandbox_html}
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


# ---------------------------------------------------------------------------
# Pooled HTTP client + tiny in-process cache for /stream-proxy.
#
# Cold-start buffering on Barncentre / game pages came down to two avoidable
# costs on every fetch this proxy makes upstream:
#   1. A fresh TCP+TLS handshake per request (~100-200 ms each), because the
#      old code opened a new httpx.AsyncClient inside every handler call.
#   2. No collapsing of duplicate work — ten viewers tapping the same chip
#      meant ten manifest fetches plus ten segment fetches end-to-end.
#
# The pooled client below keeps a small set of warm keep-alive connections
# to each upstream CDN; the cache absorbs the thundering-herd on chip clicks
# and on hover/touch warmup. Segments are immutable for their lifetime so
# they can be cached briefly; manifests get a tighter TTL because live
# playlists rotate every few seconds.
#
# HTTP/2 stays off — enabling it requires `httpx[http2]` which would add
# the `h2` dep; HTTP/1.1 keep-alive already removes the per-request
# handshake, which is the dominant cold-start cost.
# ---------------------------------------------------------------------------
_PROXY_HTTP: httpx.AsyncClient | None = None

_PROXY_MANIFEST_TTL = 1.0   # seconds; live playlists rotate every 2-6 s
_PROXY_SEGMENT_TTL = 30.0   # seconds; segments are immutable
# Cache hard-fail upstream responses. When an upstream account is auth-throttled
# (typical penalty for `/live/*` probe floods is 24-48h), every hover warmup
# fires a fresh /m3u8 probe; without caching, browsing barncentre during the
# throttle window keeps the upstream provider locked out indefinitely. 60s
# is short enough that genuine upstream recovery (rare on the same minute,
# but possible on transient outages) isn't masked for long, and long enough
# that a hover scan across 20 chips doesn't generate 20 redundant probes
# against the same throttled account.
_PROXY_4XX_TTL = 60.0
_PROXY_CACHE_MAX_BYTES = 32 * 1024 * 1024
_PROXY_CACHE: dict[str, tuple[float, str, bytes, int]] = {}
# value: (expires_at, content_type, body, status_code)
_PROXY_CACHE_BYTES = 0
_PROXY_INFLIGHT: dict[str, "asyncio.Future[tuple[str, bytes, int]]"] = {}


async def _get_proxy_http() -> httpx.AsyncClient:
    """Return the pooled client; lazy-init in case startup didn't run.

    read=30s is a deliberate bump over the old 15s. The relay's /hls
    endpoint blocks for up to HLS_STARTUP_TIMEOUT_S (15s) waiting for
    ffmpeg to produce the first manifest on cold start. With both timers
    set to 15s the API client gave up at exactly 15.04s and surfaced a
    502, killing the player before the relay even finished spinning up —
    the "video doesn't load unless we press Cast" symptom. 30s gives the
    relay headroom while still failing reasonably fast on a dead chip.
    """
    global _PROXY_HTTP
    if _PROXY_HTTP is None:
        _PROXY_HTTP = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=4.0, read=30.0, write=15.0, pool=5.0),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=100,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
    return _PROXY_HTTP


def _proxy_cache_evict_if_needed(incoming_bytes: int) -> None:
    global _PROXY_CACHE_BYTES
    if _PROXY_CACHE_BYTES + incoming_bytes <= _PROXY_CACHE_MAX_BYTES:
        return
    # Drop expired entries first, then oldest by expires_at until under cap.
    now = _time.monotonic()
    for k, (exp, _ct, body, _sc) in list(_PROXY_CACHE.items()):
        if exp <= now:
            _PROXY_CACHE_BYTES -= len(body)
            _PROXY_CACHE.pop(k, None)
    if _PROXY_CACHE_BYTES + incoming_bytes <= _PROXY_CACHE_MAX_BYTES:
        return
    for k, _v in sorted(_PROXY_CACHE.items(), key=lambda kv: kv[1][0]):
        body = _v[2]
        _PROXY_CACHE_BYTES -= len(body)
        _PROXY_CACHE.pop(k, None)
        if _PROXY_CACHE_BYTES + incoming_bytes <= _PROXY_CACHE_MAX_BYTES:
            return


def _proxy_cache_get(url: str) -> tuple[str, bytes, int] | None:
    entry = _PROXY_CACHE.get(url)
    if not entry:
        return None
    expires_at, content_type, body, status_code = entry
    if expires_at <= _time.monotonic():
        global _PROXY_CACHE_BYTES
        _PROXY_CACHE_BYTES -= len(body)
        _PROXY_CACHE.pop(url, None)
        return None
    return content_type, body, status_code


def _proxy_cache_put(url: str, content_type: str, body: bytes, status_code: int, ttl: float) -> None:
    if ttl <= 0:
        return
    # Allow success (200 + body) and hard-fail 4xx (with or without body).
    # 4xx caching breaks the retry storm where hls.js spams the relay after
    # an upstream account throttles — see _PROXY_4XX_TTL above for the why.
    if status_code == 200:
        if not body:
            return
    elif status_code not in (401, 403, 404, 410):
        return
    global _PROXY_CACHE_BYTES
    _proxy_cache_evict_if_needed(len(body))
    _PROXY_CACHE[url] = (_time.monotonic() + ttl, content_type, body, status_code)
    _PROXY_CACHE_BYTES += len(body)


@app.on_event("startup")
async def _start_proxy_http() -> None:
    await _get_proxy_http()


@app.on_event("startup")
async def _prewarm_barncentre() -> None:
    """Build the BarnCentre cache in the background at startup so the first
    real user lands on a warm cache instead of paying the full 50-65s cold
    fan-out (NHL + ESPN + MLB + NBA + IPTV sources + ~100 verifier probes).
    """
    async def _runner() -> None:
        try:
            await _build_barncentre_payload()
        except Exception:
            pass
    asyncio.create_task(_runner())


@app.on_event("startup")
async def _start_stream_resolver() -> None:
    """Initialise the stream resolver if STREAM_RESOLVER_ENABLED=1.

    Lazy: provider classes are constructed but Playwright isn't started until
    the first resolve. This costs ~5 ms at boot when disabled, ~50 ms when
    enabled (just imports + provider construction).
    """
    if os.environ.get("STREAM_RESOLVER_ENABLED") != "1":
        return
    async def _runner() -> None:
        try:
            from .stream_resolver import init_resolver
            r = await init_resolver()
            print(f"[stream_resolver] startup: ready providers={list(r.providers.keys())}",
                  flush=True)
            # Prewarm DISABLED on this 2GB VPS — running prewarm in parallel
            # with user-facing resolves doubled Chromium memory under load
            # and starved real requests. First user pays a one-time
            # ~12s cold-start; subsequent users hit the 24h cache.
        except Exception as e:
            import traceback
            print(f"[stream_resolver] startup FAILED: {e}", flush=True)
            traceback.print_exc()
    asyncio.create_task(_runner())


@app.on_event("shutdown")
async def _stop_stream_resolver() -> None:
    if os.environ.get("STREAM_RESOLVER_ENABLED") != "1":
        return
    try:
        from .stream_resolver import shutdown_resolver
        await shutdown_resolver()
    except Exception:
        pass


@app.on_event("startup")
async def _prewarm_lounge_epg() -> None:
    """Build the Lounge EPG cache at startup. Cold build is 8-15s (XMLTV
    fetch from each upstream account); the Vercel /api/[...backend] proxy
    has a 10s default function timeout, so the first cold-cache EPG hit
    after a deploy was 504-ing through Vercel and timing out the lounge
    client. Pre-warming on startup eliminates the cold-cache window.
    """
    async def _runner() -> None:
        import time as _t_pwe
        try:
            merged = await _build_lounge_epg()
            _lounge_epg_cache["data"] = merged
            _lounge_epg_cache["ts"]   = _t_pwe.time()
        except Exception:
            pass
    asyncio.create_task(_runner())


@app.on_event("shutdown")
async def _stop_proxy_http() -> None:
    global _PROXY_HTTP
    if _PROXY_HTTP is not None:
        await _PROXY_HTTP.aclose()
        _PROXY_HTTP = None


@app.get("/stream-proxy")
async def stream_proxy(url: str, request: Request) -> FastResponse:
    """Proxy an HLS stream URL, rewriting internal URLs to also go through proxy."""
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
    }

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Cache-Control": "no-cache",
    }

    # Cheap path-only check so we can pick the right cache TTL up front.
    # Manifests get a tight TTL; segments inherit a longer one.
    _path_lower = url.split("?", 1)[0].lower()
    looks_like_segment = (
        _path_lower.endswith(".ts")
        or _path_lower.endswith(".aac")
        or _path_lower.endswith(".m4s")
        or _path_lower.endswith(".fmp4")
        or _path_lower.endswith(".mp4")
    )

    # Segments are content-addressed (ffmpeg writes seg00NN.ts once and never
    # rewrites it; CDN segments are URL-stable for their lifetime). Telling
    # Cloudflare Tunnel + browser to cache for 60s lets repeat fetches —
    # back-buffer reads, second-viewer-of-same-chip, hls.js retry-on-blip —
    # skip the whole API → relay → upstream chain.
    seg_cors_headers = {
        **cors_headers,
        "Cache-Control": "public, max-age=60, immutable",
    }

    cached = _proxy_cache_get(url)
    if cached is not None:
        ct, body, _sc = cached
        # Cached hard-fail (4xx) — pass through as-is. Skip m3u8 rewrite (the
        # body is an HTML error page or empty, not a manifest) and skip the
        # immutable segment Cache-Control (don't tell CF to cache the failure
        # for 60s when the upstream is likely to recover in <30s).
        if _sc and _sc >= 400:
            return FastResponse(
                content=body,
                media_type=ct or "application/octet-stream",
                headers=cors_headers,
                status_code=_sc,
            )
        if "mpegurl" in ct.lower():
            text = body.decode("utf-8", errors="replace")
            text = _rewrite_m3u8(text, url, proxy_base)
            return FastResponse(
                content=text,
                media_type="application/vnd.apple.mpegurl",
                headers=cors_headers,
                status_code=_sc or 200,
            )
        return FastResponse(
            content=body,
            media_type=ct or "application/octet-stream",
            headers=seg_cors_headers if looks_like_segment else cors_headers,
            status_code=_sc or 200,
        )

    # Segment cache miss: stream the body through instead of buffering ~1 MB
    # into memory before responding. Avoids ~50-100 ms of Time-To-First-Byte
    # latency per segment, which compounds across the player's continuous
    # 3s-segment fetch loop. We deliberately skip the in-flight dedupe + cache
    # write here — segments are too large to keep in the 32 MB _PROXY_CACHE
    # without thrashing the manifest entries that actually benefit from
    # dedupe, and the new immutable Cache-Control above moves dedupe to
    # Cloudflare + browser where it scales without API memory pressure.
    if looks_like_segment:
        try:
            client = await _get_proxy_http()
            req = client.build_request("GET", url, headers=_stream_headers)
            upstream = await client.send(req, stream=True)
            if upstream.status_code != 200:
                # Non-200 path: drain + fall through to the existing buffered
                # branch so 4xx/5xx bodies are surfaced unchanged.
                await upstream.aclose()
            else:
                up_ct = upstream.headers.get("content-type") or "video/mp2t"
                async def _stream_segment() -> "collections.abc.AsyncIterator[bytes]":
                    try:
                        async for chunk in upstream.aiter_bytes(64 * 1024):
                            yield chunk
                    finally:
                        await upstream.aclose()
                return StreamingResponse(
                    _stream_segment(),
                    media_type=up_ct,
                    headers=seg_cors_headers,
                )
        except Exception:
            # Network blip on the streaming path — fall through to the
            # buffered path, which has its own try/except + 502 handling.
            pass

    # In-flight dedupe: if another request for the same URL is already
    # fetching upstream, await its result instead of opening a second fetch.
    inflight = _PROXY_INFLIGHT.get(url)
    if inflight is not None:
        try:
            ct, body, sc = await inflight
            if sc == 200:
                if "mpegurl" in ct.lower():
                    text = body.decode("utf-8", errors="replace")
                    text = _rewrite_m3u8(text, url, proxy_base)
                    return FastResponse(
                        content=text,
                        media_type="application/vnd.apple.mpegurl",
                        headers=cors_headers,
                    )
                return FastResponse(
                    content=body,
                    media_type=ct or "application/octet-stream",
                    headers=cors_headers,
                )
        except Exception:
            pass  # fall through and try ourselves

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[tuple[str, bytes, int]] = loop.create_future()
    _PROXY_INFLIGHT[url] = fut

    try:
        client = await _get_proxy_http()
        resp = await client.get(url, headers=_stream_headers)

        # Relay cold-start race — the relay's /hls endpoint can return 503
        # "manifest not yet available" when ffmpeg has spawned but hasn't
        # produced the first segment yet. Retry once after a short wait
        # before surfacing the error. This is the same race that used to
        # show as "video doesn't load until you press Cast" — Cast went
        # direct to the relay and the browser tolerated the wait, but the
        # proxy path saw the 503 and gave up.
        if (
            resp.status_code == 503
            and "/hls" in url.lower()
            and b"manifest not yet available" in resp.content
        ):
            await asyncio.sleep(2.0)
            resp = await client.get(url, headers=_stream_headers)

        content_type = resp.headers.get("content-type", "")
        body = resp.content

        # Hard upstream failure (401/403/404/410). Cache briefly and return
        # immediately — don't fall through to m3u8/HTML/segment processing.
        # The 4xx cache breaks the retry-storm pattern where hls.js spams the
        # relay after an upstream account throttles, compounding the provider's
        # connection-limit rejection.
        if resp.status_code in (401, 403, 404, 410):
            _proxy_cache_put(url, content_type, body, resp.status_code, _PROXY_4XX_TTL)
            if not fut.done():
                fut.set_result((content_type, body, resp.status_code))
            return FastResponse(
                content=body,
                media_type=content_type or "application/octet-stream",
                headers=cors_headers,
                status_code=resp.status_code,
            )

        # If it's an m3u8 manifest, rewrite internal URLs.
        # Only treat as M3U8 on success — a 403/404 with a .m3u8 URL returns HTML
        # which must not be passed to _rewrite_m3u8 as garbage segment lines.
        is_m3u8 = resp.status_code == 200 and (
            "mpegurl" in content_type.lower()
            or url.split("?")[0].endswith(".m3u8")
            or body[:7] == b"#EXTM3U"
        )

        if is_m3u8:
            # Cache the *raw* upstream body keyed by the upstream URL so warmup
            # hits land here on the actual click. Rewriting happens per-request
            # because proxy_base depends on x-forwarded-host.
            stored_ct = content_type or "application/vnd.apple.mpegurl"
            _proxy_cache_put(url, stored_ct, body, 200, _PROXY_MANIFEST_TTL)
            if not fut.done():
                fut.set_result((stored_ct, body, 200))
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
            # Layer 1: yt-dlp (JWPlayer/Video.js config detection, no browser, ~2s)
            real_m3u8 = await asyncio.to_thread(_ytdlp_extract_m3u8, url)

            # Layer 2: Playwright network interception (headless browser, ~8s)
            if not real_m3u8:
                real_m3u8 = await _playwright_extract_m3u8(url)

            if real_m3u8:
                if not real_m3u8.startswith("http"):
                    real_m3u8 = urljoin(url, real_m3u8)
                m3u8_resp = await client.get(real_m3u8, headers=_stream_headers)
                m3u8_body = m3u8_resp.content
                stored_ct = m3u8_resp.headers.get("content-type") or "application/vnd.apple.mpegurl"
                _proxy_cache_put(real_m3u8, stored_ct, m3u8_body, m3u8_resp.status_code, _PROXY_MANIFEST_TTL)
                if not fut.done():
                    fut.set_result(("application/vnd.apple.mpegurl", m3u8_body, m3u8_resp.status_code))
                m3u8_text = m3u8_body.decode("utf-8", errors="replace")
                m3u8_text = _rewrite_m3u8(m3u8_text, real_m3u8, proxy_base)
                return FastResponse(
                    content=m3u8_text,
                    media_type="application/vnd.apple.mpegurl",
                    headers=cors_headers,
                )
            # Layer 3: iframe fallback — tell the frontend to embed the page directly
            import json as _json
            if not fut.done():
                fut.set_result(("application/json", b"", 422))
            return FastResponse(
                content=_json.dumps({"fallback_url": url}),
                status_code=422,
                media_type="application/json",
                headers={**cors_headers, "Content-Type": "application/json"},
            )

        # Binary segment (.ts, .aac, etc.) — stream through as-is.
        # Cache only if the path looks like a segment, so we don't accidentally
        # cache an HTML error page that slipped through the checks above.
        if looks_like_segment:
            _proxy_cache_put(url, content_type, body, resp.status_code, _PROXY_SEGMENT_TTL)
        if not fut.done():
            fut.set_result((content_type, body, resp.status_code))
        # Preserve upstream status_code. Returning a fake 200 for a 4xx/5xx
        # makes hls.js retry up to 3x at 9s each (~27s of spinner) instead
        # of failing fast — the player just sits there spinning when in
        # reality the chip is dead and we should advance to the next one.
        return FastResponse(
            content=body,
            media_type=content_type or "application/octet-stream",
            headers=seg_cors_headers if looks_like_segment and resp.status_code == 200 else cors_headers,
            status_code=resp.status_code,
        )

    except Exception as exc:
        if not fut.done():
            fut.set_exception(exc)
        return FastResponse(
            content=str(exc),
            status_code=502,
            headers=cors_headers,
        )
    finally:
        # Always clear the in-flight slot. If we never set a result, awaiters
        # will see a CancelledError-like state and just re-fetch on their own.
        if not fut.done():
            fut.cancel()
        _PROXY_INFLIGHT.pop(url, None)


@app.get("/stream-success")
@app.post("/stream-success")
async def stream_success(url: str) -> FastResponse:
    """Frontend pings this when a stream actually starts playing.

    Used to rank chips on `/streams/{game_id}` — consistently-failing backup
    scrapers sink within their priority bucket, so users land on streams that
    have been recently confirmed working. GET is accepted alongside POST so
    `navigator.sendBeacon` and `fetch(..., { keepalive: true })` both work
    without a CORS preflight.
    """
    _record_stream_success(url)
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Cache-Control": "no-cache",
    }
    import json as _json
    return FastResponse(
        content=_json.dumps({"ok": True}),
        media_type="application/json",
        headers=cors_headers,
    )



@app.get("/player-shots/{player_id}")
async def player_shots(
    player_id: int,
    context: str = Query("season", description="season | playoffs"),
):
    """Return arena-adjusted shot coordinates for a player (last 2 seasons of MoneyPuck data)."""
    shots_dir = _GRETZKY_DATA_DIR / "shots"
    if not shots_dir.exists():
        return {"shots": [], "count": 0, "status": "no_data"}
    files = sorted(shots_dir.glob("*.parquet"))
    if not files:
        return {"shots": [], "count": 0, "status": "no_data"}
    import polars as pl
    want_cols = {"shooter_id", "arena_adj_x", "arena_adj_y", "x_goal", "is_goal", "shot_type", "is_playoff"}
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
    # Filter to the requested context so the heat map reflects regular-season
    # or playoff shots only.
    if "is_playoff" in player_df.columns:
        if context == "playoffs":
            player_df = player_df.filter(pl.col("is_playoff") == True)  # noqa: E712
        elif context == "season":
            player_df = player_df.filter(pl.col("is_playoff") == False)  # noqa: E712
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
async def goalie_shots(
    player_id: int,
    context: str = Query("season", description="season | playoffs"),
):
    """Return arena-adjusted shot coordinates for shots AGAINST a goalie (last 2 seasons)."""
    shots_dir = _GRETZKY_DATA_DIR / "shots"
    if not shots_dir.exists():
        return {"shots": [], "count": 0, "status": "no_data"}
    files = sorted(shots_dir.glob("*.parquet"))
    if not files:
        return {"shots": [], "count": 0, "status": "no_data"}
    import polars as pl
    want_cols = {"goalie_id", "arena_adj_x", "arena_adj_y", "x_goal", "is_goal", "shot_type", "is_playoff"}
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
    if "is_playoff" in player_df.columns:
        if context == "playoffs":
            player_df = player_df.filter(pl.col("is_playoff") == True)  # noqa: E712
        elif context == "season":
            player_df = player_df.filter(pl.col("is_playoff") == False)  # noqa: E712
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
    # FanDuel Sports Networks (formerly Bally Sports) — NHL regional US.
    # Only the slugs whose CDN backends return 200 are listed; the other
    # `fanduel-sports-network-*` slugs all 404 at e2.thetvapp.to. The albinchristo04
    # `streams.json` mirror (Source 2) carries additional working `-hd` variants.
    ("Fanduel Sports Network Detroit HD",     "fanduel-sports-network-detroit-hd", "hd"),
    ("Fanduel Sports Network Great Lakes HD", "fanduel-sports-network-great-lakes", "hd"),
]

_NHL_KEYWORDS = [
    "espn","tnt","tbs","nhl","sportsnet","tsn","msg","foxsports","fs1","fs2","abc","nbc","tva","rds",
    # US regional sports (Fanduel = rebranded Bally Sports; Victory+ = Ducks)
    "fanduel","bally","victory",
    # Other regional US nets that carry NHL
    "nesn","nbcs","nbcsp","spectrum sportsnet","nhln",
    # CBS Sports family — non-NHL carrier but surfaced in BarnCentre guide
    "cbs",
    # CBC + city variants (e.g. "CBC Montreal", "CBC Toronto"). National CBC
    # ships as "CBC" in NHL API; regional outlets simulcast the same broadcast
    # with local cut-ins, so they're surfaced via the per-team CBC city branch
    # in _matches_broadcast (regular season only).
    "cbc",
    # "fox sports" with a space — kstv & many providers spell it "Fox Sports 1"
    # rather than the contraction "FS1". Without this, FS1/FS2 chips were 0.
    "fox sports",
]

# Cable channels carried by Origin Lounge (the personal at-home Fire TV app).
# These get pulled into the same iptv_channels() candidate pool — downstream
# filters (`_BARNCENTRE_CHANNEL_NAMES` vs `_LOUNGE_CHANNEL_NAMES`) decide
# which app surfaces which channel, so broadening the enumeration keyword
# set doesn't change BarnCentre output.
_LOUNGE_CABLE_KEYWORDS = [
    # Premium movie networks
    "hbo", "cinemax", "starz", "showtime",
    # General entertainment
    "amc", " fx ", "fx hd", "fx movies", "fxx", "tnt", "a&e", "ae hd", "a e ",
    "comedy central", "comedy", "mtv",
    # Kids / animation
    "cartoon network", "adult swim", "teletoon", "boomerang",
    "pokemon", "sailor moon", "yu gi", "yu-gi",
    # Factual
    "discovery", "national geographic", "nat geo", "history",
    # Lifestyle
    "tlc", "food network", "hgtv", "oxygen",
    # Canadian premium cable
    "stack tv", "stack hd", "stacktv", "w network", "showcase", "slice",
    # Streaming-services-as-channel
    "paramount",
    # ── News — US + CA (added 2026-06-27) ──
    "cnn", "fox news", "bloomberg", "bnn", "hln", "cp24", "global news",
    "ctv news", "msnbc",
    # ── Canadian basic networks ──
    "global", "citytv", "city tv", "ctv",
    # ── Quebec French (CA) ──
    "noovo", "rdi", "lcn", "ici ", "tv5", "canal d", "canal vie",
    # ── More sports (US/CA) ──
    "dazn", "bein", "nfl", "redzone",
    # ── True crime ──
    "court tv", "true crime", "reelz",
    # ── Adult animation — 24/7 single-show channels ──
    "family guy", "american dad", "rick and morty", "south park",
    "futurama", "simpsons", "king of the hill", "bob's burgers", "bobs burgers",
]

# Combined filter used at IPTV-channel enumeration time. Sources upstream of
# this filter (upstream, M3U playlists, tvpass, etc.) get scanned for any of
# these substrings; BarnCentre + Lounge then pick what they need from the
# resulting candidate pool.
_IPTV_CHANNEL_KEYWORDS = _NHL_KEYWORDS + _LOUNGE_CABLE_KEYWORDS

# ---------------------------------------------------------------------------
# upstream Codes IPTV accounts — fetched at startup, cached 1hr alongside tvpass
# Each entry: (label, host, port, username, password)
# ---------------------------------------------------------------------------
_upstream_ACCOUNTS: list[tuple[str, str, int, str, str]] = [
    # kstv (puny243) — Bob's premium account, top priority everywhere.
    # Panel disables get.php (404), but exposes player_api.php — the fetcher
    # falls back to action=get_live_streams enumeration. ~327 NHL-keyword
    # channels, max 5 concurrent connections.
    ("kstv",        "kstv.us",             8080, "puny243",               "2033697598"),
    ("an upstream host",  "an upstream host.ddns.net", 8081, "PNbV7ywsHG",            "u7jmr3xvcM"),
    # ampztl arturo removed 2026-05-03: player_api.php auth=1 (60K catalog),
    # but every /arturo/<pw>/<id> stream returned 401 from upstream ffmpeg —
    # same API-only / no-stream-entitlement pattern as jdarnut an upstream host.
    ("ampztl-b",    "ampztl.xyz",          8080, "webtv1847",             "YsAPRy6Jq8TJ"),
    # bgdc.live — Bob's purchased accounts. Each line is 2 max concurrent.
    # Three accounts → 6 simultaneous streams; channel catalog triplicates
    # (one chip per account per channel) so users have failover capacity.
    # All three verified live via player_api.php on 2026-05-03 (auth=1, Active).
    ("bgdc",        "bgdc.live",          25461, "ionel.j.popa",          "8107963912"),
    ("bgdc-2",      "bgdc.live",          25461, "4ERkuQAz4t",            "qDvGPL9WV5"),
    ("bgdc-3",      "bgdc.live",          25461, "98nhcH9Ol6",            "XWr6xRtdBP"),
    # an upstream host.an upstream host.co — Bob's older account. jdarnut was tested and
    # removed: player_api.php returned auth=1 but every stream endpoint
    # returned 401/"Unauthorized access!" — API-only, no stream entitlement.
    # 09145054 (max=1, active=0) verified end-to-end: 200 OK on /m3u8 with
    # proper redirect handling to dal3.rocketdns.info CDN, valid MPEG-TS
    # segments. Other four creds on this panel skipped (60-220 active conns
    # suggest prior leak).
    ("an upstream host",   "an upstream host.an upstream host.co", 8080, "09145054",             "65339468"),
    # Disabled (DNS-dead 2026-05-02): rexmaximl.shop & subdomain return
    # "Not Authoritative" from Cloudflare DoH, anadolutv.shop returns SOA
    # with no A record (registered but unhosted), izletvhd.xyz SERVFAILs.
    # Domains likely seized or NS-misconfigured. Don't re-add without
    # re-verifying DNS.
    # ("rexmaximl",   "piaylgbuslxs8ua98elr7e.rexmaximl.shop", 8080, "rmz2021", "rmz.2021"),
    # ("anadolu-a",   "anadolutv.shop",      8080, "mstf6192",              "qcy8SyxFHtks"),
    # ("anadolu-b",   "anadolutv.shop",      8080, "rmzn0781",              "rmzn0781"),
    # ("izletvhd",    "izletvhd.xyz",        8080, "ft2301",                "cn.2504"),
    # Disabled: tv14s CDN tarpits our IP, lunar returns 403. Re-enable only
    # if providers start accepting requests again.
    # ("tv14s",       "tv14s.xyz",           8080, "Serentiy2@ogbtv.com",   "0306@1954"),
    # ("lunar",       "lunar.pm",            8080, "JeffOglesby",           "Marriage101"),
]

# Fresh upstream accounts discovered nightly by scripts/iptv_freshness.py (scrapes
# r/REDACTED_SOURCE, verifies via player_api). Additive + deduped: absent/empty file
# = no change, and a fresh account never displaces a hardcoded (purchased) one —
# the hardcoded accounts above keep priority; these append as supplements/failover
# and cover channels the dead/maxed providers (kstv, ampztl, an upstream host) dropped.
# Their hosts must also be in the relay allowlist (the same script writes
# data/relay_allowed_hosts.json, which iptv_relay.py reads).
def _load_dynamic_upstream_accounts() -> list[tuple[str, str, int, str, str]]:
    f = Path(__file__).resolve().parents[2] / "data" / "dynamic_upstream_accounts.json"
    try:
        rows = json.loads(f.read_text())
    except (OSError, ValueError):
        return []
    out: list[tuple[str, str, int, str, str]] = []
    for r in rows:
        try:
            label, host, port, user, pw = r
            out.append((str(label), str(host), int(port), str(user), str(pw)))
        except (ValueError, TypeError):
            continue
    return out


_seen_upstream = {(h, u) for _, h, _p, u, _pw in _upstream_ACCOUNTS}
for _row in _load_dynamic_upstream_accounts():
    if (_row[1], _row[3]) not in _seen_upstream:
        _upstream_ACCOUNTS.append(_row)
        _seen_upstream.add((_row[1], _row[3]))

# Residential relay — scripts/iptv_relay.py run on a laptop/Pi and exposed
# via Cloudflare Tunnel (localhost:8000). When IPTV_LOCAL_PROXY_URL is set,
# every stream URL we hand to the browser is rewritten to go through that
# tunnel (upstream providers block VPS IPs; the relay punches through from a
# residential address). Empty ⇒ direct-to-provider fallback.
_upstream_HOSTS: frozenset[str] = frozenset(h for _, h, *_ in _upstream_ACCOUNTS)

# ---------------------------------------------------------------------------
# Per-team dedicated NHL feeds. bgdc.live carries a "US : NHL <CITY> <NICK>"
# series for 28/32 teams. These get added as their own row in the broadcast
# slate (always shown — regular season AND playoffs), giving viewers a per-
# team fallback when the standard broadcast slate is thin (e.g. playoffs
# national-only filter, or an RSN our providers don't carry).
#
# Needle is matched as a case-insensitive substring against the raw channel
# title. The label is what the chip displays.
# ---------------------------------------------------------------------------
_TEAM_FEED_NEEDLES: dict[str, str] = {
    "ANA": "nhl anaheim ducks",
    "BOS": "nhl boston bruins",
    "BUF": "nhl buffalo sabres",
    "CGY": "nhl calgary flames",
    "CAR": "nhl carolina hurricanes",
    "CHI": "nhl chicago blackhawks",
    "COL": "nhl colorado avalanche",
    "CBJ": "nhl columbus blue jackets",
    "DAL": "nhl dallas stars",
    "DET": "nhl detroit red wings",
    "EDM": "nhl edmonton oilers",
    "FLA": "nhl florida panthers",
    "LAK": "nhl los angeles kings",
    "MIN": "nhl minnesota wild",
    "MTL": "nhl montreal canadiens",
    "NSH": "nhl nashville predators",
    "NJD": "nhl new jersey devils",
    "NYI": "nhl new york islanders",
    "NYR": "nhl new york rangers",
    "OTT": "nhl ottawa senators",
    "PHI": "nhl philadelphia flyers",
    "PIT": "nhl pittsburgh penguins",
    "SJS": "nhl san jose sharks",
    "SEA": "nhl seattle kraken",
    "STL": "nhl st louis blues",
    "TBL": "nhl tampa bay lightning",
    "TOR": "nhl toronto maple leafs",
    "UTA": "nhl utah mammoth",
    "VAN": "nhl vancouver canucks",
    "VGK": "nhl vegas golden knights",
    "WSH": "nhl washington capitals",
    "WPG": "nhl winnipeg jets",
}

_TEAM_FEED_LABELS: dict[str, str] = {
    "ANA": "Ducks Feed",
    "BOS": "Bruins Feed",
    "BUF": "Sabres Feed",
    "CGY": "Flames Feed",
    "CAR": "Hurricanes Feed",
    "CHI": "Blackhawks Feed",
    "COL": "Avalanche Feed",
    "CBJ": "Blue Jackets Feed",
    "DAL": "Stars Feed",
    "DET": "Red Wings Feed",
    "EDM": "Oilers Feed",
    "FLA": "Panthers Feed",
    "LAK": "Kings Feed",
    "MIN": "Wild Feed",
    "MTL": "Canadiens Feed",
    "NSH": "Predators Feed",
    "NJD": "Devils Feed",
    "NYI": "Islanders Feed",
    "NYR": "Rangers Feed",
    "OTT": "Senators Feed",
    "PHI": "Flyers Feed",
    "PIT": "Penguins Feed",
    "SJS": "Sharks Feed",
    "SEA": "Kraken Feed",
    "STL": "Blues Feed",
    "TBL": "Lightning Feed",
    "TOR": "Maple Leafs Feed",
    "UTA": "Mammoth Feed",
    "VAN": "Canucks Feed",
    "VGK": "Golden Knights Feed",
    "WSH": "Capitals Feed",
    "WPG": "Jets Feed",
}

# CBC regional outlets simulcast the national broadcast with local cut-ins,
# so they're a valid match for a "CBC" broadcast row in regular season only.
# Restricted to Canadian teams — US-side CBC variants don't exist.
_TEAM_CITY_FOR_CBC: dict[str, str] = {
    "MTL": "montreal",
    "TOR": "toronto",
    "OTT": "ottawa",
    "WPG": "winnipeg",
    "EDM": "edmonton",
    "CGY": "calgary",
    "VAN": "vancouver",
}


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
    # Pass through. No q param means the relay falls into `-c copy` mode (see
    # iptv_relay.py:_encode_args). XCIPTV and every native IPTV player on the
    # planet plays these upstreams smoothly because they DON'T transcode —
    # they just wrap raw MPEG-TS into a player-consumable container. We were
    # forcing q=720p which spent 30-50% of one CPU core per stream on
    # libx264 software encoding (relay falls back to libx264 when no
    # hardware encoder is available); on a multi-viewer residential box that
    # saturates the CPU and ffmpeg falls behind producing segments — surfacing
    # as mid-stream pause-and-load on the player. Passthrough is <2% CPU per
    # stream and matches XCIPTV's behaviour exactly.
    return f"{tunnel}/{endpoint}?u={quote(inner, safe='')}{tq}"


# upstream hosts that firewall the VPS IP. Channel enumeration for these has
# to go through the residential relay — same trick we already use for stream
# playback. Add a host here when a previously-working provider starts
# returning "All connection attempts failed" or 403 from the API box.
_VPS_BLOCKED_upstream_HOSTS: frozenset[str] = frozenset({
    "kstv.us",
})


def _upstream_fetch_url(upstream_url: str, host: str) -> str:
    """If the host blocks the VPS, route the JSON catalog fetch through the
    relay's /upstream-json endpoint (residential IP). Otherwise return as-is.
    """
    if host.lower() not in _VPS_BLOCKED_upstream_HOSTS:
        return upstream_url
    tunnel = (os.environ.get("IPTV_LOCAL_PROXY_URL") or "").rstrip("/")
    if not tunnel:
        return upstream_url  # no relay configured — best-effort direct fetch
    token = os.environ.get("IPTV_RELAY_TOKEN") or ""
    tq    = f"&t={quote(token, safe='')}" if token else ""
    return f"{tunnel}/upstream-json?u={quote(upstream_url, safe='')}{tq}"


async def _fetch_upstream_channels(
    label: str, host: str, port: int, username: str, password: str,
) -> list[dict]:
    """
    Fetch NHL-relevant channels from an upstream Codes panel.

    Tries (in order): get.php?type=m3u_plus&output=m3u8 → output=ts →
    player_api.php?action=get_live_streams. The last path is needed for panels
    like kstv.us where get.php is locked down (404) but player_api.php is
    open. Stream URLs are constructed as /live/<user>/<pass>/<id>.m3u8 (upstream
    convention). The relay's /m3u8 endpoint follows redirects to whatever
    CDN the panel hands back.

    Hosts in _VPS_BLOCKED_upstream_HOSTS firewall the API VPS IP, so all
    catalog fetches for those panels are routed through the relay's
    /upstream-json endpoint (residential IP) — see _upstream_fetch_url.
    """
    base = f"http://{host}:{port}"
    ua   = _BROWSER_HEADERS["User-Agent"]

    primary_fmt = "m3u8"
    secondary_fmt: str | None = "ts"
    try:
        async with _httpx_backup.AsyncClient(timeout=10.0, follow_redirects=True) as hx:
            probe = await hx.get(
                _upstream_fetch_url(
                    f"{base}/player_api.php?username={username}&password={password}",
                    host,
                ),
                headers={"User-Agent": ua},
            )
        if probe.status_code == 200:
            fmts = [str(f).lower() for f in
                    (probe.json().get("user_info") or {}).get("allowed_output_formats") or []]
            if "m3u8" in fmts and "ts" in fmts:
                primary_fmt, secondary_fmt = "m3u8", "ts"
            elif "m3u8" in fmts:
                primary_fmt, secondary_fmt = "m3u8", "ts"  # still try ts as last resort
            elif "ts" in fmts:
                primary_fmt, secondary_fmt = "ts", "m3u8"
            elif fmts:
                # Provider advertises something exotic (rtmp etc) — no path forward.
                return []
    except Exception:
        pass  # probe failure is not fatal — try m3u8 then ts

    async def _fetch_via_m3u(fmt: str) -> list[dict]:
        url = _upstream_fetch_url(
            f"{base}/get.php?username={username}&password={password}"
            f"&type=m3u_plus&output={fmt}",
            host,
        )
        try:
            async with _httpx_backup.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
                r = await hx.get(url, headers={"User-Agent": ua})
            if r.status_code != 200:
                return []
            text = r.text
        except Exception:
            return []

        out: list[dict] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            if line.startswith("#EXTINF"):
                name_m = _re_iptv.search(r'tvg-name="([^"]+)"', line)
                ch_name = name_m.group(1) if name_m else line.split(",")[-1].strip()
                stream_url = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if (
                    stream_url.startswith("http")
                    and any(k in ch_name.lower() for k in _IPTV_CHANNEL_KEYWORDS)
                ):
                    out.append({
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
        return out

    async def _fetch_via_player_api() -> list[dict]:
        """Last-resort: enumerate live streams via the JSON API and build
        upstream-convention URLs. Required for panels (kstv) that 404 get.php.
        """
        try:
            async with _httpx_backup.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
                r = await hx.get(
                    _upstream_fetch_url(
                        f"{base}/player_api.php?username={username}&password={password}"
                        "&action=get_live_streams",
                        host,
                    ),
                    headers={"User-Agent": ua},
                )
            if r.status_code != 200:
                return []
            streams = r.json()
        except Exception:
            return []
        if not isinstance(streams, list):
            return []

        out: list[dict] = []
        for s in streams:
            if not isinstance(s, dict):
                continue
            ch_name = (s.get("name") or "").strip()
            sid     = s.get("stream_id")
            if not ch_name or sid is None:
                continue
            if not any(k in ch_name.lower() for k in _IPTV_CHANNEL_KEYWORDS):
                continue
            # Prefer .m3u8 — relay's /m3u8 endpoint passthrough-rewrites cheaply
            # (no ffmpeg). Panels that only serve raw TS will redirect to a /ts
            # variant; ffmpeg transmux at the relay handles that fallback.
            stream_url = f"{base}/live/{username}/{password}/{sid}.m3u8"
            out.append({
                "title":      f"{ch_name} ({label})",
                "url":        _rewrite_iptv_url(stream_url),
                "source":     "upstream",
                "feed":       "iptv",
                "priority":   1,
                "embed_only": False,
            })
        return out

    results = await _fetch_via_m3u(primary_fmt)
    if not results and secondary_fmt and secondary_fmt != primary_fmt:
        results = await _fetch_via_m3u(secondary_fmt)
    if not results:
        results = await _fetch_via_player_api()
    return results


def _build_m3u_sources() -> list[str]:
    return [
        "https://raw.githubusercontent.com/phosani/tvpass/refs/heads/main/tvpasshd.m3u",
        "https://raw.githubusercontent.com/musicmashupstv-hue/tvpassplaylist/main/tvpassplaylist.m3u8",
        "https://raw.githubusercontent.com/jburg229/iptv-playlist/main/playlist.m3u",
    ]

_M3U_SOURCES = _build_m3u_sources()

# Approved IPTV provider hosts. The public GitHub M3U mirrors include URLs
# pointing at lots of providers we don't want (lunar.pm, tv14s.xyz, random
# regional CDNs); only URLs whose hostname matches — or is a subdomain of —
# one of these are accepted.
_APPROVED_IPTV_HOSTS: tuple[str, ...] = (
    "tvpass.org",
    "thetvapp.to",
    "ampztl.xyz",
    "an upstream host.ddns.net",
    "kstv.us",
    "bgdc.live",
    "an upstream host.an upstream host.co",
)


def _is_approved_iptv_url(u: str) -> bool:
    try:
        host = (urlparse(u).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == a or host.endswith("." + a) for a in _APPROVED_IPTV_HOSTS)


_IPTV_REFRESH_TASK: "asyncio.Task | None" = None


def _kick_iptv_refresh() -> None:
    """Spawn one background refresh of _IPTV_CACHE; no-op if one is already
    in flight. Used by the stale-while-revalidate path so the user never
    waits for a cold-start fan-out once the cache has data at all."""
    global _IPTV_REFRESH_TASK
    try:
        if _IPTV_REFRESH_TASK is not None and not _IPTV_REFRESH_TASK.done():
            return

        async def _refresh() -> None:
            try:
                fresh = await _build_iptv_channels()
                if fresh:
                    _IPTV_CACHE["data"] = fresh
                    _IPTV_CACHE["ts"] = _time_iptv.time()
            except Exception:
                pass

        _IPTV_REFRESH_TASK = asyncio.create_task(_refresh())
    except RuntimeError:
        # No running event loop — extremely unlikely from a request handler,
        # but worth defending against (e.g. import-time call from a script).
        pass


@app.get("/iptv-channels")
async def iptv_channels(force: bool = False):
    now = _time_iptv.time()
    have = bool(_IPTV_CACHE["data"])
    fresh = have and (now - _IPTV_CACHE["ts"] < _IPTV_TTL)

    if not force and have and fresh:
        return {"channels": _IPTV_CACHE["data"], "count": len(_IPTV_CACHE["data"]), "cached": True}

    if not force and have:
        # Stale-while-revalidate: data is past TTL but exists. Return it
        # immediately and refresh in the background. Hides the 30-60s
        # cold-start fan-out cost from every cache turnover.
        _kick_iptv_refresh()
        return {
            "channels": _IPTV_CACHE["data"],
            "count": len(_IPTV_CACHE["data"]),
            "cached": True,
            "stale": True,
        }

    # First request after process start (or force=True): pay the cost.
    channels = await _build_iptv_channels()
    # Guard against poisoning the cache with an empty/degraded cold build when a
    # transient upstream outage coincides with process start / TTL expiry. The
    # background refresher already guards (`if fresh:` in _kick_iptv_refresh) —
    # the cold path was the only unguarded writer, so a single bad fan-out here
    # could blank the whole channel pool for the full TTL. Keep last-known-good
    # instead, and serve it so BarnCentre/Lounge stay watchable.
    if channels or not _IPTV_CACHE["data"]:
        _IPTV_CACHE["data"] = channels
        _IPTV_CACHE["ts"] = now
    served = _IPTV_CACHE["data"]
    return {"channels": served, "count": len(served), "cached": not channels and bool(served)}


async def _src_streams_json() -> list[dict]:
    """Source 2: albinchristo04/tvpass streams.json — tvpass.org redirect URLs
    with fresh tokens. Each entry has:
      original_url: tvpass.org/live/<slug>/hd  ← use this (fresh token at stream time)
      stream_url:   thetvapp.to/hls/...?token=... ← stale within hours — do NOT use
    """
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://raw.githubusercontent.com/albinchristo04/tvpass/main/streams.json"
            )
            if r.status_code != 200:
                return out
            payload = r.json()
            entries = payload.get("channels", []) if isinstance(payload, dict) else payload
            for entry in entries:
                name_lower = (entry.get("name") or "").lower()
                original_url = entry.get("original_url") or ""
                stream_url   = entry.get("stream_url") or ""
                use_url = original_url if original_url.startswith("http") else stream_url
                if (
                    entry.get("status") == "working"
                    and use_url.startswith("http")
                    and any(k in name_lower for k in _IPTV_CHANNEL_KEYWORDS)
                ):
                    out.append({
                        "title": f"{entry['name']} (thetvapp)",
                        "url": use_url,
                        "source": "thetvapp",
                        "feed": "iptv",
                        "priority": 0,
                        "embed_only": False,
                    })
    except Exception:
        pass
    return out


async def _src_m3u_playlists() -> list[dict]:
    """Source 3: public M3U playlists. URLs are whitelisted via
    _is_approved_iptv_url so non-approved providers (lunar.pm, tv14s.xyz,
    random regional CDNs) never enter the pool — only ampztl / an upstream host /
    tvpass.org-family URLs the GitHub mirrors carry get through.

    Fires all 3 mirrors in parallel; one slow mirror no longer drags the
    others.
    """
    out: list[dict] = []

    async def _one(client: httpx.AsyncClient, m3u_url: str) -> list[dict]:
        inner: list[dict] = []
        try:
            r = await client.get(m3u_url)
            if r.status_code != 200:
                return inner
            lines = r.text.splitlines()
            i = 0
            while i < len(lines) - 1:
                line = lines[i]
                if line.startswith("#EXTINF"):
                    name_match = _re_iptv.search(r'tvg-name="([^"]+)"', line)
                    display = name_match.group(1) if name_match else line.split(",")[-1].strip()
                    url_line = lines[i + 1].strip()
                    if (
                        any(k in display.lower() for k in _IPTV_CHANNEL_KEYWORDS)
                        and url_line.startswith("http")
                        and _is_approved_iptv_url(url_line)
                    ):
                        inner.append({
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
            pass
        return inner

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            results = await asyncio.gather(
                *[_one(client, u) for u in _M3U_SOURCES],
                return_exceptions=True,
            )
        for r in results:
            if isinstance(r, list):
                out.extend(r)
    except Exception:
        pass
    return out


async def _src_upstream() -> list[dict]:
    """Source 4: upstream Codes accounts — fire all in parallel, merge results."""
    out: list[dict] = []
    try:
        results = await asyncio.gather(
            *[_fetch_upstream_channels(lbl, h, p, u, pw) for lbl, h, p, u, pw in _upstream_ACCOUNTS],
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, list):
                out.extend(res)
    except Exception:
        pass
    return out


async def _build_iptv_channels() -> list[dict]:
    """Fetch all IPTV sources and return the deduped channel list.

    Sources 2, 3, and 4 fan out concurrently via asyncio.gather — previously
    they ran one-after-another, so a slow GitHub mirror or hung upstream panel
    blocked the others. Worst-case cold-start is now max(source) instead of
    sum(sources): roughly 30s instead of 60s+.
    """
    channels: list[dict] = []

    # Source 1: static tvpass.org slugs (in-memory, instant).
    for name, slug, quality in _TVPASS_CHANNELS:
        channels.append({
            "title": name,
            "url": f"https://tvpass.org/live/{slug}/{quality}",
            "source": "tvpass",
            "feed": "iptv",
            "priority": 1,
            "embed_only": False,
        })

    # Sources 2, 3, 4 in parallel.
    src_results = await asyncio.gather(
        _src_streams_json(),
        _src_m3u_playlists(),
        _src_upstream(),
        return_exceptions=True,
    )
    for r in src_results:
        if isinstance(r, list):
            channels.extend(r)

    # Deduplicate by exact URL — keeps first occurrence (highest-priority source wins).
    seen: set[str] = set()
    deduped: list[dict] = []
    for ch in channels:
        if ch["url"] not in seen:
            seen.add(ch["url"])
            deduped.append(ch)
    return deduped


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
    # laptop's current Cloudflare Tunnel without leaking the raw token value.
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
    # US — CBS Sports (placed right before the French block)
    "CBS Sports", "CBS Sports Network",
    # Canadian French — at the back, grouped by network
    "RDS", "RDS 2", "RDS INFO",
    "TVA Sports",
]

# Per-channel upstream-host blocklist with TTL. Entries map host → epoch
# timestamp at which the host was last flagged as broken; an entry is "active"
# only while now - timestamp < _BARNCENTRE_BLOCKLIST_TTL_S. After that it
# auto-expires and the host re-enters the candidate pool. The loop-detector
# probe (`_probe_manifest_loop` below) re-adds fresh entries whenever a host
# is caught replaying segments, so a transient upstream hiccup doesn't
# permanently demote a fast source.
#
# 2026-05-04: TSN1 an upstream host served a ~30s loop; ampztl-b webtv1847 carried
# the same upstream feed and inherited the bug. Both timestamps are 24h+ old
# now, so they auto-expire on the next /barncentre-channels request — if the
# upstream is still broken, the loop probe re-adds them with a fresh stamp.
_BARNCENTRE_BLOCKLIST_TTL_S = 24 * 60 * 60  # 24h
_BARNCENTRE_HOST_BLOCKLIST: dict[str, dict[str, float]] = {
    "TSN1": {
        "an upstream host.ddns.net": 1714780000.0,  # 2026-05-04
        "ampztl.xyz":          1714810000.0,  # 2026-05-04
    },
}

# Permanent host-blocklist for BarnCentre / Lounge. These are Bob-curated
# entries: "this provider's slot for this channel is bad and we don't want
# auto-recovery." Does NOT apply to /game-iptv-streams — game broadcasts
# match against a wider candidate pool and benefit from having ampztl as a
# fallback even when its BarnCentre version is unwatchable. The TTL blocklist
# above is the auto-detection / self-healing layer; this one is the manual
# override that doesn't expire.
#
# 2026-05-05: ampztl.xyz removed from TSN1 (Bob — picture buffers worse than
# the an upstream host / bgdc alternatives even after the player buffer tuning).
_BARNCENTRE_PERMANENT_HOST_BLOCKLIST: dict[str, set[str]] = {
    "TSN1": {"ampztl.xyz"},
}


def _active_blocklist(name: str) -> set[str]:
    """Return hosts currently blocked for `name` — TTL entries that haven't
    expired, unioned with the permanent (Bob-curated) blocklist.

    Mutates the TTL dict to evict expired entries so it self-cleans over
    time. Safe to call from async contexts (dict update is sync and
    contention-free in CPython).
    """
    entries = _BARNCENTRE_HOST_BLOCKLIST.get(name)
    blocked: set[str] = set()
    if entries:
        now = _time.time()
        expired = [h for h, t in entries.items() if now - t >= _BARNCENTRE_BLOCKLIST_TTL_S]
        for h in expired:
            entries.pop(h, None)
        blocked.update(entries.keys())
    permanent = _BARNCENTRE_PERMANENT_HOST_BLOCKLIST.get(name)
    if permanent:
        blocked.update(permanent)
    return blocked


def _add_to_blocklist(name: str, host: str) -> None:
    """Mark `host` as broken for channel `name`. Resets the TTL window."""
    if not host:
        return
    _BARNCENTRE_HOST_BLOCKLIST.setdefault(name, {})[host] = _time.time()


def _is_tvpass(ch: dict) -> bool:
    """Return True if a candidate channel originates from tvpass.org.

    tvpass entries are dropped wholesale (Bob, 2026-05-05): they're slow,
    flake on token refresh, and the better thetvapp tokens we get via
    direct thetvapp playlists cover every channel they did. Catches both
    the source-tagged static slugs and any GitHub-mirrored entries whose
    URL still points back at tvpass.org even when source != "tvpass".
    """
    if ch.get("source") == "tvpass":
        return True
    return "tvpass.org" in (ch.get("url", "") or "")


# Channels whose source MPEG-TS has irregular keyframe spacing — the
# encoder swings GOP length around (e.g. 2s/4s alternating), so under
# passthrough (`-c copy`) the relay's HLS muxer can only split at keyframes
# and produces variable-duration segments (4.0s/2.0s/4.0s/2.0s ...). hls.js
# tolerates that poorly: the live edge advances unevenly, the watchdog's
# drift heuristic fires spuriously, and users see what looks like buffering
# but is actually the player resyncing every few seconds.
#
# Forcing `q=720p` makes ffmpeg re-encode with `-g 60 -keyint_min 60`
# (deterministic 2s GOP) which aligns cleanly with `-hls_time 3` → steady 3s
# segments. The historical comment in `_rewrite_iptv_url` notes this was
# removed globally because libx264 software encoding burned 30-50% of one
# core per stream on the residential relay; we re-add it ONLY for channels
# that actually need it. With ~5 TSN feeds and at most a handful of
# concurrent viewers per feed, total CPU is bounded.
#
# Live evidence (2026-05-05): TSN1 manifest from an upstream host `/play/.../ts`
# emitted alternating 4.004/2.002s segments; Sportsnet East from the same
# provider emitted steady 3.000s segments. Same provider, same relay path,
# same encoder config — the only difference is the upstream feed's GOP.
_RECODE_CHANNELS: set[str] = {"TSN1", "TSN2", "TSN3", "TSN4", "TSN5"}


def _needs_recode(ch_name: str = "", broadcast_code: str = "") -> bool:
    """Decide whether a channel/broadcast belongs to the irregular-GOP set.

    Channel-side callers (BarnCentre / Lounge) pass a canonical name like
    "TSN1". Game-page callers know only the NHL broadcast code ("TSN", "TSN1",
    sometimes the network display string) — those land here too, where any
    "TSN..." prefix qualifies.
    """
    if ch_name in _RECODE_CHANNELS:
        return True
    bc = (broadcast_code or "").upper()
    return bc.startswith("TSN")


def _apply_recode(url: str, ch_name: str = "", *, force: bool = False) -> str:
    """Append `&q=480p` to a relay /hls URL when the channel needs re-encoding.

    Leaves /m3u8 URLs (already-HLS sources, no transmux happens) and URLs
    that already carry a q= parameter untouched. `force=True` lets callers
    that already determined the recode requirement (e.g. game page knowing the
    broadcast code is TSN) bypass the channel-name lookup.

    480p (1.2 Mbps) chosen over 720p (2.5 Mbps) because TSN's source is
    a high-bitrate 1080i feed: live tests against the residential relay
    showed q=720p ffmpeg sessions timing out before producing the first
    manifest ("ffmpeg did not produce manifest in time"), while q=480p
    emitted steady 3.003s segments cleanly. The CPU on the residential
    relay can't keep up with 720p re-encode of TSN's source. The watchdog
    + auto-downgrade in the player will still flip 720p→480p mid-stream
    on other channels; this just makes 480p the *starting* tier for TSN.
    """
    if not (force or ch_name in _RECODE_CHANNELS):
        return url
    if "q=" in url:
        return url
    if "localhost:8000/hls?" not in url and "/hls?u=" not in url:
        return url
    return url + ("&q=480p" if "?" in url else "?q=480p")

import re as _re_bc

def _normalize_ch(title: str) -> str:
    """Strip source suffix and HD/SD qualifier for channel name matching.

    Handles four naming styles:
    - tvpass/thetvapp: "Sportsnet East HD"        → "sportsnet east"
    - upstream pipe:    "US | Fox Sports 1 HD"      → "fs1"
    - upstream dash:    "CA - SPORTSNET EAST"       → "sportsnet east"
    - upstream colon:   "CA: RDS (FR)"              → "rds"
    - language flag:  "CA-FR | TVA SPORTS 1"      → "tva sports"
    - upstream label:   "RDS HD (tv14s)"            → "rds"
    """
    t = title.strip()
    # Strip upstream source labels we append: "(tv14s)", "(an upstream host)", etc.
    t = _re_bc.sub(r"\s*\([a-z0-9._-]{3,30}\)\s*$", "", t, flags=_re_bc.I)
    # Strip playlist/thetvapp suffixes
    t = _re_bc.sub(r"\s*\(thetvapp\)|\s*\(playlist\)", "", t, flags=_re_bc.I)
    # Strip trailing quality flags: HD, SD, FHD, UHD, (B), (R), (E), (D)
    t = _re_bc.sub(r"\s+(?:fhd|uhd|hd|sd)\s*$", "", t, flags=_re_bc.I)
    t = _re_bc.sub(r"\s*\([a-z]\)\s*$", "", t, flags=_re_bc.I)
    # Strip upstream country/language prefixes. The optional language-code group
    # uses [-_]+ (no whitespace) so it matches "CA-FR" / "CA_EN" but does NOT
    # accidentally eat into "CA - SPORTSNET" (where the dash is just a
    # separator, not a language separator). Without this guard the regex was
    # munching "CA - SP" off "CA - SPORTSNET EAST" and breaking matches.
    t = _re_bc.sub(r"^(?:CA|US|UK)(?:[-_]+[A-Z]{2})?\s*(?:\([A-Z]{2}\))?\s*[\|:\-]?\s*", "", t, flags=_re_bc.I)
    # Strip a leading numeric sort prefix ("02. ", "01) ") and an UPPERCASE
    # provider-category pipe tag ("DEP | ", "NOT | ", "PRIME| ", "2MB | "). The
    # category strip is uppercase-only so mixed-case real names ("Fox Sports |…")
    # are never eaten. Added 2026-06-27 to match labels like "02. DAZN1" and
    # "DEP | Bein Sports 02 (USA)".
    t = _re_bc.sub(r"^\s*\d{1,3}[.\)]\s*", "", t)
    t = _re_bc.sub(r"^[A-Z0-9]{2,5}\s*\|\s*", "", t)
    # Strip trailing language flag "(FR)" / "(EN)" / "(ES)" / "(DE)" — lunar
    # keeps this after "CA:" prefix consumption.
    t = _re_bc.sub(r"\s*\((?:FR|EN|ES|DE)\)\s*$", "", t, flags=_re_bc.I)
    # Normalize "TVA SPORTS 1" → "TVA SPORTS": tv14s numbers the primary feed.
    t = _re_bc.sub(r"(?i)\btva\s+sports\s+1\s*$", "TVA Sports", t)
    # Strip remaining leading/trailing punctuation. ampztl decorates titles
    # with leading "." (e.g. ".NHL Network ✤", ".RDS ☆") which would otherwise
    # leave the dot in front and break matcher equality.
    t = t.strip(" |:-").lstrip(".")
    # Strip trailing ampztl decoration glyphs (e.g. "TSN 1 ƒ" → "TSN 1",
    # "RDS ☆" → "RDS"). Without this, the digit-collapse below misses
    # decorated titles because its $-anchor sees the glyph before the digit.
    # Uses an ASCII-only character class — Python's \w is unicode-aware so
    # ƒ (U+0192) and accented letters would otherwise be treated as "words"
    # and skipped. Requires preceding whitespace, so embedded accents in
    # legitimate words ("Société") are never touched.
    t = _re_bc.sub(r"(?:\s+[^a-zA-Z0-9_\s]+)+\s*$", "", t)
    # Collapse digit-spacing for known concatenated channel names so providers
    # emitting "TSN 1" / "ESPN 2" / "FS 1" match _BARNCENTRE_CHANNEL_NAMES
    # entries like "TSN1" / "ESPN2" / "FS1". Anchored to whole-string after
    # prefixes/suffixes are stripped so unrelated text never gets mangled.
    t = _re_bc.sub(r"^(tsn|espn|fs)\s+(\d+)$", r"\1\2", t, flags=_re_bc.I)
    # "Fox Sports 1" / "FOX SPORTS 2" → "fs1" / "fs2". kstv & most US providers
    # spell out "Fox Sports", but BarnCentre's slate uses the FS contraction.
    t = _re_bc.sub(r"(?i)^fox\s*sports\s*(\d+)$", r"FS\1", t)
    # DAZN / beIN number normalization (added 2026-06-27): providers label these
    # "DAZN1", "Bein Sports 02" — normalize to the spaced, zero-stripped form so
    # they match "DAZN 1" / "beIN Sports 2". Anchored to the whole string.
    t = _re_bc.sub(r"(?i)^dazn\s*0?(\d+)$", r"dazn \1", t)
    t = _re_bc.sub(r"(?i)^be\s?in\s*sports?\s*0?(\d+)$", r"bein sports \1", t)
    return t.strip().lower()


_SHORT_FR_DISPLAYS = {"rds", "rds2", "rds info", "tva sports"}

# Foreign-feed guard. Scraped upstream providers carry Latin-American / Brazilian /
# other-region duplicates of US channels ("ESPN Brasil 2", "ESPN HD OP 2" =
# opción) that would otherwise match the bare US channel name via the prefix
# rule and put Portuguese/Spanish audio on the US channel. Curated channels are
# US/CA English (RDS/TVA Sports are French-CANADIAN and matched separately, and
# contain none of these markers). If a title carries one of these region/language
# markers, it is NOT one of our channels — reject it. Word-ish markers use a
# boundary so "op" can't fire inside other words.
_FOREIGN_FEED_RE = _re_bc.compile(
    r"\b(?:brasil|brazil|latino|latin|deportes|espa(?:n|ñ)ol|espa(?:n|ñ)a|"
    # NOTE: no French marker here on purpose — Quebec French channels (TVA, LCN,
    # RDI, Noovo, RDS) are Canadian and wanted. France-the-country feeds are
    # excluded by not adding their channel names, not by this guard.
    r"m[eé]xico|portugu|arabic|arab|t[uü]rk|italia|italiano|deutsch|"
    r"colombia|argentina|venezuela|chile|per[uú]|opci[oó]n|op)\b",
    _re_bc.I,
)


def _ch_matches(raw_title: str, ch_name: str) -> bool:
    """Return True if an IPTV channel title matches our BarnCentre channel name."""
    norm  = _normalize_ch(raw_title)
    want  = ch_name.lower()
    # Reject foreign-region/language feeds outright (e.g. "ESPN Brasil",
    # "ESPN HD OP 2") so they don't shadow the US channel — unless the curated
    # name itself contains the marker (none currently do).
    if _FOREIGN_FEED_RE.search(norm) and not _FOREIGN_FEED_RE.search(want):
        return False
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


# Concurrency cap on the verifier. Without this, `gather()` spawns ~100
# probes that all queue behind httpx's default 5-connection pool — adding
# 20-25s of head-of-line blocking on cold start despite "parallel". 20
# concurrent slots maxes out a typical pool budget without trashing it.
_VERIFY_SEM: "asyncio.Semaphore | None" = None


def _get_verify_sem() -> "asyncio.Semaphore":
    global _VERIFY_SEM
    if _VERIFY_SEM is None:
        _VERIFY_SEM = asyncio.Semaphore(20)
    return _VERIFY_SEM


_verify_cache: dict[str, tuple[bool, float]] = {}
_VERIFY_CACHE_TTL_OK = 300.0   # 5 min for alive-confirmed URLs
_VERIFY_CACHE_TTL_BAD = 60.0   # 1 min for dead URLs (revisit faster on outages)


async def _verify_stream_alive(url: str, timeout: float = 4.0) -> bool:
    """Follow redirects and confirm the URL resolves to a reachable HLS stream.

    Result is cached per-URL (5 min OK, 1 min bad) to keep the channel-list
    build from re-probing the same chip dozens of times. Critical: each
    /lounge/live-channels build calls this for every primary across ~50
    channels; without caching, each request fans out 50+ probes (some
    spawning relay-side ffmpeg) and pushes the worker over the 2 GB
    VPS budget, OOM-killing the worker mid-response.
    """
    import time as _t_vc
    now_ts = _t_vc.time()
    cached = _verify_cache.get(url)
    if cached is not None:
        ok, ts = cached
        ttl = _VERIFY_CACHE_TTL_OK if ok else _VERIFY_CACHE_TTL_BAD
        if now_ts - ts < ttl:
            return ok

    result = await _verify_stream_alive_uncached(url, timeout)
    _verify_cache[url] = (result, now_ts)
    return result


async def _verify_stream_alive_uncached(url: str, timeout: float = 4.0) -> bool:
    """Original verifier — bypasses the cache. Internal use only.

    For tvpass.org/live/* URLs: follow 302 → check final URL is m3u8 + HTTP 200.
    For direct m3u8 URLs: do a lightweight HEAD/GET.
    Returns False on timeout, 404/403, or non-m3u8 destination.

    Relay /hls URLs: probe the relay endpoint itself, not the inner upstream URL.
    The whole point of the relay is that the VPS can't reach the upstream
    provider directly — probing the inner URL from here always fails (IP-banned)
    and incorrectly marks every working an upstream host-routed channel as offline.
    The relay returns a tiny manifest (≤1KB), so the cost is negligible.

    Timeout default is 4s for non-relay URLs. Relay /hls URLs get 8s — the
    relay's ffmpeg cold-start can take 5-7s on first probe of the day, and
    the previous 4s budget was false-flagging working channels (TSN2, TSN3,
    NHL Network, FanDuel) as offline. Combined with _VERIFY_SEM(20) this
    keeps cold-cache verifier latency bounded; once warm, probes return
    in ~50ms.
    """
    # Relay /hls calls block until ffmpeg produces the first manifest, up to
    # HLS_STARTUP_TIMEOUT_S=15s on the relay side. Our verifier needs >15s of
    # headroom or it times out before the relay decides — and we used to
    # treat that timeout as "optimistically alive," which is exactly how
    # broken upstreams (TSN's an upstream host slot today) kept getting picked as
    # primary_url. 18s gives the relay a 3s buffer to actually return its
    # 502 verdict so we can promote a verified alternate.
    if "/hls?" in url and "u=" in url:
        timeout = max(timeout, 18.0)
    # Relay /m3u8 wraps an upstream HLS manifest through the residential
    # tunnel: API → CF tunnel → relay → upstream provider. Each hop adds
    # 50-200ms; cold-cache the cumulative latency can flirt with the 4s
    # default. an upstream host TSN1 routinely measures ~1s warm but spiked over
    # 4s on the build that just deployed (verifier rejected it, alternate
    # got promoted to bgdc instead). 10s headroom keeps an upstream host eligible
    # without slowing the response noticeably (responses still cap at
    # cold ~2s in practice).
    elif "/m3u8?" in url and "u=" in url:
        timeout = max(timeout, 10.0)
    sem = _get_verify_sem()
    async with sem:
        return await _verify_stream_alive_inner(url, timeout)


async def _verify_stream_alive_inner(url: str, timeout: float) -> bool:
    import httpx as _hx_v

    # Relay /hls passthrough — probe the relay URL itself. We verify the
    # response is an actual HLS media playlist with at least one segment
    # line (#EXTINF or #EXT-X-MEDIA-SEQUENCE). A 200 with just `#EXTM3U`
    # passed the old "non-empty body" check but means ffmpeg never produced
    # a real segment. Strict on real failures (502 ffmpeg-died, 503
    # manifest-not-yet) → falls through to the verified-alternate path in
    # _build_channel. No more optimistic-on-timeout: with the 18s budget
    # set in the wrapper, a timeout here means the upstream is genuinely
    # unreachable, not a slow cold-start.
    if "/hls?" in url and "u=" in url:
        try:
            async with _hx_v.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                headers={
                    "User-Agent": "Grtzky-StreamVerifier/1.0",
                    "Range":      "bytes=0-2047",
                },
            ) as cl:
                r = await cl.get(url)
            if r.status_code not in (200, 206) or not r.text:
                return False
            body = r.text
            return ("#EXTINF" in body) or ("#EXT-X-MEDIA-SEQUENCE" in body)
        except Exception:
            return False

    try:
        async with _hx_v.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": "Grtzky-StreamVerifier/1.0",
                "Referer":    "https://tvpass.org/",
                "Origin":     "https://tvpass.org",
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
        # tvpass dropped 2026-05-05; the optimistic fallback that used to
        # live here ("trust tvpass on transient timeout") is gone. Strict
        # for everything else.
        return False


# ── Loop-detector probe ──────────────────────────────────────────────────────
# Strict liveness ("the manifest returns 200") doesn't catch the failure mode
# we care most about: a feed that auth-passes and serves bytes but is replaying
# the same ~30s of content forever. We see this when an upstream provider's
# encoder dies mid-stream and the panel keeps the session "alive" with cached
# segments — every probe says "fine," but viewers see a stuck or looping
# picture. The fix that prompted this rewrite was a hard-coded TSN1 host
# blocklist; the probe below replaces that with auto-detection so any channel
# benefits, not just the one Bob noticed.
#
# Mechanism: pull the manifest twice, 30s apart, and parse #EXT-X-MEDIA-SEQUENCE.
# In a healthy live stream the sequence number advances monotonically as new
# segments are produced. Equal-or-regressed seq across a 30s window means the
# manifest is stale (replay). When detected, the upstream host is added to
# `_BARNCENTRE_HOST_BLOCKLIST` with a fresh timestamp; the TTL helper above
# auto-evicts after 24h so a transient hiccup doesn't permanently demote a
# fast source.
#
# Cost-bounding: per-URL probe is cached for `_LOOP_PROBE_INTERVAL_S`. With
# ~25 BarnCentre channels and a 6h interval, the relay sees at most one probe
# pair per channel every six hours — negligible.
_LOOP_PROBE_INTERVAL_S = 6 * 60 * 60
_LOOP_PROBE_GAP_S      = 30
_loop_probe_cache: dict[str, float] = {}


def _extract_upstream_host(url: str) -> str:
    """Return the real provider host behind a candidate URL.

    Relay-wrapped URLs (`localhost:8000/hls?u=<urlencoded-upstream>`) hide
    the actual provider; pull it out of the `u=` query param. Non-relay URLs
    are returned as-is. Returns empty string on parse failure so callers can
    short-circuit cheaply.
    """
    if not url:
        return ""
    try:
        if "localhost:8000" in url and "u=" in url:
            from urllib.parse import parse_qs as _pq, unquote as _uq
            inner = _uq(_pq(urlparse(url).query).get("u", [""])[0])
            return urlparse(inner).hostname or ""
        return urlparse(url).hostname or ""
    except Exception:
        return ""


async def _read_media_sequence(url: str, timeout: float = 10.0) -> int | None | str:
    """GET the HLS manifest and return its `#EXT-X-MEDIA-SEQUENCE` value.

    Returns:
      - int  : the sequence number (healthy manifest)
      - "ffmpeg_dead" : relay returned 502 with "ffmpeg did not produce" (a
        broken upstream — quarantinable failure even on a single probe)
      - None : transient (timeout, master playlist, missing tag) — skip
    """
    import httpx as _hx_l
    try:
        async with _hx_l.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Grtzky-LoopProbe/1.0"},
        ) as cl:
            r = await cl.get(url)
        # Relay's 502 body for a stuck ffmpeg session looks like
        # `{"detail":"ffmpeg did not produce manifest in time"}` — when we
        # see that, the upstream feed itself is broken (the relay can't
        # produce segments because the source doesn't deliver). Treat this
        # as a hard, single-probe quarantine signal.
        if r.status_code in (502, 504) and "ffmpeg did not produce" in (r.text or ""):
            return "ffmpeg_dead"
        if r.status_code != 200 or not r.text:
            return None
        for raw in r.text.splitlines():
            line = raw.strip()
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None
        return None
    except Exception:
        return None


async def _probe_manifest_loop(url: str, channel_name: str) -> None:
    """Detect a looping manifest and quarantine its upstream host.

    Pulls the manifest twice `_LOOP_PROBE_GAP_S` apart and compares
    `#EXT-X-MEDIA-SEQUENCE`. Equal-or-regressed seq → host added to
    `_BARNCENTRE_HOST_BLOCKLIST` for 24h. Cached per-URL so a single page
    load doesn't re-probe the same channel multiple times.
    """
    if not url or not channel_name:
        return
    now = _time.time()
    last = _loop_probe_cache.get(url, 0.0)
    if now - last < _LOOP_PROBE_INTERVAL_S:
        return
    _loop_probe_cache[url] = now  # claim the slot regardless of outcome

    def _quarantine(reason: str) -> None:
        host = _extract_upstream_host(url)
        if host:
            _add_to_blocklist(channel_name, host)
            # Drop the cache so the next page load re-evaluates from
            # scratch; if the host recovers before the TTL expires we want
            # fresh data when it re-enters the candidate pool.
            _loop_probe_cache.pop(url, None)
        _ = reason  # kept for future logging once we wire structured logs

    seq1 = await _read_media_sequence(url)
    if seq1 == "ffmpeg_dead":
        _quarantine("ffmpeg_dead_probe1")
        return
    if seq1 is None:
        return
    await asyncio.sleep(_LOOP_PROBE_GAP_S)
    seq2 = await _read_media_sequence(url)
    if seq2 == "ffmpeg_dead":
        _quarantine("ffmpeg_dead_probe2")
        return
    if seq2 is None:
        return
    if isinstance(seq1, int) and isinstance(seq2, int) and seq2 > seq1:
        return  # advancing healthily
    _quarantine("seq_not_advancing")


_QUALITY_TAG_RE = _re_iptv.compile(r"\b(4K|UHD|FHD|HD|SD)\b", _re_iptv.I)


def _quality_score(title: str, url: str) -> int:
    """Higher score = lower quality (sorts first = best). Two signals:

    1. **Title tag** — providers tag titles with 4K / UHD / FHD / HD / SD.
       UHD ≈ 4K (typically 2160p), FHD = 1080p, HD = 720p, SD ≤ 480p. An
       untagged title is treated as HD-equivalent (most providers default
       to HD without saying so). 4K event-only feeds are scored as UHD —
       they're great when the event is live, otherwise they're a black
       screen, so we don't want them universally first.

    2. **Relay quality knob** — relay URLs include `?q=720p` / `q=480p` /
       (default) `q=passthrough`. Passthrough = original encode, no
       re-transcode = best quality + lowest CPU. 720p is a deliberate
       downgrade we apply mid-stream when bandwidth tanks, so a URL that
       arrives already pinned to 720p is worse than passthrough.
    """
    t_score = 4  # default: untagged ≈ HD
    m = _QUALITY_TAG_RE.search(title)
    if m:
        tag = m.group(1).upper()
        t_score = {"UHD": 0, "4K": 1, "FHD": 2, "HD": 3, "SD": 5}.get(tag, 4)

    if "q=480p" in url:
        q_score = 2
    elif "q=720p" in url:
        q_score = 1
    else:
        q_score = 0  # passthrough or non-relay direct manifest

    # Title tag is a stronger signal than the relay knob (a UHD chip pinned
    # to 720p still streams a much higher source than an SD chip at
    # passthrough), so weight it more.
    return t_score * 10 + q_score


def _sort_by_url_priority(candidates: list[dict]) -> list[dict]:
    """Sort channel candidates for initial game-page playback.

    Three-axis sort. Primary axis (Bob, 2026-05-18): an upstream host candidates
    win unconditionally — promoted to "main link" for the game page so the
    primary chip click always lands on an upstream host.an upstream host.co when present.
    Secondary axis: stream quality (UHD > FHD > untagged ≈ HD > 480p-pinned
    > SD). Tertiary axis: host reliability (kstv, an upstream host, ampztl,
    tvpass, thetvapp, other untested upstream, other) as the final
    tiebreaker among same-quality non-an upstream host candidates.

    kstv (puny243) returned auth=0 as of 2026-05-04 (Bob's premium panel
    appears to have lapsed). It still leads the host-priority tier so when
    the account comes back online its candidates re-enter the top of each
    quality bucket without code changes.
    """
    def _host_prio(m: dict) -> int:
        u = m["url"]
        if "kstv.us" in u:
            return 0
        if "an upstream host" in u:
            return 1
        if "an upstream host" in u:
            return 1
        if "ampztl" in u:
            return 2
        if "tvpass.org/live" in u:
            return 3
        if "thetvapp.to" in u:
            return 4
        if any(h in u for h in ("rexmaximl.shop", "anadolutv.shop", "izletvhd.xyz")):
            return 5
        return 6

    def _key(m: dict) -> tuple[int, int, int]:
        is_an upstream host = 0 if "an upstream host" in m["url"] else 1
        return (is_an upstream host, _quality_score(m.get("title", ""), m["url"]), _host_prio(m))

    return sorted(candidates, key=_key)


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
_NBA_NETWORK_TO_CHANNEL: dict[str, str | None] = {
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
_BARNCENTRE_REFRESH_TASK: "asyncio.Task | None" = None


def _kick_barncentre_refresh() -> None:
    """Spawn one background rebuild of _barncentre_cache; no-op if one is
    already in flight. Mirrors _kick_iptv_refresh — used by the
    stale-while-revalidate path so the user never waits for the 50-65s
    cold fan-out (NHL + ESPN + MLB + NBA + IPTV sources + ~100 verifier probes)
    once the cache has data at all.
    """
    global _BARNCENTRE_REFRESH_TASK
    try:
        if _BARNCENTRE_REFRESH_TASK is not None and not _BARNCENTRE_REFRESH_TASK.done():
            return

        async def _refresh() -> None:
            try:
                await _build_barncentre_payload()
            except Exception:
                pass

        _BARNCENTRE_REFRESH_TASK = asyncio.create_task(_refresh())
    except RuntimeError:
        pass


@app.get("/barncentre-channels")
async def barncentre_channels() -> dict:
    """Return curated, verified channel list for BarnCentre with today's NHL programs.

    Stale-while-revalidate: a hit on stale data (>1h) returns cached payload
    immediately and rebuilds in the background. First-ever request after
    process restart still pays the full cost, but every later cache turnover
    is invisible to users.
    """
    import time as _t_bc
    now_ts = _t_bc.time()
    have   = _barncentre_cache["data"] is not None
    fresh  = have and (now_ts - _barncentre_cache["ts"] < _BARNCENTRE_TTL)

    if have and fresh:
        return {"channels": _barncentre_cache["data"], "cached": True}

    if have:
        _kick_barncentre_refresh()
        return {"channels": _barncentre_cache["data"], "cached": True, "stale": True}

    # Cold cache — pay the full cost.
    return await _build_barncentre_payload()


async def _build_barncentre_payload() -> dict:
    """Full builder for /barncentre-channels. Extracted so SWR + startup
    pre-warm can both call it. Writes _barncentre_cache and returns the
    response payload.
    """
    import time as _t_bc
    import asyncio as _aio_bc
    now_ts = _t_bc.time()

    # ── 1. Full IPTV channel list (shared 1-hr cache) ─────────────────────────
    iptv_result  = await iptv_channels()
    all_channels: list[dict] = iptv_result.get("channels", [])
    # Drop tvpass entirely (2026-05-05). See `_is_tvpass` for rationale.
    all_channels = [ch for ch in all_channels if not _is_tvpass(ch)]

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
                        display = cast(str, _BROADCAST_CODE_MAP.get(code, code))
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
    from urllib.parse import parse_qs as _bl_parse_qs, unquote as _bl_unquote
    def _candidate_upstream_host(u: str) -> str:
        if "localhost:8000" in u and "u=" in u:
            try:
                inner = _bl_unquote(_bl_parse_qs(urlparse(u).query).get("u", [""])[0])
                return urlparse(inner).hostname or ""
            except Exception:
                return ""
        return urlparse(u).hostname or ""

    for name, cands in list(channel_candidates.items()):
        seen: set[str] = set()
        unique: list[dict] = []
        for m in cands:
            if m["url"] not in seen:
                seen.add(m["url"])
                unique.append(m)
        blocked = _active_blocklist(name)
        if blocked:
            unique = [c for c in unique if _candidate_upstream_host(c["url"]) not in blocked]
        channel_candidates[name] = _sort_by_url_priority(unique)

    # ── 4. Verify each channel's primary stream in parallel ───────────────────
    async def _build_channel(ch_name: str, candidates: list[dict]) -> dict:
        """Pick primary stream + run liveness check; return channel dict.

        Priority hard-bias: any upstream candidate (an upstream host / ampztl) wins
        primary if it exists, regardless of verifier outcome. The relay's
        ffmpeg cold start (~4-5s) can blow past the 10s verifier window on the
        first probe of the day, which previously demoted an upstream host below
        tvpass. We trust _sort_by_url_priority's field-tested ordering and
        only use the verifier to decide the `online` badge.
        """
        def _is_upstream(u: str) -> bool:
            return ("an upstream host" in u) or ("ampztl" in u)

        # BarnCentre primary preference (Bob, 2026-05-05): an upstream host leads.
        # an upstream host.an upstream host.co has been the most reliable provider in live
        # tests today (TSN1 picture, Sportsnet East alternate). an upstream host +
        # ampztl remain in the upstream-bias chain as fallback when an upstream host
        # isn't available for a given channel. The verifier-promotion path
        # below still applies, so a stale an upstream host slot doesn't get stuck.
        # NOTE: this only affects /barncentre-channels — game page and
        # /lounge/* keep the existing an upstream host/ampztl-first ordering.
        an upstream host_cands = [c for c in candidates if "an upstream host.an upstream host.co" in c["url"]]
        if an upstream host_cands:
            chosen = an upstream host_cands[0]["url"]
        else:
            upstream_cands = [c for c in candidates if _is_upstream(c["url"])]
            chosen = (upstream_cands[0] if upstream_cands else candidates[0])["url"]

        # Run liveness check on the chosen primary. If it's alive, keep it.
        # On a real failure (502 ffmpeg-died, 503 manifest-not-yet, body
        # missing #EXTINF), fall through to a verified alternate AND
        # promote it to primary_url. Walk DISTINCT upstream hosts rather
        # than just `candidates[:3]` — when one provider has multiple
        # accounts (3× an upstream host variants on top of the candidate list),
        # iterating by index can leave the working alternate (an upstream host at
        # index 4+) untried. Capped at 6 hosts so verification never blocks
        # the response indefinitely.
        verified_primary: str | None = None
        if await _verify_stream_alive(chosen):
            verified_primary = chosen
        else:
            tried_hosts: set[str] = set()
            chosen_host = _candidate_upstream_host(chosen)
            tried_hosts.add(chosen_host)
            for cand in candidates:
                if len(tried_hosts) >= 6:
                    break
                if cand["url"] == chosen:
                    continue
                cand_host = _candidate_upstream_host(cand["url"])
                if cand_host in tried_hosts:
                    continue
                tried_hosts.add(cand_host)
                if await _verify_stream_alive(cand["url"]):
                    verified_primary = cand["url"]
                    chosen = verified_primary  # promote alternate to primary
                    break

        # Loop-detector: kick off a background probe of the chosen primary so
        # any host that's serving stale content gets quarantined for 24h.
        # Fire-and-forget; cached per-URL by `_LOOP_PROBE_INTERVAL_S` so this
        # doesn't fire on every page load.
        if verified_primary:
            try:
                asyncio.create_task(_probe_manifest_loop(verified_primary, ch_name))
            except RuntimeError:
                pass  # no running loop (e.g. import-time call) — skip

        # Diverse-provider backup list. Without per-host capping a single
        # provider with multiple accounts (e.g. bgdc.live × 3 accounts × 2
        # naming variants = 6 URLs) eats every backup slot before other
        # providers like an upstream host.an upstream host.co get a turn. Cap each upstream
        # host to 3 backup slots so failover URLs span multiple panels.
        from urllib.parse import parse_qs as _bc_parse_qs, unquote as _bc_unquote
        def _backup_upstream_host(u: str) -> str:
            if "localhost:8000" in u and "u=" in u:
                try:
                    inner = _bc_unquote(_bc_parse_qs(urlparse(u).query).get("u", [""])[0])
                    return urlparse(inner).hostname or ""
                except Exception:
                    return ""
            return urlparse(u).hostname or ""
        host_used: dict[str, int] = {}
        backup_src: list[str] = []
        for c in candidates:
            if c["url"] == chosen:
                continue
            h = _backup_upstream_host(c["url"])
            if host_used.get(h, 0) >= 3:
                continue
            host_used[h] = host_used.get(h, 0) + 1
            backup_src.append(c["url"])
            # 5 sources max per channel (1 primary + 4 backups), per product
            # requirement — keeps the player's source list tight + all-working.
            if len(backup_src) >= 4:
                break

        # Force re-encode for channels with irregular GOPs (TSN). See the
        # comment on _RECODE_CHANNELS for the live-evidence write-up.
        chosen_out  = _apply_recode(cast(str, chosen), ch_name)
        backups_out = [_apply_recode(u, ch_name) for u in backup_src]

        return {
            "name":         ch_name,
            "primary_url":  chosen_out,
            "backup_urls":  backups_out,
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
    #
    # 2026-05-03: an upstream host TVA Sports auth-redirects but serves 0 bytes
    # (slot dead) — excluded from TVA Sports until that comes back. an upstream host
    # added as the new primary on TVA Sports/2 (verified end-to-end).
    _fr_priority: dict[str, list[str]] = {
        "RDS":          ["an upstream host", "an upstream host", "ampztl-a", "ampztl-b"],
        "RDS 2":        ["an upstream host", "an upstream host", "ampztl-a", "ampztl-b"],
        "RDS INFO":     ["an upstream host"],
        "TVA Sports":   ["an upstream host", "ampztl-a", "ampztl-b"],
    }
    _acct_re  = _re_bc.compile(r"\(([a-z0-9._-]{3,30})\)\s*$", _re_bc.I)
    _decor_re = _re_bc.compile(r"(?:\s+[^\w\s]+)+\s*$")
    def _is_fr_exact(title: str, ch_name: str) -> bool:
        norm = _normalize_ch(title)
        want = ch_name.lower()
        if norm == want:
            return True
        return _decor_re.sub("", norm).strip() == want

    # Same dead-glyph filter as /game-iptv-streams — these ampztl variants
    # don't play. Keep ✪/✤/☆.
    _FR_DEAD_MARKERS = ("ƒ", "≋")

    def _build_fr_channel(ch_name: str, priority: list[str]) -> dict:
        by_acct: dict[str, str] = {}
        for cand in all_channels:
            title = cand.get("title", "")
            if any(m in title for m in _FR_DEAD_MARKERS):
                continue
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

    # Warm the relay's ffmpeg sessions for the resolved primary chips. Reuses
    # _kick_chip_warmups (already used by /game-iptv-streams) — its expected
    # input shape is broadcasts with `network` + `channels[].url`, so map the
    # BarnCentre result into that shape. Cap-of-5 is enforced inside the
    # warmup helper so spawning here doesn't blow up ffmpeg counts.
    try:
        warmup_input = [
            {"network": ch["name"], "channels": [{"url": ch["primary_url"]}]}
            for ch in result if ch.get("primary_url")
        ]
        _kick_chip_warmups(warmup_input)
    except Exception:
        pass

    # Same poisoning guard as the Lounge builder: don't overwrite a good cache
    # with a cold build that resolved zero playable streams during a transient
    # upstream outage — keep last-known-good and serve it (flagged degraded) so
    # BarnCentre stays watchable until the next successful background refresh.
    if any(ch.get("primary_url") for ch in result) or _barncentre_cache["data"] is None:
        _barncentre_cache["data"] = result
        _barncentre_cache["ts"]   = now_ts
        return {"channels": result, "cached": False}
    return {
        "channels": _barncentre_cache["data"],
        "cached":   True,
        "stale":    True,
        "degraded": True,
    }


# ===========================================================================
# /lounge/* — Origin Lounge personal Fire TV app endpoints
# ===========================================================================
#
# At-home Fire TV companion app. Reuses the same upstream accounts, the same
# relay, and the same `grtzky_session` cookie auth as BarnCentre. Surfaces
# premium cable + sports for Live TV, plus the full upstream VOD catalog
# (movies + series) organised Netflix-style by streaming service.
#
# All endpoints are gated by Next.js cookie middleware (`/api/lounge/*`
# proxies to `http://localhost:8000/lounge/*`), so reaching this code from
# the public internet without a session is impossible.

_LOUNGE_CHANNEL_NAMES: list[str] = [
    # ── Sports (1-19) ── inherits BarnCentre lineup, channel 1+ ──
    *_BARNCENTRE_CHANNEL_NAMES,
    # ── Core Premium (20-29) — always-on movie networks ──
    "HBO Max", "HBO", "Cinemax", "Starz", "Showtime", "MGM+",
    "Paramount+", "Apple TV+", "Prime Video",
    # ── Dark Humor / Edgy / Late Night (30-39) ──
    "AMC", "FX", "FXX", "TNT", "A&E", "Comedy Central",
    "Vice TV", "Shudder", "Investigation Discovery", "Reelz",
    # ── Sci-Fi / Space / Tech (40-49) ──
    "Syfy", "Discovery Science", "NASA TV", "National Geographic",
    "History", "Curiosity Stream",
    # ── Music (50-59) — heavy rotation ──
    "MTV", "MTV Live", "MuchMusic", "Vevo Hits", "CMT Music",
    "Stingray Music", "Music Choice",
    # ── Anime / Animation (60-69) ──
    "Cartoon Network", "Adult Swim", "Teletoon", "Boomerang",
    "Crunchyroll", "24/7 Pokemon", "24/7 Sailor Moon",
    # ── Factual / Lifestyle (70-79) ──
    "Discovery", "TLC", "Food Network", "HGTV", "Oxygen",
    # ── Canadian premium cable (80-89) ──
    "Stack TV", "W Network", "Showcase", "Slice",
    # ── News — US + CA (added 2026-06-27; US/CA only) ──
    "CNN", "CNN International", "HLN", "Bloomberg", "BNN Bloomberg",
    "MSNBC", "Fox News", "CBC News Network", "CTV News", "CTV News Network",
    "CP24", "Global News",
    # ── Canadian basic networks ──
    "CBC", "CTV", "CTV 2", "Global", "Citytv",
    # ── Quebec French — TV / news / sports (CA; names match provider labels, no accents) ──
    "TVA", "Noovo", "ICI Tele", "ICI RDI", "RDI", "LCN", "TVA Sports 2",
    "TV5", "Canal D", "Canal Vie",
    # ── More sports — US/CA ──
    "DAZN 1", "DAZN 2", "DAZN 3", "DAZN 4", "DAZN 5",
    "NFL Network", "NFL RedZone",
    "beIN Sports", "beIN Sports 1", "beIN Sports 2", "beIN Sports 3",
    "beIN Sports 4", "beIN Sports 5",
    "TSN", "Sportsnet", "TVA Sports",
    # ── True crime (US/CA) ──
    "Court TV", "True Crime Network", "ID",
    # ── Adult animation — 24/7 single-show channels (common on IPTV, like 24/7 Pokemon) ──
    "24/7 Family Guy", "24/7 American Dad", "24/7 Rick and Morty",
    "24/7 South Park", "24/7 Bob's Burgers", "24/7 Futurama",
    "24/7 The Simpsons", "24/7 King of the Hill",
]

# Per-channel upstream-host blocklist. Same semantics as
# _BARNCENTRE_HOST_BLOCKLIST — drop hosts that auth-pass + serve broken
# content (looped/frozen/auth-only) on this specific channel. Start empty;
# Bob populates as he discovers them.
_LOUNGE_HOST_BLOCKLIST: dict[str, set[str]] = {}

# Channel number = position in the curated list (1-indexed). Surfaced on
# the API as `channel_number` so the client can sort the EPG by it.
_LOUNGE_CHANNEL_NUMBER: dict[str, int] = {
    name: i + 1 for i, name in enumerate(_LOUNGE_CHANNEL_NAMES)
}

_LOUNGE_TTL = _BARNCENTRE_TTL  # share the same staleness window

# Channel logo CDN base. Each channel's logo lives at
# `{base}/{slug}.png` where slug is the lowercased channel name with
# non-alphanumerics → "-". The static folder is hosted alongside the API
# (Cloudflare-cached), so adding a new logo is a drop-in operation; no
# code change. Missing assets 404 cleanly — the client is responsible
# for graceful degradation when the URL doesn't resolve.
_LOUNGE_LOGO_BASE = "http://localhost:3000/static/channel-logos"


def _lounge_channel_slug(name: str) -> str:
    """Stable filesystem slug for a channel name.

    Lowercase, non-alphanumerics collapsed to single "-", trimmed.
    Example: "HBO Max" → "hbo-max", "24/7 Pokemon" → "24-7-pokemon",
    "A&E" → "a-e", "Paramount+" → "paramount-plus".
    """
    s = (name or "").strip().lower().replace("+", "-plus")
    out: list[str] = []
    last_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-")


def _lounge_logo_url(name: str) -> str:
    return f"{_LOUNGE_LOGO_BASE}/{_lounge_channel_slug(name)}.png"


_lounge_cache: dict = {"data": None, "ts": 0.0}
_LOUNGE_REFRESH_TASK: asyncio.Task | None = None


def _kick_lounge_refresh() -> None:
    global _LOUNGE_REFRESH_TASK
    if _LOUNGE_REFRESH_TASK is not None and not _LOUNGE_REFRESH_TASK.done():
        return
    try:
        loop = asyncio.get_running_loop()

        async def _refresh():
            try:
                await _build_lounge_payload()
            except Exception:
                pass

        _LOUNGE_REFRESH_TASK = loop.create_task(_refresh())
    except RuntimeError:
        pass


@app.get("/lounge/live-channels")
async def lounge_live_channels() -> dict:
    """Curated cable + sports channel list for Origin Lounge.

    Same SWR semantics as /barncentre-channels. Reuses the IPTV channel
    candidate pool (filtered through `_IPTV_CHANNEL_KEYWORDS` which now
    includes cable substrings on top of NHL ones), then matches to
    `_LOUNGE_CHANNEL_NAMES` and ranks each chip via the quality-aware
    `_sort_by_url_priority`.
    """
    import time as _t_lc

    now_ts = _t_lc.time()
    have   = _lounge_cache["data"] is not None
    fresh  = have and (now_ts - _lounge_cache["ts"] < _LOUNGE_TTL)

    # Fire-and-forget EPG warm. By the time the user navigates from
    # Home → Live TV the cache is hot, so the EpgActivity load is instant.
    # Cheap: hits the cached path if already fresh, otherwise builds in
    # the background while we serve channels immediately.
    _kick_epg_prewarm()

    if fresh:
        return {"channels": _lounge_cache["data"], "cached": True}
    if have:
        _kick_lounge_refresh()
        return {"channels": _lounge_cache["data"], "cached": True, "stale": True}

    return await _build_lounge_payload()


_epg_prewarm_inflight: bool = False


def _kick_epg_prewarm() -> None:
    """Fire-and-forget EPG cache build. No-op if cache is fresh or another
    prewarm is already running. Always safe to call repeatedly."""
    global _epg_prewarm_inflight
    import time as _t_pw
    have  = _lounge_epg_cache["data"] is not None
    fresh = have and (_t_pw.time() - _lounge_epg_cache["ts"] < _LOUNGE_EPG_TTL)
    if fresh or _epg_prewarm_inflight:
        return
    _epg_prewarm_inflight = True

    async def _run() -> None:
        global _epg_prewarm_inflight
        try:
            await lounge_epg("")  # builds + caches
        except Exception:
            pass
        finally:
            _epg_prewarm_inflight = False

    try:
        asyncio.create_task(_run())
    except Exception:
        _epg_prewarm_inflight = False


async def _build_lounge_payload() -> dict:
    """Builder for /lounge/live-channels. Models `_build_barncentre_payload`
    closely — same IPTV pool, same matching/normalisation, but uses the
    lounge's broader curated list and per-channel blocklist.
    """
    import time as _t_lp
    import asyncio as _aio_lp
    now_ts = _t_lp.time()

    iptv_result = await iptv_channels()
    # tvpass dropped wholesale; see `_is_tvpass` for context. Filter happens
    # here too so /lounge/* mirrors /barncentre-channels.
    all_channels: list[dict] = iptv_result.get("channels", [])
    all_channels = [ch for ch in all_channels if not _is_tvpass(ch)]

    # Programs source: today's NHL/ESPN/MLB/NBA from the existing helpers
    # (sports channels in _LOUNGE_CHANNEL_NAMES still want their game guide).
    # Cable channels get programs[] populated by the EPG endpoint
    # (/lounge/epg) once it lands; the channel list itself stays light.
    import datetime as _dt_lp
    today = _dt_lp.date.today().isoformat()
    programs: dict[str, list] = {}
    try:
        import httpx as _hx_lp
        async with _hx_lp.AsyncClient(timeout=8) as _cl:
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
                        display = cast(str, _BROADCAST_CODE_MAP.get(code, code))
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

    epg_data, mlb_data, nba_data = await _aio_lp.gather(
        _fetch_espn_schedule(), _fetch_mlb_schedule(), _fetch_nba_schedule(),
    )
    for sport in (epg_data, mlb_data, nba_data):
        for ch_name, progs in sport.items():
            programs.setdefault(ch_name, []).extend(progs)
    for ch_name in programs:
        programs[ch_name].sort(key=lambda p: p.get("start_utc", ""))

    # Match candidates to curated names — longer/more-specific names first
    # so "HBO Max" and "FXX" claim before "HBO" / "FX".
    sorted_names = sorted(_LOUNGE_CHANNEL_NAMES, key=lambda n: -len(n))
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

    # Backup-host capping helper — same logic as the barncentre builder so
    # one provider with N accounts doesn't eat the whole backup chain.
    from urllib.parse import parse_qs as _bl_parse_qs, unquote as _bl_unquote
    def _backup_upstream_host(u: str) -> str:
        if "localhost:8000" in u and "u=" in u:
            try:
                inner = _bl_unquote(_bl_parse_qs(urlparse(u).query).get("u", [""])[0])
                return urlparse(inner).hostname or ""
            except Exception:
                return ""
        return urlparse(u).hostname or ""

    def _candidate_upstream_host(u: str) -> str:
        return _backup_upstream_host(u)

    for name, cands in list(channel_candidates.items()):
        seen_urls: set[str] = set()
        unique: list[dict] = []
        for c in cands:
            if c["url"] not in seen_urls:
                seen_urls.add(c["url"])
                unique.append(c)
        blocked = _LOUNGE_HOST_BLOCKLIST.get(name) or _active_blocklist(name)
        if blocked:
            unique = [c for c in unique if _candidate_upstream_host(c["url"]) not in blocked]
        channel_candidates[name] = _sort_by_url_priority(unique)

    async def _build_one(ch_name: str, candidates: list[dict]) -> dict:
        # Same upstream-bias logic as barncentre — the verifier's 10s window
        # routinely demotes valid upstream chips on cold start, so we trust
        # the field-tested order.
        def _is_upstream(u: str) -> bool:
            return ("an upstream host" in u) or ("ampztl" in u) or ("kstv.us" in u)

        if not candidates:
            return {
                "name":           ch_name,
                "channel_number": _LOUNGE_CHANNEL_NUMBER.get(ch_name, 999),
                "logo_url":       _lounge_logo_url(ch_name),
                "primary_url":    "",
                "backup_urls":    [],
                "category":       "live",
                "programs":       programs.get(ch_name, []),
                "online":         False,
            }

        # Per-channel primary host overrides. an upstream host's TSN slot
        # serves dead 0-byte tokens (auth-redirects to a dead upstream),
        # so for TSN1-5 we promote an upstream host to primary. User-confirmed
        # 2026-05-07: an upstream host TSN feeds play buttery.
        _TSN_NAMES = {"TSN1", "TSN2", "TSN3", "TSN4", "TSN5"}
        chosen = ""
        if ch_name in _TSN_NAMES:
            an upstream host_cands = [c for c in candidates if "an upstream host" in c["url"]]
            if an upstream host_cands:
                chosen = an upstream host_cands[0]["url"]
        if not chosen:
            upstream_cands = [c for c in candidates if _is_upstream(c["url"])]
            chosen = (upstream_cands[0] if upstream_cands else candidates[0])["url"]

        # NO inline verification. Cold-build verification was the OOM root
        # cause: 50+ channels × _verify_stream_alive() against /hls?u=
        # URLs spawned 30+ relay-side ffmpegs and held 50+ concurrent
        # httpx connections in the API worker, OOM-killing both workers
        # and stalling concurrent web traffic ("LOADING GAME..." forever
        # on localhost:3000). The player handles dead chips: 401/404/
        # PlaylistStuckException are detected as terminal failures and
        # rotate to the next chip immediately (see PlayerActivity.kt
        # isTerminalChipFailure). Trust the priority sort; let the
        # client do the live health check at play time.
        verified_primary: str | None = chosen

        host_used: dict[str, int] = {}
        backup_src: list[str] = []
        for c in candidates:
            if c["url"] == chosen:
                continue
            h = _backup_upstream_host(c["url"])
            if host_used.get(h, 0) >= 3:
                continue
            host_used[h] = host_used.get(h, 0) + 1
            backup_src.append(c["url"])
            # 5 sources max per channel (1 primary + 4 backups), per product
            # requirement — same cap as the BarnCentre builder.
            if len(backup_src) >= 4:
                break

        chosen_out  = _apply_recode(cast(str, chosen), ch_name)
        backups_out = [_apply_recode(u, ch_name) for u in backup_src]

        return {
            "name":           ch_name,
            "channel_number": _LOUNGE_CHANNEL_NUMBER.get(ch_name, 999),
            "logo_url":       _lounge_logo_url(ch_name),
            "primary_url":    chosen_out,
            "backup_urls":    backups_out,
            "category":       "live",
            "programs":       programs.get(ch_name, []),
            "online":         verified_primary is not None,
        }

    tasks  = [_build_one(n, c) for n, c in channel_candidates.items()]
    result = list(await _aio_lp.gather(*tasks))
    order  = {n: i for i, n in enumerate(_LOUNGE_CHANNEL_NAMES)}
    result.sort(key=lambda ch: order.get(ch["name"], 999))

    # Pre-warm relay ffmpeg sessions on the resolved primaries (same trick
    # as BarnCentre — saves 4-7s cold-start on the first click).
    try:
        warmup_input = [
            {"network": ch["name"], "channels": [{"url": ch["primary_url"]}]}
            for ch in result if ch.get("primary_url")
        ]
        _kick_chip_warmups(warmup_input)
    except Exception:
        pass

    # Only overwrite the cache when this build actually resolved playable
    # streams. A cold build during a transient upstream/relay hiccup can return
    # every channel with no primary_url; caching that would strand the Lounge
    # on an all-dead list for the full TTL. Keep last-known-good instead and
    # serve it (flagged degraded) so the at-home Fire TV stays watchable until
    # the next successful background refresh.
    if any(ch.get("primary_url") for ch in result) or _lounge_cache["data"] is None:
        _lounge_cache["data"] = result
        _lounge_cache["ts"]   = now_ts
        return {"channels": result, "cached": False}
    return {
        "channels": _lounge_cache["data"],
        "cached":   True,
        "stale":    True,
        "degraded": True,
    }


# ─── /lounge/epg — TV guide for the cable + sports channels ───────────────
#
# Source priority (highest first):
#   1. Sports channels: existing programs[] from /lounge/live-channels (NHL +
#      ESPN + MLB + NBA schedules, populated by `_build_lounge_payload`)
#   2. Each upstream account's `xmltv.php` — many panels expose this and it's
#      the cheapest source for cable EPG. Fetched per-account in parallel,
#      cached 6h.
#   3. iptv-org/epg fallback (deferred — runs as a nightly cron writing
#      Parquet to data/lounge_epg/, glued in here once it ships).

_lounge_epg_cache: dict = {"data": None, "ts": 0.0}
_LOUNGE_EPG_TTL = 6 * 3600  # 6h


async def _fetch_upstream_xmltv(label: str, host: str, port: int, user: str, pw: str) -> dict[str, list[dict]]:
    """Hit one upstream account's xmltv.php and return {channel_id: [programmes]}.

    Returns empty dict on auth failure / parse error / network timeout —
    each panel either supports XMLTV or it doesn't, and we don't retry the
    failures inline (the 6h cache absorbs them).
    """
    import httpx as _hx_xmltv
    import xml.etree.ElementTree as ET

    base = f"http://{host}:{port}"
    url  = f"{base}/xmltv.php?username={user}&password={pw}"
    try:
        async with _hx_xmltv.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
            r = await hx.get(url, headers={"User-Agent": _BROWSER_HEADERS["User-Agent"]})
        if r.status_code != 200 or not r.content or not r.content.lstrip().startswith(b"<"):
            return {}
        root = ET.fromstring(r.content)
    except Exception:
        return {}

    # Map channel id → display-name (XMLTV uses tvg-id which we'll also
    # match against). Programmes reference the channel id as `channel="…"`.
    chan_names: dict[str, str] = {}
    for ch in root.findall("channel"):
        cid = ch.get("id") or ""
        dn  = ch.findtext("display-name") or cid
        chan_names[cid] = dn

    out: dict[str, list[dict]] = {}
    for prog in root.findall("programme"):
        cid    = prog.get("channel") or ""
        start  = prog.get("start") or ""
        stop   = prog.get("stop") or ""
        title  = prog.findtext("title") or ""
        desc   = prog.findtext("desc")
        if not cid or not title:
            continue
        # Convert XMLTV start/stop ("20260504220000 +0000") to ISO 8601.
        def _parse(ts: str) -> str:
            try:
                if not ts:
                    return ""
                dt_part, _, tz_part = ts.partition(" ")
                from datetime import datetime, timezone, timedelta
                dt = datetime.strptime(dt_part, "%Y%m%d%H%M%S")
                if tz_part and len(tz_part) == 5:
                    sign = 1 if tz_part[0] == "+" else -1
                    hh = int(tz_part[1:3])
                    mm = int(tz_part[3:5])
                    dt = dt.replace(tzinfo=timezone(sign * timedelta(hours=hh, minutes=mm)))
                else:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                return ts

        # Key the output by both id and display name so we can match either.
        keys = {cid}
        if cid in chan_names:
            keys.add(chan_names[cid])
        for k in keys:
            out.setdefault(k, []).append({
                "title":     title,
                "start_utc": _parse(start),
                "stop_utc":  _parse(stop),
                "desc":      desc,
            })
    return out


_lounge_epg_build_inflight: bool = False


async def _build_lounge_epg() -> dict:
    """Build the EPG cache from XMLTV per upstream account + sports programs.
    Caller is responsible for mutating _lounge_epg_cache. Slow path —
    8-15s on cold cache; never call this on a request hot path."""
    per_account = await asyncio.gather(
        *[
            _fetch_upstream_xmltv(label, host, port, user, pw)
            for label, host, port, user, pw in _upstream_ACCOUNTS
        ],
        return_exceptions=True,
    )
    merged: dict[str, list[dict]] = {}
    for r in per_account:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            merged.setdefault(k, []).extend(v)
    # Fold in sports programs from the Lounge channel list (already
    # populated when /lounge/live-channels last ran).
    if isinstance(_lounge_cache.get("data"), list):
        for ch in _lounge_cache["data"]:
            ch_name = ch.get("name", "")
            progs   = ch.get("programs", []) or []
            if ch_name and progs:
                merged.setdefault(ch_name, []).extend(progs)
    # Dedup + sort each channel's programmes
    for k in list(merged.keys()):
        seen: set[tuple] = set()
        unique: list[dict] = []
        for p in merged[k]:
            key = (p.get("title", ""), p.get("start_utc", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        unique.sort(key=lambda p: p.get("start_utc", ""))
        merged[k] = unique
    return merged


def _kick_lounge_epg_refresh() -> None:
    """Fire-and-forget rebuild of the EPG cache. No-op if a build is
    already inflight. Always safe to call repeatedly."""
    global _lounge_epg_build_inflight
    if _lounge_epg_build_inflight:
        return
    _lounge_epg_build_inflight = True

    async def _run() -> None:
        global _lounge_epg_build_inflight
        import time as _t_kr
        try:
            merged = await _build_lounge_epg()
            _lounge_epg_cache["data"] = merged
            _lounge_epg_cache["ts"]   = _t_kr.time()
        except Exception:
            pass
        finally:
            _lounge_epg_build_inflight = False

    try:
        asyncio.create_task(_run())
    except Exception:
        _lounge_epg_build_inflight = False


@app.get("/lounge/epg")
async def lounge_epg(channels: str = "") -> dict:
    """Programme listings for the next ~48h on the requested channels.

    Returns `{channel_name: [programmes]}`. `channels` is a comma-separated
    list of names from /lounge/live-channels (e.g. "HBO,FX,AMC"). Empty
    string returns all known programmes.

    SWR semantics:
      - Fresh cache → return immediately.
      - Stale cache → return stale data + kick a background refresh. The
        client gets sub-second response and the next call sees fresh data.
      - No cache (cold start) → block on the build (~8-15s). The Android
        client's 20s read timeout fits inside this; previous behaviour
        also blocked on cold cache, so cold-start UX is unchanged.
    """
    import time as _t_epg
    now_ts = _t_epg.time()
    have   = _lounge_epg_cache["data"] is not None
    fresh  = have and (now_ts - _lounge_epg_cache["ts"] < _LOUNGE_EPG_TTL)

    if not have:
        # Cold cache — must build inline.
        merged = await _build_lounge_epg()
        _lounge_epg_cache["data"] = merged
        _lounge_epg_cache["ts"]   = now_ts
    elif not fresh:
        # Have stale data — serve it, refresh in background.
        _kick_lounge_epg_refresh()

    data: dict[str, list[dict]] = _lounge_epg_cache["data"] or {}
    if channels:
        wanted = {c.strip() for c in channels.split(",") if c.strip()}
        # Match by channel name OR by tvg-id (some panels use ids in the
        # XMLTV `channel` attribute that don't equal the display name).
        filtered: dict[str, list[dict]] = {}
        for w in wanted:
            if w in data:
                filtered[w] = data[w]
                continue
            # Loose match — strip non-alpha from both sides
            wl = "".join(c for c in w.lower() if c.isalnum())
            for k, v in data.items():
                kl = "".join(c for c in k.lower() if c.isalnum())
                if wl == kl or wl in kl or kl in wl:
                    filtered.setdefault(w, []).extend(v)
        return {"programmes": filtered, "cached": fresh}

    return {"programmes": data, "cached": fresh}


# ─── /lounge/vod/* — Movies + Series catalog from upstream VOD APIs ────────
#
# Pipeline:
#   1. Each upstream account exposes action=get_vod_streams + action=get_series.
#      We cache the union (deduped by tmdb_id when present, else by title +
#      year) for 24h in `_lounge_vod_cache`.
#   2. TMDB fills posters / overview / release_date when TMDB_API_KEY is set.
#      Without the key, we fall back to the upstream-supplied metadata
#      (often poster URL is set; overview rarely is). Endpoint flags
#      `metadata_source: "tmdb"` vs `"upstream_only"` so the UI can render
#      the gap honestly.
#   3. Watchmode classifies titles by streaming service (Netflix / Disney+ /
#      Paramount+ / Apple TV+ / Crave / Prime CA) when WATCHMODE_API_KEY is
#      set. Without the key, we group by upstream's category name (panels
#      typically tag titles as "VOD - Netflix" / "Netflix Movies" etc., so
#      this is a decent fallback).

_lounge_vod_cache: dict = {"data": None, "ts": 0.0}
_LOUNGE_VOD_TTL = 24 * 3600  # 24h — VOD catalog is slow-moving

# Streaming services we expose on the Movies/Series picker. Order is the
# order the UI renders them.
_LOUNGE_VOD_SERVICES: list[str] = [
    "Netflix", "Disney+", "Paramount+", "Apple TV+", "HBO Max",
    "Prime Video", "Crave", "Hulu", "Peacock", "Other",
]

# Service-name fragments used to classify upstream category names + TMDB
# provider names. All checks lowercase. Order matters: more specific keys
# first (HBO Max before "max", "disney+" before bare "disney").
_LOUNGE_VOD_CATEGORY_HINTS: dict[str, list[str]] = {
    "Netflix":      ["netflix", "nflx", "n+"],
    "Disney+":      ["disney+", "disney plus", "disneyplus", "disney "],
    "Paramount+":   ["paramount+", "paramount plus", "paramountplus",
                     "paramount", "p+ "],
    "Apple TV+":    ["apple tv+", "apple tv plus", "appletv+", "apple+",
                     "apple tv", "appletv", "atv+", "atv plus"],
    "HBO Max":      ["hbo max", "hbo-max", "hbomax", "hbo", "max only",
                     "max series", "max original", "max:"],
    "Prime Video":  ["amazon prime", "prime video", "primevideo", "amazon",
                     "prime"],
    "Crave":        ["crave", "crave tv", "cravetv"],
    "Hulu":         ["hulu"],
    "Peacock":      ["peacock"],
}

# Title-prefix tags that some IPTV panels use on individual titles, e.g.
# "[NF] Stranger Things", "[DSN] Loki". Checked when the category name
# alone doesn't classify (very common for series, where panels group
# everything under generic buckets like "TV - English").
_LOUNGE_VOD_TITLE_PREFIXES: dict[str, list[str]] = {
    "Netflix":      ["[nf]", "[net]", "[netflix]", "[n]"],
    "Disney+":      ["[dsn]", "[dsny]", "[disn]", "[disney]", "[d+]", "[dp]"],
    "Paramount+":   ["[p+]", "[par]", "[parm]", "[pmt]", "[paramount]",
                     "[pp]"],
    "Apple TV+":    ["[apl]", "[atv]", "[atv+]", "[apple]", "[at+]"],
    "HBO Max":      ["[hbo]", "[max]", "[hbom]", "[hb]"],
    "Prime Video":  ["[amz]", "[prime]", "[ama]", "[ap]", "[pv]"],
    "Crave":        ["[crv]", "[crave]"],
    "Hulu":         ["[hu]", "[hulu]"],
    "Peacock":      ["[pk]", "[peacock]"],
}


# Lounge is English-only. The upstream provider tags many Netflix/HBO/etc.
# Latin-American mirrors with strings like "VOD - Netflix LAT", which the
# service classifier still routes into the English Netflix bucket because
# the word "netflix" is in there. We drop those at intake.
_LOUNGE_VOD_SPANISH_CATEGORY_HINTS: list[str] = [
    "español", "espanol", "castellano",
    "latino", "latinoamerica", "latinoamérica", "latam",
    "spanish", "mexican", "argentin", "colombian", "venezolan",
    " lat ", " esp ", " mex ", " arg ", " col ",
    "lat -", "esp -", "lat–", "esp–",
    "- lat", "- esp", "- mex", "- arg",
    " lat]", " esp]", "(lat)", "(esp)", "(spa)", "(mex)",
    "[lat]", "[esp]", "[spa]", "[mex]",
    "vodlat", "vodes", "voddub", "doblada", "doblaje", "subtitulada",
]

_LOUNGE_VOD_SPANISH_TITLE_HINTS: list[str] = [
    "(lat)", "(esp)", "(spa)", "(mex)", "(arg)", "(col)", "(ven)",
    "[lat]", "[esp]", "[spa]", "[mex]", "[arg]", "[col]", "[ven]",
    "doblada", "doblaje", "castellano", "español", "espanol",
    "latinoamérica", "latinoamerica",
]


def _is_spanish_category(name: str) -> bool:
    nl = (name or "").lower()
    if not nl:
        return False
    return any(h in nl for h in _LOUNGE_VOD_SPANISH_CATEGORY_HINTS)


def _is_spanish_title(name: str) -> bool:
    nl = (name or "").lower()
    if not nl:
        return False
    return any(h in nl for h in _LOUNGE_VOD_SPANISH_TITLE_HINTS)


def _classify_vod_service(category_name: str, title: str = "") -> str:
    """Assign a streaming service from an upstream category label and/or
    title prefix. Returns "Other" when nothing matches — the UI groups
    Other into a generic rail at the end of the service picker.

    Series categories from most IPTV providers don't include service
    names (panels lump them under "TV - English"), so the title-prefix
    pass is what saves them from the Other bucket — many panels tag
    individual titles with "[NF] Show Name" / "[HBO] Show Name".
    """
    cl = (category_name or "").lower()
    if cl:
        for svc, hints in _LOUNGE_VOD_CATEGORY_HINTS.items():
            if any(h in cl for h in hints):
                return svc
    tl = (title or "").lower().lstrip()
    if tl:
        for svc, prefixes in _LOUNGE_VOD_TITLE_PREFIXES.items():
            if any(tl.startswith(p) for p in prefixes):
                return svc
    return "Other"


# Concurrency cap on the TMDB fallback. TMDB allows 50 req/s on the public
# API; 8 in flight + 24h cache TTL is well under the limit.
_TMDB_PROVIDER_CONCURRENCY = 8


async def _tmdb_provider_for(
    hx: "httpx.AsyncClient",
    api_key: str,
    kind: str,
    tmdb_id: str,
) -> str | None:
    """Look up TMDB watch providers (region CA) and translate to one of
    our service buckets. Returns None on miss / non-CA-only / errors."""
    path = "tv" if kind == "series" else "movie"
    try:
        r = await hx.get(
            f"https://api.themoviedb.org/3/{path}/{tmdb_id}/watch/providers",
            params={"api_key": api_key},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None
        ca = ((r.json() or {}).get("results") or {}).get("CA") or {}
    except Exception:
        return None
    # flatrate (subscription) is the strongest signal. ads/free are accepted
    # too — Peacock and Pluto often only show under those tiers in CA.
    for tier in ("flatrate", "ads", "free", "buy", "rent"):
        for p in ca.get(tier) or []:
            pn = (p.get("provider_name") or "").lower()
            if not pn:
                continue
            for svc, hints in _LOUNGE_VOD_CATEGORY_HINTS.items():
                if any(h in pn for h in hints):
                    return svc
    return None


async def _tmdb_classify_unknowns(items: list[dict], kind: str) -> int:
    """In-place reassign service for items currently tagged "Other" that
    have a tmdb_id. Returns how many were reclassified. No-op without
    TMDB_API_KEY env var.
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return 0
    targets = [it for it in items if it.get("service") == "Other" and it.get("tmdb_id")]
    if not targets:
        return 0
    import httpx as _hx_tp
    sem = asyncio.Semaphore(_TMDB_PROVIDER_CONCURRENCY)
    reclassified = 0
    async def _one(hx: _hx_tp.AsyncClient, item: dict) -> None:
        nonlocal reclassified
        async with sem:
            svc = await _tmdb_provider_for(hx, api_key, kind, str(item["tmdb_id"]))
            if svc:
                item["service"] = svc
                reclassified += 1
    async with _hx_tp.AsyncClient(timeout=15.0, follow_redirects=True) as hx:
        await asyncio.gather(*[_one(hx, t) for t in targets], return_exceptions=True)
    return reclassified


async def _tmdb_english_title(
    hx: "httpx.AsyncClient",
    api_key: str,
    kind: str,
    tmdb_id: str,
) -> tuple[str | None, str | None]:
    """Fetch the English-localized title + poster path for one TMDB id.
    Returns (english_title, english_poster_url) — either may be None on miss.
    """
    path = "tv" if kind == "series" else "movie"
    try:
        r = await hx.get(
            f"https://api.themoviedb.org/3/{path}/{tmdb_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=10.0,
        )
        if r.status_code != 200:
            return None, None
        body = r.json() or {}
    except Exception:
        return None, None
    title = (body.get("title") or body.get("name") or "").strip() or None
    poster = body.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster}" if poster else None
    return title, poster_url


async def _tmdb_apply_english_titles(items: list[dict], kind: str) -> int:
    """In-place set `english_name` (and replace `poster` with TMDB art when
    we get one) for every item that has a tmdb_id. Returns how many got an
    English title. No-op without TMDB_API_KEY.

    The upstream `name` is kept around as the fallback. The client-side
    PosterCardPresenter prefers `english_name` when present.
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return 0
    targets = [it for it in items if it.get("tmdb_id") and not it.get("english_name")]
    if not targets:
        return 0
    import httpx as _hx_en
    # English-title pass is per-title — we may issue 1k+ calls on a fresh
    # cache build. Bigger semaphore than the watch-providers pass; TMDB
    # tolerates ~50 req/s on the public API.
    sem = asyncio.Semaphore(16)
    applied = 0
    async def _one(hx: _hx_en.AsyncClient, item: dict) -> None:
        nonlocal applied
        async with sem:
            title, poster = await _tmdb_english_title(hx, api_key, kind, str(item["tmdb_id"]))
            if title:
                item["english_name"] = title
                applied += 1
            if poster:
                # Prefer TMDB poster over the panel's stream_icon — it's
                # cleaner art and almost always English-language design.
                item["poster"] = poster
    async with _hx_en.AsyncClient(timeout=15.0, follow_redirects=True) as hx:
        await asyncio.gather(*[_one(hx, t) for t in targets], return_exceptions=True)
    return applied


async def _fetch_upstream_vod(label: str, host: str, port: int, user: str, pw: str) -> dict:
    """Pull one account's VOD catalog. Returns
    {"categories": {category_id: name}, "movies": [...], "series": [...]}.

    DISABLED — kept as a no-op for the legacy _build_vod_catalog_legacy_upstream
    path. The active VOD catalog source is TMDB
    (see _build_vod_catalog_tmdb). Calling get_vod_streams/get_series
    against the IPTV panels was returning Spanish-dubbed titles and
    chewing up auth quota; live channels (a different endpoint) are not
    affected. To re-enable: restore the function body and point
    _build_vod_catalog at _build_vod_catalog_legacy_upstream.
    """
    return {"label": label, "categories": {}, "movies": [], "series": []}


async def _fetch_upstream_vod_legacy(label: str, host: str, port: int, user: str, pw: str) -> dict:
    """Original upstream VOD fetcher. See _fetch_upstream_vod docstring."""
    import httpx as _hx_vod

    base = f"http://{host}:{port}"
    out: dict = {"label": label, "categories": {}, "movies": [], "series": []}
    try:
        async with _hx_vod.AsyncClient(timeout=30.0, follow_redirects=True) as hx:
            cat_r, mov_r, ser_r = await asyncio.gather(
                hx.get(f"{base}/player_api.php?username={user}&password={pw}&action=get_vod_categories",
                       headers={"User-Agent": _BROWSER_HEADERS["User-Agent"]}),
                hx.get(f"{base}/player_api.php?username={user}&password={pw}&action=get_vod_streams",
                       headers={"User-Agent": _BROWSER_HEADERS["User-Agent"]}),
                hx.get(f"{base}/player_api.php?username={user}&password={pw}&action=get_series",
                       headers={"User-Agent": _BROWSER_HEADERS["User-Agent"]}),
                return_exceptions=True,
            )

        if hasattr(cat_r, "status_code") and cat_r.status_code == 200:
            try:
                cats = cat_r.json()
                if isinstance(cats, list):
                    out["categories"] = {
                        str(c.get("category_id")): c.get("category_name", "")
                        for c in cats if isinstance(c, dict)
                    }
            except Exception:
                pass

        if hasattr(mov_r, "status_code") and mov_r.status_code == 200:
            try:
                streams = mov_r.json()
                if isinstance(streams, list):
                    for s in streams:
                        if not isinstance(s, dict):
                            continue
                        sid       = s.get("stream_id")
                        cat_id    = str(s.get("category_id", ""))
                        cat_name  = out["categories"].get(cat_id, "")
                        nm        = s.get("name") or ""
                        # English-only catalog: drop LAT/ESP/Latino entries
                        # at intake — they otherwise leak into the Netflix /
                        # HBO buckets because the classifier matches the
                        # service-name half of "Netflix LAT" labels.
                        if _is_spanish_category(cat_name) or _is_spanish_title(nm):
                            continue
                        ext       = s.get("container_extension") or "mp4"
                        out["movies"].append({
                            "kind":              "movie",
                            "label":             label,
                            "stream_id":         sid,
                            "name":              s.get("name") or "",
                            "tmdb_id":           s.get("tmdb") or s.get("tmdb_id"),
                            "year":              s.get("year"),
                            "rating":            s.get("rating"),
                            "added":             s.get("added"),
                            "category_name":     cat_name,
                            "service":           _classify_vod_service(cat_name, s.get("name") or ""),
                            "url":               f"{base}/movie/{user}/{pw}/{sid}.{ext}",
                            "poster":            s.get("stream_icon"),
                        })
            except Exception:
                pass

        if hasattr(ser_r, "status_code") and ser_r.status_code == 200:
            try:
                series = ser_r.json()
                if isinstance(series, list):
                    for s in series:
                        if not isinstance(s, dict):
                            continue
                        sid       = s.get("series_id")
                        cat_id    = str(s.get("category_id", ""))
                        cat_name  = out["categories"].get(cat_id, "")
                        nm        = s.get("name") or ""
                        if _is_spanish_category(cat_name) or _is_spanish_title(nm):
                            continue
                        out["series"].append({
                            "kind":              "series",
                            "label":             label,
                            "series_id":         sid,
                            "name":              s.get("name") or "",
                            "tmdb_id":           s.get("tmdb") or s.get("tmdb_id"),
                            "year":              s.get("year"),
                            "rating":            s.get("rating"),
                            "added":             s.get("last_modified") or s.get("releaseDate"),
                            "category_name":     cat_name,
                            "service":           _classify_vod_service(cat_name, s.get("name") or ""),
                            "poster":            s.get("cover"),
                            "host":              host,
                            "port":              port,
                            "user":              user,
                            "pw":                pw,
                        })
            except Exception:
                pass
    except Exception:
        pass

    return out


# TMDB watch-provider IDs per streaming service (region CA). These let us
# query "what's on Netflix in Canada right now" via /discover endpoints,
# which is the cleanest catalog source we have post-upstream. IDs are stable
# TMDB primary keys; verified via /watch/providers/regions.
_LOUNGE_VOD_PROVIDER_IDS_CA: dict[str, list[int]] = {
    "Netflix":      [8],
    "Disney+":      [337],
    "Paramount+":   [531],
    "Apple TV+":    [350],
    "HBO Max":      [1825, 1899],   # Max + legacy "HBO Max"
    "Prime Video":  [9, 119],       # Amazon (US id 9 + worldwide 119, both surface)
    "Crave":        [230],
    "Hulu":         [15],           # mostly US-only; CA gets a thin slice
    "Peacock":      [386],
}

# Pages per service per kind. TMDB returns 20 results per page.
#   25 pages × 20 = 500 titles per service per kind.
# With 9 services × 2 kinds = 18 catalog rails of 500 titles each.
# Catalog build: ~360 TMDB calls in parallel, ~30-45s cold (httpx pooling
# keeps it bounded). Cached 24h (_LOUNGE_VOD_TTL).
_TMDB_DISCOVER_PAGES = 25


async def _tmdb_discover_titles(
    hx: "httpx.AsyncClient",
    api_key: str,
    kind: str,
    service: str,
    provider_ids: list[int],
    region: str = "CA",
) -> list[dict]:
    """Pull popular titles for one (service, kind) from TMDB /discover.

    `kind` is "movie" or "series". Returns a list of VodTitle-shaped dicts
    matching the existing VodCatalogResponse schema so the client doesn't
    change. `service` is set on every returned item so the rail picker
    keeps working. `region` is a TMDB watch_region code (CA, US, GB, etc.).
    """
    path = "tv" if kind == "series" else "movie"
    out: list[dict] = []
    seen: set[int] = set()
    for page in range(1, _TMDB_DISCOVER_PAGES + 1):
        try:
            r = await hx.get(
                f"https://api.themoviedb.org/3/discover/{path}",
                params={
                    "api_key": api_key,
                    "language": "en-US",
                    "watch_region": region,
                    "with_watch_providers": "|".join(str(x) for x in provider_ids),
                    "sort_by": "popularity.desc",
                    "page": page,
                    # Drop adult content and completely unrated items.
                    "include_adult": "false",
                },
                timeout=12.0,
            )
            if r.status_code != 200:
                break
            results = (r.json() or {}).get("results") or []
        except Exception:
            break
        if not results:
            break
        for it in results:
            tid = it.get("id")
            if not tid or tid in seen:
                continue
            seen.add(int(tid))
            poster_path = it.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            backdrop_path = it.get("backdrop_path")
            backdrop_url = f"https://image.tmdb.org/t/p/w780{backdrop_path}" if backdrop_path else None
            release_date = it.get("release_date") or it.get("first_air_date") or ""
            year = release_date[:4] if release_date else None
            name = (it.get("title") or it.get("name") or "").strip()
            rating_n = it.get("vote_average")
            rating = f"{rating_n:.1f}" if isinstance(rating_n, (int, float)) else None
            out.append({
                "kind":          kind,
                "name":          name,
                "english_name":  name,
                "tmdb_id":       int(tid),
                "year":          year,
                "rating":        rating,
                "added":         release_date or None,
                "category_name": None,
                "service":       service,
                # url/stream_id/series_id are unused now — playback goes
                # through the WebView resolver keyed on tmdb_id. Keep the
                # fields nullable so the existing client model stays
                # compatible.
                "url":           None,
                "poster":        poster_url,
                "backdrop":      backdrop_url,
                "stream_id":     None,
                # Use the tmdb_id as the series_id so the existing
                # SeriesDetailActivity flow keeps working.
                "series_id":     int(tid) if kind == "series" else None,
                "overview":      (it.get("overview") or "").strip() or None,
            })
    return out


async def _build_vod_catalog_tmdb() -> dict:
    """TMDB-sourced catalog. Replaces the upstream VOD intake entirely.

    Pulls /discover/{movie,tv} per streaming service for region CA, sorted
    by popularity desc. Each rail caps at ~200 titles (10 TMDB pages × 20).
    No upstream VOD calls — upstream is only used for live channels now.

    Returns the same shape as the legacy _build_vod_catalog so the client
    `VodCatalogResponse` model needs no changes.
    """
    import time as _t_vod
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        # No TMDB key → empty catalog. Honest gap rather than silently
        # serving stale upstream data. Operator notices via the Movies/Series
        # tile being empty and the metadata_source field.
        return {
            "by_service":      {svc: {"movies": [], "series": []} for svc in _LOUNGE_VOD_SERVICES},
            "movies_by_id":    {},
            "series_by_id":    {},
            "metadata_source": "missing_tmdb_key",
            "fetched_at":      _t_vod.time(),
        }

    import httpx as _hx_tmdb_cat
    movies: list[dict] = []
    series: list[dict] = []

    async with _hx_tmdb_cat.AsyncClient(timeout=20.0, follow_redirects=True) as hx:
        # Discovery in parallel across 2 regions (CA + US). Region matters
        # because TMDB watch-providers is region-specific: HBO Max/Hulu/
        # Peacock only return titles for region=US. Many titles dedupe
        # across regions (same TMDB id), so we keep the first occurrence
        # within each (service, kind) rail.
        tasks = []
        for region in ("CA", "US"):
            for svc, ids in _LOUNGE_VOD_PROVIDER_IDS_CA.items():
                tasks.append(_tmdb_discover_titles(hx, api_key, "movie",  svc, ids, region=region))
                tasks.append(_tmdb_discover_titles(hx, api_key, "series", svc, ids, region=region))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException) or not r:
                continue
            for item in r:
                (movies if item["kind"] == "movie" else series).append(item)

    # Group by service and dedup by tmdb_id within a service rail (a title
    # may appear under multiple provider ids, e.g. Crave + HBO Max).
    by_service: dict[str, dict[str, list[dict]]] = {
        svc: {"movies": [], "series": []} for svc in _LOUNGE_VOD_SERVICES
    }
    seen: dict[tuple[str, str], set[int]] = {}
    for it in movies + series:
        svc = it["service"]
        kind = it["kind"]
        bucket_key = "movies" if kind == "movie" else "series"
        if svc not in by_service:
            by_service[svc] = {"movies": [], "series": []}
        seen_key = (svc, kind)
        if seen_key not in seen:
            seen[seen_key] = set()
        if it["tmdb_id"] in seen[seen_key]:
            continue
        seen[seen_key].add(it["tmdb_id"])
        by_service[svc][bucket_key].append(it)

    # Sort each rail by added (release_date) desc within service.
    def _date_key(item: dict) -> str:
        return str(item.get("added") or item.get("year") or "")
    for svc in by_service:
        by_service[svc]["movies"].sort(key=_date_key, reverse=True)
        by_service[svc]["series"].sort(key=_date_key, reverse=True)

    # Index by tmdb_id for the /details endpoint. Movies map to a
    # single-element list to keep the legacy shape; we don't have a
    # "primary + backup" notion when there's only one source.
    movies_by_id: dict[str, list[dict]] = {}
    for m in movies:
        key = str(m.get("tmdb_id") or "")
        if key:
            movies_by_id.setdefault(key, []).append(m)

    series_by_id: dict[str, dict] = {}
    for s in series:
        sid = str(s.get("tmdb_id") or "")
        if sid:
            series_by_id[sid] = s

    return {
        "by_service":      by_service,
        "movies_by_id":    movies_by_id,
        "series_by_id":    series_by_id,
        "metadata_source": "tmdb",
        "fetched_at":      _t_vod.time(),
    }


async def _build_vod_catalog() -> dict:
    """Aggregate VOD catalogs across all upstream accounts. Result shape:

        {
          "by_service": {"Netflix": {"movies": [...], "series": [...]},
                         "Disney+": {...}, ...},
          "movies_by_id":  {tmdb_id: [movie_dict, ...]},  # for instance fallback
          "series_by_id":  {series_id_str: series_dict},
          "metadata_source": "tmdb" | "upstream_only",
          "fetched_at": <unix_ts>,
        }

    NOTE: upstream VOD intake is disabled — see _fetch_upstream_vod docstring.
    Movies/Series come from _build_vod_catalog_tmdb now. This wrapper is
    kept only so the cache-build path doesn't change shape and any other
    callers of _build_vod_catalog continue to work.
    """
    return await _build_vod_catalog_tmdb()


async def _build_vod_catalog_legacy_upstream() -> dict:
    """Legacy upstream VOD aggregation. Kept for emergency revert only —
    not invoked by the current cache build. To re-enable, swap
    _build_vod_catalog body to call this instead of _build_vod_catalog_tmdb.
    """
    import time as _t_vod
    per_account = await asyncio.gather(
        *[
            _fetch_upstream_vod(label, host, port, user, pw)
            for label, host, port, user, pw in _upstream_ACCOUNTS
        ],
        return_exceptions=True,
    )

    movies: list[dict] = []
    series: list[dict] = []
    for r in per_account:
        if not isinstance(r, dict):
            continue
        movies.extend(r.get("movies", []))
        series.extend(r.get("series", []))

    # Second-pass service classification for items still tagged "Other" that
    # have a tmdb_id — TMDB watch-providers (region CA) catches series that
    # IPTV panels lump under generic "TV - English" categories without a
    # title-prefix tag. No-op without TMDB_API_KEY.
    movies_reclassified = await _tmdb_classify_unknowns(movies, "movie")
    series_reclassified = await _tmdb_classify_unknowns(series, "series")

    # Third pass — English-localized titles + posters for everything with a
    # tmdb_id. The IPTV panel often labels titles with their Spanish dub
    # name ("El Padrino" instead of "The Godfather"); TMDB returns the
    # English release title which is what the user actually wants to see.
    await _tmdb_apply_english_titles(movies, "movie")
    await _tmdb_apply_english_titles(series, "series")

    metadata_source = "tmdb_providers" if (movies_reclassified or series_reclassified) else "upstream_only"

    # Group by service.
    by_service: dict[str, dict[str, list[dict]]] = {
        svc: {"movies": [], "series": []} for svc in _LOUNGE_VOD_SERVICES
    }
    for m in movies:
        by_service.setdefault(m["service"], {"movies": [], "series": []})["movies"].append(m)
    for s in series:
        by_service.setdefault(s["service"], {"movies": [], "series": []})["series"].append(s)

    # Sort each rail by added/release date desc — newest first.
    def _date_key(item: dict) -> str:
        return str(item.get("added") or item.get("year") or "")
    for svc in by_service:
        by_service[svc]["movies"].sort(key=_date_key, reverse=True)
        by_service[svc]["series"].sort(key=_date_key, reverse=True)

    # Multi-instance index — maps a tmdb_id (or fallback name+year) to all
    # movie dicts that have it. The /lounge/vod/details endpoint uses this
    # to assemble the backup-URL chain.
    movies_by_id: dict[str, list[dict]] = {}
    for m in movies:
        key = str(m.get("tmdb_id") or "") or f"name:{(m.get('name') or '').lower().strip()}|year:{m.get('year') or ''}"
        movies_by_id.setdefault(key, []).append(m)

    series_by_id: dict[str, dict] = {}
    for s in series:
        sid = str(s.get("series_id") or "")
        if sid:
            series_by_id[sid] = s

    return {
        "by_service":     by_service,
        "movies_by_id":   movies_by_id,
        "series_by_id":   series_by_id,
        "metadata_source": metadata_source,
        "fetched_at":     _t_vod.time(),
    }


@app.get("/lounge/vod/catalog")
async def lounge_vod_catalog(service: str = "") -> dict:
    """Movies + series catalog grouped by streaming service.

    `service` filters to a single service (case-insensitive). Empty returns
    every service. Sort: newest first within each rail.
    """
    import time as _t_vc
    now_ts = _t_vc.time()
    have   = _lounge_vod_cache["data"] is not None
    fresh  = have and (now_ts - _lounge_vod_cache["ts"] < _LOUNGE_VOD_TTL)

    if not fresh:
        try:
            data = await _build_vod_catalog()
            _lounge_vod_cache["data"] = data
            _lounge_vod_cache["ts"]   = now_ts
        except Exception as e:
            if not have:
                return {"error": f"vod build failed: {e}", "by_service": {}}
            data = _lounge_vod_cache["data"]
    else:
        data = _lounge_vod_cache["data"]

    by_service = data.get("by_service", {})
    if service:
        # Case-insensitive match
        sl = service.lower()
        match = next((k for k in by_service if k.lower() == sl), None)
        if not match:
            return {"service": service, "movies": [], "series": [], "metadata_source": data.get("metadata_source")}
        return {
            "service": match,
            "movies":  by_service[match]["movies"][:200],   # cap to keep payload tame
            "series":  by_service[match]["series"][:200],
            "metadata_source": data.get("metadata_source"),
            "cached":  fresh,
        }

    # Whole catalog — return per-service counts so the picker can show
    # numbers, plus a small preview rail (top 20 newest per service).
    summary = {
        svc: {
            "movies_count": len(rails["movies"]),
            "series_count": len(rails["series"]),
            "preview":      rails["movies"][:20] + rails["series"][:20],
        }
        for svc, rails in by_service.items()
    }
    return {
        "services":         _LOUNGE_VOD_SERVICES,
        "summary":          summary,
        "metadata_source":  data.get("metadata_source"),
        "cached":           fresh,
    }


# Embed providers that take a TMDB id and render a player whose network
# traffic includes an .m3u8. The Fire TV WebView resolver loads each in
# fallback order until one captures a stream.
#
# vidsrc.in/.to/.xyz/.pm/.net all gate Cloudflare Turnstile (sitekey
# 0x4AAAAAABNpWSLmOnUi7s0b) and the WebView can't solve it reliably —
# these are excluded entirely. The list below is curated to providers
# that load a working player WITHOUT a captcha, so the WebView's
# m3u8 capture actually fires.
#
# Order = "most likely to capture m3u8 fastest" first. vidlink/moviesapi
# typically autoplay and surface the master.m3u8 within 5-7s. The multi-
# embed services need a server pick before their iframes load and tend
# to be slower (~10s) but cover more obscure titles.

def _vidsrc_movie_embeds(tmdb_id: str) -> list[str]:
    return [
        f"https://vidlink.pro/movie/{tmdb_id}",
        f"https://moviesapi.club/movie/{tmdb_id}",
        f"https://www.2embed.cc/embed/{tmdb_id}",
        f"https://multiembed.mov/?video_id={tmdb_id}&tmdb=1",
        f"https://nontongo.win/embed/movie/{tmdb_id}",
        f"https://pstream.org/embed/movie/{tmdb_id}",
    ]


def _vidsrc_episode_embeds(tmdb_id: str, season: int, episode: int) -> list[str]:
    return [
        f"https://vidlink.pro/tv/{tmdb_id}/{season}/{episode}",
        f"https://moviesapi.club/tv/{tmdb_id}-{season}-{episode}",
        f"https://www.2embed.cc/embedtv/{tmdb_id}&s={season}&e={episode}",
        f"https://multiembed.mov/?video_id={tmdb_id}&tmdb=1&s={season}&e={episode}",
        f"https://nontongo.win/embed/tv/{tmdb_id}/{season}/{episode}",
        f"https://pstream.org/embed/tv/{tmdb_id}/{season}/{episode}",
    ]


@app.get("/lounge/vod/details/{tmdb_id}")
async def lounge_vod_details(tmdb_id: str) -> dict:
    """Movie detail keyed on TMDB id.

    Returns metadata + the list of TMDB-keyed embed URLs (vidlink,
    moviesapi, 2embed, multiembed, nontongo, pstream) the client's
    WebView resolver should try in order.

    When the stream resolver is enabled (env STREAM_RESOLVER_ENABLED=1) and
    is fast enough (configurable budget), this also returns `stream_urls` —
    pre-resolved m3u8 URLs wrapped through `/lounge/vod-stream-proxy`. The
    client should prefer `stream_urls` and fall back to `embed_urls` if
    they're empty or fail at play time.
    """
    if _lounge_vod_cache["data"] is None:
        await lounge_vod_catalog()  # forces a build
    data = _lounge_vod_cache["data"] or {}
    instances = data.get("movies_by_id", {}).get(tmdb_id, [])

    # Even if the catalog cache doesn't know about this tmdb_id (e.g. user
    # pasted one in, or it's been pushed off the popular list since the
    # last cache build), we can still synthesize the embed URLs and let
    # the client try playback. Metadata-only fields fall back to None.
    head: dict = instances[0] if instances else {}

    stream_urls = await _resolve_stream_urls_movie(tmdb_id)

    return {
        "title":         head.get("english_name") or head.get("name") or "",
        "year":          head.get("year"),
        "rating":        head.get("rating"),
        "service":       head.get("service"),
        "poster":        head.get("poster"),
        "backdrop":      head.get("backdrop"),
        "overview":      head.get("overview"),
        "category":      head.get("category_name"),
        "instance_count": len(_vidsrc_movie_embeds(tmdb_id)),
        # Legacy clients: stream URL list. The Lounge app's WebView
        # resolver consumes `embed_urls`; older PlayerActivity-only
        # callers fall back to nothing.
        "urls":          [],
        "embed_urls":    _vidsrc_movie_embeds(tmdb_id),
        # New: pre-resolved m3u8 URLs (already wrapped through the proxy).
        # Empty when the resolver is disabled, the title can't be resolved,
        # or we hit the 6s budget before any provider responded.
        "stream_urls":   stream_urls,
    }


@app.get("/lounge/vod/series/{series_key}")
async def lounge_vod_series(series_key: str) -> dict:
    """Episode list for a series, keyed on TMDB id.

    Returns season+episode structure populated from TMDB. Each episode
    carries the `embed_urls` list so the client's WebView resolver can
    pick one and resolve to an m3u8 at play time.
    """
    if _lounge_vod_cache["data"] is None:
        await lounge_vod_catalog()
    data = _lounge_vod_cache["data"] or {}
    s = data.get("series_by_id", {}).get(series_key) or {}

    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return {"error": "missing_tmdb_key", "seasons": []}

    import httpx as _hx_se
    try:
        async with _hx_se.AsyncClient(timeout=15.0, follow_redirects=True) as hx:
            top = await hx.get(
                f"https://api.themoviedb.org/3/tv/{series_key}",
                params={"api_key": api_key, "language": "en-US"},
            )
            if top.status_code != 200:
                return {"error": f"tmdb_http_{top.status_code}", "seasons": []}
            top_body = top.json() or {}
            season_meta = [
                sm for sm in (top_body.get("seasons") or [])
                if isinstance(sm, dict) and isinstance(sm.get("season_number"), int)
                and sm.get("season_number", 0) > 0  # skip Specials (season 0)
            ]
            # Fetch each season's episode list. Cap at 12 seasons so a
            # 20-season-long sitcom doesn't issue 20 TMDB calls per click.
            season_meta = season_meta[:12]

            async def _fetch_season(sn: int) -> dict | None:
                try:
                    r = await hx.get(
                        f"https://api.themoviedb.org/3/tv/{series_key}/season/{sn}",
                        params={"api_key": api_key, "language": "en-US"},
                    )
                    if r.status_code != 200:
                        return None
                    return r.json() or {}
                except Exception:
                    return None
            season_jsons = await asyncio.gather(
                *[_fetch_season(int(sm["season_number"])) for sm in season_meta],
                return_exceptions=True,
            )
    except Exception as e:
        return {"error": f"fetch_failed: {e}", "seasons": []}

    seasons: list[dict] = []
    for sm, sj in zip(season_meta, season_jsons):
        if not isinstance(sj, dict):
            continue
        sn = int(sm["season_number"])
        episodes: list[dict] = []
        for e in (sj.get("episodes") or []):
            if not isinstance(e, dict):
                continue
            ep_n = e.get("episode_number")
            if not isinstance(ep_n, int):
                continue
            still_path = e.get("still_path")
            still_url = f"https://image.tmdb.org/t/p/w300{still_path}" if still_path else None
            # Stream resolver: only check the cache here — running real
            # resolves for every episode would blow the per-request budget.
            # The first viewer of a given (series, season, episode) gets
            # an empty stream_urls and falls back to embed_urls. Their
            # WebView capture populates the cache via a background task,
            # so the next viewer plays instantly.
            ep_stream_urls = _resolve_stream_urls_episode_cache_only(
                series_key, sn, ep_n,
            )
            episodes.append({
                "episode_number": ep_n,
                "title":          (e.get("name") or "").strip(),
                "overview":       (e.get("overview") or "").strip() or None,
                "still_url":      still_url,
                # Legacy stream URL field — no native URL; the client uses
                # embed_urls and the WebView resolver to play.
                "url":            "",
                "embed_urls":     _vidsrc_episode_embeds(series_key, sn, ep_n),
                "stream_urls":    ep_stream_urls,
            })
        if episodes:
            seasons.append({
                "season_number": sn,
                "episodes":      episodes,
            })

    name  = s.get("english_name") or s.get("name") or top_body.get("name") or ""
    year  = s.get("year") or (top_body.get("first_air_date") or "")[:4]
    rating_n = top_body.get("vote_average")
    rating = s.get("rating") or (f"{rating_n:.1f}" if isinstance(rating_n, (int, float)) else None)
    poster_path = top_body.get("poster_path")
    poster = s.get("poster") or (f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None)
    return {
        "title":   name,
        "year":    year,
        "rating":  rating,
        "service": s.get("service"),
        "poster":  poster,
        "seasons": seasons,
    }


# ─── Stream resolver integration ──────────────────────────────────────────
#
# The stream resolver scrapes bflix-family + vidsrc-family sites server-side
# to deliver pre-resolved m3u8 URLs in the catalog response. The lounge
# client prefers `stream_urls` over `embed_urls` when present.
#
# Disabled by default until E2E verified — set STREAM_RESOLVER_ENABLED=1
# to flip the flag. The resolver is initialised lazily on first request.

def _stream_resolver_enabled() -> bool:
    return os.environ.get("STREAM_RESOLVER_ENABLED") == "1"


# Inline resolve budget for /lounge/vod/details. vidlink hits in ~1.3s
# locally but ~6s on the VPS (cloudflared + cold provider). 8s budget
# leaves headroom, falls back to embed_urls on timeout. Cached calls
# return in <100ms regardless.
_STREAM_RESOLVE_BUDGET_S = float(os.environ.get("STREAM_RESOLVE_BUDGET_S", "8.0"))

# Path the resolver should use to wrap stream URLs. The client's API base
# is the public URL; this needs to be the same so the proxied URLs work.
_RESOLVER_API_BASE = os.environ.get("RESOLVER_API_BASE", "")


def _wrap_resolver_url(url: str) -> str:
    """Wrap a stream resolver-emitted URL through this API's vod proxy.

    The resolver returns paths starting with `/lounge/vod-stream-proxy?...`
    (when RESOLVER_API_BASE is unset) — we keep them as paths so the client
    combines them with its own baseUrl. If RESOLVER_API_BASE is set in env,
    the resolver pre-qualifies the URL absolute and we return it as-is.
    """
    return url


async def _resolve_stream_urls_movie(tmdb_id: str) -> list[str]:
    """Cache-only fast path for /lounge/vod/details.

    The catalog endpoint MUST return quickly — it powers the movie list
    page. We never wait for a real resolve here:
      * Cache hit -> return stream_urls immediately
      * Cache miss -> kick a real resolve in background and return [].
        The client's VodDetailActivity shows a loading splash and
        polls /lounge/vod/details (or /lounge/stream/{id}) until the
        resolve completes and the cache populates.

    This separation keeps catalog browsing snappy AND lets the loading
    splash carry the cold-start wait visibly to the user instead of
    silently locking up the catalog request."""
    if not _stream_resolver_enabled():
        return []
    try:
        tmdb_id_int = int(tmdb_id)
    except (TypeError, ValueError):
        return []
    try:
        from .stream_resolver import get_resolver as _gr
    except Exception as e:
        print(f"[stream_resolver] import failed: {e}", flush=True)
        return []

    r = _gr()
    if r is None:
        # Fire init in background; first viewer gets embed_urls fallback.
        try:
            from .stream_resolver import init_resolver
            asyncio.create_task(init_resolver())
        except Exception:
            pass
        return []

    # Cache hit — instant return.
    key = (tmdb_id_int, "movie", None, None)
    entry = r._cache.get(key)
    if entry is not None and entry.expires_at > __import__("time").monotonic() and not entry.is_failure:
        return [_wrap_resolver_url(u) for u in entry.result.stream_urls]

    # Cache miss — kick a real resolve in the background. Don't await.
    # This way the catalog endpoint returns in <100 ms and the next
    # call hits the cache.
    try:
        async def _bg_resolve() -> None:
            try:
                await r.resolve_movie(tmdb_id_int, budget_s=30.0)
            except Exception as e:
                print(f"[stream_resolver] bg_resolve_movie({tmdb_id}) failed: {e}",
                      flush=True)
        asyncio.create_task(_bg_resolve())
    except Exception:
        pass
    return []


def _resolve_stream_urls_episode_cache_only(
    series_key: str, season: int, episode: int,
) -> list[str]:
    """Cache-only lookup for episode list. No live resolve here — running
    real resolves for every episode in a series would blow the budget."""
    if not _stream_resolver_enabled():
        return []
    try:
        tmdb_id_int = int(series_key)
    except (TypeError, ValueError):
        return []
    try:
        from .stream_resolver import get_resolver as _gr
    except Exception:
        return []
    r = _gr()
    if r is None:
        return []
    key = (tmdb_id_int, "series", season, episode)
    entry = r._cache.get(key)
    if entry is None:
        return []
    import time as _t_es
    if entry.expires_at <= _t_es.monotonic():
        return []
    if entry.is_failure:
        return []
    return [_wrap_resolver_url(u) for u in entry.result.stream_urls]


@app.get("/lounge/stream/{tmdb_id}")
async def lounge_stream_movie(tmdb_id: str, budget_s: float = 12.0) -> dict:
    """Resolve a movie to playable stream URLs. Longer budget than inline.

    Used by clients that want to show a loading spinner instead of waiting
    for /lounge/vod/details to come back. 12s budget is the default; the
    client can request more or less via ?budget_s=.
    """
    if not _stream_resolver_enabled():
        return {"stream_urls": [], "embed_urls": _vidsrc_movie_embeds(tmdb_id),
                "providers_tried": [], "cache_hit": False, "resolved_in_ms": 0,
                "disabled": True}
    try:
        tmdb_id_int = int(tmdb_id)
    except (TypeError, ValueError):
        return {"stream_urls": [], "embed_urls": _vidsrc_movie_embeds(tmdb_id),
                "providers_tried": [], "cache_hit": False, "resolved_in_ms": 0,
                "error": "invalid_tmdb_id"}
    from .stream_resolver import resolve_movie as _rm
    budget_s = max(2.0, min(30.0, float(budget_s)))
    try:
        result = await asyncio.wait_for(_rm(tmdb_id_int, budget_s=budget_s),
                                         timeout=budget_s + 1.0)
    except asyncio.TimeoutError:
        return {"stream_urls": [], "embed_urls": _vidsrc_movie_embeds(tmdb_id),
                "providers_tried": [], "cache_hit": False, "resolved_in_ms": 0,
                "error": "timeout"}
    return {
        "stream_urls":     [_wrap_resolver_url(u) for u in result.stream_urls],
        "embed_urls":      result.embed_urls,
        "providers_tried": [a.__dict__ for a in result.providers_tried],
        "cache_hit":       result.cache_hit,
        "resolved_in_ms":  result.resolved_in_ms,
    }


@app.get("/lounge/stream/{tmdb_id}/{season}/{episode}")
async def lounge_stream_episode(
    tmdb_id: str, season: int, episode: int, budget_s: float = 12.0,
) -> dict:
    if not _stream_resolver_enabled():
        return {"stream_urls": [],
                "embed_urls": _vidsrc_episode_embeds(tmdb_id, season, episode),
                "providers_tried": [], "cache_hit": False, "resolved_in_ms": 0,
                "disabled": True}
    try:
        tmdb_id_int = int(tmdb_id)
    except (TypeError, ValueError):
        return {"stream_urls": [],
                "embed_urls": _vidsrc_episode_embeds(tmdb_id, season, episode),
                "providers_tried": [], "cache_hit": False, "resolved_in_ms": 0,
                "error": "invalid_tmdb_id"}
    from .stream_resolver import resolve_episode as _re
    budget_s = max(2.0, min(30.0, float(budget_s)))
    try:
        result = await asyncio.wait_for(
            _re(tmdb_id_int, season, episode, budget_s=budget_s),
            timeout=budget_s + 1.0,
        )
    except asyncio.TimeoutError:
        return {"stream_urls": [],
                "embed_urls": _vidsrc_episode_embeds(tmdb_id, season, episode),
                "providers_tried": [], "cache_hit": False, "resolved_in_ms": 0,
                "error": "timeout"}
    return {
        "stream_urls":     [_wrap_resolver_url(u) for u in result.stream_urls],
        "embed_urls":      result.embed_urls,
        "providers_tried": [a.__dict__ for a in result.providers_tried],
        "cache_hit":       result.cache_hit,
        "resolved_in_ms":  result.resolved_in_ms,
    }


@app.get("/lounge/stream-resolver/health")
async def lounge_stream_resolver_health() -> dict:
    """Per-provider rolling success rate + state.json fields. Public — the
    info is non-sensitive and useful for debugging from any client."""
    if not _stream_resolver_enabled():
        return {"enabled": False}
    try:
        from .stream_resolver import get_resolver as _gr
    except Exception as e:
        return {"enabled": True, "error": f"import_failed: {e}"}
    r = _gr()
    if r is None:
        return {"enabled": True, "running": False}
    healths = r.health.all()
    out: dict = {"enabled": True, "running": True, "providers": {}}
    for pid, h in healths.items():
        out["providers"][pid] = {
            "rolling_successes":         h.rolling_successes,
            "rolling_failures":          h.rolling_failures,
            "consecutive_silent_empty":  h.consecutive_silent_empty,
            "success_rate":              round(h.success_rate, 3),
            "last_success_at":           h.last_success_at,
            "last_failure_at":           h.last_failure_at,
            "last_failure_reason":       h.last_failure_reason,
            "current_base_url":          h.current_base_url,
            "pattern_version":           h.pattern_version,
        }
    out["browser_started_at"] = r.browser.started_at
    out["browser_eviction_count"] = r.browser.eviction_count
    out["cache_size"] = len(r._cache)
    return out


@app.post("/lounge/stream-resolver/refresh")
async def lounge_stream_resolver_refresh(provider: str) -> dict:
    """Force a rediscovery pass for the named provider. Returns the diff
    report — patches that were applied, structural changes that need
    operator attention, captured AJAX endpoints from the trace."""
    if not _stream_resolver_enabled():
        return {"enabled": False}
    from .stream_resolver import get_resolver as _gr
    r = _gr()
    if r is None:
        return {"enabled": True, "running": False}
    diff = await r.force_rediscover(provider)
    if diff is None:
        return {"error": "unknown_provider", "provider": provider}
    return diff.to_dict()


@app.post("/lounge/stream-resolver/clear-cache")
async def lounge_stream_resolver_clear_cache(tmdb_id: str | None = None) -> dict:
    if not _stream_resolver_enabled():
        return {"enabled": False}
    from .stream_resolver import get_resolver as _gr
    r = _gr()
    if r is None:
        return {"running": False}
    try:
        n = await r.clear_cache(int(tmdb_id) if tmdb_id else None)
    except (TypeError, ValueError):
        return {"error": "invalid_tmdb_id"}
    return {"cleared": n}


@app.get("/lounge/stream-resolver/test")
async def lounge_stream_resolver_test(tmdb_id: str) -> dict:
    """Resolve a movie + return full diagnostics. Same as /lounge/stream/{id}
    but always with full budget + verbose providers_tried even on miss."""
    if not _stream_resolver_enabled():
        return {"enabled": False}
    try:
        tmdb_id_int = int(tmdb_id)
    except (TypeError, ValueError):
        return {"error": "invalid_tmdb_id"}
    from .stream_resolver import resolve_movie as _rm
    result = await _rm(tmdb_id_int, budget_s=20.0)
    return {
        "stream_urls":     [_wrap_resolver_url(u) for u in result.stream_urls],
        "embed_urls":      result.embed_urls,
        "providers_tried": [a.__dict__ for a in result.providers_tried],
        "cache_hit":       result.cache_hit,
        "resolved_in_ms":  result.resolved_in_ms,
    }


# ─── /lounge/vod-stream-proxy ─────────────────────────────────────────────
#
# Range-aware byte-stream proxy for upstream VOD .mp4 / .mkv files. The
# existing /stream-proxy is HLS-only (it rewrites manifests + segments).
# VOD is a single-file download where the player needs Range header
# forwarding so seek works.

# Hosts the WebView resolver may legitimately produce after resolving a
# vidsrc embed. Wildcard suffixes — match by endswith on the parsed
# hostname so subdomains roll up. Any other host gets 403.
_VIDSRC_RESOLVED_HOST_SUFFIXES: tuple[str, ...] = (
    ".mp4upload.com", ".doodstream.com", ".dood.cx", ".dood.li", ".dood.la",
    ".dood.pm", ".dood.so", ".dood.sh", ".dood.ws", ".dood.wf", ".dood.re",
    ".dood.work", ".dood.watch", ".dood.yt",
    ".vidcdn.pro", ".vidcdn.com", ".vidsrc.xyz", ".vidsrc.in", ".vidsrc.pm",
    ".vidsrc.net", ".vidsrc.cc", ".vidsrc.me", ".vidsrc.to",
    ".vidcloud.online", ".vidcloud.icu", ".vidcloud.io", ".vidcloud.lol",
    ".vidcloud.co", ".vidcloud.stream",
    ".upcloud.io", ".upstream.to", ".megacloud.tv", ".megacloud.club",
    ".rabbitstream.net", ".filemoon.sx", ".filemoon.in", ".filemoon.to",
    ".streamtape.com", ".streamtape.net", ".mixdrop.co", ".mixdrop.ag",
    ".mixdrop.bz", ".mixdrop.club", ".mixdrop.sx",
    ".smashy.stream", ".smashystream.com",
    ".embed.su", ".embedsu.com",
    ".2embed.cc", ".2embed.org",
    ".autoembed.cc",
    # Streaming hosts vidsrc-family embeds commonly resolve to. Each is a
    # real HLS or progressive-mp4 CDN — not a generic open relay. Add new
    # ones here when the [vod-proxy] BLOCKED log surfaces them.
    ".streamruby.com", ".streamhg.com", ".streamzy.com", ".streamzy.cc",
    ".streamlare.com", ".sltube.org", ".slmaxed.com", ".slwatch.co",
    ".streamwish.com", ".streamwish.to", ".swdyu.com",
    ".vidcloud9.com", ".vidcloud.lol", ".vidcloud.life",
    ".kerapoxy.cc", ".feurl.com", ".fembed.com", ".gomo.to",
    ".uqload.com", ".uqload.io", ".uqload.co",
    ".videovard.sx", ".videovard.to",
    ".zoechip.com", ".zorox.cc", ".zorox.fans",
    ".vidplay.online", ".vidplay.lol", ".vidplay.site",
    ".filelions.to", ".filelions.live", ".filelions.online",
    ".smashy.cloud", ".smashy.work",
    ".neonhorizonworkshops.com",
    "ployan.me",
    ".cdnshells.com", ".cdnsh.com",
    ".jp19.icu", ".vrtcdn.com",
    ".vodvidl.site", "storm.vodvidl.site",
    ".vidlink.pro",
    ".speedsterwave.app", "easy.speedsterwave.app",
    ".uns.wtf", "vidflix.uns.wtf",
    ".cyou", "flixcdn.cyou",
    # Akamai / Cloudfront / Fastly fronts that vidsrc-family upstreams hide
    # behind. We don't enumerate every shard — match on the parent.
    ".akamaized.net", ".cloudfront.net", ".fastly.net", ".bunnycdn.com",
    ".b-cdn.net",
)


def _vod_proxy_host_allowed(host: str | None) -> bool:
    if not host:
        return False
    h = host.lower()
    if h in _upstream_HOSTS:
        return True
    for suffix in _VIDSRC_RESOLVED_HOST_SUFFIXES:
        if h == suffix.lstrip(".") or h.endswith(suffix):
            return True
    return False


def _rewrite_m3u8_for_vod(
    content: str, original_url: str, proxy_base: str, referer: str | None,
) -> str:
    """Rewrite segment URLs in a vidlink/vidsrc-resolved manifest so they
    flow back through /lounge/vod-stream-proxy (preserving the referer)
    instead of trying to fetch directly from the upstream CDN.

    The upstream CDN (e.g. storm.vodvidl.site) requires a specific
    Referer header that ExoPlayer can't set per-segment. By rewriting,
    every segment fetch comes back to us and we re-add the Referer.
    """
    from urllib.parse import urljoin, quote
    out_lines: list[str] = []
    ref_param = f"&referer={quote(referer, safe='')}" if referer else ""
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            # Pass-through tags. EXT-X-KEY URI= and EXT-X-MAP URI= aren't
            # rewritten here for simplicity — most vidlink streams don't
            # encrypt or use map. If they do, we extend later.
            out_lines.append(line)
            continue
        # This line is a URI (segment or sub-playlist).
        abs_url = urljoin(original_url, s)
        wrapped = f"{proxy_base}?url={quote(abs_url, safe='')}{ref_param}"
        out_lines.append(wrapped)
    return "\n".join(out_lines) + "\n"


@app.get("/lounge/vod-stream-proxy")
async def lounge_vod_stream_proxy(url: str, request: Request):
    """Stream a VOD file (mp4 or HLS m3u8 segment) from an upstream.

    Range header is forwarded both ways so ExoPlayer / hls.js seek works.
    Allowlist combines:
      - upstream hosts (legacy — for any leftover IPTV VOD flow)
      - vidsrc-resolved CDN hosts (for the new TMDB+vidsrc playback path)
    Anything else returns 403 so this can't be turned into an open proxy.

    `referer` and `origin` query params are accepted as overrides — the
    WebView resolver captures the Referer the embed required and passes
    it along; many vidsrc-family CDNs hard-require a matching Referer.
    """
    from fastapi.responses import StreamingResponse, JSONResponse

    parsed = urlparse(url)
    if not _vod_proxy_host_allowed(parsed.hostname):
        # Log rejections so we can see which CDNs vidsrc-family embeds
        # are actually resolving to. Add these to
        # _VIDSRC_RESOLVED_HOST_SUFFIXES once observed.
        print(f"[vod-proxy] BLOCKED host={parsed.hostname!r} url={url[:200]!r}", flush=True)
        return JSONResponse(
            status_code=403,
            content={"error": "host_not_allowed", "host": parsed.hostname},
        )

    import httpx as _hx_vsp
    fwd_headers: dict[str, str] = {
        "User-Agent": _BROWSER_HEADERS["User-Agent"],
        # Force identity encoding upstream — we use aiter_raw() to stream
        # the response, which means we'd serve gzipped bytes to the
        # client without the Content-Encoding header. ExoPlayer would
        # treat that as garbage. Asking upstream for plain text avoids
        # the round-trip entirely.
        "Accept-Encoding": "identity",
    }
    range_hdr = request.headers.get("range")
    if range_hdr:
        fwd_headers["Range"] = range_hdr
    # Optional Referer/Origin override — the WebView resolver may pass
    # these as query params after capturing them from the embed page.
    qp = request.query_params
    referer_override = qp.get("referer") or qp.get("ref")
    if referer_override:
        fwd_headers["Referer"] = referer_override
    origin_override = qp.get("origin")
    if origin_override:
        fwd_headers["Origin"] = origin_override

    client = _hx_vsp.AsyncClient(timeout=httpx.Timeout(30.0, read=None), follow_redirects=True)

    # Detect manifest vs segment from the URL path. HLS manifests need
    # buffering + segment-URL rewriting so ExoPlayer's relative .ts
    # fetches resolve back to /lounge/vod-stream-proxy. Segment files
    # stream raw with Range support.
    path_lower = (urlparse(url).path or "").lower()
    is_manifest = path_lower.endswith(".m3u8") or "playlist.m3u8" in path_lower or "master.m3u8" in path_lower or "index.m3u8" in path_lower

    if is_manifest:
        from fastapi.responses import Response as _Resp
        # Buffer the small manifest, rewrite segment URLs, return as text.
        try:
            r = await client.get(url, headers=fwd_headers)
        except Exception as e:
            await client.aclose()
            return JSONResponse(status_code=502, content={"error": f"fetch: {e}"})
        if r.status_code >= 400:
            await client.aclose()
            return JSONResponse(
                status_code=r.status_code,
                content={"error": f"upstream_status_{r.status_code}"},
            )
        # Rewrite segment URLs to point back through /lounge/vod-stream-proxy
        # with the same referer carried forward, so segment fetches hit our
        # proxy which knows the upstream's Referer requirement.
        body_text = r.text
        proxy_base = f"{request.url.scheme}://{request.url.netloc}/lounge/vod-stream-proxy"
        rewritten = _rewrite_m3u8_for_vod(body_text, str(r.url), proxy_base, referer_override)
        await client.aclose()
        ct = r.headers.get("content-type") or "application/vnd.apple.mpegurl"
        return _Resp(
            content=rewritten,
            status_code=r.status_code,
            media_type=ct,
            headers={"cache-control": "no-cache", "accept-ranges": "bytes"},
        )

    # Segment path: stream raw bytes with Range pass-through.
    upstream = await client.send(
        client.build_request("GET", url, headers=fwd_headers),
        stream=True,
    )

    if upstream.status_code >= 400:
        await upstream.aclose()
        await client.aclose()
        return JSONResponse(
            status_code=upstream.status_code,
            content={"error": f"upstream_status_{upstream.status_code}"},
        )

    pass_headers: dict[str, str] = {}
    for h in ("content-length", "content-range",
              "accept-ranges", "etag", "last-modified", "cache-control",
              "content-encoding"):
        v = upstream.headers.get(h)
        if v:
            pass_headers[h] = v
    # Override content-type. Some upstream CDNs (vodvidl, doodstream)
    # disguise .ts segments as image/jpg or image/png to dodge ISP
    # filtering. ExoPlayer rejects HLS segments with image/* content-type
    # and refuses to play. Force video/mp2t for .ts files, leave others
    # as-is from upstream.
    upstream_ct = (upstream.headers.get("content-type") or "").lower()
    path_no_q = url.split("?", 1)[0].lower()
    if path_no_q.endswith(".ts") or path_no_q.endswith(".jpg") or "image/" in upstream_ct:
        # HLS .ts segment — force the right MIME so ExoPlayer accepts it
        pass_headers["content-type"] = "video/mp2t"
    elif upstream_ct:
        pass_headers["content-type"] = upstream_ct
    else:
        pass_headers["content-type"] = "video/mp2t"
    pass_headers.setdefault("accept-ranges", "bytes")

    async def _gen():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _gen(),
        status_code=upstream.status_code,
        headers=pass_headers,
    )
