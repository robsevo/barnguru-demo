#!/usr/bin/env python3
"""Run the RAPM and WAR models over real NHL rosters.

WHAT IS REAL AND WHAT IS NOT — READ THIS FIRST
----------------------------------------------
**Real**, fetched from the NHL's public API by `scripts/fetch_nhl.py`:
players, clubs, positions, ice time, and every counting stat the dashboard
shows — goals, assists, points, plus/minus, save percentage, standings.

**Simulated**: the shift-level and shot-level events this script feeds to the
models. RAPM needs to know which ten skaters were on the ice for every shot
attempt, and the league's public endpoints do not publish that. So the play is
generated, anchored to each player's real production rate, and the models fit
against it.

That means the RAPM and WAR figures here are **a demonstration of the models,
not a measurement of the players**. A leaderboard that looked authoritative
while resting on simulated play would be the dishonest version of this. The
dashboard labels these columns accordingly, and so does the README.

Everything else is the production pipeline: `models/rapm_model.py` and
`models/war_model.py` are called unmodified.

    uv run python scripts/fetch_nhl.py --season 20242025    # real data first
    uv run python scripts/make_demo_data.py                 # then the models

Seeded, so a given (seed, games) always produces the same output.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import polars as pl

from models.rapm_model import RAPMModel
from models.war_model import ROSTER_SLOTS_SKATERS, gar_from_rapm, war_from_gar

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"

PERIOD_SECS = 20 * 60
REGULATION_SECS = 3 * PERIOD_SECS


def load_league() -> tuple[dict, int]:
    path = DEMO_DIR / "league.json"
    if not path.exists():
        raise SystemExit(
            "No league data. Fetch it first:\n"
            "  uv run python scripts/fetch_nhl.py --season 20242025"
        )
    league = json.loads(path.read_text())
    return league, int(league.get("season", 2025))


def attach_skill(skaters: list[dict]) -> list[dict]:
    """Give every skater a latent rating derived from real production.

    Points per 60 minutes, standardised across the league. This is what the
    simulation samples from, so generated play reflects the real gap between a
    first-line centre and a fourth-line winger and RAPM has something genuine to
    recover rather than noise.

    It is also the honest framing of what can be shown here: RAPM is being asked
    to rediscover scoring rate from simulated shifts. That is a test of the
    estimator, not a new measurement of the player.
    """
    rated = []
    for p in skaters:
        toi_min = (p.get("toiPerGame") or 0.0) * (p.get("gp") or 0)
        if toi_min < 60:                       # under an hour all season
            continue
        p60 = (p.get("points") or 0) / (toi_min / 60.0)
        rated.append({**p, "_p60": p60})

    if not rated:
        raise SystemExit("no skaters with enough ice time — is the fetch complete?")

    mean = statistics.fmean(r["_p60"] for r in rated)
    sd = statistics.pstdev([r["_p60"] for r in rated]) or 1.0
    for r in rated:
        r["true_skill"] = round((r["_p60"] - mean) / sd, 4)
        del r["_p60"]
    return rated


def simulate(players: list[dict], season: int, n_games: int, rng: random.Random):
    """Generate shifts and shot events in the shapes `build_stints` documents."""
    by_team: dict[str, list[dict]] = {}
    for p in players:
        by_team.setdefault(p["team"], []).append(p)
    by_team = {t: ps for t, ps in by_team.items() if len(ps) >= 10}
    abbrevs = sorted(by_team)

    shifts: list[dict] = []
    shots: list[dict] = []

    for game_no in range(n_games):
        game_id = int(f"{season}{game_no:05d}")
        home, away = rng.sample(abbrevs, 2)
        t = 0.0
        while t < REGULATION_SECS:
            end = min(t + rng.uniform(35.0, 55.0), float(REGULATION_SECS))

            # ~14% special teams, roughly the real share. Without it toi_pp and
            # toi_pk are zero for everyone and two of the four GAR components
            # contribute nothing.
            roll = rng.random()
            if roll < 0.09:
                home_n, away_n = 5, 4
            elif roll < 0.14:
                home_n, away_n = 4, 5
            else:
                home_n, away_n = 5, 5

            home_unit = rng.sample(by_team[home], home_n)
            away_unit = rng.sample(by_team[away], away_n)

            for p in home_unit:
                shifts.append({"game_id": game_id, "player_id": p["player_id"],
                               "team_id": home, "game_seconds_start": t,
                               "game_seconds_end": end})
            for p in away_unit:
                shifts.append({"game_id": game_id, "player_id": p["player_id"],
                               "team_id": away, "game_seconds_start": t,
                               "game_seconds_end": end})

            home_skill = sum(p["true_skill"] for p in home_unit) / len(home_unit)
            away_skill = sum(p["true_skill"] for p in away_unit) / len(away_unit)
            edge = (home_skill - away_skill) + 2.2 * (home_n - away_n)

            minutes = (end - t) / 60.0
            for _ in range(max(0, int(round(rng.gauss(1.0 * minutes, 0.45 * minutes))))):
                to_home = rng.random() < (0.5 + 0.055 * edge)
                shot_t = rng.uniform(t, end)
                # Chance QUALITY has to depend on the unit too. With xG drawn
                # independently of who is on the ice, the only signal available
                # to a model whose target IS expected goals is shot direction,
                # and the recovered correlation collapses to noise.
                attacking = home_skill if to_home else away_skill
                defending = away_skill if to_home else home_skill
                quality = 1.0 + 0.11 * (attacking - defending)
                shots.append({
                    "game_id": game_id,
                    "period": min(3, int(shot_t // PERIOD_SECS) + 1),
                    "time": shot_t,
                    "x_goal": max(0.005, min(0.95,
                                  rng.betavariate(1.6, 18.0) * max(0.25, quality))),
                    "shooting_team": home if to_home else away,
                    "home_team": home, "away_team": away,
                    "home_skaters": home_n, "away_skaters": away_n,
                })
            t = end

    shifts_df = pl.DataFrame(shifts).with_columns([
        pl.col("game_id").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
        pl.col("game_seconds_start").cast(pl.Float64),
        pl.col("game_seconds_end").cast(pl.Float64),
    ])
    shots_df = pl.DataFrame(shots).with_columns([
        pl.col("game_id").cast(pl.Int64), pl.col("period").cast(pl.Int64),
        pl.col("time").cast(pl.Float64), pl.col("x_goal").cast(pl.Float64),
        pl.col("home_skaters").cast(pl.Int64), pl.col("away_skaters").cast(pl.Int64),
    ])
    return shifts_df, shots_df


def build_war(rapm_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Convert RAPM to GAR and WAR, with replacement level from this pool.

    `war_model._compute_replacement_level` does the same band calculation but
    ends with a guard: if the computed EV replacement is not below -0.5 goals/60
    it substitutes a constant of -2.0. That guard is calibrated for the ridge
    shrinkage in real fitted data; against this pool it fires every time and adds
    a flat +2.0 goals/60 to every player — about 32 goals each over a season, and
    it produced a +14 WAR league leader in an early run. The band is computed
    here directly and the number printed, because the dataset rests on it.
    """
    ranked = rapm_df.sort("toi_ev", descending=True)
    band = ranked.slice(ROSTER_SLOTS_SKATERS, 30)
    if len(band) >= 10:
        repl_ev = float(band["rapm_ev_off"].mean() or 0.0) + float(band["rapm_ev_def"].mean() or 0.0)
        repl_pp = float(band["rapm_pp"].mean() or 0.0)
        repl_pk = float(band["rapm_pk"].mean() or 0.0)
    else:
        repl_ev = repl_pp = repl_pk = 0.0
    print(f"  replacement level (from {len(band)}-player band): "
          f"ev={repl_ev:+.3f} pp={repl_pp:+.3f} pk={repl_pk:+.3f}")

    rows = []
    for r in rapm_df.iter_rows(named=True):
        # RAPM stores TOI in SECONDS; gar_from_rapm takes MINUTES.
        gar_ev, gar_pp, gar_pk = gar_from_rapm(
            rapm_ev_off=r["rapm_ev_off"], rapm_ev_def=r["rapm_ev_def"],
            rapm_pp=r["rapm_pp"], rapm_pk=r["rapm_pk"],
            toi_ev_min=r["toi_ev"] / 60.0,
            toi_pp_min=r["toi_pp"] / 60.0,
            toi_pk_min=r["toi_pk"] / 60.0,
            replacement_ev=repl_ev, replacement_pp=repl_pp, replacement_pk=repl_pk,
        )
        total = gar_ev + gar_pp + gar_pk
        rows.append({
            "player_id": r["player_id"], "player_name": r["player_name"],
            "team": r["team"], "season": season, "gp": r["gp"],
            "gar_ev": float(gar_ev), "gar_pp": float(gar_pp), "gar_pk": float(gar_pk),
            "gar_total": float(total), "war": float(war_from_gar(float(total))),
        })
    return pl.DataFrame(rows).sort("war", descending=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=1312,
                    help="simulated games (a real season is 1312)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    league, season = load_league()
    skaters = [p for p in league["players"] if p.get("position") != "G"]
    players = attach_skill(skaters)
    print(f"league: {len(league['teams'])} clubs, {len(players)} rated skaters "
          f"(season {season}, source: {league.get('source', 'unknown')})")

    # Persist the latent rating so the dashboard can show what the model was
    # asked to recover next to what it produced.
    skill_map = {str(p["player_id"]): p["true_skill"] for p in players}
    (DEMO_DIR / "skill.json").write_text(json.dumps(skill_map, indent=2) + "\n")

    rng = random.Random(args.seed)
    (DATA_DIR / "rapm").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "war").mkdir(parents=True, exist_ok=True)

    shifts_df, shots_df = simulate(players, season, args.games, rng)
    print(f"{season}: {len(shifts_df):,} shift rows, {len(shots_df):,} shots, "
          f"{args.games:,} games simulated")

    model = RAPMModel()
    # RAPM reads names off the shot feed, which real MoneyPuck data carries and a
    # simulation does not; without this every row reads "player_8478402".
    model.name_fallback = {p["player_id"]: p["name"] for p in players}

    rapm_df = model.fit_season(season, shifts_df, shots_df)
    out = DATA_DIR / "rapm" / f"rapm_{season}.parquet"
    rapm_df.write_parquet(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)} — {len(rapm_df)} players")

    war_df = build_war(rapm_df, season)
    war_out = DATA_DIR / "war" / f"war_{season}.parquet"
    war_df.write_parquet(war_out)
    print(f"  wrote {war_out.relative_to(REPO_ROOT)} — {len(war_df)} players")

    # Did the estimator recover the rating it never saw? A pipeline that quietly
    # produced noise would still render a convincing-looking leaderboard.
    ints = {int(k): v for k, v in skill_map.items()}
    joined = rapm_df.with_columns(
        pl.col("player_id")
          .map_elements(lambda i: ints.get(i), return_dtype=pl.Float64)
          .alias("true_skill")
    ).drop_nulls(["true_skill", "rapm_ev_off"])
    if len(joined) > 2:
        corr = joined.select(pl.corr("rapm_ev_off", "true_skill")).item()
        print(f"  corr(rapm_ev_off, production rating) = {corr:+.3f}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
