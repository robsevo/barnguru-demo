"""Tests for models/war_model.py — Feature 2.25."""
from __future__ import annotations

import warnings

import polars as pl
import pytest

from models.rapm_model import DataMissingWarning
from models.war_model import (
    GOALS_PER_WIN,
    MODEL_VERSION,
    WAR_DOLLAR_RATE,
    WAR_SCHEMA,
    WARModel,
    contract_efficiency,
    gar_from_rapm,
    war_from_gar,
    write_war,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rapm_row(pid=1, season=2025, player_name="Test Player", team="TOR",
              toi_ev=800.0, toi_pp=80.0, toi_pk=60.0,
              rapm_ev_off=1.5, rapm_ev_def=0.8, rapm_pp=1.0, rapm_pk=0.5,
              position="C"):
    return {
        "player_id": pid, "player_name": player_name, "team": team,
        "position": position, "season": season,
        "toi_ev": toi_ev, "toi_pp": toi_pp, "toi_pk": toi_pk,
        "rapm_ev_off": rapm_ev_off, "rapm_ev_def": rapm_ev_def,
        "rapm_pp": rapm_pp, "rapm_pk": rapm_pk,
    }


def _make_rapm(rows: list[dict]) -> pl.DataFrame:
    base = {
        "player_id": pl.Int64, "player_name": pl.Utf8, "team": pl.Utf8,
        "position": pl.Utf8, "season": pl.Int64,
        "toi_ev": pl.Float64, "toi_pp": pl.Float64, "toi_pk": pl.Float64,
        "rapm_ev_off": pl.Float64, "rapm_ev_def": pl.Float64,
        "rapm_pp": pl.Float64, "rapm_pk": pl.Float64,
    }
    if not rows:
        return pl.DataFrame(schema=base)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# gar_from_rapm
# ---------------------------------------------------------------------------

class TestGarFromRapm:
    def test_zero_toi_returns_zero_gar(self):
        gar_ev, gar_pp, gar_pk = gar_from_rapm(1.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0)
        assert gar_ev == pytest.approx(0.0)
        assert gar_pp == pytest.approx(0.0)
        assert gar_pk == pytest.approx(0.0)

    def test_average_player_returns_positive_gar(self):
        # average player RAPM = 0.0 at EV; replacement = -2.0 → above replacement
        gar_ev, _, _ = gar_from_rapm(0.0, 0.0, 0.0, 0.0, 600.0, 0.0, 0.0)
        # (0.0 - (-2.0)) × (600/60) = 2.0 × 10 = 20.0
        assert gar_ev == pytest.approx(20.0)

    def test_none_rapm_treated_as_zero(self):
        gar_ev, gar_pp, gar_pk = gar_from_rapm(None, None, None, None, 600.0, 80.0, 60.0)
        # None → 0.0; still above replacement
        assert gar_ev > 0

    def test_positive_rapm_gives_higher_gar(self):
        gar_low,  _, _ = gar_from_rapm(-1.0, -1.0, 0.0, 0.0, 600.0, 0.0, 0.0)
        gar_high, _, _ = gar_from_rapm(2.0,  1.0,  0.0, 0.0, 600.0, 0.0, 0.0)
        assert gar_high > gar_low

    def test_pp_gar_scales_with_toi_pp(self):
        _, gar_pp_small, _ = gar_from_rapm(0.0, 0.0, 1.0, 0.0, 600.0, 30.0, 0.0)
        _, gar_pp_large, _ = gar_from_rapm(0.0, 0.0, 1.0, 0.0, 600.0, 90.0, 0.0)
        assert gar_pp_large > gar_pp_small


# ---------------------------------------------------------------------------
# war_from_gar
# ---------------------------------------------------------------------------

class TestWarFromGar:
    def test_zero_gar_returns_zero_war(self):
        assert war_from_gar(0.0) == pytest.approx(0.0)

    def test_five_gar_returns_one_war(self):
        assert war_from_gar(5.0, goals_per_win=5.0) == pytest.approx(1.0)

    def test_negative_gar_returns_negative_war(self):
        assert war_from_gar(-10.0) < 0

    def test_custom_goals_per_win(self):
        assert war_from_gar(10.0, goals_per_win=2.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# contract_efficiency
# ---------------------------------------------------------------------------

class TestContractEfficiency:
    def test_efficient_contract(self):
        # WAR = 4.0 → value = 4.0 × 1.2 = $4.8M; cap_hit = $2.0M → eff = 2.4
        eff = contract_efficiency(4.0, 2.0)
        assert eff == pytest.approx(4.0 * WAR_DOLLAR_RATE / 2.0)

    def test_none_cap_hit_returns_none(self):
        assert contract_efficiency(4.0, None) is None

    def test_zero_cap_hit_returns_none(self):
        assert contract_efficiency(4.0, 0.0) is None

    def test_negative_cap_hit_returns_none(self):
        assert contract_efficiency(4.0, -1.0) is None

    def test_overpaid_player_efficiency_below_one(self):
        # WAR = 1.0 → value = $1.2M; cap_hit = $5.0M → eff = 0.24
        eff = contract_efficiency(1.0, 5.0)
        assert eff is not None
        assert eff < 1.0


# ---------------------------------------------------------------------------
# WARModel.fit — basic
# ---------------------------------------------------------------------------

class TestWARModelFit:
    def test_empty_input_returns_empty_schema(self):
        model = WARModel()
        df = model.fit(_make_rapm([]))
        assert df.is_empty()
        for col in WAR_SCHEMA:
            assert col in df.columns

    def test_single_player(self):
        rapm = _make_rapm([_rapm_row(pid=1)])
        model = WARModel()
        df = model.fit(rapm)
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["player_id"] == 1
        assert row["model_version"] == MODEL_VERSION

    def test_output_columns_match_schema(self):
        rapm = _make_rapm([_rapm_row()])
        model = WARModel()
        df = model.fit(rapm)
        assert set(df.columns) == set(WAR_SCHEMA.keys())

    def test_war_computed(self):
        rapm = _make_rapm([_rapm_row(rapm_ev_off=2.0, rapm_ev_def=1.0, toi_ev=600.0,
                                     toi_pp=0.0, toi_pk=0.0)])
        model = WARModel()
        df = model.fit(rapm)
        row = df.row(0, named=True)
        # ev_off_adj =   2.0 × 1.6 =  3.2     (MULTIPLIER_EV_OFF)
        # ev_def_adj = -(1.0) × 1.4 = -1.4    (sign flip: +ve RAPM_def = worse defense)
        # gar_ev = (3.2 + -1.4 - (-2.0)) × (600/60) = 3.8 × 10 = 38.0
        # war    = 38.0 / 5.5 ≈ 6.9091         (GOALS_PER_WIN)
        assert row["gar_ev"] == pytest.approx(38.0)
        assert row["war"]    == pytest.approx(38.0 / GOALS_PER_WIN)

    def test_missing_player_id_raises(self):
        bad = pl.DataFrame({"name": ["X"]})
        model = WARModel()
        with pytest.raises(ValueError, match="missing required columns"):
            model.fit(bad)

    def test_no_rapm_warns(self):
        row = _rapm_row()
        row["rapm_ev_off"] = None
        row["rapm_ev_def"] = None
        row["rapm_pp"]     = None
        row["rapm_pk"]     = None
        rapm = pl.DataFrame([row])
        model = WARModel()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(rapm)
        assert any(issubclass(x.category, DataMissingWarning) for x in w)

    def test_multiple_players(self):
        rows = [_rapm_row(pid=i) for i in range(5)]
        model = WARModel()
        df = model.fit(_make_rapm(rows))
        assert len(df) == 5

    def test_better_rapm_gives_higher_war(self):
        rows = [
            _rapm_row(pid=1, rapm_ev_off=-1.0, rapm_ev_def=-1.0),
            _rapm_row(pid=2, rapm_ev_off=3.0,  rapm_ev_def=2.0),
        ]
        model = WARModel()
        df = model.fit(_make_rapm(rows))
        war_by_pid = {r["player_id"]: r["war"] for r in df.to_dicts()}
        assert war_by_pid[2] > war_by_pid[1]

    def test_toi_alias_toi_ev_min(self):
        """toi_ev_min should be treated same as toi_ev."""
        row = _rapm_row()
        row["toi_ev_min"] = row.pop("toi_ev")
        row["toi_pp_min"] = row.pop("toi_pp")
        row["toi_pk_min"] = row.pop("toi_pk")
        model = WARModel()
        df = model.fit(pl.DataFrame([row]))
        assert len(df) == 1
        assert df["war"][0] is not None


# ---------------------------------------------------------------------------
# Optional finishing_df and cap_df
# ---------------------------------------------------------------------------

class TestOptionalInputs:
    def test_finishing_df_adds_gar(self):
        rapm = _make_rapm([_rapm_row(pid=1, toi_pp=0.0, toi_pk=0.0)])
        fin  = pl.DataFrame([{"player_id": 1, "season": 2025, "finishing_delta": 5.0}])
        model_with    = WARModel()
        model_without = WARModel()
        df_with    = model_with.fit(rapm, finishing_df=fin)
        df_without = model_without.fit(rapm)
        assert df_with["gar_total"][0] == pytest.approx(df_without["gar_total"][0] + 5.0)

    def test_cap_df_computes_efficiency(self):
        rapm = _make_rapm([_rapm_row(pid=1)])
        cap  = pl.DataFrame([{"player_id": 1, "season": 2025, "cap_hit": 3.0}])
        model = WARModel()
        df = model.fit(rapm, cap_df=cap)
        row = df.row(0, named=True)
        assert row["cap_hit"] == pytest.approx(3.0)
        assert row["contract_efficiency"] is not None
        expected = row["war"] * WAR_DOLLAR_RATE / 3.0
        assert row["contract_efficiency"] == pytest.approx(expected)

    def test_no_cap_efficiency_is_null(self):
        rapm = _make_rapm([_rapm_row(pid=1)])
        model = WARModel()
        df = model.fit(rapm)
        assert df["contract_efficiency"][0] is None

    def test_finishing_df_season_mismatch_ignored(self):
        rapm = _make_rapm([_rapm_row(pid=1, season=2025)])
        fin  = pl.DataFrame([{"player_id": 1, "season": 2024, "finishing_delta": 10.0}])
        model = WARModel()
        df = model.fit(rapm, finishing_df=fin)
        # Season mismatch → finishing_delta not applied
        assert df["gar_finishing"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

class TestLookup:
    def _trained(self):
        rapm = _make_rapm([_rapm_row(pid=1)])
        model = WARModel()
        model.fit(rapm)
        return model

    def test_get_war_returns_float(self):
        model = self._trained()
        val = model.get_war(1)
        assert isinstance(val, float)

    def test_get_war_unknown_returns_none(self):
        model = WARModel()
        assert model.get_war(9999) is None

    def test_get_gar_returns_float(self):
        model = self._trained()
        assert isinstance(model.get_gar(1), float)

    def test_get_contract_efficiency_no_cap_returns_none(self):
        model = self._trained()
        assert model.get_contract_efficiency(1) is None

    def test_league_war_percentile_empty_returns_none(self):
        model = WARModel()
        assert model.league_war_percentile(1.0) is None

    def test_league_war_percentile_top_player(self):
        rows = [_rapm_row(pid=i, rapm_ev_off=float(i)) for i in range(10)]
        model = WARModel()
        model.fit(_make_rapm(rows))
        # pid=9 has highest WAR
        top_war = model.get_war(9)
        pct = model.league_war_percentile(top_war)
        assert pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        rapm = _make_rapm([_rapm_row(pid=1)])
        model = WARModel()
        model.fit(rapm)
        path = tmp_path / "war.pkl"
        model.save(path)
        loaded = WARModel.load(path)
        assert loaded.get_war(1) == pytest.approx(model.get_war(1))

    def test_load_preserves_goals_per_win(self, tmp_path):
        model = WARModel(goals_per_win=4.5)
        model.save(tmp_path / "w.pkl")
        loaded = WARModel.load(tmp_path / "w.pkl")
        assert loaded._goals_per_win == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# write_war
# ---------------------------------------------------------------------------

class TestWriteWar:
    def test_writes_parquet(self, tmp_path):
        rapm = _make_rapm([_rapm_row()])
        model = WARModel()
        df = model.fit(rapm)
        path = write_war(df, tmp_path / "out", season=2025)
        assert path.exists()
        assert path.name == "war_2025.parquet"

    def test_readable_back(self, tmp_path):
        rapm = _make_rapm([_rapm_row()])
        model = WARModel()
        df = model.fit(rapm)
        path = write_war(df, tmp_path / "out", season=2025)
        back = pl.read_parquet(path)
        assert len(back) == 1

    def test_creates_output_dir(self, tmp_path):
        df = pl.DataFrame(schema={c: d for c, d in WAR_SCHEMA.items()})
        write_war(df, tmp_path / "deep" / "dir", season=2024)
        assert (tmp_path / "deep" / "dir").exists()
