"""Tests for PolymarketParser (feature 1.9).

Covers:
- Known-answer regression (frozen Gamma API JSON → expected MarketRecord list)
- Price parsing: float conversion, range checks, sum check
- Missing fields: conditionId (skip), outcomePrices (warn+keep), clobTokenIds (warn+keep)
- Edge cases: empty input, to_polars
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from data.polymarket_parser import EMPTY_SCHEMA, MarketRecord, PolymarketParser
from data.pbp_parser import DataMissingWarning


# ---------------------------------------------------------------------------
# Frozen test fixtures — never change without updating expected values
# ---------------------------------------------------------------------------

_FETCHED_AT = "2026-03-21T10:00:00+00:00"

_FROZEN_MARKETS: list[dict] = [
    {
        "conditionId": "0xaaa111",
        "id": "1",
        "question": "Will the Rangers beat the Bruins on March 21?",
        "slug": "rangers-vs-bruins-mar-21",
        "clobTokenIds": ["0xtoken_yes_1", "0xtoken_no_1"],
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.62", "0.38"],
        "volume": "125000.50",
        "liquidity": "50000.00",
        "endDateIso": "2026-03-22T00:00:00Z",
        "active": True,
        "closed": False,
    },
    {
        "conditionId": "0xbbb222",
        "id": "2",
        "question": "Will the Leafs beat the Habs on March 21?",
        "slug": "leafs-vs-habs-mar-21",
        "clobTokenIds": ["0xtoken_yes_2", "0xtoken_no_2"],
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.55", "0.45"],
        "volume": "88000.00",
        "liquidity": "32000.00",
        "endDateIso": "2026-03-22T00:00:00Z",
        "active": True,
        "closed": False,
    },
]


# ===========================================================================
# Known-answer regression
# ===========================================================================


class TestPolymarketParserKnownAnswer:
    def setup_method(self):
        self.records = PolymarketParser.parse(_FROZEN_MARKETS, _FETCHED_AT)

    def test_two_records_parsed(self):
        assert len(self.records) == 2

    def test_first_condition_id(self):
        assert self.records[0].condition_id == "0xaaa111"

    def test_first_question(self):
        assert "Rangers" in self.records[0].question

    def test_first_price_yes_float(self):
        assert self.records[0].price_yes == pytest.approx(0.62)

    def test_first_price_no_float(self):
        assert self.records[0].price_no == pytest.approx(0.38)

    def test_first_token_id_yes(self):
        assert self.records[0].token_id_yes == "0xtoken_yes_1"

    def test_first_token_id_no(self):
        assert self.records[0].token_id_no == "0xtoken_no_1"

    def test_first_outcome_yes(self):
        assert self.records[0].outcome_yes == "Yes"

    def test_first_outcome_no(self):
        assert self.records[0].outcome_no == "No"

    def test_first_volume(self):
        assert self.records[0].volume == pytest.approx(125000.50)

    def test_first_liquidity(self):
        assert self.records[0].liquidity == pytest.approx(50000.00)

    def test_first_end_date(self):
        assert self.records[0].end_date_iso == "2026-03-22T00:00:00Z"

    def test_first_is_active_true(self):
        assert self.records[0].is_active is True

    def test_first_is_closed_false(self):
        assert self.records[0].is_closed is False

    def test_fetched_at_propagated(self):
        for r in self.records:
            assert r.fetched_at == _FETCHED_AT

    def test_second_record_prices(self):
        assert self.records[1].price_yes == pytest.approx(0.55)
        assert self.records[1].price_no == pytest.approx(0.45)


# ===========================================================================
# Price parsing
# ===========================================================================


class TestPriceParsing:
    def _single_market(self, outcome_prices=None, clobTokenIds=None):
        return [
            {
                "conditionId": "0xtest",
                "id": "99",
                "question": "Test?",
                "slug": "test",
                "clobTokenIds": clobTokenIds or ["0xt1", "0xt2"],
                "outcomes": ["Yes", "No"],
                "outcomePrices": outcome_prices,
                "volume": "1000",
                "liquidity": "500",
                "endDateIso": "2026-01-01T00:00:00Z",
                "active": True,
                "closed": False,
            }
        ]

    def test_price_string_to_float(self):
        records = PolymarketParser.parse(self._single_market(["0.70", "0.30"]), _FETCHED_AT)
        assert records[0].price_yes == pytest.approx(0.70)
        assert records[0].price_no == pytest.approx(0.30)

    def test_missing_outcome_prices_warns_nones(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(self._single_market(None), _FETCHED_AT)
        assert len(records) == 1
        assert records[0].price_yes is None
        assert records[0].price_no is None
        assert any("outcomePrices" in str(w.message) for w in caught)

    def test_outcome_prices_len_1_warns_nones(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(self._single_market(["0.70"]), _FETCHED_AT)
        assert records[0].price_yes is None
        assert any("outcomePrices" in str(w.message) for w in caught)

    def test_malformed_price_string_warns_none(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(self._single_market(["abc", "0.30"]), _FETCHED_AT)
        assert records[0].price_yes is None
        assert records[0].price_no == pytest.approx(0.30)
        assert any("cannot be parsed as float" in str(w.message) for w in caught)

    def test_price_out_of_range_warns_keeps_value(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(self._single_market(["1.2", "0.30"]), _FETCHED_AT)
        assert records[0].price_yes == pytest.approx(1.2)  # kept
        assert any("outside [0, 1]" in str(w.message) for w in caught)

    def test_price_sum_mismatch_warns_keeps_record(self):
        # 0.70 + 0.20 = 0.90, abs(0.90 - 1.0) = 0.10 > 0.05
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(self._single_market(["0.70", "0.20"]), _FETCHED_AT)
        assert len(records) == 1
        assert records[0].price_yes == pytest.approx(0.70)
        assert any("expected ~1.0" in str(w.message) for w in caught)

    def test_price_sum_within_tolerance_no_warning(self):
        # 0.62 + 0.38 = 1.00, within tolerance
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            PolymarketParser.parse(self._single_market(["0.62", "0.38"]), _FETCHED_AT)
        sum_warnings = [w for w in caught if "expected ~1.0" in str(w.message)]
        assert sum_warnings == []

    def test_empty_string_price_warns_none(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(self._single_market(["", "0.40"]), _FETCHED_AT)
        assert records[0].price_yes is None
        assert any("missing" in str(w.message).lower() for w in caught)


# ===========================================================================
# Missing critical fields
# ===========================================================================


class TestMissingCriticalFields:
    def test_missing_condition_id_skips_record(self):
        market = {
            "id": "1",
            "question": "Test?",
            "slug": "test",
            "outcomePrices": ["0.5", "0.5"],
            "active": True,
            "closed": False,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse([market], _FETCHED_AT)
        assert records == []
        assert any("conditionId" in str(w.message) for w in caught)

    def test_empty_condition_id_skips_record(self):
        market = {"conditionId": "", "id": "1", "question": "Test?", "slug": "test",
                  "active": True, "closed": False}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse([market], _FETCHED_AT)
        assert records == []

    def test_missing_clob_token_ids_warns_keeps_record(self):
        market = {
            "conditionId": "0xaaa",
            "id": "1",
            "question": "Test?",
            "slug": "test",
            "outcomePrices": ["0.6", "0.4"],
            "outcomes": ["Yes", "No"],
            "active": True,
            "closed": False,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse([market], _FETCHED_AT)
        assert len(records) == 1
        assert records[0].token_id_yes is None
        assert records[0].token_id_no is None
        assert any("clobTokenIds" in str(w.message) for w in caught)

    def test_clob_token_ids_len_1_warns(self):
        market = {
            "conditionId": "0xaaa",
            "id": "1",
            "question": "Test?",
            "slug": "test",
            "clobTokenIds": ["0xt1"],
            "outcomePrices": ["0.6", "0.4"],
            "active": True,
            "closed": False,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse([market], _FETCHED_AT)
        assert len(records) == 1
        assert any("clobTokenIds" in str(w.message) for w in caught)

    def test_one_bad_record_does_not_prevent_good(self):
        markets = [
            {"id": "1", "question": "Bad?"},  # missing conditionId
            {
                "conditionId": "0xgood",
                "id": "2",
                "question": "Good?",
                "slug": "good",
                "outcomePrices": ["0.6", "0.4"],
                "clobTokenIds": ["0xt1", "0xt2"],
                "active": True,
                "closed": False,
            },
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", DataMissingWarning)
            records = PolymarketParser.parse(markets, _FETCHED_AT)
        assert len(records) == 1
        assert records[0].condition_id == "0xgood"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_list_returns_empty(self):
        records = PolymarketParser.parse([], _FETCHED_AT)
        assert records == []

    def test_volume_none_when_missing(self):
        market = {
            "conditionId": "0xaaa",
            "id": "1",
            "question": "Test?",
            "slug": "test",
            "clobTokenIds": ["0xt1", "0xt2"],
            "outcomePrices": ["0.6", "0.4"],
            "active": True,
            "closed": False,
            # no "volume" or "liquidity"
        }
        records = PolymarketParser.parse([market], _FETCHED_AT)
        assert records[0].volume is None
        assert records[0].liquidity is None

    def test_end_date_none_when_missing(self):
        market = {
            "conditionId": "0xaaa",
            "id": "1",
            "question": "Test?",
            "slug": "test",
            "clobTokenIds": ["0xt1", "0xt2"],
            "outcomePrices": ["0.6", "0.4"],
            "active": True,
            "closed": False,
        }
        records = PolymarketParser.parse([market], _FETCHED_AT)
        assert records[0].end_date_iso is None

    def test_is_closed_true(self):
        market = {
            "conditionId": "0xclosed",
            "id": "3",
            "question": "Closed?",
            "slug": "closed",
            "outcomePrices": ["1.0", "0.0"],
            "clobTokenIds": ["0xt1", "0xt2"],
            "active": False,
            "closed": True,
        }
        records = PolymarketParser.parse([market], _FETCHED_AT)
        assert records[0].is_active is False
        assert records[0].is_closed is True

    def test_numeric_volume_parsed(self):
        market = {
            "conditionId": "0xaaa",
            "id": "1",
            "question": "Test?",
            "slug": "test",
            "clobTokenIds": ["0xt1", "0xt2"],
            "outcomePrices": ["0.6", "0.4"],
            "volume": "99999.99",
            "liquidity": "12345.00",
            "active": True,
            "closed": False,
        }
        records = PolymarketParser.parse([market], _FETCHED_AT)
        assert records[0].volume == pytest.approx(99999.99)
        assert records[0].liquidity == pytest.approx(12345.00)


# ===========================================================================
# to_polars
# ===========================================================================


class TestToPolars:
    def test_empty_list_returns_empty_dataframe(self):
        df = PolymarketParser.to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_two_records_gives_two_rows(self):
        records = PolymarketParser.parse(_FROZEN_MARKETS, _FETCHED_AT)
        df = PolymarketParser.to_polars(records)
        assert len(df) == 2

    def test_all_schema_columns_present(self):
        records = PolymarketParser.parse(_FROZEN_MARKETS, _FETCHED_AT)
        df = PolymarketParser.to_polars(records)
        for col in EMPTY_SCHEMA:
            assert col in df.columns, f"Missing column: {col}"

    def test_price_yes_is_float_type(self):
        records = PolymarketParser.parse(_FROZEN_MARKETS, _FETCHED_AT)
        df = PolymarketParser.to_polars(records)
        assert df["price_yes"].dtype in (pl.Float64, pl.Float32)

    def test_is_active_is_boolean(self):
        records = PolymarketParser.parse(_FROZEN_MARKETS, _FETCHED_AT)
        df = PolymarketParser.to_polars(records)
        assert df["is_active"].dtype == pl.Boolean

    def test_condition_ids_unique(self):
        records = PolymarketParser.parse(_FROZEN_MARKETS, _FETCHED_AT)
        df = PolymarketParser.to_polars(records)
        assert df["condition_id"].n_unique() == len(records)
