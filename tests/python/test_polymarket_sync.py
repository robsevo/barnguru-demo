"""Tests for PolymarketSync (feature 1.9).

Covers:
- Successful sync: ok status, market_count, Parquet written
- TTL skip logic (within TTL → skip, past TTL → re-fetch)
- Error handling (error_fetch, error_parse)
- Empty markets → empty-schema Parquet
- get_markets(), get_market(), clear_errors()
- Integration smoke: sync → get_markets → correct columns
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import polars as pl
import pytest

from data.polymarket_client import PolymarketClient, PolymarketError
from data.polymarket_parser import EMPTY_SCHEMA
from data.polymarket_sync import MarketSyncResult, PolymarketSync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

_MINIMAL_MARKETS = [
    {
        "conditionId": "0xaaa111",
        "id": "1",
        "question": "Will the Rangers beat the Bruins on March 21?",
        "slug": "rangers-vs-bruins",
        "clobTokenIds": ["0xt1", "0xt2"],
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.62", "0.38"],
        "volume": "100000",
        "liquidity": "50000",
        "endDateIso": "2026-03-22T00:00:00Z",
        "active": True,
        "closed": False,
    },
    {
        "conditionId": "0xbbb222",
        "id": "2",
        "question": "Will the Leafs beat the Habs on March 21?",
        "slug": "leafs-vs-habs",
        "clobTokenIds": ["0xt3", "0xt4"],
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.55", "0.45"],
        "volume": "80000",
        "liquidity": "40000",
        "endDateIso": "2026-03-22T00:00:00Z",
        "active": True,
        "closed": False,
    },
]


# ===========================================================================
# Successful sync
# ===========================================================================


class TestPolymarketSyncSuccess:
    def test_sync_ok_status(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                result = sync.sync(client)
        assert result.status == "ok"

    def test_sync_market_count(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                result = sync.sync(client)
        assert result.market_count == 2

    def test_sync_writes_parquet(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        assert (tmp_path / f"markets_{_TODAY}.parquet").exists()

    def test_sync_updates_manifest(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        manifest = sync.get_manifest()
        assert manifest["dates"][_TODAY]["status"] == "ok"
        assert "synced_at" in manifest["dates"][_TODAY]

    def test_sync_fetch_date_is_today(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                result = sync.sync(client)
        assert result.fetch_date == _TODAY


# ===========================================================================
# TTL / skip logic
# ===========================================================================


class TestPolymarketSyncTTL:
    def test_second_call_within_ttl_returns_skipped(self, tmp_path):
        sync = PolymarketSync(tmp_path, ttl_hours=1.0)
        call_count = 0

        def mock_fetch(active_only=True):
            nonlocal call_count
            call_count += 1
            return _MINIMAL_MARKETS

        with patch.object(PolymarketClient, "fetch_nhl_markets", side_effect=mock_fetch):
            with PolymarketClient() as client:
                r1 = sync.sync(client)
                r2 = sync.sync(client)

        assert r1.status == "ok"
        assert r2.status == "skipped"
        assert call_count == 1

    def test_skipped_preserves_market_count(self, tmp_path):
        sync = PolymarketSync(tmp_path, ttl_hours=1.0)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                r1 = sync.sync(client)
                r2 = sync.sync(client)
        assert r2.market_count == r1.market_count

    def test_expired_ttl_refetches(self, tmp_path):
        sync = PolymarketSync(tmp_path, ttl_hours=1.0)
        call_count = 0

        def mock_fetch(active_only=True):
            nonlocal call_count
            call_count += 1
            return _MINIMAL_MARKETS

        stale_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                _TODAY: {
                    "status": "ok",
                    "market_count": 2,
                    "warning_count": 0,
                    "synced_at": stale_time,
                }
            },
        }))

        with patch.object(PolymarketClient, "fetch_nhl_markets", side_effect=mock_fetch):
            with PolymarketClient() as client:
                result = sync.sync(client)

        assert result.status == "ok"
        assert call_count == 1

    def test_corrupt_synced_at_triggers_refetch(self, tmp_path):
        sync = PolymarketSync(tmp_path, ttl_hours=1.0)
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                _TODAY: {
                    "status": "ok",
                    "market_count": 2,
                    "warning_count": 0,
                    "synced_at": "NOT_A_TIMESTAMP",
                }
            },
        }))

        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                result = sync.sync(client)

        assert result.status == "ok"


# ===========================================================================
# Error handling
# ===========================================================================


class TestPolymarketSyncErrors:
    def test_fetch_error_returns_error_fetch_status(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(
            PolymarketClient,
            "fetch_nhl_markets",
            side_effect=PolymarketError("HTTP 503", status_code=503),
        ):
            with PolymarketClient() as client:
                result = sync.sync(client)
        assert result.status == "error_fetch"

    def test_fetch_error_updates_manifest(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(
            PolymarketClient,
            "fetch_nhl_markets",
            side_effect=PolymarketError("HTTP 503", status_code=503),
        ):
            with PolymarketClient() as client:
                sync.sync(client)
        assert sync.get_manifest()["dates"][_TODAY]["status"] == "error_fetch"

    def test_fetch_error_message_stored(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(
            PolymarketClient,
            "fetch_nhl_markets",
            side_effect=PolymarketError("HTTP 503", status_code=503),
        ):
            with PolymarketClient() as client:
                result = sync.sync(client)
        assert result.error_message is not None
        assert "503" in result.error_message


# ===========================================================================
# Empty markets
# ===========================================================================


class TestPolymarketSyncEmpty:
    def test_empty_markets_writes_empty_schema_parquet(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=[]):
            with PolymarketClient() as client:
                result = sync.sync(client)

        assert result.status == "empty"
        parquet_path = tmp_path / f"markets_{_TODAY}.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert len(df) == 0
        for col in EMPTY_SCHEMA:
            assert col in df.columns

    def test_empty_result_market_count_zero(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=[]):
            with PolymarketClient() as client:
                result = sync.sync(client)
        assert result.market_count == 0


# ===========================================================================
# get_markets
# ===========================================================================


class TestGetMarkets:
    def test_get_markets_returns_dataframe(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        df = sync.get_markets()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

    def test_get_markets_missing_date_returns_empty(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        df = sync.get_markets("1990-01-01")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_get_markets_columns_match_schema(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        df = sync.get_markets()
        for col in EMPTY_SCHEMA:
            assert col in df.columns

    def test_get_markets_specific_date(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        df = sync.get_markets(_TODAY)
        assert len(df) == 2


# ===========================================================================
# get_market
# ===========================================================================


class TestGetMarket:
    def test_known_condition_id_returns_record(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        record = sync.get_market("0xaaa111")
        assert record is not None
        assert record.condition_id == "0xaaa111"
        assert record.price_yes == pytest.approx(0.62)

    def test_unknown_condition_id_returns_none(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        record = sync.get_market("0xnonexistent")
        assert record is None

    def test_no_parquet_for_date_returns_none(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        record = sync.get_market("0xaaa111", date="1990-01-01")
        assert record is None

    def test_get_market_correct_prices(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        record = sync.get_market("0xbbb222")
        assert record is not None
        assert record.price_yes == pytest.approx(0.55)
        assert record.price_no == pytest.approx(0.45)


# ===========================================================================
# clear_errors
# ===========================================================================


class TestClearErrors:
    def test_clear_errors_removes_error_fetch(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(
            PolymarketClient,
            "fetch_nhl_markets",
            side_effect=PolymarketError("503", status_code=503),
        ):
            with PolymarketClient() as client:
                sync.sync(client)
        removed = sync.clear_errors()
        assert removed == 1
        assert _TODAY not in sync.get_manifest()["dates"]

    def test_clear_errors_keeps_ok(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        removed = sync.clear_errors()
        assert removed == 0
        assert sync.get_manifest()["dates"][_TODAY]["status"] == "ok"

    def test_clear_errors_removes_empty(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=[]):
            with PolymarketClient() as client:
                sync.sync(client)
        removed = sync.clear_errors()
        assert removed == 1

    def test_clear_errors_count(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "dates": {
                "2026-01-01": {"status": "error_fetch", "market_count": 0, "warning_count": 0, "synced_at": ""},
                "2026-01-02": {"status": "error_parse", "market_count": 0, "warning_count": 0, "synced_at": ""},
                "2026-01-03": {"status": "ok", "market_count": 5, "warning_count": 0, "synced_at": ""},
            },
        }))
        removed = sync.clear_errors()
        assert removed == 2
        assert "2026-01-03" in sync.get_manifest()["dates"]


# ===========================================================================
# Integration smoke
# ===========================================================================


class TestPolymarketSyncIntegration:
    def test_full_pipeline(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                result = sync.sync(client)

        assert result.status == "ok"
        assert result.market_count == 2

        df = sync.get_markets()
        assert "condition_id" in df.columns
        assert "price_yes" in df.columns
        assert "price_no" in df.columns
        assert len(df) == 2

    def test_prices_are_floats_in_parquet(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        df = sync.get_markets()
        assert df["price_yes"].dtype in (pl.Float64, pl.Float32)

    def test_filter_active_markets(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        with patch.object(PolymarketClient, "fetch_nhl_markets", return_value=_MINIMAL_MARKETS):
            with PolymarketClient() as client:
                sync.sync(client)
        df = sync.get_markets()
        active = df.filter(pl.col("is_active") == True)  # noqa: E712
        assert len(active) == 2

    def test_get_manifest_structure(self, tmp_path):
        sync = PolymarketSync(tmp_path)
        assert sync.get_manifest() == {"format_version": 1, "dates": {}}
