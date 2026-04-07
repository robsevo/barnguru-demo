"""Tests for PolymarketClient (feature 1.9).

Covers:
- fetch_nhl_tag_ids: success, NHL not found
- fetch_nhl_markets: pagination, deduplication on conditionId
- HTTP error handling (404 no-retry, 5xx retry, 429 Retry-After, timeout)
- Context manager
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from data.polymarket_client import PolymarketClient, PolymarketError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SPORTS_RESPONSE = [
    {"sport": "NHL", "tags": "101,102", "image": "nhl.png", "resolution": "nhl.com"},
    {"sport": "NBA", "tags": "201", "image": "nba.png", "resolution": "nba.com"},
]

_MARKETS_PAGE_1 = [
    {
        "conditionId": f"0x{i:04d}",
        "id": str(i),
        "question": f"Market {i}",
        "slug": f"market-{i}",
        "active": True,
        "closed": False,
    }
    for i in range(100)
]

_MARKETS_PAGE_2 = [
    {
        "conditionId": f"0x{i:04d}",
        "id": str(i),
        "question": f"Market {i}",
        "slug": f"market-{i}",
        "active": True,
        "closed": False,
    }
    for i in range(100, 150)
]


def _make_json_response(data: object, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.is_error = status >= 400
    resp.headers = {}
    resp.json.return_value = data
    return resp


# ===========================================================================
# Context manager
# ===========================================================================


class TestPolymarketClientContextManager:
    def test_enter_returns_self(self):
        client = PolymarketClient()
        assert client.__enter__() is client
        client.close()

    def test_with_block_closes_client(self):
        with PolymarketClient() as client:
            assert client._client is not None


# ===========================================================================
# fetch_nhl_tag_ids
# ===========================================================================


class TestFetchNhlTagIds:
    def test_returns_tag_ids_for_nhl(self):
        with patch.object(
            httpx.Client, "get", return_value=_make_json_response(_SPORTS_RESPONSE)
        ):
            with PolymarketClient() as client:
                tag_ids = client.fetch_nhl_tag_ids()
        assert "101" in tag_ids
        assert "102" in tag_ids

    def test_nhl_not_found_raises(self):
        no_nhl = [{"sport": "NBA", "tags": "201"}]
        with patch.object(
            httpx.Client, "get", return_value=_make_json_response(no_nhl)
        ):
            with PolymarketClient() as client:
                with pytest.raises(PolymarketError, match="NHL sport not found"):
                    client.fetch_nhl_tag_ids()

    def test_empty_sports_list_raises(self):
        with patch.object(
            httpx.Client, "get", return_value=_make_json_response([])
        ):
            with PolymarketClient() as client:
                with pytest.raises(PolymarketError):
                    client.fetch_nhl_tag_ids()

    def test_comma_separated_tags_split_correctly(self):
        sports = [{"sport": "NHL", "tags": "10,20,30"}]
        with patch.object(
            httpx.Client, "get", return_value=_make_json_response(sports)
        ):
            with PolymarketClient() as client:
                tag_ids = client.fetch_nhl_tag_ids()
        assert tag_ids == ["10", "20", "30"]

    def test_single_tag_returned_as_list(self):
        sports = [{"sport": "NHL", "tags": "99"}]
        with patch.object(
            httpx.Client, "get", return_value=_make_json_response(sports)
        ):
            with PolymarketClient() as client:
                tag_ids = client.fetch_nhl_tag_ids()
        assert tag_ids == ["99"]

    def test_nhl_case_insensitive(self):
        sports = [{"sport": "nhl", "tags": "55"}]
        with patch.object(
            httpx.Client, "get", return_value=_make_json_response(sports)
        ):
            with PolymarketClient() as client:
                tag_ids = client.fetch_nhl_tag_ids()
        assert tag_ids == ["55"]


# ===========================================================================
# fetch_nhl_markets — pagination and deduplication
# ===========================================================================


class TestFetchNhlMarkets:
    def test_single_page_returns_all(self):
        small_page = _MARKETS_PAGE_2  # 50 items, less than page_size=100 → last page

        call_count = 0

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if "/sports" in url:
                return _make_json_response(_SPORTS_RESPONSE)
            # Markets endpoint — return small_page regardless of tag_id
            return _make_json_response(small_page)

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with PolymarketClient(page_size=100) as client:
                markets = client.fetch_nhl_markets()

        # Two tag IDs (101, 102) × 1 page each = 2 markets calls + 1 sports call
        # But deduplication on conditionId means 50 unique markets (same data from both tags)
        assert len(markets) == 50

    def test_pagination_two_pages(self):
        pages: dict[str, list[dict]] = {}  # tag_id → current page index

        def mock_get(url, params=None, **kwargs):
            if "/sports" in url:
                # Single tag for simplicity
                return _make_json_response([{"sport": "NHL", "tags": "101"}])
            offset = int(params.get("offset", 0)) if params else 0
            if offset == 0:
                return _make_json_response(_MARKETS_PAGE_1)  # 100 items → not last
            else:
                return _make_json_response(_MARKETS_PAGE_2)  # 50 items → last

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with PolymarketClient(page_size=100) as client:
                markets = client.fetch_nhl_markets()

        assert len(markets) == 150

    def test_deduplication_by_condition_id(self):
        # Two tag IDs return the same 10 markets
        shared_markets = [
            {"conditionId": f"0x{i:04d}", "id": str(i), "question": f"Q{i}", "slug": f"s{i}",
             "active": True, "closed": False}
            for i in range(10)
        ]

        def mock_get(url, params=None, **kwargs):
            if "/sports" in url:
                return _make_json_response([{"sport": "NHL", "tags": "101,102"}])
            return _make_json_response(shared_markets)

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with PolymarketClient(page_size=100) as client:
                markets = client.fetch_nhl_markets()

        assert len(markets) == 10  # deduplicated

    def test_active_only_passes_params(self):
        received_params: list[dict] = []

        def mock_get(url, params=None, **kwargs):
            received_params.append(params or {})
            if "/sports" in url:
                return _make_json_response([{"sport": "NHL", "tags": "55"}])
            return _make_json_response([])

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with PolymarketClient() as client:
                client.fetch_nhl_markets(active_only=True)

        markets_params = [p for p in received_params if "tag_id" in p]
        assert all(p.get("active") == "true" for p in markets_params)
        assert all(p.get("closed") == "false" for p in markets_params)


# ===========================================================================
# HTTP error handling
# ===========================================================================


class TestPolymarketClientErrors:
    def test_404_raises_immediately_no_retry(self):
        call_count = 0

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 404
            resp.is_error = True
            resp.headers = {}
            return resp

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with PolymarketClient() as client:
                with pytest.raises(PolymarketError) as exc_info:
                    client._get_json("https://example.com/test")

        assert exc_info.value.status_code == 404
        assert call_count == 1

    def test_500_retries_then_raises(self):
        call_count = 0

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 500
            resp.is_error = True
            resp.headers = {}
            return resp

        with patch("time.sleep"):
            with patch.object(httpx.Client, "get", side_effect=mock_get):
                with PolymarketClient(max_retries=2) as client:
                    with pytest.raises(PolymarketError):
                        client._get_json("https://example.com/test")

        assert call_count == 3  # initial + 2 retries

    def test_429_respects_retry_after(self):
        call_count = 0
        sleep_calls: list[float] = []

        def mock_get(url, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.is_error = True
            resp.headers = {"Retry-After": "2"}
            return resp

        def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("time.sleep", side_effect=mock_sleep):
            with patch.object(httpx.Client, "get", side_effect=mock_get):
                with PolymarketClient(max_retries=1) as client:
                    with pytest.raises(PolymarketError):
                        client._get_json("https://example.com/test")

        assert sleep_calls[0] == 2.0  # respected Retry-After header

    def test_timeout_raises_polymarket_error(self):
        def mock_get(url, params=None, **kwargs):
            raise httpx.TimeoutException("timed out")

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with PolymarketClient() as client:
                with pytest.raises(PolymarketError) as exc_info:
                    client._get_json("https://example.com/test")

        assert exc_info.value.status_code is None

    def test_500_then_200_succeeds(self):
        attempts = [500, 200]
        idx = 0
        data = {"result": "ok"}

        def mock_get(url, params=None, **kwargs):
            nonlocal idx
            status = attempts[idx]
            idx += 1
            resp = MagicMock()
            resp.status_code = status
            resp.is_error = status >= 400
            resp.headers = {}
            resp.json.return_value = data
            return resp

        with patch("time.sleep"):
            with patch.object(httpx.Client, "get", side_effect=mock_get):
                with PolymarketClient(max_retries=2) as client:
                    result = client._get_json("https://example.com/test")

        assert result == data
        assert idx == 2
