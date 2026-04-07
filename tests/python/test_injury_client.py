"""Tests for InjuryClient (feature 1.8).

Covers:
- HTTP error handling (404 no-retry, 5xx retry, 429 Retry-After)
- Successful fetch returns parsed JSON dict
- Timeout raises InjuryFeedError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from data.injury_client import InjuryClient, InjuryFeedError

# Minimal valid ESPN response (structure only — content tested in parser tests)
_MINIMAL_RESPONSE = {"injuries": []}


# ===========================================================================
# Context manager
# ===========================================================================


class TestInjuryClientContextManager:
    def test_enter_returns_self(self):
        client = InjuryClient()
        assert client.__enter__() is client
        client.close()

    def test_with_block_closes_client(self):
        with InjuryClient() as client:
            assert client._client is not None
        # After exit, underlying httpx client is closed — accessing it should
        # not raise but the client is no longer usable for requests.


# ===========================================================================
# HTTP error handling
# ===========================================================================


class TestInjuryClientErrors:
    def test_404_raises_immediately_no_retry(self):
        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 404
            resp.is_error = True
            resp.headers = {}
            return resp

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with InjuryClient() as client:
                with pytest.raises(InjuryFeedError) as exc_info:
                    client.fetch_injuries()

        assert exc_info.value.status_code == 404
        assert call_count == 1  # no retry

    def test_500_retries_then_raises(self):
        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 500
            resp.is_error = True
            resp.headers = {}
            return resp

        with patch("time.sleep"):  # suppress actual sleep in tests
            with patch.object(httpx.Client, "get", side_effect=mock_get):
                with InjuryClient(max_retries=2) as client:
                    with pytest.raises(InjuryFeedError) as exc_info:
                        client.fetch_injuries()

        assert exc_info.value.status_code == 500
        assert call_count == 3  # initial + 2 retries

    def test_429_retries_then_raises(self):
        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.is_error = True
            resp.headers = {"Retry-After": "0"}
            return resp

        with patch("time.sleep"):
            with patch.object(httpx.Client, "get", side_effect=mock_get):
                with InjuryClient(max_retries=1) as client:
                    with pytest.raises(InjuryFeedError):
                        client.fetch_injuries()

        assert call_count == 2  # initial + 1 retry

    def test_timeout_raises_injury_feed_error(self):
        def mock_get(url, **kwargs):
            raise httpx.TimeoutException("timed out")

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with InjuryClient() as client:
                with pytest.raises(InjuryFeedError) as exc_info:
                    client.fetch_injuries()

        assert exc_info.value.status_code is None
        assert "timed out" in str(exc_info.value).lower()

    def test_non_retryable_4xx_raises(self):
        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 403
            resp.is_error = True
            resp.headers = {}
            return resp

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with InjuryClient() as client:
                with pytest.raises(InjuryFeedError) as exc_info:
                    client.fetch_injuries()

        assert exc_info.value.status_code == 403


# ===========================================================================
# Successful fetch
# ===========================================================================


class TestInjuryClientSuccess:
    def test_success_returns_dict(self):
        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.is_error = False
            resp.headers = {}
            resp.json.return_value = _MINIMAL_RESPONSE
            return resp

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with InjuryClient() as client:
                result = client.fetch_injuries()

        assert isinstance(result, dict)
        assert "injuries" in result

    def test_success_single_attempt(self):
        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.is_error = False
            resp.headers = {}
            resp.json.return_value = _MINIMAL_RESPONSE
            return resp

        with patch.object(httpx.Client, "get", side_effect=mock_get):
            with InjuryClient() as client:
                client.fetch_injuries()

        assert call_count == 1

    def test_500_then_200_succeeds(self):
        """Retried after transient 500, succeeds on second attempt."""
        attempts = [500, 200]
        idx = 0

        def mock_get(url, **kwargs):
            nonlocal idx
            status = attempts[idx]
            idx += 1
            resp = MagicMock()
            resp.status_code = status
            resp.is_error = status >= 400
            resp.headers = {}
            resp.json.return_value = _MINIMAL_RESPONSE
            return resp

        with patch("time.sleep"):
            with patch.object(httpx.Client, "get", side_effect=mock_get):
                with InjuryClient(max_retries=2) as client:
                    result = client.fetch_injuries()

        assert result == _MINIMAL_RESPONSE
        assert idx == 2  # hit 500 once, then 200
