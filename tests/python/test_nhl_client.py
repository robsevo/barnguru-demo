"""Tests for data.nhl_client — NHL API async HTTP client."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from data.nhl_client import NHLClient, NHLApiError


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_get_roster_empty_string_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError):
            await c.get_roster("")


async def test_get_roster_whitespace_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError):
            await c.get_roster("   ")


async def test_get_play_by_play_zero_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError):
            await c.get_play_by_play(0)


async def test_get_play_by_play_negative_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError):
            await c.get_play_by_play(-1)


async def test_get_schedule_invalid_date_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError):
            await c.get_schedule("not-a-date")


async def test_get_schedule_bad_format_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError):
            await c.get_schedule("2024/11/15")


# ---------------------------------------------------------------------------
# Known-answer (mocked response) tests
# ---------------------------------------------------------------------------

_TEAMS_RESPONSE = {
    "data": [
        {"id": 10, "triCode": "TOR", "fullName": "Toronto Maple Leafs"},
        {"id": 8, "triCode": "MTL", "fullName": "Montréal Canadiens"},
    ]
}

_SCHEDULE_RESPONSE = {
    "gameWeek": [
        {"date": "2024-11-15", "games": [{"id": 2024020001}]},
    ]
}

_ROSTER_RESPONSE = {
    "forwards": [{"id": 8480801, "lastName": {"default": "Matthews"}}],
    "defensemen": [{"id": 8480803, "lastName": {"default": "Rielly"}}],
    "goalies": [{"id": 8480353, "lastName": {"default": "Woll"}}],
}

_PBP_RESPONSE = {
    "id": 2024020001,
    "plays": [
        {"eventId": 1, "typeDescKey": "faceoff"},
        {"eventId": 2, "typeDescKey": "shot-on-goal"},
    ],
}


@respx.mock
async def test_get_teams_returns_list():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(200, json=_TEAMS_RESPONSE)
    )
    async with NHLClient() as c:
        teams = await c.get_teams()
    assert isinstance(teams, list)
    assert len(teams) == 2
    assert teams[0]["triCode"] == "TOR"
    assert "id" in teams[0]
    assert "fullName" in teams[0]


@respx.mock
async def test_get_schedule_now_returns_dict():
    respx.get("https://api-web.nhle.com/v1/schedule/now").mock(
        return_value=httpx.Response(200, json=_SCHEDULE_RESPONSE)
    )
    async with NHLClient() as c:
        sched = await c.get_schedule("now")
    assert "gameWeek" in sched


@respx.mock
async def test_get_schedule_date_returns_dict():
    respx.get("https://api-web.nhle.com/v1/schedule/2024-11-15").mock(
        return_value=httpx.Response(200, json=_SCHEDULE_RESPONSE)
    )
    async with NHLClient() as c:
        sched = await c.get_schedule("2024-11-15")
    assert "gameWeek" in sched


@respx.mock
async def test_get_roster_returns_dict():
    respx.get("https://api-web.nhle.com/v1/roster/TOR/current").mock(
        return_value=httpx.Response(200, json=_ROSTER_RESPONSE)
    )
    async with NHLClient() as c:
        roster = await c.get_roster("TOR")
    assert "forwards" in roster
    assert "defensemen" in roster
    assert "goalies" in roster


@respx.mock
async def test_get_play_by_play_returns_dict():
    respx.get("https://api-web.nhle.com/v1/gamecenter/2024020001/play-by-play").mock(
        return_value=httpx.Response(200, json=_PBP_RESPONSE)
    )
    async with NHLClient() as c:
        pbp = await c.get_play_by_play(2024020001)
    assert "plays" in pbp
    assert isinstance(pbp["plays"], list)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_404_raises_nhl_api_error():
    respx.get("https://api-web.nhle.com/v1/roster/FAKE/current").mock(
        return_value=httpx.Response(404)
    )
    async with NHLClient() as c:
        with pytest.raises(NHLApiError) as exc_info:
            await c.get_roster("FAKE")
    err = exc_info.value
    assert err.status_code == 404
    assert "404" in str(err)


@respx.mock
async def test_500_raises_nhl_api_error():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(500)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock):
        async with NHLClient(max_retries=0) as c:
            with pytest.raises(NHLApiError) as exc_info:
                await c.get_teams()
    assert exc_info.value.status_code == 500


@respx.mock
async def test_timeout_raises_nhl_api_error():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    async with NHLClient() as c:
        with pytest.raises(NHLApiError) as exc_info:
            await c.get_teams()
    assert exc_info.value.status_code is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_teams_empty_data():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    async with NHLClient() as c:
        teams = await c.get_teams()
    assert teams == []


@respx.mock
async def test_get_schedule_no_games():
    respx.get("https://api-web.nhle.com/v1/schedule/2024-07-01").mock(
        return_value=httpx.Response(200, json={"gameWeek": [{"date": "2024-07-01", "games": []}]})
    )
    async with NHLClient() as c:
        sched = await c.get_schedule("2024-07-01")
    assert sched["gameWeek"][0]["games"] == []


@respx.mock
async def test_get_play_by_play_future_game_empty_plays():
    respx.get("https://api-web.nhle.com/v1/gamecenter/2024029999/play-by-play").mock(
        return_value=httpx.Response(200, json={"id": 2024029999, "plays": []})
    )
    async with NHLClient() as c:
        pbp = await c.get_play_by_play(2024029999)
    assert pbp["plays"] == []


# ---------------------------------------------------------------------------
# Lifecycle / integration smoke tests
# ---------------------------------------------------------------------------


async def test_client_instantiates_and_closes():
    client = NHLClient()
    await client.close()


async def test_async_context_manager():
    async with NHLClient() as client:
        assert client is not None


async def test_custom_base_urls():
    client = NHLClient(
        base_stats="http://mock-stats.test",
        base_web="http://mock-web.test",
    )
    await client.close()


# ---------------------------------------------------------------------------
# Rate limiting and retries
# ---------------------------------------------------------------------------


@respx.mock
async def test_429_with_retry_after_header_retries():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json=_TEAMS_RESPONSE),
        ]
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with NHLClient(requests_per_second=1000) as c:
            teams = await c.get_teams()
    assert len(teams) == 2
    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    assert 2.0 in sleep_calls


@respx.mock
async def test_429_without_retry_after_uses_backoff():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=_TEAMS_RESPONSE),
        ]
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with NHLClient(requests_per_second=1000) as c:
            teams = await c.get_teams()
    assert len(teams) == 2
    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    assert any(d > 0 for d in sleep_calls)


@respx.mock
async def test_429_exhausts_retries_raises():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(429)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock):
        async with NHLClient(requests_per_second=1000, max_retries=3) as c:
            with pytest.raises(NHLApiError) as exc_info:
                await c.get_teams()
    assert exc_info.value.status_code == 429


@respx.mock
async def test_503_retries_and_succeeds():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_TEAMS_RESPONSE),
        ]
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock):
        async with NHLClient(requests_per_second=1000) as c:
            teams = await c.get_teams()
    assert len(teams) == 2


@respx.mock
async def test_503_exhausts_retries_raises():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(503)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock):
        async with NHLClient(requests_per_second=1000, max_retries=3) as c:
            with pytest.raises(NHLApiError) as exc_info:
                await c.get_teams()
    assert exc_info.value.status_code == 503


@respx.mock
async def test_404_raises_immediately_no_retry():
    respx.get("https://api-web.nhle.com/v1/roster/FAKE/current").mock(
        return_value=httpx.Response(404)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with NHLClient(requests_per_second=1000) as c:
            with pytest.raises(NHLApiError) as exc_info:
                await c.get_roster("FAKE")
    assert exc_info.value.status_code == 404
    # sleep should not have been called for a retry (only possibly for rate limiter,
    # but at 1000 rps the gap is negligible and won't trigger sleep)
    retry_sleeps = [call.args[0] for call in mock_sleep.call_args_list if call.args[0] >= 0.5]
    assert retry_sleeps == []


@respx.mock
async def test_max_retries_zero_no_retry():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(429)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with NHLClient(requests_per_second=1000, max_retries=0) as c:
            with pytest.raises(NHLApiError) as exc_info:
                await c.get_teams()
    assert exc_info.value.status_code == 429
    retry_sleeps = [call.args[0] for call in mock_sleep.call_args_list if call.args[0] >= 0.5]
    assert retry_sleeps == []


@respx.mock
async def test_rate_limiter_throttles_sequential_requests():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(200, json=_TEAMS_RESPONSE)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with NHLClient(requests_per_second=1, max_retries=0) as c:
            await c.get_teams()
            await c.get_teams()
    # At 1 rps the second request must wait ~1s — sleep should have been called
    assert mock_sleep.called


@respx.mock
async def test_rate_limiter_high_rate_no_sleep():
    respx.get("https://api.nhle.com/stats/rest/en/team").mock(
        return_value=httpx.Response(200, json=_TEAMS_RESPONSE)
    )
    with patch("data.nhl_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        async with NHLClient(requests_per_second=1000, max_retries=0) as c:
            await c.get_teams()
            await c.get_teams()
    # At 1000 rps any sleep call should be negligible (< 1ms), not a real throttle
    significant_sleeps = [call.args[0] for call in mock_sleep.call_args_list if call.args[0] >= 0.001]
    assert significant_sleeps == []


# ---------------------------------------------------------------------------
# EDGE endpoint input validation
# ---------------------------------------------------------------------------


async def test_get_edge_skating_pre_2019_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="2019"):
            await c.get_edge_skating(2018)


async def test_get_edge_skating_bad_game_type_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="game_type"):
            await c.get_edge_skating(2023, game_type=1)


async def test_get_edge_shot_speed_pre_2019_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="2019"):
            await c.get_edge_shot_speed(2018)


async def test_get_edge_shot_speed_bad_game_type_raises():
    async with NHLClient() as c:
        with pytest.raises(ValueError, match="game_type"):
            await c.get_edge_shot_speed(2023, game_type=5)
