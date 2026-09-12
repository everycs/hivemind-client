"""Tests for HivemindClient and CircuitBreaker."""

from __future__ import annotations

import httpx

from makersite_hivemind.client import CircuitBreaker, HivemindClient

# --- CircuitBreaker ---


def test_breaker_stays_closed_under_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open


def test_breaker_opens_at_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3, open_seconds=60.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open


def test_breaker_resets_on_success() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert not cb.is_open


def test_breaker_closes_after_timeout() -> None:
    cb = CircuitBreaker(failure_threshold=1, open_seconds=0.0)
    cb.record_failure()
    # open_seconds=0 means it expires immediately
    assert not cb.is_open


# --- HivemindClient.ask ---


async def test_ask_returns_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/ask" in str(request.url)
        assert "question=" in str(request.url)
        return httpx.Response(200, json={"result": "Use dataclasses."})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("how to define models")
    assert result == "Use dataclasses."


async def test_ask_passes_directory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "directory=database" in str(request.url)
        return httpx.Response(200, json={"result": "ok"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("backup schedule", directory="database")
    assert result == "ok"


async def test_ask_passes_vault() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "vault=work" in str(request.url)
        return httpx.Response(200, json={"result": "ok"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("anything", vault="work")
    assert result == "ok"


async def test_ask_returns_empty_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("anything")
    assert result == ""


async def test_ask_returns_empty_on_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("anything")
    assert result == ""


# --- HivemindClient.search ---


async def test_search_returns_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/search" in str(request.url)
        assert "limit=3" in str(request.url)
        return httpx.Response(200, json={"result": "Found 1 result"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.search("orientdb", limit=3)
    assert "Found 1 result" in result


async def test_search_passes_directory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "directory=infra" in str(request.url)
        return httpx.Response(200, json={"result": "ok"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.search("terraform", directory="infra")
    assert result == "ok"


# --- HivemindClient.smart_search ---


async def test_smart_search_returns_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/smart-search" in str(request.url)
        return httpx.Response(200, json={"result": "Found via smart search"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.smart_search("database crashing")
    assert "smart search" in result


# --- HivemindClient.get_document ---


async def test_get_document_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/get" in str(request.url)
        return httpx.Response(200, json={"result": "# My Doc\nContent here"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.get_document("database/orientdb.md")
    assert "My Doc" in result


# --- HivemindClient.health ---


async def test_health_returns_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "documents": 42})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.health()
    assert result == {"status": "ok", "documents": 42}


async def test_health_returns_empty_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.health()
    assert result == {}


# --- Circuit breaker integration ---


async def test_breaker_prevents_calls_after_repeated_failures() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("refused")

    async with HivemindClient("http://test:8300", failure_threshold=2, circuit_open_seconds=60.0) as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        await client.ask("q1")
        await client.ask("q2")
        assert client.breaker.is_open

        before = call_count
        result = await client.ask("q3")
        assert result == ""
        assert call_count == before  # no HTTP call made


async def test_breaker_resets_after_success() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 2:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"result": "recovered"})

    client = HivemindClient("http://test:8300", failure_threshold=3)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.ask("q1")  # fail
    await client.ask("q2")  # fail
    assert not client.breaker.is_open  # threshold=3, only 2 failures

    result = await client.ask("q3")  # success
    assert result == "recovered"
    assert not client.breaker.is_open
    await client.close()


# --- Context manager ---


async def test_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": "ok"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("test")
        assert result == "ok"


# --- Edge cases ---


async def test_non_string_result_is_coerced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": ["a", "b"]})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("test")
    assert result == "['a', 'b']"


async def test_missing_result_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "something"})

    async with HivemindClient("http://test:8300") as client:
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await client.ask("test")
    assert result == ""
