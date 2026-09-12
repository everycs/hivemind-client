"""Async HTTP client for the Hivemind vault-search API.

Endpoints:
    /ask     - Compressed answer (1-3 sentences), server-side cached (24h TTL).
    /search  - BM25-ranked full-text search results.
    /smart-search - Search with automatic LLM query correction.
    /get     - Retrieve a single document by path.
    /health  - Health check.

All methods return empty string on failure. A circuit breaker prevents
cascading failures when the server is unreachable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0
_CIRCUIT_OPEN_SECONDS = 60.0
_FAILURE_THRESHOLD = 3


@dataclass
class CircuitBreaker:
    """Consecutive-failure circuit breaker.

    Opens after ``failure_threshold`` consecutive failures, stays open for
    ``open_seconds``, then allows one probe request (half-open). A success
    resets the counter immediately.
    """

    failure_threshold: int = _FAILURE_THRESHOLD
    open_seconds: float = _CIRCUIT_OPEN_SECONDS
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _open_until: float = field(default=0.0, init=False, repr=False)

    @property
    def is_open(self) -> bool:
        if self._consecutive_failures < self.failure_threshold:
            return False
        return time.monotonic() < self._open_until

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.open_seconds
            logger.warning(
                "Hivemind circuit breaker open for %.0fs after %d consecutive failures",
                self.open_seconds,
                self._consecutive_failures,
            )


class HivemindClient:
    """Async client for the Hivemind vault-search HTTP API.

    Usage::

        client = HivemindClient("http://hivemind.orion.makersite.org:8300")
        answer = await client.ask("what are the OrientDB backup conventions?")
        results = await client.search("orientdb backup", limit=5)
        await client.close()

    Or as an async context manager::

        async with HivemindClient("http://hivemind:8300") as client:
            answer = await client.ask("deployment checklist")
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        failure_threshold: int = _FAILURE_THRESHOLD,
        circuit_open_seconds: float = _CIRCUIT_OPEN_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http: httpx.AsyncClient | None = None
        self._owns_http = True
        self.breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            open_seconds=circuit_open_seconds,
        )

    async def __aenter__(self) -> HivemindClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self._timeout)
            self._owns_http = True
        return self._http

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._http and not self._http.is_closed and self._owns_http:
            await self._http.aclose()

    # --- Public API ---

    async def ask(self, question: str, *, directory: str = "", vault: str = "") -> str:
        """Get a compressed answer (1-3 sentences) from Hivemind.

        Uses hybrid BM25 + vector retrieval with LLM compression.
        Responses are cached server-side for 24 hours.

        Returns empty string on failure or when circuit breaker is open.
        """
        params: dict[str, str] = {"question": question}
        if directory:
            params["directory"] = directory
        if vault:
            params["vault"] = vault
        return await self._get("/ask", params)

    async def search(self, query: str, *, limit: int = 15, directory: str = "", vault: str = "") -> str:
        """BM25-ranked full-text search.

        Returns formatted search results as text.
        Returns empty string on failure or when circuit breaker is open.
        """
        params: dict[str, str | int] = {"query": query, "limit": limit}
        if directory:
            params["directory"] = directory
        if vault:
            params["vault"] = vault
        return await self._get("/search", params)

    async def smart_search(self, query: str, *, limit: int = 15, directory: str = "", vault: str = "") -> str:
        """Search with automatic LLM query correction on weak results.

        Returns empty string on failure or when circuit breaker is open.
        """
        params: dict[str, str | int] = {"query": query, "limit": limit}
        if directory:
            params["directory"] = directory
        if vault:
            params["vault"] = vault
        return await self._get("/smart-search", params)

    async def get_document(self, path: str, *, vault: str = "") -> str:
        """Retrieve a single document by its relative path.

        Returns empty string if not found or on failure.
        """
        params: dict[str, str] = {"path": path}
        if vault:
            params["vault"] = vault
        return await self._get("/get", params)

    async def health(self) -> dict:
        """Check server health. Returns {"status": "ok", "documents": N} or empty dict."""
        if self.breaker.is_open:
            return {}
        try:
            client = await self._client()
            resp = await client.get(f"{self._base_url}/health")
            resp.raise_for_status()
            self.breaker.record_success()
            return resp.json()
        except Exception:
            self.breaker.record_failure()
            return {}

    # --- Internal ---

    async def _get(self, path: str, params: dict) -> str:
        if self.breaker.is_open:
            return ""
        try:
            client = await self._client()
            resp = await client.get(f"{self._base_url}{path}", params=params)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", "")
            if not isinstance(result, str):
                result = str(result)
            self.breaker.record_success()
            return result
        except httpx.TimeoutException:
            self.breaker.record_failure()
            logger.warning("Hivemind request timed out: %s", path)
            return ""
        except Exception:
            self.breaker.record_failure()
            logger.warning("Hivemind request failed: %s", path, exc_info=True)
            return ""
