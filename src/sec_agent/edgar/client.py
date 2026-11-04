"""Minimal EDGAR HTTP client that follows the SEC's fair access rules."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# The SEC's published fair access limit. Going over it gets the caller's address blocked.
SEC_MAX_REQUESTS_PER_SECOND = 10
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# A company's filing list. Older filings are in extra pages (SUBMISSIONS_PAGE_URL).
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


class RateLimiter:
    """Spaces requests evenly. The SEC's limit applies across a user's whole traffic."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if delay > 0:
            time.sleep(delay)


def _is_retryable(error: BaseException) -> bool:
    # Retry rate limiting (429), server errors and network failures. Other errors, such as
    # 404 for a wrong URL, would fail the same way again, so they are raised immediately.
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        *,
        # 8, not 10, leaves headroom under the SEC's limit for retries and clock jitter.
        requests_per_second: float = 8,
        http_client: httpx.Client | None = None,
    ) -> None:
        if requests_per_second > SEC_MAX_REQUESTS_PER_SECOND:
            raise ValueError(f"SEC allows at most {SEC_MAX_REQUESTS_PER_SECOND} requests/second.")
        self._limiter = RateLimiter(requests_per_second)
        self._http = http_client or httpx.Client(timeout=60, follow_redirects=True)
        # EDGAR requires a User-Agent that identifies the caller with contact details.
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

    @retry(
        retry=retry_if_exception(_is_retryable),
        # Up to 5 tries, waiting 1 s, 2 s, 4 s... (never more than 30 s) between them.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        # Every attempt, including retries, passes the rate limiter.
        self._limiter.wait()
        response = self._http.get(url, headers=self._headers)
        response.raise_for_status()
        return response

    def get_json(self, url: str) -> Any:
        return self._get(url).json()

    def download(self, url: str, destination: Path) -> bool:
        """Save a URL to disk. Returns False if the file already exists."""
        if destination.exists() and destination.stat().st_size > 0:
            return False
        content = self._get(url).content
        if not content:
            raise ValueError(f"EDGAR returned an empty file for {url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name, then rename, so an interrupted download can never be
        # mistaken for a complete file on the next run.
        partial = destination.with_name(destination.name + ".partial")
        partial.write_bytes(content)
        os.replace(partial, destination)
        return True
