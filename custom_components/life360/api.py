"""Life360 API wrapper with curl_cffi fallback for Cloudflare 403s."""

from __future__ import annotations

import logging
from typing import Any

from curl_cffi.requests import AsyncSession, RequestsError
from life360 import (
    CommError,
    Life360Error,
    LoginError,
    NotFound,
    NotModified,
    RateLimited,
    Unauthorized,
)
from life360.api import _HEADERS, Life360 as _BaseLife360
from life360.const import HTTP_Error

_LOGGER = logging.getLogger(__name__)

_CFFI_RETRY_STATUS_CODES = frozenset(
    {
        HTTP_Error.BAD_GATEWAY,
        HTTP_Error.SERVICE_UNAVAILABLE,
        HTTP_Error.GATEWAY_TIME_OUT,
        HTTP_Error.SERVER_UNKNOWN_ERROR,
    }
)


class Life360(_BaseLife360):
    """Life360 client that falls back to curl_cffi after a Cloudflare 403.

    Uses the upstream aiohttp-based transport first.  On the first LoginError
    (HTTP 403) a single retry is attempted via a curl_cffi AsyncSession that
    impersonates a real Chrome TLS+HTTP/2 fingerprint.  If the curl_cffi request
    succeeds all subsequent requests are routed through it, bypassing Cloudflare's
    bot-detection checks.
    """

    def __init__(
        self,
        session: Any,
        max_retries: int,
        authorization: str | None = None,
        *,
        name: str | None = None,
        verbosity: int = 0,
    ) -> None:
        """Initialize API."""
        super().__init__(
            session,
            max_retries,
            authorization,
            name=name,
            verbosity=verbosity,
        )
        self._cffi_session: AsyncSession | None = None
        self._prefer_curl: bool = False

    # ------------------------------------------------------------------
    # Public helpers used by coordinator for cleanup
    # ------------------------------------------------------------------

    def clear_cookies(self) -> None:
        """Clear cookies for the aiohttp transport."""
        self._session.cookie_jar.clear()

    async def async_close(self) -> None:
        """Close the curl_cffi session if one was created."""
        if self._cffi_session is not None:
            await self._cffi_session.close()
            self._cffi_session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cffi_session(self) -> AsyncSession:
        """Return the curl_cffi session, creating it lazily."""
        if self._cffi_session is None:
            self._cffi_session = AsyncSession(impersonate="chrome")
        return self._cffi_session

    # ------------------------------------------------------------------
    # Core request override
    # ------------------------------------------------------------------

    async def _request(
        self,
        url: str,
        /,
        raise_not_modified: bool,
        method: str = "get",
        *,
        authorization: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make a request; transparently fall back to curl_cffi on 403."""
        if self._prefer_curl:
            return await self._cffi_request(
                url,
                raise_not_modified,
                method,
                authorization=authorization,
                **kwargs,
            )

        try:
            return await super()._request(
                url,
                raise_not_modified,
                method,
                authorization=authorization,
                **kwargs,
            )
        except LoginError as aiohttp_exc:
            # aiohttp was blocked (Cloudflare 403).  Try once with curl_cffi.
            try:
                result = await self._cffi_request(
                    url,
                    raise_not_modified,
                    method,
                    authorization=authorization,
                    **kwargs,
                )
            except Life360Error:
                # curl_cffi also failed – propagate the original error.
                raise aiohttp_exc

            # curl_cffi succeeded; switch all future requests to it.
            self._prefer_curl = True
            _LOGGER.warning(
                "Switching to curl_cffi transport after aiohttp 403 for %s", url
            )
            return result

    # ------------------------------------------------------------------
    # curl_cffi transport
    # ------------------------------------------------------------------

    async def _cffi_request(
        self,
        url: str,
        /,
        raise_not_modified: bool,
        method: str = "get",
        *,
        authorization: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make a request using a curl_cffi AsyncSession."""
        if authorization is None:
            authorization = self.authorization
        if authorization is None:
            raise LoginError("Must login")

        # Build headers the same way the upstream library does.
        headers: dict[str, str] = dict(_HEADERS)
        if authorization:
            headers["authorization"] = authorization
        if raise_not_modified and (etag := self._etags.get(url)):
            headers["if-none-match"] = etag
        # Merge any caller-supplied header overrides.
        headers.update(kwargs.pop("headers", {}))

        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = await self._get_cffi_session().request(
                    method.upper(),
                    url,
                    headers=headers,
                    **kwargs,
                )
            except RequestsError as exc:
                if attempt < self._max_attempts:
                    continue
                raise CommError(str(exc), None) from exc

            status = resp.status_code

            # Retry on transient server errors.
            if status in _CFFI_RETRY_STATUS_CODES and attempt < self._max_attempts:
                continue

            if status == HTTP_Error.NOT_MODIFIED:
                raise NotModified

            if etag := resp.headers.get("l360-etag"):
                self._etags[url] = etag

            if status >= 400:
                self._raise_cffi_error(status, resp)

            try:
                return resp.json()
            except Exception as exc:
                raise Life360Error(f"Could not parse curl_cffi response: {exc}") from exc

        raise Life360Error("Unexpected curl_cffi request flow")

    def _raise_cffi_error(self, status: int, resp: Any) -> None:
        """Raise the appropriate life360 exception for an HTTP error status."""
        try:
            err_msg = str(resp.json().get("errorMessage", resp.text)).lower()
        except Exception:
            err_msg = resp.text or f"HTTP {status}"

        match status:
            case HTTP_Error.UNAUTHORIZED:
                raise Unauthorized(err_msg, resp.headers.get("www-authenticate"))
            case HTTP_Error.FORBIDDEN:
                raise LoginError(err_msg)
            case HTTP_Error.NOT_FOUND:
                raise NotFound(err_msg)
            case HTTP_Error.TOO_MANY_REQUESTS:
                retry_after = resp.headers.get("retry-after")
                try:
                    retry_after_value = float(retry_after) if retry_after else None
                except ValueError:
                    retry_after_value = None
                raise RateLimited(err_msg, retry_after_value)
            case _:
                raise CommError(err_msg, status)
