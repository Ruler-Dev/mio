"""Browser security boundary for Mio's loopback Web UI.

Mio exposes local files, model tools, and (when explicitly enabled) code
execution.  A loopback bind alone is not an authentication boundary: DNS
rebinding and cross-site browser requests can still reach it.  This module
therefore owns four small, composable controls:

* strict Host validation before HTTP/WebSocket routing;
* same-origin checks plus a server-issued session/CSRF token for mutations;
* the matching session check for browser WebSocket handshakes; and
* a fail-closed risk classification for WebUI skill execution.

Native API clients are unaffected outside ``/ui``.  Browser sessions are
process-local by design: restarting Mio invalidates them.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import urlsplit

_SESSION_COOKIE = "mio_ui_session"
_CSRF_COOKIE = "mio_csrf"
CSRF_HEADER = "x-mio-csrf-token"
DANGEROUS_ACTION_HEADER = "x-mio-dangerous-action"
WEBHOOK_SECRET_HEADER = "x-mio-webhook-secret"

_MODEL_REQUEST_SKILL_GRANTS: ContextVar[frozenset[str] | None] = ContextVar(
    "mio_model_request_skill_grants",
    default=None,
)

_SESSION_TTL_SECONDS = 12 * 60 * 60
_MAX_SESSIONS = 256
MAX_HTTP_BODY_BYTES = 32 * 1024 * 1024
_UI_DOCUMENT_PATHS = {
    "/dashboard",
    "/ui",
    "/ui/",
    "/ui/playground",
    "/ui/compare",
    "/ui/stats",
    "/ui/attachments",
    "/ui/dashboard",
}
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_PRIVATE_LAN_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)
_WEBUI_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com https://esm.sh; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "font-src 'self' data:; img-src 'self' data: blob:; "
    "connect-src 'self' ws://127.0.0.1:* ws://localhost:* ws://[::1]:* "
    "wss://127.0.0.1:* wss://localhost:* wss://[::1]:*; "
    "worker-src 'self' blob:; frame-src 'self' blob: https://www.youtube.com https://www.youtube-nocookie.com; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)


@dataclass(frozen=True)
class _BrowserSession:
    csrf: str
    expires_at: float


class _RequestBodyTooLarge(Exception):
    """Internal signal raised before an oversized request reaches a route."""


class _SessionStore:
    """Small bounded in-memory store; cookies alone are never trusted."""

    def __init__(self) -> None:
        self._items: OrderedDict[str, _BrowserSession] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_locked(self, now: float) -> None:
        expired = [sid for sid, item in self._items.items() if item.expires_at <= now]
        for sid in expired:
            self._items.pop(sid, None)
        while len(self._items) >= _MAX_SESSIONS:
            self._items.popitem(last=False)

    def create(self) -> tuple[str, str]:
        now = time.monotonic()
        sid = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_locked(now)
            self._items[sid] = _BrowserSession(csrf, now + _SESSION_TTL_SECONDS)
        return sid, csrf

    def lookup(self, sid: str | None) -> _BrowserSession | None:
        if not sid:
            return None
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            item = self._items.get(sid)
            if item is not None:
                self._items.move_to_end(sid)
            return item

    def validate(self, sid: str | None, csrf: str | None) -> bool:
        item = self.lookup(sid)
        return bool(item and csrf and hmac.compare_digest(item.csrf, csrf))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_sessions = _SessionStore()


@dataclass(frozen=True)
class _RuntimeWebPolicy:
    """Explicit authorities enabled by the current ``mio serve`` bind."""

    authorities: frozenset[tuple[str, int | None]] = frozenset()
    browser_origins: frozenset[str] = frozenset()
    private_ip_ports: frozenset[int] = frozenset()


_runtime_policy_lock = threading.Lock()
_runtime_policy = _RuntimeWebPolicy()


def _format_authority(hostname: str, port: int) -> str:
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{rendered_host}:{port}"


def configure_runtime_web_security(
    bind_host: str,
    port: int,
    *,
    allow_remote: bool,
) -> None:
    """Configure the exact browser boundary implied by a server bind.

    A concrete remote bind authorizes only that authority and its same-origin
    browser origin.  A wildcard bind cannot know which local interface the
    browser will use, so its explicit unsafe opt-in authorizes numeric private
    and link-local addresses on the configured port.  Hostnames still require
    ``MIO_TRUSTED_HOSTS`` and cross-origin deployments still require
    ``MIO_CORS_ORIGINS``.
    """

    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    normalized = str(bind_host).strip().lower()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]

    authorities: set[tuple[str, int | None]] = set()
    browser_origins: set[str] = set()
    private_ip_ports: set[int] = set()
    if allow_remote and not _is_loopback_host(normalized):
        if normalized in {"0.0.0.0", "::"}:
            private_ip_ports.add(port)
        else:
            candidate = f"[{normalized}]" if ":" in normalized else normalized
            parsed = _parse_authority(candidate)
            if parsed is None:
                raise ValueError(f"invalid bind host {bind_host!r}")
            hostname, _unused_port = parsed
            authorities.add((hostname, port))
            if port == 80:
                authorities.add((hostname, None))
            browser_origins.add(f"http://{_format_authority(hostname, port)}")

    policy = _RuntimeWebPolicy(
        authorities=frozenset(authorities),
        browser_origins=frozenset(browser_origins),
        private_ip_ports=frozenset(private_ip_ports),
    )
    global _runtime_policy
    with _runtime_policy_lock:
        _runtime_policy = policy


def runtime_web_origins() -> list[str]:
    """Return concrete same-origin browser origins for CORS configuration."""

    with _runtime_policy_lock:
        return sorted(_runtime_policy.browser_origins)


def _runtime_host_allowed(hostname: str, port: int | None) -> bool:
    with _runtime_policy_lock:
        policy = _runtime_policy
    if (hostname, port) in policy.authorities:
        return True
    if port is None or port not in policy.private_ip_ports:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return bool(
        (
            address.is_link_local
            or any(address in network for network in _PRIVATE_LAN_NETWORKS)
        )
        and not address.is_multicast
        and not address.is_unspecified
    )


def reset_web_security_state() -> None:
    """Clear browser sessions. Intended for isolated tests and server restart."""

    _sessions.clear()
    configure_runtime_web_security("127.0.0.1", 9090, allow_remote=False)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _header_values(scope: dict[str, Any], name: str) -> list[str]:
    target = name.lower().encode("ascii")
    return [
        value.decode("latin-1").strip()
        for key, value in scope.get("headers", [])
        if key.lower() == target
    ]


def _single_header(scope: dict[str, Any], name: str) -> str | None:
    values = _header_values(scope, name)
    return values[0] if len(values) == 1 else None


def _cookies(scope: dict[str, Any]) -> dict[str, str]:
    raw = _single_header(scope, "cookie")
    if not raw:
        return {}
    parsed = SimpleCookie()
    try:
        parsed.load(raw)
    except Exception:
        return {}
    return {name: morsel.value for name, morsel in parsed.items()}


def _parse_authority(value: str) -> tuple[str, int | None] | None:
    """Parse a Host/authority without accepting userinfo or ambiguous syntax."""

    if not value or any(ch.isspace() for ch in value) or any(ch in value for ch in ",/\\#?"):
        return None
    try:
        parsed = urlsplit(f"//{value}")
        port = parsed.port
    except ValueError:
        return None
    if parsed.username is not None or parsed.password is not None or not parsed.hostname:
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    if not hostname or "%" in hostname:
        # IPv6 zone identifiers can redirect loopback-looking authorities to
        # an interface selected by the requester; Mio has no reason to allow them.
        return None
    if ":" in hostname and not value.startswith("["):
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    return hostname, port


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _configured_hosts() -> set[tuple[str, int | None]]:
    configured: set[tuple[str, int | None]] = set()
    for raw in os.environ.get("MIO_TRUSTED_HOSTS", "").split(","):
        parsed = _parse_authority(raw.strip())
        if parsed is not None:
            configured.add(parsed)
    return configured


def host_allowed(authority: str | None, *, allow_test_host: bool = False) -> bool:
    """Allow exact loopback authorities and explicit operator additions only."""

    parsed = _parse_authority(authority or "")
    if parsed is None:
        return False
    hostname, port = parsed
    if _is_loopback_host(hostname):
        return True
    if hostname == "testserver" and (allow_test_host or _truthy_env("MIO_ALLOW_TEST_HOST")):
        return True
    for configured_host, configured_port in _configured_hosts():
        if hostname == configured_host and (configured_port is None or configured_port == port):
            return True
    return _runtime_host_allowed(hostname, port)


def _effective_origin(value: str, *, websocket: bool = False) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    schemes = {"http", "https"}
    if websocket:
        schemes |= {"ws", "wss"}
    if parsed.scheme not in schemes or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    if not hostname or "%" in hostname or any(character.isspace() for character in hostname):
        return None
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    effective_port = port or (443 if scheme == "https" else 80)
    return scheme, hostname, effective_port


def _render_origin(origin: tuple[str, str, int]) -> str:
    """Render one normalized HTTP(S) origin without a redundant default port."""

    scheme, hostname, port = origin
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if scheme == "https" else 80
    suffix = "" if port == default_port else f":{port}"
    return f"{scheme}://{host}{suffix}"


def configured_cors_origins() -> list[str]:
    """Return the validated, canonical operator CORS allowlist.

    The environment variable is an explicit capability grant, not a pattern
    language. Wildcards, credentials, paths, and non-HTTP schemes therefore
    fail closed instead of broadening Mio's unauthenticated browser surface.
    """

    configured = os.environ.get("MIO_CORS_ORIGINS")
    if configured is None:
        return []
    origins: list[str] = []
    for raw in configured.split(","):
        value = raw.strip()
        if not value:
            continue
        parsed = _effective_origin(value)
        if value == "*" or parsed is None:
            raise ValueError(f"invalid explicit CORS origin: {raw.strip()!r}")
        origins.append(_render_origin(parsed))
    return list(dict.fromkeys(origins))


def _configured_origin_keys() -> set[tuple[str, str, int]]:
    try:
        return {
            parsed
            for value in configured_cors_origins()
            if (parsed := _effective_origin(value)) is not None
        }
    except ValueError:
        # ``start_server`` reports malformed configuration before binding. A
        # directly embedded middleware instance must still fail closed.
        return set()


def _request_origin(scope: dict[str, Any]) -> tuple[str, str, int] | None:
    host = _single_header(scope, "host")
    parsed = _parse_authority(host or "")
    if parsed is None:
        return None
    hostname, port = parsed
    scheme = scope.get("scheme", "http")
    scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
    return scheme, hostname, port or (443 if scheme == "https" else 80)


def _same_origin_request(scope: dict[str, Any]) -> bool:
    expected = _request_origin(scope)
    if expected is None:
        return False
    if any(len(_header_values(scope, name)) > 1 for name in ("origin", "referer", "sec-fetch-site")):
        return False
    origin = _single_header(scope, "origin")
    if origin is not None and _effective_origin(origin) != expected:
        return False
    referer = _single_header(scope, "referer")
    if origin is None and referer:
        try:
            parsed = urlsplit(referer)
            referer_origin = f"{parsed.scheme}://{parsed.netloc}"
        except ValueError:
            return False
        if _effective_origin(referer_origin) != expected:
            return False
    fetch_site = _single_header(scope, "sec-fetch-site")
    if fetch_site and fetch_site.lower() not in {"same-origin", "none"}:
        return False
    return True


def _explicit_cross_origin_request(scope: dict[str, Any]) -> bool:
    """Authorize an unsafe browser request only via the exact operator list."""

    origins = _header_values(scope, "origin")
    if len(origins) != 1:
        return False
    actual = _effective_origin(origins[0])
    return bool(actual and actual in _configured_origin_keys())


def _cookie_header(name: str, value: str, *, http_only: bool, secure: bool) -> bytes:
    cookie = SimpleCookie()
    cookie[name] = value
    morsel = cookie[name]
    morsel["path"] = "/"
    morsel["max-age"] = str(_SESSION_TTL_SECONDS)
    morsel["samesite"] = "Strict"
    if http_only:
        morsel["httponly"] = True
    if secure:
        morsel["secure"] = True
    return morsel.OutputString().encode("latin-1")


def _security_headers(existing: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    headers = list(existing)
    names = {key.lower() for key, _value in headers}
    additions = {
        b"content-security-policy": _WEBUI_CSP.encode("ascii"),
        b"referrer-policy": b"no-referrer",
        b"x-content-type-options": b"nosniff",
        b"x-frame-options": b"DENY",
        b"cross-origin-opener-policy": b"same-origin",
        b"permissions-policy": b"camera=(), geolocation=(), payment=(), usb=()",
        b"cache-control": b"no-store",
    }
    for key, value in additions.items():
        if key not in names:
            headers.append((key, value))
    return headers


async def _http_error(send: Callable[[dict[str, Any]], Awaitable[None]], status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class WebSecurityMiddleware:
    """Pure-ASGI Host/session/CSRF middleware.

    ``allow_test_host`` is deliberately constructor-only so production never
    trusts Starlette's synthetic ``testserver`` hostname by accident.
    """

    def __init__(
        self,
        app: Any,
        *,
        allow_test_host: bool = False,
        max_body_bytes: int = MAX_HTTP_BODY_BYTES,
        webui_enabled: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(max_body_bytes, int) or max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be a positive integer")
        self.app = app
        self.allow_test_host = allow_test_host
        self.max_body_bytes = max_body_bytes
        self.webui_enabled = webui_enabled

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        host_values = _header_values(scope, "host")
        if len(host_values) != 1 or not host_allowed(
            host_values[0] if host_values else None,
            allow_test_host=self.allow_test_host,
        ):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": "Host is not allowed"})
            else:
                await _http_error(send, 400, "Host is not allowed")
            return

        path = scope.get("path", "")
        ui_path = path == "/ui" or path.startswith("/ui/")
        if ui_path and self.webui_enabled is not None and not self.webui_enabled():
            if scope["type"] == "websocket":
                await send(
                    {
                        "type": "websocket.close",
                        "code": 1008,
                        "reason": "Mio UI is disabled",
                    }
                )
            else:
                await _http_error(send, 404, "Mio UI is disabled")
            return

        if scope["type"] == "websocket":
            if self.allow_test_host:
                scope.setdefault("state", {})["mio_allow_test_host"] = True
            if not websocket_scope_allowed(scope):
                await send(
                    {
                        "type": "websocket.close",
                        "code": 1008,
                        "reason": "WebSocket origin or session is not allowed",
                    }
                )
                return
            await self.app(scope, receive, send)
            return

        content_lengths = _header_values(scope, "content-length")
        if len(content_lengths) > 1:
            await _http_error(send, 400, "Ambiguous Content-Length")
            return
        if content_lengths:
            try:
                declared_length = int(content_lengths[0], 10)
            except ValueError:
                await _http_error(send, 400, "Invalid Content-Length")
                return
            if declared_length < 0:
                await _http_error(send, 400, "Invalid Content-Length")
                return
            if declared_length > self.max_body_bytes:
                await _http_error(send, 413, "Request body is too large")
                return

        method = scope.get("method", "GET").upper()
        cookies = _cookies(scope)
        session = _sessions.lookup(cookies.get(_SESSION_COOKIE))
        new_session: tuple[str, str] | None = None
        if method == "GET" and path in _UI_DOCUMENT_PATHS and session is None:
            new_session = _sessions.create()
            scope.setdefault("state", {})["mio_ui_session"] = new_session[0]
            scope["state"]["mio_csrf_token"] = new_session[1]

        protected = method in _UNSAFE_METHODS and path.startswith("/ui/api/")
        browser_signaled = any(
            _header_values(scope, header)
            for header in ("origin", "referer", "sec-fetch-site")
        )
        if (
            method in _UNSAFE_METHODS
            and browser_signaled
            and not (
                _same_origin_request(scope)
                or _explicit_cross_origin_request(scope)
            )
        ):
            await _http_error(send, 403, "Cross-origin browser mutation denied")
            return
        webhook_secret = _single_header(scope, WEBHOOK_SECRET_HEADER)
        # External webhooks use their configured secret header instead of a
        # browser session. The endpoint still verifies it against the hook.
        external_webhook = path.startswith("/ui/api/webhook/") and bool(webhook_secret)
        if protected and not external_webhook:
            csrf = _single_header(scope, CSRF_HEADER)
            if not _sessions.validate(cookies.get(_SESSION_COOKIE), csrf):
                await _http_error(send, 403, "Missing or invalid WebUI CSRF session")
                return

        response_started = False

        async def send_with_security(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            if message["type"] == "http.response.start" and (
                path.startswith("/ui") or path == "/dashboard"
            ):
                headers = _security_headers(message.get("headers", []))
                if new_session is not None:
                    secure = scope.get("scheme") == "https"
                    headers.append(
                        (b"set-cookie", _cookie_header(_SESSION_COOKIE, new_session[0], http_only=True, secure=secure))
                    )
                    headers.append(
                        (b"set-cookie", _cookie_header(_CSRF_COOKIE, new_session[1], http_only=False, secure=secure))
                    )
                message = {**message, "headers": headers}
            await send(message)

        received_bytes = 0

        async def receive_limited() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, receive_limited, send_with_security)
        except _RequestBodyTooLarge:
            # FastAPI/Starlette consumes request bodies before starting route
            # responses. Keep the guard defensive for any future streaming
            # endpoint that might violate that ordering.
            if response_started:
                raise
            await _http_error(send, 413, "Request body is too large")


def websocket_origin_allowed(websocket: Any) -> bool:
    """Accept native clients without Origin; require exact origin for browsers."""

    origin = websocket.headers.get("origin")
    if not origin:
        return True
    actual = _effective_origin(origin)
    configured = _configured_origin_keys()
    try:
        parsed_url = urlsplit(str(websocket.url))
        expected = _effective_origin(
            f"{parsed_url.scheme}://{parsed_url.netloc}",
            websocket=True,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    allow_test = bool(
        getattr(websocket, "scope", {}).get("state", {}).get("mio_allow_test_host")
        or _truthy_env("MIO_ALLOW_TEST_HOST")
    )
    return bool(
        actual
        and (
            actual in configured
            or (
                expected
                and actual == expected
                and host_allowed(
                    _format_authority(actual[1], actual[2]),
                    allow_test_host=allow_test,
                )
            )
        )
    )


def _websocket_csrf_token(websocket: Any) -> str | None:
    protocols = websocket.headers.get("sec-websocket-protocol", "")
    for protocol in protocols.split(","):
        protocol = protocol.strip()
        if protocol.startswith("mio-csrf."):
            return protocol.removeprefix("mio-csrf.")
    return None


def _websocket_scope_csrf_token(scope: dict[str, Any]) -> str | None:
    protocols = _single_header(scope, "sec-websocket-protocol") or ""
    for protocol in protocols.split(","):
        protocol = protocol.strip()
        if protocol.startswith("mio-csrf."):
            return protocol.removeprefix("mio-csrf.")
    return None


def websocket_scope_allowed(scope: dict[str, Any]) -> bool:
    """Apply the browser WebSocket boundary before any endpoint is routed."""

    origins = _header_values(scope, "origin")
    if not origins:
        return True
    if len(origins) != 1:
        return False
    origin = origins[0]
    actual = _effective_origin(origin)
    expected = _request_origin(scope)
    allow_test = bool(
        scope.get("state", {}).get("mio_allow_test_host")
        or _truthy_env("MIO_ALLOW_TEST_HOST")
    )
    explicitly_allowed = bool(actual and actual in _configured_origin_keys())
    same_origin = bool(
        actual
        and expected
        and actual == expected
        and host_allowed(
            _format_authority(actual[1], actual[2]),
            allow_test_host=allow_test,
        )
    )
    origin_allowed = explicitly_allowed or same_origin
    if not origin_allowed:
        return False
    fetch_site = _single_header(scope, "sec-fetch-site")
    if fetch_site:
        allowed_fetch_sites = (
            {"same-origin", "same-site", "cross-site", "none"}
            if explicitly_allowed
            else {"same-origin", "none"}
        )
        if fetch_site.lower() not in allowed_fetch_sites:
            return False
    cookies = _cookies(scope)
    return _sessions.validate(cookies.get(_SESSION_COOKIE), _websocket_scope_csrf_token(scope))


def websocket_session_allowed(websocket: Any) -> bool:
    """Validate browser WebSockets against the shell-issued session token."""

    if not websocket.headers.get("origin"):
        return True
    fetch_site = websocket.headers.get("sec-fetch-site")
    if fetch_site:
        actual = _effective_origin(websocket.headers.get("origin", ""))
        explicitly_allowed = bool(actual and actual in _configured_origin_keys())
        allowed_fetch_sites = (
            {"same-origin", "same-site", "cross-site", "none"}
            if explicitly_allowed
            else {"same-origin", "none"}
        )
        if fetch_site.lower() not in allowed_fetch_sites:
            return False
    try:
        sid = websocket.cookies.get(_SESSION_COOKIE)
    except (AttributeError, TypeError):
        raw_cookie = websocket.headers.get("cookie", "")
        parsed = SimpleCookie()
        parsed.load(raw_cookie)
        sid = parsed.get(_SESSION_COOKIE).value if parsed.get(_SESSION_COOKIE) else None
    return _sessions.validate(sid, _websocket_csrf_token(websocket))


async def reject_untrusted_websocket(websocket: Any) -> bool:
    """Close an untrusted handshake and report whether it was rejected."""

    if websocket_origin_allowed(websocket) and websocket_session_allowed(websocket):
        return False
    await websocket.close(code=1008, reason="WebSocket origin or session is not allowed")
    return True


# Explicit read-only allowlist. Any future/unclassified skill is sensitive by
# default and cannot silently become model-auto-executable through the WebUI.
READ_ONLY_WEBUI_SKILLS = frozenset(
    {
        "analyze_csv", "analyze_json",
        "color_palette", "convert_currency", "date_math", "decode_jwt",
        "detect_language", "encode_decode", "explain_regex", "extract_links", "fetch_url",
        "find_anime", "find_game", "find_manga", "find_movie_tv", "flip_coin",
        "format_json", "generate_fake_data", "generate_names", "generate_password",
        "generate_slug", "generate_uuid", "get_weather", "hash_text", "hn_top",
        "html_to_markdown", "json_query", "json_to_yaml", "markdown_to_html",
        "meeting_notes", "pick_random", "quote", "reddit_top", "review_code",
        "roll_dice", "scale_recipe", "search_images", "search_youtube", "symbolic_math",
        "text_stats", "timezone_convert", "translate_text", "unit_convert", "web_search",
        "wiki_summary", "wordle_helper", "yaml_to_json",
    }
)

_EXECUTION_SKILLS = frozenset({"execute_python", "execute_python_safe", "blender_exec"})
_PRIVATE_NETWORK_SKILLS = frozenset(
    {"http_request", "blender_status", "blender_snapshot", "fetch_rss", "url_preview"}
)
_ORCHESTRATION_SKILLS = frozenset({"call_mcp_tool", "list_mcp_tools", "run_flow_skill"})
_LOCAL_READ_SKILLS = frozenset(
    {
        "bookmark_list", "bookmark_search", "describe_image", "extract_pdf_text",
        "habit_list", "image_info", "journal_read", "journal_search", "list_flow_skills",
        "list_indexes", "list_mio_skills", "read_mio_skill", "reading_briefing",
        "search_local_folder", "todo_list",
    }
)
_LOCAL_WRITE_SKILLS = frozenset(
    {
        "bookmark_save", "drop_index", "generate_brochure", "generate_business_card",
        "generate_certificate", "generate_chart", "generate_csv", "generate_docx",
        "generate_flyer", "generate_ical", "generate_invoice", "generate_letter",
        "generate_markdown", "generate_menu", "generate_newsletter", "generate_pdf",
        "generate_pdf_report", "generate_pptx", "generate_qr_code", "generate_resume",
        "generate_sqlite_db", "generate_xlsx", "habit_add", "habit_checkin",
        "image_convert", "image_resize", "index_folder", "journal_append", "merge_pdfs",
        "split_pdf", "todo_add", "todo_delete", "todo_done", "unzip_file", "zip_files",
    }
)


def webui_skill_risk(name: str) -> str:
    if name in READ_ONLY_WEBUI_SKILLS:
        return "read"
    if name in _EXECUTION_SKILLS or name.startswith("execute_"):
        return "execute"
    if name in _PRIVATE_NETWORK_SKILLS:
        return "private-network"
    if name in _ORCHESTRATION_SKILLS:
        return "orchestrator"
    if name in _LOCAL_READ_SKILLS:
        return "local-read"
    if name in _LOCAL_WRITE_SKILLS:
        return "write"
    return "sensitive"


def _skill_grants() -> set[str]:
    return {
        name.strip()
        for name in os.environ.get("MIO_WEBUI_SKILL_GRANTS", "").split(",")
        if name.strip()
    }


def webui_skill_operator_granted(name: str) -> bool:
    return webui_skill_risk(name) == "read" or name in _skill_grants()


def webui_skill_direct_authorized(name: str, *, confirmed: Any, action_header: str | None) -> bool:
    """Sensitive direct runs require both operator grant and click-level consent."""

    if webui_skill_risk(name) == "read":
        return True
    return bool(
        webui_skill_operator_granted(name)
        and confirmed is True
        and action_header
        and hmac.compare_digest(action_header, name)
    )


def webui_model_skill_allowed(name: str, request_grants: Iterable[str] = ()) -> bool:
    """Auto tool loop is read-only unless both operator and request grant a name."""

    if webui_skill_risk(name) == "read":
        return True
    requested = {str(value) for value in request_grants}
    return name in requested and webui_skill_operator_granted(name)


@contextmanager
def model_request_skill_grants(grants: Iterable[str]) -> Iterator[None]:
    """Bind request-local model grants across ``asyncio.to_thread`` calls."""

    token = _MODEL_REQUEST_SKILL_GRANTS.set(
        frozenset(str(name) for name in grants if str(name))
    )
    try:
        yield
    finally:
        _MODEL_REQUEST_SKILL_GRANTS.reset(token)


def current_model_request_skill_grants() -> frozenset[str] | None:
    """Return the active model request's grants, or ``None`` outside one."""

    return _MODEL_REQUEST_SKILL_GRANTS.get()
