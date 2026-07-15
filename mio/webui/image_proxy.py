"""Bounded, DNS-pinned image retrieval for the Mio Web UI.

This module deliberately does not use urllib's automatic redirects: every hop
must pass the host/IP policy and the connection is opened against the exact IP
that was validated.  Response bytes are streamed under byte and wall-clock
budgets, then checked against both the upstream MIME type and image magic.
"""

from __future__ import annotations

import http.client
import ipaddress
import math
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Collection
from urllib.parse import SplitResult, urljoin, urlsplit


DEFAULT_IMAGE_HOSTS = frozenset({
    "myanimelist.net",
    "cdn.myanimelist.net",
    "i.ytimg.com",
    "yt3.ggpht.com",
    "commons.wikimedia.org",
    "upload.wikimedia.org",
    "tvmaze.com",
    "static.tvmaze.com",
})
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 12.0
DEFAULT_MAX_REDIRECTS = 5
_READ_CHUNK_BYTES = 64 * 1024


class ImageFetchError(ValueError):
    """The upstream target or response failed Mio's image proxy policy."""


@dataclass(frozen=True)
class ImagePayload:
    data: bytes
    media_type: str
    extension: str
    final_url: str


@dataclass(frozen=True)
class _Target:
    parsed: SplitResult
    hostname: str
    port: int
    pinned_ip: str


_CONTENT_TYPES = {
    "image/jpeg": (".jpg", "image/jpeg"),
    "image/jpg": (".jpg", "image/jpeg"),
    "image/png": (".png", "image/png"),
    "image/gif": (".gif", "image/gif"),
    "image/webp": (".webp", "image/webp"),
    "image/avif": (".avif", "image/avif"),
}


def _normalize_hostname(hostname: str) -> str:
    try:
        return hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ImageFetchError("invalid image host") from exc


def _host_allowed(hostname: str, allowed_hosts: Collection[str] | None) -> bool:
    if allowed_hosts is None:
        return True
    normalized = {_normalize_hostname(host) for host in allowed_hosts}
    return any(
        hostname == allowed or hostname.endswith("." + allowed)
        for allowed in normalized
    )


def _validated_target(
    url: str,
    *,
    allowed_hosts: Collection[str] | None,
) -> _Target:
    try:
        parsed = urlsplit(url)
        explicit_port = parsed.port
    except ValueError as exc:
        raise ImageFetchError("invalid image URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ImageFetchError("only http:// and https:// image URLs are allowed")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ImageFetchError("image URL host is required and credentials are forbidden")

    hostname = _normalize_hostname(parsed.hostname)
    if not _host_allowed(hostname, allowed_hosts):
        raise ImageFetchError("image host is not allowed")
    port = explicit_port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        raise ImageFetchError("image proxy only permits ports 80 and 443")

    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ImageFetchError(f"cannot resolve image host: {hostname}") from exc
    if not addresses:
        raise ImageFetchError(f"cannot resolve image host: {hostname}")

    resolved: list[str] = []
    for address in addresses:
        raw_ip = address[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ImageFetchError("image host resolved to an invalid address") from exc
        if not ip.is_global or ip.is_multicast or ip.is_unspecified:
            raise ImageFetchError(
                "private, loopback, link-local and reserved image targets are blocked"
            )
        normalized_ip = str(ip)
        if normalized_ip not in resolved:
            resolved.append(normalized_ip)
    return _Target(parsed, hostname, port, resolved[0])


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, target: _Target, *, timeout: float):
        super().__init__(target.hostname, port=target.port, timeout=timeout)
        self._pinned_ip = target.pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, target: _Target, *, timeout: float):
        super().__init__(
            target.hostname,
            port=target.port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_ip = target.pinned_ip

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            # self.host remains the validated hostname, preserving SNI and
            # certificate verification while avoiding a second DNS lookup.
            self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)
        except Exception:
            raw_socket.close()
            raise


def _connection_for(target: _Target, timeout: float):
    connection_type = (
        _PinnedHTTPSConnection if target.parsed.scheme == "https" else _PinnedHTTPConnection
    )
    return connection_type(target, timeout=timeout)


def _host_header(target: _Target) -> str:
    hostname = f"[{target.hostname}]" if ":" in target.hostname else target.hostname
    default_port = 443 if target.parsed.scheme == "https" else 80
    return hostname if target.port == default_port else f"{hostname}:{target.port}"


def _sniff_image(data: bytes) -> tuple[str, str] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if len(data) >= 16 and data[4:8] == b"ftyp":
        brands = {data[index:index + 4] for index in range(8, min(len(data), 40), 4)}
        if brands.intersection({b"avif", b"avis"}):
            return ".avif", "image/avif"
    return None


def _validated_image_type(data: bytes, content_type: str | None) -> tuple[str, str]:
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    declared = _CONTENT_TYPES.get(normalized_type)
    if declared is None:
        raise ImageFetchError("upstream Content-Type is not a supported image type")
    detected = _sniff_image(data)
    if detected is None:
        raise ImageFetchError("upstream body is not a supported raster image")
    if detected != declared:
        raise ImageFetchError("upstream Content-Type does not match image bytes")
    return detected


def _read_bounded_body(
    response: http.client.HTTPResponse,
    connection: http.client.HTTPConnection,
    *,
    max_bytes: int,
    deadline: float,
) -> bytes:
    raw_length = response.getheader("Content-Length")
    if raw_length:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise ImageFetchError("upstream sent an invalid Content-Length") from exc
        if declared_length < 0 or declared_length > max_bytes:
            raise ImageFetchError(f"image response exceeds {max_bytes} bytes")

    chunks: list[bytes] = []
    total = 0
    while True:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise ImageFetchError("image fetch exceeded its time budget")
        if connection.sock is not None:
            connection.sock.settimeout(remaining_time)
        try:
            chunk = response.read(min(_READ_CHUNK_BYTES, max_bytes + 1 - total))
        except (TimeoutError, socket.timeout) as exc:
            raise ImageFetchError("image fetch exceeded its time budget") from exc
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise ImageFetchError(f"image response exceeds {max_bytes} bytes")
    return b"".join(chunks)


def fetch_image(
    url: str,
    *,
    allowed_hosts: Collection[str] | None = DEFAULT_IMAGE_HOSTS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> ImagePayload:
    """Fetch and validate one image with redirect/DNS/size/time confinement."""
    if not isinstance(url, str) or not url:
        raise ImageFetchError("image URL is required")
    if (
        not isinstance(max_bytes, int)
        or max_bytes <= 0
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or not isinstance(max_redirects, int)
        or max_redirects < 0
    ):
        raise ImageFetchError("invalid image fetch limits")

    deadline = time.monotonic() + timeout_seconds
    current_url = url
    previous_scheme: str | None = None
    for redirect_count in range(max_redirects + 1):
        target = _validated_target(current_url, allowed_hosts=allowed_hosts)
        if previous_scheme == "https" and target.parsed.scheme != "https":
            raise ImageFetchError("HTTPS image redirects may not downgrade to HTTP")
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise ImageFetchError("image fetch exceeded its time budget")
        connection = _connection_for(target, remaining_time)
        path = target.parsed.path or "/"
        if target.parsed.query:
            path += "?" + target.parsed.query
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": ", ".join(sorted(_CONTENT_TYPES)),
                    "Accept-Encoding": "identity",
                    "Host": _host_header(target),
                    "User-Agent": "Mio-ImageProxy/0.1",
                },
            )
            response = connection.getresponse()
            status = response.status
            location = response.getheader("Location")
            if status in {301, 302, 303, 307, 308}:
                if not location:
                    raise ImageFetchError("image redirect omitted Location")
                if redirect_count >= max_redirects:
                    raise ImageFetchError("image redirect limit exceeded")
                previous_scheme = target.parsed.scheme
                current_url = urljoin(current_url, location)
                continue
            if not 200 <= status < 300:
                raise ImageFetchError(f"image server returned status {status}")
            content_type = response.getheader("Content-Type")
            data = _read_bounded_body(
                response,
                connection,
                max_bytes=max_bytes,
                deadline=deadline,
            )
            extension, media_type = _validated_image_type(data, content_type)
            return ImagePayload(data, media_type, extension, current_url)
        finally:
            connection.close()
    raise ImageFetchError("image redirect limit exceeded")
