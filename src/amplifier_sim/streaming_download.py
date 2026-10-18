"""Local, on-demand HTTP streaming for large UI result downloads.

Streamlit's built-in download storage is intentionally memory-backed.  Large
``.ampres.zip`` files therefore need a separate loopback-only response path so
the archive can be written directly to Chrome without retaining the completed
payload in RAM or on disk.
"""

from __future__ import annotations

import atexit
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, Callable
from urllib.parse import quote, urlsplit


_LOGGER = logging.getLogger(__name__)
_DEFAULT_TTL_S = 30.0 * 60.0
_MAX_REGISTERED_DOWNLOADS = 128


@dataclass
class _DownloadEntry:
    token: str
    cache_key: str
    writer: Callable[[BinaryIO], None]
    file_name: str
    mime_type: str
    expires_at_monotonic_s: float
    active_requests: int = 0


class _DownloadRegistry:
    """Keep small capability records; archive bytes never enter this registry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _DownloadEntry] = {}
        self._tokens_by_cache_key: dict[str, str] = {}

    def _remove_locked(self, token: str) -> None:
        entry = self._entries.pop(token, None)
        if entry is None:
            return
        if self._tokens_by_cache_key.get(entry.cache_key) == token:
            del self._tokens_by_cache_key[entry.cache_key]

    def _purge_locked(self, now: float) -> None:
        for token, entry in tuple(self._entries.items()):
            if entry.active_requests == 0 and entry.expires_at_monotonic_s <= now:
                self._remove_locked(token)
        if len(self._entries) <= _MAX_REGISTERED_DOWNLOADS:
            return
        inactive = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.active_requests == 0
            ),
            key=lambda entry: entry.expires_at_monotonic_s,
        )
        for entry in inactive[: len(self._entries) - _MAX_REGISTERED_DOWNLOADS]:
            self._remove_locked(entry.token)

    def register(
        self,
        *,
        cache_key: str,
        writer: Callable[[BinaryIO], None],
        file_name: str,
        mime_type: str,
        ttl_s: float,
    ) -> str:
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            existing_token = self._tokens_by_cache_key.get(cache_key)
            existing = (
                self._entries.get(existing_token)
                if existing_token is not None
                else None
            )
            if existing is not None:
                existing.writer = writer
                existing.file_name = file_name
                existing.mime_type = mime_type
                existing.expires_at_monotonic_s = now + ttl_s
                return existing.token

            token = secrets.token_urlsafe(32)
            self._entries[token] = _DownloadEntry(
                token=token,
                cache_key=cache_key,
                writer=writer,
                file_name=file_name,
                mime_type=mime_type,
                expires_at_monotonic_s=now + ttl_s,
            )
            self._tokens_by_cache_key[cache_key] = token
            return token

    def acquire(self, token: str) -> _DownloadEntry | None:
        now = time.monotonic()
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(token)
            if entry is None or entry.expires_at_monotonic_s <= now:
                return None
            entry.active_requests += 1
            return entry

    def release(self, token: str) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return
            entry.active_requests = max(entry.active_requests - 1, 0)
            if entry.active_requests == 0 and entry.expires_at_monotonic_s <= now:
                self._remove_locked(token)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._tokens_by_cache_key.clear()


_REGISTRY = _DownloadRegistry()


def _attachment_header(file_name: str) -> str:
    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
    if not fallback:
        fallback = "amplifier_results.ampres.zip"
    encoded = quote(file_name, safe="")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


class _StreamingDownloadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        path = urlsplit(self.path).path
        parts = path.strip("/").split("/", 2)
        if len(parts) < 2 or parts[0] != "download":
            self.send_error(404)
            return
        token = parts[1]
        entry = _REGISTRY.acquire(token)
        if entry is None:
            self.send_error(404, "Download link expired; refresh the results page.")
            return

        try:
            self.send_response(200)
            self.send_header("Content-Type", entry.mime_type)
            self.send_header("Content-Disposition", _attachment_header(entry.file_name))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            # No Content-Length is needed: closing this loopback connection marks
            # the end of the streamed ZIP and lets the browser start immediately.
            self.send_header("Connection", "close")
            self.end_headers()
            entry.writer(self.wfile)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The user cancelled the browser download.  The archive writer's
            # temporary-directory context performs the actual disk cleanup.
            pass
        except OSError as exc:
            _LOGGER.info("Local streaming download ended early: %s", exc)
        except Exception:
            _LOGGER.exception("Local streaming download failed")
        finally:
            self.close_connection = True
            _REGISTRY.release(token)

    def log_message(self, _format: str, *args: object) -> None:
        # Avoid a noisy access log for large local downloads. Exceptions still
        # use the application's normal logger; no log file is created here.
        return


class _LoopbackDownloadServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_SERVER_LOCK = threading.Lock()
_SERVER: _LoopbackDownloadServer | None = None
_SERVER_THREAD: threading.Thread | None = None


def _get_server() -> _LoopbackDownloadServer:
    global _SERVER, _SERVER_THREAD
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER
        server = _LoopbackDownloadServer(
            ("127.0.0.1", 0),
            _StreamingDownloadHandler,
        )
        thread = threading.Thread(
            target=server.serve_forever,
            name="amplifier-streaming-download",
            daemon=True,
        )
        thread.start()
        _SERVER = server
        _SERVER_THREAD = thread
        return server


def register_streaming_download(
    writer: Callable[[BinaryIO], None],
    *,
    cache_key: str,
    file_name: str,
    mime_type: str = "application/zip",
    ttl_s: float = _DEFAULT_TTL_S,
) -> str:
    """Return a loopback URL whose first byte is generated only on a GET.

    Registration creates no archive and no temporary file. Re-registering the
    same cache key refreshes a small in-memory capability rather than leaking a
    new token on every Streamlit rerun.
    """

    if not file_name or "/" in file_name or "\\" in file_name:
        raise ValueError("file_name must be a plain file name")
    if ttl_s <= 0.0:
        raise ValueError("ttl_s must be positive")
    server = _get_server()
    token = _REGISTRY.register(
        cache_key=cache_key,
        writer=writer,
        file_name=file_name,
        mime_type=mime_type,
        ttl_s=ttl_s,
    )
    port = int(server.server_address[1])
    return f"http://127.0.0.1:{port}/download/{token}/{quote(file_name, safe='')}"


def _shutdown_server() -> None:
    global _SERVER, _SERVER_THREAD
    with _SERVER_LOCK:
        server = _SERVER
        thread = _SERVER_THREAD
        _SERVER = None
        _SERVER_THREAD = None
    _REGISTRY.clear()
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2.0)


atexit.register(_shutdown_server)

