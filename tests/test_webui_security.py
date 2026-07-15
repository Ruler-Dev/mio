"""Security regressions for Web UI storage, uploads, and local rendering."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import zipfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from starlette.websockets import WebSocketDisconnect

from mio.webui import router as webui
from mio.webui import scheduler
from mio.web_security import (
    WebSecurityMiddleware,
    configure_runtime_web_security,
    host_allowed,
    reject_untrusted_websocket,
    reset_web_security_state,
    webui_model_skill_allowed,
    webui_skill_risk,
    websocket_origin_allowed,
)


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.mark.parametrize(
    "identifier",
    ["../outside", "..", ".hidden", "a/b", r"a\b", "has space", "x" * 65, 42, None],
)
def test_storage_identifiers_reject_path_traversal(identifier):
    with pytest.raises(HTTPException) as error:
        webui._json_storage_path(Path("/tmp/sessions"), identifier, label="session")
    assert error.value.status_code == 400


def test_session_persistence_stays_inside_configured_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_sessions_dir", tmp_path)

    result = run(
        webui.save_session(
            {
                "id": "safe-session_1",
                "messages": [{"role": "user", "content": "A safe title"}],
            }
        )
    )

    assert result["id"] == "safe-session_1"
    assert (tmp_path / "safe-session_1.json").is_file()
    assert run(webui.load_session("safe-session_1"))["title"] == "A safe title"
    with pytest.raises(HTTPException) as error:
        run(webui.save_session({"id": "../escape", "messages": []}))
    assert error.value.status_code == 400
    assert not (tmp_path.parent / "escape.json").exists()


def test_session_scans_and_workspace_export_never_follow_symlinks(tmp_path, monkeypatch):
    home = tmp_path / "home"
    mio_root = home / ".mio"
    sessions = mio_root / "sessions"
    sessions.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"title":"SECRET","messages":[]}', encoding="utf-8")
    (sessions / "inside.json").write_text(
        '{"id":"inside","title":"Inside","messages":[]}',
        encoding="utf-8",
    )
    (sessions / "linked.json").symlink_to(outside)
    monkeypatch.setattr(webui, "_sessions_dir", sessions)

    listed = run(webui.list_sessions())
    assert [item["id"] for item in listed["sessions"]] == ["inside"]
    assert webui._regular_files_confined(sessions) == [(sessions / "inside.json").resolve()]

    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    response = run(webui.export_workspace())

    async def collect_body():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        return b"".join(chunks)

    archive = zipfile.ZipFile(BytesIO(run(collect_body())))
    assert "sessions/inside.json" in archive.namelist()
    assert "sessions/linked.json" not in archive.namelist()
    assert b"SECRET" not in b"".join(archive.read(name) for name in archive.namelist())


def test_direct_session_access_rejects_symlink_aliases(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "real.json").write_text(
        '{"id":"real","title":"Real","messages":[]}',
        encoding="utf-8",
    )
    (sessions / "alias.json").symlink_to(sessions / "real.json")
    monkeypatch.setattr(webui, "_sessions_dir", sessions)

    with pytest.raises(HTTPException) as load_error:
        run(webui.load_session("alias"))
    assert load_error.value.status_code == 400

    with pytest.raises(HTTPException) as save_error:
        run(webui.save_session({"id": "alias", "messages": []}))
    assert save_error.value.status_code == 400
    assert run(webui.load_session("real"))["title"] == "Real"


def test_upload_normalizes_name_does_not_overwrite_and_extracts_text(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_downloads_dir", lambda: tmp_path)

    unsafe_name = "..\\folder/unsafe:\x00?.txt"
    first = UploadFile(BytesIO(b"first"), filename=unsafe_name, size=5)
    second = UploadFile(BytesIO(b"second"), filename=unsafe_name, size=6)
    one = run(webui.upload_attachment(first))
    two = run(webui.upload_attachment(second))

    assert one["filename"] == "unsafe___.txt"
    assert two["filename"] == "unsafe___ (1).txt"
    assert Path(one["path"]).parent == tmp_path
    assert Path(one["path"]).read_bytes() == b"first"
    assert Path(two["path"]).read_bytes() == b"second"
    assert one["extracted_text"] == "first"


def test_upload_limit_is_enforced_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_downloads_dir", lambda: tmp_path)
    monkeypatch.setattr(webui, "_MAX_UPLOAD_BYTES", 4)
    upload = UploadFile(BytesIO(b"12345"), filename="large.txt", size=None)

    with pytest.raises(HTTPException) as error:
        run(webui.upload_attachment(upload))

    assert error.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_download_listing_and_serving_never_follow_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_downloads_dir", lambda: tmp_path)
    (tmp_path / "report.txt").write_text("public", encoding="utf-8")
    outside = tmp_path.parent / "private.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)

    listed = run(webui.list_attachments())
    assert [entry["name"] for entry in listed["files"]] == ["report.txt"]
    assert run(webui.serve_generated_file("linked.txt")).status_code == 404

    response = run(webui.serve_generated_file("report.txt"))

    async def collect_body():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        return b"".join(chunks)

    assert response.status_code == 200
    assert run(collect_body()) == b"public"


def test_project_files_are_reduced_to_safe_download_names(tmp_path, monkeypatch):
    projects_path = tmp_path / "projects.json"
    monkeypatch.setattr(webui, "_projects_path", lambda: projects_path)

    result = run(
        webui.save_project(
            {
                "id": "safe-project",
                "files": ["notes.txt", "../../secret.txt", "folder/file.pdf", None],
            }
        )
    )

    assert result["files"] == ["notes.txt"]
    assert webui._load_projects()[0]["files"] == ["notes.txt"]


def test_shared_artifact_json_cannot_break_out_of_script():
    webui._shared_artifacts.clear()
    payload = '</script><script id="owned">window.owned=true</script>'
    run(
        webui.share_artifact(
            {
                "identifier": "safe-share",
                "type": "text/html",
                "title": "Unsafe payload",
                "content": payload,
            }
        )
    )

    response = run(webui.view_shared_artifact("safe-share"))
    page = response.body.decode()
    assert payload not in page
    assert "\\u003c/script\\u003e" in page
    assert "allow-same-origin" not in page


def test_markdown_and_artifact_renderers_use_local_security_boundary():
    root = Path(__file__).parents[1]
    shell = (root / "mio/webui/mio_ui.html").read_text()
    sanitizer = (root / "mio/webui/assets/sanitize.js").read_text()
    browser_security = (root / "mio/webui/assets/security.js").read_text()

    assert "'sanitize'" in shell
    assert "Mio.sanitizeHtml(rendered)" in shell
    assert "allow-same-origin" not in shell
    assert "wrap.innerHTML = art.content" not in shell
    assert "makeSandboxedIframe(art.content, false)" in shell
    assert '"SCRIPT"' in sanitizer
    assert '"IFRAME"' in sanitizer
    assert "SAFE_PROTOCOLS" in sanitizer
    assert "data-artifact-id=\"${artifactId}\"" in shell
    assert "map[t] || 'Unsupported artifact'" in shell
    assert "image.src = url" in shell
    assert "openArtifact('${art.id}')" not in shell
    assert "openArtifact('${a.id}')" not in shell
    assert "@xenova/transformers" not in shell.lower()
    assert '<script src="/ui/assets/security.js"></script>' in shell
    assert "X-Mio-CSRF-Token" in browser_security
    assert "mio-csrf." in browser_security
    assert "new WebSocket" not in shell
    top_level_head = shell.split("<!-- Modular feature modules", 1)[0]
    parser = _RemoteAssetParser()
    parser.feed(top_level_head)
    assert parser.remote_assets == []
    vendored = [
        "vendor_marked_12.0.2.min.js",
        "vendor_prism_1.29.0.js",
        "vendor_prism_python_1.29.0.min.js",
        "vendor_prism_typescript_1.29.0.min.js",
        "vendor_prism_jsx_1.29.0.min.js",
        "vendor_prism_tsx_1.29.0.min.js",
        "vendor_prism_bash_1.29.0.min.js",
        "vendor_prism_json_1.29.0.min.js",
        "vendor_prism_css_1.29.0.min.js",
        "vendor_prism_sql_1.29.0.min.js",
        "vendor_prism_rust_1.29.0.min.js",
        "vendor_prism_go_1.29.0.min.js",
        "vendor_prism_yaml_1.29.0.min.js",
        "vendor_prism_tomorrow_1.29.0.min.css",
    ]
    for name in vendored:
        assert f'/ui/assets/{name}' in top_level_head
        assert (root / "mio/webui/assets" / name).stat().st_size > 0


class _RemoteAssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.remote_assets: list[dict[str, str]] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        url = values.get("src") if tag == "script" else values.get("href") if tag == "link" else None
        if url and url.startswith("https://"):
            self.remote_assets.append(values)


def test_ui_response_sets_browser_security_boundary(monkeypatch):
    monkeypatch.setattr(scheduler, "init", lambda *args, **kwargs: None)
    response = run(webui.serve_ui())
    csp = response.headers["content-security-policy"]
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'none'" in csp
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"


class _FakeWebSocket:
    def __init__(self, origin: str | None, url: str = "ws://127.0.0.1:9090/ui/ws/chat"):
        from starlette.datastructures import Headers, URL

        self.headers = Headers({"origin": origin} if origin is not None else {})
        self.url = URL(url)


def test_websocket_origins_default_to_same_port_loopback(monkeypatch):
    monkeypatch.delenv("MIO_CORS_ORIGINS", raising=False)
    assert websocket_origin_allowed(_FakeWebSocket(None))
    assert websocket_origin_allowed(_FakeWebSocket("http://127.0.0.1:9090"))
    # Loopback aliases are different browser origins; exact host + port wins.
    assert not websocket_origin_allowed(_FakeWebSocket("http://localhost:9090"))
    assert websocket_origin_allowed(
        _FakeWebSocket("http://localhost:9090", "ws://localhost:9090/ui/ws/chat")
    )
    assert not websocket_origin_allowed(_FakeWebSocket("https://attacker.example"))
    assert not websocket_origin_allowed(_FakeWebSocket("http://localhost:9091"))
    assert not websocket_origin_allowed(_FakeWebSocket("file://localhost"))


def test_websocket_origin_allowlist_is_explicit_and_additive(monkeypatch):
    monkeypatch.setenv("MIO_CORS_ORIGINS", "https://mio.example, http://localhost:7777/")
    assert websocket_origin_allowed(_FakeWebSocket("https://mio.example"))
    assert websocket_origin_allowed(_FakeWebSocket("http://localhost:7777"))
    assert websocket_origin_allowed(_FakeWebSocket("http://127.0.0.1:9090"))
    assert not websocket_origin_allowed(_FakeWebSocket("https://attacker.example"))


def _security_test_app(
    *,
    allow_test_host: bool = False,
    max_body_bytes: int | None = None,
    webui_enabled=None,
) -> FastAPI:
    application = FastAPI()
    middleware_options = {"allow_test_host": allow_test_host}
    if max_body_bytes is not None:
        middleware_options["max_body_bytes"] = max_body_bytes
    if webui_enabled is not None:
        middleware_options["webui_enabled"] = webui_enabled
    application.add_middleware(WebSecurityMiddleware, **middleware_options)

    @application.get("/ui")
    async def shell():
        return HTMLResponse("<h1>Mio</h1>")

    @application.post("/ui/api/mutate")
    async def mutate():
        return {"ok": True}

    @application.post("/v1/mutate")
    async def api_mutate():
        return {"native": True}

    @application.post("/v1/mcp/health")
    async def mcp_health_probe():
        return {"probed": True}

    @application.post("/v1/read-body")
    async def api_read_body(request: Request):
        return {"bytes": len(await request.body())}

    @application.get("/dashboard")
    async def dashboard_shell():
        return HTMLResponse("<h1>Metrics</h1>")

    @application.websocket("/ui/ws/test")
    async def websocket_endpoint(websocket: WebSocket):
        if await reject_untrusted_websocket(websocket):
            return
        await websocket.accept()
        await websocket.send_text("ok")

    # Models endpoints such as /ws/metrics may not own their own guard. The
    # middleware must enforce the boundary before routing every browser WS.
    @application.websocket("/ws/unguarded")
    async def unguarded_websocket(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("unguarded-ok")

    return application


@pytest.mark.parametrize(
    "authority",
    ["127.0.0.1", "127.9.8.7:9090", "localhost:9090", "[::1]:9090"],
)
def test_host_allowlist_accepts_only_well_formed_loopback(authority, monkeypatch):
    monkeypatch.delenv("MIO_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("MIO_ALLOW_TEST_HOST", raising=False)
    assert host_allowed(authority)


@pytest.mark.parametrize(
    "authority",
    [
        "evil.example",
        "evil.example:9090",
        "localhost.evil.example",
        "localhost@evil.example",
        "127.0.0.1.evil.example",
        "[::1]@evil.example",
        "127.0.0.1,evil.example",
        "127.0.0.1:bad",
        "testserver",
        "",
    ],
)
def test_host_allowlist_rejects_rebinding_and_ambiguous_authorities(authority, monkeypatch):
    monkeypatch.delenv("MIO_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("MIO_ALLOW_TEST_HOST", raising=False)
    assert not host_allowed(authority)


def test_testclient_hostname_requires_explicit_test_only_opt_in(monkeypatch):
    monkeypatch.delenv("MIO_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("MIO_ALLOW_TEST_HOST", raising=False)
    assert not host_allowed("testserver")
    assert host_allowed("testserver", allow_test_host=True)
    blocked = TestClient(_security_test_app())
    assert blocked.get("/ui").status_code == 400
    explicit = TestClient(_security_test_app(allow_test_host=True))
    assert explicit.get("/ui").status_code == 200


def test_remote_bind_policy_allows_only_concrete_same_origin_websocket(monkeypatch):
    monkeypatch.delenv("MIO_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("MIO_CORS_ORIGINS", raising=False)
    reset_web_security_state()
    configure_runtime_web_security("192.168.50.20", 9090, allow_remote=True)
    try:
        assert host_allowed("192.168.50.20:9090")
        assert not host_allowed("192.168.50.20:9091")
        assert not host_allowed("mio.lan:9090")

        client = TestClient(
            _security_test_app(),
            base_url="http://192.168.50.20:9090",
        )
        assert client.get("/ui").status_code == 200
        csrf = client.cookies.get("mio_csrf")
        session = client.cookies.get("mio_ui_session")
        websocket_headers = {
            "Host": "192.168.50.20:9090",
            "Origin": "http://192.168.50.20:9090",
            # Starlette's WebSocket TestClient keeps a synthetic ws://testserver
            # URL even when the HTTP base URL is remote, so forward the shell
            # cookies explicitly to exercise Mio's real handshake boundary.
            "Cookie": f"mio_ui_session={session}; mio_csrf={csrf}",
        }
        with client.websocket_connect(
            "/ui/ws/test",
            headers=websocket_headers,
            subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
        ) as websocket:
            assert websocket.receive_text() == "ok"

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/ui/ws/test",
                headers={
                    **websocket_headers,
                    "Origin": "https://attacker.example",
                },
                subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
            ):
                pass
    finally:
        reset_web_security_state()


def test_wildcard_remote_bind_limits_lan_hosts_to_numeric_private_ip_and_port(monkeypatch):
    monkeypatch.delenv("MIO_TRUSTED_HOSTS", raising=False)
    reset_web_security_state()
    configure_runtime_web_security("0.0.0.0", 9090, allow_remote=True)
    try:
        assert host_allowed("192.168.1.44:9090")
        assert host_allowed("[fd00::44]:9090")
        assert not host_allowed("192.168.1.44:9091")
        assert not host_allowed("mio.local:9090")
        assert not host_allowed("203.0.113.44:9090")
        assert not host_allowed("8.8.8.8:9090")
    finally:
        reset_web_security_state()


def test_webui_gate_disables_http_and_websocket_after_mount():
    enabled = {"value": True}
    reset_web_security_state()
    client = TestClient(
        _security_test_app(webui_enabled=lambda: enabled["value"]),
        base_url="http://127.0.0.1:9090",
    )
    assert client.get("/ui").status_code == 200
    enabled["value"] = False
    response = client.get("/ui")
    assert response.status_code == 404
    assert response.json()["detail"] == "Mio UI is disabled"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ui/ws/test"):
            pass


def test_csrf_session_and_same_origin_are_required_for_mutations():
    reset_web_security_state()
    client = TestClient(_security_test_app(), base_url="http://127.0.0.1:9090")
    shell = client.get("/ui")
    assert shell.status_code == 200
    assert client.cookies.get("mio_ui_session")
    csrf = client.cookies.get("mio_csrf")
    assert csrf

    assert client.post("/ui/api/mutate").status_code == 403
    assert client.post(
        "/ui/api/mutate",
        headers={"X-Mio-CSRF-Token": csrf, "Origin": "https://attacker.example"},
    ).status_code == 403
    assert client.post(
        "/ui/api/mutate",
        headers={
            "X-Mio-CSRF-Token": csrf,
            "Origin": "http://127.0.0.1:9090",
            "Sec-Fetch-Site": "cross-site",
        },
    ).status_code == 403
    assert client.post(
        "/ui/api/mutate",
        headers={
            "X-Mio-CSRF-Token": csrf,
            "Origin": "http://127.0.0.1:9090",
            "Sec-Fetch-Site": "same-origin",
        },
    ).json() == {"ok": True}

    no_session = TestClient(_security_test_app(), base_url="http://127.0.0.1:9090")
    assert no_session.post(
        "/ui/api/mutate",
        headers={"X-Mio-CSRF-Token": csrf, "Origin": "http://127.0.0.1:9090"},
    ).status_code == 403


def test_cross_origin_browser_v1_mutation_is_denied_but_native_client_is_allowed():
    application = _security_test_app()
    native = TestClient(application, base_url="http://127.0.0.1:9090")
    assert native.post("/v1/mutate").json() == {"native": True}
    assert native.post(
        "/v1/mutate",
        headers={"Origin": "https://attacker.example"},
    ).status_code == 403
    assert native.post(
        "/v1/mutate",
        headers={"Sec-Fetch-Site": "cross-site"},
    ).status_code == 403
    assert native.post(
        "/v1/mutate",
        headers={"Origin": "http://127.0.0.1:9090", "Sec-Fetch-Site": "same-origin"},
    ).json() == {"native": True}


def test_side_effectful_mcp_health_probe_requires_webui_csrf_session():
    reset_web_security_state()
    client = TestClient(_security_test_app(), base_url="http://127.0.0.1:9090")

    # A drive-by page cannot trigger provider process launches through an
    # image/no-CORS GET or an unauthenticated form POST.
    assert client.get("/v1/mcp/health").status_code == 405
    assert client.post("/v1/mcp/health").status_code == 403
    assert client.post(
        "/v1/mcp/health",
        headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
    ).status_code == 403

    assert client.get("/ui").status_code == 200
    csrf = client.cookies.get("mio_csrf")
    assert csrf
    response = client.post(
        "/v1/mcp/health",
        headers={
            "X-Mio-CSRF-Token": csrf,
            "Origin": "http://127.0.0.1:9090",
            "Sec-Fetch-Site": "same-origin",
        },
    )
    assert response.json() == {"probed": True}


def test_explicit_cors_origin_can_call_api_but_does_not_broaden_other_origins(monkeypatch):
    monkeypatch.setenv("MIO_CORS_ORIGINS", "https://allowed.example")
    client = TestClient(_security_test_app(), base_url="http://127.0.0.1:9090")

    allowed = client.post(
        "/v1/mutate",
        headers={"Origin": "https://allowed.example", "Sec-Fetch-Site": "cross-site"},
    )
    denied = client.post(
        "/v1/mutate",
        headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
    )
    same_origin = client.post(
        "/v1/mutate",
        headers={"Origin": "http://127.0.0.1:9090", "Sec-Fetch-Site": "same-origin"},
    )

    assert allowed.json() == {"native": True}
    assert denied.status_code == 403
    assert same_origin.json() == {"native": True}


def test_http_body_limit_rejects_declared_and_streamed_oversize_requests():
    client = TestClient(
        _security_test_app(max_body_bytes=8),
        base_url="http://127.0.0.1:9090",
    )

    declared = client.post("/v1/read-body", content=b"123456789")
    assert declared.status_code == 413

    def chunks():
        yield b"1234"
        yield b"56789"

    streamed = client.post("/v1/read-body", content=chunks())
    assert streamed.status_code == 413
    assert client.post("/v1/read-body", content=b"12345678").json() == {"bytes": 8}


def test_dashboard_shell_issues_session_for_metrics_websocket():
    reset_web_security_state()
    client = TestClient(_security_test_app(allow_test_host=True))
    response = client.get("/dashboard")
    assert response.status_code == 200
    csrf = client.cookies.get("mio_csrf")
    assert csrf
    with client.websocket_connect(
        "/ws/unguarded",
        headers={"Origin": "http://testserver"},
        subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
    ) as websocket:
        assert websocket.receive_text() == "unguarded-ok"


def test_browser_websocket_requires_origin_session_and_csrf_protocol(monkeypatch):
    reset_web_security_state()
    client = TestClient(_security_test_app(allow_test_host=True))
    assert client.get("/ui").status_code == 200
    csrf = client.cookies.get("mio_csrf")

    with client.websocket_connect(
        "/ui/ws/test",
        headers={"Origin": "http://testserver"},
        subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
    ) as websocket:
        assert websocket.receive_text() == "ok"

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ui/ws/test",
            headers={"Origin": "http://testserver"},
            subprotocols=["mio-ui"],
        ):
            pass

    monkeypatch.setenv("MIO_CORS_ORIGINS", "https://allowed.example")
    with client.websocket_connect(
        "/ui/ws/test",
        headers={
            "Origin": "https://allowed.example",
            "Sec-Fetch-Site": "cross-site",
        },
        subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
    ) as websocket:
        assert websocket.receive_text() == "ok"

    with client.websocket_connect(
        "/ws/unguarded",
        headers={"Origin": "http://testserver"},
        subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
    ) as websocket:
        assert websocket.receive_text() == "unguarded-ok"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/unguarded",
            headers={"Origin": "https://attacker.example"},
            subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
        ):
            pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ui/ws/test",
            headers={"Origin": "https://attacker.example"},
            subprotocols=["mio-ui", f"mio-csrf.{csrf}"],
        ):
            pass


def test_chat_websocket_negotiates_stable_protocol_without_echoing_csrf(monkeypatch):
    reset_web_security_state()
    monkeypatch.setattr(scheduler, "init", lambda *args, **kwargs: None)
    application = FastAPI()
    application.add_middleware(WebSecurityMiddleware, allow_test_host=True)
    application.include_router(webui.router)
    client = TestClient(application, base_url="http://testserver")

    assert client.get("/ui").status_code == 200
    csrf = client.cookies.get("mio_csrf")
    assert csrf
    csrf_protocol = f"mio-csrf.{csrf}"

    with client.websocket_connect(
        "/ui/ws/chat",
        headers={"Origin": "http://testserver"},
        subprotocols=["mio-ui", csrf_protocol],
    ) as websocket:
        assert websocket.accepted_subprotocol == "mio-ui"
        assert websocket.accepted_subprotocol != csrf_protocol


def _request_with_headers(**headers: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/ui/api/skills/run",
            "headers": [(key.replace("_", "-").encode(), value.encode()) for key, value in headers.items()],
        }
    )


def test_sensitive_skill_direct_run_is_deny_by_default_then_double_opt_in(monkeypatch):
    from mio.webui.skills import SKILLS

    calls = []
    monkeypatch.setitem(SKILLS["execute_python"], "function", lambda **kwargs: calls.append(kwargs) or {"ok": True})
    monkeypatch.delenv("MIO_WEBUI_SKILL_GRANTS", raising=False)
    request = _request_with_headers(x_mio_dangerous_action="execute_python")

    with pytest.raises(HTTPException) as denied:
        run(webui.run_skill(
            {"name": "execute_python", "args": {"code": "print(1)"}, "confirm_sensitive": True},
            request,
        ))
    assert denied.value.status_code == 403
    assert calls == []

    monkeypatch.setenv("MIO_WEBUI_SKILL_GRANTS", "execute_python")
    with pytest.raises(HTTPException):
        run(webui.run_skill({"name": "execute_python", "args": {"code": "print(1)"}}, request))
    result = run(webui.run_skill(
        {"name": "execute_python", "args": {"code": "print(1)"}, "confirm_sensitive": True},
        request,
    ))
    assert result == {"ok": True, "result": {"ok": True}}
    assert calls == [{"code": "print(1)"}]


def test_model_tool_policy_is_fail_closed_for_local_and_executable_skills(monkeypatch):
    monkeypatch.delenv("MIO_WEBUI_SKILL_GRANTS", raising=False)
    assert webui_skill_risk("format_json") == "read"
    assert webui_skill_risk("journal_read") == "local-read"
    assert webui_skill_risk("execute_python") == "execute"
    assert webui_skill_risk("new_unclassified_plugin") == "sensitive"
    assert webui_model_skill_allowed("format_json")
    assert not webui_model_skill_allowed("journal_read", ["journal_read"])
    assert not webui_model_skill_allowed("execute_python", ["execute_python"])

    monkeypatch.setenv("MIO_WEBUI_SKILL_GRANTS", "journal_read,execute_python")
    assert not webui_model_skill_allowed("execute_python")
    assert webui_model_skill_allowed("journal_read", ["journal_read"])
    assert webui_model_skill_allowed("execute_python", ["execute_python"])


def test_fetch_url_and_image_cache_block_loopback_ssrf(tmp_path, monkeypatch):
    from mio.webui.skills import fetch_url

    result = fetch_url("http://127.0.0.1:9090/private")
    assert result["error"].startswith("blocked_url:")
    monkeypatch.setattr(webui, "IMAGE_CACHE_DIR", tmp_path)
    assert webui.cache_image_to_disk("http://127.0.0.1:9090/private.png") is None


def test_image_proxy_host_allowlist_uses_domain_boundaries():
    response = run(webui.proxy_image("https://evilmyanimelist.net/poster.jpg"))
    assert response.status_code == 403


def test_scheduler_starts_only_on_a_running_event_loop(monkeypatch):
    started = asyncio.Event()

    async def fake_loop():
        started.set()

    monkeypatch.setattr(scheduler, "_task", None)
    monkeypatch.setattr(scheduler, "_run_loop", fake_loop)
    scheduler.init(object())
    assert scheduler._task is None

    async def start_and_check():
        scheduler.init(object())
        await asyncio.sleep(0)
        assert started.is_set()
        assert scheduler._task is not None
        assert scheduler._task.get_name() == "mio-scheduler"

    asyncio.run(start_and_check())


def test_weekly_schedule_runs_at_most_once_per_calendar_day():
    import datetime as dt

    now = dt.datetime(2026, 7, 15, 12, 0)  # Wednesday
    schedule = {
        "enabled": True,
        "cadence": {"kind": "weekly", "weekday": now.weekday(), "hour": 9, "minute": 0},
    }
    assert scheduler._should_run_now(schedule, now, None)
    assert not scheduler._should_run_now(schedule, now, "2026-07-15T09:00:00")
    assert scheduler._should_run_now(schedule, now, "2026-07-08T09:00:00")


def test_scheduler_generation_runs_in_worker_under_gpu_lock(monkeypatch):
    import threading

    main_thread = threading.get_ident()

    class Lock:
        active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, *_args):
            self.active = False

    lock = Lock()

    class Engine:
        def generate_stream(self, messages, max_tokens):
            assert threading.get_ident() != main_thread
            assert lock.active
            assert messages[-1]["content"] == "scheduled"
            assert max_tokens == 1500
            yield "done", None

    class Manager:
        def loaded_tiers(self):
            return ["small"]

        def get_engine(self, tier):
            assert tier == "small"
            return Engine()

    monkeypatch.setattr(scheduler, "_manager_ref", Manager())
    monkeypatch.setattr(scheduler, "_gpu_lock_ref", lock)
    result = asyncio.run(scheduler._fire({"prompt": "scheduled", "tier": "small"}))
    assert result == {"ok": True, "tier": "small", "output": "done"}


class _InlineScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts: list[str] = []
        self._parts: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "script" and not dict(attrs).get("src"):
            self._parts = []

    def handle_data(self, data):
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._parts is not None:
            self.scripts.append("".join(self._parts))
            self._parts = None


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_all_webui_javascript_has_valid_syntax(tmp_path):
    root = Path(__file__).parents[1]
    assets = sorted((root / "mio/webui/assets").glob("*.js"))
    failures = []
    for path in assets:
        result = subprocess.run(
            ["node", "--check", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            failures.append(f"{path.name}: {result.stderr}")

    html_pages = [
        root / "mio/webui/mio_ui.html",
        root / "mio/webui/assets/compare.html",
    ]
    for page in html_pages:
        collector = _InlineScriptCollector()
        collector.feed(page.read_text())
        assert collector.scripts, f"no inline scripts found in {page.name}"
        for index, source in enumerate(collector.scripts):
            path = tmp_path / f"{page.stem}-inline-{index}.js"
            path.write_text(source)
            result = subprocess.run(
                ["node", "--check", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                failures.append(f"{page.name} script {index}: {result.stderr}")

    assert failures == []
