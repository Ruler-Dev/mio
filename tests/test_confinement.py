"""Filesystem and image-proxy confinement regressions."""

from __future__ import annotations

import asyncio
import sqlite3
import stat
import zipfile

import pytest

from mio.webui import (
    image_proxy,
    router as webui,
    safe_files,
    skills_docs,
    skills_misc,
    skills_python,
    skills_rag,
)


def test_skills_python_output_is_confined_to_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()

    output = skills_python._out("report", ".pdf")
    assert output == downloads / "report.pdf"

    for filename in ("../escape", "../../escape", "/tmp/escape", "sub/escape", "..\\escape"):
        with pytest.raises(safe_files.UnsafePathError):
            skills_python._out(filename, ".pdf")


def test_skills_python_output_rejects_symlinked_root_or_leaf(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    downloads = tmp_path / "Downloads"
    downloads.symlink_to(outside, target_is_directory=True)

    with pytest.raises(safe_files.UnsafePathError, match="symlinked directory"):
        skills_python._out("report", ".pdf")

    downloads.unlink()
    downloads.mkdir()
    (downloads / "report.pdf").symlink_to(outside / "stolen.pdf")
    with pytest.raises(safe_files.UnsafePathError, match="symlinked path"):
        skills_python._out("report", ".pdf")


def test_all_builtin_output_helpers_reject_downloads_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Downloads").mkdir()

    for output_helper in (
        skills_python._out,
        skills_docs._output_path,
        skills_misc._output_path,
    ):
        for filename in ("../escape", "/tmp/escape", "sub/escape", "..\\escape"):
            with pytest.raises(safe_files.UnsafePathError):
                output_helper(filename, ".pdf")


def test_downloads_input_rejects_traversal_absolute_escape_and_symlinks(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    outside = tmp_path / "outside"
    downloads.mkdir()
    outside.mkdir()
    (downloads / "inside.txt").write_text("inside")
    (outside / "secret.txt").write_text("secret")
    (downloads / "linked.txt").symlink_to(outside / "secret.txt")

    assert safe_files.downloads_input_path("inside.txt") == (
        downloads / "inside.txt"
    ).resolve()
    for value in (
        "../outside/secret.txt",
        "..\\outside\\secret.txt",
        str(outside / "secret.txt"),
        "linked.txt",
    ):
        with pytest.raises(safe_files.UnsafePathError):
            safe_files.downloads_input_path(value)


def test_zip_files_reads_only_regular_download_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    outside = tmp_path / "outside"
    downloads.mkdir()
    outside.mkdir()
    (downloads / "inside.txt").write_text("inside")
    (outside / "secret.txt").write_text("secret")
    (downloads / "linked.txt").symlink_to(outside / "secret.txt")

    result = skills_python.zip_files(["inside.txt"], "bundle.zip")
    assert result["files"] == ["inside.txt"]
    with zipfile.ZipFile(downloads / "bundle.zip") as archive:
        assert archive.read("inside.txt") == b"inside"

    for value in ("../outside/secret.txt", str(outside / "secret.txt"), "linked.txt"):
        rejected = skills_python.zip_files([value], "rejected.zip")
        assert "error" in rejected


def _write_zip(path, members, *, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, content in members:
            if isinstance(name, zipfile.ZipInfo):
                archive.writestr(name, content)
            else:
                archive.writestr(name, content)


def test_unzip_streams_valid_members_without_extractall(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    archive_path = downloads / "valid.zip"
    _write_zip(archive_path, [("nested/readme.txt", b"safe")])
    monkeypatch.setattr(
        zipfile.ZipFile,
        "extractall",
        lambda *_args, **_kwargs: pytest.fail("extractall must never be used"),
    )

    result = skills_python.unzip_file("valid.zip", "expanded")

    assert result["file_count"] == 1
    assert result["files"] == ["nested/readme.txt"]
    assert (downloads / "expanded" / "nested" / "readme.txt").read_bytes() == b"safe"


@pytest.mark.parametrize(
    "member_name",
    ["../escape.txt", "/tmp/escape.txt", "nested\\escape.txt"],
)
def test_unzip_rejects_traversal_and_absolute_members(
    tmp_path,
    monkeypatch,
    member_name,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    _write_zip(downloads / "unsafe.zip", [(member_name, b"no")])

    result = skills_python.unzip_file("unsafe.zip", "expanded")

    assert "error" in result
    assert not (tmp_path / "escape.txt").exists()
    assert not (downloads / "expanded").exists()


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600])
def test_unzip_rejects_symlink_and_special_members(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    member = zipfile.ZipInfo("unsafe-entry")
    member.create_system = 3
    member.external_attr = mode << 16
    _write_zip(downloads / "unsafe.zip", [(member, b"../outside")])

    result = skills_python.unzip_file("unsafe.zip", "expanded")

    assert "error" in result
    assert not (downloads / "expanded").exists()


def test_unzip_rejects_destination_escape_or_symlink(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    outside = tmp_path / "outside"
    downloads.mkdir()
    outside.mkdir()
    _write_zip(downloads / "safe.zip", [("file.txt", b"safe")])

    assert "error" in skills_python.unzip_file("safe.zip", "../outside")
    (downloads / "linked").symlink_to(outside, target_is_directory=True)
    assert "error" in skills_python.unzip_file("safe.zip", "linked")
    assert not (outside / "file.txt").exists()


def test_unzip_enforces_member_count_size_total_and_ratio_caps(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads = tmp_path / "Downloads"
    downloads.mkdir()

    _write_zip(downloads / "count.zip", [("a", b"a"), ("b", b"b")])
    monkeypatch.setattr(skills_python, "_MAX_ZIP_MEMBERS", 1)
    assert "more than" in skills_python.unzip_file("count.zip", "count")["error"]

    _write_zip(downloads / "size.zip", [("large", b"1234")])
    monkeypatch.setattr(skills_python, "_MAX_ZIP_MEMBERS", 1024)
    monkeypatch.setattr(skills_python, "_MAX_ZIP_MEMBER_BYTES", 3)
    assert "member exceeds" in skills_python.unzip_file("size.zip", "size")["error"]

    _write_zip(downloads / "total.zip", [("a", b"12"), ("b", b"34")])
    monkeypatch.setattr(skills_python, "_MAX_ZIP_MEMBER_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(skills_python, "_MAX_ZIP_TOTAL_BYTES", 3)
    assert "expands beyond" in skills_python.unzip_file("total.zip", "total")["error"]

    _write_zip(downloads / "ratio.zip", [("zeros", b"0" * 10_000)])
    monkeypatch.setattr(skills_python, "_MAX_ZIP_TOTAL_BYTES", 256 * 1024 * 1024)
    monkeypatch.setattr(skills_python, "_MAX_ZIP_COMPRESSION_RATIO", 2.0)
    assert "compression-ratio" in skills_python.unzip_file("ratio.zip", "ratio")["error"]


def test_confined_listing_and_obsidian_tree_never_follow_symlinks(tmp_path):
    root = tmp_path / "vault"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "inside.md").write_text("inside")
    (outside / "secret.md").write_text("secret")
    (root / "linked.md").symlink_to(outside / "secret.md")
    (root / "linked-folder").symlink_to(outside, target_is_directory=True)

    files = list(
        safe_files.iter_confined_regular_files(root, suffixes={".md"}, recursive=True)
    )
    assert files == [(root / "inside.md").resolve()]

    tree = safe_files.confined_markdown_tree(root)
    assert [item["name"] for item in tree] == ["inside.md"]


def test_ingest_state_and_obsidian_write_reject_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MIO_HOME", raising=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".mio").symlink_to(outside, target_is_directory=True)
    with pytest.raises(safe_files.UnsafePathError, match="symlinked directory"):
        safe_files.mio_state_directory("ingest", create=True)

    (tmp_path / ".mio").unlink()
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(safe_files.UnsafePathError, match="symlinked directory"):
        safe_files.write_confined_text(
            vault,
            "escape/note.md",
            "must stay confined",
            create_parents=True,
        )
    assert not (outside / "note.md").exists()


def test_mio_state_directory_honors_custom_mio_home_and_rejects_symlink(
    tmp_path,
    monkeypatch,
):
    custom_home = tmp_path / "custom-state"
    monkeypatch.setenv("MIO_HOME", str(custom_home))

    ingest = safe_files.mio_state_directory("ingest", create=True)
    assert ingest == (custom_home / "ingest").resolve()

    ingest.rmdir()
    custom_home.rmdir()
    outside = tmp_path / "outside-state"
    outside.mkdir()
    custom_home.symlink_to(outside, target_is_directory=True)
    with pytest.raises(safe_files.UnsafePathError, match="symlinked directory"):
        safe_files.mio_state_directory("ingest", create=True)


def test_router_ingest_uses_confined_mio_state_and_ignores_symlinks(
    tmp_path,
    monkeypatch,
):
    state = tmp_path / "state"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret")
    monkeypatch.setenv("MIO_HOME", str(state))

    result = asyncio.run(
        webui.ingest_from_browser({
            "url": "https://example.com/page",
            "title": "Safe page",
            "text": "inside",
            "target": "attach",
        })
    )
    assert result["path"].startswith(str(state / "ingest"))
    ingest = state / "ingest"
    (ingest / "linked.md").symlink_to(outside / "secret.md")

    listing = asyncio.run(webui.list_ingested())
    assert [item["title"] for item in listing["items"]] == ["Safe page"]
    assert asyncio.run(webui.delete_ingested("linked"))["deleted"] is False
    assert (outside / "secret.md").read_text() == "secret"


def test_router_obsidian_tree_read_and_write_reject_symlink_escape(
    tmp_path,
    monkeypatch,
):
    state = tmp_path / "state"
    vault = tmp_path / "vault"
    outside = tmp_path / "outside"
    vault.mkdir()
    outside.mkdir()
    (vault / "inside.md").write_text("inside")
    (outside / "secret.md").write_text("secret")
    (vault / "linked.md").symlink_to(outside / "secret.md")
    (vault / "linked-dir").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("MIO_HOME", str(state))

    configured = asyncio.run(webui.obsidian_set_config({"vault_path": str(vault)}))
    assert configured["ok"] is True
    tree = asyncio.run(webui.obsidian_tree())
    assert [item["name"] for item in tree["tree"]] == ["inside.md"]
    assert "error" in asyncio.run(webui.obsidian_read_note("linked.md"))
    assert "error" in asyncio.run(webui.obsidian_read_note("../outside/secret.md"))
    rejected = asyncio.run(
        webui.obsidian_write_note({"path": "linked-dir/stolen.md", "content": "no"})
    )
    assert "error" in rejected
    assert not (outside / "stolen.md").exists()


def test_router_image_proxy_and_disk_cache_use_validated_payload(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "images"
    cache.mkdir()
    png = b"\x89PNG\r\n\x1a\nvalidated"
    payload = image_proxy.ImagePayload(
        png,
        "image/png",
        ".png",
        "https://cdn.myanimelist.net/poster.png",
    )
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append((url, kwargs))
        return payload

    monkeypatch.setattr(webui, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(webui, "fetch_image", fake_fetch)
    webui._img_proxy_cache.clear()

    local = webui.cache_image_to_disk("https://images.example/wrong.jpg")
    assert local and local.endswith(".png")
    assert (cache / local.rsplit("/", 1)[-1]).read_bytes() == png
    assert calls[-1][1] == {"allowed_hosts": None}

    response = asyncio.run(
        webui.proxy_image("https://cdn.myanimelist.net/poster.png")
    )
    assert response.status_code == 200
    assert response.media_type == "image/png"
    assert response.body == png


def test_router_cache_reads_and_serving_never_follow_symlink_leaf(
    tmp_path,
    monkeypatch,
):
    image_cache = tmp_path / "images"
    web_cache = tmp_path / "web"
    outside = tmp_path / "outside"
    image_cache.mkdir()
    web_cache.mkdir()
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"secret")
    (image_cache / "deadbeef.png").symlink_to(outside / "secret.png")
    monkeypatch.setattr(webui, "IMAGE_CACHE_DIR", image_cache)
    monkeypatch.setattr(webui, "WEB_CACHE_DIR", web_cache)

    response = asyncio.run(webui.serve_cached_image("deadbeef.png"))
    assert response.status_code == 404

    url = "https://example.com/page"
    key = webui.hashlib.sha1(url.encode()).hexdigest()
    outside_cache = outside / "cache.json"
    outside_cache.write_text('{"content": "secret"}')
    (web_cache / f"{key}.json").symlink_to(outside_cache)
    assert webui.web_cache_get(url) is None
    webui.web_cache_put(url, "overwrite")
    assert outside_cache.read_text() == '{"content": "secret"}'


def test_rag_index_excludes_symlinked_files_and_directories(tmp_path, monkeypatch):
    root = tmp_path / "notes"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "inside.md").write_text("inside searchable")
    (outside / "secret.md").write_text("outside secret")
    (root / "linked.md").symlink_to(outside / "secret.md")
    (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(skills_rag, "RAG_DB", tmp_path / "rag.sqlite")

    result = skills_rag.index_folder(str(root), label="notes")
    assert result["file_count"] == 1
    with sqlite3.connect(tmp_path / "rag.sqlite") as connection:
        paths = [row[0] for row in connection.execute("SELECT path FROM documents")]
    assert paths == [str((root / "inside.md").resolve())]

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    rejected = skills_rag.index_folder(str(linked_root))
    assert "symlinked directory" in rejected["error"]


def _public_dns(host: str, port: int, **_kwargs):
    suffix = 10 if host.startswith("cdn") else 9
    return [
        (
            image_proxy.socket.AF_INET,
            image_proxy.socket.SOCK_STREAM,
            6,
            "",
            (f"93.184.216.{suffix}", port),
        )
    ]


class _Socket:
    def __init__(self):
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _Response:
    def __init__(self, status: int, headers: dict[str, str], chunks: list[bytes]):
        self.status = status
        self.headers = headers
        self.chunks = list(chunks)

    def getheader(self, name: str):
        return self.headers.get(name)

    def read(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""


class _Connection:
    def __init__(self, response: _Response):
        self.response = response
        self.sock = _Socket()
        self.requests = []
        self.closed = False

    def request(self, *args, **kwargs):
        self.requests.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def test_image_proxy_pins_dns_and_revalidates_redirect_target(monkeypatch):
    def dns(host: str, port: int, **_kwargs):
        ip = "93.184.216.9" if host == "images.example" else "127.0.0.1"
        return [
            (
                image_proxy.socket.AF_INET,
                image_proxy.socket.SOCK_STREAM,
                6,
                "",
                (ip, port),
            )
        ]

    redirect = _Response(
        302,
        {"Location": "https://cdn.example/private.png"},
        [],
    )
    connections = []

    def connection_for(target, _timeout):
        connections.append(target)
        return _Connection(redirect)

    monkeypatch.setattr(image_proxy.socket, "getaddrinfo", dns)
    monkeypatch.setattr(image_proxy, "_connection_for", connection_for)

    with pytest.raises(image_proxy.ImageFetchError, match="private, loopback"):
        image_proxy.fetch_image(
            "https://images.example/start",
            allowed_hosts={"images.example", "cdn.example"},
        )
    assert [target.pinned_ip for target in connections] == ["93.184.216.9"]


def test_image_proxy_uses_magic_and_content_type_for_real_extension(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n" + b"payload"
    response = _Response(
        200,
        {"Content-Type": "image/png", "Content-Length": str(len(png))},
        [png],
    )
    connection = _Connection(response)
    targets = []
    monkeypatch.setattr(image_proxy.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(
        image_proxy,
        "_connection_for",
        lambda target, _timeout: targets.append(target) or connection,
    )

    payload = image_proxy.fetch_image(
        "https://images.example/wrong.jpg",
        allowed_hosts={"images.example"},
    )

    assert payload.data == png
    assert payload.extension == ".png"
    assert payload.media_type == "image/png"
    assert targets[0].pinned_ip == "93.184.216.9"
    assert connection.closed


def test_image_proxy_rejects_mime_magic_mismatch(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n" + b"payload"
    response = _Response(200, {"Content-Type": "image/jpeg"}, [png])
    monkeypatch.setattr(image_proxy.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(
        image_proxy,
        "_connection_for",
        lambda _target, _timeout: _Connection(response),
    )

    with pytest.raises(image_proxy.ImageFetchError, match="does not match"):
        image_proxy.fetch_image(
            "https://images.example/file.jpg",
            allowed_hosts={"images.example"},
        )


def test_image_proxy_streaming_byte_limit(monkeypatch):
    response = _Response(
        200,
        {"Content-Type": "image/png"},
        [b"\x89PNG\r\n\x1a\n", b"abc"],
    )
    monkeypatch.setattr(image_proxy.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(
        image_proxy,
        "_connection_for",
        lambda _target, _timeout: _Connection(response),
    )

    with pytest.raises(image_proxy.ImageFetchError, match="exceeds 10 bytes"):
        image_proxy.fetch_image(
            "https://images.example/file.png",
            allowed_hosts={"images.example"},
            max_bytes=10,
        )


def test_image_proxy_total_time_budget_applies_between_chunks(monkeypatch):
    response = _Response(
        200,
        {"Content-Type": "image/png"},
        [b"\x89PNG\r\n\x1a\n", b"payload"],
    )
    ticks = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(image_proxy.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(image_proxy.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        image_proxy,
        "_connection_for",
        lambda _target, _timeout: _Connection(response),
    )

    with pytest.raises(image_proxy.ImageFetchError, match="time budget"):
        image_proxy.fetch_image(
            "https://images.example/file.png",
            allowed_hosts={"images.example"},
            timeout_seconds=1.0,
        )


def test_image_proxy_host_allowlist_uses_domain_boundaries(monkeypatch):
    monkeypatch.setattr(
        image_proxy.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: pytest.fail("disallowed host must not resolve"),
    )
    with pytest.raises(image_proxy.ImageFetchError, match="host is not allowed"):
        image_proxy.fetch_image(
            "https://evilmyanimelist.net/poster.jpg",
            allowed_hosts=image_proxy.DEFAULT_IMAGE_HOSTS,
        )
