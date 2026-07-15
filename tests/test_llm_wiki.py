from __future__ import annotations

from mio.mcp.llm_wiki_server import WikiStore, handle_request, slugify, tool_definitions


def test_wiki_write_read_search_and_revision(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    first = store.write(
        title="Prefix Cache Research",
        content="A sufficiently long evidence note about prefix cache behavior on MLX inference workloads.",
        sources=["benchmark://run-001"],
        tags=["mlx", "prefill"],
    )
    assert first["slug"] == "prefix-cache-research"
    assert store.read(first["slug"])["content"].startswith("A sufficiently")
    assert store.search("MLX prefix")[0]["slug"] == first["slug"]

    second = store.write(
        title="Prefix Cache Research",
        content="Updated evidence that remains long enough for the wiki linter and its quality checks.",
        sources=["benchmark://run-002"],
    )
    assert second["revision"] == 2
    assert second["created_at"] == first["created_at"]


def test_wiki_ingest_accumulates_sources_and_lint_finds_broken_link(tmp_path):
    store = WikiStore(tmp_path)
    store.ingest(
        title="DFlash",
        content="Initial measured evidence about speculative decoding and [[missing-page]] on Apple Silicon.",
        source="benchmark://dflash-1",
        tags=["inference"],
    )
    page = store.ingest(
        title="DFlash",
        content="A second independent observation with enough detail to preserve accumulated research history.",
        source="benchmark://dflash-2",
    )
    assert page["revision"] == 2
    assert page["sources"] == ["benchmark://dflash-1", "benchmark://dflash-2"]
    lint = store.lint("dflash")
    assert any(issue["code"] == "broken-link" for issue in lint["issues"])


def test_wiki_paths_are_slugged_and_confined(tmp_path):
    store = WikiStore(tmp_path)
    page = store.write(title="../../Outside", content="safe local content", sources=["test"])
    assert page["slug"] == "outside"
    assert (tmp_path / "pages" / "outside.json").exists()
    assert not (tmp_path.parent / "Outside.json").exists()
    assert slugify("Caffè MLX") == "caffe-mlx"


def test_wiki_mcp_tools_and_error_shape(tmp_path):
    store = WikiStore(tmp_path)
    initialize = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, store)
    assert initialize["result"]["serverInfo"]["name"] == "mio-llm-wiki"
    names = {tool["name"] for tool in tool_definitions()}
    assert names == {
        "llm_wiki_list",
        "llm_wiki_search",
        "llm_wiki_read",
        "llm_wiki_write",
        "llm_wiki_ingest",
        "llm_wiki_lint",
    }
    result = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "llm_wiki_read", "arguments": {"slug": "absent"}},
        },
        store,
    )
    assert result["result"]["isError"] is True
