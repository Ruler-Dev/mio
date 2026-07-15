"""Concurrency and corruption regressions for shared Web UI JSON stores."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Callable

import pytest
from fastapi import HTTPException

from mio.webui import router, scheduler, webhooks


StoreOperation = tuple[
    Path,
    Callable[[int], object],
    Callable[[], object],
    Callable[[], list[dict]],
]


def _configure_store(name: str, path: Path, monkeypatch) -> StoreOperation:
    if name == "prompts":
        monkeypatch.setattr(router, "_prompts_path", lambda: path)

        def add(index: int) -> object:
            return asyncio.run(
                router.save_prompt(
                    {"id": f"prompt-{index}", "name": f"Prompt {index}"}
                )
            )

        return (
            path,
            add,
            lambda: asyncio.run(router.delete_prompt("missing")),
            router._load_prompts,
        )

    if name == "memory":
        monkeypatch.setattr(router, "_memory_path", lambda: path)

        def add(index: int) -> object:
            return asyncio.run(
                router.add_memory({"id": f"memory-{index}", "text": str(index)})
            )

        return (
            path,
            add,
            lambda: asyncio.run(router.delete_memory("missing")),
            router._load_memory,
        )

    if name == "projects":
        monkeypatch.setattr(router, "_projects_path", lambda: path)

        def add(index: int) -> object:
            return asyncio.run(
                router.save_project({"id": f"project-{index}", "name": str(index)})
            )

        return (
            path,
            add,
            lambda: asyncio.run(router.delete_project("missing")),
            router._load_projects,
        )

    if name == "schedules":
        monkeypatch.setattr(scheduler, "_SCHED_FILE", path)

        def add(index: int) -> object:
            return scheduler.create_schedule(
                f"Schedule {index}",
                str(index),
                {"kind": "interval", "every_seconds": 60},
            )

        return path, add, lambda: scheduler.delete_schedule("missing"), scheduler.load_schedules

    if name == "webhooks":
        monkeypatch.setattr(webhooks, "_WEBHOOKS_FILE", path)

        def add(index: int) -> object:
            return webhooks.create_webhook(
                f"hook-{index}",
                f"Run {index}",
                secret=f"concurrency-secret-{index:04d}",
            )

        return path, add, lambda: webhooks.delete_webhook("missing"), webhooks.load_webhooks

    raise AssertionError(f"unknown store {name}")


@pytest.mark.parametrize(
    "store_name",
    ["prompts", "memory", "projects", "schedules", "webhooks"],
)
def test_store_concurrent_creates_do_not_lose_updates(
    store_name,
    tmp_path,
    monkeypatch,
):
    _path, add, _delete_missing, load = _configure_store(
        store_name,
        tmp_path / f"{store_name}.json",
        monkeypatch,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(24)))

    records = load()
    identity = "slug" if store_name == "webhooks" else "id"
    assert len(records) == 24
    assert len({record[identity] for record in records}) == 24


@pytest.mark.parametrize(
    "store_name",
    ["prompts", "memory", "projects", "schedules", "webhooks"],
)
@pytest.mark.parametrize(
    "corrupt_payload, error_type",
    [
        (b'{"truncated"', json.JSONDecodeError),
        (b'{"valid_json":"wrong_schema"}\n', ValueError),
    ],
)
def test_store_mutations_fail_closed_without_replacing_corrupt_state(
    store_name,
    corrupt_payload,
    error_type,
    tmp_path,
    monkeypatch,
):
    path, add, delete_missing, load = _configure_store(
        store_name,
        tmp_path / f"{store_name}.json",
        monkeypatch,
    )
    path.write_bytes(corrupt_payload)

    with pytest.raises(error_type):
        load()
    mutation_error = HTTPException if store_name in {"prompts", "memory", "projects"} else error_type
    with pytest.raises(mutation_error) as add_error:
        add(99)
    with pytest.raises(mutation_error) as delete_error:
        delete_missing()
    if mutation_error is HTTPException:
        assert add_error.value.status_code == 409
        assert delete_error.value.status_code == 409

    assert path.read_bytes() == corrupt_payload


@pytest.mark.parametrize(
    "store_name",
    ["prompts", "memory", "projects", "schedules", "webhooks"],
)
@pytest.mark.parametrize(
    "corrupt_payload",
    [b'{"truncated"', b'{"valid_json":"wrong_schema"}\n'],
)
def test_store_http_endpoints_report_conflict_without_data_loss(
    store_name,
    corrupt_payload,
    tmp_path,
    monkeypatch,
):
    path = tmp_path / f"{store_name}.json"
    path.write_bytes(corrupt_payload)
    if store_name == "prompts":
        monkeypatch.setattr(router, "_prompts_path", lambda: path)
        call = router.list_prompts
    elif store_name == "memory":
        monkeypatch.setattr(router, "_memory_path", lambda: path)
        call = router.list_memory
    elif store_name == "projects":
        monkeypatch.setattr(router, "_projects_path", lambda: path)
        call = router.list_projects
    elif store_name == "schedules":
        monkeypatch.setattr(scheduler, "_SCHED_FILE", path)
        call = router.schedules_list
    else:
        monkeypatch.setattr(webhooks, "_WEBHOOKS_FILE", path)
        call = router.webhooks_list

    with pytest.raises(HTTPException) as raised:
        asyncio.run(call())

    assert raised.value.status_code == 409
    assert "no changes were written" in str(raised.value.detail)
    assert path.read_bytes() == corrupt_payload
