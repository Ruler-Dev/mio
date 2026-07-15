"""Mio-owned persistent state must follow the MIO_HOME boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_persistent_subsystems_share_isolated_mio_home(tmp_path):
    state = tmp_path / "state"
    fake_user_home = tmp_path / "user-home"
    fake_user_home.mkdir()
    env = os.environ.copy()
    env["MIO_HOME"] = str(state)
    env["HOME"] = str(fake_user_home)

    program = textwrap.dedent(
        """
        import os
        from pathlib import Path

        from mio.cache_store import CacheStore
        from mio.webui import flow_runner, flow_skills, skills_life
        from mio.webui import skills_productivity as productivity
        from mio.webui import skills_rag

        root = Path(os.environ["MIO_HOME"]).resolve()

        CacheStore()
        skills_rag.list_indexes()
        productivity.todo_add("isolated todo")
        productivity.habit_add("isolated habit")
        productivity.journal_append("isolated journal")
        skills_life.bookmark_save("https://example.test/mio-home")
        flow_runner._memory_save({"isolated": True})
        flow_skills._root()

        expected = (
            root / "cache",
            root / "rag.sqlite",
            root / "todos.sqlite",
            root / "habits.sqlite",
            root / "journal",
            root / "bookmarks.sqlite",
            root / "flow-memory.json",
            root / "flows",
        )
        assert all(path.exists() for path in expected), expected
        assert not (Path.home() / ".mio").exists()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
