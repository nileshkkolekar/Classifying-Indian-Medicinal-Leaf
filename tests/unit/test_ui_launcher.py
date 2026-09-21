"""The `leaf-ui` console script.

Nothing here launches Streamlit; the point is that the command it would run
is well formed and that the script path resolves from the installed package
rather than from the current working directory.
"""

from __future__ import annotations

import sys

from medicinal_leaf.ui.launcher import DEFAULT_UI_PORT, app_path, build_command


def test_app_path_points_at_a_real_file():
    path = app_path()
    assert path.is_file()
    assert path.name == "streamlit_app.py"


def test_command_runs_streamlit_through_this_interpreter():
    """Using sys.executable keeps the venv's Streamlit, not whatever is on PATH."""
    command = build_command("127.0.0.1", DEFAULT_UI_PORT)
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[4] == str(app_path())


def test_command_carries_address_and_port():
    command = build_command("0.0.0.0", 9000)
    assert "--server.address" in command
    assert command[command.index("--server.address") + 1] == "0.0.0.0"
    assert command[command.index("--server.port") + 1] == "9000"


def test_command_is_headless():
    """A container has no browser, and the prompt would block startup."""
    command = build_command("0.0.0.0", DEFAULT_UI_PORT)
    assert command[command.index("--server.headless") + 1] == "true"
    assert command[command.index("--browser.gatherUsageStats") + 1] == "false"


def test_command_raises_the_upload_ceiling():
    """Streamlit defaults to 200 MB, which would reject bulk archives."""
    command = build_command("0.0.0.0", DEFAULT_UI_PORT, max_upload_mb=1024)
    assert command[command.index("--server.maxUploadSize") + 1] == "1024"
