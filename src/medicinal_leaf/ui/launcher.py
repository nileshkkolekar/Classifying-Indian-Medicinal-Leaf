"""Console-script entry point for the Streamlit UI.

``streamlit run`` needs a filesystem path to a script, which is awkward once
the package is installed somewhere like ``site-packages``. This resolves that
path from the module itself, so ``leaf-ui`` works identically in a checkout
and in a container.

Invoked as a subprocess rather than through ``streamlit.web.cli``: that
module is Streamlit's internal CLI and its shape changes between releases,
whereas the command line is the stable, documented interface.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from medicinal_leaf.config.settings import load_settings

#: Streamlit's own default; the API port comes from configuration.
DEFAULT_UI_PORT = 8501


def app_path() -> Path:
    """Filesystem location of the Streamlit script."""
    return Path(__file__).with_name("streamlit_app.py")


def build_command(host: str, port: int) -> list[str]:
    """The argv Streamlit is launched with."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path()),
        "--server.address",
        host,
        "--server.port",
        str(port),
        # Containers have no browser to open, and the prompt blocks startup.
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def main() -> None:
    """Launch the UI, inheriting host and port from configuration."""
    settings = load_settings()
    command = build_command(settings.serving.host, DEFAULT_UI_PORT)
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
