"""Drive the `browser-use` CLI: feed the in-page fetch script its spec via env."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve().parent / "browser_scripts" / "fetch_batch.py"


def run_fetch(spec: dict[str, Any], timeout: float | None = None) -> int:
    """Execute fetch_batch.py under the browser-use harness; return its exit code."""
    env = os.environ.copy()
    env["VPC_FETCH_SPEC"] = json.dumps(spec)
    proc = subprocess.run(
        ["browser-use"],
        input=SCRIPT_PATH.read_text(encoding="utf-8"),
        text=True,
        env=env,
        timeout=timeout,
    )
    return proc.returncode
