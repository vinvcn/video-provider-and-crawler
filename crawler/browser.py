"""Drive the `browser-use` CLI: feed the in-page fetch script its spec via env."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve().parent / "browser_scripts" / "fetch_batch.py"


def run_fetch(spec: dict[str, Any], timeout: float | None = None) -> int:
    """Execute fetch_batch.py under the browser-use harness; return its exit code.

    The spec goes through a temp file, not an environment variable: large specs
    (thousands of search targets) exceed the per-env-var size limit (~128KB).
    """
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="vpc-spec-", delete=False, encoding="utf-8"
    )
    path = handle.name
    try:
        json.dump(spec, handle)
        handle.close()
        env = os.environ.copy()
        env["VPC_FETCH_SPEC_FILE"] = path
        proc = subprocess.run(
            ["browser-use"],
            input=SCRIPT_PATH.read_text(encoding="utf-8"),
            text=True,
            env=env,
            timeout=timeout,
        )
        return proc.returncode
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
