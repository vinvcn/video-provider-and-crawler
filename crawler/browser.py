"""Drive the `browser-use` CLI: feed the in-page fetch script its spec via env."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve().parent / "browser_scripts" / "fetch_batch.py"


def _progress(state_file: str) -> int:
    """Cheap progress probe over a run's state file (search keys or seed pages)."""
    try:
        data = json.loads(Path(state_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return -1
    done = data.get("done_keys")
    if isinstance(done, list):
        return len(done)
    try:
        return int(data.get("seed_pages", 0) or 0)
    except (TypeError, ValueError):
        return -1


def _run_once(spec: dict[str, Any], timeout: float | None) -> int:
    """One `browser-use` invocation. The spec goes through a temp file, not an
    environment variable: large specs (thousands of search targets) exceed the
    per-env-var size limit (~128KB)."""
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


def run_fetch(spec: dict[str, Any], timeout: float | None = None, attempts: int = 1) -> int:
    """Execute fetch_batch.py under the browser-use harness; return its exit code.

    With `attempts` > 1 every aborted run is retried with backoff: a fresh
    harness process re-establishes a dropped CDP connection, and the in-page
    script resumes from the state file (completed work is skipped).
    """
    state_file = str(spec.get("state_file") or "")
    progress = _progress(state_file)
    total = max(1, int(attempts))
    result = 1
    for attempt in range(1, total + 1):
        result = _run_once(spec, timeout)
        if result == 0:
            return 0
        current = _progress(state_file)
        print(f"[vpc] attempt {attempt}/{total} exited {result}; progress {progress} -> {current}")
        progress = max(progress, current)
        if attempt < total:
            time.sleep(min(5.0 * attempt, 30.0))
    print("[vpc] attempts exhausted")
    return result
