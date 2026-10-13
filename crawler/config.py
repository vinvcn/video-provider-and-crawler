"""Runtime paths and environment-backed settings for the crawler."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from crawler.pexels_v3 import PEXELS_PUBLIC_SECRET

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing vars win).

    The Makefile already treats .env as the machine-local config surface;
    this makes plain `vpc` invocations (embed/search/bench-adjacent) see the
    same VPC_EMBED_*/VPC_JUDGE_* values the bench subcommands load.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one CLI invocation."""

    db_dsn: str
    spool_dir: Path
    thumbnail_dir: Path
    state_dir: Path
    use_proxy: bool
    pexels_secret: str


def load_settings() -> Settings:
    """Read settings from VPC_* environment variables (sane local defaults)."""
    from store.db import DEFAULT_DSN

    _load_env_file()
    return Settings(
        db_dsn=os.environ.get("VPC_DB_DSN", DEFAULT_DSN),
        spool_dir=Path(os.environ.get("VPC_SPOOL_DIR", REPO_ROOT / "storage" / "spool")),
        thumbnail_dir=Path(
            os.environ.get("VPC_THUMBNAIL_DIR", REPO_ROOT / "storage" / "thumbnails")
        ),
        state_dir=Path(os.environ.get("VPC_STATE_DIR", REPO_ROOT / "storage" / "state")),
        use_proxy=os.environ.get("VPC_USE_PROXY", "1") == "1",
        pexels_secret=os.environ.get("VPC_PEXELS_SECRET", PEXELS_PUBLIC_SECRET),
    )
