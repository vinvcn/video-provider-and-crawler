"""Bench settings: environment variables plus a minimal `.env` loader.

The repo Makefile already includes `.env` for make-level variables; the CLI
needs the values in-process, so we parse the same file. Existing environment
variables always win (the loader never overrides).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a `.env`-style file and return values not already in os.environ.

    Only `KEY=VALUE` lines are honoured; comments (`#`) and blank lines are
    skipped. Quotes around the value are stripped.
    """
    if not path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


@dataclass(frozen=True)
class JudgeConfig:
    """OpenAI-compatible chat endpoint used for relevance judging."""

    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class EmbedConfig:
    """OpenAI-compatible embeddings endpoint used by the dense arms."""

    base_url: str
    api_key: str
    model: str
    dim: int | None
    batch: int
    query_prefix: str


def judge_config() -> JudgeConfig:
    """Judge endpoint from VPC_JUDGE_* (errors list what is missing)."""
    missing = [
        name
        for name in (
            "VPC_JUDGE_BASE_URL",
            "VPC_JUDGE_API_KEY",
            "VPC_JUDGE_MODEL",
        )
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)} (set them in .env)")
    return JudgeConfig(
        base_url=os.environ["VPC_JUDGE_BASE_URL"].rstrip("/"),
        api_key=os.environ["VPC_JUDGE_API_KEY"],
        model=os.environ["VPC_JUDGE_MODEL"],
    )


def embed_config() -> EmbedConfig:
    """Embeddings endpoint from VPC_EMBED_* (errors list what is missing)."""
    missing = [
        name
        for name in ("VPC_EMBED_BASE_URL", "VPC_EMBED_API_KEY", "VPC_EMBED_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"missing environment variables: {', '.join(missing)} (set them in .env)")
    dim_raw = os.environ.get("VPC_EMBED_DIM", "").strip()
    return EmbedConfig(
        base_url=os.environ["VPC_EMBED_BASE_URL"].rstrip("/"),
        api_key=os.environ["VPC_EMBED_API_KEY"],
        model=os.environ["VPC_EMBED_MODEL"],
        dim=int(dim_raw) if dim_raw else None,
        batch=max(1, int(os.environ.get("VPC_EMBED_BATCH", "16"))),
        query_prefix=os.environ.get("VPC_EMBED_QUERY_PREFIX", ""),
    )


def bench_dir() -> Path:
    """Root directory for all bench artifacts (VPC_BENCH_DIR overrides)."""
    path = Path(os.environ.get("VPC_BENCH_DIR", REPO_ROOT / "storage" / "bench"))
    path.mkdir(parents=True, exist_ok=True)
    return path
