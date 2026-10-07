"""Thumbnail downloader: fetch pending images and record local paths."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import psycopg

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
)


def download_pending(
    conn: psycopg.Connection,
    out_dir: Path,
    limit: int = 2000,
    concurrency: int = 8,
    use_proxy: bool = True,
) -> dict[str, int]:
    """Download up to `limit` thumbnails that have no local path yet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        """
        SELECT pexels_id, thumbnail_url FROM stock_videos
        WHERE thumbnail_path IS NULL AND thumbnail_url IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return {"downloaded": 0, "failed": 0}

    downloaded: list[tuple[str, int]] = []
    failed = 0
    lock = threading.Lock()

    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": _UA},
        trust_env=use_proxy,
    ) as client:

        def fetch_one(pid: int, url: str) -> tuple[int, str | None]:
            target = out_dir / f"{pid}.jpg"
            if target.exists() and target.stat().st_size > 0:
                return pid, str(target)
            try:
                resp = client.get(url)
                resp.raise_for_status()
                tmp = target.with_suffix(".jpg.tmp")
                tmp.write_bytes(resp.content)
                tmp.replace(target)
                return pid, str(target)
            except Exception:
                return pid, None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(fetch_one, pid, url) for pid, url in rows]
            for future in as_completed(futures):
                pid, path = future.result()
                with lock:
                    if path:
                        downloaded.append((path, pid))
                    else:
                        failed += 1

    if downloaded:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE stock_videos SET thumbnail_path = %s WHERE pexels_id = %s",
                downloaded,
            )
        conn.commit()
    return {"downloaded": len(downloaded), "failed": failed}
