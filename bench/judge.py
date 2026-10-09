"""Judge: pool candidates from runs, grade 0-3 with an LLM, cache labels.

Protocol (docs/bench-harness-design-2026-10-09.md §5):
- pool  = union of each run's top-`depth` per query + `negatives` deterministic
          random materials docs (calibration guard + future hard negatives);
- grade = one independent LLM call per (query, doc) pair — the judge sees only
          the query text and the document text, never the strategy or the
          other candidates (no position bias);
- cache = append-only labels.jsonl keyed by (query_text, doc_id, doc_hash,
          judge_model, prompt_version); interrupted runs resume for free;
- mock  = deterministic hash grading for pipeline verification only; mock
          labels never mix into a real judge version's metrics.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from bench import util
from bench.materials import Materials
from bench.queries import QuerySet
from bench.run import load_run

PROMPT_VERSION = "v1"
GRADE_RETRY_ATTEMPTS = 2  # extra attempts after an unparsable response

JUDGE_SYSTEM = """You are a relevance assessor for a stock video library.
Grade how well a video matches a creator's search query on a 0-3 scale:
3 = exactly what the creator wants, directly usable for this query
2 = relevant and usable, a good substitute
1 = marginally related, only loosely matches the intent
0 = not relevant
The query may be in a different language than the document; relevance is about
whether a creator searching this query would find this clip useful.
Judge only from the document text (title, description, tags) provided.
Respond with JSON only: {"grade": <0|1|2|3>, "reason": "<one short sentence>"}"""


def judge_version_for(model: str) -> str:
    """Version stamp tying labels to the judge model + prompt revision."""
    return f"{model}@prompt-{PROMPT_VERSION}"


def label_key(query_text: str, doc_id: int, doc_hash: str, judge_model: str) -> str:
    """Cache key for one judgment (prompt version folded in via model stamp)."""
    return util.hash_hex(query_text, doc_id, doc_hash, judge_model, PROMPT_VERSION)


def parse_grade(text: str) -> tuple[int | None, str]:
    """Extract {grade, reason} from a model response; (None, reason) if unparsable."""
    candidates: list[str] = []
    try:
        payload = json.loads(text)
        candidates.append(json.dumps(payload))
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    while start != -1:
        end = text.find("}", start)
        if end == -1:
            break
        candidates.append(text[start : end + 1])
        start = text.find("{", end)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        grade = payload.get("grade")
        if isinstance(grade, str) and grade.strip().isdigit():
            grade = int(grade.strip())
        reason = str(payload.get("reason", ""))[:500]
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3:
            continue
        return grade, reason
    return None, text[:200]


class JudgeClient:
    """OpenAI-compatible chat judge (temperature 0)."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0),
            trust_env=True,
        )

    def grade(self, query_text: str, doc_text: str) -> str:
        """Raw model response text for one (query, document) pair."""
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Query: {query_text}\n\nDocument: {doc_text}",
                    },
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    def close(self) -> None:
        self._client.close()


class LabelStore:
    """Append-only judgment cache with an in-memory existing-key index."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._keys: set[str] = set()
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            self._keys.add(str(json.loads(line)["key"]))
                        except json.JSONDecodeError:
                            continue

    def has(self, key: str) -> bool:
        return key in self._keys

    def append(self, record: dict) -> None:
        """Write one label (thread-safe); duplicate keys are refused."""
        with self._lock:
            if record["key"] in self._keys:
                return
            self._keys.add(str(record["key"]))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(util.canonical_json(record) + "\n")


def load_labels(path: Path, judge_version: str) -> dict[str, dict[int, int | None]]:
    """{qid: {doc_id: grade}} for one judge version (unparsed kept as None)."""
    grades: dict[str, dict[int, int | None]] = {}
    if not path.is_file():
        return grades
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if judge_version_for(str(record["judge_model"])) != judge_version:
                continue
            grades.setdefault(str(record["qid"]), {})[int(record["doc_id"])] = record.get("grade")
    return grades


def label_records(path: Path, judge_version: str) -> list[dict]:
    """All label records for one judge version (calibration input)."""
    out = []
    if path.is_file():
        for record in util.read_jsonl(path):
            if judge_version_for(str(record["judge_model"])) == judge_version:
                out.append(record)
    return out


def pool_for_query(
    run_rows: Sequence[Mapping[str, object]],
    qid: str,
    all_doc_ids: Sequence[int],
    materials_version: str,
    depth: int,
    negatives: int,
) -> list[int]:
    """Union of top-`depth` hits across runs + deterministic random negatives."""
    pooled: set[int] = set()
    for row in run_rows:
        hits = (row.get("hits") or [])[:depth]  # type: ignore[union-attr]
        pooled.update(int(doc_id) for doc_id, _ in hits)
    rng = random.Random(int.from_bytes(util.hash_key(qid, materials_version), "big"))
    candidates = [doc_id for doc_id in all_doc_ids if doc_id not in pooled]
    negatives_picked = sorted(rng.sample(candidates, min(negatives, len(candidates))))
    return sorted(pooled) + negatives_picked


@dataclass
class JudgeOutcome:
    """What a judging pass did (for CLI reporting)."""

    judged: int
    cached: int
    unparsed: int
    judge_version: str
    pool_size: int


def judge_runs(
    run_dirs: Sequence[Path],
    materials: Materials,
    queries: QuerySet,
    root: Path,
    depth: int,
    negatives: int,
    client: JudgeClient | None,
    concurrency: int,
    max_pairs: int | None,
) -> JudgeOutcome:
    """Pool + grade every (query, doc) pair for the given runs.

    `client=None` selects mock grading (deterministic hash, pipeline testing).
    All runs must share the materials/queries versions of the passed objects.
    """
    run_rows: dict[str, list[dict]] = {}
    versions = set()
    for run_dir in run_dirs:
        run = load_run(run_dir)
        versions.add((run["manifest"]["materials"]["hash"], run["manifest"]["queries"]["hash"]))
        for qid, row in run["rows"].items():
            run_rows.setdefault(qid, []).append(row)
    if len(versions) != 1:
        raise SystemExit("judge: runs mix materials/queries versions; pool them per version")
    if versions != {(materials.content_hash, queries.content_hash)}:
        raise SystemExit("judge: runs do not match the loaded materials/queries versions")

    judge_model = client.model if client else "mock"
    judge_version = judge_version_for(judge_model)
    store = LabelStore(root / "judgments" / "labels.jsonl")

    pool: dict[str, list[int]] = {
        qid: pool_for_query(
            run_rows.get(qid, []), qid, materials.ids, materials.version, depth, negatives
        )
        for qid in queries.ordered
    }
    missing: list[tuple[str, int, str]] = []
    for qid in queries.ordered:
        record = queries.records[qid]
        for doc_id in pool[qid]:
            row = materials.rows[doc_id]
            key = label_key(record.text, doc_id, row.doc_hash, judge_model)
            if not store.has(key):
                missing.append((qid, doc_id, key))
    pending = missing[:max_pairs] if max_pairs is not None else missing

    def judge_one(item: tuple[str, int, str]) -> dict:
        qid, doc_id, key = item
        record = queries.records[qid]
        row = materials.rows[doc_id]
        status = "ok"
        grade: int | None = None
        reason = ""
        if client is None:
            grade = int(util.hash_key("mock", key)[0]) % 4
            reason = "mock grade (pipeline verification)"
            status = "mock"
        else:
            for attempt in range(GRADE_RETRY_ATTEMPTS + 1):
                try:
                    raw = client.grade(record.text, row.embed_text)
                except httpx.HTTPError as exc:
                    if attempt == GRADE_RETRY_ATTEMPTS:
                        raise SystemExit(f"judge endpoint failed repeatedly: {exc}") from exc
                    time.sleep(2**attempt)
                    continue
                grade, reason = parse_grade(raw)
                if grade is not None:
                    break
                if attempt == GRADE_RETRY_ATTEMPTS:
                    status = "unparsed"
        return {
            "key": key,
            "qid": qid,
            "query_text": record.text,
            "doc_id": doc_id,
            "doc_hash": row.doc_hash,
            "doc_text": row.embed_text,
            "judge_model": judge_model,
            "prompt_version": PROMPT_VERSION,
            "grade": grade,
            "reason": reason,
            "status": status,
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }

    judged = 0
    unparsed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            for record in executor.map(judge_one, pending):
                store.append(record)
                judged += 1
                if record["status"] == "unparsed":
                    unparsed += 1
    pool_size = sum(len(docs) for docs in pool.values())
    return JudgeOutcome(
        judged=judged,
        cached=pool_size - len(missing),
        unparsed=unparsed,
        judge_version=judge_version,
        pool_size=pool_size,
    )
