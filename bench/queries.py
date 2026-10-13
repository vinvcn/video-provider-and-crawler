"""Query set: real catalog terms + designed queries, deterministically split.

Poles of every benchmark comparison: materials fixed, queries fixed. Both are
versioned artifacts with content hashes. See docs/bench-harness-design-2026-10-09.md §3.

- 90 real terms from catalog_queries: junk-filtered, token-matched against the
  materials (>= 2 docs), stratified over head/torso/tail match bands;
- 60 designed queries from a versioned data file (20 concept / 15 facet /
  15 filter / 10 composition, 10 of them Chinese probes);
- train/holdout split 70/30 stratified by (category, language), hash-ordered
  inside each group. The leaderboard ranks on holdout.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bench import util
from bench.materials import Materials
from store.embedding import tokenize

RULE_VERSION = "queries-v1"
DATA_PATH = Path(__file__).resolve().parent / "data" / "designed_queries.v1.json"

REAL_PER_BAND = 30
TRAIN_FRACTION = 0.7

# Terms made only of these tokens carry no retrievable intent.
STOPWORDS = frozenset(
    "the of and for with a an in on to is are at by or type what how best new http www com".split()
)
BANDS = {"head": 1000, "torso": 100, "tail": 2}
ALLOWED_HARD_KEYS = frozenset({"orientation", "duration_min", "duration_max", "min_width"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
FACET_VOCAB = frozenset(
    ("subject", "motion", "camera", "setting", "lighting", "time", "mood", "colour")
)
LANGS = frozenset({"en", "zh"})
DESIGNED_CATEGORY_COUNTS = {"concept": 20, "facet": 15, "filter": 15, "composition": 10}
DESIGNED_ZH_COUNT = 10


@dataclass(frozen=True)
class QueryRecord:
    """One benchmark query with its metadata and split assignment."""

    qid: str
    text: str
    source: str  # catalog | designed
    category: str  # real | concept | facet | filter | composition
    lang: str  # en | zh
    facets: tuple[str, ...] = ()
    hard: dict = field(default_factory=dict)
    difficulty: str = "medium"
    match_docs: int = 0
    split: str = "train"

    def to_json(self) -> dict:
        return {
            "qid": self.qid,
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "lang": self.lang,
            "facets": list(self.facets),
            "hard": dict(self.hard),
            "difficulty": self.difficulty,
            "match_docs": self.match_docs,
            "split": self.split,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> QueryRecord:
        return cls(
            qid=str(data["qid"]),
            text=str(data["text"]),
            source=str(data["source"]),
            category=str(data["category"]),
            lang=str(data["lang"]),
            facets=tuple(str(facet) for facet in (data.get("facets") or [])),
            hard=dict(data.get("hard") or {}),  # type: ignore[arg-type]
            difficulty=str(data.get("difficulty") or "medium"),
            match_docs=int(data.get("match_docs") or 0),
            split=str(data.get("split") or "train"),
        )


def make_qid(text: str) -> str:
    """Content-derived stable query id."""
    return "q" + util.hash_hex(text)[:12]


def is_junk(term: str) -> bool:
    """True when a catalog term carries no retrievable intent."""
    lowered = term.lower()
    if "http" in lowered or "www" in lowered:
        return True
    tokens = tokenize(term)
    if not 1 <= len(tokens) <= 6:
        return True
    if all(token.isdigit() for token in tokens):
        return True
    if not any(len(token) >= 2 and token.isalpha() for token in tokens):
        return True
    return all(token in STOPWORDS for token in tokens)


def match_proxy(tokens: Sequence[str], postings: Mapping[str, int]) -> int:
    """Lower bound of matching materials docs: max token posting size."""
    return max((postings.get(token, 0) for token in tokens), default=0)


def band_of(match: int) -> str | None:
    """head >= 1000, torso 100-999, tail 2-99; None when below the floor."""
    if match >= BANDS["head"]:
        return "head"
    if match >= BANDS["torso"]:
        return "torso"
    if match >= BANDS["tail"]:
        return "tail"
    return None


def detect_lang(text: str) -> str:
    """zh when the text contains CJK codepoints, en otherwise."""
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"


def _postings(materials: Materials) -> dict[str, int]:
    """Doc-frequency counts over title+tags tokens of the materials set."""
    counts: Counter[str] = Counter()
    for doc_id in materials.ids:
        row = materials.rows[doc_id]
        tokens = set(tokenize(row.title)) | set(tokenize(" ".join(row.tags)))
        counts.update(tokens)
    return dict(counts)


def _sample_real(terms: Sequence[str], postings: Mapping[str, int]) -> list[QueryRecord]:
    """Deterministic head/torso/tail sample of catalog terms."""
    by_band: dict[str, list[tuple[bytes, str]]] = {"head": [], "torso": [], "tail": []}
    for term in terms:
        if is_junk(term):
            continue
        tokens = tokenize(term)
        band = band_of(match_proxy(tokens, postings))
        if band is None:
            continue
        by_band[band].append((util.hash_key(term), term))
    records: list[QueryRecord] = []
    for band in ("head", "torso", "tail"):
        for _, term in sorted(by_band[band])[:REAL_PER_BAND]:
            records.append(
                QueryRecord(
                    qid=make_qid(term),
                    text=term,
                    source="catalog",
                    category="real",
                    lang=detect_lang(term),
                    facets=(),
                    match_docs=match_proxy(tokenize(term), postings),
                )
            )
    return records


def load_designed(path: Path = DATA_PATH) -> list[dict]:
    """Load + validate the designed-queries data file (fails loudly)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data["queries"]
    counts = Counter(entry["category"] for entry in queries)
    if dict(counts) != DESIGNED_CATEGORY_COUNTS:
        raise SystemExit(f"designed query category counts drifted: {dict(counts)}")
    zh = sum(1 for entry in queries if entry["lang"] == "zh")
    if zh != DESIGNED_ZH_COUNT:
        raise SystemExit(f"designed Chinese probe count drifted: {zh} != {DESIGNED_ZH_COUNT}")
    for entry in queries:
        unknown_facets = set(entry["facets"]) - FACET_VOCAB
        if unknown_facets:
            raise SystemExit(f"unknown facets {sorted(unknown_facets)} in {entry['text']!r}")
        unknown_hard = set(entry["hard"]) - ALLOWED_HARD_KEYS
        if unknown_hard:
            raise SystemExit(f"unknown hard keys {sorted(unknown_hard)} in {entry['text']!r}")
        if entry["difficulty"] not in DIFFICULTIES:
            raise SystemExit(f"unknown difficulty in {entry['text']!r}")
        if entry["lang"] not in LANGS or detect_lang(entry["text"]) != entry["lang"]:
            raise SystemExit(f"lang mismatch in {entry['text']!r}")
        if entry["category"] == "composition" and len(entry["facets"]) < 2:
            raise SystemExit(f"composition query needs >= 2 facets: {entry['text']!r}")
    return queries


def _assign_split(records: Sequence[QueryRecord]) -> list[QueryRecord]:
    """70/30 train/holdout, stratified by (category, lang), hash-ordered."""
    groups: dict[tuple[str, str], list[QueryRecord]] = {}
    for record in records:
        groups.setdefault((record.category, record.lang), []).append(record)
    out: list[QueryRecord] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: (util.hash_key(r.text), r.text))
        n_train = round(TRAIN_FRACTION * len(members))
        for index, record in enumerate(members):
            merged = {**record.to_json(), "split": "train" if index < n_train else "holdout"}
            out.append(QueryRecord.from_json(merged))
    return out


def build_queries(
    dsn: str | None,
    materials: Materials,
    version: str,
    root: Path,
    designed_path: Path = DATA_PATH,
) -> dict:
    """Build + freeze the query set; returns the manifest."""
    out_dir = root / "queries" / version
    queries_path = out_dir / "queries.jsonl"
    if queries_path.exists():
        raise SystemExit(f"queries version {version!r} already exists: {queries_path}")

    import psycopg

    from store.db import DEFAULT_DSN

    with psycopg.connect(dsn or DEFAULT_DSN) as conn:
        terms = [row[0] for row in conn.execute("SELECT term FROM catalog_queries").fetchall()]

    postings = _postings(materials)
    records = _sample_real(terms, postings)

    for entry in load_designed(designed_path):
        tokens = tokenize(entry["text"])
        records.append(
            QueryRecord(
                qid=make_qid(entry["text"]),
                text=entry["text"],
                source="designed",
                category=entry["category"],
                lang=entry["lang"],
                facets=tuple(entry["facets"]),
                hard=dict(entry["hard"]),
                difficulty=entry["difficulty"],
                match_docs=match_proxy(tokens, postings),
            )
        )

    qids = [record.qid for record in records]
    if len(set(qids)) != len(qids):
        raise SystemExit("duplicate query texts produced identical qids")
    records = sorted(_assign_split(records), key=lambda r: r.qid)

    manifest = {
        "version": version,
        "rule": RULE_VERSION,
        "built_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n": len(records),
        "counts": {
            "source": dict(Counter(r.source for r in records)),
            "category": dict(Counter(r.category for r in records)),
            "lang": dict(Counter(r.lang for r in records)),
            "split": dict(Counter(r.split for r in records)),
            "difficulty": dict(Counter(r.difficulty for r in records)),
            "real_band": dict(
                Counter(band_of(r.match_docs) or "none" for r in records if r.source == "catalog")
            ),
        },
        "materials_version": materials.version,
        "materials_hash": materials.content_hash,
        "designed_file": designed_path.name,
        "designed_file_hash": util.content_hash(load_designed(designed_path)),
        "content_hash": util.content_hash([record.to_json() for record in records]),
        "code": util.code_version(),
    }
    util.write_jsonl(queries_path, (record.to_json() for record in records))
    util.write_json(out_dir / "manifest.json", manifest)
    return manifest


class QuerySet:
    """In-memory view of a frozen query set (what runs and the judge see)."""

    def __init__(self, version: str, root: Path) -> None:
        self.version = version
        out_dir = root / "queries" / version
        self.manifest = util.load_json(out_dir / "manifest.json")
        self.records: dict[str, QueryRecord] = {
            record.qid: record
            for record in (
                QueryRecord.from_json(data) for data in util.read_jsonl(out_dir / "queries.jsonl")
            )
        }
        self.ordered = sorted(self.records)
        self.content_hash = str(self.manifest["content_hash"])

    def by_split(self, split: str) -> list[QueryRecord]:
        return [self.records[qid] for qid in self.ordered if self.records[qid].split == split]
