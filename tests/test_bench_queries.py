"""Query set: junk filter, match bands, designed-file validation, split."""

import pytest

from bench import queries as q
from bench.queries import (
    QueryRecord,
    _assign_split,
    band_of,
    detect_lang,
    is_junk,
    load_designed,
    make_qid,
    match_proxy,
)


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("type of", True),
        ("best", True),
        ("12345", True),
        ("", True),
        ("the of and", True),
        ("http://pexels.com/sea", True),
        ("ocean waves", False),
        ("hd wallpaper for android", False),
        ("vancouver beach", False),
        ("drone", False),
    ],
)
def test_is_junk(term, expected):
    assert is_junk(term) is expected


def test_band_boundaries():
    assert band_of(0) is None
    assert band_of(1) is None
    assert band_of(2) == "tail"
    assert band_of(99) == "tail"
    assert band_of(100) == "torso"
    assert band_of(999) == "torso"
    assert band_of(1000) == "head"


def test_match_proxy_takes_rarest_token_max():
    postings = {"ocean": 500, "waves": 3, "drone": 7}
    assert match_proxy(["ocean", "waves"], postings) == 500
    assert match_proxy(["waves"], postings) == 3
    assert match_proxy(["zebra"], postings) == 0


def test_detect_lang():
    assert detect_lang("雨夜城市") == "zh"
    assert detect_lang("ocean waves") == "en"


def test_make_qid_stable_and_distinct():
    assert make_qid("ocean waves") == make_qid("ocean waves")
    assert make_qid("ocean waves") != make_qid("ocean wave")


def test_designed_file_is_valid():
    entries = load_designed()
    assert len(entries) == 60
    categories = {}
    for entry in entries:
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1
    assert categories == {"concept": 20, "facet": 15, "filter": 15, "composition": 10}
    assert sum(1 for entry in entries if entry["lang"] == "zh") == 10


def test_assign_split_is_stratified_and_deterministic():
    records = [
        QueryRecord(
            qid=make_qid(f"text-{i}"),
            text=f"text-{i}",
            source="designed",
            category="concept",
            lang="en",
        )
        for i in range(10)
    ] + [
        QueryRecord(
            qid=make_qid(f"雨-{i}"),
            text=f"雨-{i}",
            source="designed",
            category="concept",
            lang="zh",
        )
        for i in range(4)
    ]
    assigned = _assign_split(records)
    assert len(assigned) == 14

    en = [r for r in assigned if r.lang == "en"]
    zh = [r for r in assigned if r.lang == "zh"]
    assert sum(1 for r in en if r.split == "holdout") == 3  # round(0.3 * 10)
    assert sum(1 for r in zh if r.split == "holdout") == 1  # round(0.3 * 4)

    again = _assign_split(records)
    assert {r.qid: r.split for r in again} == {r.qid: r.split for r in assigned}


def test_assign_split_covers_every_record():
    records = [
        QueryRecord(
            qid=make_qid(f"t{i}"), text=f"t{i}", source="catalog", category="real", lang="en"
        )
        for i in range(7)
    ]
    assigned = _assign_split(records)
    assert sorted(r.qid for r in assigned) == sorted(r.qid for r in records)


def test_stopword_list_matches_design():
    assert "type" in q.STOPWORDS and "of" in q.STOPWORDS
