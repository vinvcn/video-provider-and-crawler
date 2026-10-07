"""Unit tests for the jsonb preparation helper (pure logic, no I/O)."""

from psycopg.types.json import Jsonb

from store.db import _wrap_jsonb


def test_wraps_jsonb_bound_values():
    row = _wrap_jsonb({"video_files": [{"link": "x"}], "raw": {"id": 1}, "title": "t"})
    assert isinstance(row["video_files"], Jsonb)
    assert isinstance(row["raw"], Jsonb)
    assert row["video_files"].obj == [{"link": "x"}]
    assert row["title"] == "t"


def test_leaves_none_untouched():
    row = _wrap_jsonb({"video_files": None, "raw": None})
    assert row["video_files"] is None
    assert row["raw"] is None


def test_does_not_mutate_input():
    original = {"raw": {"a": 1}}
    _wrap_jsonb(original)
    assert isinstance(original["raw"], dict)
