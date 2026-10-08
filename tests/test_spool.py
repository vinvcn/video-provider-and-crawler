"""Unit tests for spool naming conventions and record shapes."""

import os
from pathlib import Path

from crawler.spool import (
    iter_records,
    load_ingest_marks,
    record_attributes,
    save_ingest_marks,
    term_slug,
    write_record,
)


def test_record_attributes_reads_list_payloads():
    record = {
        "body": {
            "data": [
                {"attributes": {"id": 1, "slug": "a"}},
                {"attributes": {"id": 2}},
                {"id": 3},
                "junk",
            ]
        }
    }
    assert [attrs["id"] for attrs in record_attributes(record)] == [1, 2]


def test_record_attributes_reads_per_id_payloads():
    record = {"attributes": {"id": 40083338, "slug": "wildebeest-migration-across-river"}}
    assert record_attributes(record) == [
        {"id": 40083338, "slug": "wildebeest-migration-across-river"}
    ]


def test_record_attributes_ignores_failures_and_unknown_shapes():
    assert record_attributes({"status": 0, "error": "boom"}) == []
    assert record_attributes({"attributes": {"slug": "no-id"}}) == []
    assert record_attributes({"body": {"data": "nope"}}) == []


def test_term_slug_basic():
    assert term_slug("Panda Eating Bamboo!") == "panda-eating-bamboo"


def test_term_slug_collapses_runs_and_trims():
    assert term_slug("a  b__c--d") == "a-b-c-d"
    assert term_slug("--edge--") == "edge"


def test_term_slug_truncates():
    assert len(term_slug("x" * 200)) == 60


def test_term_slug_empty_fallback():
    assert term_slug("!!!") == "term"


def test_iter_records_replays_everything_without_floors(tmp_path, monkeypatch):
    monkeypatch.setenv("VPC_SPOOL_DIR", str(tmp_path / "spool"))
    write_record("search", "a", "u", {"body": {"data": []}})
    write_record("search", "b", "u", {"body": {"data": []}})
    keys = [Path(record["_path"]).stem for record in iter_records("search")]
    assert keys == ["a", "b"]


def test_iter_records_skips_files_at_or_below_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("VPC_SPOOL_DIR", str(tmp_path / "spool"))
    old = write_record("search", "a-old", "u", {"body": {"data": []}})
    new = write_record("search", "b-new", "u", {"body": {"data": []}})
    os.utime(old, ns=(1_000, 1_000))
    os.utime(new, ns=(2_000, 2_000))

    keys = [record["_path"] for record in iter_records("search", since_ns={"search": 1_000})]
    assert [Path(key).stem for key in keys] == ["b-new"]

    # A floor recorded for another kind must not hide this kind's records.
    keys = [record["_path"] for record in iter_records("search", since_ns={"seed": 5_000})]
    assert [Path(key).stem for key in keys] == ["a-old", "b-new"]


def test_iter_records_unknown_kind_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("VPC_SPOOL_DIR", str(tmp_path / "spool"))
    assert list(iter_records("nope")) == []


def test_ingest_marks_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VPC_STATE_DIR", str(tmp_path / "state"))
    assert load_ingest_marks() == {}
    save_ingest_marks({"search": 123, "ids": 4})
    assert load_ingest_marks() == {"search": 123, "ids": 4}


def test_ingest_marks_ignore_corrupt_state(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("VPC_STATE_DIR", str(state))
    (state / "ingest.json").write_text("{not json", encoding="utf-8")
    assert load_ingest_marks() == {}
