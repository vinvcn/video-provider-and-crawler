"""Unit tests for spool naming conventions and record shapes."""

from crawler.spool import record_attributes, term_slug


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
