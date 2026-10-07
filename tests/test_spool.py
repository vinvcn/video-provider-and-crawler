"""Unit tests for spool naming conventions."""

from crawler.spool import term_slug


def test_term_slug_basic():
    assert term_slug("Panda Eating Bamboo!") == "panda-eating-bamboo"


def test_term_slug_collapses_runs_and_trims():
    assert term_slug("a  b__c--d") == "a-b-c-d"
    assert term_slug("--edge--") == "edge"


def test_term_slug_truncates():
    assert len(term_slug("x" * 200)) == 60


def test_term_slug_empty_fallback():
    assert term_slug("!!!") == "term"
