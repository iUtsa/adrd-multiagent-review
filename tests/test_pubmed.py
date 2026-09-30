import os
from pathlib import Path

import pytest

from adrd_review.tools import pubmed
from adrd_review.tools.pubmed import (
    fetch_articles,
    normalize_pmids,
    parse_pubmed_xml,
    read_pmids_file,
)

FIXTURE = (Path(__file__).parent / "fixtures" / "pubmed_sample.xml").read_text()


class FakeClient:
    """Stands in for PubMedClient: serves the fixture, counts calls."""

    def __init__(self):
        self.calls = []

    def efetch_xml(self, pmids):
        self.calls.append(list(pmids))
        return FIXTURE


# ---- parsing ---------------------------------------------------------------


def test_parses_articles_and_skips_books():
    articles = parse_pubmed_xml(FIXTURE)
    assert [a.pmid for a in articles] == ["11111111", "22222222"]


def test_structured_abstract_keeps_section_labels():
    a = parse_pubmed_xml(FIXTURE)[0]
    assert [s.label for s in a.abstract_sections] == ["BACKGROUND", "METHODS", "RESULTS"]
    assert a.abstract.startswith("BACKGROUND: Dementia is often undiagnosed.")
    assert "METHODS: We screened 500 adults aged 65+ with the MoCA (cut-off <26)." in a.abstract


def test_title_inline_tags_are_flattened():
    assert parse_pubmed_xml(FIXTURE)[0].title == "Screening with the MoCA in rural older adults."


def test_ids_come_from_the_article_not_its_references():
    a = parse_pubmed_xml(FIXTURE)[0]
    assert a.doi == "10.9999/test.doi.1"
    assert a.pmcid == "PMC1234567"


def test_doi_falls_back_to_elocation_id():
    assert parse_pubmed_xml(FIXTURE)[1].doi == "10.9999/letter.doi"


def test_metadata_fields():
    a = parse_pubmed_xml(FIXTURE)[0]
    assert a.journal == "Journal of Test Neurology"
    assert a.year == 2024
    assert a.authors == ["Rivera, Ana", "Okafor, C", "Test Dementia Consortium"]
    assert a.mesh_terms == ["Dementia", "Mass Screening"]
    assert a.keywords == ["MoCA", "rural health"]
    assert a.publication_types == ["Journal Article"]
    assert a.language == ["eng"]


def test_missing_abstract_and_medline_date():
    b = parse_pubmed_xml(FIXTURE)[1]
    assert b.abstract == "" and not b.has_abstract
    assert b.year == 1998


# ---- PMID cleaning ---------------------------------------------------------


def test_normalize_pmids_cleans_and_dedupes():
    raw = [" 111 ", "222.0", 333, "", "nan", "111", float("nan")]
    assert normalize_pmids(raw) == ["111", "222", "333"]


def test_normalize_pmids_rejects_non_numbers():
    with pytest.raises(ValueError, match="10.1000/abc"):
        normalize_pmids(["123", "10.1000/abc"])


def test_read_pmids_from_txt_and_csv(tmp_path):
    txt = tmp_path / "ids.txt"
    txt.write_text("111\n\n222\n")
    assert read_pmids_file(txt) == ["111", "222"]

    csv = tmp_path / "gold.csv"
    csv.write_text("Title,PMID\nA,111\nB,222\n")
    assert read_pmids_file(csv, column="pmid") == ["111", "222"]

    with pytest.raises(ValueError, match="No column named 'doi'"):
        read_pmids_file(csv, column="doi")


# ---- fetching + cache ------------------------------------------------------


def test_fetch_saves_cache_and_reports(tmp_path):
    client = FakeClient()
    articles, report = fetch_articles(
        ["11111111", "22222222", "33333333"], cache_dir=tmp_path, client=client
    )
    assert list(articles) == ["11111111", "22222222"]
    assert report.downloaded == ["11111111", "22222222"]
    assert report.missing == ["33333333"]  # book record: not a journal article
    assert report.no_abstract == ["22222222"]
    assert (tmp_path / "11111111.json").exists()


def test_second_fetch_uses_cache_without_network(tmp_path):
    fetch_articles(["11111111"], cache_dir=tmp_path, client=FakeClient())
    client = FakeClient()
    articles, report = fetch_articles(["11111111"], cache_dir=tmp_path, client=client)
    assert client.calls == []
    assert report.from_cache == ["11111111"]
    assert articles["11111111"].title.startswith("Screening with the MoCA")


def test_refresh_redownloads(tmp_path):
    fetch_articles(["11111111"], cache_dir=tmp_path, client=FakeClient())
    client = FakeClient()
    fetch_articles(["11111111"], cache_dir=tmp_path, client=client, refresh=True)
    assert client.calls == [["11111111"]]


def test_large_lists_are_split_into_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(pubmed, "BATCH_SIZE", 2)
    client = FakeClient()
    fetch_articles(["1", "2", "3", "4", "5"], cache_dir=tmp_path, client=client)
    assert client.calls == [["1", "2"], ["3", "4"], ["5"]]


# ---- live PubMed (opt-in) --------------------------------------------------


@pytest.mark.skipif(not os.getenv("RUN_NETWORK_TESTS"), reason="set RUN_NETWORK_TESTS=1")
def test_live_search_and_fetch(tmp_path):
    ids, count = pubmed.search('"Montreal Cognitive Assessment"[tiab] AND dementia[tiab]')
    assert count > 0 and ids
    articles, report = fetch_articles(ids[:3], cache_dir=tmp_path)
    assert not report.missing
    assert all(a.title for a in articles.values())
