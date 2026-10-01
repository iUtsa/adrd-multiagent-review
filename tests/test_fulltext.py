"""PubMed Central full text: parsing and when the pipeline uses it."""

from pathlib import Path

from adrd_review.agents.common import check_quotes, render_article
from adrd_review.codebook import load_codebook
from adrd_review.fake_model import default_responders
from adrd_review.graph.pipeline import PipelineOptions, Reviewer, build_pipeline
from adrd_review.llm import FakeLLM, ModelConfig
from adrd_review.schemas import Article, EvidenceSpan
from adrd_review.tools.pmc import FullText, Section, excerpt, parse_jats

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = load_codebook(ROOT / "configs/codebook.yaml")
XML = (Path(__file__).parent / "fixtures" / "pmc_sample.xml").read_text()
FT = FullText(pmid="9", pmcid="PMC1", sections=[Section(title="Methods", text="We recruited 412 adults from general practice.")])


def test_parse_keeps_body_sections_and_skips_back_matter():
    sections, license_ = parse_jats(XML)
    assert [s.title for s in sections] == ["Methods", "Methods / Reference standard", "Results"]
    assert sections[0].text == "We recruited 412 adults aged 65 and over from general practice."
    assert license_ == "https://creativecommons.org/licenses/by/4.0/"


def test_excerpt_caps_length_at_a_sentence():
    long = FullText(pmid="1", pmcid="PMC1", sections=[Section(title="", text="A sentence here. " * 100)])
    out = excerpt(long, 200)
    assert len(out) <= 210 and out.endswith("[...]")
    assert excerpt(FT, 10_000) == "METHODS: We recruited 412 adults from general practice."


def test_fulltext_is_shown_to_agents_and_quotes_can_come_from_it():
    art = Article(pmid="9", title="T", abstract="Short.", fulltext="METHODS: We recruited 412 adults.")
    assert "Full text of the paper:" in render_article(art)
    assert check_quotes([EvidenceSpan(quote="We recruited 412 adults.", source="fulltext")], art) == []


def graph(mode, fetcher):
    llm = FakeLLM(ModelConfig(provider="fake", model="keyword-rules"), default_responders(CODEBOOK))
    opts = PipelineOptions(fulltext=mode, fulltext_fetcher=fetcher)
    return build_pipeline(CODEBOOK, llm, llm, [Reviewer("A", llm, llm)], opts)


def test_when_needed_fetches_only_for_thin_abstracts_with_a_pmcid():
    calls = []

    def fetcher(a):
        calls.append(a.pmid)
        return FT

    thin = Article(pmid="9", title="MMSE letter", pmcid="PMC1")  # no abstract -> triage: not enough
    s = graph("when_needed", fetcher).invoke({"article": thin})
    assert calls == ["9"] and s["fulltext_info"]["used"] and "412 adults" in s["article"].fulltext

    full = Article(pmid="8", title="T", abstract="A complete abstract.", pmcid="PMC2")
    s = graph("when_needed", fetcher).invoke({"article": full})
    assert calls == ["9"] and s["fulltext_info"] == {"used": False, "reason": "not needed"}

    s = graph("when_needed", fetcher).invoke({"article": Article(pmid="7", title="No PMC")})
    assert s["fulltext_info"] == {"used": False, "reason": "no PMCID"}


def test_never_and_fetch_errors_do_not_fail_the_article():
    s = graph("never", lambda a: FT).invoke({"article": Article(pmid="9", title="T", pmcid="PMC1")})
    assert s["fulltext_info"]["reason"] == "off" and s["article"].fulltext is None

    def broken(a):
        raise ConnectionError("PMC down")

    s = graph("always", broken).invoke({"article": Article(pmid="9", title="T", pmcid="PMC1")})
    assert s["fulltext_info"]["reason"] == "fetch failed: ConnectionError"
    assert s["screening_decision"].labels  # the article still got a decision
