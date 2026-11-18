"""Phase 2 with the strict age rule (65+) applied to the majority vote's result."""

import csv
import json
from types import SimpleNamespace

from adrd_review import age_check, phase2_age
from adrd_review.schemas import Article
from adrd_review.tools.fulltext import FullTextResult
from adrd_review.tools.pmc import FullText, Section

TEXTS = {"1": "Participants had a mean age of 74 years.", "2": "The cohort had a mean age of 58 years.",
         "3": "Adults were recruited from memory clinics.", "4": "Participants had a mean age of 71 years.",
         "5": "The cohort had a mean age of 52 years."}


class AgeStub:
    """Quotes the mean age from the paper when it states one, like the real extraction model."""

    def structured(self, schema, system, user, role):
        import re

        m = re.search(r"mean age of (\d+) years", user)
        ans = age_check.AgeAnswer(mean_age=age_check.AgeFact(value=float(m.group(1)), quote=m.group(0)) if m else
                                  age_check.AgeFact())
        return SimpleNamespace(parsed=ans, usage=SimpleNamespace(model="stub"))


def _write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_the_age_rule_excludes_under_65_and_sends_unclear_age_to_a_person(tmp_path, monkeypatch):
    outcomes = {"1": "included", "2": "included", "3": "included", "4": "human review", "5": "human review",
                "6": "excluded"}
    _write(tmp_path / "final.csv", [{"pmid": p, "title": f"t{p}", "outcome": o,
                                     "stage": "title" if o == "excluded" else "full text", "reason": "",
                                     "reviewer A": "", "reviewer B": "", "reviewer C": ""} for p, o in outcomes.items()])
    # paper 4 is open only on age (unclear on I4); paper 5 also has I2 open
    reasons = {"4": "I4 Older adults (65 or over)", "5": "I2 Uses AI or machine learning, I4 Older adults (65 or over)"}
    for r in "ABC":
        _write(tmp_path / f"reviewer_{r}.csv", [{"pmid": p, "verdict": "unclear" if p in reasons and r == "A" else "included",
                                                 "reason": reasons.get(p, "")} for p in outcomes])
    (tmp_path / "cfg.yaml").write_text(json.dumps({"name": "t", "codebook": "configs/codebook_aiml_v0.4.yaml",
                                                   "pmids_file": "x", "reviewers": []}))
    arts = {p: Article(pmid=p, title=f"t{p}", abstract="a") for p in outcomes}
    monkeypatch.setattr("adrd_review.tools.pubmed.fetch_articles", lambda pmids, cache_dir: (arts, {}))

    def source(a):
        return FullTextResult(FullText(pmid=a.pmid, pmcid="PMC", sections=[Section(title="Methods", text=TEXTS[a.pmid])]),
                              "pmc", "PMC")

    s = phase2_age.run(tmp_path, tmp_path / "cfg.yaml", concurrency=1, llm=AgeStub(), fulltext_source=source)
    new = {r["pmid"]: r for r in csv.DictReader((tmp_path / "final_age65.csv").open())}
    assert new["1"]["outcome"] == "included"  # mean 74
    assert (new["2"]["outcome"], new["2"]["reason"]) == ("excluded", "I4 Older adults (65 or over)")  # mean 58
    assert new["3"]["outcome"] == "human review"  # no quoted age: can no longer be confirmed 65+
    assert new["4"]["outcome"] == "included"  # only age was open, and it is met
    assert new["5"]["outcome"] == "excluded"  # under 65 excludes whatever else is open
    assert new["6"]["outcome"] == "excluded" and new["2"]["before age rule"] == "included"
    assert s["age_rule_changes"] == {"included -> excluded": 1, "included -> human review": 1,
                                     "human review -> included": 1, "human review -> excluded": 1}
    assert s["stage_table"]["full text"]["included"] == 2 and s["failed"] == 0
