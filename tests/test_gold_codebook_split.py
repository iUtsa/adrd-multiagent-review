from pathlib import Path

import pytest

from adrd_review.codebook import load_codebook
from adrd_review.gold import GoldError, GoldRecord, load_gold, summarize, write_gold
from adrd_review.split import load_split, make_split, save_split

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = load_codebook(ROOT / "configs/codebook.yaml")


def write_csv(tmp_path, text):
    p = tmp_path / "gold.csv"
    p.write_text(text)
    return p


# ---- codebook --------------------------------------------------------------


def test_codebook_loads_with_unique_rules_and_labels():
    ids = [r.id for r in CODEBOOK.rules]
    assert ids == ["I1", "I2", "I3", "I4", "E1", "E2", "E3"]
    assert set(CODEBOOK.dimensions) == {"instrument", "setting"}
    assert "MoCA" in CODEBOOK.dimensions["instrument"].label_ids
    assert len(CODEBOOK.sha256) == 64


def test_codebook_rejects_duplicate_rule_ids(tmp_path):
    bad = (ROOT / "configs/codebook.yaml").read_text().replace("id: I2", "id: I1")
    p = tmp_path / "cb.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="Duplicate rule IDs"):
        load_codebook(p)


# ---- gold loader -----------------------------------------------------------


def test_gold_loads_standard_format(tmp_path):
    p = write_csv(
        tmp_path,
        "pmid,screening,fulltext,instrument,setting,screening__r1,screening__r2,source\n"
        "111,include,include,MMSE;MoCA,community,include,exclude,test\n"
        "222.0,Excluded,,,,exclude,exclude,test\n",
    )
    a, b = load_gold(p, CODEBOOK)
    assert a.pmid == "111" and a.screening == "include" and a.fulltext == "include"
    assert a.labels == {"instrument": ["MMSE", "MoCA"], "setting": ["community"]}
    assert a.reviewers == {"r1": {"screening": "include"}, "r2": {"screening": "exclude"}}
    assert b.pmid == "222" and b.screening == "exclude" and b.labels == {}


def test_gold_reports_every_problem_at_once(tmp_path):
    p = write_csv(
        tmp_path,
        "pmid,screening,instrument\n"
        "111,include,SLUMS\n"      # label not in codebook
        "abc,include,\n"           # bad PMID
        "333,maybe,\n"             # bad decision
        "444,exclude,MMSE\n"       # excluded article with labels
        "555,include,\n"
        "555,include,\n",          # duplicate
    )
    with pytest.raises(GoldError) as e:
        load_gold(p, CODEBOOK)
    text = str(e.value)
    for expected in ["SLUMS", "'abc'", "'maybe'", "excluded article has annotation labels", "duplicate PMIDs: ['555']"]:
        assert expected in text


def test_gold_requires_pmid_and_screening(tmp_path):
    p = write_csv(tmp_path, "id,decision\n111,include\n")
    with pytest.raises(GoldError, match="missing required column 'pmid'"):
        load_gold(p, CODEBOOK)


def test_write_then_load_roundtrip(tmp_path):
    recs = [
        GoldRecord(pmid="1", screening="include", labels={"instrument": ["IQCODE"]}),
        GoldRecord(pmid="2", screening="exclude"),
    ]
    p = tmp_path / "g.csv"
    write_gold(recs, p, dimensions=list(CODEBOOK.dimensions))
    back = load_gold(p, CODEBOOK)
    assert [(r.pmid, r.screening, r.labels) for r in back] == [
        ("1", "include", {"instrument": ["IQCODE"]}),
        ("2", "exclude", {}),
    ]
    assert summarize(back)["screening"] == {"include": 1, "exclude": 1}


# ---- split -----------------------------------------------------------------


def fake_gold(n_include=90, n_exclude=178):
    return [GoldRecord(pmid=str(i), screening="include") for i in range(1, n_include + 1)] + [
        GoldRecord(pmid=str(i), screening="exclude")
        for i in range(n_include + 1, n_include + n_exclude + 1)
    ]


def test_split_sizes_no_overlap_and_stratified():
    gold = fake_gold()
    s = make_split(gold, n_calibration=40, seed=1)
    cal, test = set(s["calibration"]), set(s["test"])
    assert len(cal) == 40 and len(test) == 228
    assert not cal & test and cal | test == {r.pmid for r in gold}
    includes = {r.pmid for r in gold if r.screening == "include"}
    assert len(cal & includes) == 13  # 40 * 90/268 = 13.4 -> 13


def test_split_is_deterministic_for_a_seed():
    gold = fake_gold()
    assert make_split(gold, 40, seed=7) == make_split(gold, 40, seed=7)
    assert make_split(gold, 40, seed=7)["calibration"] != make_split(gold, 40, seed=8)["calibration"]


def test_saved_split_is_frozen(tmp_path):
    p = tmp_path / "split.json"
    save_split({"calibration": ["1"], "test": ["2"]}, p)
    with pytest.raises(FileExistsError, match="frozen"):
        save_split({"calibration": ["2"], "test": ["1"]}, p)
    assert load_split(p)["calibration"] == ["1"]
    save_split({"calibration": ["2"], "test": ["1"]}, p, force=True)
    assert load_split(p)["calibration"] == ["2"]


# ---- the real placeholder files (skipped if not built) ----------------------

GOLD = ROOT / "data/gold/gold.csv"
SPLIT = ROOT / "configs/splits/placeholder.json"


@pytest.mark.skipif(not GOLD.exists(), reason="run scripts/build_placeholder_gold.py first")
def test_placeholder_gold_matches_codebook_and_split():
    gold = load_gold(GOLD, CODEBOOK)  # raises if any label is not in the codebook
    assert len(gold) == 268
    split = load_split(SPLIT)
    assert set(split["calibration"]) | set(split["test"]) == {r.pmid for r in gold}
    from adrd_review.gold import file_sha256

    assert split["gold_sha256"] == file_sha256(GOLD), "gold file changed after the split was frozen"
