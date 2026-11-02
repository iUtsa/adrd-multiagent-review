"""Phase 1 vs phase 2: the blind check of the papers the two runs decided differently."""

import csv

from adrd_review.eval import tiebreak


def _write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_only_opposite_decisions_go_to_the_sheet_and_the_run_right_more_often_wins(tmp_path):
    p1 = [{"pmid": str(i), "title": f"t{i}", "year": "2024", "link": f"https://pubmed/{i}", "stage": s, "outcome": o,
           "reason": "r"} for i, (o, s) in enumerate([("included", "full text"), ("excluded", "title"),
                                                      ("included", "full text"), ("for human review", "full text"),
                                                      ("excluded", "abstract")])]
    p2 = [{"pmid": str(i), "title": f"t{i}", "outcome": o, "stage": s, "reason": "q", "reviewer A": "included",
           "reviewer B": "excluded (title)", "reviewer C": "included"}
          for i, (o, s) in enumerate([("included", "full text"), ("included", "full text"),
                                      ("excluded", "abstract"), ("excluded", "title"), ("excluded", "title")])]
    _write(tmp_path / "p1.csv", p1)
    _write(tmp_path / "p2.csv", p2)
    pairs = tiebreak.disagreements(tmp_path / "p1.csv", tmp_path / "p2.csv")
    assert sorted(x["p1"]["pmid"] for x in pairs) == ["1", "2"]  # human review / same decisions are left out
    rows = tiebreak.make_sheet(pairs, tmp_path / "out", {"1": "abs"})
    assert (tmp_path / "out/phase_check.xlsx").exists()
    assert "phase 1 outcome" not in open(tmp_path / "out/phase_check.csv").readline()  # blind
    # the person includes paper 1 (phase 2 included it) and excludes paper 2 (phase 1 included it): phase 2 right twice
    filled = [{"#": r["#"], tiebreak.DECISION: "include" if r["PMID"] == "1" else "exclude"} for r in rows]
    _write(tmp_path / "filled.csv", filled)
    res = tiebreak.score(tmp_path / "filled.csv", tmp_path / "out/phase_check_answer_key.csv", decided_by_both=4)
    assert (res["phase 1 right"], res["phase 2 right"], res["better"]) == (0, 2, None)  # 2 papers: not significant
    assert res["accuracy_gap_phase2_minus_phase1"] == 0.5
    assert tiebreak.sign_test(0, 10) < 0.01 and tiebreak.sign_test(5, 10) == 1.0
