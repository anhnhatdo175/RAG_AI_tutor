import pandas as pd

from lmsrag.datasets.skill_map import MapTagger
from lmsrag.labels.build_skill_map import apply_recheck, collect, read_sheet, rule_skill, split_list


def _sheets(a_rows, b_rows=(), c_rows=()):
    a = pd.DataFrame(a_rows, columns=["row", "assistments_skill", "khan_slug", "decision", "khan_slugs", "note"])
    b = pd.DataFrame(b_rows, columns=["row", "assistments_skill", "decision", "khan_slugs", "note"])
    c = pd.DataFrame(c_rows, columns=["row", "khan_slug", "tag_now", "decision", "assistments_skill", "note"])
    return a, b, c


def _pairs(rows):
    return {(r["source_skill"], r["khan_slug"], r["review"]) for r in rows}


def test_sheet_a_keep_drop_fix_and_template_less_rows():
    a, b, c = _sheets([
        ["1", "Mean", "mean_median_and_mode", "keep", "", ""],
        ["2", "Unit Rate", "rate_problems_2", "drop", "", ""],
        ["3", "Estimation", "rounding_whole_numbers", "fix", "rounding_whole_numbers;estimation_with_decimals", ""],
        ["4", "Median", "mean_median_and_mode", "fix", "average_word_problems", ""],
        ["5", "Venn Diagram", "", "drop", "", ""],
    ])
    rows, rejected, errors = collect(a, b, c)
    assert errors == []
    assert _pairs(rows) == {
        ("Mean", "mean_median_and_mode", "A:keep"),
        ("Estimation", "rounding_whole_numbers", "A:fix"), ("Estimation", "estimation_with_decimals", "A:fix"),
        ("Median", "average_word_problems", "A:fix"),
        ("Venn Diagram", "", "A:none"),
        ("Unit Rate", "", "A:all_dropped"),  # every pair dropped: the skill stays, with no template
    }
    # a fix that keeps the old slug does not reject it; one that replaces it does
    assert _pairs(rejected) == {("Unit Rate", "rate_problems_2", "A:drop"),
                                ("Median", "mean_median_and_mode", "A:fix_replaced")}


def test_sheets_b_and_c():
    a, b, c = _sheets(
        [],
        [["1", "Factoring Polynomials Standard", "add", "factoring_polynomials_1;factoring_polynomials_2", ""],
         ["2", "Volume Cone", "none", "", "prisms only"]],
        [["1", "making_a_decade", "khan:making_a_decade", "add", "Addition Whole Numbers", ""],
         ["2", "probability_with_perm_comb", "rule:Counting-Methods", "add",
          "Probability of a Single Event;Counting Methods", ""],
         ["3", "volume_2", "rule:Volume-Rectangular-Prism", "keep", "", ""],
         ["4", "matrix_dimensions", "khan:matrix_dimensions", "none", "", "matrices"]],
    )
    rows, rejected, errors = collect(a, b, c)
    assert errors == [] and rejected == []
    assert _pairs(rows) == {
        ("Factoring Polynomials Standard", "factoring_polynomials_1", "B:add"),
        ("Factoring Polynomials Standard", "factoring_polynomials_2", "B:add"),
        ("Volume Cone", "", "B:none"),
        ("Addition Whole Numbers", "making_a_decade", "C:add"),
        ("Probability of a Single Event", "probability_with_perm_comb", "C:add"),
        ("Counting Methods", "probability_with_perm_comb", "C:add"),
        ("Volume Rectangular Prism", "volume_2", "C:keep"),  # keep resolves the rule tag back to its skill
        ("", "matrix_dimensions", "C:none"),
    }
    none_row = next(r for r in rows if r["review"] == "C:none")
    assert none_row["canonical_tag"] == "khan:matrix_dimensions"


def test_invalid_rows_are_reported_not_written():
    a, b, c = _sheets(
        [["1", "Mean", "mean_median_and_mode", "kep", "", ""],
         ["2", "Mean", "average_word_problems", "fix", "", ""]],
        [["1", "Volume Cone", "add", "", ""]],
        [["1", "volume_2", "khan:volume_2", "keep", "", ""]],
    )
    _, _, errors = collect(a, b, c)
    assert len(errors) == 4
    assert any("A row 1" in e and "'kep'" in e for e in errors)
    assert any("A row 2" in e and "fix needs khan_slugs" in e for e in errors)
    assert any("B row 1" in e and "add needs khan_slugs" in e for e in errors)
    assert any("C row 1" in e and "keep needs a rule:<tag>" in e for e in errors)


def test_recheck_sheet_overrides_sheet_a():
    a, _, _ = _sheets([["1", "Addition Whole Numbers", "addition_1", "drop", "", ""],
                       ["2", "Median", "mean_median_and_mode", "drop", "", ""]])
    d = pd.DataFrame([["1", "1", "addition_1", "keep", "chỉ xét kỹ năng"],
                      ["2", "2", "mean_median_and_mode", "", ""],
                      ["3", "9", "volume_1", "keep", ""],
                      ["4", "2", "mean_median_and_mode", "maybe", ""]],
                     columns=["row", "a_row", "khan_slug", "decision", "note"])
    out, errors = apply_recheck(a, d)
    assert out["decision"].tolist() == ["keep", "drop"]  # an empty D decision leaves sheet A as it is
    assert out.loc[0, "note"] == "phiếu D: keep; chỉ xét kỹ năng"
    assert len(errors) == 2 and "D row 3" in errors[0] and "'maybe'" in errors[1]


def test_map_tagger_uses_the_dictionary_first(tmp_path):
    path = tmp_path / "map.csv"
    pd.DataFrame({"source_skill": ["Circumference", "Circumference", "Table", ""],
                  "canonical_tag": ["Circumference", "Circumference", "", "khan:x"],
                  "note": ["", "", "cần hình", ""]}).to_csv(path, index=False)
    t = MapTagger(path)
    assert t.to_tag("Circumference ") == "Circumference"  # 2009 label with a trailing space
    assert t.to_tag("Table") is None and t.unmapped_reason("table") == "cần hình"
    assert t.to_tag("Addition and Subtraction Fractions") == "Addition-and-Subtraction-Fractions"  # fallback


def test_helpers(tmp_path):
    assert split_list(" a ; b;;") == ["a", "b"]
    assert rule_skill("Polynomial-Factors") == "Polynomial Factors"
    assert rule_skill("Not-A-Skill") is None
    # Excel's "CSV (Comma delimited)" writes the ANSI code page; the reader falls back to it
    path = tmp_path / "sheet.csv"
    path.write_bytes("row,note\n1, xác suất \n".encode("cp1252", errors="replace"))
    assert read_sheet(path).loc[0, "row"] == "1"
    path.write_bytes("row,note\n1, phân vân \n".encode("utf-8-sig"))
    assert read_sheet(path).loc[0, "note"] == "phân vân"


def test_plain_math_for_annotators():
    from lmsrag.labels.make_sheets import plain_math

    assert plain_math(r"$\dfrac{3}{4} + \blue{\frac{1}{2}}$") == "(3)/(4) + (1)/(2)"
    assert plain_math(r"$6 \times 7 \div 2 = \ ?$") == "6 × 7 ÷ 2 = ?"
    assert plain_math(r"Express $0.6\overline{3}$ as a fraction.") == "Express 0.63 (lặp lại) as a fraction."
    assert plain_math(r"$\overline{AB} = 7$, $\angle ABC = 90^\circ$") == "đoạn AB = 7, ∠ABC = 90°"
    # found by both annotators in round 1: \pink was read as \pi + "nk"
    assert plain_math(r"$\Huge\blue{-5} + \green{-3}=\pink{{?}}$") == "-5 + -3=?"
    assert plain_math(r"$12\pi$ units") == "12π units"
    assert plain_math(r"$\sqrt{16} \leq 5$ and $\lvert -2 \rvert$, 15\% off \$20") == "√(16) ≤ 5 and | -2 |, 15% off $20"


def test_sheet_e_drops_restricts_and_adds_pairs():
    from lmsrag.labels.build_skill_map import apply_audit_fixes

    a, b, c = _sheets([["1", "Mean", "mean_median_and_mode", "keep", "", ""],
                       ["2", "Pattern Finding", "geometric_sequences_2", "keep", "", ""]])
    rows, rejected, _ = collect(a, b, c)
    e = pd.DataFrame([["1", "m007", "Mean", "mean_median_and_mode", "types", "mean", "ba dạng"],
                      ["2", "m024", "Pattern Finding", "geometric_sequences_2", "drop", "", "thay công thức"],
                      ["3", "", "Median", "mean_median_and_mode", "add", "median", ""],
                      ["4", "", "Mode", "mean_median_and_mode", "types", "mode", ""],
                      ["5", "", "Mean", "mean_median_and_mode", "fix", "", ""]],
                     columns=["row", "evidence", "source_skill", "khan_slug", "decision", "khan_types", "note"])
    errors = apply_audit_fixes(rows, rejected, e)
    assert len(errors) == 2 and "E row 4" in errors[0] and "'fix'" in errors[1]
    got = {(r["source_skill"], r["khan_slug"], r.get("khan_types", ""), r["review"]) for r in rows}
    assert got == {("Mean", "mean_median_and_mode", "mean", "E:types"),
                   ("Median", "mean_median_and_mode", "median", "E:add"),
                   ("Pattern Finding", "", "", "E:all_dropped")}  # its only template was dropped
    assert [(r["khan_slug"], r["review"]) for r in rejected] == [("geometric_sequences_2", "E:drop")]
    assert rows[0]["note"] == "phiếu E (m007): ba dạng"


def test_type_level_tags(tmp_path):
    from lmsrag.datasets.junyi2015 import khan_type_tag_lists

    m, t = tmp_path / "map.csv", tmp_path / "types.csv"
    pd.DataFrame({"source_skill": ["Mean", "Median", "Order All", "Order Pos"],
                  "canonical_tag": ["Mean", "Median", "OA", "OP"],
                  "khan_slug": ["mmm", "mmm", "ops", "ops"],
                  "khan_types": ["mean", "median", "", "a;1"]}).to_csv(m, index=False)
    pd.DataFrame({"slug": ["mmm", "mmm", "mmm", "ops", "ops", "ops"],
                  "problem_index": ["0", "1", "2", "0", "1", "2"],
                  "problem_id": ["mean", "median", "mode", "a", "1", "2"]}).to_csv(t, index=False)
    got = khan_type_tag_lists(m, t)
    assert got[("mmm", "median")] == got[("mmm", "1")] == ["Median"]
    assert got[("mmm", "mode")] == []  # no pair covers it
    assert got[("ops", "0")] == ["OA", "OP"]
    assert got[("ops", "1")] == ["OA", "OP"]  # type id "1", not position 1 (id "1" is also at position 1 here)
    assert got[("ops", "2")] == ["OA"]
