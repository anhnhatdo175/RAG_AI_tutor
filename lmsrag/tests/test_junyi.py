"""Schema test on a tiny SYNTHETIC file shaped like the 2020 Kaggle release (hashed ids, Chinese titles). Not Junyi data."""
import numpy as np
import pandas as pd

from lmsrag.layer1_data import junyi

TITLES = {"c1": "分數的加減: 異分母", "c2": "最大公因數", "c3": "四捨五入到整數", "c4": "【基礎】怎樣解題: 數量關係", "c5": "影片: 質數"}


def _fake(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(10):
        for i in range(80):
            uc = ["c1", "c2", "c3", "c4"][i % 4]
            rows.append(dict(timestamp_TW=f"2019-05-{1 + i // 4:02d} 10:{i % 60:02d}:00 UTC", uuid=f"u{u}", ucid=uc, upid=f"{uc}-p{i % 5}",
                             problem_number=i, is_correct=bool(rng.random() < 0.6), total_sec_taken=int(rng.integers(5, 60)),
                             total_attempt_cnt=int(rng.integers(1, 3))))
    pd.DataFrame(rows).to_csv(tmp_path / "Log_Problem.csv", index=False)
    pd.DataFrame({"ucid": list(TITLES), "content_pretty_name": list(TITLES.values()), "content_kind": ["Exercise"] * 4 + ["Video"],
                  "subject": ["math"] * 5}).to_csv(tmp_path / "Info_Content.csv", index=False)


def test_zh_rules():
    assert junyi.tag_title("分數的加減: 異分母", junyi.SkillMapper(), {})[0] == "Addition-and-Subtraction-Fractions"
    assert junyi.tag_title("最小公倍數", junyi.SkillMapper(), {})[0] == "Least-Common-Multiple"
    assert junyi.tag_title("分數的除法", junyi.SkillMapper(), {})[0] == "Division-Fractions"


def test_adapter_schema_and_mapping(tmp_path):
    _fake(tmp_path)
    out, ex = junyi.load_junyi_logs(tmp_path / "Log_Problem.csv", tmp_path / "Info_Content.csv", min_user_rows=20)
    for c in ["user_id", "problem_id", "correct", "skill_id", "skill_name", "ms", "split", "opp", "tag", "item_d", "bucket", "item_known"]:
        assert c in out.columns
    assert set(out.split) == {"history", "val", "test"} and out.correct.isin([0, 1]).all()
    m = ex.set_index("title")
    assert m.loc["最大公因數", "tag"] == "Greatest-Common-Factor" and m.loc["【基礎】怎樣解題: 數量關係", "route"] == "unmapped"
    assert "影片: 質數" not in set(ex.title)                       # videos are dropped
    assert out.tag.notna().all() and set(out.tag) == {"Addition-and-Subtraction-Fractions", "Greatest-Common-Factor", "Rounding"}
