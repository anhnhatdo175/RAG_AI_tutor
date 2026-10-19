"""
LAYER 1 / Junyi Academy adapter (docx 2.8.1): second interaction log, same common schema as ASSISTments -> ``data.finalize_logs``.

Files (Kaggle "Learning Activity Public Dataset by Junyi Academy", 2020 release, all ids hashed):
  * Log_Problem.csv   timestamp_TW, uuid (learner), ucid (content = exercise), upid (concrete problem), problem_number,
                      is_correct, total_sec_taken, total_attempt_cnt, used_hint_cnt, ...      (~1.5 GB, read in chunks)
  * Info_Content.csv  ucid, content_pretty_name (CHINESE title), content_kind, difficulty, subject, learning_stage, level*_id (hashed)

Design:
  * learner = uuid, skill = corpus skill tag (many Junyi exercises collapse into one tag), item = upid (a concrete problem, so the
    empirical item difficulty of ``finalize_logs`` works exactly as for ASSISTments); falls back to ucid when upid is absent.
  * first-try correctness = is_correct and total_attempt_cnt == 1.
  * exercise -> tag: only Info_Content rows with content_kind == 'Exercise' and subject == 'math'.  Route 1: English khan-exercises name
    (older releases); route 2: Traditional-Chinese keyword rules (``ZH_RULES``, ordered, first match wins); route 3: English keyword
    rules of the SkillMapper on the title.  The route is recorded, unmapped titles are listed in the report so the rules can be refined.
  * Column names are resolved through ALIASES; a missing required column raises a clear error.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, data
from ..skill_mapping import SkillMapper

ALIASES = {
    "user": ["uuid", "user_id"],
    "ucid": ["ucid"],
    "exercise": ["exercise", "exercise_name"],            # English-name logs (older release)
    "problem": ["upid"],
    "time": ["timestamp_TW", "upload_time", "timestamp"],
    "correct": ["is_correct", "correct"],
    "secs": ["total_sec_taken", "sec_taken"],
    "attempts": ["total_attempt_cnt", "attempt_cnt"],
}
OPTIONAL = {"exercise", "ucid", "problem", "attempts", "secs"}
VERSION_SUFFIX = re.compile(r"_(\d+(?:\.\d+)?)$")

# Ordered (specific before generic).  Traditional Chinese, as used in Junyi titles.  First match wins.
ZH_RULES: list[tuple[str, str]] = [
    ("Division-Fractions", r"分數.{0,6}除|除以分數|分數的除法"),
    ("Multiplication-Fractions", r"分數.{0,6}乘|乘以分數|分數的乘法"),
    ("Addition-and-Subtraction-Fractions", r"(分數|異分母|同分母|帶分數).{0,8}(加|減)|分數的加|分數的減|異分母|同分母"),
    ("Conversion-of-Fraction-Decimals-Percents", r"(分數|小數|百分率|百分數|成數).{0,6}(化|換|轉).{0,6}(分數|小數|百分|成)|化成.{0,4}(分數|小數|百分)|分數與小數|小數與分數|百分數"),
    ("Least-Common-Multiple", r"最小公倍數"),
    ("Greatest-Common-Factor", r"最大公因數|最大公因|互質"),
    ("Distributive-Property", r"分配律|分配性質"),
    ("Divisibility-Rules", r"整除|倍數.{0,4}判別|判別.{0,4}倍數"),
    ("Prime-Number", r"質數|質因數|質因"),
    ("Percent-Discount", r"折扣|打折|折\b|幾折"),
    ("Finding-Percents", r"求百分率|百分率是多少|成數是多少"),
    ("Percent-Of", r"百分率|百分比|成數|%|基準量|比較量"),
    ("Rounding", r"四捨五入|概數|無條件進位|無條件捨去"),
    ("Mean", r"平均"),
    ("Midpoint", r"中點"),
    ("Perimeter-of-a-Polygon", r"周長"),
    ("Probability-of-Two-Distinct-Events", r"機率.{0,8}(兩|獨立|同時|連續)|獨立事件|互斥"),
    ("Probability-of-a-Single-Event", r"機率|可能性"),
    ("Counting-Methods", r"排列|組合|樹狀圖|計數|乘法原理|加法原理|幾種"),
    ("Proportion", r"比例|比值|正比|反比|連比|相等的比"),
    ("Unit-Rate", r"單價|速率|時速|速度|單位量|每.{0,3}(小時|分鐘|秒|公斤|公升)"),
    ("Pythagorean-Theorem", r"畢氏定理|畢式定理"),
    ("Square-Root", r"平方根|開根號|根號"),
    ("Volume-Rectangular-Prism", r"(長方體|正方體|立方體).{0,6}體積|體積.{0,6}(長方體|正方體|立方體)"),
    ("Finding-Slope-From-Equation", r"斜率"),
    ("Equation-Solving-Two-or-Fewer-Steps", r"一元一次方程|解方程|方程式"),
]
ZH_COMPILED = [(t, re.compile(p)) for t, p in ZH_RULES]


def _resolve(cols: list[str], key: str, required: bool = True) -> str | None:
    for a in ALIASES[key]:
        if a in cols:
            return a
    if required:
        raise KeyError(f"Junyi log: no column for '{key}' (looked for {ALIASES[key]}); found {cols}")
    return None


def khan_name_tags(khan_jsonl: Path, mapper: SkillMapper) -> dict[str, str]:
    """khan-exercises template name -> majority skill tag of its rendered items (used when titles are English ids)."""
    votes: dict[str, Counter] = defaultdict(Counter)
    if not khan_jsonl.exists():
        return {}
    for ln in khan_jsonl.read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            r = json.loads(ln)
            tag, _, _ = mapper.tag_text(r["text"].lower(), r.get("subject"), None, False)
            if tag:
                votes[str(r["subject"]).replace("khan:", "")][tag] += 1
    return {name: c.most_common(1)[0][0] for name, c in votes.items()}


class JunyiMapper(SkillMapper):
    """``to_tag(skill_name)`` accepts the tag strings produced by ``build_exercise_tags``."""

    def __post_init__(self):
        super().__post_init__()
        self.known_tags: set[str] = set()

    def to_tag(self, log_skill: str):
        return log_skill if log_skill in self.known_tags else None


def tag_title(title: str, mapper: SkillMapper, khan: dict[str, str]) -> tuple[str | None, str]:
    base = VERSION_SUFFIX.sub("", title)
    t = khan.get(title) or khan.get(base)
    if t:
        return t, "khan_name"
    for tag, rx in ZH_COMPILED:
        if rx.search(title):
            return tag, "zh_keyword"
    t, _, _ = mapper.tag_text(base.replace("_", " ").lower(), "khan:" + title, None, False)
    return (t, "en_keyword") if t else (None, "unmapped")


def read_exercise_table(path: Path) -> pd.DataFrame:
    """Info_Content.csv -> ucid, title (math exercises only when the columns exist)."""
    et = pd.read_csv(path)
    title = next((c for c in ("content_pretty_name", "name", "exercise") if c in et.columns), None)
    if "ucid" not in et.columns or title is None:
        raise KeyError(f"exercise table needs ucid + one of content_pretty_name/name; found {list(et.columns)}")
    if "content_kind" in et.columns:
        et = et[et.content_kind.astype(str).str.lower() == "exercise"]
    if "subject" in et.columns:
        et = et[et.subject.astype(str).str.lower() == "math"]
    out = et[["ucid", title]].rename(columns={title: "title"}).drop_duplicates("ucid")
    out["title"] = out.title.astype(str)
    return out.reset_index(drop=True)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def read_junyi(log_path: Path, exercise_table: Path | None, keep_ucids: set | None = None, nrows: int | None = None,
               chunksize: int = 2_000_000) -> pd.DataFrame:
    """Raw Junyi problem log -> common-schema columns (order_id,user_id,problem_id,correct,ucid,ms); chunked, filtered to ``keep_ucids``."""
    head = list(pd.read_csv(log_path, nrows=0).columns)
    c = {k: _resolve(head, k, required=k not in OPTIONAL) for k in ALIASES}
    use = sorted({v for v in c.values() if v})
    parts = []
    for ch in pd.read_csv(log_path, usecols=use, chunksize=chunksize, nrows=nrows, dtype={c["user"]: "category"} if False else None):
        key = c["ucid"] or c["exercise"]
        if keep_ucids is not None:
            ch = ch[ch[key].isin(keep_ucids)]
        first = _truthy(ch[c["correct"]])
        if c["attempts"]:
            first &= pd.to_numeric(ch[c["attempts"]], errors="coerce") == 1
        t = pd.to_datetime(ch[c["time"]], errors="coerce", utc=True)
        secs = pd.to_numeric(ch[c["secs"]], errors="coerce") if c["secs"] else pd.Series(10.0, index=ch.index)
        parts.append(pd.DataFrame({
            "user_id": ch[c["user"]].astype(str).to_numpy(), "ucid": ch[key].astype(str).to_numpy(),
            "problem_id": (ch[c["problem"]] if c["problem"] else ch[key]).astype(str).to_numpy(),
            "correct": first.astype(int).to_numpy(), "ms": (secs * 1000).clip(500, 120_000).to_numpy(),
            "_t": ((t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).to_numpy()}))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["_t", "ms"])
    out = out.sort_values("_t", kind="stable").reset_index(drop=True)
    out["order_id"] = np.arange(len(out))
    return out.drop(columns="_t")


def load_junyi_logs(log_path: Path, exercise_table: Path | None = None, khan_jsonl: Path | None = None, nrows: int | None = None,
                    min_user_rows: int = 20):
    """Junyi -> evaluation frame (same columns as ``data.load_logs``) and the exercise->tag table used."""
    if exercise_table is None:
        guess = Path(log_path).with_name("Info_Content.csv")
        exercise_table = guess if guess.exists() else None
    head = list(pd.read_csv(log_path, nrows=0).columns)
    english_log = _resolve(head, "exercise", required=False) is not None
    mapper0 = SkillMapper()
    khan = khan_name_tags(khan_jsonl or (config.KHAN_DIR / "khan_exercises.jsonl"), mapper0)
    if english_log and exercise_table is None:               # log carries the exercise name itself
        names = pd.read_csv(log_path, usecols=[_resolve(head, "exercise")]).iloc[:, 0].astype(str).unique()
        ex = pd.DataFrame({"ucid": names, "title": names})
    else:
        if exercise_table is None:
            raise FileNotFoundError("Info_Content.csv (exercise table) is needed to name the exercises; pass --exercise-table")
        ex = read_exercise_table(exercise_table)
    tagged = [tag_title(t, mapper0, khan) for t in ex.title]
    ex["tag"], ex["route"] = [t for t, _ in tagged], [r for _, r in tagged]
    keep = set(ex.ucid[ex.tag.notna()].astype(str))
    raw = read_junyi(log_path, exercise_table, keep_ucids=keep, nrows=nrows)
    raw = raw.merge(ex[["ucid", "tag"]].assign(ucid=ex.ucid.astype(str)), on="ucid", how="left")
    mapper = JunyiMapper()
    mapper.known_tags = set(ex.tag.dropna())
    raw["skill_name"] = raw.tag
    raw["skill_id"] = pd.factorize(raw.skill_name)[0]
    raw["problem_id"] = pd.factorize(raw.problem_id)[0]
    raw["user_id"] = pd.factorize(raw.user_id)[0]
    df = data.finalize_logs(raw[["order_id", "user_id", "problem_id", "correct", "skill_id", "skill_name", "ms"]], mapper, min_user_rows)
    return df, ex


def report(log_path: Path, exercise_table: Path | None = None, nrows: int | None = None) -> dict:
    df, ex = load_junyi_logs(log_path, exercise_table, nrows=nrows)
    eligible = None
    rj = config.ART / "results.json"
    if rj.exists():
        eligible = set(json.loads(rj.read_text())["data"]["retrievable_skills"])
    unm = ex[ex.tag.isna()].title.tolist()
    rep = {"events": int(len(df)), "learners": int(df.user_id.nunique()), "split_rows": df.split.value_counts().to_dict(),
           "math_exercises": int(len(ex)), "exercises_mapped": int(ex.tag.notna().sum()), "route": ex.route.value_counts().to_dict(),
           "exercises_per_tag": ex.tag.value_counts().to_dict(), "events_per_tag": df.tag.value_counts().to_dict(),
           "problems_per_tag": df.groupby("tag").problem_id.nunique().to_dict(), "first_try_accuracy": float(df.correct.mean()),
           "known_item_share_test": float(df[df.split == "test"].item_known.mean()),
           "unmapped_titles_sample": unm[:60]}
    if eligible is not None:
        rep["events_in_retrievable_tags"] = float(df.tag.isin(eligible).mean())
        rep["retrievable_tags_present"] = sorted(set(df.tag) & eligible)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Junyi adapter report")
    ap.add_argument("log"); ap.add_argument("--exercise-table"); ap.add_argument("--nrows", type=int)
    a = ap.parse_args()
    r = report(Path(a.log), Path(a.exercise_table) if a.exercise_table else None, a.nrows)
    out = config.ART / "junyi_report.json"
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps(r, indent=1, ensure_ascii=False)[:4500], "\n->", out)
