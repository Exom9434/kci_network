"""
새 임계값 로직 테스트 스크립트
================================
기존 로직과 새 로직 비교.

[기존 로직]
  SINGLE=0.35, MULTI=0.175
  - 단독: 최고 카테고리 ≥ 0.35
  - 융복합: 2개 이상이 각각 ≥ 0.175
  - 기초법: 나머지

[새 로직 - A안]
  1. 키워드 매칭 0개           → 기초법
  2. 최고 카테고리 ≥ 50%      → 단독 분류
  3. 25% 이상 카테고리 2개+   → 융복합 (최대 4개)
  4. 25% 이상 카테고리 1개    → 단독 분류
  5. 나머지                   → 기초법

사용법:
    python 04.analyze_new_threshold.py
"""

import re
import ahocorasick
import pandas as pd
from collections import defaultdict

KEYWORD_CSV     = "00.법률별_키워드_카테고리.csv"
PAPER_CSV       = "AI_법학_논문목록.csv"
KEYWORD_MIN_LEN = 3

T_SINGLE = 0.50   # 단독 확정 기준
T_MULTI  = 0.25   # 융복합 참여 기준


# ── 유틸 ─────────────────────────────────────────────────

def build_cat_kw():
    df = pd.read_csv(KEYWORD_CSV, encoding="utf-8-sig")
    df = df[df["카테고리"].notna() & df["키워드 목록"].notna()]
    cat_kw = defaultdict(set)
    for _, row in df.iterrows():
        for kw in str(row["키워드 목록"]).split(","):
            kw = kw.strip()
            if kw and re.search(r"[가-힣]", kw) and len(kw) >= KEYWORD_MIN_LEN:
                cat_kw[row["카테고리"]].add(kw)
    return dict(cat_kw)


def build_automaton(cat_kw):
    kw_to_cats = defaultdict(set)
    for cat, kws in cat_kw.items():
        for kw in kws:
            kw_to_cats[kw].add(cat)
    A = ahocorasick.Automaton()
    for kw, cats in kw_to_cats.items():
        A.add_word(kw, (cats, kw))
    A.make_automaton()
    return A


def classify_old(title, abstract, automaton):
    """기존 로직: SINGLE=0.35, MULTI=0.175"""
    title    = str(title)    if pd.notna(title)    else ""
    abstract = str(abstract) if pd.notna(abstract) else ""
    for s in ["리걸테크", "자율무기"]:
        if s in title: return s
    text = title + " " + abstract
    matched = defaultdict(set)
    for _, (cats, kw) in automaton.iter(text):
        for cat in cats: matched[cat].add(kw)
    counts = {c: len(k) for c, k in matched.items()}
    total  = sum(counts.values())
    if total == 0: return "기초법"
    ratios = {c: n / total for c, n in counts.items()}
    above  = [c for c, r in ratios.items() if r >= 0.175]
    if len(above) >= 2:
        top2 = sorted(above, key=lambda c: ratios[c], reverse=True)[:2]
        return f"융복합({'|'.join(sorted(top2))})"
    top = max(ratios, key=ratios.get)
    return top if ratios[top] >= 0.35 else "기초법"


def classify_new(title, abstract, automaton):
    """새 로직 A안: 1→기초법 2→단독(≥50%) 3→융복합(≥25% 2개+) 4→단독(≥25% 1개) 5→기초법"""
    title    = str(title)    if pd.notna(title)    else ""
    abstract = str(abstract) if pd.notna(abstract) else ""
    for s in ["리걸테크", "자율무기"]:
        if s in title: return s
    text = title + " " + abstract

    matched = defaultdict(set)
    for _, (cats, kw) in automaton.iter(text):
        for cat in cats: matched[cat].add(kw)
    counts = {c: len(k) for c, k in matched.items()}
    total  = sum(counts.values())

    # 1. 키워드 미매칭
    if total == 0:
        return "기초법"

    ratios = {c: n / total for c, n in counts.items()}
    top    = max(ratios, key=ratios.get)

    # 2. 단독 확정 (과반수)
    if ratios[top] >= T_SINGLE:
        return top

    # 3. 25% 이상 카테고리 2개+ → 융복합 (최대 4개)
    above_multi = sorted(
        [c for c, r in ratios.items() if r >= T_MULTI],
        key=lambda c: ratios[c], reverse=True
    )[:4]
    if len(above_multi) >= 2:
        return f"융복합({'|'.join(sorted(above_multi))})"

    # 4. 25% 이상 카테고리 1개 → 단독 분류 (A안 추가)
    if len(above_multi) == 1:
        return above_multi[0]

    # 5. 나머지 → 기초법
    return "기초법"


# ── 메인 ─────────────────────────────────────────────────

def main():
    cat_kw    = build_cat_kw()
    automaton = build_automaton(cat_kw)
    papers    = pd.read_csv(PAPER_CSV, encoding="utf-8-sig")
    N         = len(papers)

    old_labels, new_labels = [], []
    for _, row in papers.iterrows():
        old_labels.append(classify_old(row.get("제목", ""), row.get("초록", ""), automaton))
        new_labels.append(classify_new(row.get("제목", ""), row.get("초록", ""), automaton))

    old_vc = pd.Series(old_labels).value_counts()
    new_vc = pd.Series(new_labels).value_counts()

    fixed_cats = ["공법", "민사법", "형사법", "데이터법", "지식재산권법",
                  "노동법", "사회보장법", "경제법", "금융법", "조세법", "의료법"]

    def summarize(vc, label):
        n기 = vc.get("기초법", 0)
        n융 = sum(v for k, v in vc.items() if k.startswith("융복합"))
        n단 = N - n기 - n융 - vc.get("리걸테크", 0) - vc.get("자율무기", 0)
        print(f"\n[{label}]")
        print(f"  기초법:   {n기:4d}편 ({n기/N*100:.1f}%)")
        print(f"  융복합:   {n융:4d}편 ({n융/N*100:.1f}%)")
        print(f"  단독분류: {n단:4d}편 ({n단/N*100:.1f}%)")
        print(f"  단독 상세: " + " / ".join(f"{c}={vc.get(c, 0)}" for c in fixed_cats))

    summarize(old_vc, "기존 (SINGLE=0.35, MULTI=0.175)")
    summarize(new_vc, "새 로직 A안 (SINGLE=0.50, MULTI=0.25, 1개≥25%→단독)")

    print("\n=== 기존: 융복합 조합 Top 15 ===")
    for k, v in sorted({k: v for k, v in old_vc.items()
                        if k.startswith("융복합")}.items(),
                       key=lambda x: -x[1])[:15]:
        print(f"  {v:4d}편  {k}")

    print("\n=== 새 로직: 융복합 조합 Top 15 ===")
    for k, v in sorted({k: v for k, v in new_vc.items()
                        if k.startswith("융복합")}.items(),
                       key=lambda x: -x[1])[:15]:
        print(f"  {v:4d}편  {k}")

    print("\n=== 새 로직: 3개 이상 융복합 ===")
    multi3 = {k: v for k, v in new_vc.items()
               if k.startswith("융복합") and k.count("|") >= 2}
    if multi3:
        for k, v in sorted(multi3.items(), key=lambda x: -x[1]):
            print(f"  {v:4d}편  {k}")
    else:
        print("  없음")

    print(f"\n=== 기초법 변화 ===")
    n기초_old = old_vc.get("기초법", 0)
    n기초_new = new_vc.get("기초법", 0)
    print(f"  기존: {n기초_old}편 → 새 로직: {n기초_new}편  ({n기초_new - n기초_old:+d}편)")


if __name__ == "__main__":
    main()
