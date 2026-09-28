"""
기초법 분류 원인 분석 스크립트
================================
03.classify_papers.py 실행 결과에서 '기초법'으로 분류된 논문의 원인을 두 가지로 분리:
  1. 키워드_미매칭 : 키워드 사전과 논문 텍스트가 전혀 겹치지 않음
  2. 임계값_미달   : 키워드는 매칭됐으나 단독/융복합 기준 모두 미달

사용법:
    python 04.analyze_기초법.py
"""

import re
import ahocorasick
import pandas as pd
from collections import defaultdict

KEYWORD_CSV  = "00.법률별_키워드_카테고리.csv"
PAPER_CSV    = "AI_법학_논문목록.csv"
RESULT_CSV   = "03.논문_카테고리_분류결과.csv"

THRESHOLD_SINGLE = 0.35
THRESHOLD_MULTI  = 0.5 * THRESHOLD_SINGLE
KEYWORD_MIN_LEN  = 3


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


def get_reason(title, abstract, automaton):
    """기초법으로 분류된 논문의 원인을 반환"""
    title    = str(title)    if pd.notna(title)    else ""
    abstract = str(abstract) if pd.notna(abstract) else ""
    text     = title + " " + abstract

    matched = defaultdict(set)
    for _, (cats, kw) in automaton.iter(text):
        for cat in cats:
            matched[cat].add(kw)

    counts = {c: len(k) for c, k in matched.items()}
    total  = sum(counts.values())

    if total == 0:
        return "키워드_미매칭", 0

    ratios  = {c: n / total for c, n in counts.items()}
    top_cat = max(ratios, key=ratios.get)
    above_m = [c for c, r in ratios.items() if r >= THRESHOLD_MULTI]

    if len(above_m) >= 2 or ratios[top_cat] >= THRESHOLD_SINGLE:
        return "기초법_아님", total   # 실제로는 융복합 or 단독 분류

    return "임계값_미달", total


# ── 메인 ─────────────────────────────────────────────────

def main():
    cat_kw    = build_cat_kw()
    automaton = build_automaton(cat_kw)
    papers    = pd.read_csv(PAPER_CSV, encoding="utf-8-sig")
    result    = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")

    기초법_idx = result[result["분류결과"] == "기초법"].index

    reasons = []
    for i in 기초법_idx:
        row    = papers.loc[i]
        reason, total_kw = get_reason(row.get("제목", ""), row.get("초록", ""), automaton)
        reasons.append({
            "논문ID":    row.get("논문ID", ""),
            "제목":      row.get("제목", ""),
            "초록유무":  "있음" if str(row.get("초록", "")).strip() else "없음",
            "매칭_키워드수": total_kw,
            "원인":      reason,
        })

    df_result = pd.DataFrame(reasons)

    print(f"기초법 총 {len(df_result)}편\n")

    print("=== 원인 분포 ===")
    print(df_result["원인"].value_counts().to_string())

    print("\n=== 키워드 미매칭 — 초록 유무 ===")
    no_match = df_result[df_result["원인"] == "키워드_미매칭"]
    print(no_match["초록유무"].value_counts().to_string())

    print("\n=== 임계값 미달 — 매칭 키워드 수 분포 ===")
    미달 = df_result[df_result["원인"] == "임계값_미달"]
    print(미달["매칭_키워드수"].describe().to_string())

    print("\n=== 임계값 미달 논문 예시 (매칭 키워드수 기준 상위 10개) ===")
    for _, row in 미달.sort_values("매칭_키워드수", ascending=False).head(10).iterrows():
        print(f"  [{row['매칭_키워드수']}개 매칭] {str(row['제목'])[:60]}")


if __name__ == "__main__":
    main()
