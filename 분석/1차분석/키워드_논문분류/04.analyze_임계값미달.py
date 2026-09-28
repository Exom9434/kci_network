"""
임계값 미달 논문 분석 스크립트
================================
기초법으로 분류된 논문 중 '임계값 미달' 케이스만 추려서,
threshold를 낮출 경우 몇 편이 구제되는지 시뮬레이션.

사용법:
    python 04.analyze_임계값미달.py
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

TEST_THRESHOLDS  = [0.15, 0.20, 0.25, 0.30, 0.35]


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


def get_ratios(title, abstract, automaton):
    """카테고리별 키워드 매칭 비율 반환"""
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
        return {}, 0
    return {c: n / total for c, n in counts.items()}, total


def classify_with_threshold(ratios, t_s):
    """주어진 threshold로 분류 결과 반환"""
    if not ratios:
        return "기초법(미매칭)"
    t_m    = t_s * 0.5
    above  = [c for c, r in ratios.items() if r >= t_m]
    if len(above) >= 2:
        top2 = sorted(above, key=lambda c: ratios[c], reverse=True)[:2]
        return f"융복합({'|'.join(sorted(top2))})"
    top = max(ratios, key=ratios.get)
    return top if ratios[top] >= t_s else "기초법(임계값미달)"


# ── 메인 ─────────────────────────────────────────────────

def main():
    cat_kw    = build_cat_kw()
    automaton = build_automaton(cat_kw)
    papers    = pd.read_csv(PAPER_CSV,   encoding="utf-8-sig")
    result    = pd.read_csv(RESULT_CSV,  encoding="utf-8-sig")

    기초법_idx = result[result["분류결과"] == "기초법"].index

    # 임계값 미달 논문만 추출 (키워드 매칭 있는 것)
    미달_rows = []
    for i in 기초법_idx:
        row = papers.loc[i]
        ratios, total = get_ratios(row.get("제목", ""), row.get("초록", ""), automaton)
        if total == 0:
            continue   # 미매칭은 제외
        top_cat = max(ratios, key=ratios.get)
        미달_rows.append({
            "idx":       i,
            "제목":      row.get("제목", ""),
            "ratios":    ratios,
            "total_kw":  total,
            "top_cat":   top_cat,
            "top_ratio": ratios[top_cat],
        })

    print(f"임계값 미달 논문: {len(미달_rows)}편\n")

    # ── threshold별 구제 현황 ────────────────────────────
    print("=== threshold별 구제 현황 ===")
    print(f"{'threshold':>12}  {'구제됨':>8}  {'여전히 기초법':>14}  {'구제율':>8}")
    print("-" * 50)
    for t_s in TEST_THRESHOLDS:
        rescued    = 0
        still_기초법 = 0
        for r in 미달_rows:
            label = classify_with_threshold(r["ratios"], t_s)
            if "기초법" in label:
                still_기초법 += 1
            else:
                rescued += 1
        print(f"  t={t_s:.2f}      {rescued:>6}편  {still_기초법:>12}편  "
              f"{rescued/len(미달_rows)*100:>7.1f}%")

    # ── 구제됐을 때 어느 카테고리로 가는지 (t=0.25 기준) ──
    print(f"\n=== t=0.25 기준 구제 논문의 분류 결과 ===")
    rescued_labels = []
    for r in 미달_rows:
        label = classify_with_threshold(r["ratios"], 0.25)
        if "기초법" not in label:
            rescued_labels.append(label)
    label_counts = pd.Series(rescued_labels).value_counts()
    print(label_counts.to_string())

    # ── 미달 논문의 최고 비율 분포 ──────────────────────
    print(f"\n=== 임계값 미달 논문의 최고 카테고리 비율 분포 ===")
    top_ratios = pd.Series([r["top_ratio"] for r in 미달_rows])
    bins = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    labels = ["~10%", "10~15%", "15~20%", "20~25%", "25~30%", "30~35%"]
    print(pd.cut(top_ratios, bins=bins, labels=labels).value_counts().sort_index().to_string())

    # ── 여전히 기초법인 논문 예시 (t=0.25에서도 미달) ───
    print(f"\n=== t=0.25에서도 기초법인 논문 예시 (top 10) ===")
    still = [r for r in 미달_rows
             if "기초법" in classify_with_threshold(r["ratios"], 0.25)]
    for r in sorted(still, key=lambda x: x["top_ratio"])[:10]:
        cats = ", ".join(f"{c}:{v:.0%}" for c, v in
                         sorted(r["ratios"].items(), key=lambda x: -x[1])[:3])
        print(f"  [{r['top_ratio']:.0%}] {str(r['제목'])[:50]}")
        print(f"         → {cats}")


if __name__ == "__main__":
    main()
