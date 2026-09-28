"""
임계값 스윕 스크립트
=============================================
THRESHOLD_SINGLE을 여러 값으로 바꿔가며 분류 결과를 비교 테이블로 출력.
THRESHOLD_MULTI는 THRESHOLD_SINGLE × 0.5 로 연동.

사용법:
    python 03.classify_sweep.py
"""

import re
import ahocorasick
import pandas as pd
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────
KEYWORD_CSV = "00.법률별_키워드_카테고리.csv"
PAPER_CSV   = "AI_법학_논문목록.csv"
OUTPUT_CSV  = "03.classify_sweep_결과.csv"   # 전체 상세 결과 저장

# 테스트할 THRESHOLD_SINGLE 값 목록
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# ── 유틸 ─────────────────────────────────────────────────

def is_pure_english(kw: str) -> bool:
    return not bool(re.search(r'[가-힣]', kw))


def build_category_keywords(path: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df['카테고리'].notna() & (df['키워드 목록'].notna())]
    df = df[df['키워드 목록'] != '']
    cat_kw = defaultdict(set)
    for _, row in df.iterrows():
        cat = row['카테고리']
        for kw in str(row['키워드 목록']).split(','):
            kw = kw.strip()
            if kw and not is_pure_english(kw):
                cat_kw[cat].add(kw)
    return dict(cat_kw)


def build_automaton(cat_kw: dict):
    kw_to_cats = defaultdict(set)
    for cat, kws in cat_kw.items():
        for kw in kws:
            kw_to_cats[kw].add(cat)
    A = ahocorasick.Automaton()
    for kw, cats in kw_to_cats.items():
        A.add_word(kw, (cats, kw))
    A.make_automaton()
    return A


def classify_paper(title, abstract, automaton, t_single, t_multi) -> str:
    title    = str(title)    if pd.notna(title)    else ""
    abstract = str(abstract) if pd.notna(abstract) else ""

    for special in ['리걸테크', '자율무기']:
        if special in title:
            return special

    text = title + " " + abstract
    matched_by_cat = defaultdict(set)
    for _, (cats, kw) in automaton.iter(text):
        for cat in cats:
            matched_by_cat[cat].add(kw)

    cat_counts = {cat: len(kws) for cat, kws in matched_by_cat.items()}
    total = sum(cat_counts.values())
    if total == 0:
        return "기초법"

    cat_ratios = {cat: cnt / total for cat, cnt in cat_counts.items()}

    above_multi = [cat for cat, r in cat_ratios.items() if r >= t_multi]
    if len(above_multi) >= 2:
        top2 = sorted(above_multi, key=lambda c: cat_ratios[c], reverse=True)[:2]
        return f"융복합({'|'.join(sorted(top2))})"

    top_cat = max(cat_ratios, key=cat_ratios.get)
    if cat_ratios[top_cat] >= t_single:
        return top_cat

    return "기초법"


# ── 메인 ─────────────────────────────────────────────────

def main():
    print("키워드 사전 로드 중...")
    cat_kw   = build_category_keywords(KEYWORD_CSV)
    automaton = build_automaton(cat_kw)
    papers   = pd.read_csv(PAPER_CSV, encoding="utf-8-sig")
    n        = len(papers)

    total_kw = sum(len(v) for v in cat_kw.values())
    print(f"카테고리 {len(cat_kw)}개 / 키워드 {total_kw}개 / 논문 {n}편\n")

    # 카테고리 목록 (나중에 컬럼 정렬용)
    fixed_cats = ['공법','민사법','형사법','데이터법','지식재산권법',
                  '노동법','사회보장법','경제법','금융법','조세법','의료법']

    sweep_rows  = []   # 비교 테이블용
    detail_rows = []   # 전체 상세 저장용

    for t_s in THRESHOLDS:
        t_m = round(t_s * 0.5, 4)
        labels = []
        for _, row in papers.iterrows():
            lbl = classify_paper(row.get('제목',''), row.get('초록',''),
                                 automaton, t_s, t_m)
            labels.append(lbl)
            detail_rows.append({
                'threshold_single': t_s,
                '논문ID': row.get('논문ID',''),
                '제목':   row.get('제목',''),
                '분류결과': lbl,
            })

        vc = pd.Series(labels).value_counts()

        n_기초법  = vc.get('기초법', 0)
        n_융복합  = sum(v for k, v in vc.items() if k.startswith('융복합'))
        n_특수    = vc.get('리걸테크', 0) + vc.get('자율무기', 0)
        n_단독    = n - n_기초법 - n_융복합 - n_특수

        row_dict = {
            'threshold_single': t_s,
            'threshold_multi':  t_m,
            '기초법(%)':   f"{n_기초법}({n_기초법/n*100:.0f}%)",
            '융복합(%)':   f"{n_융복합}({n_융복합/n*100:.0f}%)",
            '단독분류(%)': f"{n_단독}({n_단독/n*100:.0f}%)",
        }
        # 카테고리별 단독 분류 수
        for cat in fixed_cats:
            row_dict[cat] = vc.get(cat, 0)

        sweep_rows.append(row_dict)

        print(f"threshold={t_s:.2f}  "
              f"기초법:{n_기초법:4d}({n_기초법/n*100:.0f}%)  "
              f"융복합:{n_융복합:4d}({n_융복합/n*100:.0f}%)  "
              f"단독:{n_단독:4d}({n_단독/n*100:.0f}%)")

    # ── 비교 테이블 출력 ──────────────────────────────────
    print("\n" + "=" * 80)
    print("  카테고리별 단독 분류 수 (threshold별)")
    print("=" * 80)
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))

    # ── 저장 ─────────────────────────────────────────────
    sweep_df.to_csv(OUTPUT_CSV.replace('.csv', '_summary.csv'),
                    index=False, encoding='utf-8-sig')
    pd.DataFrame(detail_rows).to_csv(OUTPUT_CSV,
                                      index=False, encoding='utf-8-sig')
    print(f"\n[저장] {OUTPUT_CSV.replace('.csv','_summary.csv')}  (요약)")
    print(f"[저장] {OUTPUT_CSV}  (전체 상세)")


if __name__ == "__main__":
    main()
