"""
논문 카테고리 분류 스크립트 - 논문 키워드 매칭 버전
=====================================
법률 정의 조항 키워드와 논문 자체 키워드(KCI 메타데이터)를 매칭하여 카테고리 분류

[입력 파일]
- 00.KCI_인공_논문_상세_및_인용데이터.csv (00.fetch_kci_details.py 출력)
  컬럼: source_id, title, pub_year, abstract, keywords, category, target_arti_id, ref_type
  → source_id 기준으로 중복 제거 후 분류

[실험 결과 및 한계]
- 기초법(키워드없음) 47.0% (482/1025편) — 제목+초록 버전(12.7%)보다 훨씬 높음
- 원인: 법률 키워드 사전은 법령 정의 조항 기반(개인정보처리자 등 구체적 법적 용어),
        논문 저자 키워드는 연구 개념어(설명가능성, 투명성, 진보성 등) 중심 → 집합이 거의 안 겹침
- 향후 과제: 법률 키워드 사전을 개념어 중심으로 재구성하면 커버리지 개선 가능

[분류 규칙 - A안 5단계]
1. 제목에 '리걸테크' / '자율무기' 포함 → 해당 특수 라벨
2. 키워드 매칭 0개 → '기초법(키워드없음)'
3. 최고 카테고리 비율 ≥ THRESHOLD_SINGLE(50%) → 단독 분류
4. THRESHOLD_MULTI(25%) 이상 카테고리 2개 이상 → '융복합' (최대 4개)
5. THRESHOLD_MULTI(25%) 이상 카테고리 1개 → 단독 분류
6. top 카테고리 비율 ≥ 15% → 단독 분류, 미만 → '기초법(임계값미달)'

[키워드 처리]
- 매칭 대상: 논문 자체 키워드(KCI 메타데이터)만 사용 (제목/초록 전문 제외)
- 순수 영어(한글 없음) 법률 키워드는 매칭에서 제외
- 동일 논문에서 같은 키워드는 1회만 카운트
- KEYWORD_MIN_LEN(3) 미만 한글 키워드 제외

사용법:
    python 03.classify_papers_논문키워드버전.py
"""

import re
import ahocorasick
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

KEYWORD_CSV = "00.법령_키워드_TFIDF_v2.csv"
PAPER_CSV   = "00.KCI_인공_논문_상세_및_인용데이터.csv"
OUTPUT_CSV  = "03.논문_카테고리_분류결과_tfidf_논문키워드버전.csv"

THRESHOLD_SINGLE = 0.50   # 단독 카테고리 확정 기준 (50%, 과반수)
THRESHOLD_MULTI  = 0.25   # 융복합 참여 기준 (25%)

KEYWORD_MIN_LEN  = 3      # 한글 키워드 최소 길이 (2글자 이하 일반 명사 제거)


# ─────────────────────────────────────────────
# 🔧 키워드 사전 구축
# ─────────────────────────────────────────────

def is_pure_english(keyword: str) -> bool:
    """한글 글자가 없으면 순수 영어로 간주"""
    return not bool(re.search(r'[가-힣]', keyword))


def build_category_keywords(keyword_csv: str) -> dict[str, set[str]]:
    """카테고리 → 키워드 집합 딕셔너리 생성 (순수 영어 제외, 중복 제거)"""
    df = pd.read_csv(keyword_csv, encoding="utf-8-sig")
    df = df[df['카테고리'].notna() & (df['키워드 목록'].notna())]
    df = df[df['키워드 목록'] != '']

    cat_keywords = defaultdict(set)
    for _, row in df.iterrows():
        cat = row['카테고리']
        for kw in str(row['키워드 목록']).split(','):
            kw = kw.strip()
            if kw and not is_pure_english(kw) and len(kw) >= KEYWORD_MIN_LEN:
                cat_keywords[cat].add(kw)

    return dict(cat_keywords)


def build_automaton(cat_keywords: dict[str, set[str]]):
    """Aho-Corasick 오토마톤 생성: 모든 키워드를 한 번에 탐색"""
    kw_to_cats = defaultdict(set)
    for cat, keywords in cat_keywords.items():
        for kw in keywords:
            kw_to_cats[kw].add(cat)

    A = ahocorasick.Automaton()
    for kw, cats in kw_to_cats.items():
        A.add_word(kw, (cats, kw))
    A.make_automaton()
    return A


# ─────────────────────────────────────────────
# 📊 논문 분류 로직
# ─────────────────────────────────────────────

def classify_paper(title: str, paper_keywords: str,
                   automaton) -> tuple[str, dict, str]:
    """
    Args:
        title:          논문 제목 (특수 라벨 판별에만 사용)
        paper_keywords: KCI 논문 자체 키워드 (매칭 대상 텍스트)
    Returns:
        (최종분류, {카테고리: 매칭키워드수}, 매칭_키워드_목록_문자열)
    """
    title          = str(title)          if pd.notna(title)          else ""
    paper_keywords = str(paper_keywords) if pd.notna(paper_keywords) else ""

    # 규칙 1: 제목 특수 키워드 우선
    for special in ['리걸테크', '자율무기']:
        if special in title:
            return special, {}, ""

    # 매칭 대상: 논문 자체 키워드만 사용
    text = paper_keywords

    # Aho-Corasick으로 한 번에 모든 키워드 탐색
    matched_by_cat = defaultdict(set)
    for _, (cats, kw) in automaton.iter(text):
        for cat in cats:
            matched_by_cat[cat].add(kw)

    cat_counts  = {cat: len(kws) for cat, kws in matched_by_cat.items()}
    matched_kws = [f"{kw}({cat})" for cat, kws in matched_by_cat.items() for kw in kws]

    total = sum(cat_counts.values())

    # Step 2: 키워드 미매칭 → 기초법(키워드없음)
    if total == 0:
        return "기초법(키워드없음)", {}, ""

    # 비율 계산
    cat_ratios = {cat: cnt / total for cat, cnt in cat_counts.items()}
    top_cat    = max(cat_ratios, key=cat_ratios.get)

    # Step 3: 최고 카테고리 ≥ 50% → 단독 분류
    if cat_ratios[top_cat] >= THRESHOLD_SINGLE:
        return top_cat, cat_counts, ", ".join(matched_kws[:20])

    # Step 4: 25% 이상 카테고리 2개+ → 융복합 (최대 4개, 가나다순)
    above_multi = sorted(
        [cat for cat, r in cat_ratios.items() if r >= THRESHOLD_MULTI],
        key=lambda c: cat_ratios[c], reverse=True
    )[:4]
    if len(above_multi) >= 2:
        label = f"융복합({'|'.join(sorted(above_multi))})"
        return label, cat_counts, ", ".join(matched_kws[:20])

    # Step 5: 25% 이상 카테고리 1개 → 단독 분류 (A안)
    if len(above_multi) == 1:
        return above_multi[0], cat_counts, ", ".join(matched_kws[:20])

    # Step 6: top 카테고리 비율 ≥ 15% → 단독 분류, 미만 → 기초법(임계값미달)
    top_ratio = cat_ratios[top_cat]
    if top_ratio >= 0.15:
        return top_cat, cat_counts, ", ".join(matched_kws[:20])
    return "기초법(임계값미달)", cat_counts, ", ".join(matched_kws[:20])


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  논문 카테고리 분류 (논문 키워드 매칭 버전)")
    print("=" * 60)

    # 키워드 사전 로드 & 오토마톤 구축
    cat_keywords = build_category_keywords(KEYWORD_CSV)
    total_kw = sum(len(v) for v in cat_keywords.values())
    print(f"[키워드 사전] {len(cat_keywords)}개 카테고리 / 총 {total_kw}개 키워드")
    for cat, kws in sorted(cat_keywords.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<12} {len(kws):>4}개")

    print("\n[오토마톤] Aho-Corasick 구축 중...", end=" ")
    automaton = build_automaton(cat_keywords)
    print("완료")

    # 논문 데이터 로드 (source_id 기준 중복 제거 — 인용 행 제거)
    raw = pd.read_csv(PAPER_CSV, encoding="utf-8-sig")
    papers = raw.drop_duplicates(subset="source_id").reset_index(drop=True)
    print(f"[논문] {len(raw)}행 로드 → 중복 제거 후 {len(papers)}편\n")

    # 분류 실행
    results = []
    for _, row in tqdm(papers.iterrows(), total=len(papers), desc="분류 중", unit="편"):
        label, cat_counts, matched = classify_paper(
            row.get('title', ''),
            row.get('keywords', ''),
            automaton,
        )
        results.append({
            "논문ID":       row.get('source_id', ''),
            "제목":         row.get('title', ''),
            "발행연도":     row.get('pub_year', ''),
            "논문키워드":   row.get('keywords', ''),
            "분류결과":     label,
            "매칭_키워드":  matched,
        })

    df_out = pd.DataFrame(results)

    # 결과 요약
    print("\n" + "=" * 60)
    print("  분류 결과 요약")
    print("=" * 60)
    vc = df_out['분류결과'].value_counts()
    total = len(df_out)
    for cat, cnt in vc.items():
        print(f"  {cat:<20} {cnt:>6}편  ({cnt/total*100:.1f}%)")

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {OUTPUT_CSV}")
    print("완료!")


if __name__ == "__main__":
    main()
