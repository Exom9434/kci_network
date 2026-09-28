"""
AI 법학 논문 카테고리 자동 분류 (A안)
====================================
키워드 사전(00.법률별_키워드_카테고리.csv)을 기반으로
논문 제목·초록에서 카테고리를 자동 분류합니다.

출력: 03.논문_카테고리_분류결과.csv
"""

import pandas as pd
import ahocorasick
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────────────────────────────────
KEYWORD_CSV      = "00.법률별_키워드_카테고리.csv"
PAPER_CSV        = "AI_법학_논문목록.csv"
OUTPUT_CSV       = "03.논문_카테고리_분류결과.csv"

THRESHOLD_SINGLE = 0.50   # 단독 분류 임계값 (단일 카테고리만 매칭 시)
THRESHOLD_MULTI  = 0.25   # 융복합 분류 임계값 (복수 카테고리 매칭 시)
KEYWORD_MIN_LEN  = 3      # 키워드 최소 길이

# 제목에 해당 키워드가 포함되면 다른 매칭보다 우선 적용
TITLE_PRIORITY   = ['리걸테크', '자율무기']

# ── 키워드 사전 로드 ──────────────────────────────────────────────────────────
kw_df = pd.read_csv(KEYWORD_CSV, encoding='utf-8-sig')
category_keywords = {}
for _, row in kw_df.iterrows():
    cat = row['카테고리']
    if pd.isna(cat) or pd.isna(row['키워드 목록']):
        continue
    for k in str(row['키워드 목록']).split(','):
        k = k.strip()
        if len(k) >= KEYWORD_MIN_LEN:
            category_keywords.setdefault(cat, set()).add(k)

print("카테고리별 키워드 수:")
for cat, kws in sorted(category_keywords.items()):
    print(f"  {cat}: {len(kws)}개")

# ── Aho-Corasick 자동화 구축 ──────────────────────────────────────────────────
A = ahocorasick.Automaton()
for cat, kws in category_keywords.items():
    for kw in kws:
        A.add_word(kw, (kw, cat))
A.make_automaton()

# ── 논문 분류 ─────────────────────────────────────────────────────────────────
papers = pd.read_csv(PAPER_CSV, encoding='utf-8-sig')
results = []

for _, row in tqdm(papers.iterrows(), total=len(papers)):
    title = str(row.get('제목', ''))
    text  = title + ' ' + str(row.get('초록', ''))

    # 규칙 1: 제목 특수 키워드 우선
    special_hit = None
    for special in TITLE_PRIORITY:
        if special in title:
            special_hit = special
            break

    if special_hit:
        label   = special_hit
        matched = f'{special_hit}(제목직접매칭)'
    else:
        # 규칙 2: 키워드 매칭 + 비율 기반 분류
        cat_matches = {}
        for _, (kw, cat) in A.iter(text):
            cat_matches.setdefault(cat, set()).add(kw)

        total_kw = sum(len(v) for v in cat_matches.values())

        if total_kw == 0:
            label   = '기초법(키워드없음)'
            matched = ''
        else:
            ratios   = {c: len(v) / total_kw for c, v in cat_matches.items()}
            threshold = THRESHOLD_SINGLE if len(ratios) == 1 else THRESHOLD_MULTI
            top_cats  = [c for c, r in ratios.items() if r >= threshold]

            if not top_cats:
                label = '기초법(임계값미달)'
            elif len(top_cats) == 1:
                label = top_cats[0]
            else:
                label = '융복합(' + '|'.join(sorted(top_cats)) + ')'

            matched = ', '.join(
                f'{k}({c})' for c, kws in cat_matches.items() for k in kws
            )

    results.append({
        '논문ID':    row['논문ID'],
        '제목':      row['제목'],
        '발행연도':  row.get('발행연도', ''),
        '학술지명':  row.get('학술지명', ''),
        '분류결과':  label,
        '매칭_키워드': matched,
    })

# ── 저장 ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"\n저장 완료: {OUTPUT_CSV}")
print(f"총 {len(out)}편\n")
print("=== 분류 결과 상위 20 ===")
print(out['분류결과'].value_counts().head(20).to_string())
