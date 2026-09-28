"""
04z.table2_rebuild.py
=====================
원고 표 2(키워드 공출현 네트워크 연구 주제군) 재생성

기존 표에 '최다 분과(비중)' 열을 추가한다. 4.2절이 주장하는 분과별 분포를
독자가 표만 보고 검증할 수 있게 하는 것이 목적이다.

전제:
    04.keyword_build.py → 04.keyword_leiden.py 를 먼저 실행해
    results/04_keyword_network/ 에 아래 두 파일이 있어야 한다.
        community_paper_map.csv    논문 → 주제군 배정 (1,337편)
        keyword_communities.csv    키워드 → 주제군 배정 (228개)

입력 (같은 폴더. 심볼릭 링크 가능):
    KCI_AI_논문_카테고리_확정.csv
    KCI_AI_논문_카테고리_검토필요.csv

출력:
    results/04z_table2/table2_주제군.csv     원자료
    results/04z_table2/table2_주제군.xlsx    한글 붙여넣기용
    results/04z_table2/table2_주제군.md      마크다운 표
    results/04z_table2/table2_분과분포_상세.csv  군집×분과 교차표 (검증용)

사용법:
    cd 2025인용_재수집/분석_2025
    python 04z.table2_rebuild.py

옵션:
    TOP_KEYWORDS   대표 키워드 개수 (기본 8)
    LABEL_OVERRIDE 원고에서 쓰는 주제군 명칭. 비워두면 자동 라벨을 그대로 쓴다.
"""

import os
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
KW = BASE / 'results' / '04_keyword_network'
OUT = BASE / 'results' / '04z_table2'
OUT.mkdir(parents=True, exist_ok=True)

TOP_KEYWORDS = 8

# 융복합 카테고리 처리 방식. 결과가 꽤 달라지므로 원고 3장에 반드시 밝힐 것.
#   True  : 융복합(A+B) → A 로 접어 전통 분과에 합산 (03.network_build.py와 동일)
#   False : 융복합(A+B) 를 독립 분과로 취급 (community_summary.txt와 동일)
# 두 방식의 결과는 실행 시 모두 출력되므로 비교한 뒤 정하면 된다.
COLLAPSE_FUSION = True

# 자동 라벨 → 원고 표기. 원고를 고치면 여기도 고칠 것.
LABEL_OVERRIDE = {
    '생성형 인공지능 / 저작권 / 공정이용': '생성형 인공지능·저작권·공정이용',
    '빅데이터 / 개인정보 / 개인정보보호': '빅데이터·개인정보',
    '4차 산업혁명 / 블록체인 / 지능정보사회': '4차 산업혁명·지능정보사회',
    '알고리즘 / 머신러닝 / 딥러닝': '알고리즘·머신러닝·투명성',
    'EU AI Act / 인공지능기본법 / 인공지능법': 'EU AI Act·인공지능기본법',
    '자율주행자동차 / 규제 / 책임': '자율주행자동차·규제·책임',
    '법인격 / 로봇 / 형사책임': '법인격·로봇·형사책임',
    '제조물책임 / 위험책임 / 소프트웨어': '제조물책임·위험책임·소프트웨어',
    '딥페이크 / 표현의 자유 / 인격권': '딥페이크·표현의 자유·인격권',
    '리걸테크 / 변호사법': '리걸테크·변호사법',
    '자율운항선박 / Maritime Autonomous Surface Ships / 원격운항자': '자율운항선박·원격운항자',
    '발명 / 발명자 / 특허법': '발명·발명자·특허법',
    '중국 / China': '중국·China',
    '부정경쟁방지법 / Unfair Competition Prevention Act': '부정경쟁방지법',
}


def find_file(keyword):
    for f in os.listdir(BASE):
        if keyword in unicodedata.normalize('NFC', f):
            return BASE / f
    raise FileNotFoundError(f'파일을 찾을 수 없습니다: {keyword}')


def primary_category(cat):
    """융복합(A+B) → A. 최다 분과 집계를 전통 분과 기준으로 맞춘다."""
    cat = str(cat).strip()
    if cat.startswith('융복합(') and cat.endswith(')'):
        return cat[4:-1].split('+')[0].strip()
    return cat


# ── 1. 로드 ────────────────────────────────────────────────────────
print('▶ 로드 중...')
pmap = pd.read_csv(KW / 'community_paper_map.csv', encoding='utf-8-sig')
kcom = pd.read_csv(KW / 'keyword_communities.csv', encoding='utf-8-sig')

cols = ['논문ID', '최종_카테고리']
cat = pd.concat([
    pd.read_csv(find_file('카테고리_확정'), encoding='utf-8-sig')[cols],
    pd.read_csv(find_file('검토필요'), encoding='utf-8-sig')[cols],
], ignore_index=True).drop_duplicates(subset='논문ID')

pmap['논문ID'] = pmap['논문ID'].astype(str)
cat['논문ID'] = cat['논문ID'].astype(str)
df = pmap.merge(cat, on='논문ID', how='left')
df['분과_접음'] = df['최종_카테고리'].apply(primary_category)
df['분과_원본'] = df['최종_카테고리'].astype(str)
df['주분과'] = df['분과_접음'] if COLLAPSE_FUSION else df['분과_원본']

print(f'  매핑 논문 {len(df):,}편 / 주제군 {df["community_label"].nunique()}개')
if df['최종_카테고리'].isna().any():
    print(f'  [경고] 카테고리 결측 {int(df["최종_카테고리"].isna().sum())}편')


# ── 2. 주제군별 집계 ───────────────────────────────────────────────
rows = []
for label, g in df.groupby('community_label'):
    n = len(g)
    dist = g['주분과'].value_counts()
    top1, top1_n = dist.index[0], int(dist.iloc[0])
    top2 = f'{dist.index[1]} {int(dist.iloc[1])}' if len(dist) > 1 else '-'

    kws = (kcom[kcom['community_label'] == label]
           .sort_values('freq', ascending=False)
           .head(TOP_KEYWORDS)['keyword'].tolist())

    rows.append({
        '연구 주제군': LABEL_OVERRIDE.get(label, label),
        '논문 수': n,
        '최다 분과': top1,
        '최다 분과 비중': f'{top1_n / n * 100:.0f}%',
        '최다 분과(비중)': f'{top1} ({top1_n / n * 100:.0f}%)',
        '2위 분과': top2,
        '대표 키워드': ', '.join(kws),
        '_자동라벨': label,
    })

t = pd.DataFrame(rows).sort_values('논문 수', ascending=False).reset_index(drop=True)


# ── 3. 저장 ────────────────────────────────────────────────────────
final_cols = ['연구 주제군', '논문 수', '최다 분과(비중)', '대표 키워드']
t[final_cols].to_csv(OUT / 'table2_주제군.csv', index=False, encoding='utf-8-sig')
try:
    t[final_cols].to_excel(OUT / 'table2_주제군.xlsx', index=False)
except Exception as e:
    print(f'  [건너뜀] xlsx 저장 실패: {e}')

# 검증용 상세 교차표
cross = pd.crosstab(df['community_label'], df['주분과'])
cross['합계'] = cross.sum(axis=1)
cross.sort_values('합계', ascending=False).to_csv(
    OUT / 'table2_분과분포_상세.csv', encoding='utf-8-sig')

# 마크다운
md = ['| 연구 주제군 | 논문 수 | 최다 분과(비중) | 대표 키워드 |',
      '|---|---:|---|---|']
for _, r in t.iterrows():
    md.append(f'| {r["연구 주제군"]} | {r["논문 수"]} | {r["최다 분과(비중)"]} | {r["대표 키워드"]} |')
(OUT / 'table2_주제군.md').write_text('\n'.join(md), encoding='utf-8')


# ── 4. 4.2절 서술용 분과 블록 집계 ──────────────────────────────────
print('\n' + '=' * 78)
print(t[['연구 주제군', '논문 수', '최다 분과(비중)', '2위 분과']].to_string(index=False))

total = int(t['논문 수'].sum())

for mode, col in [('융복합을 첫 분과로 접음', '분과_접음'),
                  ('융복합을 독립 분과로 둠', '분과_원본')]:
    print('\n' + '=' * 78)
    mark = ' ← 현재 설정' if (col == '분과_접음') == COLLAPSE_FUSION else ''
    print(f'  최다 분과별 블록 [{mode}]{mark}')
    print('=' * 78)
    tops = (df.groupby('community_label')[col]
            .agg(lambda s: s.value_counts().index[0]))
    sizes = df.groupby('community_label').size()
    tmp = pd.DataFrame({'분과': tops, 'n': sizes})
    tmp['라벨'] = [LABEL_OVERRIDE.get(i, i) for i in tmp.index]
    for k, g in sorted(tmp.groupby('분과'), key=lambda x: -x[1]['n'].sum()):
        v = int(g['n'].sum())
        print(f'  {k:22} {v:5}편 ({v / total * 100:4.1f}%)  ← {", ".join(g["라벨"])}')

print(f'\n  합계 {total:,}편')
print('\n  ※ 두 방식의 결과가 다르다. 어느 쪽을 쓸지 정하고 3장에 명시할 것.')
print(f'\n[저장] {OUT}')
