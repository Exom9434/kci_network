"""
03-2.network_build.py
=====================
확장 인용 네트워크 구축 — AI 법학 논문 + 피인용 KCI 논문

[네트워크 구조]
  노드 타입 A (node_type='ai'):
      KCI AI 법학 논문 2,014편 — 카테고리 분류 정보 보유
  노드 타입 B (node_type='kci'):
      AI 논문이 인용한 비AI KCI 법학 논문 — 메타데이터만 보유, 카테고리 없음

  엣지:
      A→A : AI 논문끼리의 내부 인용
      A→B : AI 논문이 비AI KCI 논문을 인용

  ※ B→B 엣지는 포함하지 않음 (B 논문 자체의 인용 관계는 미추적)

- 입력:
    00.KCI_AI_논문_상세_및_인용데이터.csv
    KCI_AI_논문_카테고리_확정.csv
    KCI_AI_논문_카테고리_검토필요.csv
    전체_법학_논문목록_중복제거.csv

- 출력:
    results/03-2_network/network_extended.graphml     전체 확장 네트워크
    results/03-2_network/network_extended_giant.graphml Giant component
    results/03-2_network/network_stats.txt            기본 통계
    results/03-2_network/degree_distribution.png      Degree 분포

사용법:
    python 03-2.network_build.py
"""

import os
import unicodedata
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

# ── 한글 폰트 ─────────────────────────────────────────────────────
_font_candidates = [
    # macOS
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/Library/Fonts/NanumGothic.ttf',
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    # Linux
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]
_font_path = next((p for p in _font_candidates if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUT  = BASE / 'results' / '03-2_network'
OUT.mkdir(parents=True, exist_ok=True)

def find_file(keyword):
    for f in os.listdir(BASE):
        if keyword in unicodedata.normalize('NFC', f):
            return BASE / f
    raise FileNotFoundError(f'파일을 찾을 수 없습니다: {keyword}')

CATEGORIES = [
    '공법', '민사법', '형사법', '경제법', '노동법',
    '사회보장법', '금융법', '의료법', '조세법',
    '데이터법', '인공지능법', '지식재산권법', '기초법',
    '리걸테크', '자율무기법', '자율주행자동차법', '해외법',
]


# ════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════════
print('=' * 60)
print('  03-2.network_build.py — 확장 인용 네트워크 구축')
print('=' * 60)

print('\n▶ 데이터 로드 중...')

# AI 법학 논문 카테고리
df_ok = pd.read_csv(find_file('카테고리_확정'), encoding='utf-8-sig')
rv_path = next(
    BASE / f for f in os.listdir(BASE)
    if '검토필요' in unicodedata.normalize('NFC', f)
)
df_rv = pd.read_csv(rv_path, encoding='utf-8-sig')

df_ai = pd.concat([
    df_ok[['논문ID', '제목', '발행연도', '최종_카테고리']],
    df_rv[['논문ID', '제목', '발행연도', '최종_카테고리']],
], ignore_index=True)
df_ai['논문ID'] = df_ai['논문ID'].astype(str)

def primary_category(cat):
    cat = str(cat).strip()
    if cat.startswith('융복합(') and cat.endswith(')'):
        return cat[4:-1].split('+')[0].strip()
    return cat

df_ai['주카테고리'] = df_ai['최종_카테고리'].apply(primary_category)
ai_ids = set(df_ai['논문ID'])

# 전체 KCI 법학 논문 메타데이터
df_all = pd.read_csv(find_file('전체_법학_논문목록_중복'), encoding='utf-8-sig')
df_all['논문ID'] = df_all['논문ID'].astype(str)
all_kci_ids = set(df_all['논문ID'])
kci_meta = df_all.set_index('논문ID')

# 인용 데이터
df_cite = pd.read_csv(find_file('인용데이터'), encoding='utf-8-sig')
df_cite['source_id']      = df_cite['source_id'].astype(str)
df_cite['target_arti_id'] = df_cite['target_arti_id'].fillna('').astype(str)

print(f'  AI 법학 논문:           {len(ai_ids):,}편')
print(f'  전체 KCI 법학 논문:     {len(all_kci_ids):,}편')
print(f'  전체 인용 레코드:       {len(df_cite):,}건')


# ════════════════════════════════════════════════════════════════════
# 2. 엣지 추출
# ════════════════════════════════════════════════════════════════════
print('\n▶ 엣지 추출 중...')

# A→A: AI 내부 인용
edges_aa = df_cite[
    df_cite['source_id'].isin(ai_ids) &
    df_cite['target_arti_id'].isin(ai_ids) &
    (df_cite['source_id'] != df_cite['target_arti_id'])
][['source_id', 'target_arti_id']].drop_duplicates()

# A→B: AI → 비AI KCI
edges_ab = df_cite[
    df_cite['source_id'].isin(ai_ids) &
    df_cite['target_arti_id'].isin(all_kci_ids) &
    ~df_cite['target_arti_id'].isin(ai_ids) &
    (df_cite['target_arti_id'] != '')
][['source_id', 'target_arti_id']].drop_duplicates()

print(f'  A→A (AI 내부 인용):     {len(edges_aa):,}건')
print(f'  A→B (AI→비AI KCI):     {len(edges_ab):,}건')
print(f'  합계:                   {len(edges_aa) + len(edges_ab):,}건')

# 비AI KCI 노드 집합
ext_ids = set(edges_ab['target_arti_id'])
print(f'\n  비AI KCI 피인용 노드:   {len(ext_ids):,}편')
print(f'  확장 네트워크 총 노드:  {len(ai_ids) + len(ext_ids):,}편')


# ════════════════════════════════════════════════════════════════════
# 3. 네트워크 구축
# ════════════════════════════════════════════════════════════════════
print('\n▶ 네트워크 구축 중...')

G = nx.Graph()

# 타입 A 노드 (AI 논문)
ai_meta = df_ai.set_index('논문ID')
for pid in ai_ids:
    row = ai_meta.loc[pid] if pid in ai_meta.index else {}
    G.add_node(pid,
               node_type='ai',
               title=str(row.get('제목', '')),
               year=int(row.get('발행연도', 0)),
               category=str(row.get('최종_카테고리', '')),
               primary_category=str(row.get('주카테고리', '')))

# 타입 B 노드 (비AI KCI 논문)
for pid in ext_ids:
    row = kci_meta.loc[pid] if pid in kci_meta.index else {}
    G.add_node(pid,
               node_type='kci',
               title=str(row.get('제목', '')),
               year=int(row.get('발행연도', 0)) if pd.notna(row.get('발행연도', None)) else 0,
               category='',
               primary_category='')

# 엣지 추가
for _, row in pd.concat([edges_aa, edges_ab]).iterrows():
    s, t = row['source_id'], row['target_arti_id']
    if G.has_edge(s, t):
        G[s][t]['weight'] += 1
    else:
        G.add_edge(s, t, weight=1)

print(f'  전체 노드: {G.number_of_nodes():,}  (AI: {len(ai_ids):,} / KCI: {len(ext_ids):,})')
print(f'  전체 엣지: {G.number_of_edges():,}')
print(f'  네트워크 밀도: {nx.density(G):.6f}')


# ════════════════════════════════════════════════════════════════════
# 4. Giant Component
# ════════════════════════════════════════════════════════════════════
print('\n▶ Giant Component 분석 중...')

components  = sorted(nx.connected_components(G), key=len, reverse=True)
isolated    = sum(1 for c in components if len(c) == 1)
giant_nodes = components[0]
G_giant     = G.subgraph(giant_nodes).copy()

# giant 내 AI/KCI 비율
giant_ai  = sum(1 for n in giant_nodes if G.nodes[n].get('node_type') == 'ai')
giant_kci = sum(1 for n in giant_nodes if G.nodes[n].get('node_type') == 'kci')

print(f'  연결 컴포넌트 수:   {len(components):,}')
print(f'  고립 노드:          {isolated:,}편 ({isolated/G.number_of_nodes()*100:.1f}%)')
print(f'  Giant component:    {G_giant.number_of_nodes():,}노드 / {G_giant.number_of_edges():,}엣지')
print(f'    └ AI 논문:        {giant_ai:,}편 ({giant_ai/G_giant.number_of_nodes()*100:.1f}%)')
print(f'    └ 비AI KCI 논문:  {giant_kci:,}편 ({giant_kci/G_giant.number_of_nodes()*100:.1f}%)')

comp_sizes = Counter(len(c) for c in components)
print(f'\n  [컴포넌트 크기 분포]')
for size in sorted(comp_sizes.keys(), reverse=True)[:8]:
    print(f'    크기 {size:5d}: {comp_sizes[size]:3d}개')


# ════════════════════════════════════════════════════════════════════
# 5. 기본 통계
# ════════════════════════════════════════════════════════════════════
print('\n▶ 네트워크 통계 계산 중...')

degrees    = [d for _, d in G_giant.degree()]
pagerank   = nx.pagerank(G_giant, weight='weight')
clustering = nx.clustering(G_giant)

# AI 논문 한정 PageRank 상위 10
pr_ai_top10 = sorted(
    [(n, pr) for n, pr in pagerank.items()
     if G_giant.nodes[n].get('node_type') == 'ai'],
    key=lambda x: -x[1]
)[:10]

# 비AI KCI 논문 PageRank 상위 10 (어떤 외부 논문이 많이 인용됐는지)
pr_kci_top10 = sorted(
    [(n, pr) for n, pr in pagerank.items()
     if G_giant.nodes[n].get('node_type') == 'kci'],
    key=lambda x: -x[1]
)[:10]

stats_lines = [
    '=' * 60,
    '  KCI AI 법학 논문 확장 인용 네트워크 기본 통계',
    '=' * 60,
    '',
    '[전체 확장 네트워크]',
    f'  AI 논문 노드:         {len(ai_ids):>6,}편',
    f'  비AI KCI 논문 노드:   {len(ext_ids):>6,}편',
    f'  총 노드:              {G.number_of_nodes():>6,}',
    f'  총 엣지:              {G.number_of_edges():>6,}건',
    f'  네트워크 밀도:        {nx.density(G):.6f}',
    f'  연결 컴포넌트:        {len(components):>6,}개',
    f'  고립 노드:            {isolated:>6,}편 ({isolated/G.number_of_nodes()*100:.1f}%)',
    '',
    '[Giant Component]',
    f'  전체 노드:            {G_giant.number_of_nodes():>6,} ({G_giant.number_of_nodes()/G.number_of_nodes()*100:.1f}%)',
    f'  AI 논문:              {giant_ai:>6,}편',
    f'  비AI KCI 논문:        {giant_kci:>6,}편',
    f'  엣지 수:              {G_giant.number_of_edges():>6,}건',
    f'  평균 degree:          {np.mean(degrees):.2f}',
    f'  최대 degree:          {max(degrees)}',
    f'  평균 군집계수:         {np.mean(list(clustering.values())):.4f}',
    '',
    '[PageRank 상위 10 — AI 논문]',
]
for rank, (pid, pr) in enumerate(pr_ai_top10, 1):
    title = str(G_giant.nodes[pid].get('title', pid))[:40]
    year  = G_giant.nodes[pid].get('year', '?')
    cat   = G_giant.nodes[pid].get('primary_category', '?')
    stats_lines.append(f'  {rank:2d}. [{year}] {title}')
    stats_lines.append(f'      카테고리: {cat}  PageRank: {pr:.5f}')

stats_lines += ['', '[PageRank 상위 10 — 비AI KCI 논문 (많이 인용된 외부 논문)]']
for rank, (pid, pr) in enumerate(pr_kci_top10, 1):
    title = str(G_giant.nodes[pid].get('title', pid))[:40]
    year  = G_giant.nodes[pid].get('year', '?')
    stats_lines.append(f'  {rank:2d}. [{year}] {title}')
    stats_lines.append(f'      PageRank: {pr:.5f}')

stats_text = '\n'.join(stats_lines)
print(stats_text)

with open(OUT / 'network_stats.txt', 'w', encoding='utf-8') as f:
    f.write(stats_text)
print(f'\n  [저장] network_stats.txt')


# ════════════════════════════════════════════════════════════════════
# 6. Degree 분포
# ════════════════════════════════════════════════════════════════════
print('\n▶ Degree 분포 시각화 중...')

deg_ai  = [d for n, d in G_giant.degree()
           if G_giant.nodes[n].get('node_type') == 'ai']
deg_kci = [d for n, d in G_giant.degree()
           if G_giant.nodes[n].get('node_type') == 'kci']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(deg_ai,  bins=30, color='steelblue', alpha=0.75, label='AI 논문',    edgecolor='white')
axes[0].hist(deg_kci, bins=30, color='tomato',    alpha=0.60, label='비AI KCI', edgecolor='white')
axes[0].set_title('Degree 분포 (Giant Component)')
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('논문 수')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

all_deg_cnt = Counter(degrees)
x_v = sorted(all_deg_cnt.keys())
y_v = [all_deg_cnt[d] for d in x_v]
axes[1].scatter(x_v, y_v, color='steelblue', s=20, alpha=0.7)
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_title('Degree 분포 (로그 스케일)')
axes[1].set_xlabel('Degree (log)'); axes[1].set_ylabel('논문 수 (log)')
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'degree_distribution.png', dpi=150)
plt.close(fig)
print('  [저장] degree_distribution.png')


# ════════════════════════════════════════════════════════════════════
# 7. GraphML 저장
# ════════════════════════════════════════════════════════════════════
print('\n▶ GraphML 저장 중...')
nx.write_graphml(G,       str(OUT / 'network_extended.graphml'))
nx.write_graphml(G_giant, str(OUT / 'network_extended_giant.graphml'))
print(f'  [저장] network_extended.graphml        ({G.number_of_nodes()}노드)')
print(f'  [저장] network_extended_giant.graphml  ({G_giant.number_of_nodes()}노드)')

print('\n' + '=' * 60)
print('  완료! 다음: python 03-2.network_leiden.py')
print('=' * 60)
