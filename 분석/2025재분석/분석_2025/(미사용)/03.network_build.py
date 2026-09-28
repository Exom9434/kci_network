"""
03.network_build.py
===================
KCI AI 법학 논문 인용 네트워크 구축 및 기본 통계

- 입력:
    00.KCI_AI_논문_상세_및_인용데이터.csv  (인용 데이터)
    KCI_AI_논문_카테고리_확정.csv           (LLM 확정 분류)
    KCI_AI_논문_카테고리_검토필요.csv       (수동 검토 분류)

- 처리:
    1. 분석 대상 2,014편 + 카테고리 매핑
    2. 내부 인용 엣지 (양쪽 모두 분석 대상) 추출
    3. Giant component 추출
    4. 기본 네트워크 통계 출력
    5. 네트워크를 GraphML로 저장 (03.network_leiden.py에서 재사용)

- 출력:
    results/03_network/network_full.graphml       전체 (고립 노드 포함)
    results/03_network/network_giant.graphml      Giant component만
    results/03_network/network_stats.txt          기본 통계 요약
    results/03_network/degree_distribution.png    Degree 분포

사용법:
    python 03.network_build.py
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
BASE = Path(__file__).parent          # 스크립트가 있는 폴더 기준
OUT  = BASE / 'results' / '03_network'
OUT.mkdir(parents=True, exist_ok=True)

def find_file(keyword):
    for f in os.listdir(BASE):
        if keyword in unicodedata.normalize('NFC', f):
            return BASE / f
    raise FileNotFoundError(f'파일을 찾을 수 없습니다: {keyword}')

# ── 카테고리 정의 ─────────────────────────────────────────────────
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
print('  03.network_build.py — 인용 네트워크 구축')
print('=' * 60)

print('\n▶ 데이터 로드 중...')

# 카테고리 분류 결과 (확정 + 수동검토)
df_ok = pd.read_csv(find_file('카테고리_확정'), encoding='utf-8-sig')
df_rv_path = None
for f in os.listdir(BASE):
    if '검토필요' in unicodedata.normalize('NFC', f):
        df_rv_path = BASE / f
        break
df_rv = pd.read_csv(df_rv_path, encoding='utf-8-sig')

df_papers = pd.concat([
    df_ok[['논문ID', '제목', '발행연도', '최종_카테고리']],
    df_rv[['논문ID', '제목', '발행연도', '최종_카테고리']],
], ignore_index=True)
df_papers['논문ID'] = df_papers['논문ID'].astype(str)

# 카테고리에서 기본(1차) 카테고리 추출 (융복합이면 첫 번째)
def primary_category(cat):
    cat = str(cat).strip()
    if cat.startswith('융복합(') and cat.endswith(')'):
        return cat[4:-1].split('+')[0].strip()
    return cat

df_papers['주카테고리'] = df_papers['최종_카테고리'].apply(primary_category)

# ID → 메타데이터 매핑
id2title = df_papers.set_index('논문ID')['제목'].to_dict()
id2year  = df_papers.set_index('논문ID')['발행연도'].to_dict()
id2cat   = df_papers.set_index('논문ID')['최종_카테고리'].to_dict()
id2pcat  = df_papers.set_index('논문ID')['주카테고리'].to_dict()
all_ids  = set(df_papers['논문ID'])

print(f'  분석 대상 논문: {len(all_ids):,}편')

# 인용 데이터
df_cite = pd.read_csv(find_file('인용데이터'), encoding='utf-8-sig')
df_cite['source_id']      = df_cite['source_id'].astype(str)
df_cite['target_arti_id'] = df_cite['target_arti_id'].fillna('').astype(str)

print(f'  전체 인용 레코드: {len(df_cite):,}건')


# ════════════════════════════════════════════════════════════════════
# 2. 엣지 추출 (양쪽 모두 분석 대상인 내부 인용만)
# ════════════════════════════════════════════════════════════════════
print('\n▶ 내부 인용 엣지 추출 중...')

df_edges = df_cite[
    df_cite['source_id'].isin(all_ids) &
    df_cite['target_arti_id'].isin(all_ids) &
    (df_cite['source_id'] != df_cite['target_arti_id'])  # self-loop 제거
][['source_id', 'target_arti_id']].drop_duplicates()

df_edges.columns = ['source', 'target']
print(f'  내부 인용 엣지 (중복 제거): {len(df_edges):,}건')
print(f'  엣지에 등장하는 source 논문: {df_edges["source"].nunique():,}편')
print(f'  엣지에 등장하는 target 논문: {df_edges["target"].nunique():,}편')


# ════════════════════════════════════════════════════════════════════
# 3. 네트워크 구축
# ════════════════════════════════════════════════════════════════════
print('\n▶ 네트워크 구축 중...')

G = nx.Graph()

# 모든 분석 대상 논문을 노드로 추가 (고립 노드 포함)
for pid in all_ids:
    G.add_node(pid,
               title=str(id2title.get(pid, '')),
               year=int(id2year.get(pid, 0)),
               category=str(id2cat.get(pid, '')),
               primary_category=str(id2pcat.get(pid, '')))

# 엣지 추가 (무방향)
for _, row in df_edges.iterrows():
    if G.has_edge(row['source'], row['target']):
        G[row['source']][row['target']]['weight'] += 1
    else:
        G.add_edge(row['source'], row['target'], weight=1)

print(f'  전체 노드: {G.number_of_nodes():,}')
print(f'  전체 엣지: {G.number_of_edges():,}')
print(f'  네트워크 밀도: {nx.density(G):.6f}')


# ════════════════════════════════════════════════════════════════════
# 4. Giant Component 추출
# ════════════════════════════════════════════════════════════════════
print('\n▶ Giant Component 분석 중...')

components = sorted(nx.connected_components(G), key=len, reverse=True)
isolated   = sum(1 for c in components if len(c) == 1)
giant_nodes = components[0]
G_giant = G.subgraph(giant_nodes).copy()

print(f'  연결 컴포넌트 수: {len(components):,}')
print(f'  고립 노드(단독): {isolated:,}편 ({isolated/G.number_of_nodes()*100:.1f}%)')
print(f'  Giant component: {G_giant.number_of_nodes():,}노드 / '
      f'{G_giant.number_of_edges():,}엣지 '
      f'({G_giant.number_of_nodes()/G.number_of_nodes()*100:.1f}%)')

# 컴포넌트 크기 분포
comp_sizes = Counter(len(c) for c in components)
print(f'\n  [컴포넌트 크기 분포]')
for size in sorted(comp_sizes.keys(), reverse=True)[:8]:
    print(f'    크기 {size:4d}: {comp_sizes[size]:3d}개')
if len(comp_sizes) > 8:
    print(f'    ... (총 {len(comp_sizes)}종)')


# ════════════════════════════════════════════════════════════════════
# 5. 기본 통계
# ════════════════════════════════════════════════════════════════════
print('\n▶ 네트워크 통계 계산 중...')

degree_dict  = dict(G_giant.degree())
degrees      = list(degree_dict.values())
pagerank     = nx.pagerank(G_giant, weight='weight')
clustering   = nx.clustering(G_giant)

pr_top10 = sorted(pagerank.items(), key=lambda x: -x[1])[:10]

stats_lines = [
    '=' * 60,
    '  KCI AI 법학 논문 인용 네트워크 기본 통계',
    '=' * 60,
    '',
    '[전체 네트워크]',
    f'  분석 대상 논문:      {G.number_of_nodes():>6,}편',
    f'  내부 인용 엣지:      {G.number_of_edges():>6,}건',
    f'  네트워크 밀도:       {nx.density(G):.6f}',
    f'  연결 컴포넌트 수:    {len(components):>6,}개',
    f'  고립 노드:          {isolated:>6,}편 ({isolated/G.number_of_nodes()*100:.1f}%)',
    '',
    '[Giant Component]',
    f'  노드 수:            {G_giant.number_of_nodes():>6,}편 ({G_giant.number_of_nodes()/G.number_of_nodes()*100:.1f}%)',
    f'  엣지 수:            {G_giant.number_of_edges():>6,}건',
    f'  평균 degree:        {np.mean(degrees):.2f}',
    f'  최대 degree:        {max(degrees)}',
    f'  평균 군집계수:       {np.mean(list(clustering.values())):.4f}',
    '',
    '[PageRank 상위 10 논문 (Giant Component)]',
]
for rank, (pid, pr) in enumerate(pr_top10, 1):
    title = id2title.get(pid, pid)[:40]
    year  = id2year.get(pid, '?')
    cat   = id2pcat.get(pid, '?')
    stats_lines.append(f'  {rank:2d}. [{year}] {title}')
    stats_lines.append(f'      카테고리: {cat}  PageRank: {pr:.5f}')

stats_text = '\n'.join(stats_lines)
print(stats_text)

with open(OUT / 'network_stats.txt', 'w', encoding='utf-8') as f:
    f.write(stats_text)
print(f'\n  [저장] network_stats.txt')


# ════════════════════════════════════════════════════════════════════
# 6. Degree 분포 시각화
# ════════════════════════════════════════════════════════════════════
print('\n▶ Degree 분포 시각화 중...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 선형 스케일
axes[0].hist(degrees, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('Degree 분포 (Giant Component)', fontsize=12)
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('논문 수')
axes[0].grid(axis='y', alpha=0.3)

# 로그 스케일 (power-law 확인)
deg_cnt = Counter(degrees)
x_vals = sorted(deg_cnt.keys())
y_vals = [deg_cnt[d] for d in x_vals]
axes[1].scatter(x_vals, y_vals, color='tomato', s=20, alpha=0.7)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('Degree 분포 (로그 스케일)', fontsize=12)
axes[1].set_xlabel('Degree (log)')
axes[1].set_ylabel('논문 수 (log)')
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'degree_distribution.png', dpi=150)
plt.close(fig)
print('  [저장] degree_distribution.png')


# ════════════════════════════════════════════════════════════════════
# 7. GraphML 저장 (Leiden 스크립트에서 재사용)
# ════════════════════════════════════════════════════════════════════
print('\n▶ GraphML 저장 중...')

nx.write_graphml(G,       str(OUT / 'network_full.graphml'))
nx.write_graphml(G_giant, str(OUT / 'network_giant.graphml'))
print(f'  [저장] network_full.graphml  ({G.number_of_nodes()}노드)')
print(f'  [저장] network_giant.graphml ({G_giant.number_of_nodes()}노드)')

print('\n' + '=' * 60)
print('  완료! 다음 단계: python 03.network_leiden.py')
print('=' * 60)
