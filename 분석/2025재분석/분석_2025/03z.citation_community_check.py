"""
03z.citation_community_check.py
================================
인용네트워크 커뮤니티 집중도 점검 (원고 3.6절 근거 수치 산출)

목적:
    V11~V13 원고 3.6절의 "전체 논문의 약 70%가 하나의 커뮤니티에 집중" 문장을
    2025 인용데이터 반영본으로 재검증한다. 커뮤니티 탐지 결과를 주제분류로
    쓰지 않는 근거가 되는 수치(고립 노드 비율, 거대 컴포넌트 비율, 커뮤니티 수,
    최대 커뮤니티 비중)를 해상도별로 산출한다.

입력 (같은 폴더에 있어야 함. 심볼릭 링크 가능):
    00.KCI_AI_논문_상세_및_인용데이터.csv   (2025 반영본을 링크)
    KCI_AI_논문_카테고리_확정.csv
    KCI_AI_논문_카테고리_검토필요.csv

출력:
    results/03z_citation_community/community_concentration.txt   요약 (본문 붙여넣기용)
    results/03z_citation_community/resolution_sweep.csv          해상도별 수치
    results/03z_citation_community/node_community_res{R}.csv     선택 해상도 노드-커뮤니티 배정

사용법:
    cd 2025인용_재수집/분석_2025
    python 03z.citation_community_check.py

엣지 구성 로직은 03.network_build.py와 동일하다(양쪽 모두 분석대상인 내부 인용,
self-loop 제거, 중복 제거, 무방향). Leiden 설정도 03.network_leiden.py와 같다
(RBConfigurationVertexPartition, weights='weight', seed=42).
"""

import os
import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg

warnings.filterwarnings('ignore')

# ── 설정 ──────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUT = BASE / 'results' / '03z_citation_community'
OUT.mkdir(parents=True, exist_ok=True)

RESOLUTION_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]
FINAL_RESOLUTION = 1.5      # 03.network_leiden.py와 동일
MIN_COMMUNITY_SIZE = 10     # 03.network_leiden.py와 동일
RANDOM_SEED = 42


def find_file(keyword):
    for f in os.listdir(BASE):
        if keyword in unicodedata.normalize('NFC', f):
            return BASE / f
    raise FileNotFoundError(f'파일을 찾을 수 없습니다: {keyword}')


def run_leiden(G_ig, resolution, seed=RANDOM_SEED):
    part = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights='weight',
        seed=seed,
    )
    return part


# ── 1. 데이터 로드 ─────────────────────────────────────────────────
print('▶ 데이터 로드 중...')

df_ok = pd.read_csv(find_file('카테고리_확정'), encoding='utf-8-sig')
df_rv = pd.read_csv(find_file('검토필요'), encoding='utf-8-sig')

cols = ['논문ID', '제목', '발행연도', '최종_카테고리']
df_papers = pd.concat([df_ok[cols], df_rv[cols]], ignore_index=True)
df_papers['논문ID'] = df_papers['논문ID'].astype(str)
df_papers = df_papers.drop_duplicates(subset='논문ID')

all_ids = set(df_papers['논문ID'])
n_papers = len(all_ids)
print(f'  분석 대상 논문: {n_papers:,}편')

df_cite = pd.read_csv(find_file('인용데이터'), encoding='utf-8-sig',
                      usecols=['source_id', 'target_arti_id'])
df_cite['source_id'] = df_cite['source_id'].astype(str)
df_cite['target_arti_id'] = df_cite['target_arti_id'].fillna('').astype(str)
print(f'  전체 인용 레코드: {len(df_cite):,}건')


# ── 2. 내부 인용 엣지 (03.network_build.py와 동일 로직) ─────────────
print('\n▶ 내부 인용 엣지 추출 중...')

df_edges = df_cite[
    df_cite['source_id'].isin(all_ids)
    & df_cite['target_arti_id'].isin(all_ids)
    & (df_cite['source_id'] != df_cite['target_arti_id'])
][['source_id', 'target_arti_id']].drop_duplicates()
df_edges.columns = ['source', 'target']
print(f'  내부 인용 엣지 (중복 제거): {len(df_edges):,}건')


# ── 3. 네트워크 구축 (고립 노드 포함) ───────────────────────────────
G = nx.Graph()
G.add_nodes_from(all_ids)
for s, t in df_edges.itertuples(index=False):
    if G.has_edge(s, t):
        G[s][t]['weight'] += 1
    else:
        G.add_edge(s, t, weight=1)

n_iso = sum(1 for _, d in G.degree() if d == 0)
comps = sorted(nx.connected_components(G), key=len, reverse=True)
G_giant = G.subgraph(comps[0]).copy()
n_giant = G_giant.number_of_nodes()

print(f'\n  전체 노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}')
print(f'  고립 노드: {n_iso:,}편 ({n_iso / n_papers * 100:.1f}%)')
print(f'  연결 컴포넌트: {len(comps):,}개')
print(f'  거대 컴포넌트: {n_giant:,}편 ({n_giant / n_papers * 100:.1f}%)')


# ── 4. 해상도별 Leiden ─────────────────────────────────────────────
print(f'\n▶ Leiden 해상도 sweep (seed={RANDOM_SEED}) ...')

G_ig = ig.Graph()
nodes = list(G_giant.nodes())
G_ig.add_vertices(len(nodes))
idx = {n: i for i, n in enumerate(nodes)}
G_ig.add_edges([(idx[u], idx[v]) for u, v in G_giant.edges()])
G_ig.es['weight'] = [G_giant[u][v]['weight'] for u, v in G_giant.edges()]

rows = []
final_membership = None
for res in RESOLUTION_SWEEP:
    part = run_leiden(G_ig, res)
    sizes = sorted(pd.Series(part.membership).value_counts().tolist(), reverse=True)
    top1 = sizes[0]
    top3 = sum(sizes[:3])
    rows.append({
        'resolution': res,
        '커뮤니티수': len(sizes),
        f'크기≥{MIN_COMMUNITY_SIZE}': sum(1 for s in sizes if s >= MIN_COMMUNITY_SIZE),
        '모듈성': round(part.modularity, 4),
        '최대커뮤니티_편수': top1,
        '최대커뮤니티_거대성분대비%': round(top1 / n_giant * 100, 1),
        '최대커뮤니티_전체논문대비%': round(top1 / n_papers * 100, 1),
        '상위3_전체논문대비%': round(top3 / n_papers * 100, 1),
    })
    print(f'  res={res:<4} 커뮤니티={len(sizes):3}개  최대={top1:4}편 '
          f'(거대성분 {top1/n_giant*100:5.1f}% / 전체 {top1/n_papers*100:5.1f}%)  '
          f'모듈성={part.modularity:.3f}')
    if res == FINAL_RESOLUTION:
        final_membership = part.membership

df_sweep = pd.DataFrame(rows)
df_sweep.to_csv(OUT / 'resolution_sweep.csv', index=False, encoding='utf-8-sig')


# ── 5. 선택 해상도의 노드-커뮤니티 배정 저장 ─────────────────────────
if final_membership is not None:
    id2title = df_papers.set_index('논문ID')['제목'].to_dict()
    id2year = df_papers.set_index('논문ID')['발행연도'].to_dict()
    id2cat = df_papers.set_index('논문ID')['최종_카테고리'].to_dict()
    pd.DataFrame({
        'node_id': nodes,
        'community_id': final_membership,
        'title': [id2title.get(n, '') for n in nodes],
        'year': [id2year.get(n, '') for n in nodes],
        'category': [id2cat.get(n, '') for n in nodes],
    }).to_csv(OUT / f'node_community_res{FINAL_RESOLUTION}.csv',
              index=False, encoding='utf-8-sig')


# ── 6. 요약 (본문 붙여넣기용) ───────────────────────────────────────
fin = df_sweep[df_sweep['resolution'] == FINAL_RESOLUTION].iloc[0]
lines = [
    '=' * 62,
    '  인용네트워크 커뮤니티 집중도 점검 (2025 인용데이터 반영)',
    '=' * 62,
    '',
    f'  분석 대상 논문      : {n_papers:,}편',
    f'  내부 인용 엣지      : {G.number_of_edges():,}건',
    f'  고립 노드           : {n_iso:,}편 ({n_iso / n_papers * 100:.1f}%)',
    f'  연결 컴포넌트       : {len(comps):,}개',
    f'  거대 컴포넌트       : {n_giant:,}편 ({n_giant / n_papers * 100:.1f}%)',
    '',
    f'  [resolution={FINAL_RESOLUTION}, seed={RANDOM_SEED}]',
    f'  커뮤니티 수         : {int(fin["커뮤니티수"])}개 '
    f'(크기≥{MIN_COMMUNITY_SIZE}: {int(fin[f"크기≥{MIN_COMMUNITY_SIZE}"])}개)',
    f'  최대 커뮤니티       : {int(fin["최대커뮤니티_편수"])}편',
    f'    - 거대 컴포넌트 대비 {fin["최대커뮤니티_거대성분대비%"]}%',
    f'    - 전체 논문 대비    {fin["최대커뮤니티_전체논문대비%"]}%',
    f'  상위 3개 누적       : 전체 논문 대비 {fin["상위3_전체논문대비%"]}%',
    '',
    '  ※ 해상도별 수치는 resolution_sweep.csv 참조.',
    '  ※ 원고 3.6절의 "전체 논문의 약 70%가 하나의 커뮤니티에 집중"이',
    '     어느 해상도에서도 재현되지 않으면 문장을 교체해야 한다.',
    '',
]
txt = '\n'.join(lines)
(OUT / 'community_concentration.txt').write_text(txt, encoding='utf-8')
print('\n' + txt)
print(f'[저장] {OUT}')
