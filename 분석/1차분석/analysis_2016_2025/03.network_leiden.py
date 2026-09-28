"""
03.network_leiden.py
====================
Leiden 알고리즘 기반 커뮤니티 감지 + 카테고리 분포 기반 레이블링

- 입력:
    results/03_network/network_giant.graphml   (03.network_build.py 출력)

- 처리:
    1. 여러 resolution으로 Leiden 실행 → modularity 비교
    2. 최적 resolution으로 최종 커뮤니티 감지
    3. 소규모 커뮤니티(< MIN_COMMUNITY_SIZE) 처리
    4. 카테고리 분포 기반 자동 레이블 초안 생성
    5. 커뮤니티별 요약 통계 (연도 분포, 핵심 논문 etc.)

- 출력:
    results/03_network/leiden_resolution_sweep.png    resolution별 modularity/커뮤니티수
    results/03_network/community_summary.csv          커뮤니티별 요약 (레이블 초안 포함)
    results/03_network/community_category_dist.png    커뮤니티별 카테고리 분포 히트맵
    results/03_network/community_year_dist.png        커뮤니티별 연도 분포
    results/03_network/node_community.csv             노드별 커뮤니티 ID + 레이블
    results/03_network/network_giant_with_community.graphml  커뮤니티 속성 추가된 그래프

설정:
    RESOLUTION        Leiden resolution parameter (높을수록 커뮤니티 잘게 쪼개짐)
    MIN_COMMUNITY_SIZE 이 이상이어야 독립 커뮤니티로 인정 (미만은 '기타'로 묶음)

사용법:
    python 03.network_leiden.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
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

# ── 설정 ─────────────────────────────────────────────────────────
RESOLUTION         = 1.5    # 최종 사용할 resolution (sweep 후 조정 가능)
MIN_COMMUNITY_SIZE = 10     # 이 미만은 '기타'로 묶음
RESOLUTION_SWEEP   = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0,2.5, 3.0, 4.0, 5.0]
RANDOM_SEED        = 42

CATEGORIES = [
    '공법', '민사법', '형사법', '경제법', '노동법',
    '사회보장법', '금융법', '의료법', '조세법',
    '데이터법', '인공지능법', '지식재산권법', '기초법',
    '리걸테크', '자율무기법', '자율주행자동차법', '해외법',
]


# ════════════════════════════════════════════════════════════════════
# 유틸 함수
# ════════════════════════════════════════════════════════════════════

def nx_to_igraph(G_nx):
    """NetworkX 무방향 그래프 → igraph 변환"""
    nodes = list(G_nx.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G_nx.edges()]
    weights = [G_nx[u][v].get('weight', 1) for u, v in G_nx.edges()]

    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    G_ig.vs['name'] = nodes
    G_ig.es['weight'] = weights

    # 노드 속성 복사
    for attr in ['title', 'year', 'category', 'primary_category']:
        G_ig.vs[attr] = [str(G_nx.nodes[n].get(attr, '')) for n in nodes]

    return G_ig, nodes


def run_leiden(G_ig, resolution, seed=RANDOM_SEED):
    """Leiden 실행, (partition, modularity) 반환"""
    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights='weight',
        seed=seed,
    )
    return partition, partition.quality()


def auto_label(cat_dist: dict, threshold_dominant=0.50, threshold_secondary=0.20) -> str:
    """
    카테고리 분포 → 자동 레이블 초안
    - 1위가 threshold_dominant 이상이면 단일 레이블
    - 1위+2위 모두 threshold_secondary 이상이면 '융복합(A+B)'
    - 그 외는 '혼합'
    """
    if not cat_dist:
        return '미분류'
    total = sum(cat_dist.values())
    if total == 0:
        return '미분류'

    sorted_cats = sorted(cat_dist.items(), key=lambda x: -x[1])
    top1_cat, top1_cnt = sorted_cats[0]
    top1_ratio = top1_cnt / total

    if top1_ratio >= threshold_dominant:
        return top1_cat

    if len(sorted_cats) >= 2:
        top2_cat, top2_cnt = sorted_cats[1]
        top2_ratio = top2_cnt / total
        if top2_ratio >= threshold_secondary:
            # CATEGORIES 순서대로 정렬
            pair = sorted([top1_cat, top2_cat],
                          key=lambda x: CATEGORIES.index(x) if x in CATEGORIES else 99)
            return f'융복합({pair[0]}+{pair[1]})'

    return '혼합'


# ════════════════════════════════════════════════════════════════════
# 1. 그래프 로드
# ════════════════════════════════════════════════════════════════════
print('=' * 60)
print('  03.network_leiden.py — Leiden 커뮤니티 감지')
print('=' * 60)

print('\n▶ Giant component 그래프 로드 중...')
G_nx = nx.read_graphml(str(OUT / 'network_giant.graphml'))
print(f'  노드: {G_nx.number_of_nodes():,}  엣지: {G_nx.number_of_edges():,}')

G_ig, node_list = nx_to_igraph(G_nx)
print(f'  igraph 변환 완료')


# ════════════════════════════════════════════════════════════════════
# 2. Resolution Sweep
# ════════════════════════════════════════════════════════════════════
print(f'\n▶ Resolution sweep ({RESOLUTION_SWEEP}) ...')

sweep_results = []
for res in RESOLUTION_SWEEP:
    part, mod = run_leiden(G_ig, res)
    n_comm = len(set(part.membership))
    sizes  = Counter(part.membership)
    large  = sum(1 for s in sizes.values() if s >= MIN_COMMUNITY_SIZE)
    sweep_results.append({
        'resolution': res,
        'modularity': round(mod, 4),
        'n_communities': n_comm,
        'n_large_communities': large,
    })
    print(f'  res={res:.1f}  modularity={mod:.4f}  '
          f'커뮤니티={n_comm}개 (크기≥{MIN_COMMUNITY_SIZE}: {large}개)')

# sweep 시각화
df_sweep = pd.DataFrame(sweep_results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(df_sweep['resolution'], df_sweep['modularity'],
             marker='o', color='steelblue', linewidth=2)
axes[0].axvline(RESOLUTION, color='tomato', linestyle='--',
                label=f'선택 resolution={RESOLUTION}')
axes[0].set_title('Resolution vs Modularity')
axes[0].set_xlabel('Resolution')
axes[0].set_ylabel('Modularity')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df_sweep['resolution'], df_sweep['n_large_communities'],
             marker='s', color='darkorange', linewidth=2,
             label=f'크기≥{MIN_COMMUNITY_SIZE}')
axes[1].plot(df_sweep['resolution'], df_sweep['n_communities'],
             marker='o', color='gray', linewidth=1, linestyle='--',
             label='전체')
axes[1].axvline(RESOLUTION, color='tomato', linestyle='--',
                label=f'선택 resolution={RESOLUTION}')
axes[1].set_title('Resolution vs 커뮤니티 수')
axes[1].set_xlabel('Resolution')
axes[1].set_ylabel('커뮤니티 수')
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'leiden_resolution_sweep.png', dpi=150)
plt.close(fig)
print(f'\n  [저장] leiden_resolution_sweep.png')


# ════════════════════════════════════════════════════════════════════
# 3. 최종 Leiden 실행
# ════════════════════════════════════════════════════════════════════
print(f'\n▶ 최종 Leiden 실행 (resolution={RESOLUTION}) ...')

final_partition, final_mod = run_leiden(G_ig, RESOLUTION)
raw_membership = final_partition.membership

print(f'  Modularity: {final_mod:.4f}')
print(f'  원시 커뮤니티 수: {len(set(raw_membership))}')


# ════════════════════════════════════════════════════════════════════
# 4. 소규모 커뮤니티 → '기타' 처리
# ════════════════════════════════════════════════════════════════════
comm_sizes = Counter(raw_membership)
large_comms = {cid for cid, sz in comm_sizes.items() if sz >= MIN_COMMUNITY_SIZE}
small_comms = {cid for cid, sz in comm_sizes.items() if sz < MIN_COMMUNITY_SIZE}

# 크기 기준 내림차순으로 커뮤니티 ID 재부여 (0부터 시작)
sorted_large = sorted(large_comms, key=lambda c: -comm_sizes[c])
old2new = {old: new for new, old in enumerate(sorted_large)}
SMALL_ID = len(sorted_large)  # '기타' 커뮤니티 ID

membership_final = []
for cid in raw_membership:
    if cid in large_comms:
        membership_final.append(old2new[cid])
    else:
        membership_final.append(SMALL_ID)

n_large = len(sorted_large)
n_small_nodes = sum(1 for m in membership_final if m == SMALL_ID)
print(f'  유효 커뮤니티 (크기≥{MIN_COMMUNITY_SIZE}): {n_large}개')
print(f'  기타(소규모) 편입 노드: {n_small_nodes}개')


# ════════════════════════════════════════════════════════════════════
# 5. 커뮤니티별 카테고리 분포 + 자동 레이블링
# ════════════════════════════════════════════════════════════════════
print('\n▶ 카테고리 분포 기반 자동 레이블 생성 중...')

# 노드별 커뮤니티 + 속성
node_data = []
for i, node_id in enumerate(node_list):
    attrs = G_nx.nodes[node_id]
    node_data.append({
        'node_id':          node_id,
        'community_id':     membership_final[i],
        'title':            attrs.get('title', ''),
        'year':             int(attrs.get('year', 0)),
        'category':         attrs.get('category', ''),
        'primary_category': attrs.get('primary_category', ''),
    })
df_nodes = pd.DataFrame(node_data)

# 융복합 카테고리 분해 함수 (중복집계용)
def expand_cats(cat_str):
    cat_str = str(cat_str).strip()
    if cat_str.startswith('융복합(') and cat_str.endswith(')'):
        return [p.strip() for p in cat_str[4:-1].split('+')]
    return [cat_str]

# 커뮤니티별 요약
pagerank_nx = nx.pagerank(G_nx, weight='weight')

comm_summaries = []
for cid in list(range(n_large)) + [SMALL_ID]:
    mask = df_nodes['community_id'] == cid
    sub  = df_nodes[mask]
    n    = len(sub)
    if n == 0:
        continue

    label_name = '기타' if cid == SMALL_ID else f'C{cid+1:02d}'

    # 카테고리 분포 (융복합 분해 포함)
    cat_counts = Counter()
    for cat in sub['category']:
        for c in expand_cats(cat):
            if c in CATEGORIES:
                cat_counts[c] += 1

    auto_lbl = '기타' if cid == SMALL_ID else auto_label(cat_counts)

    # PageRank 상위 3 논문
    pr_sub = {nid: pagerank_nx.get(nid, 0) for nid in sub['node_id']}
    top3   = sorted(pr_sub.items(), key=lambda x: -x[1])[:3]
    top3_titles = ' | '.join(
        str(G_nx.nodes[nid].get('title', nid))[:30] for nid, _ in top3
    )

    # 연도 분포
    year_dist = sub['year'].value_counts().sort_index().to_dict()

    comm_summaries.append({
        'community_id':    cid,
        'label_id':        label_name,
        'auto_label':      auto_lbl,
        'final_label':     '',          # 연구자가 직접 채울 컬럼
        'size':            n,
        'top1_category':   cat_counts.most_common(1)[0][0] if cat_counts else '',
        'top2_category':   cat_counts.most_common(2)[1][0] if len(cat_counts) >= 2 else '',
        'top3_category':   cat_counts.most_common(3)[2][0] if len(cat_counts) >= 3 else '',
        'top1_ratio':      round(cat_counts.most_common(1)[0][1] / sum(cat_counts.values()), 3)
                           if cat_counts else 0,
        'top3_papers':     top3_titles,
        'year_min':        int(sub['year'].min()),
        'year_max':        int(sub['year'].max()),
        'year_mode':       int(sub['year'].mode().iloc[0]) if len(sub) > 0 else 0,
        'cat_dist_json':   str(dict(cat_counts.most_common())),
    })

df_summary = pd.DataFrame(comm_summaries)
df_summary.to_csv(OUT / 'community_summary.csv', index=False, encoding='utf-8-sig')
print(f'  [저장] community_summary.csv')

# 요약 출력
print(f'\n  {"ID":5s} {"레이블(초안)":22s} {"크기":>5s}  {"1위 카테고리":14s}  {"비율":>5s}')
print('  ' + '-' * 62)
for _, r in df_summary.iterrows():
    print(f"  {r['label_id']:5s} {r['auto_label']:22s} {r['size']:>5d}  "
          f"{r['top1_category']:14s}  {r['top1_ratio']:>5.1%}")


# ════════════════════════════════════════════════════════════════════
# 6. 커뮤니티별 카테고리 분포 히트맵
# ════════════════════════════════════════════════════════════════════
print('\n▶ 시각화 생성 중...')

# 히트맵용 행렬 (기타 제외)
df_large = df_summary[df_summary['community_id'] != SMALL_ID].copy()
import json, ast

heatmap_data = []
for _, r in df_large.iterrows():
    cat_dist = ast.literal_eval(r['cat_dist_json'])
    total    = sum(cat_dist.values()) or 1
    row_pct  = {cat: cat_dist.get(cat, 0) / total * 100 for cat in CATEGORIES}
    row_pct['label'] = r['auto_label']
    heatmap_data.append(row_pct)

df_heatmap = pd.DataFrame(heatmap_data).set_index('label')[CATEGORIES]

fig, ax = plt.subplots(figsize=(16, max(4, len(df_heatmap) * 0.5 + 2)))
im = ax.imshow(df_heatmap.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
ax.set_xticks(range(len(CATEGORIES)))
ax.set_xticklabels(CATEGORIES, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(df_heatmap)))
ax.set_yticklabels(df_heatmap.index, fontsize=9)
ax.set_title(f'커뮤니티별 카테고리 분포 (%) — resolution={RESOLUTION}', fontsize=13, pad=12)
plt.colorbar(im, ax=ax, label='비율 (%)', shrink=0.8)

# 수치 표기 (10% 이상)
for i in range(len(df_heatmap)):
    for j in range(len(CATEGORIES)):
        val = df_heatmap.values[i, j]
        if val >= 10:
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color='black' if val < 60 else 'white')

fig.tight_layout()
fig.savefig(OUT / 'community_category_dist.png', dpi=150)
plt.close(fig)
print('  [저장] community_category_dist.png')

# 연도 분포 (커뮤니티별 stacked bar)
years = list(range(2016, 2026))
fig, ax = plt.subplots(figsize=(14, 6))
palette = plt.cm.tab20.colors
bottom = np.zeros(len(years))

for idx, (_, r) in enumerate(df_large.iterrows()):
    year_vals = []
    sub = df_nodes[df_nodes['community_id'] == r['community_id']]
    yc  = sub['year'].value_counts()
    for yr in years:
        year_vals.append(yc.get(yr, 0))
    ax.bar(years, year_vals, bottom=bottom,
           color=palette[idx % len(palette)], label=r['auto_label'], width=0.7)
    bottom += np.array(year_vals)

ax.set_title('커뮤니티별 연도 분포', fontsize=13)
ax.set_xlabel('발행연도')
ax.set_ylabel('논문 수')
ax.set_xticks(years)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'community_year_dist.png', dpi=150)
plt.close(fig)
print('  [저장] community_year_dist.png')


# ════════════════════════════════════════════════════════════════════
# 7. 노드별 커뮤니티 CSV + GraphML 업데이트
# ════════════════════════════════════════════════════════════════════

# 커뮤니티 ID → 레이블 매핑
id2label = {r['community_id']: r['auto_label'] for _, r in df_summary.iterrows()}

df_nodes['label_id']   = df_nodes['community_id'].apply(
    lambda c: f'C{c+1:02d}' if c != SMALL_ID else '기타')
df_nodes['auto_label'] = df_nodes['community_id'].map(id2label)

df_nodes.to_csv(OUT / 'node_community.csv', index=False, encoding='utf-8-sig')
print('  [저장] node_community.csv')

# GraphML에 커뮤니티 속성 추가
for i, node_id in enumerate(node_list):
    cid = membership_final[i]
    G_nx.nodes[node_id]['community_id']  = cid
    G_nx.nodes[node_id]['auto_label']    = id2label.get(cid, '기타')
    G_nx.nodes[node_id]['label_id']      = f'C{cid+1:02d}' if cid != SMALL_ID else '기타'

nx.write_graphml(G_nx, str(OUT / 'network_giant_with_community.graphml'))
print('  [저장] network_giant_with_community.graphml')


print('\n' + '=' * 60)
print(f'  완료!')
print(f'  → community_summary.csv 의 final_label 컬럼을 채워주세요')
print(f'  → 이후 python 03.network_viz.py 실행')
print('=' * 60)
