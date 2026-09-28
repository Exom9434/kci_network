"""
03-2.network_leiden.py
======================
확장 네트워크 Leiden 커뮤니티 감지 + 카테고리 레이블링

[03.network_leiden.py와 차이점]
  - 노드가 AI(타입A) / 비AI KCI(타입B) 두 종류
  - 커뮤니티 레이블링은 타입A(AI 논문) 카테고리 분포 기준
  - 커뮤니티 요약에 "비AI KCI 논문 비율" 추가
    → 어느 커뮤니티가 외부 문헌을 많이 참고하는지 파악 가능

- 입력:
    results/03-2_network/network_extended_giant.graphml

- 출력:
    results/03-2_network/leiden_resolution_sweep.png
    results/03-2_network/community_summary.csv
    results/03-2_network/community_category_dist.png
    results/03-2_network/community_year_dist.png
    results/03-2_network/node_community.csv
    results/03-2_network/network_extended_giant_with_community.graphml

설정:
    RESOLUTION          Leiden resolution parameter
    MIN_COMMUNITY_SIZE  독립 커뮤니티 최소 논문 수 (AI 논문 기준)

사용법:
    python 03-2.network_leiden.py
"""

import os, ast, warnings
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
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

# ── 설정 ─────────────────────────────────────────────────────────
RESOLUTION         = 1.0
MIN_COMMUNITY_SIZE = 10     # AI 논문 기준
RESOLUTION_SWEEP   = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
RANDOM_SEED        = 42

CATEGORIES = [
    '공법', '민사법', '형사법', '경제법', '노동법',
    '사회보장법', '금융법', '의료법', '조세법',
    '데이터법', '인공지능법', '지식재산권법', '기초법',
    '리걸테크', '자율무기법', '자율주행자동차법', '해외법',
]


# ════════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════════

def nx_to_igraph(G_nx):
    nodes     = list(G_nx.nodes())
    node_idx  = {n: i for i, n in enumerate(nodes)}
    edges     = [(node_idx[u], node_idx[v]) for u, v in G_nx.edges()]
    weights   = [G_nx[u][v].get('weight', 1) for u, v in G_nx.edges()]
    G_ig      = ig.Graph(n=len(nodes), edges=edges, directed=False)
    G_ig.vs['name'] = nodes
    G_ig.es['weight'] = weights
    for attr in ['node_type', 'title', 'year', 'category', 'primary_category']:
        G_ig.vs[attr] = [str(G_nx.nodes[n].get(attr, '')) for n in nodes]
    return G_ig, nodes


def run_leiden(G_ig, resolution, seed=RANDOM_SEED):
    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights='weight',
        seed=seed,
    )
    return partition, partition.quality()


def expand_cats(cat_str):
    cat_str = str(cat_str).strip()
    if cat_str.startswith('융복합(') and cat_str.endswith(')'):
        return [p.strip() for p in cat_str[4:-1].split('+')]
    return [cat_str]


def auto_label(cat_dist, threshold_dominant=0.50, threshold_secondary=0.20):
    if not cat_dist:
        return '미분류'
    total = sum(cat_dist.values())
    if total == 0:
        return '미분류'
    sorted_cats = sorted(cat_dist.items(), key=lambda x: -x[1])
    top1_cat, top1_cnt = sorted_cats[0]
    if top1_cnt / total >= threshold_dominant:
        return top1_cat
    if len(sorted_cats) >= 2:
        top2_cat, top2_cnt = sorted_cats[1]
        if top2_cnt / total >= threshold_secondary:
            pair = sorted([top1_cat, top2_cat],
                          key=lambda x: CATEGORIES.index(x) if x in CATEGORIES else 99)
            return f'융복합({pair[0]}+{pair[1]})'
    return '혼합'


# ════════════════════════════════════════════════════════════════════
# 1. 그래프 로드
# ════════════════════════════════════════════════════════════════════
print('=' * 60)
print('  03-2.network_leiden.py — Leiden 커뮤니티 감지 (확장)')
print('=' * 60)

print('\n▶ 확장 Giant component 그래프 로드 중...')
G_nx = nx.read_graphml(str(OUT / 'network_extended_giant.graphml'))
n_ai  = sum(1 for n in G_nx.nodes() if G_nx.nodes[n].get('node_type') == 'ai')
n_kci = sum(1 for n in G_nx.nodes() if G_nx.nodes[n].get('node_type') == 'kci')
print(f'  전체 노드: {G_nx.number_of_nodes():,}  엣지: {G_nx.number_of_edges():,}')
print(f'    └ AI 논문:      {n_ai:,}')
print(f'    └ 비AI KCI 논문: {n_kci:,}')

G_ig, node_list = nx_to_igraph(G_nx)


# ════════════════════════════════════════════════════════════════════
# 2. Resolution Sweep (AI 논문 기준으로 커뮤니티 크기 카운트)
# ════════════════════════════════════════════════════════════════════
print(f'\n▶ Resolution sweep ...')

sweep_results = []
for res in RESOLUTION_SWEEP:
    part, mod = run_leiden(G_ig, res)
    membership = part.membership
    # AI 논문만 기준으로 커뮤니티 크기 계산
    ai_per_comm = Counter()
    for i, cid in enumerate(membership):
        if G_ig.vs[i]['node_type'] == 'ai':
            ai_per_comm[cid] += 1
    n_large = sum(1 for s in ai_per_comm.values() if s >= MIN_COMMUNITY_SIZE)
    n_comm  = len(set(membership))
    sweep_results.append({
        'resolution': res, 'modularity': round(mod, 2),
        'n_communities': n_comm, 'n_large': n_large,
    })
    print(f'  res={res:.1f}  modularity={mod:.1f}  '
          f'커뮤니티={n_comm}개  (AI≥{MIN_COMMUNITY_SIZE}편: {n_large}개)')

df_sweep = pd.DataFrame(sweep_results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(df_sweep['resolution'], df_sweep['modularity'], marker='o', color='steelblue', linewidth=2)
axes[0].axvline(RESOLUTION, color='tomato', linestyle='--', label=f'선택={RESOLUTION}')
axes[0].set_title('Resolution vs Modularity'); axes[0].set_xlabel('Resolution')
axes[0].set_ylabel('Modularity'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(df_sweep['resolution'], df_sweep['n_large'], marker='s', color='darkorange',
             linewidth=2, label=f'AI≥{MIN_COMMUNITY_SIZE}')
axes[1].plot(df_sweep['resolution'], df_sweep['n_communities'], marker='o', color='gray',
             linewidth=1, linestyle='--', label='전체')
axes[1].axvline(RESOLUTION, color='tomato', linestyle='--', label=f'선택={RESOLUTION}')
axes[1].set_title('Resolution vs 커뮤니티 수'); axes[1].set_xlabel('Resolution')
axes[1].legend(); axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'leiden_resolution_sweep.png', dpi=150)
plt.close(fig)
print(f'\n  [저장] leiden_resolution_sweep.png')


# ════════════════════════════════════════════════════════════════════
# 3. 최종 Leiden
# ════════════════════════════════════════════════════════════════════
print(f'\n▶ 최종 Leiden (resolution={RESOLUTION}) ...')
final_partition, final_mod = run_leiden(G_ig, RESOLUTION)
raw_membership = final_partition.membership
print(f'  Modularity: {final_mod:.2f}')
print(f'  원시 커뮤니티: {len(set(raw_membership))}개')


# ════════════════════════════════════════════════════════════════════
# 4. 소규모 커뮤니티 처리 (AI 논문 기준)
# ════════════════════════════════════════════════════════════════════
# AI 논문 수 기준으로 커뮤니티 크기 측정
ai_per_comm = Counter()
for i, cid in enumerate(raw_membership):
    if G_ig.vs[i]['node_type'] == 'ai':
        ai_per_comm[cid] += 1

large_comms  = {cid for cid, sz in ai_per_comm.items() if sz >= MIN_COMMUNITY_SIZE}
sorted_large = sorted(large_comms, key=lambda c: -ai_per_comm[c])
old2new      = {old: new for new, old in enumerate(sorted_large)}
SMALL_ID     = len(sorted_large)

membership_final = [
    old2new[cid] if cid in large_comms else SMALL_ID
    for cid in raw_membership
]

n_large       = len(sorted_large)
n_small_nodes = membership_final.count(SMALL_ID)
print(f'  유효 커뮤니티 (AI≥{MIN_COMMUNITY_SIZE}편): {n_large}개')
print(f'  기타 편입 노드: {n_small_nodes}개')


# ════════════════════════════════════════════════════════════════════
# 5. 커뮤니티별 카테고리 분포 + 자동 레이블
# ════════════════════════════════════════════════════════════════════
print('\n▶ 카테고리 분포 기반 자동 레이블 생성 중...')

node_data = []
for i, nid in enumerate(node_list):
    attrs = G_nx.nodes[nid]
    node_data.append({
        'node_id':      nid,
        'community_id': membership_final[i],
        'node_type':    attrs.get('node_type', ''),
        'title':        attrs.get('title', ''),
        'year':         int(attrs.get('year', 0)),
        'category':     attrs.get('category', ''),
        'primary_category': attrs.get('primary_category', ''),
    })
df_nodes = pd.DataFrame(node_data)

pagerank_nx = nx.pagerank(G_nx, weight='weight')

comm_summaries = []
for cid in list(range(n_large)) + [SMALL_ID]:
    mask     = df_nodes['community_id'] == cid
    sub      = df_nodes[mask]
    sub_ai   = sub[sub['node_type'] == 'ai']
    sub_kci  = sub[sub['node_type'] == 'kci']
    n_total  = len(sub)
    n_ai_c   = len(sub_ai)
    n_kci_c  = len(sub_kci)
    if n_total == 0:
        continue

    label_id = '기타' if cid == SMALL_ID else f'C{cid+1:02d}'

    # 카테고리 분포 (AI 논문만, 융복합 분해 포함)
    cat_counts = Counter()
    for cat in sub_ai['category']:
        for c in expand_cats(cat):
            if c in CATEGORIES:
                cat_counts[c] += 1

    auto_lbl = '기타' if cid == SMALL_ID else auto_label(cat_counts)

    # PageRank 상위 3 (AI 논문 한정)
    pr_ai = {nid: pagerank_nx.get(nid, 0) for nid in sub_ai['node_id']}
    top3  = sorted(pr_ai.items(), key=lambda x: -x[1])[:3]
    top3_titles = ' | '.join(
        str(G_nx.nodes[nid].get('title', nid))[:30] for nid, _ in top3
    )

    comm_summaries.append({
        'community_id':  cid,
        'label_id':      label_id,
        'auto_label':    auto_lbl,
        'final_label':   '',
        'size_total':    n_total,
        'size_ai':       n_ai_c,
        'size_kci':      n_kci_c,
        'kci_ratio':     round(n_kci_c / n_total, 3) if n_total > 0 else 0,
        'top1_category': cat_counts.most_common(1)[0][0] if cat_counts else '',
        'top2_category': cat_counts.most_common(2)[1][0] if len(cat_counts) >= 2 else '',
        'top1_ratio':    round(cat_counts.most_common(1)[0][1] / sum(cat_counts.values()), 3)
                         if cat_counts else 0,
        'top3_ai_papers': top3_titles,
        'year_min':      int(sub_ai['year'].min()) if n_ai_c > 0 else 0,
        'year_max':      int(sub_ai['year'].max()) if n_ai_c > 0 else 0,
        'cat_dist_json': str(dict(cat_counts.most_common())),
    })

df_summary = pd.DataFrame(comm_summaries)
df_summary.to_csv(OUT / 'community_summary.csv', index=False, encoding='utf-8-sig')
print(f'  [저장] community_summary.csv')

print(f'\n  {"ID":5s} {"레이블(초안)":22s} {"AI":>5s}  {"KCI":>5s}  {"KCI비율":>7s}  {"1위카테고리":14s}')
print('  ' + '-' * 68)
for _, r in df_summary.iterrows():
    print(f"  {r['label_id']:5s} {r['auto_label']:22s} {r['size_ai']:>5d}  "
          f"{r['size_kci']:>5d}  {r['kci_ratio']:>7.1%}  {r['top1_category']:14s}")


# ════════════════════════════════════════════════════════════════════
# 6. 시각화
# ════════════════════════════════════════════════════════════════════
print('\n▶ 시각화 생성 중...')

df_large = df_summary[df_summary['community_id'] != SMALL_ID].copy()

# 카테고리 분포 히트맵
heatmap_data = []
for _, r in df_large.iterrows():
    cat_dist = ast.literal_eval(r['cat_dist_json'])
    total    = sum(cat_dist.values()) or 1
    row_pct  = {cat: cat_dist.get(cat, 0) / total * 100 for cat in CATEGORIES}
    row_pct['label'] = r['auto_label']
    heatmap_data.append(row_pct)

if heatmap_data:
    df_hm = pd.DataFrame(heatmap_data).set_index('label')[CATEGORIES]
    fig, ax = plt.subplots(figsize=(16, max(4, len(df_hm) * 0.5 + 2)))
    im = ax.imshow(df_hm.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(df_hm)))
    ax.set_yticklabels(df_hm.index, fontsize=9)
    ax.set_title(f'커뮤니티별 카테고리 분포 (AI 논문 기준, %) — resolution={RESOLUTION}',
                 fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, label='비율 (%)', shrink=0.8)
    for i in range(len(df_hm)):
        for j in range(len(CATEGORIES)):
            val = df_hm.values[i, j]
            if val >= 10:
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=7, color='black' if val < 60 else 'white')
    fig.tight_layout()
    fig.savefig(OUT / 'community_category_dist.png', dpi=150)
    plt.close(fig)
    print('  [저장] community_category_dist.png')

# 연도 분포
years   = list(range(2016, 2026))
palette = plt.cm.tab20.colors
fig, ax = plt.subplots(figsize=(14, 6))
bottom  = np.zeros(len(years))
for idx, (_, r) in enumerate(df_large.iterrows()):
    sub_ai = df_nodes[(df_nodes['community_id'] == r['community_id']) &
                      (df_nodes['node_type'] == 'ai')]
    yc     = sub_ai['year'].value_counts()
    vals   = [yc.get(yr, 0) for yr in years]
    ax.bar(years, vals, bottom=bottom, color=palette[idx % len(palette)],
           label=r['auto_label'], width=0.7)
    bottom += np.array(vals, dtype=float)

ax.set_title('커뮤니티별 연도 분포 (AI 논문 기준)')
ax.set_xlabel('발행연도'); ax.set_ylabel('논문 수')
ax.set_xticks(years)
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'community_year_dist.png', dpi=150)
plt.close(fig)
print('  [저장] community_year_dist.png')

# KCI 비율 막대 (커뮤니티별 외부 참조 의존도)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(df_large['label_id'], df_large['kci_ratio'] * 100,
              color=palette[:len(df_large)], edgecolor='white', alpha=0.85)
ax.set_title('커뮤니티별 비AI KCI 논문 비율\n(높을수록 외부 문헌 참조 의존도 높음)')
ax.set_xlabel('커뮤니티'); ax.set_ylabel('비AI KCI 논문 비율 (%)')
ax.set_xticks(range(len(df_large)))
ax.set_xticklabels(df_large['auto_label'], rotation=30, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, (_, r) in zip(bars, df_large.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{r['kci_ratio']*100:.1f}%",
            ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'community_kci_ratio.png', dpi=150)
plt.close(fig)
print('  [저장] community_kci_ratio.png')


# ════════════════════════════════════════════════════════════════════
# 7. 노드별 CSV + GraphML 업데이트
# ════════════════════════════════════════════════════════════════════
id2label = {r['community_id']: r['auto_label'] for _, r in df_summary.iterrows()}

df_nodes['label_id']   = df_nodes['community_id'].apply(
    lambda c: f'C{c+1:02d}' if c != SMALL_ID else '기타')
df_nodes['auto_label'] = df_nodes['community_id'].map(id2label)
df_nodes.to_csv(OUT / 'node_community.csv', index=False, encoding='utf-8-sig')
print('  [저장] node_community.csv')

for i, nid in enumerate(node_list):
    cid = membership_final[i]
    G_nx.nodes[nid]['community_id'] = cid
    G_nx.nodes[nid]['auto_label']   = id2label.get(cid, '기타')
    G_nx.nodes[nid]['label_id']     = f'C{cid+1:02d}' if cid != SMALL_ID else '기타'

nx.write_graphml(G_nx, str(OUT / 'network_extended_giant_with_community.graphml'))
print('  [저장] network_extended_giant_with_community.graphml')

print('\n' + '=' * 60)
print('  완료! → community_summary.csv 의 final_label 채우기')
print('  → 이후 python 03-2.network_viz.py')
print('=' * 60)
