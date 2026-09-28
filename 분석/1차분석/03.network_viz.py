"""
03.network_viz.py
=================
인용 네트워크 시각화 — 커뮤니티 색상, PageRank 노드 크기

- 입력:
    results/03_network/network_giant_with_community.graphml
    results/03_network/community_summary.csv

- 출력:
    results/03_network/viz_network.png      정적 고해상도 이미지 (논문용)
    results/03_network/viz_network.html     인터랙티브 Plotly (탐색용)

- 레이아웃: Fruchterman-Reingold (spring layout)
  ※ fa2 라이브러리 없이도 동작하도록 spring layout 사용
  ※ fa2 설치 시 Force Atlas 2 자동 사용: pip install fa2

설정:
    LABEL_TOP_N    라벨 표시할 상위 PageRank 논문 수
    NODE_SCALE     노드 크기 배율

사용법:
    python 03.network_viz.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from pathlib import Path

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
OUT = Path(__file__).parent / 'results' / '03_network'
OUT.mkdir(parents=True, exist_ok=True)

# ── 설정 ─────────────────────────────────────────────────────────
LABEL_TOP_N  = 20    # 라벨 표시할 상위 PageRank 논문 수
NODE_SCALE   = 3000  # 노드 크기 배율 (PageRank 정규화 후 곱함)
LAYOUT_SEED  = 42


# ════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════════
print('=' * 60)
print('  03.network_viz.py — 네트워크 시각화')
print('=' * 60)

print('\n▶ 그래프 + 커뮤니티 정보 로드 중...')
G = nx.read_graphml(str(OUT / 'network_giant_with_community.graphml'))
df_summary = pd.read_csv(OUT / 'community_summary.csv', encoding='utf-8-sig')
print(f'  노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}')

# 커뮤니티 ID → 레이블 매핑
#   final_label가 채워져 있으면 우선 사용, 없으면 auto_label 사용
def get_label(row):
    fl = str(row.get('final_label', '')).strip()
    return fl if fl else str(row.get('auto_label', '기타')).strip()

id2label = {str(r['community_id']): get_label(r)
            for _, r in df_summary.iterrows()}

# 커뮤니티 목록 (기타 포함)
all_labels = list(dict.fromkeys(
    id2label[str(r['community_id'])] for _, r in df_summary.iterrows()
))

# 색상 팔레트 (커뮤니티별)
base_colors = (list(plt.cm.Set2.colors) + list(plt.cm.Set1.colors) +
               list(plt.cm.tab20.colors))
label_color = {}
for i, lbl in enumerate(all_labels):
    label_color[lbl] = '#cccccc' if lbl == '기타' else base_colors[i % len(base_colors)]


# ════════════════════════════════════════════════════════════════════
# 2. 레이아웃 계산
# ════════════════════════════════════════════════════════════════════
print('\n▶ 레이아웃 계산 중 (spring layout)...')

# Force Atlas 2 시도 (설치돼 있으면 사용)
try:
    from fa2 import ForceAtlas2
    fa2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        verbose=False,
    )
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    print('  레이아웃: Force Atlas 2')
except ImportError:
    pos = nx.spring_layout(G, k=2.0, seed=LAYOUT_SEED, weight='weight',
                           iterations=100)
    print('  레이아웃: Fruchterman-Reingold (spring)')
    print('  ※ Force Atlas 2 사용 시: pip install fa2')


# ════════════════════════════════════════════════════════════════════
# 3. PageRank 계산
# ════════════════════════════════════════════════════════════════════
print('\n▶ PageRank 계산 중...')
pr = nx.pagerank(G, weight='weight')
max_pr = max(pr.values())

# 상위 N개 노드 (라벨 표시용)
top_nodes = set(n for n, _ in sorted(pr.items(), key=lambda x: -x[1])[:LABEL_TOP_N])


# ════════════════════════════════════════════════════════════════════
# 4. 정적 시각화 (matplotlib) — 논문용
# ════════════════════════════════════════════════════════════════════
print('\n▶ 정적 시각화 (PNG) 생성 중...')

fig, ax = plt.subplots(figsize=(18, 14))
ax.set_facecolor('#f8f8f8')
fig.patch.set_facecolor('#f8f8f8')

# 엣지 (얇게, 반투명)
nx.draw_networkx_edges(
    G, pos, ax=ax,
    alpha=0.12, width=0.4,
    edge_color='#888888'
)

# 노드 (커뮤니티별 색상, PageRank 크기)
for lbl in all_labels:
    nodes_in_lbl = [
        n for n in G.nodes()
        if id2label.get(str(G.nodes[n].get('community_id', '')), '기타') == lbl
    ]
    if not nodes_in_lbl:
        continue
    sizes = [pr.get(n, 0) / max_pr * NODE_SCALE + 20 for n in nodes_in_lbl]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=nodes_in_lbl,
        node_size=sizes,
        node_color=label_color[lbl],
        alpha=0.85,
        linewidths=0.3,
        edgecolors='white',
    )

# 상위 노드 라벨
for n in top_nodes:
    if n not in pos:
        continue
    title = str(G.nodes[n].get('title', n))
    short = title[:18] + '…' if len(title) > 18 else title
    year  = G.nodes[n].get('year', '')
    x, y  = pos[n]
    ax.annotate(
        f'{short}\n({year})',
        xy=(x, y), fontsize=6.5,
        ha='center', va='bottom',
        xytext=(0, 6), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
    )

# 범례
legend_patches = [
    mpatches.Patch(color=label_color[lbl], label=f'{lbl} ({sum(1 for n in G.nodes() if id2label.get(str(G.nodes[n].get("community_id","")), "기타") == lbl)}편)')
    for lbl in all_labels
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=9,
          framealpha=0.85, title='커뮤니티', title_fontsize=10)

ax.set_title('KCI AI 법학 논문 인용 네트워크\n(노드 크기: PageRank, 색상: 커뮤니티)',
             fontsize=14, pad=15)
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'viz_network.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('  [저장] viz_network.png')


# ════════════════════════════════════════════════════════════════════
# 5. 인터랙티브 시각화 (Plotly HTML) — 탐색용
# ════════════════════════════════════════════════════════════════════
print('\n▶ 인터랙티브 시각화 (HTML) 생성 중...')

try:
    import plotly.graph_objects as go
    import plotly.express as px

    traces = []

    # 엣지 (한 번에)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    traces.append(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(150,150,150,0.2)'),
        hoverinfo='none', showlegend=False,
    ))

    # 노드 (커뮤니티별)
    plotly_colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
    for i, lbl in enumerate(all_labels):
        nodes_in_lbl = [
            n for n in G.nodes()
            if id2label.get(str(G.nodes[n].get('community_id', '')), '기타') == lbl
        ]
        if not nodes_in_lbl:
            continue

        x_vals = [pos[n][0] for n in nodes_in_lbl]
        y_vals = [pos[n][1] for n in nodes_in_lbl]
        sizes  = [max(5, pr.get(n, 0) / max_pr * 40) for n in nodes_in_lbl]
        texts  = [
            f"{G.nodes[n].get('title','')[:50]}<br>"
            f"연도: {G.nodes[n].get('year','')}<br>"
            f"카테고리: {G.nodes[n].get('category','')}<br>"
            f"커뮤니티: {lbl}<br>"
            f"PageRank: {pr.get(n,0):.5f}"
            for n in nodes_in_lbl
        ]
        color = '#bbbbbb' if lbl == '기타' else plotly_colors[i % len(plotly_colors)]
        traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(size=sizes, color=color, opacity=0.85,
                        line=dict(width=0.5, color='white')),
            text=texts,
            hoverinfo='text',
            name=lbl,
        ))

    fig_html = go.Figure(
        data=traces,
        layout=go.Layout(
            title='KCI AI 법학 논문 인용 네트워크 (인터랙티브)',
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='#f8f8f8',
            plot_bgcolor='#f8f8f8',
            width=1200, height=900,
        )
    )
    fig_html.write_html(str(OUT / 'viz_network.html'))
    print('  [저장] viz_network.html')

except ImportError:
    print('  ※ plotly 미설치 — HTML 시각화 건너뜀')


print('\n' + '=' * 60)
print('  완료!')
print(f'  논문용:  results/03_network/viz_network.png')
print(f'  탐색용:  results/03_network/viz_network.html')
print('=' * 60)
