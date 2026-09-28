"""
03-2.network_viz.py
===================
확장 인용 네트워크 시각화

[03.network_viz.py와 차이점]
  - AI 논문(타입A): 커뮤니티별 색상, 채워진 원
  - 비AI KCI 논문(타입B): 회색 빈 원(작게) — 배경 역할
  - 범례에 노드 타입 구분 표시

- 입력:
    results/03-2_network/network_extended_giant_with_community.graphml
    results/03-2_network/community_summary.csv

- 출력:
    results/03-2_network/viz_network.png
    results/03-2_network/viz_network.html

사용법:
    python 03-2.network_viz.py
"""

import os, warnings
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
OUT = Path(__file__).parent / 'results' / '03-2_network'
OUT.mkdir(parents=True, exist_ok=True)

# ── 설정 ─────────────────────────────────────────────────────────
LABEL_TOP_N  = 20
NODE_SCALE   = 3000
LAYOUT_SEED  = 42


# ════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════════
print('=' * 60)
print('  03-2.network_viz.py — 확장 네트워크 시각화')
print('=' * 60)

print('\n▶ 그래프 + 커뮤니티 정보 로드 중...')
G = nx.read_graphml(str(OUT / 'network_extended_giant_with_community.graphml'))
df_summary = pd.read_csv(OUT / 'community_summary.csv', encoding='utf-8-sig')

n_ai  = sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'ai')
n_kci = sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'kci')
print(f'  노드: {G.number_of_nodes():,} (AI: {n_ai:,} / KCI: {n_kci:,})  '
      f'엣지: {G.number_of_edges():,}')

def get_label(row):
    fl = str(row.get('final_label', '')).strip()
    return fl if fl else str(row.get('auto_label', '기타')).strip()

id2label   = {str(r['community_id']): get_label(r) for _, r in df_summary.iterrows()}
all_labels = list(dict.fromkeys(
    id2label[str(r['community_id'])] for _, r in df_summary.iterrows()
))

base_colors = (list(plt.cm.Set2.colors) + list(plt.cm.Set1.colors) +
               list(plt.cm.tab20.colors))
label_color = {}
for i, lbl in enumerate(all_labels):
    label_color[lbl] = '#cccccc' if lbl == '기타' else base_colors[i % len(base_colors)]


# ════════════════════════════════════════════════════════════════════
# 2. 레이아웃
# ════════════════════════════════════════════════════════════════════
print('\n▶ 레이아웃 계산 중...')
try:
    from fa2 import ForceAtlas2
    fa2 = ForceAtlas2(outboundAttractionDistribution=True, verbose=False,
                      barnesHutOptimize=True, gravity=1.0)
    pos = fa2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    print('  레이아웃: Force Atlas 2')
except ImportError:
    pos = nx.spring_layout(G, k=1.5, seed=LAYOUT_SEED, weight='weight', iterations=80)
    print('  레이아웃: Fruchterman-Reingold (spring)')


# ════════════════════════════════════════════════════════════════════
# 3. PageRank
# ════════════════════════════════════════════════════════════════════
print('\n▶ PageRank 계산 중...')
pr      = nx.pagerank(G, weight='weight')
max_pr  = max(pr.values())
# 상위 N은 AI 논문 한정
top_nodes = set(n for n, _ in
    sorted([(n, v) for n, v in pr.items()
            if G.nodes[n].get('node_type') == 'ai'],
           key=lambda x: -x[1])[:LABEL_TOP_N])


# ════════════════════════════════════════════════════════════════════
# 4. 정적 시각화
# ════════════════════════════════════════════════════════════════════
print('\n▶ 정적 시각화 (PNG) 생성 중...')

fig, ax = plt.subplots(figsize=(20, 16))
ax.set_facecolor('#f5f5f5')
fig.patch.set_facecolor('#f5f5f5')

# 엣지
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, width=0.3, edge_color='#999999')

# 비AI KCI 노드 (작은 회색 빈 원) — 먼저 그려서 뒤에 깔리도록
kci_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'kci']
if kci_nodes:
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=kci_nodes,
                           node_size=15, node_color='white',
                           linewidths=0.5, edgecolors='#aaaaaa', alpha=0.6)

# AI 노드 (커뮤니티별 색상, PageRank 크기)
for lbl in all_labels:
    ai_in_lbl = [
        n for n in G.nodes()
        if G.nodes[n].get('node_type') == 'ai' and
           id2label.get(str(G.nodes[n].get('community_id', '')), '기타') == lbl
    ]
    if not ai_in_lbl:
        continue
    sizes = [pr.get(n, 0) / max_pr * NODE_SCALE + 25 for n in ai_in_lbl]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ai_in_lbl,
                           node_size=sizes, node_color=label_color[lbl],
                           alpha=0.88, linewidths=0.4, edgecolors='white')

# 상위 노드 라벨 (AI 논문만)
for n in top_nodes:
    if n not in pos:
        continue
    title = str(G.nodes[n].get('title', n))
    short = title[:18] + '…' if len(title) > 18 else title
    year  = G.nodes[n].get('year', '')
    x, y  = pos[n]
    ax.annotate(f'{short}\n({year})', xy=(x, y), fontsize=6,
                ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

# 범례 — 커뮤니티
legend_patches = [
    mpatches.Patch(
        color=label_color[lbl],
        label=f'{lbl} (AI {sum(1 for n in G.nodes() if G.nodes[n].get("node_type")=="ai" and id2label.get(str(G.nodes[n].get("community_id","")), "기타")==lbl)}편)'
    )
    for lbl in all_labels
]
# 비AI KCI 범례 추가
legend_patches.append(
    mpatches.Patch(facecolor='white', edgecolor='#aaaaaa',
                   label=f'비AI KCI 논문 ({n_kci:,}편)')
)
ax.legend(handles=legend_patches, loc='lower left', fontsize=8,
          framealpha=0.85, title='커뮤니티 / 노드 타입', title_fontsize=9)

ax.set_title('KCI AI 법학 논문 확장 인용 네트워크\n'
             '(색상: AI 논문 커뮤니티 / 회색 빈 원: 비AI KCI 피인용 논문 / 노드 크기: PageRank)',
             fontsize=13, pad=15)
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'viz_network.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('  [저장] viz_network.png')


# ════════════════════════════════════════════════════════════════════
# 5. 인터랙티브 (Plotly HTML)
# ════════════════════════════════════════════════════════════════════
print('\n▶ 인터랙티브 시각화 (HTML) 생성 중...')

try:
    import plotly.graph_objects as go
    import plotly.express as px

    traces = []

    # 엣지
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    traces.append(go.Scatter(x=ex, y=ey, mode='lines',
                             line=dict(width=0.4, color='rgba(150,150,150,0.15)'),
                             hoverinfo='none', showlegend=False))

    # 비AI KCI 노드
    if kci_nodes:
        kci_x = [pos[n][0] for n in kci_nodes]
        kci_y = [pos[n][1] for n in kci_nodes]
        kci_text = [f"{G.nodes[n].get('title','')[:40]}<br>연도: {G.nodes[n].get('year','')}"
                    for n in kci_nodes]
        traces.append(go.Scatter(
            x=kci_x, y=kci_y, mode='markers',
            marker=dict(size=4, color='white', opacity=0.6,
                        line=dict(width=0.5, color='#aaaaaa')),
            text=kci_text, hoverinfo='text', name=f'비AI KCI ({n_kci:,}편)',
        ))

    # AI 노드 (커뮤니티별)
    plotly_colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
    for i, lbl in enumerate(all_labels):
        ai_in_lbl = [
            n for n in G.nodes()
            if G.nodes[n].get('node_type') == 'ai' and
               id2label.get(str(G.nodes[n].get('community_id', '')), '기타') == lbl
        ]
        if not ai_in_lbl:
            continue
        color = '#bbbbbb' if lbl == '기타' else plotly_colors[i % len(plotly_colors)]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in ai_in_lbl],
            y=[pos[n][1] for n in ai_in_lbl],
            mode='markers',
            marker=dict(size=[max(5, pr.get(n,0)/max_pr*40) for n in ai_in_lbl],
                        color=color, opacity=0.88,
                        line=dict(width=0.5, color='white')),
            text=[f"{G.nodes[n].get('title','')[:50]}<br>"
                  f"연도: {G.nodes[n].get('year','')}<br>"
                  f"카테고리: {G.nodes[n].get('category','')}<br>"
                  f"커뮤니티: {lbl}<br>PageRank: {pr.get(n,0):.5f}"
                  for n in ai_in_lbl],
            hoverinfo='text', name=lbl,
        ))

    fig_html = go.Figure(
        data=traces,
        layout=go.Layout(
            title='KCI AI 법학 논문 확장 인용 네트워크 (인터랙티브)',
            showlegend=True, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='#f5f5f5', plot_bgcolor='#f5f5f5',
            width=1300, height=1000,
        )
    )
    fig_html.write_html(str(OUT / 'viz_network.html'))
    print('  [저장] viz_network.html')

except ImportError:
    print('  ※ plotly 미설치 — HTML 시각화 건너뜀')

print('\n' + '=' * 60)
print('  완료!')
print(f'  논문용:  results/03-2_network/viz_network.png')
print(f'  탐색용:  results/03-2_network/viz_network.html')
print('=' * 60)
