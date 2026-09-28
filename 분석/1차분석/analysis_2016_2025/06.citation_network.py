"""
KCI 인공지능 논문 인용 네트워크 분석 스크립트

사전 설치: pip install networkx plotly pandas

입력: 00.KCI_AI_논문_상세_및_인용데이터.csv

출력 (A. 내부 네트워크 — AI논문 2014개끼리):
  08a_internal_network_full.html       전체 누적 네트워크
  09a_internal_pagerank_trend.html     연도별 PageRank 상위 논문 변화
  10a_internal_community_time.html     연도별 군집 구성 변화

출력 (B. 확장 네트워크 — AI논문 + 인용된 KCI논문 포함):
  08b_extended_network_full.html       전체 누적 네트워크
  09b_cited_domain_trend.html          연도별 피인용 외부논문 카테고리 분포
  10b_extended_community_time.html     연도별 군집 변화

공통:
  citation_stats.txt                   분석 요약 텍스트

※ 2025년은 인용 데이터 없어서 분석 제외 (2016-2024 기준)
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
INPUT_FILE = '00.KCI_AI_논문_상세_및_인용데이터.csv'
OUT_DIR    = 'results/03_citation'
os.makedirs(OUT_DIR, exist_ok=True)
# PNG 저장을 위해 kaleido 필요: pip install kaleido

def save_fig(fig, name):
    """HTML + PNG 동시 저장 (논문용 scale=2 고해상도)"""
    fig.write_html(f'{OUT_DIR}/{name}.html')
    fig.write_image(f'{OUT_DIR}/{name}.png', scale=2)
    print(f"  ✅ {OUT_DIR}/{name}.html / .png")
YEAR_START = 2016
YEAR_END   = 2024   # 2025 제외 (인용 데이터 없음)

# 시기 구분 (슬라이딩 + 누적 둘 다 사용)
PERIODS = {
    '2016-2018': (2016, 2018),
    '2019-2020': (2019, 2020),
    '2021-2022': (2021, 2022),
    '2023-2024': (2023, 2024),
}

# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════
print("▶ 데이터 로드 중...")
df = pd.read_csv(INPUT_FILE)
df['source_id']      = df['source_id'].astype(str)
df['target_arti_id'] = df['target_arti_id'].fillna('').astype(str)

# AI 논문 메타데이터 (2016-2024)
papers = (df.drop_duplicates(subset='source_id')
            [['source_id','title','pub_year','category']]
            .query(f"{YEAR_START} <= pub_year <= {YEAR_END}")
            .copy())
ai_ids = set(papers['source_id'])
id2year  = papers.set_index('source_id')['pub_year'].to_dict()
id2title = papers.set_index('source_id')['title'].to_dict()

print(f"  AI 논문 (2016-2024): {len(ai_ids):,}개")

# 인용 엣지 (source가 AI논문이고 연도 범위 내)
edges_df = df[
    (df['source_id'].isin(ai_ids)) &
    (df['target_arti_id'] != '') &
    (df['target_arti_id'].str.startswith('ART'))
][['source_id','target_arti_id']].drop_duplicates()

# 내부 엣지: target도 AI논문
internal_edges = edges_df[edges_df['target_arti_id'].isin(ai_ids)]
# 외부 엣지: target이 AI논문 밖 KCI 논문
external_edges = edges_df[~edges_df['target_arti_id'].isin(ai_ids)]

print(f"  내부 인용 엣지: {len(internal_edges):,}건")
print(f"  외부 KCI 인용 엣지: {len(external_edges):,}건")
print()


# ══════════════════════════════════════════════════════════════════
# 유틸: Plotly 네트워크 그래프 생성
# ══════════════════════════════════════════════════════════════════
def build_plotly_network(G, title, node_year=None, top_n_label=30):
    """NetworkX 그래프 → Plotly Figure"""
    if G.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(G, k=1.5, seed=42,
                           weight='weight' if nx.is_weighted(G) else None)

    # PageRank로 노드 크기
    try:
        pr = nx.pagerank(G, weight='weight')
    except Exception:
        pr = {n: 1 for n in G.nodes()}
    max_pr = max(pr.values()) if pr else 1

    # 군집 감지 (색상)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        node_comm = {n: i for i, c in enumerate(communities) for n in c}
    except Exception:
        node_comm = {n: 0 for n in G.nodes()}

    color_pal = px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1

    traces = []
    # 엣지
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
            hoverinfo='none', showlegend=False
        ))

    # 노드 (군집별로 분리해서 legend 표시)
    top_nodes = sorted(pr, key=pr.get, reverse=True)[:top_n_label]
    for cid in sorted(set(node_comm.values())):
        nodes = [n for n, c in node_comm.items() if c == cid]
        yr_label = [str(node_year.get(n, '?')) if node_year else '' for n in nodes]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode='markers+text',
            marker=dict(
                size=[8 + 30*(pr.get(n,0)/max_pr) for n in nodes],
                color=color_pal[cid % len(color_pal)],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[id2title.get(n, n)[:20] if n in top_nodes else '' for n in nodes],
            textposition='top center',
            textfont=dict(size=9),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                '연도: %{customdata[1]}<br>'
                'PageRank: %{customdata[2]}<extra></extra>'
            ),
            customdata=[[id2title.get(n, n), node_year.get(n,'?') if node_year else '?',
                         f"{pr.get(n,0):.4f}"] for n in nodes],
            name=f'군집 {cid+1}'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True, hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f8f8f8',
        margin=dict(b=20, l=5, r=5, t=60),
        width=1100, height=800,
        legend=dict(title='군집', x=1.01, y=0.5)
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# A. 내부 네트워크 분석
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("A. 내부 네트워크 (AI논문 → AI논문)")
print("=" * 55)

# A-1. 전체 누적 네트워크
G_int = nx.DiGraph()
for nid in ai_ids:
    G_int.add_node(nid)
for _, row in internal_edges.iterrows():
    G_int.add_edge(row['source_id'], row['target_arti_id'])

# 고립 노드 제거
G_int.remove_nodes_from(list(nx.isolates(G_int)))
print(f"전체 누적: 노드 {G_int.number_of_nodes()}, 엣지 {G_int.number_of_edges()}")

fig = build_plotly_network(
    G_int.to_undirected(),
    title=f'AI논문 내부 인용 네트워크 전체 (2016-2024)',
    node_year=id2year
)
save_fig(fig, '08a_internal_network_full')

# A-2. PageRank 상위 논문 연도별 변화 (누적)
print("\n  PageRank 상위 논문 누적 추이 계산 중...")
pr_records = []
for year in range(YEAR_START, YEAR_END + 1):
    # 해당 연도까지 누적된 엣지
    cum_src = {nid for nid in ai_ids if id2year.get(nid, 9999) <= year}
    cum_edges = internal_edges[
        internal_edges['source_id'].isin(cum_src) &
        internal_edges['target_arti_id'].isin(cum_src)
    ]
    if len(cum_edges) == 0:
        continue
    Gc = nx.DiGraph()
    Gc.add_nodes_from(cum_src)
    for _, r in cum_edges.iterrows():
        Gc.add_edge(r['source_id'], r['target_arti_id'])
    pr = nx.pagerank(Gc, weight=None)
    top10 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (nid, score) in enumerate(top10, 1):
        pr_records.append({
            '연도': year, '순위': rank,
            '논문ID': nid,
            '제목': id2title.get(nid, nid)[:35],
            '발행연도': id2year.get(nid, '?'),
            'PageRank': round(score, 5)
        })

df_pr = pd.DataFrame(pr_records)
df_pr.to_csv(f'{OUT_DIR}/pagerank_trend_data.csv', index=False, encoding='utf-8-sig')

# 상위 5개 논문의 PageRank 시계열
top5_ids = (df_pr[df_pr['연도'] == YEAR_END]
            .nsmallest(5, '순위')['논문ID'].tolist())

fig2 = go.Figure()
for nid in top5_ids:
    sub = df_pr[df_pr['논문ID'] == nid]
    fig2.add_trace(go.Scatter(
        x=sub['연도'], y=sub['PageRank'],
        mode='lines+markers',
        name=id2title.get(nid, nid)[:30],
        hovertemplate='%{x}년<br>PageRank: %{y:.5f}<extra></extra>'
    ))
fig2.update_layout(
    title='AI논문 내부 인용 — PageRank 상위 논문 누적 추이',
    xaxis=dict(title='기준 연도', dtick=1),
    yaxis=dict(title='PageRank'),
    width=1000, height=550, font=dict(size=12),
    hovermode='x unified', plot_bgcolor='white',
    xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
    xaxis_showgrid=True, yaxis_showgrid=True,
    legend=dict(x=0, y=1)
)
save_fig(fig2, '09a_internal_pagerank_trend')

# A-3. 시기별 군집 변화 (슬라이딩)
print("\n  시기별 군집 분석 중...")
period_stats = []
for period_name, (y1, y2) in PERIODS.items():
    period_src = {nid for nid in ai_ids if y1 <= id2year.get(nid, 0) <= y2}
    period_edges = internal_edges[
        internal_edges['source_id'].isin(period_src) &
        internal_edges['target_arti_id'].isin(period_src)
    ]
    if len(period_edges) == 0:
        continue
    Gp = nx.Graph()
    for _, r in period_edges.iterrows():
        Gp.add_edge(r['source_id'], r['target_arti_id'])
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(Gp))
        modularity = nx.algorithms.community.quality.modularity(Gp, comms)
    except Exception:
        comms = []; modularity = 0
    period_stats.append({
        '시기': period_name, '논문수': len(period_src),
        '엣지수': Gp.number_of_edges(), '노드수': Gp.number_of_nodes(),
        '군집수': len(comms), '모듈성': round(modularity, 3)
    })
    print(f"  {period_name}: 노드 {Gp.number_of_nodes()}, 엣지 {Gp.number_of_edges()}, 군집 {len(comms)}개, 모듈성 {modularity:.3f}")

df_period = pd.DataFrame(period_stats)

fig3 = go.Figure()
fig3.add_trace(go.Bar(x=df_period['시기'], y=df_period['군집수'],
                       name='군집 수', marker_color='#4daf4a'))
fig3.add_trace(go.Scatter(x=df_period['시기'], y=df_period['모듈성'],
                           name='모듈성', yaxis='y2',
                           mode='lines+markers', line=dict(color='#e41a1c', width=2.5),
                           marker=dict(size=10)))
fig3.update_layout(
    title='시기별 내부 인용 네트워크 군집 구조 변화',
    yaxis=dict(title='군집 수'),
    yaxis2=dict(title='모듈성 (0~1)', overlaying='y', side='right'),
    width=900, height=500, font=dict(size=13),
    plot_bgcolor='white', legend=dict(x=0.7, y=0.95)
)
save_fig(fig3, '10a_internal_community_time')


# ══════════════════════════════════════════════════════════════════
# B. 확장 네트워크 분석 (AI논문 + 외부 KCI 피인용 논문)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("B. 확장 네트워크 (AI논문 + 인용된 외부 KCI논문)")
print("=" * 55)

# B-1. 전체 누적 확장 네트워크
all_edges = pd.concat([internal_edges, external_edges], ignore_index=True)
ext_ids = set(external_edges['target_arti_id'].unique())  # 외부 KCI 논문
all_nodes = ai_ids | ext_ids

G_ext = nx.DiGraph()
G_ext.add_nodes_from(all_nodes)
for _, row in all_edges.iterrows():
    G_ext.add_edge(row['source_id'], row['target_arti_id'])
G_ext.remove_nodes_from(list(nx.isolates(G_ext)))

print(f"확장 네트워크: 노드 {G_ext.number_of_nodes()}, 엣지 {G_ext.number_of_edges()}")
print(f"  AI논문 노드: {len(ai_ids & set(G_ext.nodes()))}")
print(f"  외부KCI 노드: {len(ext_ids & set(G_ext.nodes()))}")

# 외부 노드가 많아서 시각화는 AI논문 노드만 표시
G_ext_vis = G_ext.subgraph(ai_ids & set(G_ext.nodes())).copy()
# 단 외부 노드로의 엣지 수를 노드 속성으로
out_to_ext = external_edges.groupby('source_id').size().to_dict()
for n in G_ext_vis.nodes():
    G_ext_vis.nodes[n]['ext_out'] = out_to_ext.get(n, 0)

fig_b1 = build_plotly_network(
    G_ext_vis.to_undirected(),
    title='확장 인용 네트워크 — AI논문 노드 (외부 인용 연결 포함)',
    node_year=id2year
)
save_fig(fig_b1, '08b_extended_network_full')

# B-2. 연도별 외부 KCI 인용 카테고리 분포 (외부 논문의 category 정보 필요)
# → 외부 논문 category는 우리 데이터에 없으므로 대신 "피인용 빈도 Top 논문" 분석
print("\n  외부 피인용 빈도 분석 중...")
ext_cited_cnt = external_edges.groupby('target_arti_id').size().sort_values(ascending=False)
print(f"  외부 KCI 피인용 고유 논문: {len(ext_cited_cnt):,}개")
print(f"  상위 10개 피인용 외부 논문:")
for tid, cnt in ext_cited_cnt.head(10).items():
    print(f"    {tid}: {cnt}회")

# 연도별 외부 인용 건수 추이
ext_by_year = (external_edges
               .merge(papers[['source_id','pub_year']], on='source_id')
               .groupby('pub_year').size()
               .reset_index(name='외부인용건수'))
int_by_year = (internal_edges
               .merge(papers[['source_id','pub_year']], on='source_id')
               .groupby('pub_year').size()
               .reset_index(name='내부인용건수'))
cnt_by_year = ext_by_year.merge(int_by_year, on='pub_year', how='outer').fillna(0)
cnt_by_year['총인용'] = cnt_by_year['외부인용건수'] + cnt_by_year['내부인용건수']
cnt_by_year['내부비율(%)'] = (cnt_by_year['내부인용건수'] / cnt_by_year['총인용'] * 100).round(1)

fig_b2 = go.Figure()
fig_b2.add_trace(go.Bar(x=cnt_by_year['pub_year'], y=cnt_by_year['내부인용건수'],
                         name='AI→AI 내부', marker_color='#377eb8'))
fig_b2.add_trace(go.Bar(x=cnt_by_year['pub_year'], y=cnt_by_year['외부인용건수'],
                         name='AI→외부KCI', marker_color='#ff7f00'))
fig_b2.add_trace(go.Scatter(x=cnt_by_year['pub_year'], y=cnt_by_year['내부비율(%)'],
                              name='내부인용 비율(%)', yaxis='y2',
                              mode='lines+markers', line=dict(color='#e41a1c', width=2.5),
                              marker=dict(size=9)))
fig_b2.update_layout(
    title='연도별 인용 유형 구성 (내부 AI논문 vs 외부 KCI논문)',
    barmode='stack',
    xaxis=dict(title='발행 연도', dtick=1),
    yaxis=dict(title='인용 건수'),
    yaxis2=dict(title='내부인용 비율(%)', overlaying='y', side='right', range=[0,100]),
    width=1000, height=550, font=dict(size=13),
    plot_bgcolor='white', legend=dict(x=0.01, y=0.95)
)
save_fig(fig_b2, '09b_cited_domain_trend')

# B-3. 시기별 확장 네트워크 군집 변화
print("\n  시기별 확장 네트워크 군집 분석 중...")
period_stats_b = []
for period_name, (y1, y2) in PERIODS.items():
    period_src = {nid for nid in ai_ids if y1 <= id2year.get(nid, 0) <= y2}
    p_edges = all_edges[all_edges['source_id'].isin(period_src)]
    if len(p_edges) == 0:
        continue
    Gp = nx.Graph()
    for _, r in p_edges.iterrows():
        Gp.add_edge(r['source_id'], r['target_arti_id'])
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(Gp))
        modularity = nx.algorithms.community.quality.modularity(Gp, comms)
    except Exception:
        comms = []; modularity = 0
    period_stats_b.append({
        '시기': period_name, '총노드': Gp.number_of_nodes(),
        '엣지수': Gp.number_of_edges(), '군집수': len(comms),
        '모듈성': round(modularity, 3)
    })
    print(f"  {period_name}: 노드 {Gp.number_of_nodes()}, 엣지 {Gp.number_of_edges()}, 군집 {len(comms)}개")

df_period_b = pd.DataFrame(period_stats_b)
fig_b3 = go.Figure()
fig_b3.add_trace(go.Bar(x=df_period_b['시기'], y=df_period_b['군집수'],
                         name='군집 수', marker_color='#984ea3'))
fig_b3.add_trace(go.Scatter(x=df_period_b['시기'], y=df_period_b['모듈성'],
                              name='모듈성', yaxis='y2',
                              mode='lines+markers', line=dict(color='#ff7f00', width=2.5),
                              marker=dict(size=10)))
fig_b3.update_layout(
    title='시기별 확장 인용 네트워크 군집 구조 변화',
    yaxis=dict(title='군집 수'),
    yaxis2=dict(title='모듈성', overlaying='y', side='right'),
    width=900, height=500, font=dict(size=13),
    plot_bgcolor='white', legend=dict(x=0.7, y=0.95)
)
save_fig(fig_b3, '10b_extended_community_time')


# ══════════════════════════════════════════════════════════════════
# 요약 텍스트 저장
# ══════════════════════════════════════════════════════════════════
with open(f'{OUT_DIR}/citation_stats.txt', 'w', encoding='utf-8') as f:
    f.write("KCI 인공지능 논문 인용 네트워크 분석 요약\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"분석 기간: {YEAR_START} ~ {YEAR_END} (2025 제외)\n")
    f.write(f"AI 논문 수: {len(ai_ids):,}개\n")
    f.write(f"내부 인용 엣지: {len(internal_edges):,}건\n")
    f.write(f"외부 KCI 인용 엣지: {len(external_edges):,}건\n\n")
    f.write("[A. 내부 네트워크]\n")
    f.write(f"  연결 노드: {G_int.number_of_nodes()}\n")
    f.write(f"  엣지: {G_int.number_of_edges()}\n\n")
    f.write("[시기별 내부 네트워크 요약]\n")
    f.write(df_period.to_string(index=False) + "\n\n")
    f.write("[B. 확장 네트워크]\n")
    f.write(f"  총 노드: {G_ext.number_of_nodes()}\n")
    f.write(f"  엣지: {G_ext.number_of_edges()}\n\n")
    f.write("[연도별 인용 유형 구성]\n")
    f.write(cnt_by_year.to_string(index=False) + "\n\n")
    f.write("[외부 피인용 상위 20개 논문]\n")
    for tid, cnt in ext_cited_cnt.head(20).items():
        f.write(f"  {tid}: {cnt}회\n")

print()
print("=" * 55)
print("✅ 인용 네트워크 분석 완료! 생성된 파일:")
print("  [A 내부]")
print("   08a_internal_network_full.html")
print("   09a_internal_pagerank_trend.html")
print("   10a_internal_community_time.html")
print("  [B 확장]")
print("   08b_extended_network_full.html")
print("   09b_cited_domain_trend.html")
print("   10b_extended_community_time.html")
print("  [공통]")ㄴ
print(f"   {OUT_DIR}/citation_stats.txt")
print(f"   {OUT_DIR}/pagerank_trend_data.csv")
print("=" * 55)
