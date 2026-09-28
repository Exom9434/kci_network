"""
KCI 인공지능 논문 키워드 분석 스크립트
입력: 00.KCI_AI_논문_상세_및_인용데이터.csv
출력:
  - 01_keyword_heatmap.html   : 연도×키워드 히트맵
  - 02_keyword_trends.html    : 상위 12개 키워드 트렌드 라인차트
  - 03_network_XXXX_XXXX.html : 4개 시기별 공출현 네트워크
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import Counter, defaultdict
import itertools
# PNG 저장을 위해 kaleido 필요: pip install kaleido

# ── 설정 ────────────────────────────────────────────────────────
INPUT_FILE = '00.KCI_AI_논문_상세_및_인용데이터.csv'
OUT_DIR    = 'results/01_keyword'
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(fig, name):
    """HTML + PNG 동시 저장 (논문용 scale=2 고해상도)"""
    fig.write_html(f'{OUT_DIR}/{name}.html')
    fig.write_image(f'{OUT_DIR}/{name}.png', scale=2)
    print(f"  ✅ {OUT_DIR}/{name}.html / .png")

# 분석 제외 키워드 (모든 논문에 공통 → 변별력 없음)
EXCLUDE_KWS = {'인공지능', 'Artificial Intelligence', 'artificial intelligence',
               'AI', '인공지능(AI)', 'Artificial Intelligence(AI)',
               'Artificial Intelligence (AI)', 'Artificial intelligence'}

# 한영 통합 정규화 맵
NORMALIZE_MAP = {
    # 생성형 AI
    'Generative AI': '생성형 AI', 'generative AI': '생성형 AI',
    'Generative Artificial Intelligence': '생성형 AI',
    '생성형 인공지능': '생성형 AI', 'generative artificial intelligence': '생성형 AI',
    # 머신러닝
    'Machine Learning': '머신러닝', 'machine learning': '머신러닝',
    # 딥러닝
    'Deep Learning': '딥러닝', 'deep learning': '딥러닝',
    # 알고리즘
    'Algorithm': '알고리즘', 'algorithm': '알고리즘',
    # 저작권
    'Copyright': '저작권', 'copyright': '저작권',
    # 빅데이터
    'Big Data': '빅데이터', 'big data': '빅데이터',
    # 개인정보
    '개인정보보호': '개인정보', 'Personal Data': '개인정보',
    'personal data': '개인정보', 'Privacy': '개인정보', 'privacy': '개인정보',
    # 투명성
    'Transparency': '투명성', 'transparency': '투명성',
    # 거버넌스
    'Governance': '거버넌스', 'governance': '거버넌스',
    # AI Act
    'AI Act': 'EU AI Act',
    # 책임
    'Liability': '책임', 'liability': '책임',
    'Accountability': '책임', 'accountability': '책임',
    '법적 책임': '책임', '민사책임': '책임',
    # 자연어처리
    'Natural Language Processing': '자연어처리', 'NLP': '자연어처리',
    # ChatGPT/LLM
    'ChatGPT': 'ChatGPT/LLM', 'GPT': 'ChatGPT/LLM', 'LLM': 'ChatGPT/LLM',
    'Large Language Model': 'ChatGPT/LLM',
    '대규모 언어 모델': 'ChatGPT/LLM', '거대언어모델': 'ChatGPT/LLM',
    # 자율주행
    '자율주행자동차': '자율주행', '자율주행차': '자율주행',
    'Autonomous Vehicle': '자율주행', 'Autonomous Driving': '자율주행',
    # 로봇
    'Robot': '로봇', 'robot': '로봇',
    # 공정이용
    'Fair Use': '공정이용', 'fair use': '공정이용',
}

# 시기 구분
PERIODS = {
    '2016-2018': (2016, 2018),
    '2019-2020': (2019, 2020),
    '2021-2022': (2021, 2022),
    '2023-2025': (2023, 2025),
}

# ── 데이터 로드 및 전처리 ────────────────────────────────────────
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    papers = df.drop_duplicates(subset='source_id')[
        ['source_id', 'title', 'pub_year', 'keywords']].copy()
    papers = papers.dropna(subset=['keywords'])

    def parse_and_normalize(kw_str):
        kws = [k.strip() for k in kw_str.split(',') if len(k.strip()) > 1]
        kws = [NORMALIZE_MAP.get(k, k) for k in kws]
        kws = [k for k in kws if k not in EXCLUDE_KWS]
        return kws

    papers['kw_final'] = papers['keywords'].apply(parse_and_normalize)
    print(f"분석 대상: {len(papers)}개 논문 (키워드 있는 것)")
    print("연도별:")
    print(papers['pub_year'].value_counts().sort_index().to_string())
    return papers


# ── 히트맵 ──────────────────────────────────────────────────────
def make_heatmap(papers, top_n=20):
    years = sorted(papers['pub_year'].unique())
    all_kws = [kw for kws in papers['kw_final'] for kw in kws]
    top_kws = [k for k, v in Counter(all_kws).most_common(top_n)]

    freq_data = {}
    for year in years:
        yp = papers[papers['pub_year'] == year]
        ykws = [kw for kws in yp['kw_final'] for kw in kws]
        cnt = Counter(ykws)
        n = len(yp)
        freq_data[year] = {k: round(cnt.get(k, 0) / n * 100, 1) for k in top_kws}

    pivot = pd.DataFrame(freq_data, index=top_kws).T

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(y) for y in pivot.columns],
        y=pivot.index.tolist(),
        colorscale='YlOrRd',
        text=pivot.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='연도: %{x}<br>키워드: %{y}<br>출현율: %{z}%<extra></extra>',
        colorbar=dict(title='논문 대비<br>출현율(%)')
    ))
    fig.update_layout(
        title=dict(text='KCI 인공지능 논문 키워드 트렌드 히트맵 (2016-2025)', font=dict(size=18)),
        xaxis_title='발행 연도', yaxis_title='키워드',
        width=1000, height=700, font=dict(size=13),
        margin=dict(l=150, r=80, t=80, b=60)
    )
    save_fig(fig, '01_keyword_heatmap')


# ── 트렌드 라인차트 ──────────────────────────────────────────────
def make_trend_chart(papers, top_n=12):
    years = sorted(papers['pub_year'].unique())
    all_kws = [kw for kws in papers['kw_final'] for kw in kws]
    top_kws = [k for k, v in Counter(all_kws).most_common(top_n)]

    freq_data = {}
    for year in years:
        yp = papers[papers['pub_year'] == year]
        ykws = [kw for kws in yp['kw_final'] for kw in kws]
        cnt = Counter(ykws)
        n = len(yp)
        freq_data[year] = {k: cnt.get(k, 0) / n * 100 for k in top_kws}

    color_seq = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
    fig = go.Figure()
    for i, kw in enumerate(top_kws):
        fig.add_trace(go.Scatter(
            x=list(years),
            y=[freq_data[yr].get(kw, 0) for yr in years],
            mode='lines+markers',
            name=kw,
            line=dict(width=2.5, color=color_seq[i]),
            marker=dict(size=8)
        ))
    fig.update_layout(
        title=dict(text='주요 키워드 연도별 출현 추이 (논문 대비 %)', font=dict(size=18)),
        xaxis=dict(title='발행 연도', tickmode='linear', dtick=1),
        yaxis=dict(title='출현율 (%)'),
        legend=dict(x=1.01, y=1),
        width=1100, height=600, font=dict(size=13),
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray',
        xaxis_showgrid=True, yaxis_showgrid=True,
    )
    save_fig(fig, '02_keyword_trends')


# ── 공출현 네트워크 ──────────────────────────────────────────────
def build_cooccurrence(df_period, top_n=35, min_cooc=2):
    all_kws = [kw for kws in df_period['kw_final'] for kw in kws]
    top_kws = set([k for k, v in Counter(all_kws).most_common(top_n)])
    node_freq = Counter(all_kws)
    cooc = defaultdict(int)
    for kws in df_period['kw_final']:
        filtered = list(set(k for k in kws if k in top_kws))
        for pair in itertools.combinations(sorted(filtered), 2):
            cooc[pair] += 1
    G = nx.Graph()
    for kw in top_kws:
        if node_freq[kw] >= 2:
            G.add_node(kw, freq=node_freq[kw])
    for (a, b), cnt in cooc.items():
        if cnt >= min_cooc and G.has_node(a) and G.has_node(b):
            G.add_edge(a, b, weight=cnt)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def make_network(papers, period_name, y1, y2):
    df_p = papers[(papers['pub_year'] >= y1) & (papers['pub_year'] <= y2)]
    n = len(df_p)
    min_c = 2 if n < 80 else 3
    G = build_cooccurrence(df_p, min_cooc=min_c)
    print(f"  {period_name}: 논문 {n}개 → 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, k=2.5, seed=42, weight='weight')
    node_freq = nx.get_node_attributes(G, 'freq')
    max_freq = max(node_freq.values()) if node_freq else 1

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        node_comm = {n: i for i, comm in enumerate(communities) for n in comm}
    except Exception:
        node_comm = {n: 0 for n in G.nodes()}

    edge_weights = nx.get_edge_attributes(G, 'weight')
    max_w = max(edge_weights.values()) if edge_weights else 1
    color_palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
                     '#a65628','#f781bf','#999999','#66c2a5','#fc8d62']

    traces = []
    for (u, v), w in edge_weights.items():
        x0, y0 = pos[u]; x1, y1c = pos[v]
        op = 0.3 + 0.5 * (w / max_w)
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1c, None],
            mode='lines',
            line=dict(width=1 + 3*(w/max_w), color=f'rgba(100,100,100,{op:.2f})'),
            hoverinfo='none', showlegend=False
        ))
    for cid in set(node_comm.values()):
        nodes = [nd for nd, c in node_comm.items() if c == cid]
        traces.append(go.Scatter(
            x=[pos[nd][0] for nd in nodes],
            y=[pos[nd][1] for nd in nodes],
            mode='markers+text',
            marker=dict(
                size=[15 + 25*(node_freq.get(nd,1)/max_freq) for nd in nodes],
                color=color_palette[cid % len(color_palette)],
                opacity=0.85, line=dict(width=1.5, color='white')
            ),
            text=nodes, textposition='top center', textfont=dict(size=11),
            hovertemplate='<b>%{text}</b><br>출현: %{customdata}회<extra></extra>',
            customdata=[node_freq.get(nd, 0) for nd in nodes],
            name=f'군집 {cid+1}'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f'키워드 공출현 네트워크: {period_name} (논문 {n}편)', font=dict(size=16)),
        showlegend=True, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#fafafa', width=900, height=700,
        legend=dict(title='군집', x=1.01, y=0.5)
    )
    save_fig(fig, f'03_network_{period_name.replace("-","_")}')


# ── 메인 실행 ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("KCI 인공지능 논문 키워드 분석 시작")
    print("=" * 50)

    papers = load_and_preprocess(INPUT_FILE)
    print()

    print("[1/3] 히트맵 생성 중...")
    make_heatmap(papers, top_n=20)

    print("[2/3] 트렌드 라인차트 생성 중...")
    make_trend_chart(papers, top_n=12)

    print("[3/3] 공출현 네트워크 생성 중...")
    for period_name, (y1, y2) in PERIODS.items():
        make_network(papers, period_name, y1, y2)

    print()
    print("=" * 50)
    print("✅ 모든 분석 완료! 생성된 파일:")
    print(f"   {OUT_DIR}/01_keyword_heatmap.html")
    print(f"   {OUT_DIR}/02_keyword_trends.html")
    print(f"   {OUT_DIR}/03_network_2016_2018.html")
    print(f"   {OUT_DIR}/03_network_2019_2020.html")
    print(f"   {OUT_DIR}/03_network_2021_2022.html")
    print(f"   {OUT_DIR}/03_network_2023_2025.html")
    print("=" * 50)
