"""
AI 법학 논문 카테고리 기반 네트워크/클러스터 분석
=====================================================
입력: 05.논문_카테고리_최종분류.csv  (카테고리 분류 결과)
      ../AI_법학_논문목록.csv         (저자 정보)

출력 (results/09_category_network/ 폴더):
  01_category_cooccur_network.html/png  카테고리 공존 네트워크 (노드=분야, 엣지=공존 논문 수)
  02_author_collab_network.html/png     공저자 네트워크 (노드=저자, 색상=주요분야)
  03_journal_category_heatmap.html/png  학술지 × 카테고리 분포 히트맵
  09_network_stats.csv                  네트워크 기본 통계
"""

import os
import re
import math
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter, defaultdict

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
INPUT_CLASS  = "05.논문_카테고리_최종분류.csv"
INPUT_PAPERS = "../AI_법학_논문목록.csv"
OUT_DIR      = "results/09_category_network"
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_START = 2016
YEAR_END   = 2025

CORE_CATS = [
    "데이터법", "인공지능법", "경제법", "지식재산권법",
    "형사법", "공법", "민사법", "금융법",
    "노동법", "사회보장법", "조세법", "의료법", "기초법",
]

# 분야별 고정 색상 (네트워크 노드 컬러)
CAT_COLORS = {
    "데이터법":     "#1f77b4",
    "인공지능법":   "#ff7f0e",
    "경제법":       "#2ca02c",
    "지식재산권법": "#d62728",
    "형사법":       "#9467bd",
    "공법":         "#8c564b",
    "민사법":       "#e377c2",
    "금융법":       "#7f7f7f",
    "노동법":       "#bcbd22",
    "사회보장법":   "#17becf",
    "조세법":       "#aec7e8",
    "의료법":       "#ffbb78",
    "기초법":       "#98df8a",
    "기타":         "#cccccc",
}


def save_fig(fig, name: str):
    fig.write_html(f"{OUT_DIR}/{name}.html")
    try:
        fig.write_image(f"{OUT_DIR}/{name}.png", scale=2)
    except Exception as e:
        print(f"  ⚠ PNG 저장 실패 ({name}): {e}")
    print(f"  ✅ {OUT_DIR}/{name}.html / .png")


def extract_primary_cats(cat_str: str) -> list[str]:
    cat_str = str(cat_str).strip()
    if cat_str.startswith("융복합"):
        inner = re.search(r"\((.+)\)", cat_str)
        if inner:
            return [c.strip() for c in inner.group(1).split("|") if c.strip()]
    return [cat_str]


def get_main_cat(cat_str: str) -> str:
    """단일 대표 카테고리 반환 (융복합이면 첫 번째)"""
    cats = extract_primary_cats(cat_str)
    return cats[0] if cats else "기타"


# ══════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════
print("▶ 데이터 로드 중...")
df_cls = pd.read_csv(INPUT_CLASS, encoding="utf-8-sig")
df_cls = df_cls[df_cls["발행연도"].between(YEAR_START, YEAR_END)].copy()
df_cls["구성분야"] = df_cls["최종카테고리"].apply(extract_primary_cats)
df_cls["대표분야"] = df_cls["최종카테고리"].apply(get_main_cat)
print(f"  분류 데이터: {len(df_cls)}편")

# 저자 정보 합치기
df_papers = pd.read_csv(INPUT_PAPERS, encoding="utf-8-sig")
df = df_cls.merge(df_papers[["논문ID", "저자", "학술지명"]], on="논문ID", how="left",
                  suffixes=("_cls", "_p"))
# 학술지명: 분류 파일 우선, 없으면 논문목록에서
if "학술지명_cls" in df.columns:
    df["학술지명"] = df["학술지명_cls"].fillna(df.get("학술지명_p", ""))
elif "학술지명" not in df.columns:
    df["학술지명"] = ""
print(f"  저자 정보 병합 후: {len(df)}편")


# ══════════════════════════════════════════════════════
# 2. 카테고리 공존 네트워크 (노드=분야, 엣지=공존 논문 수)
# ══════════════════════════════════════════════════════
print("\n▶ [1/3] 카테고리 공존 네트워크 구성 중...")

# 각 분야의 단독 출현 빈도 (노드 크기)
cat_freq: Counter = Counter()
for cats in df["구성분야"]:
    for c in cats:
        cat_freq[c] += 1

# 분야 쌍 공존 빈도 (엣지 가중치)
pair_freq: Counter = Counter()
for cats in df["구성분야"]:
    cats_u = sorted(set(cats))
    for i in range(len(cats_u)):
        for j in range(i + 1, len(cats_u)):
            pair_freq[(cats_u[i], cats_u[j])] += 1

# NetworkX 그래프 생성
G_cat = nx.Graph()
for cat, freq in cat_freq.items():
    G_cat.add_node(cat, freq=freq, color=CAT_COLORS.get(cat, "#cccccc"))

MIN_EDGE = 3  # 최소 공존 논문 수 이상인 엣지만 포함
for (c1, c2), w in pair_freq.items():
    if w >= MIN_EDGE:
        G_cat.add_edge(c1, c2, weight=w)

print(f"  노드: {G_cat.number_of_nodes()}개  엣지: {G_cat.number_of_edges()}개 (공존 ≥ {MIN_EDGE}편)")

# 레이아웃 계산 (spring layout)
np.random.seed(42)
pos = nx.spring_layout(G_cat, k=2.5, weight="weight", seed=42)

# Plotly 시각화
edge_traces = []
for u, v, data in G_cat.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    w = data.get("weight", 1)
    edge_traces.append(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(width=max(0.5, w / 8), color="rgba(150,150,150,0.6)"),
        hoverinfo="none",
        showlegend=False,
    ))

# 엣지 레이블 (가중치 표시)
edge_label_traces = []
for u, v, data in G_cat.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    w = data.get("weight", 1)
    if w >= 10:  # 10편 이상인 엣지만 수치 표시
        edge_label_traces.append(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            mode="text",
            text=[str(w)],
            textfont=dict(size=8, color="#555"),
            showlegend=False,
            hoverinfo="none",
        ))

node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
for node in G_cat.nodes():
    x, y = pos[node]
    freq = cat_freq[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_color.append(CAT_COLORS.get(node, "#cccccc"))
    node_size.append(max(20, min(60, freq / 3)))
    node_hover.append(f"{node}<br>단독 출현: {freq}편<br>연결 분야: {G_cat.degree(node)}개")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="white")),
    text=node_text,
    textposition="top center",
    textfont=dict(size=11),
    hovertext=node_hover,
    hoverinfo="text",
    showlegend=False,
)

fig_cat_net = go.Figure(
    data=edge_traces + edge_label_traces + [node_trace],
    layout=go.Layout(
        title=dict(text="AI 법학 논문 분야 간 공존 네트워크", font=dict(size=18)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=900, height=750,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
        annotations=[dict(
            text=f"노드 크기 = 논문 수 (단독 기준), 엣지 굵기 = 공존 빈도 (≥{MIN_EDGE}편)",
            xref="paper", yref="paper", x=0.5, y=-0.02,
            showarrow=False, font=dict(size=10, color="#888"),
        )],
    ),
)
save_fig(fig_cat_net, "01_category_cooccur_network")


# ══════════════════════════════════════════════════════
# 3. 공저자 네트워크 (노드=저자, 색상=대표분야)
# ══════════════════════════════════════════════════════
print("▶ [2/3] 공저자 네트워크 구성 중...")

# 저자 파싱: '홍길동(서울대학교), 이순신(연세대)' → ['홍길동', '이순신']
def parse_authors(author_str) -> list[str]:
    if pd.isna(author_str) or not str(author_str).strip():
        return []
    # 괄호 안 소속 제거
    cleaned = re.sub(r"\([^)]*\)", "", str(author_str))
    # 쉼표/세미콜론/공백 등으로 분리
    authors = [a.strip() for a in re.split(r"[,;]", cleaned) if a.strip()]
    return authors

df["저자목록"] = df["저자"].apply(parse_authors)

# 저자별 대표 분야 (가장 많이 쓴 카테고리)
author_cat: defaultdict = defaultdict(Counter)
for _, row in df.iterrows():
    for author in row["저자목록"]:
        author_cat[author][row["대표분야"]] += 1

author_main_cat = {
    author: cnt.most_common(1)[0][0]
    for author, cnt in author_cat.items()
}

# 공저 쌍 추출
coauthor_freq: Counter = Counter()
for _, row in df.iterrows():
    authors = row["저자목록"]
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            pair = tuple(sorted([authors[i], authors[j]]))
            coauthor_freq[pair] += 1

# 저자 출현 빈도
author_freq: Counter = Counter()
for authors in df["저자목록"]:
    for a in authors:
        author_freq[a] += 1

# 논문 2편 이상인 저자만 포함 (네트워크 가독성)
MIN_PAPER  = 2
MIN_COAUTH = 1  # 최소 공저 횟수

active_authors = {a for a, cnt in author_freq.items() if cnt >= MIN_PAPER}
print(f"  논문 {MIN_PAPER}편 이상 저자: {len(active_authors)}명")

G_author = nx.Graph()
for author in active_authors:
    G_author.add_node(
        author,
        freq=author_freq[author],
        cat=author_main_cat.get(author, "기타"),
    )

for (a1, a2), w in coauthor_freq.items():
    if a1 in active_authors and a2 in active_authors and w >= MIN_COAUTH:
        G_author.add_edge(a1, a2, weight=w)

# 고립 노드 제거 (엣지 없는 저자)
isolates = list(nx.isolates(G_author))
G_author.remove_nodes_from(isolates)
print(f"  저자 네트워크: 노드 {G_author.number_of_nodes()}명, 엣지 {G_author.number_of_edges()}개")

if G_author.number_of_nodes() > 0:
    np.random.seed(42)
    pos_a = nx.spring_layout(G_author, k=1.5, weight="weight", seed=42)

    edge_traces_a = []
    for u, v, data in G_author.edges(data=True):
        x0, y0 = pos_a[u]
        x1, y1 = pos_a[v]
        edge_traces_a.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=1, color="rgba(180,180,180,0.5)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # 분야별 레이어로 분리 (범례용)
    cats_present = sorted(set(nx.get_node_attributes(G_author, "cat").values()))
    for cat in cats_present:
        nodes_in_cat = [n for n, d in G_author.nodes(data=True) if d.get("cat") == cat]
        if not nodes_in_cat:
            continue
        nx_list = [pos_a[n][0] for n in nodes_in_cat]
        ny_list = [pos_a[n][1] for n in nodes_in_cat]
        sizes   = [max(8, min(25, G_author.nodes[n]["freq"] * 4)) for n in nodes_in_cat]
        labels  = nodes_in_cat
        hover   = [f"{n}<br>논문: {G_author.nodes[n]['freq']}편<br>분야: {cat}" for n in nodes_in_cat]

        edge_traces_a.append(go.Scatter(
            x=nx_list, y=ny_list,
            mode="markers+text",
            name=cat,
            marker=dict(size=sizes, color=CAT_COLORS.get(cat, "#ccc"),
                        line=dict(width=1, color="white")),
            text=labels,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover,
            hoverinfo="text",
        ))

    fig_author = go.Figure(
        data=edge_traces_a,
        layout=go.Layout(
            title=dict(text="AI 법학 논문 공저자 네트워크 (논문 2편↑ 저자)", font=dict(size=17)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(title="주요 연구분야"),
            width=1050, height=800,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor="white",
        ),
    )
    save_fig(fig_author, "02_author_collab_network")
else:
    print("  ⚠ 공저자 네트워크 노드가 없어 건너뜁니다.")


# ══════════════════════════════════════════════════════
# 4. 학술지 × 카테고리 히트맵
# ══════════════════════════════════════════════════════
print("▶ [3/3] 학술지 × 카테고리 히트맵...")

journal_cat = (
    df.groupby(["학술지명", "대표분야"])
    .size()
    .reset_index(name="편수")
)

# 상위 학술지 20개만 표시
top_journals = (
    df["학술지명"].value_counts()
    .head(20)
    .index.tolist()
)
jc_filtered = journal_cat[journal_cat["학술지명"].isin(top_journals)]

jc_pivot = jc_filtered.pivot(index="학술지명", columns="대표분야", values="편수").fillna(0)
# CORE_CATS 순서 정렬
col_order = [c for c in CORE_CATS if c in jc_pivot.columns] + \
            [c for c in jc_pivot.columns if c not in CORE_CATS]
jc_pivot = jc_pivot[col_order]

# 학술지별 총 논문 수로 정렬
jc_pivot = jc_pivot.loc[
    jc_pivot.sum(axis=1).sort_values(ascending=True).index
]

fig_jc = go.Figure(data=go.Heatmap(
    z=jc_pivot.values,
    x=jc_pivot.columns.tolist(),
    y=jc_pivot.index.tolist(),
    colorscale="Blues",
    text=jc_pivot.values.astype(int),
    texttemplate="%{text}",
    textfont={"size": 9},
    hovertemplate="학술지: %{y}<br>분야: %{x}<br>편수: %{z}<extra></extra>",
    colorbar=dict(title="논문 수"),
))
fig_jc.update_layout(
    title=dict(text="주요 학술지별 카테고리 분포 (상위 20개 학술지)", font=dict(size=16)),
    xaxis_title="카테고리",
    yaxis_title="학술지",
    width=1000, height=700,
    font=dict(size=10),
    margin=dict(l=280, r=60, t=70, b=100),
    xaxis=dict(tickangle=-30),
)
save_fig(fig_jc, "03_journal_category_heatmap")


# ══════════════════════════════════════════════════════
# 5. 네트워크 기본 통계 저장
# ══════════════════════════════════════════════════════
print("\n▶ 네트워크 통계 저장...")

stats_rows = []
# 카테고리 공존 네트워크
stats_rows.append({"네트워크": "카테고리공존", "노드수": G_cat.number_of_nodes(),
                   "엣지수": G_cat.number_of_edges(),
                   "평균연결도": round(sum(dict(G_cat.degree()).values()) / max(G_cat.number_of_nodes(), 1), 2),
                   "밀도": round(nx.density(G_cat), 4)})
# 공저자 네트워크
if G_author.number_of_nodes() > 0:
    components = list(nx.connected_components(G_author))
    stats_rows.append({"네트워크": "공저자", "노드수": G_author.number_of_nodes(),
                       "엣지수": G_author.number_of_edges(),
                       "평균연결도": round(sum(dict(G_author.degree()).values()) / max(G_author.number_of_nodes(), 1), 2),
                       "밀도": round(nx.density(G_author), 4)})

pd.DataFrame(stats_rows).to_csv(f"{OUT_DIR}/09_network_stats.csv",
                                 index=False, encoding="utf-8-sig")

# 상위 허브 저자
if G_author.number_of_nodes() > 0:
    degree_sorted = sorted(G_author.degree(), key=lambda x: x[1], reverse=True)
    print("\n  📌 공저 연결도 상위 10명:")
    for name, deg in degree_sorted[:10]:
        cat  = G_author.nodes[name].get("cat", "?")
        freq = G_author.nodes[name].get("freq", 0)
        print(f"    {name:12s}  연결도={deg:3d}  논문={freq:3d}편  분야={cat}")

print("\n" + "=" * 55)
print("✅ 카테고리 네트워크 분석 완료! 생성된 파일:")
print(f"   {OUT_DIR}/01_category_cooccur_network.html/png")
print(f"   {OUT_DIR}/02_author_collab_network.html/png")
print(f"   {OUT_DIR}/03_journal_category_heatmap.html/png")
print(f"   {OUT_DIR}/09_network_stats.csv")
print("=" * 55)
