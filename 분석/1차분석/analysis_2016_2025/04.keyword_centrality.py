"""
04.keyword_centrality.py
-------------------------
키워드 공출현 네트워크의 중심성 지표 분석
  - 연결 중심성 (Degree Centrality): 정규화된 연결 수
  - 가중 연결 중심성 (Weighted Degree): 엣지 가중치(공출현 횟수) 합산
  - 매개 중심성 (Betweenness Centrality): 커뮤니티 간 다리 키워드 탐지
  - 페이지랭크 (PageRank): 영향력 있는 키워드 탐지
      → 인용 횟수 없는 공출현 네트워크에서는 엣지 가중치(공출현 횟수)를
        링크 강도로 사용 (가중 PageRank)

핵심 질문:
  1. 빈도 상위 키워드와 구조적 중심 키워드는 일치하는가? (Degree vs Freq)
  2. 커뮤니티 간 가교 역할을 하는 키워드는 무엇인가? (Betweenness)
  3. 영향력 기준으로 본 핵심 키워드는 빈도/연결 기준과 다른가? (PageRank)

출력:
  results/04_keyword_network/
    keyword_centrality.csv         — 키워드별 모든 중심성 지표 + 커뮤니티
    centrality_freq_vs_degree.png  — 빈도 vs 연결 중심성 산점도
    centrality_betweenness_top.png — 매개 중심성 상위 키워드 바차트
    centrality_pagerank_top.png    — PageRank 상위 키워드 바차트
    centrality_bridge_network.png  — 다리 키워드 강조 네트워크 시각화
"""

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ────────────────────────── 경로 설정 ──────────────────────────────────
BASE = Path(__file__).parent
NET  = BASE / "results" / "04_keyword_network" / "keyword_giant.graphml"
COM  = BASE / "results" / "04_keyword_network" / "keyword_communities.csv"
OUT  = BASE / "results" / "04_keyword_network"
OUT.mkdir(parents=True, exist_ok=True)

# ────────────────────────── 파라미터 ───────────────────────────────────
TOP_N_BETWEENNESS = 20   # 매개 중심성 상위 표시 개수
TOP_N_BRIDGE_NET  = 30   # 네트워크 시각화 시 강조할 상위 다리 키워드 수

# ────────────────────────── 폰트 설정 ───────────────────────────────────
_font_candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _font_candidates:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════════════
def load_data() -> tuple[nx.Graph, pd.DataFrame]:
    print("[load] 네트워크 로드 중...")
    G = nx.read_graphml(NET)
    # graphml 저장 시 weight가 문자열로 저장될 수 있으므로 float 변환
    # distance = 1/weight 추가: betweenness 계산 시 공출현 강도를 "가까움"으로 해석
    for u, v, d in G.edges(data=True):
        if "weight" in d:
            d["weight"] = float(d["weight"])
            d["distance"] = 1.0 / float(d["weight"])
    print(f"  노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}")

    print("[load] 커뮤니티 정보 로드 중...")
    df_com = pd.read_csv(COM, encoding="utf-8-sig")
    print(f"  키워드-커뮤니티 매핑: {len(df_com)}개")
    return G, df_com


# ═══════════════════════════════════════════════════════════════════════
# 2. 중심성 계산
# ═══════════════════════════════════════════════════════════════════════
def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    print("\n[centrality] 연결 중심성 계산 중...")
    deg_centrality = nx.degree_centrality(G)

    print("[centrality] 가중 연결 중심성 계산 중...")
    weighted_deg = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
                    for n in G.nodes()}

    print("[centrality] 매개 중심성 계산 중 (시간 소요)...")
    # 방식 A: distance = 1/weight 사용
    # 공출현 횟수(weight)가 클수록 두 키워드가 "가깝다"고 해석
    # NetworkX betweenness는 weight를 거리로 처리하므로 역수 변환 필요
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    print("[centrality] PageRank 계산 중...")
    # 가중 PageRank: 공출현 횟수(weight)를 링크 강도로 사용
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    df = pd.DataFrame({
        "keyword":            list(G.nodes()),
        "freq":               [G.nodes[n].get("freq", 0) for n in G.nodes()],
        "degree":             [G.degree(n) for n in G.nodes()],
        "degree_centrality":  [deg_centrality[n] for n in G.nodes()],
        "weighted_degree":    [weighted_deg[n] for n in G.nodes()],
        "betweenness":        [betweenness[n] for n in G.nodes()],
        "pagerank":           [pagerank[n] for n in G.nodes()],
    })
    df = df.sort_values("betweenness", ascending=False).reset_index(drop=True)
    print(f"[centrality] 완료. 키워드 수: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. 커뮤니티 정보 병합
# ═══════════════════════════════════════════════════════════════════════
def merge_community(df_cent: pd.DataFrame, df_com: pd.DataFrame) -> pd.DataFrame:
    df_com_slim = df_com[["keyword", "community_id", "community_label"]].copy()
    df = df_cent.merge(df_com_slim, on="keyword", how="left")
    df["community_id"]    = df["community_id"].fillna(-1).astype(int)
    df["community_label"] = df["community_label"].fillna("미배정")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. 시각화 1 — 빈도 vs 연결 중심성 산점도
# ═══════════════════════════════════════════════════════════════════════
def plot_freq_vs_degree(df: pd.DataFrame) -> None:
    """
    x축: 논문 등장 빈도(freq)
    y축: 연결 중심성(degree_centrality)
    → 빈도는 낮지만 연결 중심성 높은 '구조적 허브' 키워드 탐지
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # 커뮤니티별 색상
    cids = sorted(df["community_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(cids))
    color_map = {cid: cmap(i) for i, cid in enumerate(cids)}
    colors = df["community_id"].map(color_map)

    scatter = ax.scatter(
        df["freq"], df["degree_centrality"],
        c=colors, alpha=0.7, s=40, edgecolors="none"
    )

    # 상위 레이블 표시 (빈도 상위 OR 연결 중심성 상위)
    top_freq = set(df.nlargest(10, "freq")["keyword"])
    top_deg  = set(df.nlargest(10, "degree_centrality")["keyword"])
    label_set = top_freq | top_deg

    for _, row in df[df["keyword"].isin(label_set)].iterrows():
        ax.annotate(
            row["keyword"],
            xy=(row["freq"], row["degree_centrality"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=7.5, alpha=0.9
        )

    ax.set_xlabel("논문 등장 빈도 (Frequency)", fontsize=11)
    ax.set_ylabel("연결 중심성 (Degree Centrality)", fontsize=11)
    ax.set_title("키워드 빈도 vs 연결 중심성\n(빈도는 낮지만 허브인 키워드 탐지)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT / "centrality_freq_vs_degree.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# 5. 시각화 2 — 매개 중심성 상위 키워드 바차트
# ═══════════════════════════════════════════════════════════════════════
def plot_betweenness_top(df: pd.DataFrame, top_n: int = TOP_N_BETWEENNESS) -> None:
    """커뮤니티 색상으로 구분된 매개 중심성 상위 키워드 바차트."""
    df_top = df.head(top_n).copy()

    # 커뮤니티별 색상 (상위 N개에 등장하는 커뮤니티만)
    cids = sorted(df_top["community_id"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(cids), 1))
    color_map = {cid: cmap(i) for i, cid in enumerate(cids)}
    bar_colors = df_top["community_id"].map(color_map)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        range(top_n), df_top["betweenness"].values,
        color=bar_colors, edgecolor="white", linewidth=0.5
    )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(df_top["keyword"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("매개 중심성 (Betweenness Centrality, normalized)", fontsize=10)
    ax.set_title(f"커뮤니티 간 다리 키워드 Top {top_n}\n(매개 중심성 기준)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    # 커뮤니티 범례
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=color_map[cid],
              label=f"C{cid}: {df_top[df_top['community_id']==cid]['community_label'].iloc[0][:15]}...")
        for cid in cids if cid != -1
    ]
    if legend_items:
        ax.legend(handles=legend_items, loc="lower right", fontsize=7,
                  title="커뮤니티", title_fontsize=8)

    plt.tight_layout()
    out_path = OUT / "centrality_betweenness_top.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# 6. 시각화 3 — PageRank 상위 키워드 바차트
# ═══════════════════════════════════════════════════════════════════════
def plot_pagerank_top(df: pd.DataFrame, top_n: int = TOP_N_BETWEENNESS) -> None:
    """
    PageRank 상위 키워드 바차트.
    빈도·연결 중심성과 비교해 '영향력' 기준 핵심 키워드를 식별.
    """
    df_top = df.nlargest(top_n, "pagerank").copy()

    cids = sorted(df_top["community_id"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(cids), 1))
    color_map = {cid: cmap(i) for i, cid in enumerate(cids)}
    bar_colors = df_top["community_id"].map(color_map)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        range(top_n), df_top["pagerank"].values,
        color=bar_colors, edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(df_top["keyword"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("PageRank (가중, α=0.85)", fontsize=10)
    ax.set_title(
        f"키워드 영향력 Top {top_n} — PageRank 기준\n"
        f"(공출현 횟수를 링크 강도로 사용한 가중 PageRank)",
        fontsize=12
    )
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=color_map[cid],
              label=f"C{cid}: {df_top[df_top['community_id']==cid]['community_label'].iloc[0][:15]}...")
        for cid in cids if cid != -1
    ]
    if legend_items:
        ax.legend(handles=legend_items, loc="lower right", fontsize=7,
                  title="커뮤니티", title_fontsize=8)

    plt.tight_layout()
    out_path = OUT / "centrality_pagerank_top.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# 7. 시각화 4 — 다리 키워드 강조 네트워크
# ═══════════════════════════════════════════════════════════════════════
def plot_bridge_network(G: nx.Graph, df: pd.DataFrame,
                        top_n: int = TOP_N_BRIDGE_NET) -> None:
    """
    매개 중심성 상위 키워드를 붉은색/대형 노드로 강조한 네트워크 시각화.
    가독성을 위해 거대 성분 중 상위 연결 노드만 표시.
    """
    bridge_keywords = set(df.head(top_n)["keyword"])

    # 시각화용 서브그래프: degree 기준 상위 120개 노드 + 다리 키워드 포함
    top_degree_nodes = set(
        [n for n, _ in sorted(G.degree(), key=lambda x: -x[1])[:120]]
    )
    nodes_to_show = top_degree_nodes | bridge_keywords
    Gsub = G.subgraph(nodes_to_show).copy()

    # Spring layout (seed 고정으로 재현성 확보)
    pos = nx.spring_layout(Gsub, seed=42, k=0.6, weight="weight")

    # 커뮤니티 색상
    com_lookup = dict(zip(df["keyword"], df["community_id"]))
    cids_all = sorted(set(df["community_id"].unique()) - {-1})
    cmap = plt.cm.get_cmap("tab20", max(len(cids_all), 1))
    color_map = {cid: cmap(i) for i, cid in enumerate(cids_all)}

    node_colors = []
    node_sizes  = []
    node_edges  = []
    for n in Gsub.nodes():
        cid = com_lookup.get(n, -1)
        is_bridge = n in bridge_keywords
        node_colors.append(color_map.get(cid, (0.7, 0.7, 0.7, 1.0)))
        node_sizes.append(500 if is_bridge else 80)
        node_edges.append("crimson" if is_bridge else "none")

    edge_weights = [Gsub[u][v].get("weight", 1) for u, v in Gsub.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.2 + 1.8 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(
        Gsub, pos, ax=ax,
        width=edge_widths, alpha=0.25, edge_color="gray"
    )
    nx.draw_networkx_nodes(
        Gsub, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edgecolors=node_edges, linewidths=1.2
    )
    # 다리 키워드만 레이블 표시
    bridge_labels = {n: n for n in Gsub.nodes() if n in bridge_keywords}
    nx.draw_networkx_labels(
        Gsub, pos, labels=bridge_labels, ax=ax,
        font_size=7.5, font_color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none")
    )

    ax.set_title(
        f"키워드 공출현 네트워크 — 매개 중심성 상위 {top_n} 다리 키워드 강조\n"
        f"(붉은 테두리: 다리 키워드, 노드 색상: 커뮤니티)",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    out_path = OUT / "centrality_bridge_network.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
# 8. 텍스트 요약 출력
# ═══════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("매개 중심성 (Betweenness) 상위 20 — 커뮤니티 간 다리 키워드")
    print("=" * 60)
    cols = ["keyword", "betweenness", "freq", "degree", "community_label"]
    print(df[cols].head(20).to_string(index=False))

    print("\n" + "=" * 60)
    print("PageRank 상위 20 — 영향력 기준 핵심 키워드")
    print("=" * 60)
    cols_pr = ["keyword", "pagerank", "freq", "degree", "betweenness", "community_label"]
    print(df.nlargest(20, "pagerank")[cols_pr].to_string(index=False))

    print("\n" + "=" * 60)
    print("빈도 / 연결 중심성 / PageRank 순위 3-way 비교 (빈도 상위 20)")
    print("=" * 60)
    df2 = df.copy()
    df2["freq_rank"] = df2["freq"].rank(ascending=False).astype(int)
    df2["deg_rank"]  = df2["degree_centrality"].rank(ascending=False).astype(int)
    df2["pr_rank"]   = df2["pagerank"].rank(ascending=False).astype(int)
    df2["btw_rank"]  = df2["betweenness"].rank(ascending=False).astype(int)
    comparison = df2.nlargest(20, "freq")[
        ["keyword", "freq_rank", "deg_rank", "pr_rank", "btw_rank", "community_label"]
    ]
    print(comparison.to_string(index=False))

    # 빈도↑ but PageRank↓ → 많이 등장하지만 영향력은 낮은 키워드
    df2["pr_freq_diff"] = (df2["freq_rank"] - df2["pr_rank"]).abs()
    print("\n" + "=" * 60)
    print("빈도-PageRank 순위 괴리 상위 10 (= 빈도와 영향력이 불일치하는 키워드)")
    print("=" * 60)
    top_diff = df2.nlargest(10, "pr_freq_diff")[
        ["keyword", "freq_rank", "pr_rank", "pr_freq_diff", "community_label"]
    ]
    print(top_diff.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    G, df_com = load_data()

    df_cent = compute_centrality(G)
    df      = merge_community(df_cent, df_com)

    # CSV 저장 (소수점 4자리 반올림)
    df_save = df.copy()
    for col in ["degree_centrality", "weighted_degree", "betweenness", "pagerank"]:
        df_save[col] = df_save[col].round(4)
    csv_path = OUT / "keyword_centrality.csv"
    df_save.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {csv_path.name}  ({len(df)}행)")

    print_summary(df)

    print("\n── 시각화 생성 중...")
    plot_freq_vs_degree(df)
    plot_betweenness_top(df)
    plot_pagerank_top(df)
    plot_bridge_network(G, df)

    print(f"\n✓ 완료: {OUT}")
    print("생성 파일:")
    print("  keyword_centrality.csv         — 키워드별 중심성 지표 전체")
    print("  centrality_freq_vs_degree.png  — 빈도 vs 연결 중심성 산점도")
    print("  centrality_betweenness_top.png — 매개 중심성 상위 바차트")
    print("  centrality_pagerank_top.png    — PageRank 상위 바차트")
    print("  centrality_bridge_network.png  — 다리 키워드 강조 네트워크")


if __name__ == "__main__":
    main()
