"""
04.keyword_viz.py
------------------
키워드 공동 사용 네트워크 시각화

입력:
  results/04_keyword_network/keyword_giant.graphml
  results/04_keyword_network/keyword_communities.csv

출력:
  results/04_keyword_network/
    keyword_network.png          — 정적 PNG (고해상도)
    keyword_network.html         — Plotly 인터랙티브 HTML
"""

from pathlib import Path
import math
import collections

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import plotly.graph_objects as go

# ────────────────────────────── 경로 설정 ─────────────────────────────
BASE = Path(__file__).parent
NET  = BASE / "results" / "04_keyword_network" / "keyword_giant.graphml"
COM  = BASE / "results" / "04_keyword_network" / "keyword_communities.csv"
OUT  = BASE / "results" / "04_keyword_network"

# ────────────────────────────── 파라미터 ──────────────────────────────
LABEL_TOP_N     = 30    # PNG: 상위 N개 키워드 레이블 표시
PLOTLY_LABEL_N  = 50    # Plotly: 상위 N개 레이블 표시

# 서로 구분이 잘 되는 고대비 커뮤니티 색상 팔레트 (16색)
# 통계/시각화용 Palette(Tab10, Set1, Dark2 등)를 기반으로 가독성을 고려해 조합함
PALETTE = [
    # --- 1~8번: 주요 구분을 위한 강렬한 색상 ---
    "#E63946", # 0: 빨강 (앵커)
    "#1D3557", # 1: 짙은 남색 (앵커)
    "#2A9D8F", # 2: 청록색
    "#F4A261", # 3: 살구/주황
    "#6A4C93", # 4: 보라
    "#FFD166", # 5: 노랑/골드 (밝음)
    "#F72585", # 6: 핫핑크
    "#4CC9F0", # 7: 하늘색

    # --- 9~16번: 세부 군집 구분을 위한 중간톤/파스텔톤 ---
    "#8AB17D", # 8: 올리브 그린
    "#C77DFF", # 9: 연보라
    "#FB8500", # 10: 진한 주황
    "#606C38", # 11: 국방색/카키
    "#90E0EF", # 12: 연한 남색
    "#FF9F1C", # 13: 귤색
    "#A7C957", # 14: 라임 그린
    "#ADB5BD"  # 15: 회색 (기타/작은 군집용)
]
# ────────────────────────────── 폰트 설정 ─────────────────────────────
_font_candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_font_path = None
for _fp in _font_candidates:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        _font_path = _fp
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════
def load_data() -> tuple[nx.Graph, pd.DataFrame]:
    G = nx.read_graphml(NET)
    com_df = pd.read_csv(COM, encoding="utf-8-sig")
    print(f"[load] 노드: {G.number_of_nodes()}  엣지: {G.number_of_edges()}")
    print(f"[load] 커뮤니티: {com_df['community_id'].nunique()}개")
    return G, com_df


# ══════════════════════════════════════════════════════════════════════
# 2. 레이아웃 & 속성 계산
# ══════════════════════════════════════════════════════════════════════
def compute_layout(G: nx.Graph) -> dict[str, tuple[float, float]]:
    """spring_layout with weight. 재현성을 위해 seed 고정."""
    return nx.spring_layout(G, weight="weight", seed=42, k=1.5 / math.sqrt(G.number_of_nodes()))


def build_node_attrs(
    G: nx.Graph, com_df: pd.DataFrame
) -> dict[str, dict]:
    """노드별 freq, community_id, community_label, color 계산."""
    kw_to_com  = dict(zip(com_df["keyword"],         com_df["community_id"]))
    kw_to_label= dict(zip(com_df["keyword"],         com_df["community_label"]))

    attrs = {}
    for node in G.nodes():
        freq  = G.nodes[node].get("freq", 1)
        cid   = kw_to_com.get(node, -1)
        color = PALETTE[cid % len(PALETTE)] if cid >= 0 else "#CCCCCC"
        attrs[node] = {
            "freq":    freq,
            "cid":     cid,
            "label":   kw_to_label.get(node, ""),
            "color":   color,
        }
    return attrs


# ══════════════════════════════════════════════════════════════════════
# 3. PNG 시각화
# ══════════════════════════════════════════════════════════════════════
def draw_png(
    G: nx.Graph,
    pos: dict,
    attrs: dict,
    com_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 18))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    # ── 엣지
    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    for (u, v), w in zip(G.edges(), edge_weights):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        alpha = 0.08 + 0.35 * (w / max_w)
        lw    = 0.3 + 1.5 * (w / max_w)
        ax.plot([x0, x1], [y0, y1], color="#AAAAAA", alpha=alpha, linewidth=lw, zorder=1)

    # ── 노드
    freqs  = [attrs[n]["freq"] for n in G.nodes()]
    max_f  = max(freqs)
    node_sizes  = [80 + 1200 * (attrs[n]["freq"] / max_f) ** 0.6 for n in G.nodes()]
    node_colors = [attrs[n]["color"] for n in G.nodes()]
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]

    sc = ax.scatter(xs, ys, s=node_sizes, c=node_colors, alpha=0.85, zorder=2,
                    linewidths=0.5, edgecolors="white")

    # ── 레이블 (상위 LABEL_TOP_N)
    top_nodes = sorted(G.nodes(), key=lambda n: attrs[n]["freq"], reverse=True)[:LABEL_TOP_N]
    for node in top_nodes:
        x, y = pos[node]
        size = attrs[node]["freq"]
        fontsize = 7 + min(5, size / 20)
        ax.text(x, y, node, fontsize=fontsize, ha="center", va="center",
                fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, linewidth=0))

    # ── 범례 (커뮤니티)
    cid_to_label = com_df.drop_duplicates("community_id").set_index("community_id")["community_label"]
    cids = sorted(com_df["community_id"].unique())
    patches = []
    for cid in cids:
        lbl = cid_to_label.get(cid, f"커뮤니티 {cid}")
        # 레이블 간소화 (30자 초과 시 줄임)
        short = lbl if len(lbl) <= 30 else lbl[:28] + "…"
        patches.append(mpatches.Patch(color=PALETTE[cid % len(PALETTE)], label=f"C{cid}: {short}"))
    ax.legend(handles=patches, loc="lower left", fontsize=7.5,
              framealpha=0.85, edgecolor="#CCCCCC",
              title="키워드 커뮤니티", title_fontsize=8)

    ax.set_title("한국 법학 AI 논문 키워드 공동 사용 네트워크\n(노드 크기 ∝ 출현 논문 수 / 색상 = Leiden 커뮤니티)",
                 fontsize=14, pad=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "keyword_network.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[saved] keyword_network.png")


# ══════════════════════════════════════════════════════════════════════
# 4. Plotly 인터랙티브 HTML
# ══════════════════════════════════════════════════════════════════════
def draw_plotly(
    G: nx.Graph,
    pos: dict,
    attrs: dict,
    com_df: pd.DataFrame,
) -> None:
    # ── 엣지 트레이스
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#BBBBBB"),
        hoverinfo="none",
        showlegend=False,
    )

    # ── 노드별 커뮤니티 트레이스 (범례 분리)
    cid_to_label = com_df.drop_duplicates("community_id").set_index("community_id")["community_label"]
    cids = sorted(com_df["community_id"].unique())

    top_set = set(
        sorted(G.nodes(), key=lambda n: attrs[n]["freq"], reverse=True)[:PLOTLY_LABEL_N]
    )
    freqs  = [attrs[n]["freq"] for n in G.nodes()]
    max_f  = max(freqs)

    traces = [edge_trace]
    for cid in cids:
        nodes_in = [n for n in G.nodes() if attrs[n]["cid"] == cid]
        if not nodes_in:
            continue
        color = PALETTE[cid % len(PALETTE)]
        lbl   = cid_to_label.get(cid, f"커뮤니티 {cid}")

        nx_arr, ny_arr, sizes, texts, hovers = [], [], [], [], []
        for node in nodes_in:
            x, y = pos[node]
            nx_arr.append(x)
            ny_arr.append(y)
            freq = attrs[node]["freq"]
            sizes.append(8 + 30 * (freq / max_f) ** 0.5)
            texts.append(node if node in top_set else "")
            hovers.append(f"<b>{node}</b><br>출현 논문 수: {freq}<br>커뮤니티: C{cid} {lbl}")

        traces.append(go.Scatter(
            x=nx_arr, y=ny_arr,
            mode="markers+text",
            name=f"C{cid}: {lbl[:25]}",
            marker=dict(size=sizes, color=color, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            text=texts,
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hovers,
            hoverinfo="text",
        ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text="한국 법학 AI 논문 키워드 공동 사용 네트워크",
                font=dict(size=18),
            ),
            showlegend=True,
            legend=dict(font=dict(size=10), itemsizing="constant"),
            hovermode="closest",
            paper_bgcolor="#F8F9FA",
            plot_bgcolor="#F8F9FA",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200, height=900,
            margin=dict(l=20, r=20, t=60, b=20),
        )
    )

    fig.write_html(OUT / "keyword_network.html")
    print("[saved] keyword_network.html")


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    G, com_df = load_data()

    print("\n── 레이아웃 계산 중 (spring_layout)...")
    pos = compute_layout(G)

    print("── 노드 속성 계산 중...")
    attrs = build_node_attrs(G, com_df)

    print("── PNG 저장 중...")
    draw_png(G, pos, attrs, com_df)

    print("── Plotly HTML 저장 중...")
    draw_plotly(G, pos, attrs, com_df)

    print(f"\n✓ 완료: {OUT}")


if __name__ == "__main__":
    main()
