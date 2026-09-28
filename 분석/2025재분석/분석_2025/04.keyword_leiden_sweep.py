"""
04.keyword_leiden_sweep.py
---------------------------
Resolution 스윕: 여러 resolution 값에서 Leiden을 반복 실행해
커뮤니티 수·modularity·안정성을 측정하고 시각화.

분석 항목:
  1. resolution vs 커뮤니티 수 / modularity
  2. 각 resolution에서 N_REPEAT회 반복 → 커뮤니티 수 분산(안정성) 측정
  3. 현재 사용 중인 resolution=1.0 기준선 표시

출력:
  results/04_keyword_network/
    resolution_sweep.png   — 2×1 패널 차트
    resolution_sweep.csv   — 수치 테이블
"""

from pathlib import Path
import statistics

import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ────────────────────────────── 경로 ──────────────────────────────────
BASE = Path(__file__).parent
NET  = BASE / "results" / "04_keyword_network" / "keyword_giant.graphml"
OUT  = BASE / "results" / "04_keyword_network"

# ────────────────────────────── 파라미터 ──────────────────────────────
RESOLUTIONS = [
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0,
]
N_REPEAT     = 10     # 각 resolution에서 반복 실행 횟수 (안정성 측정용)
CURRENT_RES  = 1.0    # 현재 사용 중인 값 (기준선 표시용)

# ────────────────────────────── 폰트 ──────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════
# 유틸: NetworkX → igraph
# ══════════════════════════════════════════════════════════════════════
def nx_to_igraph(G_nx: nx.Graph) -> ig.Graph:
    nodes   = list(G_nx.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    edges   = [(idx_map[u], idx_map[v]) for u, v in G_nx.edges()]
    weights = [G_nx[u][v].get("weight", 1) for u, v in G_nx.edges()]
    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    G_ig.es["weight"] = weights
    return G_ig


# ══════════════════════════════════════════════════════════════════════
# 스윕 실행
# ══════════════════════════════════════════════════════════════════════
def sweep(G_ig: ig.Graph) -> pd.DataFrame:
    records = []
    total = len(RESOLUTIONS) * N_REPEAT
    done  = 0

    for res in RESOLUTIONS:
        n_coms_list  = []
        mod_list     = []

        for seed in range(N_REPEAT):
            part = leidenalg.find_partition(
                G_ig,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                n_iterations=10,
                seed=seed,
            )
            n_coms_list.append(len(set(part.membership)))
            mod_list.append(part.modularity)
            done += 1

        records.append({
            "resolution":    res,
            "n_com_mean":    statistics.mean(n_coms_list),
            "n_com_min":     min(n_coms_list),
            "n_com_max":     max(n_coms_list),
            "n_com_std":     statistics.stdev(n_coms_list) if len(n_coms_list) > 1 else 0,
            "modularity_mean": statistics.mean(mod_list),
            "modularity_std":  statistics.stdev(mod_list) if len(mod_list) > 1 else 0,
        })
        print(f"  res={res:.1f}  →  커뮤니티 {min(n_coms_list)}~{max(n_coms_list)}개  "
              f"(평균 {statistics.mean(n_coms_list):.1f})  "
              f"modularity {statistics.mean(mod_list):.4f}")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════
def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor("#F8F9FA")
    for ax in axes:
        ax.set_facecolor("#F8F9FA")

    res   = df["resolution"].values
    c_mu  = df["n_com_mean"].values
    c_min = df["n_com_min"].values
    c_max = df["n_com_max"].values
    m_mu  = df["modularity_mean"].values
    m_std = df["modularity_std"].values

    # ── 패널 1: 커뮤니티 수
    ax0 = axes[0]
    ax0.fill_between(res, c_min, c_max, alpha=0.20, color="#457B9D", label="min–max 범위")
    ax0.plot(res, c_mu, "o-", color="#457B9D", linewidth=2, markersize=6, label="평균 커뮤니티 수")
    ax0.axvline(CURRENT_RES, color="#E63946", linestyle="--", linewidth=1.5,
                label=f"현재 사용 값 (res={CURRENT_RES})")
    # 현재 값에서의 커뮤니티 수 주석
    cur_row = df[df["resolution"] == CURRENT_RES]
    if not cur_row.empty:
        cy = cur_row["n_com_mean"].values[0]
        ax0.annotate(f"{cy:.0f}개", xy=(CURRENT_RES, cy),
                     xytext=(CURRENT_RES + 0.1, cy + 0.5),
                     fontsize=9, color="#E63946",
                     arrowprops=dict(arrowstyle="->", color="#E63946", lw=1))
    ax0.set_ylabel("커뮤니티 수", fontsize=11)
    ax0.set_title(f"Leiden Resolution 스윕 (반복 {N_REPEAT}회 / 거대 성분 {224}개 노드)", fontsize=12)
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── 패널 2: Modularity
    ax1 = axes[1]
    ax1.fill_between(res, m_mu - m_std, m_mu + m_std,
                     alpha=0.20, color="#2A9D8F", label="±1σ 범위")
    ax1.plot(res, m_mu, "s-", color="#2A9D8F", linewidth=2, markersize=6, label="평균 Modularity")
    ax1.axvline(CURRENT_RES, color="#E63946", linestyle="--", linewidth=1.5,
                label=f"현재 사용 값 (res={CURRENT_RES})")
    ax1.set_ylabel("Modularity", fontsize=11)
    ax1.set_xlabel("Resolution", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "resolution_sweep.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("\n[saved] resolution_sweep.png")


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    print("── 그래프 로드 중...")
    G_nx = nx.read_graphml(NET)
    G_ig = nx_to_igraph(G_nx)
    print(f"[graph] 노드: {G_ig.vcount()}  엣지: {G_ig.ecount()}")

    print(f"\n── Resolution 스윕 시작 ({len(RESOLUTIONS)}개 값 × {N_REPEAT}회 반복) ──")
    df = sweep(G_ig)

    print("\n── 결과 저장 중...")
    df.to_csv(OUT / "resolution_sweep.csv", index=False, encoding="utf-8-sig",
              float_format="%.4f")
    print("[saved] resolution_sweep.csv")

    plot(df)

    # ── 콘솔 요약 테이블
    print("\n" + "=" * 65)
    print(f"{'resolution':>12}  {'커뮤니티(평균)':>12}  {'범위':>10}  {'modularity':>12}")
    print("-" * 65)
    for _, row in df.iterrows():
        marker = " ◀ 현재" if row["resolution"] == CURRENT_RES else ""
        print(f"  {row['resolution']:>8.1f}    {row['n_com_mean']:>8.1f}      "
              f"{row['n_com_min']:.0f}~{row['n_com_max']:.0f}       {row['modularity_mean']:>8.4f}{marker}")
    print("=" * 65)
    print(f"\n✓ 완료: {OUT}")


if __name__ == "__main__":
    main()
