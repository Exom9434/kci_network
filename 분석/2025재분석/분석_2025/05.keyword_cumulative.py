"""
05.keyword_cumulative.py
=========================
키워드 레이블 기반 누적 분석

[Part 1] 인용 네트워크 누적 분석 (키워드 레이블 버전)
  03/03-2 커뮤니티에 키워드 레이블을 적용해
  기존 03.network_cumulative.py 와 동일한 4종 차트를 재생성.
  → 출력: results/03_network/   (파일명 _kw 접미사)
           results/03-2_network/

[Part 2] 키워드 자체 시계열 분석
  논문별 키워드 + 발행연도 기반으로
  주요 키워드의 연도별 등장 추이를 직접 시각화.
  → 출력: results/05_keyword_trend/

사용법:
    python 05.keyword_cumulative.py
"""

import os, warnings, importlib.util, collections
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 한글 폰트 ──────────────────────────────────────────────────────
BASE = Path(__file__).parent
_font_candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_font_path = next((p for p in _font_candidates if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ── keyword_build 전처리 임포트 ────────────────────────────────────
_spec = importlib.util.spec_from_file_location("kb", BASE / "04.keyword_build.py")
_kb   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kb)
parse_keywords = _kb.parse_keywords

YEARS      = list(range(2016, 2026))
SNAP_YEARS = [2017, 2019, 2021, 2023, 2025]
EMERGE_MIN = 5

PALETTE = (list(plt.cm.Set2.colors) + list(plt.cm.Set1.colors) +
           list(plt.cm.tab20.colors))


# ══════════════════════════════════════════════════════════════════════
# Part 1: 인용 네트워크 누적 분석 (키워드 레이블)
# ══════════════════════════════════════════════════════════════════════

def run_citation_cumulative(tag: str, graphml_path: Path,
                             node_csv: Path, summary_csv: Path,
                             out_dir: Path) -> None:
    print(f'\n{"="*60}')
    print(f'  [{tag}] 인용 네트워크 누적 분석 (키워드 레이블)')
    print(f'{"="*60}')

    G          = nx.read_graphml(str(graphml_path))
    df_nodes   = pd.read_csv(node_csv,    encoding="utf-8-sig")
    df_summary = pd.read_csv(summary_csv, encoding="utf-8-sig")

    # ── 키워드 레이블 우선, 없으면 auto_label 폴백 ────────────────
    def resolve_label(row):
        kl = str(row.get("keyword_label", "")).strip()
        if kl and kl not in ("", "nan"):
            return kl
        al = str(row.get("auto_label", "")).strip()
        return al if al and al != "nan" else "기타"

    id2label = {int(r["community_id"]): resolve_label(r)
                for _, r in df_summary.iterrows()}

    df_nodes["community_id"] = (pd.to_numeric(df_nodes["community_id"],
                                               errors="coerce")
                                  .fillna(-1).astype(int))
    df_nodes["label"] = df_nodes["community_id"].map(id2label).fillna("기타")

    if "node_type" in df_nodes.columns:
        df_plot = df_nodes[df_nodes["node_type"] == "ai"].copy()
    else:
        df_plot = df_nodes.copy()

    df_plot = df_plot[df_plot["year"].between(2016, 2025)]
    df_plot["year"] = df_plot["year"].astype(int)
    df_plot = df_plot[df_plot["label"] != "기타"]

    label_order = (df_plot.groupby("label")["node_id"]
                   .count().sort_values(ascending=False).index.tolist())
    lbl_color = {lbl: PALETTE[i % len(PALETTE)]
                 for i, lbl in enumerate(label_order)}

    # 연도별 누적
    annual = collections.defaultdict(lambda: collections.defaultdict(int))
    for _, r in df_plot.iterrows():
        annual[r["label"]][r["year"]] += 1

    cumul = {}
    for lbl in label_order:
        total = 0
        cumul[lbl] = []
        for yr in YEARS:
            total += annual[lbl].get(yr, 0)
            cumul[lbl].append(total)

    # ① 누적 성장 곡선
    n_col = 2
    n_row = (len(label_order) + 1) // 2
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(14, max(6, n_row * 3)), sharex=True)
    axes = np.array(axes).flatten()
    for i, lbl in enumerate(label_order):
        ax = axes[i]
        ax.fill_between(YEARS, cumul[lbl], alpha=0.3, color=lbl_color[lbl])
        ax.plot(YEARS, cumul[lbl], marker="o", markersize=4,
                color=lbl_color[lbl], linewidth=2)
        ax.set_title(lbl, fontsize=9, pad=4)
        ax.set_xticks(YEARS)
        ax.set_xticklabels(YEARS, rotation=45, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for j in range(len(label_order), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"[{tag}] 커뮤니티별 누적 논문 수 (키워드 레이블, 2016–2025)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_growth_kw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [저장] cumulative_growth_kw.png")

    # ② 스택 영역 차트
    fig, ax = plt.subplots(figsize=(13, 6))
    cumul_matrix = np.array([cumul[lbl] for lbl in label_order])
    ax.stackplot(YEARS, cumul_matrix,
                 labels=label_order,
                 colors=[lbl_color[lbl] for lbl in label_order],
                 alpha=0.85)
    ax.set_title(f"[{tag}] 커뮤니티별 누적 논문 수 변화 — 키워드 레이블", fontsize=12)
    ax.set_xlabel("발행연도")
    ax.set_ylabel("누적 논문 수")
    ax.set_xticks(YEARS)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_stacked_kw.png", dpi=150)
    plt.close(fig)
    print("  [저장] cumulative_stacked_kw.png")

    # ③ 등장 시점 타임라인
    emerge_year = {}
    for lbl in label_order:
        total = 0
        for yr in YEARS:
            total += annual[lbl].get(yr, 0)
            if total >= EMERGE_MIN:
                emerge_year[lbl] = yr
                break

    fig, ax = plt.subplots(figsize=(13, max(5, len(label_order) * 0.5 + 2)))
    for i, lbl in enumerate(reversed(label_order)):
        yr = emerge_year.get(lbl)
        if yr is None:
            continue
        ax.barh(i, 2025 - yr + 1, left=yr, height=0.6,
                color=lbl_color[lbl], alpha=0.8)
        ax.plot(yr, i, "o", color=lbl_color[lbl],
                markersize=8, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(2025.3, i, f"{cumul[lbl][-1]}편", va="center", fontsize=8)
    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(list(reversed(label_order)), fontsize=8)
    ax.set_xlim(2015.5, 2026.5)
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=45)
    ax.set_title(f"[{tag}] 커뮤니티 등장 시점 타임라인 (키워드 레이블)\n"
                 f"(● 누적 {EMERGE_MIN}편 도달 시점)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_emergence_kw.png", dpi=150)
    plt.close(fig)
    print("  [저장] cumulative_emergence_kw.png")

    # 콘솔 요약
    print(f"\n  커뮤니티 등장 시점 (누적 {EMERGE_MIN}편 기준)")
    print(f"  {'커뮤니티':30s}  {'등장':>6s}  {'최종편수':>6s}")
    print("  " + "-" * 48)
    for lbl in label_order:
        yr  = emerge_year.get(lbl, "?")
        cnt = cumul[lbl][-1]
        print(f"  {lbl:30s}  {str(yr):>6s}  {cnt:>6d}")

    # ④ 주요 연도 스냅샷
    ai_nodes_all = [n for n in G.nodes()
                    if G.nodes[n].get("node_type", "ai") == "ai"
                    or "node_type" not in G.nodes[n]]
    G_ai = G.subgraph(ai_nodes_all)
    pos  = nx.spring_layout(G_ai, k=2.0, seed=42, weight="weight", iterations=80)

    node2label = {}
    for n in G.nodes():
        raw = G.nodes[n].get("community_id", -1)
        try:
            cid = int(float(raw))
        except (ValueError, TypeError):
            cid = -1
        node2label[n] = id2label.get(cid, "기타")

    fig, axes = plt.subplots(1, len(SNAP_YEARS),
                             figsize=(5 * len(SNAP_YEARS), 5))
    if len(SNAP_YEARS) == 1:
        axes = [axes]
    for ax, snap_yr in zip(axes, SNAP_YEARS):
        visible = [n for n in ai_nodes_all
                   if int(G.nodes[n].get("year", 0)) <= snap_yr]
        if not visible:
            ax.set_title(f"{snap_yr}년\n(데이터 없음)")
            ax.axis("off")
            continue
        G_snap   = G.subgraph(visible)
        pos_snap = {n: pos[n] for n in visible if n in pos}
        nx.draw_networkx_edges(G_snap, pos_snap, ax=ax,
                               alpha=0.12, width=0.4, edge_color="#888888")
        for lbl in label_order:
            nodes_lbl = [n for n in visible
                         if node2label.get(n) == lbl and n in pos_snap]
            if not nodes_lbl:
                continue
            nx.draw_networkx_nodes(G_snap, pos_snap, ax=ax,
                                   nodelist=nodes_lbl, node_size=20,
                                   node_color=lbl_color[lbl],
                                   alpha=0.85, linewidths=0)
        ax.set_title(f"{snap_yr}년 이전\n{len(visible)}편", fontsize=10)
        ax.axis("off")

    patches = [mpatches.Patch(color=lbl_color[lbl], label=lbl)
               for lbl in label_order]
    fig.legend(handles=patches, loc="lower center",
               fontsize=6, ncol=min(5, len(label_order)),
               framealpha=0.8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"[{tag}] 주요 연도 누적 스냅샷 (키워드 레이블)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_snapshot_kw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [저장] cumulative_snapshot_kw.png")


# ══════════════════════════════════════════════════════════════════════
# Part 2: 키워드 자체 시계열 분석
# ══════════════════════════════════════════════════════════════════════

def run_keyword_trend() -> None:
    print(f'\n{"="*60}')
    print(f'  [Part 2] 키워드 시계열 분석')
    print(f'{"="*60}')

    OUT_KW = BASE / "results" / "05_keyword_trend"
    OUT_KW.mkdir(parents=True, exist_ok=True)

    # 논문 데이터 로드
    df_all = pd.read_csv(BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv",
                         encoding="utf-8-sig")

    # AI 논문 ID 수집
    import unicodedata
    def find(keyword):
        for f in os.listdir(BASE):
            if keyword in unicodedata.normalize("NFC", f):
                return BASE / f
        raise FileNotFoundError(keyword)

    ai_ids = (set(pd.read_csv(find("카테고리_확정"), encoding="utf-8-sig")
                  ["논문ID"].astype(str)) |
              set(pd.read_csv(find("검토필요"), encoding="utf-8-sig")
                  ["논문ID"].astype(str)))

    df = (df_all[df_all["source_id"].astype(str).isin(ai_ids)]
          .drop_duplicates(subset="source_id")[["source_id", "pub_year", "keywords"]]
          .copy())
    df = df[df["pub_year"].between(2016, 2025)]
    df["pub_year"] = df["pub_year"].astype(int)
    print(f"[load] AI 논문 {len(df)}편 (2016–2025, 키워드 있음: {df['keywords'].notna().sum()})")

    # 연도별 키워드 빈도 집계
    year_kw: dict[int, collections.Counter] = {yr: collections.Counter() for yr in YEARS}
    for _, row in df.iterrows():
        if pd.isna(row["keywords"]):
            continue
        kws = parse_keywords(str(row["keywords"]))
        year_kw[row["pub_year"]].update(kws)

    # 전체 빈도 기준 상위 키워드 선정
    total_freq: collections.Counter = collections.Counter()
    for cnt in year_kw.values():
        total_freq.update(cnt)

    TOP_N_LINE  = 15   # 연도별 추이 라인 차트 키워드 수
    TOP_N_HEAT  = 30   # 히트맵 키워드 수
    TOP_N_EMERG = 20   # 등장 시점 타임라인 키워드 수
    EMERGE_KW_MIN = 3  # 키워드 "등장"으로 인정할 최소 연간 등장 수

    top_line  = [kw for kw, _ in total_freq.most_common(TOP_N_LINE)]
    top_heat  = [kw for kw, _ in total_freq.most_common(TOP_N_HEAT)]
    top_emerg = [kw for kw, _ in total_freq.most_common(TOP_N_EMERG)]

    # ── ① 상위 키워드 연도별 추이 (라인 차트) ────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    line_palette = PALETTE[:TOP_N_LINE]
    for i, kw in enumerate(top_line):
        ys = [year_kw[yr].get(kw, 0) for yr in YEARS]
        ax.plot(YEARS, ys, marker="o", markersize=4,
                linewidth=1.8, color=line_palette[i], label=kw)
    ax.set_title("주요 키워드 연도별 등장 논문 수 추이 (상위 15개)", fontsize=12)
    ax.set_xlabel("발행연도")
    ax.set_ylabel("등장 논문 수")
    ax.set_xticks(YEARS)
    ax.legend(fontsize=8, ncol=3, loc="upper left", framealpha=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_KW / "keyword_trend_line.png", dpi=150)
    plt.close(fig)
    print("  [저장] keyword_trend_line.png")

    # ── ② 히트맵 (키워드 × 연도) ─────────────────────────────────
    heat_matrix = np.array([
        [year_kw[yr].get(kw, 0) for yr in YEARS]
        for kw in top_heat
    ], dtype=float)
    # 행 정규화 (각 키워드의 연도별 비중)
    row_max = heat_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    heat_norm = heat_matrix / row_max

    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, TOP_N_HEAT * 0.4 + 2)))
    for ax, mat, title, cmap in [
        (axes[0], heat_matrix, "절대 빈도 (등장 논문 수)", "Blues"),
        (axes[1], heat_norm,   "상대 비중 (키워드별 최댓값 기준 정규화)", "Oranges"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(YEARS)))
        ax.set_xticklabels(YEARS, rotation=45, fontsize=9)
        ax.set_yticks(range(len(top_heat)))
        ax.set_yticklabels(top_heat, fontsize=9)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # 셀 숫자 표시 (절대 빈도만)
        if cmap == "Blues":
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = int(mat[r, c])
                    if v > 0:
                        ax.text(c, r, str(v), ha="center", va="center",
                                fontsize=6.5,
                                color="white" if v > mat.max() * 0.6 else "black")
    fig.suptitle("주요 키워드 × 연도 히트맵 (상위 30개)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_KW / "keyword_trend_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [저장] keyword_trend_heatmap.png")

    # ── ③ 키워드 등장 시점 타임라인 ──────────────────────────────
    emerge_kw: dict[str, int] = {}
    for kw in top_emerg:
        for yr in YEARS:
            if year_kw[yr].get(kw, 0) >= EMERGE_KW_MIN:
                emerge_kw[kw] = yr
                break

    fig, ax = plt.subplots(figsize=(13, max(5, len(top_emerg) * 0.55 + 2)))
    kw_palette = PALETTE[:len(top_emerg)]
    for i, kw in enumerate(reversed(top_emerg)):
        yr = emerge_kw.get(kw)
        if yr is None:
            continue
        total_cnt = total_freq[kw]
        color = kw_palette[top_emerg.index(kw) % len(kw_palette)]
        ax.barh(i, 2025 - yr + 1, left=yr, height=0.6, color=color, alpha=0.8)
        ax.plot(yr, i, "o", color=color,
                markersize=8, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(2025.3, i, f"{total_cnt}편", va="center", fontsize=8)
    ax.set_yticks(range(len(top_emerg)))
    ax.set_yticklabels(list(reversed(top_emerg)), fontsize=9)
    ax.set_xlim(2015.5, 2026.5)
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=45)
    ax.set_title(f"주요 키워드 등장 시점 타임라인\n"
                 f"(● 연간 {EMERGE_KW_MIN}편 이상 첫 등장, 막대 끝: 전체 누적 편수)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_KW / "keyword_trend_emergence.png", dpi=150)
    plt.close(fig)
    print("  [저장] keyword_trend_emergence.png")

    # ── ④ 연도별 top10 키워드 변화 (3열×4행 패널, A4 세로 최적화) ──
    NCOLS, NROWS = 3, 4                        # 3열 4행 = 12칸 (10년 + 빈칸 2개)
    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(14, 20),  # A4 세로 비율에 맞춤
                             sharey=False)
    axes = axes.flatten()
    for i, yr in enumerate(YEARS):
        ax = axes[i]
        top10 = year_kw[yr].most_common(10)
        if not top10:
            ax.set_title(str(yr))
            ax.axis("off")
            continue
        kws_yr = [k for k, _ in top10]
        cnts   = [c for _, c in top10]
        colors_yr = [PALETTE[top_line.index(k) % len(PALETTE)]
                     if k in top_line else "#AAAAAA"
                     for k in kws_yr]
        bars = ax.barh(range(len(kws_yr)), cnts, color=colors_yr,
                       alpha=0.85, height=0.65)
        # 막대 끝 수치 표시
        for bar, cnt in zip(bars, cnts):
            ax.text(cnt + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", ha="left", fontsize=8)
        ax.set_yticks(range(len(kws_yr)))
        ax.set_yticklabels(kws_yr, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"{yr}년", fontsize=12, fontweight="bold", pad=6)
        ax.set_xlim(0, max(cnts) * 1.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
    # 빈 패널 숨기기
    for j in range(len(YEARS), NROWS * NCOLS):
        axes[j].set_visible(False)
    fig.suptitle("연도별 상위 10 키워드 (등장 논문 수)", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    fig.savefig(OUT_KW / "keyword_top10_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [저장] keyword_top10_by_year.png")

    # 연도별 키워드 빈도 CSV 저장
    rows = []
    for yr in YEARS:
        for kw, cnt in year_kw[yr].most_common():
            rows.append({"year": yr, "keyword": kw, "count": cnt})
    pd.DataFrame(rows).to_csv(OUT_KW / "keyword_trend_data.csv",
                               index=False, encoding="utf-8-sig")
    print("  [저장] keyword_trend_data.csv")
    print(f"\n  출력 위치: {OUT_KW}")


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    out_03  = BASE / "results" / "03_network"
    out_032 = BASE / "results" / "03-2_network"

    # Part 1 (인용 네트워크 누적) — 논문 미사용(03/03-2 커뮤니티 라벨 제외)으로 건너뜀
    # Part 2 — 키워드 시계열 (필요한 부분만 실행)
    run_keyword_trend()

    print(f'\n{"="*60}')
    print("  모든 누적 분석 완료!")
    print(f'{"="*60}')


if __name__ == "__main__":
    main()
