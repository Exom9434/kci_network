"""
05.keyword_community_trend.py
==============================
5장 전용: 키워드 커뮤니티(04) 기반 누적 변천사 분석

4장에서 도출한 키워드 커뮤니티(community_paper_map.csv)에
발행연도를 붙여 시계열 분석.

→ 4장(키워드 커뮤니티 정의) → 5장(그 커뮤니티의 시간적 성장) 논리 완결

출력:
  results/05_keyword_community_trend/
    kw_community_growth.png       커뮤니티별 누적 성장 곡선
    kw_community_stacked.png      누적 스택 영역 차트 (비중 변화)
    kw_community_emergence.png    등장 시점 타임라인
    kw_community_heatmap.png      커뮤니티 × 연도 히트맵 (절대/비율)
    kw_community_trend_data.csv   수치 테이블
"""

from pathlib import Path
import importlib.util, os, collections
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

# ────────────────────────────── 경로 ──────────────────────────────
BASE    = Path(__file__).parent
COM_MAP = BASE / "results" / "04_keyword_network" / "community_paper_map.csv"
DATA    = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
OUT     = BASE / "results" / "05_keyword_community_trend"
OUT.mkdir(parents=True, exist_ok=True)

YEARS      = list(range(2016, 2026))
EMERGE_MIN = 5   # 커뮤니티 "등장"으로 인정할 최소 누적 논문 수

# ────────────────────────────── 폰트 ──────────────────────────────
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

PALETTE = [
    "#E63946", "#1D3557", "#2A9D8F", "#F4A261", "#6A4C93",
    "#FFD166", "#F72585", "#4CC9F0", "#8AB17D", "#C77DFF",
    "#FB8500", "#606C38", "#90E0EF", "#FF9F1C", "#A7C957",
    "#ADB5BD",
]


# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 조인
# ══════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    community_paper_map.csv (논문ID + 키워드 커뮤니티)
    + 00.KCI 데이터 (논문ID + 발행연도)
    → 논문별 [논문ID, 발행연도, community_id, community_label]
    """
    com_map = pd.read_csv(COM_MAP, encoding="utf-8-sig")
    com_map["논문ID"] = com_map["논문ID"].astype(str)

    # 발행연도 추출
    df_all = pd.read_csv(DATA, encoding="utf-8-sig",
                         usecols=["source_id", "pub_year"])
    df_all = df_all.drop_duplicates(subset="source_id")
    df_all["source_id"] = df_all["source_id"].astype(str)

    merged = com_map.merge(
        df_all, left_on="논문ID", right_on="source_id", how="left"
    )
    merged = merged.rename(columns={"pub_year": "year"})
    merged = merged[merged["year"].between(2016, 2025)].copy()
    merged["year"] = merged["year"].astype(int)

    print(f"[load] 조인 결과: {len(merged)}편  "
          f"(키워드 커뮤니티 {merged['primary_community'].nunique()}개)")
    print(f"[load] 연도 범위: {merged['year'].min()}~{merged['year'].max()}")
    return merged


# ══════════════════════════════════════════════════════════════════
# 2. 집계
# ══════════════════════════════════════════════════════════════════
def aggregate(df: pd.DataFrame):
    """
    커뮤니티별 연도별 논문 수 집계 →
    label_order, annual, cumul 반환
    """
    # 커뮤니티 총 크기 내림차순으로 레이블 정렬
    label_order = (df.groupby("community_label")["논문ID"]
                   .count().sort_values(ascending=False).index.tolist())

    annual = collections.defaultdict(lambda: collections.defaultdict(int))
    for _, row in df.iterrows():
        annual[row["community_label"]][row["year"]] += 1

    cumul = {}
    for lbl in label_order:
        total = 0
        cumul[lbl] = []
        for yr in YEARS:
            total += annual[lbl].get(yr, 0)
            cumul[lbl].append(total)

    return label_order, annual, cumul


# ══════════════════════════════════════════════════════════════════
# 3. 차트 ①: 커뮤니티별 누적 성장 곡선
# ══════════════════════════════════════════════════════════════════
def plot_growth(label_order, cumul, lbl_color) -> None:
    # 3열 레이아웃 — 14개 커뮤니티 → 5행×3열 (빈칸 1개)
    N_COL = 3
    N_ROW = -(-len(label_order) // N_COL)   # ceiling division
    fig, axes = plt.subplots(N_ROW, N_COL,
                             figsize=(15, N_ROW * 3.6),
                             sharex=True)
    axes = np.array(axes).flatten()

    # x축 표시는 짝수 연도만 (공간 절약)
    tick_years  = [yr for yr in YEARS if yr % 2 == 0]
    tick_labels = [str(yr) for yr in tick_years]

    for i, lbl in enumerate(label_order):
        ax = axes[i]
        vals = cumul[lbl]

        # 영역 채우기 (누적)
        ax.fill_between(YEARS, vals, alpha=0.18, color=lbl_color[lbl])
        # 누적 실선
        ax.plot(YEARS, vals, marker="o", markersize=3.5,
                color=lbl_color[lbl], linewidth=2.2, zorder=3)

        # 패널 우상단 — 최종 누적 편수 배지
        ax.text(0.97, 0.93, f"{vals[-1]}편",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9.5, fontweight="bold",
                color=lbl_color[lbl])

        ax.set_title(lbl, fontsize=10, fontweight="bold", pad=5)
        ax.set_xticks(tick_years)
        ax.set_xticklabels(tick_labels, fontsize=8.5)
        ax.set_xlim(YEARS[0] - 0.3, YEARS[-1] + 0.3)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
        ax.tick_params(axis="y", labelsize=8)

    for j in range(len(label_order), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("키워드 커뮤니티별 누적 논문 수 (2016–2025)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    fig.savefig(OUT / "kw_community_growth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[저장] kw_community_growth.png")


# ══════════════════════════════════════════════════════════════════
# 4. 차트 ②: 누적 스택 영역 차트
# ══════════════════════════════════════════════════════════════════
def plot_stacked(label_order, cumul, lbl_color) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    cumul_matrix = np.array([cumul[lbl] for lbl in label_order])
    ax.stackplot(YEARS, cumul_matrix,
                 labels=label_order,
                 colors=[lbl_color[lbl] for lbl in label_order],
                 alpha=0.85)
    ax.set_title("키워드 커뮤니티별 누적 논문 수 변화 — 스택 영역", fontsize=12)
    ax.set_xlabel("발행연도")
    ax.set_ylabel("누적 논문 수")
    ax.set_xticks(YEARS)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kw_community_stacked.png", dpi=160)
    plt.close(fig)
    print("[저장] kw_community_stacked.png")


# ══════════════════════════════════════════════════════════════════
# 5. 차트 ③: 등장 시점 타임라인
# ══════════════════════════════════════════════════════════════════
def plot_emergence(label_order, annual, cumul, lbl_color) -> None:
    emerge_year = {}
    for lbl in label_order:
        total = 0
        for yr in YEARS:
            total += annual[lbl].get(yr, 0)
            if total >= EMERGE_MIN:
                emerge_year[lbl] = yr
                break

    fig, ax = plt.subplots(figsize=(13, max(5, len(label_order) * 0.55 + 2)))
    for i, lbl in enumerate(reversed(label_order)):
        yr = emerge_year.get(lbl)
        if yr is None:
            continue
        ax.barh(i, 2025 - yr + 1, left=yr, height=0.6,
                color=lbl_color[lbl], alpha=0.85)
        ax.plot(yr, i, "o", color=lbl_color[lbl],
                markersize=9, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(2025.3, i, f"{cumul[lbl][-1]}편",
                va="center", fontsize=8.5)

    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(list(reversed(label_order)), fontsize=8.5)
    ax.set_xlim(2015.5, 2026.8)
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=45)
    ax.set_title(
        f"키워드 커뮤니티 등장 시점 타임라인\n"
        f"(● 누적 {EMERGE_MIN}편 도달 시점  /  막대 끝 숫자: 2025년까지 누적 편수)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kw_community_emergence.png", dpi=160)
    plt.close(fig)
    print("[저장] kw_community_emergence.png")

    # 콘솔 요약
    print(f"\n  커뮤니티 등장 시점 (누적 {EMERGE_MIN}편 기준)")
    print(f"  {'커뮤니티':35s}  {'등장':>6s}  {'최종':>6s}")
    print("  " + "-" * 52)
    for lbl in label_order:
        yr  = emerge_year.get(lbl, "?")
        cnt = cumul[lbl][-1]
        print(f"  {lbl:35s}  {str(yr):>6s}  {cnt:>6d}")


# ══════════════════════════════════════════════════════════════════
# 6. 차트 ④: 커뮤니티 × 연도 히트맵
# ══════════════════════════════════════════════════════════════════
def plot_heatmap(label_order, annual, cumul) -> None:
    # 연간 신규 편수 행렬
    mat = np.array([
        [annual[lbl].get(yr, 0) for yr in YEARS]
        for lbl in label_order
    ], dtype=float)

    # 비율 행렬 (전체 연간 발행 대비)
    col_sum = mat.sum(axis=0)
    col_sum[col_sum == 0] = 1
    mat_ratio = mat / col_sum * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, max(6, len(label_order) * 0.5 + 2)))

    for ax, data, title, cmap, fmt in [
        (axes[0], mat,       "연간 신규 논문 수 (편)", "Blues",   True),
        (axes[1], mat_ratio, "연간 점유율 (%)",        "Oranges", False),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(YEARS)))
        ax.set_xticklabels(YEARS, rotation=45, fontsize=9)
        ax.set_yticks(range(len(label_order)))
        ax.set_yticklabels(label_order, fontsize=8.5)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
        if fmt:
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    v = int(data[r, c])
                    if v > 0:
                        ax.text(c, r, str(v), ha="center", va="center",
                                fontsize=7,
                                color="white" if v > data.max() * 0.65 else "black")
        else:
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    v = data[r, c]
                    if v >= 1:
                        ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                                fontsize=6.5,
                                color="white" if v > data.max() * 0.65 else "black")

    fig.suptitle("키워드 커뮤니티 × 연도 히트맵", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "kw_community_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[저장] kw_community_heatmap.png")


# ══════════════════════════════════════════════════════════════════
# 7. 수치 테이블 저장
# ══════════════════════════════════════════════════════════════════
def save_table(label_order, annual, cumul) -> None:
    rows = []
    for lbl in label_order:
        for i, yr in enumerate(YEARS):
            rows.append({
                "community_label": lbl,
                "year":            yr,
                "annual_count":    annual[lbl].get(yr, 0),
                "cumul_count":     cumul[lbl][i],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kw_community_trend_data.csv",
              index=False, encoding="utf-8-sig")
    print("[저장] kw_community_trend_data.csv")


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    print("── 데이터 로드 중...")
    df = load_data()

    print("\n── 집계 중...")
    label_order, annual, cumul = aggregate(df)
    lbl_color = {lbl: PALETTE[i % len(PALETTE)]
                 for i, lbl in enumerate(label_order)}

    print(f"\n  커뮤니티 수: {len(label_order)}개")
    print(f"  분석 대상 논문: {len(df)}편\n")

    print("── 차트 생성 중...")
    plot_growth(label_order, cumul, lbl_color)
    plot_stacked(label_order, cumul, lbl_color)
    plot_emergence(label_order, annual, cumul, lbl_color)
    plot_heatmap(label_order, annual, cumul)
    save_table(label_order, annual, cumul)

    print(f"\n✓ 완료: {OUT}")


if __name__ == "__main__":
    main()
