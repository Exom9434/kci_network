"""
02.category_line_charts.py
────────────────────────────────────────────────────────────
연도별 카테고리 꺾은선 그래프 (개선판)
 - 상위 N개: 색깔 선 + 레이블
 - 나머지: 연한 회색 배경선 (맥락만 제공)

입력:
  results/02_descriptive/category_stats_single.csv
  results/02_descriptive/category_stats_fusion.csv
  (02.category_stats.py 실행 후 생성됨)

출력:
  results/02_descriptive/category_line_single_v2.png
  results/02_descriptive/category_line_fusion_v2.png

실행:
  python 02.category_line_charts.py
────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ── 한글 폰트 ─────────────────────────────────────────────────
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        fe = fm.FontEntry(fname=fp, name="KoreanFont")
        fm.fontManager.ttflist.insert(0, fe)
        rcParams["font.family"] = "KoreanFont"
        print(f"폰트 로드: {fp}")
        break
else:
    print("⚠️  한글 폰트를 찾지 못했습니다. 로컬(Mac)에서 실행하세요.")

rcParams["axes.unicode_minus"] = False

# ── 경로 ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results", "02_descriptive")

# ── 공통 설정 ─────────────────────────────────────────────────
TOP_N       = 6          # 강조할 상위 카테고리 수
GRAY        = "#CCCCCC"  # 배경선 색
GRAY_ALPHA  = 0.6        # 배경선 투명도
GRAY_LW     = 1.0        # 배경선 굵기
TOP_COLORS  = [
    "#2563EB", "#DC2626", "#16A34A",
    "#D97706", "#7C3AED", "#0891B2",
]
TOP_LW      = 2.4        # 강조선 굵기


def draw_focus_line_chart(ax, pivot, top_n, title, ylabel="논문 수 (편)"):
    """
    pivot: 카테고리(행) × 연도(열) DataFrame, '합계' 컬럼 포함
    """
    years = [c for c in pivot.columns if c != "합계"]
    pivot_data = pivot[years]

    # 합계 기준 순위
    ranking = pivot["합계"].sort_values(ascending=False)
    top_cats  = ranking.index[:top_n].tolist()
    rest_cats = ranking.index[top_n:].tolist()

    # ── 배경선 (하위 카테고리) ───────────────────────────────
    for cat in rest_cats:
        ax.plot(years, pivot_data.loc[cat].values,
                color=GRAY, alpha=GRAY_ALPHA,
                linewidth=GRAY_LW, zorder=1)

    # ── 강조선 (상위 카테고리) ───────────────────────────────
    for i, cat in enumerate(top_cats):
        vals = pivot_data.loc[cat].values
        color = TOP_COLORS[i % len(TOP_COLORS)]
        ax.plot(years, vals,
                color=color, linewidth=TOP_LW,
                marker="o", markersize=4.5,
                zorder=3, label=cat)

        # 마지막 연도 끝에 레이블 직접 표기
        ax.text(years[-1] + 0.15, vals[-1],
                cat, color=color,
                fontsize=8.5, va="center", fontweight="bold")

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("발행연도", fontsize=10)
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.3, years[-1] + 2.5)   # 레이블 공간 확보
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # 범례: 회색 배경선 설명 추가
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GRAY, linewidth=GRAY_LW))
    labels.append(f"기타 ({len(rest_cats)}개 카테고리)")
    ax.legend(handles, labels, fontsize=8.5,
              loc="upper left", framealpha=0.85)


# ════════════════════════════════════════════════════════════
# 1. 단일 카테고리 꺾은선
# ════════════════════════════════════════════════════════════
s = pd.read_csv(os.path.join(OUT_DIR, "category_stats_single.csv"),
                encoding="utf-8-sig", index_col=0)
s = s.drop(index="전체 합계")
# 연도 컬럼을 int로
s.columns = [int(c) if str(c).isdigit() else c for c in s.columns]

fig, ax = plt.subplots(figsize=(11, 6))
draw_focus_line_chart(
    ax, s, TOP_N,
    title=f"연도별 단일 카테고리 논문 출판 추이 (2016–2025)\n상위 {TOP_N}개 강조, 나머지 회색"
)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "category_line_single_v2.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ 단일 꺾은선 → {out1}")

# ════════════════════════════════════════════════════════════
# 2. 융복합 카테고리 꺾은선
# ════════════════════════════════════════════════════════════
f = pd.read_csv(os.path.join(OUT_DIR, "category_stats_fusion.csv"),
                encoding="utf-8-sig", index_col=0)
f = f.drop(index="전체 합계")
f.columns = [int(c) if str(c).isdigit() else c for c in f.columns]

# 레이블 단순화: '융복합(' 제거
f.index = f.index.str.replace(r"융복합\(", "", regex=True).str.rstrip(")")

fig, ax = plt.subplots(figsize=(11, 6))
draw_focus_line_chart(
    ax, f, TOP_N,
    title=f"연도별 융복합 카테고리 논문 출판 추이 (2016–2025)\n상위 {TOP_N}개 강조, 나머지 회색"
)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "category_line_fusion_v2.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ 융복합 꺾은선 → {out2}")
