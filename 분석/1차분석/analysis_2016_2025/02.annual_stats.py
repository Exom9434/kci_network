"""
02.annual_stats.py
==================
2장 기술통계 - 연도별 통계 차트 (단일/융복합 비교)

출력:
  results/10_annual_stats/AB_comparison.png   단일 vs 융복합 연도별 비교 막대
  results/10_annual_stats/A_heatmap.png       단일 카테고리×연도 히트맵
  results/10_annual_stats/B_heatmap.png       융복합 카테고리×연도 히트맵
  results/10_annual_stats/A_stacked_bar.png   단일 카테고리 누적 막대
  results/10_annual_stats/B_stacked_bar.png   융복합 카테고리 누적 막대

실행: python 02.annual_stats.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

# ── 한글 폰트 ────────────────────────────────────────────────────────
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

BASE = Path(__file__).parent
OUT  = BASE / "results" / "10_annual_stats"
OUT.mkdir(parents=True, exist_ok=True)

# ── 데이터 ──────────────────────────────────────────────────────────
df_main = pd.read_csv(BASE / "KCI_AI_논문_카테고리_확정.csv",    encoding="utf-8-sig")
df_rev  = pd.read_csv(BASE / " KCI_AI_논문_카테고리_검토필요.csv", encoding="utf-8-sig")
df = pd.concat([
    df_main[["논문ID", "발행연도", "최종_카테고리"]],
    df_rev[["논문ID",  "발행연도", "최종_카테고리"]],
], ignore_index=True)

df["is_fusion"] = df["최종_카테고리"].str.startswith("융복합", na=False)
A = df[~df["is_fusion"]].copy()
B = df[df["is_fusion"]].copy()
years = sorted(df["발행연도"].unique())

BLUE  = "#1D3557"
TEAL  = "#2A9D8F"
RED   = "#E63946"
BG    = "#F8F9FA"

# ════════════════════════════════════════════════════════════════════
# 1. AB_comparison: 단일 vs 융복합 연도별 비교 막대
# ════════════════════════════════════════════════════════════════════
a_cnt = A.groupby("발행연도").size().reindex(years, fill_value=0)
b_cnt = B.groupby("발행연도").size().reindex(years, fill_value=0)

x = np.arange(len(years))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
ax.set_facecolor(BG)

bars_a = ax.bar(x - w/2, a_cnt.values, w, label=f"단일 분야 (총 {a_cnt.sum():,}편)",
                color=BLUE, alpha=0.82, edgecolor="white", linewidth=0.7)
bars_b = ax.bar(x + w/2, b_cnt.values, w, label=f"융복합 (총 {b_cnt.sum():,}편)",
                color=TEAL, alpha=0.82, edgecolor="white", linewidth=0.7)

for bar, val in zip(bars_a, a_cnt.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=8.5, color=BLUE)
for bar, val in zip(bars_b, b_cnt.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=8.5, color="#1a6b61")

ax.set_xticks(x)
ax.set_xticklabels(years, fontsize=11)
ax.set_xlabel("발행연도", fontsize=12)
ax.set_ylabel("논문 수 (편)", fontsize=12)
ax.set_ylim(0, max(a_cnt.max(), b_cnt.max()) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.88)
fig.suptitle("단일 분야 vs. 융복합 논문 수 비교 (2016–2025)",
             fontsize=14, fontweight="bold", color=BLUE)
plt.tight_layout()
p = OUT / "AB_comparison.png"
plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[저장] {p}")

# ════════════════════════════════════════════════════════════════════
# 공통 함수: 카테고리×연도 히트맵
# ════════════════════════════════════════════════════════════════════
def plot_heatmap(df_sub, top_n, title, filename, cmap="Blues"):
    top_cats = df_sub["최종_카테고리"].value_counts().head(top_n).index.tolist()
    pivot = (df_sub[df_sub["최종_카테고리"].isin(top_cats)]
             .groupby(["최종_카테고리", "발행연도"])
             .size()
             .unstack(fill_value=0)
             .reindex(columns=years, fill_value=0))
    pivot = pivot.loc[top_cats]  # 빈도 순 유지

    # 융복합 레이블 축약
    idx_labels = [l.replace("융복합(", "").replace(")", "") for l in pivot.index]

    fig, ax = plt.subplots(figsize=(13, max(5, top_n * 0.52)), facecolor=BG)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, shrink=0.7, label="논문 수")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=11)
    ax.set_yticks(range(len(idx_labels)))
    ax.set_yticklabels(idx_labels, fontsize=10)

    for i in range(len(pivot)):
        for j in range(len(years)):
            val = pivot.values[i, j]
            if val > 0:
                color = "white" if val > pivot.values.max() * 0.6 else "#333333"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8.5, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold", color=BLUE, pad=12)
    ax.set_xlabel("발행연도", fontsize=11)
    plt.tight_layout()
    p = OUT / filename
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[저장] {p}")

plot_heatmap(A, top_n=12, title="단일 분야 카테고리×연도 논문 수 히트맵",
             filename="A_heatmap.png", cmap="Blues")
plot_heatmap(B, top_n=12, title="융복합 카테고리×연도 논문 수 히트맵",
             filename="B_heatmap.png", cmap="Purples")

# ════════════════════════════════════════════════════════════════════
# 공통 함수: 카테고리 누적 막대
# ════════════════════════════════════════════════════════════════════
PALETTE = [
    "#1D3557","#2A9D8F","#E07B39","#C4882B","#6A4C93",
    "#E63946","#4D6B8A","#5C7A3E","#9B4521","#3A5570",
    "#2B4C7E","#888888","#C4AEE6","#A8DADC",
]

def plot_stacked(df_sub, top_n, title, filename):
    top_cats = df_sub["최종_카테고리"].value_counts().head(top_n).index.tolist()
    others_mask = ~df_sub["최종_카테고리"].isin(top_cats)
    df_plot = df_sub.copy()
    df_plot.loc[others_mask, "최종_카테고리"] = "기타"

    pivot = (df_plot.groupby(["발행연도", "최종_카테고리"])
             .size().unstack(fill_value=0)
             .reindex(index=years, fill_value=0))

    # 상위 카테고리 + 기타 순서
    col_order = top_cats + (["기타"] if "기타" in pivot.columns else [])
    pivot = pivot.reindex(columns=col_order, fill_value=0)
    # 융복합 레이블 축약
    pivot.columns = [c.replace("융복합(", "").replace(")", "") for c in pivot.columns]

    colors = PALETTE[:len(pivot.columns)-1] + ["#CCCCCC"]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG)
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=colors, edgecolor="white", linewidth=0.5, width=0.7)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=0, fontsize=11)
    ax.set_xlabel("발행연도", fontsize=12)
    ax.set_ylabel("논문 수 (편)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.88,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_title(title, fontsize=14, fontweight="bold", color=BLUE, pad=12)
    plt.tight_layout()
    p = OUT / filename
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[저장] {p}")

plot_stacked(A, top_n=10, title="단일 분야 카테고리별 연도 누적 막대",
             filename="A_stacked_bar.png")
plot_stacked(B, top_n=10, title="융복합 카테고리별 연도 누적 막대",
             filename="B_stacked_bar.png")

print("\n완료!")
