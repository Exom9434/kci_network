"""
02.category_charts.py
=====================
2장 기술통계 - 카테고리 차트 생성

데이터:
  - KCI_AI_논문_카테고리_확정.csv     (1,895편)
  - KCI_AI_논문_카테고리_검토필요.csv  (119편)
  합계: 2,014편 / 단일 1,410 / 융복합 604

출력:
  results/02_descriptive/category_donut_bar.png      (도넛 + 단일 카테고리 가로막대)
  results/02_descriptive/category_fusion_detail.png  (상위 15 융복합 카테고리)

실행: python 02.category_charts.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
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
OUT  = BASE / "results" / "02_descriptive"
OUT.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 & 병합 ────────────────────────────────────────────────
df_main = pd.read_csv(BASE / "KCI_AI_논문_카테고리_확정.csv",    encoding="utf-8-sig")
df_rev  = pd.read_csv(BASE / " KCI_AI_논문_카테고리_검토필요.csv", encoding="utf-8-sig")

df = pd.concat([
    df_main[["논문ID", "발행연도", "최종_카테고리"]],
    df_rev[["논문ID",  "발행연도", "최종_카테고리"]],
], ignore_index=True)

df["is_fusion"] = df["최종_카테고리"].str.startswith("융복합", na=False)
n_total  = len(df)
n_single = (~df["is_fusion"]).sum()
n_fusion = df["is_fusion"].sum()

assert n_total == 2014, f"총 논문 수 불일치: {n_total}"

# ── 단일 카테고리 상위 ───────────────────────────────────────────────
single_top = (df[~df["is_fusion"]]["최종_카테고리"]
              .value_counts()
              .head(12))

# ── 색상 팔레트 ─────────────────────────────────────────────────────
BLUE   = "#1D3557"
TEAL   = "#2A9D8F"
GRAY   = "#A8DADC"
COLORS_SINGLE = [
    "#1D3557","#2B4C7E","#2A9D8F","#E07B39","#C4882B",
    "#6A4C93","#E63946","#4D6B8A","#5C7A3E","#9B4521",
    "#3A5570","#888888",
]
COLORS_FUSION = [
    "#6A4C93","#8B6BBF","#A98FD4","#C4AEE6","#DAC9F0",
    "#E8DEF7","#F0EAFC","#B39DDB","#9575CD","#7E57C2",
    "#673AB7","#5E35B1","#512DA8","#4527A0","#311B92",
]

# ════════════════════════════════════════════════════════════════════
# 그림 1-A: 도넛 차트
# ════════════════════════════════════════════════════════════════════
fig, ax_donut = plt.subplots(figsize=(8, 7), facecolor="#F8F9FA")
fig.suptitle("KCI 법학 AI 논문 카테고리 구성 (2016–2025)",
             fontsize=15, fontweight="bold", color=BLUE, y=0.98)

donut_sizes  = [n_single, n_fusion]
donut_labels = [f"단일 분야\n{n_single:,}편\n({n_single/n_total*100:.1f}%)",
                f"융복합\n{n_fusion:,}편\n({n_fusion/n_total*100:.1f}%)"]
wedges, _ = ax_donut.pie(
    donut_sizes,
    colors=[BLUE, TEAL],
    startangle=90,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2.5),
    counterclock=False,
)
for i, (wedge, label) in enumerate(zip(wedges, donut_labels)):
    ang = (wedge.theta2 + wedge.theta1) / 2
    x = 0.72 * np.cos(np.radians(ang))
    y = 0.72 * np.sin(np.radians(ang))
    ax_donut.text(x, y, label, ha="center", va="center",
                  fontsize=13, fontweight="bold", color="white")
ax_donut.text(0, 0, f"총\n{n_total:,}편",
              ha="center", va="center", fontsize=14,
              fontweight="bold", color=BLUE)
ax_donut.set_title("단일 vs. 융복합 구성", fontsize=13,
                    fontweight="bold", pad=14, color="#333333")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out1a = OUT / "category_donut.png"
plt.savefig(out1a, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"[저장] {out1a}")

# ════════════════════════════════════════════════════════════════════
# 그림 1-B: 단일 카테고리 가로막대
# ════════════════════════════════════════════════════════════════════
fig, ax_bar = plt.subplots(figsize=(10, 7), facecolor="#F8F9FA")
ax_bar.set_facecolor("#F8F9FA")
fig.suptitle("단일 분야 상위 카테고리 (2016–2025)",
             fontsize=15, fontweight="bold", color=BLUE, y=0.98)

bars = ax_bar.barh(range(len(single_top)), single_top.values,
                   color=COLORS_SINGLE[:len(single_top)],
                   edgecolor="white", linewidth=0.8, height=0.7)
ax_bar.set_yticks(range(len(single_top)))
ax_bar.set_yticklabels(single_top.index, fontsize=11)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("논문 수", fontsize=11)
ax_bar.spines[["top","right","left"]].set_visible(False)
ax_bar.tick_params(left=False)
ax_bar.xaxis.grid(True, linestyle="--", alpha=0.5)
ax_bar.set_axisbelow(True)
for bar, val in zip(bars, single_top.values):
    ax_bar.text(val + 2, bar.get_y() + bar.get_height()/2,
                f"{val}편", va="center", fontsize=9.5, color="#444444")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out1b = OUT / "category_bar.png"
plt.savefig(out1b, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"[저장] {out1b}")

# ════════════════════════════════════════════════════════════════════
# 그림 2: 상위 15개 융복합 카테고리 상세
# ════════════════════════════════════════════════════════════════════
fusion_top15 = (df[df["is_fusion"]]["최종_카테고리"]
                .value_counts()
                .head(15))

fig, ax = plt.subplots(figsize=(14, 7), facecolor="#F8F9FA")
ax.set_facecolor("#F8F9FA")

# 카테고리명에서 괄호 제거해서 더 짧게 표시
labels_short = [l.replace("융복합(", "").replace(")", "") for l in fusion_top15.index]

bars = ax.barh(range(len(fusion_top15)), fusion_top15.values,
               color=COLORS_FUSION[:len(fusion_top15)],
               edgecolor="white", linewidth=0.8, height=0.72)
ax.set_yticks(range(len(fusion_top15)))
ax.set_yticklabels(labels_short, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("논문 수", fontsize=11)
ax.set_title("상위 15개 융복합 카테고리 (전체 융복합 604편)",
             fontsize=14, fontweight="bold", color=BLUE, pad=14)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(left=False)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for bar, val in zip(bars, fusion_top15.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val}편", va="center", fontsize=9.5, color="#444444")

plt.tight_layout()
out2 = OUT / "category_fusion_detail.png"
plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"[저장] {out2}")
print(f"\n완료: 단일 {n_single}편 / 융복합 {n_fusion}편 / 총 {n_total}편")
