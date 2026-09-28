"""
05b.rank_bump_redesign.py
=========================
'연구 주제군 순위 변동 차트'(슬로프/범프) 재설계본 — 순위 정의 B.

순위 정의(B, 기간 합산 방식):
  각 기간(초기 2016–18 / 중기 2019–22 / 후기 2023–25)에서 주제군별 논문을 먼저
  '합산'해 기간 점유율을 구하고, 그 점유율로 한 번만 순위(1~14, 중복 없음)를 매긴다.
  → 연도별 순위를 평균내던 기존 방식(A)의 '소표본 요동 → 중복(3이 여러 개)' 문제를 제거.

가독성 설계: x축 여백 최소화, 폰트·마커 확대, 좌/우 거터 라벨 최소간격 패킹 + 유도선.
입력 : results/05_keyword_community_trend/kw_community_trend_data.csv
출력 : results/05_keyword_community_trend/community_rank_bump_v2.png (+ .pdf)
실행 : python 05b.rank_bump_redesign.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator

for fp in [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "results/05_keyword_community_trend")
df = pd.read_csv(os.path.join(OUT, "kw_community_trend_data.csv"))
df.columns = df.columns.str.strip()

PERIODS = {"초기": (2016, 2018), "중기": (2019, 2022), "후기": (2023, 2025)}

# ── 정의 B: 기간 합산 점유율 → 유일 순위 ──────────────────────────────────
def period_rank_share(s, e):
    sub = df[df.year.between(s, e)]
    agg = sub.groupby("community_label")["annual_count"].sum()
    share = agg / agg.sum() * 100
    rank = share.rank(ascending=False, method="first").astype(int)
    return rank, share

er, es = period_rank_share(*PERIODS["초기"])
mr, ms = period_rank_share(*PERIODS["중기"])
lr, ls = period_rank_share(*PERIODS["후기"])

summary = pd.DataFrame({
    "early_rank": er, "mid_rank": mr, "late_rank": lr,
    "late_share": ls,
}).dropna().sort_values("late_rank")

cmap = plt.get_cmap("tab20")
comm = list(summary.index)
colors = {l: cmap(i / max(len(comm) - 1, 1)) for i, l in enumerate(comm)}

# 라벨 최소간격 패킹
def pack_labels(node_ys, min_gap, y_lo, y_hi):
    order = list(np.argsort(node_ys))
    y = np.array(node_ys, float)
    for _ in range(200):
        moved = False
        for i in range(1, len(order)):
            a, b = order[i-1], order[i]
            if y[b] - y[a] < min_gap:
                push = (min_gap - (y[b] - y[a])) / 2
                y[a] -= push; y[b] += push; moved = True
        y = np.clip(y, y_lo, y_hi)
        if not moved:
            break
    return y

x_pos = [0, 1, 2]
x_labels = ["초기\n2016–2018", "중기\n2019–2022", "후기\n2023–2025"]
N = len(summary)
Y_LO, Y_HI = 0.6, N + 0.4
MIN_GAP = 0.62

fig, ax = plt.subplots(figsize=(12.5, 11))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for xp in x_pos:
    ax.axvline(xp, color="#DDDDDD", lw=1.0, zorder=0)

node_left, node_right = {}, {}
for lbl, r in summary.iterrows():
    color = colors[lbl]
    ys = [r["early_rank"], r["mid_rank"], r["late_rank"]]  # 유일 정수 순위
    ax.plot(x_pos, ys, color=color, lw=2.6, alpha=0.9,
            marker="o", markersize=15, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2.6,
            zorder=3, solid_capstyle="round")
    for xi, rv in zip(x_pos, ys):
        ax.text(xi, rv, f"{int(rv)}", ha="center", va="center",
                fontsize=9, color=color, fontweight="bold", zorder=5)
    node_left[lbl]  = ys[0]
    node_right[lbl] = ys[2]

labels = list(summary.index)
lyL = pack_labels([node_left[l]  for l in labels], MIN_GAP, Y_LO, Y_HI)
lyR = pack_labels([node_right[l] for l in labels], MIN_GAP, Y_LO, Y_HI)

for l, yy in zip(labels, lyL):
    color = colors[l]; short = str(l).split("/")[0].strip()
    ax.plot([-0.045, -0.008], [yy, node_left[l]], color=color, lw=0.8, alpha=0.55, zorder=2)
    ax.text(-0.06, yy, short, ha="right", va="center",
            fontsize=11, color=color, fontweight="bold")
for l, yy in zip(labels, lyR):
    color = colors[l]; short = str(l).split("/")[0].strip()
    ax.plot([2.008, 2.045], [node_right[l], yy], color=color, lw=0.8, alpha=0.55, zorder=2)
    ax.text(2.06, yy, f"{short}  ({summary.loc[l,'late_share']:.1f}%)",
            ha="left", va="center", fontsize=11, color=color, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=14, fontweight="bold")
ax.set_xlim(-1.05, 3.05)
ax.invert_yaxis()
ax.set_ylim(N + 0.7, 0.3)
ax.set_ylabel("점유율 순위  (1 = 최상위, 기간 합산 기준)", fontsize=13)
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.tick_params(axis="y", labelsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.22)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title("연구 주제군 순위 변동 (기간 합산 점유율 기준)\n"
             "초기 2016–2018  ·  중기 2019–2022  ·  후기 2023–2025",
             fontsize=16, fontweight="bold", pad=18)

plt.tight_layout()
for ext, dpi in [("png", 200), ("pdf", None)]:
    p = os.path.join(OUT, f"community_rank_bump_v2.{ext}")
    plt.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"저장: {p}")
plt.close(fig)
