"""
02.annual_trend.py
==================
2장 기술통계 - 연도별 논문 수 추이 차트

출력:
  results/10_annual_stats/A_trend.png

실행: python 02.annual_trend.py
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
df = pd.read_csv(BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv", encoding="utf-8-sig")
annual = (df.drop_duplicates("source_id")["pub_year"]
            .value_counts()
            .sort_index())

years  = annual.index.tolist()
counts = annual.values.tolist()
total  = sum(counts)

# 전년 대비 증감률
growth = [None] + [
    (counts[i] - counts[i-1]) / counts[i-1] * 100
    for i in range(1, len(counts))
]

# ── 그리기 ──────────────────────────────────────────────────────────
BLUE  = "#1D3557"
TEAL  = "#2A9D8F"
LIGHT = "#A8DADC"

fig, ax1 = plt.subplots(figsize=(12, 6), facecolor="#F8F9FA")
ax1.set_facecolor("#F8F9FA")

# 막대 (논문 수)
bars = ax1.bar(years, counts, color=TEAL, alpha=0.75,
               edgecolor="white", linewidth=0.8, width=0.6, zorder=2)

# 추세선 (점선)
ax1.plot(years, counts, color=BLUE, linewidth=2.0,
         linestyle="--", marker="o", markersize=6,
         markerfacecolor="white", markeredgecolor=BLUE,
         markeredgewidth=2, zorder=3)

# 막대 위 수치
for year, cnt in zip(years, counts):
    ax1.text(year, cnt + 4, f"{cnt}", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=BLUE)

# 주요 이벤트 주석
events = {
    2018: ("GDPR 시행", -0.05),
    2022: ("ChatGPT 등장", -0.05),
    2023: ("EU AI Act\n발의·통과", -0.05),
}
for yr, (label, _) in events.items():
    if yr in years:
        idx = years.index(yr)
        ax1.axvline(yr, color="#888888", linewidth=1, linestyle=":", alpha=0.6, zorder=1)
        ax1.text(yr, counts[idx] + 22, label, ha="center", va="bottom",
                 fontsize=8.5, color="#555555",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#CCCCCC", alpha=0.85))

# 축 설정
ax1.set_xlabel("발행연도", fontsize=12)
ax1.set_ylabel("논문 수 (편)", fontsize=12, color=BLUE)
ax1.tick_params(axis="y", labelsize=10)
ax1.tick_params(axis="x", labelsize=11)
ax1.set_xticks(years)
ax1.set_xlim(years[0] - 0.6, years[-1] + 0.6)
ax1.set_ylim(0, max(counts) * 1.22)
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

fig.suptitle(f"KCI 법학 AI 논문 연도별 현황 (총 {total:,}편, 2016–2025)",
             fontsize=14, fontweight="bold", color=BLUE, y=1.01)

plt.tight_layout()
out = OUT / "A_trend.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"[저장] {out}")
