"""
인용 네트워크 분석 — 연도별 내부 인용 비율 추이 시각화
결과: results/03_citation/06_citation_ratio_trend.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from pathlib import Path

import pandas as pd
import matplotlib.font_manager as fm


BASE = Path(__file__).parent
OUT_DIR = BASE / "results" / "03_citation"
STATS_PATH = OUT_DIR / "citation_stats.txt"


def load_citation_ratio_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "[연도별 인용 유형 구성]":
            start = idx + 2  # skip section title and header line
            break
    if start is None:
        raise ValueError(f"연도별 인용 유형 구성 표를 찾지 못했습니다: {path}")

    rows = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "year": int(parts[0]),
                "external": int(parts[1]),
                "internal": int(parts[2]),
                "total": int(parts[3]),
                "ratio_pct": float(parts[4]),
            }
        )
    if not rows:
        raise ValueError(f"연도별 인용 유형 구성 표에서 수치를 읽지 못했습니다: {path}")
    return pd.DataFrame(rows)


data = load_citation_ratio_table(STATS_PATH)
years = data["year"]
ratio = data["ratio_pct"]
internal = data["internal"]
external = data["external"]

# ── 스타일 ────────────────────────────────────────────────────────────────
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
font_family = "DejaVu Sans"
for font_path in FONT_CANDIDATES:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        font_family = fm.FontProperties(fname=font_path).get_name()
        break

plt.rcParams.update({
    "font.family": [font_family, "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "font.size": 12,
})

fig, ax1 = plt.subplots(figsize=(10, 5.5))

# 막대: 내부 / 외부 인용 건수 (stacked)
bar_width = 0.6
bars_ext = ax1.bar(years, external, width=bar_width,
                   color="#B8CCE4", label="외부 인용", zorder=2)
bars_int = ax1.bar(years, internal, width=bar_width,
                   bottom=external, color="#4472C4", label="내부 인용", zorder=2)

ax1.set_xlabel("연도", fontsize=12)
ax1.set_ylabel("인용 건수", fontsize=12)
ax1.set_xticks(years)
ax1.tick_params(axis="y", labelcolor="#333333")
ax1.set_ylim(0, 4200)
ax1.yaxis.set_major_locator(plt.MultipleLocator(500))
ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

# 꺾은선: 내부 인용 비율 (%)
ax2 = ax1.twinx()
ax2.plot(years, ratio, color="#E36C09", linewidth=2.2,
         marker="o", markersize=7, label="내부 인용 비율 (%)", zorder=3)
for x, y in zip(years, ratio):
    ax2.annotate(f"{y:.1f}%", xy=(x, y),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=10, color="#E36C09", fontweight="bold")

ax2.set_ylabel("내부 인용 비율 (%)", fontsize=12, color="#E36C09")
ax2.tick_params(axis="y", labelcolor="#E36C09")
ax2.set_ylim(0, 60)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

# 범례 통합
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=11, framealpha=0.85)

plt.title("연도별 인용 유형 구성 및 내부 인용 비율 추이", fontsize=14, pad=14)
fig.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
out_path = OUT_DIR / "06_citation_ratio_trend.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장 완료: {out_path}")
