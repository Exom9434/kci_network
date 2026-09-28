"""
02.category_top10_bar.py
────────────────────────────────────────────────────────────
단일 카테고리 상위 10 + 융복합 카테고리 상위 10
가로 막대 그래프 (공시적 현황)

출력:
  results/02_descriptive/category_top10_bar.png

실행:
  python 02.category_top10_bar.py
────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ── 한글 폰트 설정 ────────────────────────────────────────────
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

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results", "02_descriptive")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 데이터 로드 ───────────────────────────────────────────────
s = pd.read_csv(os.path.join(OUT_DIR, "category_stats_single.csv"),
                encoding="utf-8-sig", index_col=0)
f = pd.read_csv(os.path.join(OUT_DIR, "category_stats_fusion.csv"),
                encoding="utf-8-sig", index_col=0)

# 상위 10, 오름차순 (barh는 위→아래 순서라 reverse)
top10_s = (s.drop(index="전체 합계")
            .nlargest(10, "합계")["합계"]
            .sort_values(ascending=True))   # 아래→위 = 작→큰

top10_f = (f.drop(index="전체 합계")
            .nlargest(10, "합계")["합계"]
            .sort_values(ascending=True))
# 레이블 단순화: '융복합(' 제거
top10_f.index = top10_f.index.str.replace(r"융복합\(", "", regex=True).str.rstrip(")")

total_s = int(s.loc["전체 합계", "합계"])
total_f = int(f.loc["전체 합계", "합계"])

# ── 그래프 ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("KCI 법학 AI 논문 카테고리 현황 (2016–2025)", fontsize=14, fontweight="bold", y=1.01)

COLOR_SINGLE = "#3B82F6"   # 파랑
COLOR_FUSION = "#10B981"   # 초록

# — 왼쪽: 단일 카테고리 —
bars1 = ax1.barh(top10_s.index, top10_s.values, color=COLOR_SINGLE, height=0.65)
ax1.set_title(f"단일 카테고리 상위 10\n(전체 {total_s:,}편)", fontsize=11, pad=8)
ax1.set_xlabel("논문 수 (편)", fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="x", linestyle="--", alpha=0.45)
ax1.set_axisbelow(True)

# 막대 끝에 숫자 표시
for bar, val in zip(bars1, top10_s.values):
    ax1.text(val + 2, bar.get_y() + bar.get_height() / 2,
             f"{int(val)}편", va="center", ha="left", fontsize=8.5)
ax1.set_xlim(0, top10_s.max() * 1.18)

# — 오른쪽: 융복합 카테고리 —
bars2 = ax2.barh(top10_f.index, top10_f.values, color=COLOR_FUSION, height=0.65)
ax2.set_title(f"융복합 카테고리 상위 10\n(전체 {total_f:,}편)", fontsize=11, pad=8)
ax2.set_xlabel("논문 수 (편)", fontsize=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", linestyle="--", alpha=0.45)
ax2.set_axisbelow(True)

for bar, val in zip(bars2, top10_f.values):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{int(val)}편", va="center", ha="left", fontsize=8.5)
ax2.set_xlim(0, top10_f.max() * 1.2)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "category_top10_bar.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ 저장 완료 → {out_path}")
