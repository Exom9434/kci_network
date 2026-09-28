"""
02.category_stats.py
────────────────────────────────────────────────────────────
카테고리별 기술 통계 표 + 연도별 꺾은선 그래프 (단일 / 융복합)

입력:
  - KCI_AI_논문_카테고리_확정.csv          (다수결 확정 1,895편)
  - KCI_AI_논문_카테고리_미확정_수동검토완료.csv (수동검토 완료 119편)

출력:
  - results/02_descriptive/category_stats_single.csv   (단일 통계표)
  - results/02_descriptive/category_stats_fusion.csv   (융복합 통계표)
  - results/02_descriptive/category_line_single.png    (단일 꺾은선)
  - results/02_descriptive/category_line_fusion.png    (융복합 꺾은선)

실행:
  python 02.category_stats.py
────────────────────────────────────────────────────────────
"""

import os
import re
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
OUT_DIR = os.path.join(BASE_DIR, "results", "02_descriptive")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 오타 정규화 맵 (수동검토 파일에서 발견된 오타) ──────────────
TYPO_MAP = {
    "기럴테크": "리걸테크",
    "사회보자장법": "사회보장법",
    "인공지능자동차법": "자율주행자동차법",
}

def normalize_category(cat: str) -> str:
    """카테고리 오타 수정 + 괄호 닫기 처리"""
    if not isinstance(cat, str):
        return cat
    cat = cat.strip()
    # 개별 오타 수정
    for wrong, right in TYPO_MAP.items():
        cat = cat.replace(wrong, right)
    # 융복합 괄호가 닫히지 않은 경우
    if cat.startswith("융복합(") and not cat.endswith(")"):
        cat = cat + ")"
    return cat

def normalize_fusion_order(cat: str) -> str:
    """융복합(A+B) → 구성 요소를 가나다 순으로 정렬해 표준화"""
    if not isinstance(cat, str) or not cat.startswith("융복합("):
        return cat
    m = re.match(r"융복합\((.+)\+(.+)\)", cat)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        return f"융복합({'+'.join(sorted([a, b]))})"
    return cat

# ── 데이터 로드 & 전처리 ──────────────────────────────────────
df_confirmed = pd.read_csv(
    os.path.join(BASE_DIR, "KCI_AI_논문_카테고리_확정.csv"), encoding="utf-8-sig"
)
df_manual = pd.read_csv(
    os.path.join(BASE_DIR, "KCI_AI_논문_카테고리_미확정_수동검토완료.csv"),
    encoding="utf-8-sig",
)

# 수동검토 파일 컬럼 맞추기
df_manual = df_manual[["논문ID", "발행연도", "최종_카테고리"]].copy()

# 확정 파일에서 필요한 컬럼만
df_confirmed = df_confirmed[["논문ID", "발행연도", "최종_카테고리"]].copy()

# 합치기 (중복 없음 — ID 기준 검증)
assert len(set(df_confirmed["논문ID"]) & set(df_manual["논문ID"])) == 0, \
    "두 파일 간 중복 논문 ID 발견!"

df = pd.concat([df_confirmed, df_manual], ignore_index=True)
print(f"총 논문 수: {len(df)}편")

# 오타 정규화
df["최종_카테고리"] = df["최종_카테고리"].apply(normalize_category)
# 융복합 순서 표준화
df["최종_카테고리"] = df["최종_카테고리"].apply(normalize_fusion_order)

# 단일 / 융복합 분리
df["is_fusion"] = df["최종_카테고리"].str.startswith("융복합")
df_single = df[~df["is_fusion"]].copy()
df_fusion = df[df["is_fusion"]].copy()

print(f"  단일 카테고리: {len(df_single)}편")
print(f"  융복합 카테고리: {len(df_fusion)}편")

YEARS = sorted(df["발행연도"].unique())

# ════════════════════════════════════════════════════════════
# 1. 통계표 — 단일 카테고리
# ════════════════════════════════════════════════════════════
# 카테고리 × 연도 피벗
pivot_single = (
    df_single.groupby(["최종_카테고리", "발행연도"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=YEARS, fill_value=0)
)
pivot_single["합계"] = pivot_single.sum(axis=1)
pivot_single = pivot_single.sort_values("합계", ascending=False)
pivot_single.index.name = "카테고리"

# 합계 행 추가
total_row_s = pivot_single.sum(axis=0).rename("전체 합계")
pivot_single = pd.concat([pivot_single, total_row_s.to_frame().T])

out_single_csv = os.path.join(OUT_DIR, "category_stats_single.csv")
pivot_single.to_csv(out_single_csv, encoding="utf-8-sig")
print(f"\n[단일 통계표] → {out_single_csv}")
print(pivot_single.to_string())

# ════════════════════════════════════════════════════════════
# 2. 통계표 — 융복합 카테고리
# ════════════════════════════════════════════════════════════
pivot_fusion = (
    df_fusion.groupby(["최종_카테고리", "발행연도"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=YEARS, fill_value=0)
)
pivot_fusion["합계"] = pivot_fusion.sum(axis=1)
pivot_fusion = pivot_fusion.sort_values("합계", ascending=False)
pivot_fusion.index.name = "카테고리"

total_row_f = pivot_fusion.sum(axis=0).rename("전체 합계")
pivot_fusion = pd.concat([pivot_fusion, total_row_f.to_frame().T])

out_fusion_csv = os.path.join(OUT_DIR, "category_stats_fusion.csv")
pivot_fusion.to_csv(out_fusion_csv, encoding="utf-8-sig")
print(f"\n[융복합 통계표] → {out_fusion_csv}")
print(pivot_fusion.head(20).to_string())

# ════════════════════════════════════════════════════════════
# 3. 꺾은선 그래프 — 연도별 단일 카테고리
# ════════════════════════════════════════════════════════════
# 합계 행 제외, 합계 컬럼 제외해서 그래프용 데이터
plot_single = pivot_single.drop(index="전체 합계").drop(columns="합계")

fig, ax = plt.subplots(figsize=(13, 7))

# 색상 팔레트
colors = plt.cm.tab20.colors
for i, (cat, row) in enumerate(plot_single.iterrows()):
    ax.plot(YEARS, row.values, marker="o", linewidth=1.8,
            markersize=5, label=cat, color=colors[i % len(colors)])

ax.set_title("연도별 단일 카테고리 논문 출판 추이 (2016–2025)", fontsize=14, pad=14)
ax.set_xlabel("발행연도", fontsize=11)
ax.set_ylabel("논문 수 (편)", fontsize=11)
ax.set_xticks(YEARS)
ax.legend(
    loc="upper left", fontsize=8.5, ncol=2,
    framealpha=0.85, bbox_to_anchor=(1.01, 1), borderaxespad=0
)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_line_single = os.path.join(OUT_DIR, "category_line_single.png")
plt.savefig(out_line_single, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[단일 꺾은선] → {out_line_single}")

# ════════════════════════════════════════════════════════════
# 4. 꺾은선 그래프 — 연도별 융복합 카테고리 (합계 ≥ 5편인 조합만)
# ════════════════════════════════════════════════════════════
plot_fusion = pivot_fusion.drop(index="전체 합계").drop(columns="합계")

# 너무 작은 조합은 '기타'로 묶기
MIN_COUNT = 5
major = pivot_fusion.loc[
    pivot_fusion.index != "전체 합계", "합계"
]
top_labels = major[major >= MIN_COUNT].index.tolist()
other_labels = major[major < MIN_COUNT].index.tolist()

plot_fusion_top = plot_fusion.loc[top_labels]
if other_labels:
    plot_fusion_top.loc["기타"] = plot_fusion.loc[other_labels].sum()

fig, ax = plt.subplots(figsize=(13, 7))

colors2 = plt.cm.tab20b.colors
for i, (cat, row) in enumerate(plot_fusion_top.iterrows()):
    # 레이블 단순화: '융복합(' 제거
    label = cat.replace("융복합(", "").rstrip(")")
    ax.plot(YEARS, row.values, marker="o", linewidth=1.8,
            markersize=5, label=label, color=colors2[i % len(colors2)])

ax.set_title(
    f"연도별 융복합 카테고리 논문 출판 추이 (2016–2025)\n(합계 {MIN_COUNT}편 미만은 '기타' 통합)",
    fontsize=13, pad=14
)
ax.set_xlabel("발행연도", fontsize=11)
ax.set_ylabel("논문 수 (편)", fontsize=11)
ax.set_xticks(YEARS)
ax.legend(
    loc="upper left", fontsize=8, ncol=2,
    framealpha=0.85, bbox_to_anchor=(1.01, 1), borderaxespad=0
)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_line_fusion = os.path.join(OUT_DIR, "category_line_fusion.png")
plt.savefig(out_line_fusion, dpi=150, bbox_inches="tight")
plt.close()
print(f"[융복합 꺾은선] → {out_line_fusion}")

print("\n✅ 완료!")
print(f"  통계표(단일): {out_single_csv}")
print(f"  통계표(융복합): {out_fusion_csv}")
print(f"  그래프(단일): {out_line_single}")
print(f"  그래프(융복합): {out_line_fusion}")
