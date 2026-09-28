"""
AI 법학 논문 카테고리별 연도별 기초 통계 시각화
=====================================================
입력: 05.논문_카테고리_최종분류.csv
출력 (results/08_category/ 폴더):
  01_annual_stacked_bar.html/png    연도별 카테고리 스택 막대그래프
  02_category_heatmap.html/png      연도 × 주요카테고리 히트맵
  03_category_trend.html/png        주요 카테고리 연도별 추이 (라인)
  04_fusion_map.html/png            융복합 카테고리 연결망 (분야 간 공존 빈도)
  08_category_annual_table.csv      연도×카테고리 집계표 (원시 수치)
"""

import os
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════
INPUT_CSV = "05.논문_카테고리_최종분류.csv"
OUT_DIR   = "results/08_category"
os.makedirs(OUT_DIR, exist_ok=True)

# 분석 연도 범위 (2010~2014는 논문 수가 매우 적어 제외 가능)
YEAR_START = 2016
YEAR_END   = 2025

# 12개 핵심 분야 (표시 순서)
CORE_CATS = [
    "데이터법", "인공지능법", "경제법", "지식재산권법",
    "형사법", "공법", "민사법", "금융법",
    "노동법", "사회보장법", "조세법", "의료법", "기초법",
]

# Plotly 색상 팔레트 (CORE_CATS + 기타/융복합)
COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel


def save_fig(fig, name: str):
    """HTML + PNG 저장"""
    fig.write_html(f"{OUT_DIR}/{name}.html")
    try:
        fig.write_image(f"{OUT_DIR}/{name}.png", scale=2)
    except Exception as e:
        print(f"  ⚠ PNG 저장 실패 ({name}): {e}")
    print(f"  ✅ {OUT_DIR}/{name}.html / .png")


# ══════════════════════════════════════════════════════
# 1. 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════
print("▶ 데이터 로드 중...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
df = df[df["발행연도"].between(YEAR_START, YEAR_END)].copy()
df["발행연도"] = df["발행연도"].astype(int)
total = len(df)
print(f"  분석 대상: {total}편 ({YEAR_START}~{YEAR_END})")


# ── 카테고리 단순화 ────────────────────────────────────
def extract_primary_cats(cat_str: str) -> list[str]:
    """
    '융복합(데이터법|인공지능법)' → ['데이터법', '인공지능법']
    '인공지능법'                  → ['인공지능법']
    """
    cat_str = str(cat_str).strip()
    if cat_str.startswith("융복합"):
        inner = re.search(r"\((.+)\)", cat_str)
        if inner:
            return [c.strip() for c in inner.group(1).split("|") if c.strip()]
    return [cat_str]


def get_simplified(cat_str: str) -> str:
    """주요 표시용 단일 레이블: 융복합이면 첫 번째 분야 + '(융복합)' 태그"""
    cats = extract_primary_cats(cat_str)
    if len(cats) > 1:
        return "융복합"
    return cats[0] if cats else "기타"


df["주요분야"]    = df["최종카테고리"].apply(get_simplified)
df["구성분야목록"] = df["최종카테고리"].apply(extract_primary_cats)

print(f"\n  단독 논문: {(df['주요분야'] != '융복합').sum()}편")
print(f"  융복합 논문: {(df['주요분야'] == '융복합').sum()}편")


# ══════════════════════════════════════════════════════
# 2. 연도 × 카테고리 집계 (단독 기준 + 융복합 별도)
# ══════════════════════════════════════════════════════
# (A) 단독 논문만 카테고리별 집계
df_single = df[df["주요분야"] != "융복합"].copy()
annual_single = (
    df_single.groupby(["발행연도", "주요분야"])
    .size()
    .reset_index(name="편수")
)

# (B) 융복합 논문은 구성 분야 각각에 0.5씩 분배 (또는 별도 행)
fusion_rows = []
for _, row in df[df["주요분야"] == "융복합"].iterrows():
    cats = row["구성분야목록"]
    share = 1.0 / len(cats)
    for c in cats:
        fusion_rows.append({
            "발행연도": row["발행연도"],
            "주요분야": c,
            "편수": share,
        })
df_fusion_dist = pd.DataFrame(fusion_rows)

# (C) 전체 통합 (단독 + 융복합 분산)
df_combined = pd.concat([annual_single, df_fusion_dist], ignore_index=True)
pivot_all = (
    df_combined.groupby(["발행연도", "주요분야"])["편수"]
    .sum()
    .reset_index()
)

# 연도×카테고리 피벗
years = sorted(df["발행연도"].unique())
cats_in_data = sorted(df_combined["주요분야"].unique())
cats_ordered = [c for c in CORE_CATS if c in cats_in_data] + \
               [c for c in cats_in_data if c not in CORE_CATS]

pivot_wide = pivot_all.pivot(index="주요분야", columns="발행연도", values="편수").fillna(0)
pivot_wide = pivot_wide.reindex(cats_ordered).fillna(0)

# CSV 저장
pivot_wide.round(1).to_csv(f"{OUT_DIR}/08_category_annual_table.csv", encoding="utf-8-sig")
print(f"\n  ✅ {OUT_DIR}/08_category_annual_table.csv")


# ══════════════════════════════════════════════════════
# 3. 시각화 ① — 연도별 스택 막대그래프
# ══════════════════════════════════════════════════════
print("\n▶ [1/4] 연도별 카테고리 스택 막대그래프...")

# 단독 vs 융복합 구분 표시 (단순 버전)
annual_count_raw = (
    df.groupby(["발행연도", "주요분야"])
    .size()
    .reset_index(name="편수")
)
pivot_raw = annual_count_raw.pivot(
    index="발행연도", columns="주요분야", values="편수"
).fillna(0)

disp_cats = [c for c in CORE_CATS if c in pivot_raw.columns] + \
            ["융복합"] if "융복합" in pivot_raw.columns else \
            [c for c in CORE_CATS if c in pivot_raw.columns]

if "융복합" in pivot_raw.columns and "융복합" not in disp_cats:
    disp_cats.append("융복합")

fig_bar = go.Figure()
color_map = {cat: COLORS[i % len(COLORS)] for i, cat in enumerate(cats_ordered)}

for cat in disp_cats:
    if cat not in pivot_raw.columns:
        continue
    fig_bar.add_trace(go.Bar(
        name=cat,
        x=pivot_raw.index.tolist(),
        y=pivot_raw[cat].tolist(),
        marker_color=color_map.get(cat, "#aaa"),
        hovertemplate=f"{cat}: %{{y}}편<extra></extra>",
    ))

fig_bar.update_layout(
    barmode="stack",
    title=dict(text="연도별 AI 법학 논문 카테고리 분포", font=dict(size=18)),
    xaxis=dict(title="발행연도", tickmode="linear"),
    yaxis=dict(title="논문 수 (편)"),
    legend=dict(title="카테고리", orientation="v"),
    width=1100, height=600,
    font=dict(size=12),
)
save_fig(fig_bar, "01_annual_stacked_bar")


# ══════════════════════════════════════════════════════
# 4. 시각화 ② — 연도 × 카테고리 히트맵 (비율 기준)
# ══════════════════════════════════════════════════════
print("▶ [2/4] 연도 × 카테고리 히트맵...")

# 연도별 전체 편수 (분모)
year_totals = df.groupby("발행연도").size().to_dict()

# pivot_wide를 비율(%)로 변환
pivot_pct = pivot_wide.copy()
for yr in pivot_pct.columns:
    total_yr = year_totals.get(yr, 1)
    pivot_pct[yr] = (pivot_pct[yr] / total_yr * 100).round(1)

# 상위 카테고리만 표시 (평균 편수 기준 상위 15개)
top_cats = (
    pivot_wide.mean(axis=1)
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)
pivot_heat = pivot_pct.loc[[c for c in top_cats if c in pivot_pct.index]]

fig_heat = go.Figure(data=go.Heatmap(
    z=pivot_heat.values,
    x=[str(y) for y in pivot_heat.columns],
    y=pivot_heat.index.tolist(),
    colorscale="Blues",
    text=pivot_heat.values.round(1),
    texttemplate="%{text}%",
    textfont={"size": 9},
    hovertemplate="연도: %{x}<br>카테고리: %{y}<br>비율: %{z}%<extra></extra>",
    colorbar=dict(title="비율(%)"),
))
fig_heat.update_layout(
    title=dict(text="연도별 카테고리 비율 히트맵 (융복합 포함 분산)", font=dict(size=17)),
    xaxis_title="발행연도",
    yaxis_title="카테고리",
    width=1000, height=600,
    font=dict(size=11),
    margin=dict(l=200, r=60, t=70, b=60),
)
save_fig(fig_heat, "02_category_heatmap")


# ══════════════════════════════════════════════════════
# 5. 시각화 ③ — 주요 카테고리 연도별 추이 (라인)
# ══════════════════════════════════════════════════════
print("▶ [3/4] 주요 카테고리 추이 라인차트...")

top8 = (
    pivot_wide.mean(axis=1)
    .sort_values(ascending=False)
    .head(8)
    .index.tolist()
)

fig_line = go.Figure()
for cat in top8:
    if cat not in pivot_wide.index:
        continue
    vals = pivot_wide.loc[cat].tolist()
    yr_list = [str(y) for y in pivot_wide.columns]
    fig_line.add_trace(go.Scatter(
        name=cat,
        x=yr_list,
        y=vals,
        mode="lines+markers",
        marker=dict(size=7),
        line=dict(width=2),
        hovertemplate=f"{cat}: %{{y:.1f}}편 (%{{x}})<extra></extra>",
    ))

fig_line.update_layout(
    title=dict(text="주요 카테고리 연도별 논문 수 추이 (상위 8개)", font=dict(size=17)),
    xaxis_title="발행연도",
    yaxis_title="논문 수 (편, 융복합 분산 포함)",
    legend=dict(title="카테고리"),
    width=1100, height=550,
    font=dict(size=12),
)
save_fig(fig_line, "03_category_trend")


# ══════════════════════════════════════════════════════
# 6. 시각화 ④ — 융복합 분야 간 공존 네트워크 (버블 히트맵)
# ══════════════════════════════════════════════════════
print("▶ [4/4] 융복합 분야 간 공존 맵...")

# 융복합 논문에서 분야 쌍(pair) 추출
pair_counter: Counter = Counter()
for cats in df[df["주요분야"] == "융복합"]["구성분야목록"]:
    cats_sorted = sorted(set(cats))
    for i in range(len(cats_sorted)):
        for j in range(i + 1, len(cats_sorted)):
            pair_counter[(cats_sorted[i], cats_sorted[j])] += 1

# 공존 행렬 만들기
all_pair_cats = sorted(set(c for pair in pair_counter for c in pair))
# CORE_CATS 순서 우선 정렬
all_pair_cats = [c for c in CORE_CATS if c in all_pair_cats] + \
                [c for c in all_pair_cats if c not in CORE_CATS]

n = len(all_pair_cats)
matrix = [[0] * n for _ in range(n)]
idx_map = {c: i for i, c in enumerate(all_pair_cats)}
for (c1, c2), cnt in pair_counter.items():
    if c1 in idx_map and c2 in idx_map:
        i, j = idx_map[c1], idx_map[c2]
        matrix[i][j] = cnt
        matrix[j][i] = cnt

fig_fuse = go.Figure(data=go.Heatmap(
    z=matrix,
    x=all_pair_cats,
    y=all_pair_cats,
    colorscale="YlOrRd",
    text=matrix,
    texttemplate="%{text}",
    textfont={"size": 8},
    hovertemplate="%{y} × %{x}: %{z}편<extra></extra>",
    colorbar=dict(title="공존 논문 수"),
))
fig_fuse.update_layout(
    title=dict(text="융복합 논문 분야 간 공존 빈도 (카테고리 간 교차 분석)", font=dict(size=16)),
    width=850, height=800,
    font=dict(size=10),
    margin=dict(l=160, r=60, t=70, b=160),
    xaxis=dict(tickangle=-45),
)
save_fig(fig_fuse, "04_fusion_map")


# ══════════════════════════════════════════════════════
# 7. 요약 통계 출력
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  📊 카테고리별 전체 요약 통계")
print("=" * 55)

cat_summary = (
    df.groupby("주요분야")
    .size()
    .reset_index(name="편수")
    .sort_values("편수", ascending=False)
)
cat_summary["비율(%)"] = (cat_summary["편수"] / total * 100).round(1)
print(cat_summary.to_string(index=False))

print(f"\n  연도별 논문 수:")
yr_counts = df.groupby("발행연도").size().reset_index(name="편수")
print(yr_counts.to_string(index=False))

print("\n" + "=" * 55)
print("✅ 분석 완료! 생성된 파일:")
print(f"   {OUT_DIR}/01_annual_stacked_bar.html/png")
print(f"   {OUT_DIR}/02_category_heatmap.html/png")
print(f"   {OUT_DIR}/03_category_trend.html/png")
print(f"   {OUT_DIR}/04_fusion_map.html/png")
print(f"   {OUT_DIR}/08_category_annual_table.csv")
print("=" * 55)
