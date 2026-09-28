"""
키워드 커뮤니티 순위 변동 분석
- 기간: 초기(2016-2018) / 중기(2019-2022) / 후기(2023-2025)
- ChatGPT 출시(2022.11) 이후 연구 반응이 나타나는 2023년을 후기 기점으로 설정
- 연도별 점유율(share) 기준 순위 계산
- 유형 분류 없이 커뮤니티별 수치를 그대로 제시
- 출력: community_share_trend.png, community_rank_bubble.png, community_rank_bump.png, community_period_table.xlsx/csv
"""

# ── 임포트 ────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from scipy import stats
from adjustText import adjust_text
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── 폰트 설정 ──────────────────────────────────────────────────────────────
font_candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]
for fp in font_candidates:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        plt.rcParams["font.family"] = prop.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 데이터 로드 ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(
    os.path.join(BASE, "results/05_keyword_community_trend/kw_community_trend_data.csv")
)
df.columns = df.columns.str.strip()

# ── 연도별 전체 논문 수 & 점유율 계산 ─────────────────────────────────────
total_by_year = df.groupby("year")["annual_count"].sum().rename("total")
df = df.merge(total_by_year, on="year")
df["share"] = df["annual_count"] / df["total"] * 100

# ── 연도별 순위 (share 기준) ───────────────────────────────────────────────
df["rank"] = df.groupby("year")["share"].rank(ascending=False, method="min").astype(int)

# ── 기간 정의 ──────────────────────────────────────────────────────────────
PERIODS = {
    "초기": (2016, 2018),
    "중기": (2019, 2022),
    "후기": (2023, 2025),
}

def period_avg(col: str, start: int, end: int) -> pd.Series:
    mask = (df["year"] >= start) & (df["year"] <= end)
    return df[mask].groupby("community_label")[col].mean()

early_rank  = period_avg("rank",         *PERIODS["초기"])
mid_rank    = period_avg("rank",         *PERIODS["중기"])
late_rank   = period_avg("rank",         *PERIODS["후기"])
early_share = period_avg("share",        *PERIODS["초기"])
mid_share   = period_avg("share",        *PERIODS["중기"])
late_share  = period_avg("share",        *PERIODS["후기"])
early_count = period_avg("annual_count", *PERIODS["초기"])
mid_count   = period_avg("annual_count", *PERIODS["중기"])
late_count  = period_avg("annual_count", *PERIODS["후기"])

# ── slope & cv ─────────────────────────────────────────────────────────────
communities = df["community_label"].unique()

def get_slope(label: str) -> float:
    sub = df[df["community_label"] == label].sort_values("year")
    if len(sub) < 3:
        return 0.0
    slope, *_ = stats.linregress(sub["year"], sub["share"])
    return slope

def cv_share(label: str) -> float:
    sub = df[df["community_label"] == label]["share"]
    return sub.std() / sub.mean() if sub.mean() > 0 else np.nan

slopes = {c: get_slope(c) for c in communities}

# ── 요약 테이블 ────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "early_rank":  early_rank,  "mid_rank":  mid_rank,  "late_rank":  late_rank,
    "early_share": early_share, "mid_share": mid_share, "late_share": late_share,
    "early_count": early_count, "mid_count": mid_count, "late_count": late_count,
    "rank_e2l":    late_rank - early_rank,
    "slope":       pd.Series(slopes),
}).dropna()
summary["cv"] = summary.index.map(cv_share)
# 후기 점유율 내림차순 정렬
summary = summary.sort_values("late_share", ascending=False)

# ── 커뮤니티별 고정 색상 (tab20) ──────────────────────────────────────────
cmap = plt.get_cmap("tab20")
comm_list = list(summary.index)
comm_colors = {lbl: cmap(i / max(len(comm_list) - 1, 1)) for i, lbl in enumerate(comm_list)}

# ── 결과 출력 ──────────────────────────────────────────────────────────────
print("=" * 100)
print("[ 키워드 커뮤니티 3기 비교 ]  초기 2016-18 / 중기 2019-22 / 후기 2023-25")
print("=" * 100)
print(f"  {'커뮤니티':<40} {'초기순위':>6} {'중기순위':>6} {'후기순위':>6} "
      f"{'초기%':>7} {'중기%':>7} {'후기%':>7} {'순위변동':>8}")
print("  " + "-" * 96)
for lbl, r in summary.iterrows():
    arrow = "↑" if r["rank_e2l"] < 0 else ("↓" if r["rank_e2l"] > 0 else "─")
    print(f"  {str(lbl)[:40]:<40} {r['early_rank']:>6.1f} {r['mid_rank']:>6.1f} "
          f"{r['late_rank']:>6.1f} {r['early_share']:>6.1f}% "
          f"{r['mid_share']:>6.1f}% {r['late_share']:>6.1f}%  "
          f"{arrow}{abs(r['rank_e2l']):.1f}")

# ── 표 저장 (CSV + Excel) ──────────────────────────────────────────────────
table = summary.copy()
table.index.name = "커뮤니티"
table = table.reset_index()

cols = [
    "커뮤니티",
    "early_count", "early_share", "early_rank",
    "mid_count",   "mid_share",   "mid_rank",
    "late_count",  "late_share",  "late_rank",
    "rank_e2l",    "slope",       "cv",
]
col_kr = {
    "커뮤니티":    "커뮤니티",
    "early_count": "초기_평균편수", "early_share": "초기_점유율(%)", "early_rank": "초기_순위",
    "mid_count":   "중기_평균편수", "mid_share":   "중기_점유율(%)", "mid_rank":   "중기_순위",
    "late_count":  "후기_평균편수", "late_share":  "후기_점유율(%)", "late_rank":  "후기_순위",
    "rank_e2l":    "순위변동(후기-초기)", "slope": "연간기울기", "cv": "변동계수(CV)",
}
out_table = table[cols].rename(columns=col_kr).round({
    "초기_평균편수": 1, "초기_점유율(%)": 1, "초기_순위": 1,
    "중기_평균편수": 1, "중기_점유율(%)": 1, "중기_순위": 1,
    "후기_평균편수": 1, "후기_점유율(%)": 1, "후기_순위": 1,
    "순위변동(후기-초기)": 1, "연간기울기": 3, "변동계수(CV)": 3,
})

out_dir   = os.path.join(BASE, "results/05_keyword_community_trend")
csv_path  = os.path.join(out_dir, "community_period_table.csv")
xlsx_path = os.path.join(out_dir, "community_period_table.xlsx")
out_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    out_table.to_excel(writer, index=False, sheet_name="커뮤니티_3기비교")
    ws = writer.sheets["커뮤니티_3기비교"]

    header_fill = PatternFill("solid", fgColor="2C3E50")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:
        cell.fill      = header_fill
        cell.font      = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = border

    # 행 교대 음영 (유형 구분 없이 단순 줄무늬)
    for i, excel_row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=1):
        fill_color = "F8F9FA" if i % 2 == 0 else "FFFFFF"
        row_fill   = PatternFill("solid", fgColor=fill_color)
        for cell in excel_row:
            cell.fill      = row_fill
            cell.border    = border
            cell.alignment = Alignment(horizontal="center", vertical="center")
        excel_row[0].alignment = Alignment(horizontal="left", vertical="center")

    ws.column_dimensions["A"].width = 38
    for col_idx in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 13
    ws.row_dimensions[1].height = 32

print(f"\n✅ 표 저장: {csv_path}")
print(f"✅ 표 저장: {xlsx_path}")

# ── 시각화 1: 점유율 시계열 ────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(13, 7))

for yr, vline_label in [(2019, "중기 시작"), (2023, "후기 시작\n(ChatGPT 반응)")]:
    ax1.axvline(x=yr, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax1.text(yr + 0.1, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 30,
             vline_label, fontsize=7.5, color="gray", va="top")

texts_trend = []
for comm_label in comm_list:
    sub_c = df[df["community_label"] == comm_label].sort_values("year")
    color = comm_colors[comm_label]
    ax1.plot(sub_c["year"], sub_c["share"],
             color=color, lw=1.8, alpha=0.85,
             marker="o", markersize=4)
    last = sub_c.iloc[-1]
    texts_trend.append(
        ax1.text(last["year"] + 0.1, last["share"],
                 str(comm_label).split("/")[0].strip(),
                 fontsize=7.5, color=color, va="center")
    )

adjust_text(texts_trend, ax=ax1, expand=(1.0, 1.4))

ax1.set_xlabel("연도", fontsize=11)
ax1.set_ylabel("연간 점유율 (%)", fontsize=11)
ax1.set_title("키워드 커뮤니티별 연간 점유율 추이\n(초기 2016–2018 / 중기 2019–2022 / 후기 2023–2025)",
              fontsize=13)
ax1.set_xticks(sorted(df["year"].unique()))
ax1.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
out1 = os.path.join(out_dir, "community_share_trend.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"✅ 저장: {out1}")

# ── 시각화 2: 초기 vs 후기 순위 버블 차트 ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 8))

for lbl, r in summary.iterrows():
    color = comm_colors[lbl]
    size  = max(r["late_share"] * 70, 40)
    ax2.scatter(r["early_rank"], r["late_rank"],
                s=size, color=color, alpha=0.82,
                edgecolors="white", linewidth=1.2, zorder=3)
    ax2.annotate(str(lbl).split("/")[0].strip(),
                 xy=(r["early_rank"], r["late_rank"]),
                 xytext=(5, 4), textcoords="offset points",
                 fontsize=8, color=color)

lim = 14
ax2.plot([1, lim], [1, lim], "k--", alpha=0.25, lw=1.2)
ax2.fill_between([1, lim], [1, 1], [1, lim], alpha=0.04, color="#2980b9")
ax2.fill_between([1, lim], [1, lim], [lim, lim], alpha=0.04, color="#e74c3c")
ax2.text(1.3, 13.5, "▲ 순위 상승", fontsize=9, color="#2980b9", alpha=0.8)
ax2.text(9.0, 1.3,  "▼ 순위 하락", fontsize=9, color="#e74c3c", alpha=0.8)

ax2.set_xlabel("초기 평균 순위 (2016–2018)  ※ 1 = 1위", fontsize=11)
ax2.set_ylabel("후기 평균 순위 (2023–2025)  ※ 1 = 1위", fontsize=11)
ax2.set_title("연구군 순위 변동 버블 차트\n(버블 크기 = 후기 점유율 / 초기 2016–18 / 후기 2023–25)",
              fontsize=12)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.set_xlim(lim + 0.5, 0.5)
ax2.set_ylim(lim + 0.5, 0.5)
ax2.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
out2 = os.path.join(out_dir, "community_rank_bubble.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"✅ 저장: {out2}")

# ── 시각화 3: 범프 차트 ───────────────────────────────────────────────────
def jitter_ranks(summary_df, period_cols, spread=0.22):
    jittered = {lbl: [summary_df.loc[lbl, c] for c in period_cols]
                for lbl in summary_df.index}
    for xi, col in enumerate(period_cols):
        groups = {}
        for lbl in summary_df.index:
            rv = float(summary_df.loc[lbl, col])
            groups.setdefault(rv, []).append(lbl)
        for rv, lbls in groups.items():
            n = len(lbls)
            if n > 1:
                offsets = np.linspace(-spread * (n - 1) / 2,
                                       spread * (n - 1) / 2, n)
                for lbl, off in zip(lbls, offsets):
                    jittered[lbl][xi] = rv + off
    return jittered

period_cols = ["early_rank", "mid_rank", "late_rank"]
jittered    = jitter_ranks(summary, period_cols)

x_pos    = [0, 1, 2]
x_labels = ["초기\n(2016–2018)", "중기\n(2019–2022)", "후기\n(2023–2025)"]

fig3, ax3 = plt.subplots(figsize=(16, 12))

texts_left:  list = []
texts_right: list = []

for lbl, r in summary.iterrows():
    color = comm_colors[lbl]
    jy    = jittered[lbl]
    ry    = [r["early_rank"], r["mid_rank"], r["late_rank"]]

    ax3.plot(x_pos, jy,
             color=color, lw=2.2, alpha=0.85,
             marker="o", markersize=9,
             markerfacecolor="white", markeredgewidth=2.2,
             zorder=3, solid_capstyle="round")

    short = str(lbl).split("/")[0].strip()
    texts_left.append(
        ax3.text(-0.12, jy[0], short,
                 ha="right", va="center", fontsize=8,
                 color=color, fontweight="bold")
    )
    texts_right.append(
        ax3.text(2.12, jy[2],
                 f"{short}  {r['late_share']:.1f}%",
                 ha="left", va="center", fontsize=8,
                 color=color, fontweight="bold")
    )
    # 마커 안에 실제 순위 숫자 표시
    for xi, (jv, rv) in enumerate(zip(jy, ry)):
        ax3.text(xi, jv, f"{int(rv)}",
                 ha="center", va="center", fontsize=7,
                 color="white", fontweight="bold", zorder=5)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels, fontsize=13, fontweight="bold")
ax3.set_xlim(-3.2, 5.8)
ax3.invert_yaxis()
ax3.set_ylim(14.5, 0.0)
ax3.set_ylabel("점유율 순위  (1위 = 최상위)", fontsize=11)
ax3.set_title("연구 주제군 순위 변동 차트\n"
              "(초기 2016–2018  /  중기 2019–2022  /  후기 2023–2025)",
              fontsize=13, pad=16)
ax3.yaxis.set_major_locator(MultipleLocator(1))
ax3.grid(axis="y", linestyle="--", alpha=0.2)
ax3.spines[["top", "right", "left"]].set_visible(False)

adjust_text(texts_left,  ax=ax3, expand=(1.0, 1.5))
adjust_text(texts_right, ax=ax3, expand=(1.0, 1.5))

plt.tight_layout()
out3 = os.path.join(out_dir, "community_rank_bump.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"✅ 저장: {out3}")
