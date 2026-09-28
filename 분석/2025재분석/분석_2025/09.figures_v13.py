"""
09.figures_v13.py
=================
V13 원고용 그림 4종을 2025 인용데이터 반영본으로 다시 만든다.

배경
----
V12 원고에 박혀 있는 그림 2·3은 2025 재분석 이전 데이터로 그려진 것이다.
해당 스크립트(05.keyword_community_selected_growth.py, 05b.rank_bump_redesign.py)가
`분석_2025/` 폴더로 복사되지 않아 재실행 대상에서 빠졌기 때문이다.
이 스크립트는 두 그림을 새 데이터로 다시 그리고, 이미 갱신된 그림 1·4를 함께 모아
`results/원고그림_V13/`에 그림1~4로 정리한다. 동시에 본문에 들어갈 수치를 출력한다.

위치
----
`2025인용_재수집/분석_2025/09.figures_v13.py` 로 두고 실행한다.

입력
----
  results/05_keyword_community_trend/kw_community_trend_data.csv   (그림2·3)
  results/05_keyword_trend/keyword_top10_by_year.png               (그림1, 복사만)
  results/03_citation_topic_flow/topic_citation_matrix_heatmap.png (그림4, 복사만)

출력
----
  results/원고그림_V13/그림1_연도별_주요키워드.png
  results/원고그림_V13/그림2_주제군_누적논문수.png  (+ .pdf)
  results/원고그림_V13/그림3_주제군_순위변동.png    (+ .pdf)
  results/원고그림_V13/그림4_주제군간_인용행렬.png
  results/원고그림_V13/그림2_데이터.csv
  results/원고그림_V13/그림3_데이터.csv
  results/원고그림_V13/본문수치_점검.txt

실행
----
  python 09.figures_v13.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
TREND = RESULTS / "05_keyword_community_trend" / "kw_community_trend_data.csv"
FIG1_SRC = RESULTS / "05_keyword_trend" / "keyword_top10_by_year.png"
FIG4_SRC = RESULTS / "03_citation_topic_flow" / "topic_citation_matrix_heatmap.png"
OUT = RESULTS / "원고그림_V13"

YEARS = list(range(2016, 2026))
PERIODS = {"초기": (2016, 2018), "중기": (2019, 2022), "후기": (2023, 2025)}

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def setup_font() -> None:
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


# 그림2에 표시할 5개 주제군 (V12와 동일한 선택·색·라벨)
SELECTED = [
    {
        "source": "생성형 인공지능 / 저작권 / 공정이용",
        "label": "생성형 AI·저작권",
        "color": "#E63946",
    },
    {
        "source": "빅데이터 / 개인정보 / 개인정보보호",
        "label": "빅데이터·개인정보",
        "color": "#2A9D8F",
    },
    {
        "source": "4차 산업혁명 / 블록체인 / 지능정보사회",
        "label": "4차 산업혁명·거버넌스",
        "color": "#1D3557",
    },
    {
        "source": "알고리즘 / 머신러닝 / 딥러닝",
        "label": "알고리즘·머신러닝",
        "color": "#F4A261",
    },
    {
        "source": "EU AI Act / 인공지능기본법 / 인공지능법",
        "label": "EU AI Act·인공지능기본법",
        "color": "#6A4C93",
    },
]

EVENTS = {
    2018: "GDPR",
    2021: "EU AI Act\n초안",
    2022: "ChatGPT",
    2024: "EU AI Act\n채택",
    2025: "인공지능\n기본법",
}


def load_trend() -> pd.DataFrame:
    if not TREND.exists():
        raise FileNotFoundError(
            f"{TREND} 가 없다. 04.keyword_leiden → 05.keyword_community_trend 를 먼저 실행할 것."
        )
    df = pd.read_csv(TREND, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


# ────────────────────────────────────────────────────────────────────
# 그림2 — 주요 주제군 누적 논문 수
# ────────────────────────────────────────────────────────────────────
def fig2(df: pd.DataFrame) -> pd.DataFrame:
    sources = {s["source"] for s in SELECTED}
    missing = sources - set(df["community_label"].unique())
    if missing:
        raise ValueError(f"입력 데이터에 없는 주제군: {sorted(missing)}")

    label_map = {s["source"]: s["label"] for s in SELECTED}
    color_map = {s["label"]: s["color"] for s in SELECTED}
    order = [s["label"] for s in SELECTED]

    sel = df[df["community_label"].isin(sources)].copy()
    sel["display_label"] = sel["community_label"].map(label_map)
    sel = sel.sort_values(["display_label", "year"])
    sel.to_csv(OUT / "그림2_데이터.csv", index=False, encoding="utf-8-sig")

    pivot = (
        sel.pivot_table(
            index="year", columns="display_label", values="cumul_count", aggfunc="sum"
        )
        .reindex(index=YEARS, columns=order)
        .ffill()
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor="#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for label in order:
        values = pivot[label].astype(int)
        ax.plot(
            YEARS,
            values,
            marker="o",
            markersize=5,
            linewidth=2.6,
            color=color_map[label],
            label=label,
        )
        ax.text(
            YEARS[-1] + 0.12,
            values.iloc[-1],
            f"{values.iloc[-1]}편",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=color_map[label],
        )

    ymax = float(pivot.to_numpy().max()) * 1.14
    for year, label in EVENTS.items():
        ax.axvline(year, color="#777777", linewidth=0.8, linestyle=":", alpha=0.55)
        ax.text(
            year,
            ymax * 0.98,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="#DDDDDD",
                alpha=0.86,
            ),
        )

    ax.set_title(
        "주요 키워드 커뮤니티의 누적 논문 수 변화(2016-2025)",
        fontsize=15,
        fontweight="bold",
        color="#1D3557",
        pad=16,
    )
    ax.set_xlabel("발행연도", fontsize=11)
    ax.set_ylabel("누적 논문 수", fontsize=11)
    ax.set_xticks(YEARS)
    ax.set_xlim(YEARS[0] - 0.25, YEARS[-1] + 1.0)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.36)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.02,
        "주: 논문-커뮤니티 매핑 결과를 기준으로 집계. 주요 5개 커뮤니티만 표시.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for ext, dpi in [("png", 220), ("pdf", None)]:
        p = OUT / f"그림2_주제군_누적논문수.{ext}"
        plt.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="#F8F9FA")
        print(f"[저장] {p}")
    plt.close(fig)
    return pivot


# ────────────────────────────────────────────────────────────────────
# 그림3 — 주제군 순위 변동 (기간 합산 점유율)
# ────────────────────────────────────────────────────────────────────
def period_rank_share(df: pd.DataFrame, s: int, e: int):
    sub = df[df.year.between(s, e)]
    agg = sub.groupby("community_label")["annual_count"].sum()
    share = agg / agg.sum() * 100
    rank = share.rank(ascending=False, method="first").astype(int)
    return rank, share


def pack_labels(node_ys, min_gap, y_lo, y_hi):
    order = list(np.argsort(node_ys))
    y = np.array(node_ys, float)
    for _ in range(200):
        moved = False
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            if y[b] - y[a] < min_gap:
                push = (min_gap - (y[b] - y[a])) / 2
                y[a] -= push
                y[b] += push
                moved = True
        y = np.clip(y, y_lo, y_hi)
        if not moved:
            break
    return y


def fig3(df: pd.DataFrame) -> pd.DataFrame:
    er, es = period_rank_share(df, *PERIODS["초기"])
    mr, ms = period_rank_share(df, *PERIODS["중기"])
    lr, ls = period_rank_share(df, *PERIODS["후기"])

    summary = (
        pd.DataFrame(
            {
                "early_rank": er,
                "early_share": es,
                "mid_rank": mr,
                "mid_share": ms,
                "late_rank": lr,
                "late_share": ls,
            }
        )
        .dropna()
        .sort_values("late_rank")
    )
    summary.round(1).to_csv(OUT / "그림3_데이터.csv", encoding="utf-8-sig")

    cmap = plt.get_cmap("tab20")
    comm = list(summary.index)
    colors = {l: cmap(i / max(len(comm) - 1, 1)) for i, l in enumerate(comm)}

    x_pos = [0, 1, 2]
    x_labels = ["초기\n2016–2018", "중기\n2019–2022", "후기\n2023–2025"]
    N = len(summary)
    Y_LO, Y_HI = 0.6, N + 0.4
    MIN_GAP = 0.62

    fig, ax = plt.subplots(figsize=(12.5, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for xp in x_pos:
        ax.axvline(xp, color="#DDDDDD", lw=1.0, zorder=0)

    node_left, node_right = {}, {}
    for lbl, r in summary.iterrows():
        color = colors[lbl]
        ys = [r["early_rank"], r["mid_rank"], r["late_rank"]]
        ax.plot(
            x_pos,
            ys,
            color=color,
            lw=2.6,
            alpha=0.9,
            marker="o",
            markersize=15,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.6,
            zorder=3,
            solid_capstyle="round",
        )
        for xi, rv in zip(x_pos, ys):
            ax.text(
                xi,
                rv,
                f"{int(rv)}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold",
                zorder=5,
            )
        node_left[lbl] = ys[0]
        node_right[lbl] = ys[2]

    labels = list(summary.index)
    lyL = pack_labels([node_left[l] for l in labels], MIN_GAP, Y_LO, Y_HI)
    lyR = pack_labels([node_right[l] for l in labels], MIN_GAP, Y_LO, Y_HI)

    for l, yy in zip(labels, lyL):
        color = colors[l]
        short = str(l).split("/")[0].strip()
        ax.plot(
            [-0.045, -0.008], [yy, node_left[l]], color=color, lw=0.8, alpha=0.55, zorder=2
        )
        ax.text(-0.06, yy, short, ha="right", va="center", fontsize=11, color=color, fontweight="bold")
    for l, yy in zip(labels, lyR):
        color = colors[l]
        short = str(l).split("/")[0].strip()
        ax.plot(
            [2.008, 2.045], [node_right[l], yy], color=color, lw=0.8, alpha=0.55, zorder=2
        )
        ax.text(
            2.06,
            yy,
            f"{short}  ({summary.loc[l, 'late_share']:.1f}%)",
            ha="left",
            va="center",
            fontsize=11,
            color=color,
            fontweight="bold",
        )

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
    ax.set_title(
        "연구 주제군 순위 변동 (기간 합산 점유율 기준)\n"
        "초기 2016–2018  ·  중기 2019–2022  ·  후기 2023–2025",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    plt.tight_layout()
    for ext, dpi in [("png", 200), ("pdf", None)]:
        p = OUT / f"그림3_주제군_순위변동.{ext}"
        plt.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[저장] {p}")
    plt.close(fig)
    return summary


# ────────────────────────────────────────────────────────────────────
# 본문 수치 점검표
# ────────────────────────────────────────────────────────────────────
def digest(df: pd.DataFrame, pivot: pd.DataFrame, summary: pd.DataFrame) -> str:
    L = []
    add = L.append
    add("본문에 들어갈 수치 (2025 인용데이터 반영본)")
    add("=" * 58)
    add("")
    add("[그림2 관련 — 4.3절 / 5.3절]")
    for label in pivot.columns:
        v = pivot[label].astype(int)
        add(f"  {label:24s} 2022 누적 {v.loc[2022]:>4d}편 → 2025 누적 {v.loc[2025]:>4d}편"
            f" (3년간 +{v.loc[2025]-v.loc[2022]}편)")
    add("")
    add("[그림3 관련 — 4.3절] 기간 합산 점유율·순위")
    add(f"  {'주제군':<28s} {'초기':>12s} {'중기':>12s} {'후기':>12s}")
    for lbl, r in summary.iterrows():
        short = str(lbl).split("/")[0].strip()
        add(
            f"  {short:<28s} {int(r.early_rank):>3d}위({r.early_share:4.1f}%)"
            f" {int(r.mid_rank):>3d}위({r.mid_share:4.1f}%)"
            f" {int(r.late_rank):>3d}위({r.late_share:4.1f}%)"
        )
    add("")
    add("[5.3절] 4차 산업혁명·지능정보사회 주제군 연도별")
    tot = df.groupby("year")["annual_count"].sum()
    lab4 = "4차 산업혁명 / 블록체인 / 지능정보사회"
    s4 = df[df.community_label == lab4].set_index("year")["annual_count"]
    for y in [2017, 2020, 2024, 2025]:
        sub = df[df.year == y].sort_values("annual_count", ascending=False).reset_index(drop=True)
        rank = sub[sub.community_label == lab4].index[0] + 1
        add(f"  {y}: {s4[y]:>3d}편  점유율 {s4[y]/tot[y]*100:4.1f}%  {rank}위")
    txt = "\n".join(L)
    (OUT / "본문수치_점검.txt").write_text(txt, encoding="utf-8")
    return txt


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_trend()

    pivot = fig2(df)
    summary = fig3(df)

    for src, dst in [
        (FIG1_SRC, OUT / "그림1_연도별_주요키워드.png"),
        (FIG4_SRC, OUT / "그림4_주제군간_인용행렬.png"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[복사] {dst}")
        else:
            print(f"[경고] {src} 없음 — 해당 단계를 먼저 실행할 것")

    print()
    print(digest(df, pivot, summary))
    print()
    print(f"→ 그림 4종 모두 {OUT} 에 있음. 한글 원고에서 이 파일들로 교체.")


if __name__ == "__main__":
    main()
