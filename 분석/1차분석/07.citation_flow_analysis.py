"""
4.6 후보 분석: 인용 관계로 본 AI 법학 연구 흐름의 축적

입력:
  - 00.KCI_AI_논문_상세_및_인용데이터.csv
  - KCI_AI_논문_카테고리_확정.csv (있으면 분야 간 내부 인용 흐름 계산)

출력:
  - results/03_citation/07_citation_flow_metrics.csv
  - results/03_citation/07_internal_citation_lag.csv
  - results/03_citation/07_period_top_internal_cited.csv
  - results/03_citation/07_category_citation_flow.csv
  - results/03_citation/07_citation_flow_summary.txt
  - results/03_citation/07_internal_citation_flow.png
  - results/03_citation/07_internal_citation_lag.png

해석 초점:
  - AI 법학 연구가 시간이 지나며 내부 선행연구를 얼마나 더 자주 참조하는가
  - 내부 인용이 단기 최신 논문 중심인지, 더 오래된 선행연구까지 축적되는지
  - 시기별로 어떤 논문이 인용 기반의 중심 참고점으로 작동하는지
  - 분야 간 내부 인용이 특정 분야 내부에 머무는지, 교차 분야로 확장되는지

※ 2025년은 인용 데이터가 불완전하므로 기존 06번 분석과 동일하게 제외한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


BASE = Path(__file__).parent
INPUT_FILE = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
CATEGORY_FILE = BASE / "KCI_AI_논문_카테고리_확정.csv"
OUT_DIR = BASE / "results" / "03_citation"

YEAR_START = 2016
YEAR_END = 2024
PERIODS = {
    "2016-2018": (2016, 2018),
    "2019-2020": (2019, 2020),
    "2021-2022": (2021, 2022),
    "2023-2024": (2023, 2024),
}


def setup_korean_font() -> None:
    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_family = "DejaVu Sans"
    for font_path in font_candidates:
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
            font_family = fm.FontProperties(fname=font_path).get_name()
            break

    plt.rcParams.update(
        {
            "font.family": [font_family, "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, str]]:
    df = pd.read_csv(INPUT_FILE)
    df["source_id"] = df["source_id"].astype(str)
    df["target_arti_id"] = df["target_arti_id"].fillna("").astype(str)

    papers = (
        df.drop_duplicates(subset="source_id")[
            ["source_id", "title", "pub_year", "category"]
        ]
        .query(f"{YEAR_START} <= pub_year <= {YEAR_END}")
        .copy()
    )
    papers["pub_year"] = papers["pub_year"].astype(int)

    ai_ids = set(papers["source_id"])
    id2year = papers.set_index("source_id")["pub_year"].to_dict()
    id2title = papers.set_index("source_id")["title"].to_dict()

    edges = (
        df[
            (df["source_id"].isin(ai_ids))
            & (df["target_arti_id"] != "")
            & (df["target_arti_id"].str.startswith("ART"))
        ][["source_id", "target_arti_id"]]
        .drop_duplicates()
        .copy()
    )
    edges["source_year"] = edges["source_id"].map(id2year)
    edges["target_year"] = edges["target_arti_id"].map(id2year)
    edges["is_internal"] = edges["target_arti_id"].isin(ai_ids)

    internal_edges = edges[edges["is_internal"]].copy()
    external_edges = edges[~edges["is_internal"]].copy()
    return papers, internal_edges, external_edges, id2year, id2title


def build_annual_metrics(
    papers: pd.DataFrame,
    internal_edges: pd.DataFrame,
    external_edges: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for year in range(YEAR_START, YEAR_END + 1):
        year_papers = papers[papers["pub_year"] == year]
        paper_count = len(year_papers)
        paper_ids = set(year_papers["source_id"])

        internal_count_by_paper = (
            internal_edges[internal_edges["source_year"] == year]
            .groupby("source_id")
            .size()
        )
        external_count_by_paper = (
            external_edges[external_edges["source_year"] == year]
            .groupby("source_id")
            .size()
        )

        internal_count = int(internal_count_by_paper.sum())
        external_count = int(external_count_by_paper.sum())
        total_count = internal_count + external_count
        papers_with_internal = len(set(internal_count_by_paper.index) & paper_ids)
        cumulative_prior = int((papers["pub_year"] < year).sum())

        rows.append(
            {
                "year": year,
                "paper_count": paper_count,
                "cumulative_prior_ai_papers": cumulative_prior,
                "internal_citations": internal_count,
                "external_kci_citations": external_count,
                "total_kci_citations": total_count,
                "internal_ratio_pct": round(
                    internal_count / total_count * 100, 1
                )
                if total_count
                else 0.0,
                "avg_internal_citations_per_paper": round(
                    internal_count / paper_count, 2
                )
                if paper_count
                else 0.0,
                "avg_external_kci_citations_per_paper": round(
                    external_count / paper_count, 2
                )
                if paper_count
                else 0.0,
                "papers_with_internal_citation": papers_with_internal,
                "papers_with_internal_citation_pct": round(
                    papers_with_internal / paper_count * 100, 1
                )
                if paper_count
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_lag_metrics(internal_edges: pd.DataFrame) -> pd.DataFrame:
    lag_edges = internal_edges.dropna(subset=["source_year", "target_year"]).copy()
    lag_edges["citation_lag"] = (
        lag_edges["source_year"].astype(int) - lag_edges["target_year"].astype(int)
    )
    # 발행연도가 뒤인 논문을 인용한 것으로 표시되는 데이터 오류/동시 온라인 출판 흔적은 제외한다.
    lag_edges = lag_edges[lag_edges["citation_lag"] >= 0]

    rows = []
    for year, sub in lag_edges.groupby("source_year"):
        lag = sub["citation_lag"]
        rows.append(
            {
                "year": int(year),
                "internal_citations_with_year": len(sub),
                "avg_lag_years": round(lag.mean(), 2),
                "median_lag_years": round(lag.median(), 2),
                "same_year_pct": round((lag == 0).mean() * 100, 1),
                "recent_1_2_year_pct": round(lag.between(1, 2).mean() * 100, 1),
                "mid_3_5_year_pct": round(lag.between(3, 5).mean() * 100, 1),
                "older_6plus_year_pct": round((lag >= 6).mean() * 100, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def build_period_top_cited(
    papers: pd.DataFrame,
    internal_edges: pd.DataFrame,
    id2title: dict[str, str],
    id2year: dict[str, int],
) -> pd.DataFrame:
    paper_count_by_period = {
        period: len(papers[papers["pub_year"].between(start, end)])
        for period, (start, end) in PERIODS.items()
    }

    rows = []
    for period, (start, end) in PERIODS.items():
        sub = internal_edges[internal_edges["source_year"].between(start, end)]
        cited = sub.groupby("target_arti_id").size().sort_values(ascending=False)
        for rank, (paper_id, citation_count) in enumerate(cited.head(15).items(), 1):
            rows.append(
                {
                    "period": period,
                    "rank": rank,
                    "paper_id": paper_id,
                    "title": id2title.get(paper_id, paper_id),
                    "pub_year": id2year.get(paper_id),
                    "internal_citations_from_period": int(citation_count),
                    "citations_per_100_papers": round(
                        citation_count / paper_count_by_period[period] * 100, 1
                    )
                    if paper_count_by_period[period]
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_category_flow(internal_edges: pd.DataFrame) -> pd.DataFrame:
    if not CATEGORY_FILE.exists():
        return pd.DataFrame()

    categories = pd.read_csv(CATEGORY_FILE)[["논문ID", "최종_카테고리"]].drop_duplicates()
    id2cat = categories.set_index("논문ID")["최종_카테고리"].to_dict()

    flow = internal_edges.copy()
    flow["source_category"] = flow["source_id"].map(id2cat)
    flow["target_category"] = flow["target_arti_id"].map(id2cat)
    flow = flow.dropna(subset=["source_category", "target_category"])

    rows = []
    for period, (start, end) in PERIODS.items():
        sub = flow[flow["source_year"].between(start, end)]
        if sub.empty:
            continue

        grouped = (
            sub.groupby(["source_category", "target_category"])
            .size()
            .reset_index(name="citation_count")
            .sort_values("citation_count", ascending=False)
        )
        total = grouped["citation_count"].sum()
        grouped["period"] = period
        grouped["share_pct"] = (grouped["citation_count"] / total * 100).round(1)
        grouped["is_same_category"] = (
            grouped["source_category"] == grouped["target_category"]
        )
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    return result[
        [
            "period",
            "source_category",
            "target_category",
            "citation_count",
            "share_pct",
            "is_same_category",
        ]
    ]


def plot_annual_metrics(metrics: pd.DataFrame) -> None:
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    years = metrics["year"]

    ax1.bar(
        years,
        metrics["external_kci_citations"],
        width=0.62,
        color="#C7D7E8",
        label="외부 KCI 인용",
        zorder=2,
    )
    ax1.bar(
        years,
        metrics["internal_citations"],
        bottom=metrics["external_kci_citations"],
        width=0.62,
        color="#3B6EA8",
        label="내부 AI 법학 인용",
        zorder=2,
    )
    ax1.set_ylabel("KCI 인용 건수")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        metrics["internal_ratio_pct"],
        color="#D95F02",
        marker="o",
        linewidth=2.2,
        label="내부 인용 비율",
    )
    ax2.set_ylabel("내부 인용 비율(%)")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.88)

    ax3.plot(
        years,
        metrics["avg_internal_citations_per_paper"],
        color="#1B9E77",
        marker="s",
        linewidth=2.0,
        label="논문당 평균 내부 인용 수",
    )
    ax3.set_ylabel("평균 건수")
    ax3.set_xlabel("연도")
    ax3.set_xticks(years)
    ax3.set_ylim(0, max(metrics["avg_internal_citations_per_paper"]) + 0.8)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)

    ax4 = ax3.twinx()
    ax4.plot(
        years,
        metrics["papers_with_internal_citation_pct"],
        color="#7570B3",
        marker="^",
        linewidth=2.0,
        label="내부 인용 논문 비중",
    )
    ax4.set_ylabel("논문 비중(%)")
    ax4.set_ylim(0, 100)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper left", framealpha=0.88)

    fig.suptitle("AI 법학 연구의 내부 인용 축적 추이", fontsize=14)
    fig.tight_layout()
    plt.savefig(OUT_DIR / "07_internal_citation_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lag_metrics(lag_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    years = lag_metrics["year"]

    bottom = pd.Series([0] * len(lag_metrics), index=lag_metrics.index, dtype=float)
    parts = [
        ("same_year_pct", "동년", "#4C78A8"),
        ("recent_1_2_year_pct", "1-2년 전", "#72B7B2"),
        ("mid_3_5_year_pct", "3-5년 전", "#F2CF5B"),
        ("older_6plus_year_pct", "6년 이상 전", "#E45756"),
    ]
    for column, label, color in parts:
        ax.bar(years, lag_metrics[column], bottom=bottom, label=label, color=color)
        bottom = bottom + lag_metrics[column]

    ax.set_xlabel("연도")
    ax.set_ylabel("내부 인용 시차 구성(%)")
    ax.set_xticks(years)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper center", ncol=4, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.title("연도별 내부 인용의 선행연구 시차")
    fig.tight_layout()
    plt.savefig(OUT_DIR / "07_internal_citation_lag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    papers: pd.DataFrame,
    internal_edges: pd.DataFrame,
    external_edges: pd.DataFrame,
    metrics: pd.DataFrame,
    lag_metrics: pd.DataFrame,
    top_cited: pd.DataFrame,
    category_flow: pd.DataFrame,
) -> None:
    first = metrics.iloc[0]
    last = metrics.iloc[-1]

    with (OUT_DIR / "07_citation_flow_summary.txt").open("w", encoding="utf-8") as f:
        f.write("4.6 후보 분석: 인용 관계로 본 연구 흐름의 축적\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"분석 기간: {YEAR_START}~{YEAR_END} (2025 제외)\n")
        f.write(f"AI 법학 논문 수: {len(papers):,}편\n")
        f.write(f"내부 인용 엣지: {len(internal_edges):,}건\n")
        f.write(f"외부 KCI 인용 엣지: {len(external_edges):,}건\n\n")

        f.write("[핵심 흐름 지표]\n")
        f.write(
            "- 내부 인용 비율: "
            f"{int(first['year'])}년 {first['internal_ratio_pct']:.1f}% -> "
            f"{int(last['year'])}년 {last['internal_ratio_pct']:.1f}%\n"
        )
        f.write(
            "- 논문당 평균 내부 인용 수: "
            f"{int(first['year'])}년 {first['avg_internal_citations_per_paper']:.2f}건 -> "
            f"{int(last['year'])}년 {last['avg_internal_citations_per_paper']:.2f}건\n"
        )
        f.write(
            "- 내부 선행연구를 1건 이상 인용한 논문 비중: "
            f"{int(first['year'])}년 {first['papers_with_internal_citation_pct']:.1f}% -> "
            f"{int(last['year'])}년 {last['papers_with_internal_citation_pct']:.1f}%\n\n"
        )

        if not lag_metrics.empty:
            lag_first = lag_metrics.iloc[0]
            lag_last = lag_metrics.iloc[-1]
            f.write("[내부 인용 시차]\n")
            f.write(
                "- 평균 내부 인용 시차: "
                f"{int(lag_first['year'])}년 {lag_first['avg_lag_years']:.2f}년 -> "
                f"{int(lag_last['year'])}년 {lag_last['avg_lag_years']:.2f}년\n"
            )
            f.write(
                "- 6년 이상 전 선행연구 인용 비중: "
                f"{int(lag_first['year'])}년 {lag_first['older_6plus_year_pct']:.1f}% -> "
                f"{int(lag_last['year'])}년 {lag_last['older_6plus_year_pct']:.1f}%\n\n"
            )

        f.write("[시기별 내부 피인용 상위 논문 Top 5]\n")
        for period in PERIODS:
            sub = top_cited[top_cited["period"] == period].head(5)
            if sub.empty:
                continue
            f.write(f"\n{period}\n")
            for _, row in sub.iterrows():
                title = str(row["title"]).replace("\n", " ")
                if len(title) > 55:
                    title = title[:55] + "..."
                f.write(
                    f"  {int(row['rank'])}. {title} "
                    f"({int(row['pub_year'])}, {int(row['internal_citations_from_period'])}회)\n"
                )

        if not category_flow.empty:
            f.write("\n[분야 간 내부 인용 흐름]\n")
            for period in PERIODS:
                sub = category_flow[category_flow["period"] == period]
                if sub.empty:
                    continue
                same_share = (
                    sub.loc[sub["is_same_category"], "citation_count"].sum()
                    / sub["citation_count"].sum()
                    * 100
                )
                f.write(f"- {period}: 동일 분야 내부 인용 비중 {same_share:.1f}%\n")

        f.write("\n[해석 메모]\n")
        f.write(
            "앞선 키워드 분석이 주제의 변화를 보여준다면, 이 분석은 AI 법학 연구가 "
            "시간이 지나며 자기 분야의 선행연구를 얼마나 축적하고 재참조하는지를 "
            "보여주는 보조 지표로 사용할 수 있다.\n"
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_korean_font()

    papers, internal_edges, external_edges, id2year, id2title = load_data()

    metrics = build_annual_metrics(papers, internal_edges, external_edges)
    lag_metrics = build_lag_metrics(internal_edges)
    top_cited = build_period_top_cited(papers, internal_edges, id2title, id2year)
    category_flow = build_category_flow(internal_edges)

    metrics.to_csv(
        OUT_DIR / "07_citation_flow_metrics.csv", index=False, encoding="utf-8-sig"
    )
    lag_metrics.to_csv(
        OUT_DIR / "07_internal_citation_lag.csv", index=False, encoding="utf-8-sig"
    )
    top_cited.to_csv(
        OUT_DIR / "07_period_top_internal_cited.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not category_flow.empty:
        category_flow.to_csv(
            OUT_DIR / "07_category_citation_flow.csv",
            index=False,
            encoding="utf-8-sig",
        )

    plot_annual_metrics(metrics)
    plot_lag_metrics(lag_metrics)
    write_summary(
        papers,
        internal_edges,
        external_edges,
        metrics,
        lag_metrics,
        top_cited,
        category_flow,
    )

    print("4.6 후보 인용 흐름 분석 완료")
    print(f"- {OUT_DIR / '07_citation_flow_metrics.csv'}")
    print(f"- {OUT_DIR / '07_internal_citation_lag.csv'}")
    print(f"- {OUT_DIR / '07_period_top_internal_cited.csv'}")
    if not category_flow.empty:
        print(f"- {OUT_DIR / '07_category_citation_flow.csv'}")
    print(f"- {OUT_DIR / '07_citation_flow_summary.txt'}")
    print(f"- {OUT_DIR / '07_internal_citation_flow.png'}")
    print(f"- {OUT_DIR / '07_internal_citation_lag.png'}")


if __name__ == "__main__":
    main()
