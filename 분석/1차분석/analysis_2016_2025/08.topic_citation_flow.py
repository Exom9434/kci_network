"""
키워드 주제군별 내부/외부 인용 구조 분석

목적:
  4.4의 키워드 공출현 주제군을 논문 단위 라벨로 사용하여,
  내부 인용 네트워크에서 각 주제군이 자기 주제군을 얼마나 인용하는지,
  다른 주제군과는 어떤 방향으로 연결되는지 집계한다.

입력:
  - 00.KCI_AI_논문_상세_및_인용데이터.csv
  - results/04_keyword_network/community_paper_map.csv

출력:
  - results/03_citation_topic_flow/topic_citation_summary.csv
  - results/03_citation_topic_flow/topic_citation_matrix_counts.csv
  - results/03_citation_topic_flow/topic_citation_matrix_row_pct.csv
  - results/03_citation_topic_flow/topic_citation_top_cross_pairs.csv
  - results/03_citation_topic_flow/topic_external_kci_summary.csv
  - results/03_citation_topic_flow/topic_citation_matrix_heatmap.png

주의:
  2025년 인용 데이터는 불완전하므로, 인용을 하는 source 논문은 2016-2024년으로 제한한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).parent
DATA = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
PAPER_COMMUNITY = BASE / "results" / "04_keyword_network" / "community_paper_map.csv"
OUT_DIR = BASE / "results" / "03_citation_topic_flow"

YEAR_START = 2016
YEAR_END = 2024


def setup_korean_font() -> None:
    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    papers_raw = pd.read_csv(DATA, encoding="utf-8-sig")
    papers_raw["source_id"] = papers_raw["source_id"].astype(str)
    papers_raw["target_arti_id"] = papers_raw["target_arti_id"].fillna("").astype(str)

    paper_meta = (
        papers_raw.drop_duplicates(subset="source_id")[
            ["source_id", "title", "pub_year", "category"]
        ]
        .copy()
    )
    paper_meta["pub_year"] = paper_meta["pub_year"].astype(int)

    paper_comm = pd.read_csv(PAPER_COMMUNITY, encoding="utf-8-sig")
    paper_comm["논문ID"] = paper_comm["논문ID"].astype(str)
    paper_comm = paper_comm.drop_duplicates(subset="논문ID")

    papers = paper_meta.merge(
        paper_comm[["논문ID", "primary_community", "community_label"]],
        left_on="source_id",
        right_on="논문ID",
        how="left",
    )
    papers = papers.drop(columns=["논문ID"])

    source_ids = set(
        papers.loc[
            papers["pub_year"].between(YEAR_START, YEAR_END),
            "source_id",
        ]
    )
    ai_ids = set(papers["source_id"])
    id2year = papers.set_index("source_id")["pub_year"].to_dict()

    edges = (
        papers_raw[
            (papers_raw["source_id"].isin(source_ids))
            & (papers_raw["target_arti_id"] != "")
            & (papers_raw["target_arti_id"].str.startswith("ART"))
        ][["source_id", "target_arti_id"]]
        .drop_duplicates()
        .copy()
    )
    edges["source_year"] = edges["source_id"].map(id2year)
    edges["target_year"] = edges["target_arti_id"].map(id2year)
    edges["is_internal_ai"] = edges["target_arti_id"].isin(ai_ids)

    return papers, paper_comm, edges


def build_topic_edges(papers: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    id2community = papers.set_index("source_id")["primary_community"].to_dict()
    id2label = papers.set_index("source_id")["community_label"].to_dict()

    topic_edges = edges[edges["is_internal_ai"]].copy()
    topic_edges["source_community"] = topic_edges["source_id"].map(id2community)
    topic_edges["target_community"] = topic_edges["target_arti_id"].map(id2community)
    topic_edges["source_label"] = topic_edges["source_id"].map(id2label)
    topic_edges["target_label"] = topic_edges["target_arti_id"].map(id2label)

    topic_edges = topic_edges.dropna(
        subset=["source_community", "target_community", "source_year", "target_year"]
    )

    # 뒤 연도 논문을 인용한 것으로 표시되는 예외는 인용 시차 분석과 동일하게 제외한다.
    topic_edges["citation_lag"] = (
        topic_edges["source_year"].astype(int) - topic_edges["target_year"].astype(int)
    )
    topic_edges = topic_edges[topic_edges["citation_lag"] >= 0].copy()

    topic_edges["source_community"] = topic_edges["source_community"].astype(int)
    topic_edges["target_community"] = topic_edges["target_community"].astype(int)
    topic_edges["is_same_topic"] = (
        topic_edges["source_community"] == topic_edges["target_community"]
    )
    return topic_edges


def build_summary(papers: pd.DataFrame, edges: pd.DataFrame, topic_edges: pd.DataFrame) -> pd.DataFrame:
    labeled_papers = papers.dropna(subset=["primary_community"]).copy()
    labeled_papers["primary_community"] = labeled_papers["primary_community"].astype(int)

    base = (
        labeled_papers.groupby(["primary_community", "community_label"])
        .agg(
            paper_count_all=("source_id", "nunique"),
            paper_count_2016_2024=(
                "source_id",
                lambda s: s[
                    labeled_papers.loc[s.index, "pub_year"].between(YEAR_START, YEAR_END)
                ].nunique(),
            ),
        )
        .reset_index()
        .rename(columns={"primary_community": "community_id"})
    )

    outgoing = (
        topic_edges.groupby("source_community")
        .agg(
            outgoing_internal_ai_citations=("source_id", "size"),
            outgoing_same_topic_citations=("is_same_topic", "sum"),
            citing_papers_with_internal_ai_citation=("source_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"source_community": "community_id"})
    )

    outgoing_cross = (
        topic_edges[~topic_edges["is_same_topic"]]
        .groupby("source_community")
        .agg(
            outgoing_cross_topic_citations=("source_id", "size"),
            citing_papers_with_cross_topic_citation=("source_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"source_community": "community_id"})
    )

    incoming = (
        topic_edges.groupby("target_community")
        .agg(
            incoming_internal_ai_citations=("target_arti_id", "size"),
            cited_papers_by_internal_ai_citation=("target_arti_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"target_community": "community_id"})
    )

    incoming_cross = (
        topic_edges[~topic_edges["is_same_topic"]]
        .groupby("target_community")
        .agg(incoming_from_other_topic_citations=("target_arti_id", "size"))
        .reset_index()
        .rename(columns={"target_community": "community_id"})
    )

    external_kci = edges[~edges["is_internal_ai"]].copy()
    id2community = papers.set_index("source_id")["primary_community"].to_dict()
    external_kci["community_id"] = external_kci["source_id"].map(id2community)
    external_kci = external_kci.dropna(subset=["community_id"])
    external_kci["community_id"] = external_kci["community_id"].astype(int)
    external_summary = (
        external_kci.groupby("community_id")
        .agg(
            outgoing_external_kci_citations=("target_arti_id", "size"),
            citing_papers_with_external_kci_citation=("source_id", "nunique"),
        )
        .reset_index()
    )

    summary = base
    for part in [outgoing, outgoing_cross, incoming, incoming_cross, external_summary]:
        summary = summary.merge(part, on="community_id", how="left")

    count_cols = [
        c
        for c in summary.columns
        if c.endswith("_citations")
        or c.startswith("citing_papers")
        or c.startswith("cited_papers")
    ]
    summary[count_cols] = summary[count_cols].fillna(0).astype(int)

    summary["outgoing_cross_topic_citations"] = summary[
        "outgoing_cross_topic_citations"
    ].fillna(0).astype(int)
    summary["outgoing_same_topic_citations"] = summary[
        "outgoing_same_topic_citations"
    ].fillna(0).astype(int)
    summary["incoming_from_other_topic_citations"] = summary[
        "incoming_from_other_topic_citations"
    ].fillna(0).astype(int)

    summary["same_topic_ratio_pct"] = (
        summary["outgoing_same_topic_citations"]
        / summary["outgoing_internal_ai_citations"].replace(0, pd.NA)
        * 100
    ).round(1)
    summary["cross_topic_ratio_pct"] = (
        summary["outgoing_cross_topic_citations"]
        / summary["outgoing_internal_ai_citations"].replace(0, pd.NA)
        * 100
    ).round(1)
    summary["external_kci_ratio_among_all_kci_refs_pct"] = (
        summary["outgoing_external_kci_citations"]
        / (
            summary["outgoing_external_kci_citations"]
            + summary["outgoing_internal_ai_citations"]
        ).replace(0, pd.NA)
        * 100
    ).round(1)
    summary["avg_outgoing_internal_ai_per_paper_2016_2024"] = (
        summary["outgoing_internal_ai_citations"]
        / summary["paper_count_2016_2024"].replace(0, pd.NA)
    ).round(2)

    return summary.sort_values(
        ["outgoing_internal_ai_citations", "paper_count_all"], ascending=False
    )


def build_matrices(topic_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = (
        topic_edges[["source_community", "source_label"]]
        .drop_duplicates()
        .rename(columns={"source_community": "community_id", "source_label": "label"})
    )
    target_labels = (
        topic_edges[["target_community", "target_label"]]
        .drop_duplicates()
        .rename(columns={"target_community": "community_id", "target_label": "label"})
    )
    labels = pd.concat([labels, target_labels]).drop_duplicates("community_id")
    labels = labels.sort_values("community_id")
    label_order = labels["label"].tolist()

    matrix = pd.crosstab(topic_edges["source_label"], topic_edges["target_label"])
    matrix = matrix.reindex(index=label_order, columns=label_order, fill_value=0)

    row_pct = matrix.div(matrix.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0) * 100
    row_pct = row_pct.round(1)

    top_cross = (
        topic_edges[~topic_edges["is_same_topic"]]
        .groupby(["source_label", "target_label"])
        .size()
        .reset_index(name="citation_count")
        .sort_values("citation_count", ascending=False)
    )
    source_total = matrix.sum(axis=1).rename("source_total_internal_ai_citations")
    top_cross = top_cross.merge(
        source_total,
        left_on="source_label",
        right_index=True,
        how="left",
    )
    top_cross["share_of_source_internal_ai_citations_pct"] = (
        top_cross["citation_count"]
        / top_cross["source_total_internal_ai_citations"].replace(0, pd.NA)
        * 100
    ).round(1)

    return matrix, row_pct, top_cross


def save_heatmap(row_pct: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(row_pct.values, cmap="Blues", vmin=0, vmax=max(40, row_pct.values.max()))
    ax.set_xticks(range(len(row_pct.columns)))
    ax.set_yticks(range(len(row_pct.index)))
    ax.set_xticklabels(row_pct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_pct.index, fontsize=8)
    ax.set_xlabel("피인용 논문의 주제군")
    ax.set_ylabel("인용 논문의 주제군")
    ax.set_title("주제군 간 내부 AI 법학 인용 비율(행 기준 %)")

    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            val = row_pct.iat[i, j]
            if val >= 8:
                color = "white" if val >= 25 else "#222222"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="행 기준 비율(%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "topic_citation_matrix_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_korean_font()

    papers, _paper_comm, edges = load_inputs()
    topic_edges = build_topic_edges(papers, edges)
    summary = build_summary(papers, edges, topic_edges)
    matrix, row_pct, top_cross = build_matrices(topic_edges)

    summary.to_csv(OUT_DIR / "topic_citation_summary.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(OUT_DIR / "topic_citation_matrix_counts.csv", encoding="utf-8-sig")
    row_pct.to_csv(OUT_DIR / "topic_citation_matrix_row_pct.csv", encoding="utf-8-sig")
    top_cross.to_csv(
        OUT_DIR / "topic_citation_top_cross_pairs.csv", index=False, encoding="utf-8-sig"
    )

    external_cols = [
        "community_id",
        "community_label",
        "paper_count_2016_2024",
        "outgoing_internal_ai_citations",
        "outgoing_external_kci_citations",
        "external_kci_ratio_among_all_kci_refs_pct",
    ]
    summary[external_cols].to_csv(
        OUT_DIR / "topic_external_kci_summary.csv", index=False, encoding="utf-8-sig"
    )
    save_heatmap(row_pct)

    print(f"[saved] {OUT_DIR}")
    print("\n상위 요약")
    print(
        summary[
            [
                "community_label",
                "paper_count_2016_2024",
                "outgoing_internal_ai_citations",
                "outgoing_same_topic_citations",
                "outgoing_cross_topic_citations",
                "same_topic_ratio_pct",
                "cross_topic_ratio_pct",
                "incoming_from_other_topic_citations",
            ]
        ]
        .head(14)
        .to_string(index=False)
    )

    print("\n주제군 간 교차 인용 상위 15개")
    print(top_cross.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
