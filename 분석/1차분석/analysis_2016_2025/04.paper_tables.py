"""
04.paper_tables.py
==================
논문 4장(분석 결과)용 표 생성 스크립트

출력 폴더:
  results/paper_tables/

생성 표:
  table_01_annual_counts.csv
  table_02_category_distribution.csv
  table_03_keyword_emergence.csv
  table_04_keyword_network_stats.csv
  table_05_community_features.csv
  table_06_centrality_top10.csv
  paper_tables.xlsx

실행:
  python 04.paper_tables.py
"""

from __future__ import annotations

from pathlib import Path
import re

import networkx as nx
import pandas as pd


BASE = Path(__file__).parent
OUT = BASE / "results" / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

FINAL_CATEGORY_FILES = [
    BASE / "KCI_AI_논문_카테고리_확정.csv",
    BASE / "KCI_AI_논문_카테고리_미확정_수동검토완료.csv",
]
RAW_CATEGORY = BASE / "KCI_AI_논문_카테고리_분류결과.csv"
TOP_KEYWORDS = BASE / "results" / "04_keyword_network" / "top_keywords.csv"
KEYWORD_TREND = BASE / "results" / "05_keyword_trend" / "keyword_trend_data.csv"
KEYWORD_GRAPH = BASE / "results" / "04_keyword_network" / "keyword_cooccurrence.graphml"
KEYWORD_GIANT = BASE / "results" / "04_keyword_network" / "keyword_giant.graphml"
KEYWORD_COMMUNITIES = BASE / "results" / "04_keyword_network" / "keyword_communities.csv"
COMMUNITY_PAPER_MAP = BASE / "results" / "04_keyword_network" / "community_paper_map.csv"
CENTRALITY = BASE / "results" / "04_keyword_network" / "keyword_centrality.csv"
DETAIL = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"

YEARS = list(range(2016, 2026))

TYPO_MAP = {
    "기럴테크": "리걸테크",
    "사회보자장법": "사회보장법",
    "인공지능자동차법": "자율주행자동차법",
}


COMMUNITY_INTERPRETATION = {
    "생성형 인공지능 / 저작권 / 공정이용": "생성형 AI 산출물, 학습데이터, 저작권 침해와 공정이용 쟁점",
    "빅데이터 / 개인정보 / 개인정보보호": "빅데이터 활용, 개인정보 보호, GDPR 및 데이터 보호 법제 쟁점",
    "4차 산업혁명 / 블록체인 / 지능정보사회": "초기 AI 법학 담론을 이끈 포괄적 기술·사회 변화 담론",
    "알고리즘 / 머신러닝 / 딥러닝": "알고리즘 의사결정, 투명성, 공정성, 설명가능성 쟁점",
    "제조물책임 / 위험책임 / 소프트웨어": "AI 시스템 결함, 위험책임, 의료 AI와 불법행위책임 쟁점",
    "자율주행자동차 / 규제 / 책임": "자율주행차 사고 책임, 교통규제, 제조물책임법 쟁점",
    "EU AI Act / 인공지능기본법 / 인공지능법": "AI 기본법, 고위험 AI, 국내외 인공지능 기본법제 쟁점",
    "법인격 / 로봇 / 형사책임": "AI·로봇의 법인격, 권리능력, 형사책임 귀속 쟁점",
    "딥페이크 / 표현의 자유 / 인격권": "딥페이크, 허위조작정보, 표현의 자유와 인격권 충돌",
    "발명 / 발명자 / 특허법": "AI 발명, 발명자성, 특허법상 보호 가능성",
    "자율운항선박 / Maritime Autonomous Surface Ships / 원격운항자": "자율운항선박과 원격운항자의 책임 및 해사법 쟁점",
    "부정경쟁방지법 / Unfair Competition Prevention Act": "AI 생성물·데이터 활용과 부정경쟁방지법 쟁점",
    "리걸테크 / 변호사법": "법률 AI 서비스, 리걸테크와 변호사법 규제 쟁점",
    "중국 / China": "중국 AI 법제 및 해외법 비교 연구",
}


def load_final_categories() -> pd.DataFrame:
    frames = []
    for path in FINAL_CATEGORY_FILES:
        df = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(df[["논문ID", "발행연도", "최종_카테고리"]])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("논문ID")
    out["논문ID"] = out["논문ID"].astype(str)
    out["최종_카테고리"] = out["최종_카테고리"].fillna("").astype(str).map(normalize_category)
    return out


def normalize_category(label: str) -> str:
    label = str(label).strip()
    for wrong, right in TYPO_MAP.items():
        label = label.replace(wrong, right)
    if label.startswith("융복합(") and not label.endswith(")"):
        label = label + ")"
    if label.startswith("융복합("):
        inner = label.replace("융복합(", "").rstrip(")")
        parts = [part.strip() for part in inner.split("+") if part.strip()]
        if len(parts) == 2:
            return f"융복합({'+'.join(sorted(parts))})"
    return label


def normalize_fusion_label(label: str) -> str:
    if not label.startswith("융복합("):
        return label
    return label.replace("융복합(", "").rstrip(")")


def table_annual_counts(raw_df: pd.DataFrame) -> pd.DataFrame:
    annual = (
        raw_df.drop_duplicates("논문ID")
        .groupby("발행연도")
        .size()
        .reindex(YEARS, fill_value=0)
    )
    total = int(annual.sum())
    rows = []
    prev = None
    for year, count in annual.items():
        yoy = None if prev in (None, 0) else (count - prev) / prev * 100
        rows.append(
            {
                "연도": year,
                "논문 수": int(count),
                "전체 대비 비율(%)": round(count / total * 100, 1),
                "전년 대비 증감률(%)": "" if yoy is None else round(yoy, 1),
            }
        )
        prev = count
    rows.append({"연도": "합계", "논문 수": total, "전체 대비 비율(%)": 100.0, "전년 대비 증감률(%)": ""})
    return pd.DataFrame(rows)


def table_category_distribution(final_df: pd.DataFrame) -> pd.DataFrame:
    df = final_df.copy()
    df["분류 유형"] = df["최종_카테고리"].where(
        ~df["최종_카테고리"].str.startswith("융복합"),
        "융복합",
    )
    total = len(df)

    summary = (
        df["분류 유형"]
        .value_counts()
        .rename_axis("법 분야")
        .reset_index(name="논문 수")
    )
    summary["구분"] = summary["법 분야"].where(summary["법 분야"].eq("융복합"), "단일 분야")

    detail = (
        df["최종_카테고리"]
        .value_counts()
        .rename_axis("법 분야")
        .reset_index(name="논문 수")
    )
    detail["구분"] = detail["법 분야"].apply(lambda x: "융복합 세부" if str(x).startswith("융복합") else "단일 분야")
    detail["법 분야"] = detail["법 분야"].apply(normalize_fusion_label)

    combined = pd.concat([summary, detail], ignore_index=True)
    combined["비율(%)"] = (combined["논문 수"] / total * 100).round(1)
    return combined[["구분", "법 분야", "논문 수", "비율(%)"]]


def table_keyword_emergence(top_n: int = 20, emergence_min: int = 3) -> pd.DataFrame:
    top = pd.read_csv(TOP_KEYWORDS, encoding="utf-8-sig").head(top_n)
    trend = pd.read_csv(KEYWORD_TREND, encoding="utf-8-sig")

    emergence = {}
    peak = {}
    for keyword, sub in trend.groupby("keyword"):
        sub = sub.sort_values("year")
        met = sub[sub["count"] >= emergence_min]
        emergence[keyword] = "" if met.empty else int(met.iloc[0]["year"])
        peak_row = sub.loc[sub["count"].idxmax()]
        peak[keyword] = f"{int(peak_row['year'])}년 {int(peak_row['count'])}편"

    rows = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        keyword = row.keyword
        rows.append(
            {
                "순위": rank,
                "키워드": keyword,
                "누적 출현 논문 수": int(row.paper_count),
                f"본격 등장 연도(연 {emergence_min}편 이상)": emergence.get(keyword, ""),
                "최고 출현 연도": peak.get(keyword, ""),
            }
        )
    return pd.DataFrame(rows)


def table_keyword_network_stats() -> pd.DataFrame:
    detail = pd.read_csv(DETAIL, encoding="utf-8-sig", usecols=["source_id", "keywords"])
    papers = detail.drop_duplicates("source_id")
    graph = nx.read_graphml(KEYWORD_GRAPH)
    giant = nx.read_graphml(KEYWORD_GIANT)
    communities = pd.read_csv(KEYWORD_COMMUNITIES, encoding="utf-8-sig")
    paper_map = pd.read_csv(COMMUNITY_PAPER_MAP, encoding="utf-8-sig")

    components = list(nx.connected_components(graph))
    degrees = [d for _, d in graph.degree()]
    rows = [
        ("분석 대상 논문 수", len(papers), "편"),
        ("키워드 보유 논문 수", int(papers["keywords"].notna().sum()), "편"),
        ("전처리 후 유니크 키워드 수", 14195, "개"),
        ("필터 후 네트워크 노드 수", graph.number_of_nodes(), "개"),
        ("필터 후 네트워크 엣지 수", graph.number_of_edges(), "개"),
        ("평균 연결 정도", round(sum(degrees) / len(degrees), 2), ""),
        ("연결 성분 수", len(components), "개"),
        ("거대 컴포넌트 노드 수", giant.number_of_nodes(), "개"),
        ("고립 노드 수", sum(1 for d in degrees if d == 0), "개"),
        ("Leiden 커뮤니티 수", communities["community_id"].nunique(), "개"),
        ("논문-커뮤니티 매핑 수", len(paper_map), "편"),
    ]
    return pd.DataFrame(rows, columns=["항목", "값", "단위"])


def table_community_features() -> pd.DataFrame:
    communities = pd.read_csv(KEYWORD_COMMUNITIES, encoding="utf-8-sig")
    paper_map = pd.read_csv(COMMUNITY_PAPER_MAP, encoding="utf-8-sig")

    paper_counts = paper_map["community_label"].value_counts()
    rows = []
    grouped = communities.groupby(["community_id", "community_label"], sort=True)
    for (cid, label), sub in grouped:
        sub = sub.sort_values("freq", ascending=False)
        top_keywords = ", ".join(sub["keyword"].head(8).tolist())
        rows.append(
            {
                "커뮤니티": f"C{int(cid)}",
                "커뮤니티명": label,
                "키워드 수": len(sub),
                "매핑 논문 수": int(paper_counts.get(label, 0)),
                "대표 키워드": top_keywords,
                "법학적 해석": COMMUNITY_INTERPRETATION.get(label, ""),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("매핑 논문 수", ascending=False).reset_index(drop=True)


def table_centrality_top10() -> pd.DataFrame:
    df = pd.read_csv(CENTRALITY, encoding="utf-8-sig")
    top = df.sort_values("betweenness", ascending=False).head(10).copy()
    top.insert(0, "순위", range(1, len(top) + 1))
    top = top[
        [
            "순위",
            "keyword",
            "freq",
            "degree",
            "weighted_degree",
            "betweenness",
            "pagerank",
            "community_label",
        ]
    ].rename(
        columns={
            "keyword": "키워드",
            "freq": "출현 논문 수",
            "degree": "Degree",
            "weighted_degree": "Weighted degree",
            "betweenness": "Betweenness",
            "pagerank": "PageRank",
            "community_label": "커뮤니티",
        }
    )
    top["Betweenness"] = top["Betweenness"].round(4)
    top["PageRank"] = top["PageRank"].round(4)
    return top


def autosize_excel_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns, start=1):
        values = [str(col)] + [str(v) for v in df[col].head(200).tolist()]
        width = min(max(len(v) for v in values) + 2, 60)
        worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = width


def main() -> None:
    raw_df = pd.read_csv(RAW_CATEGORY, encoding="utf-8-sig")
    final_df = load_final_categories()

    tables = {
        "01_연도별_논문수": table_annual_counts(raw_df),
        "02_법분야_분포": table_category_distribution(final_df),
        "03_주요키워드": table_keyword_emergence(),
        "04_네트워크_기초통계": table_keyword_network_stats(),
        "05_커뮤니티_특징": table_community_features(),
        "06_중심성_Top10": table_centrality_top10(),
    }

    filenames = {
        "01_연도별_논문수": "table_01_annual_counts.csv",
        "02_법분야_분포": "table_02_category_distribution.csv",
        "03_주요키워드": "table_03_keyword_emergence.csv",
        "04_네트워크_기초통계": "table_04_keyword_network_stats.csv",
        "05_커뮤니티_특징": "table_05_community_features.csv",
        "06_중심성_Top10": "table_06_centrality_top10.csv",
    }

    for key, df in tables.items():
        path = OUT / filenames[key]
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[저장] {path}")

    xlsx_path = OUT / "paper_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            safe_sheet = re.sub(r"[\[\]\:\*\?\/\\]", "_", sheet)[:31]
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
            autosize_excel_columns(writer, safe_sheet, df)
    print(f"[저장] {xlsx_path}")


if __name__ == "__main__":
    main()
