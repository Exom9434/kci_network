"""
10.hub_check.py
===============
"어떤 주제군이 허브인가"를 판별하는 지표표를 만든다.

문제의식
--------
피인용 총량만 보면 생성형 인공지능·저작권 주제군이 847건으로 법인격(912건) 다음이다.
그러나 피인용 총량에는 주제군 내부에서 순환한 자기인용이 섞여 있다.
허브인지 아닌지는 **다른 주제군에서 들어온 인용(교차 유입)** 으로 판별해야 하고,
주제군 규모 차이를 없애려면 **논문 한 편당 교차 유입**으로 봐야 한다.

이 스크립트는 그 지표를 계산해 표로 만들고, 상위 후보 주제군의 유입 내역을 뜯어본다.

위치
----
`2025인용_재수집/분석_2025/10.hub_check.py` 로 두고 실행한다.

입력
----
  results/03_citation_topic_flow/topic_citation_summary.csv
  results/03_citation_topic_flow/topic_citation_matrix_row_pct.csv

출력
----
  results/03_citation_topic_flow/허브판별_지표.csv
  results/03_citation_topic_flow/허브판별_유입내역.csv
  (표는 화면에도 출력)

실행
----
  python 10.hub_check.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DIR = BASE / "results" / "03_citation_topic_flow"
SUMMARY = DIR / "topic_citation_summary.csv"
ROW_PCT = DIR / "topic_citation_matrix_row_pct.csv"

# 유입 내역을 뜯어볼 주제군
FOCUS = ["법인격", "알고리즘", "생성형 인공지능"]


def short(label: str) -> str:
    return str(label).split(" / ")[0]


def build_metrics(s: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(
        {
            "주제군": s.community_label.map(short),
            "논문수": s.paper_count_all,
            "피인용총량": s.incoming_internal_ai_citations,
            "자기인용_유입": s.incoming_internal_ai_citations
            - s.incoming_from_other_topic_citations,
            "교차유입": s.incoming_from_other_topic_citations,
            "유출총량": s.outgoing_internal_ai_citations,
            "자기인용비율%": s.same_topic_ratio_pct,
        }
    )
    m["논문당_피인용"] = (m.피인용총량 / m.논문수).round(2)
    m["논문당_교차유입"] = (m.교차유입 / m.논문수).round(2)
    m["교차유입_비중%"] = (m.교차유입 / m.피인용총량 * 100).round(1)
    return m.sort_values("논문당_교차유입", ascending=False).reset_index(drop=True)


def build_inflow(s: pd.DataFrame, pct: pd.DataFrame) -> pd.DataFrame:
    """행 기준 % × 행 유출총량 → 셀 건수 복원."""
    out = s.set_index("community_label").outgoing_internal_ai_citations.reindex(pct.index)
    cnt = pct.div(100).mul(out, axis=0).round(0).astype(int)
    cnt.index = [short(i) for i in cnt.index]
    cnt.columns = [short(c) for c in cnt.columns]

    rows = []
    for col in cnt.columns:
        d = cnt[col].drop(col)
        for src, v in d[d > 0].sort_values(ascending=False).items():
            rows.append({"피인용_주제군": col, "인용한_주제군": src, "건수": int(v)})
    return pd.DataFrame(rows)


def main() -> None:
    for p in (SUMMARY, ROW_PCT):
        if not p.exists():
            raise FileNotFoundError(f"{p} 없음 — 08.topic_citation_flow.py 를 먼저 실행할 것.")

    s = pd.read_csv(SUMMARY, encoding="utf-8-sig")
    pct = pd.read_csv(ROW_PCT, index_col=0, encoding="utf-8-sig")

    metrics = build_metrics(s)
    inflow = build_inflow(s, pct)

    metrics.to_csv(DIR / "허브판별_지표.csv", index=False, encoding="utf-8-sig")
    inflow.to_csv(DIR / "허브판별_유입내역.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)

    print("■ 허브 판별 지표 (논문당 교차 유입 내림차순)")
    print("  = 다른 주제군에서 들어온 인용을 그 주제군의 논문 수로 나눈 값")
    print()
    print(
        metrics[
            [
                "주제군",
                "논문수",
                "피인용총량",
                "자기인용_유입",
                "교차유입",
                "논문당_피인용",
                "논문당_교차유입",
                "교차유입_비중%",
                "자기인용비율%",
            ]
        ].to_string(index=False)
    )

    print()
    print("■ 유입 내역 (교차 인용, 건수)")
    for f in FOCUS:
        sub = inflow[inflow.피인용_주제군 == f]
        if sub.empty:
            print(f"\n  [{f}] 해당 주제군 없음 — FOCUS 목록을 확인할 것")
            continue
        tot = int(sub.건수.sum())
        print(f"\n  [{f}] 교차 유입 합계 {tot}건")
        for _, r in sub.head(8).iterrows():
            print(f"     {r.인용한_주제군:<14s} {r.건수:>4d}건")

    print()
    print(f"→ {DIR / '허브판별_지표.csv'}")
    print(f"→ {DIR / '허브판별_유입내역.csv'}")


if __name__ == "__main__":
    main()
