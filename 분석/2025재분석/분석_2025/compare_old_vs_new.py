"""
compare_old_vs_new.py
=====================
원고(옛, 2025 미반영) 결과 vs 재분석(새, 2025 반영) 결과를 나란히 비교.
원고에서 어느 수치를 고쳐야 하는지 바로 잡기 위한 대조표.

입력: 옛 = ../../results/...   새 = ./results/...
출력: ./results/comparison/{community_features,citation_flow_metrics,annual_counts,network_stats}_diff.csv
실행: (분석_2025 폴더) python compare_old_vs_new.py
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OLD = HERE / ".." / ".." / "1차분석" / "results"
NEW = HERE / "results"
OUT = NEW / "comparison"; OUT.mkdir(parents=True, exist_ok=True)

def load(p):
    return pd.read_csv(p, encoding="utf-8-sig")

def show(title, df):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(df.to_string(index=False))

# ── 1) 주제군 논문 수 (table_05) — 커뮤니티명(키워드라벨)으로 대응 ──
def cmp_community():
    o = load(OLD/"paper_tables"/"table_05_community_features.csv")[["커뮤니티명","매핑 논문 수"]]
    n = load(NEW/"paper_tables"/"table_05_community_features.csv")[["커뮤니티명","매핑 논문 수"]]
    m = o.merge(n, on="커뮤니티명", how="outer", suffixes=("_옛","_새"))
    m["증감"] = m["매핑 논문 수_새"].fillna(0) - m["매핑 논문 수_옛"].fillna(0)
    m = m.sort_values("매핑 논문 수_새", ascending=False)
    m.to_csv(OUT/"community_features_diff.csv", index=False, encoding="utf-8-sig")
    show("① 주제군별 매핑 논문 수 (옛 vs 새)", m)
    print(f"  합계: 옛 {int(m['매핑 논문 수_옛'].sum())} → 새 {int(m['매핑 논문 수_새'].sum())}")

# ── 2) 인용 흐름 지표 (07_citation_flow_metrics) — 연도별 ──
def cmp_citation():
    cols = ["year","internal_citations","internal_ratio_pct",
            "avg_internal_citations_per_paper","papers_with_internal_citation_pct"]
    o = load(OLD/"03_citation"/"07_citation_flow_metrics.csv")[cols]
    n = load(NEW/"03_citation"/"07_citation_flow_metrics.csv")[cols]
    m = o.merge(n, on="year", how="outer", suffixes=("_옛","_새")).sort_values("year")
    m.to_csv(OUT/"citation_flow_metrics_diff.csv", index=False, encoding="utf-8-sig")
    # 보기 좋게 핵심만
    view = m[["year","internal_ratio_pct_옛","internal_ratio_pct_새",
              "avg_internal_citations_per_paper_옛","avg_internal_citations_per_paper_새"]]
    show("② 내부 인용비율 · 논문당 내부인용수 (옛 vs 새) — 2025는 옛 분석엔 없던 행", view)

# ── 3) 연도별 논문 수 (table_01) — 모집단 동일 여부 확인 ──
def cmp_annual():
    o = load(OLD/"paper_tables"/"table_01_annual_counts.csv")[["연도","논문 수"]]
    n = load(NEW/"paper_tables"/"table_01_annual_counts.csv")[["연도","논문 수"]]
    m = o.merge(n, on="연도", how="outer", suffixes=("_옛","_새"))
    m["동일"] = m["논문 수_옛"] == m["논문 수_새"]
    m.to_csv(OUT/"annual_counts_diff.csv", index=False, encoding="utf-8-sig")
    show("③ 연도별 논문 수 (모집단 동일해야 정상)", m)

# ── 4) 키워드 네트워크 통계 (table_04) ──
def cmp_netstats():
    try:
        o = load(OLD/"paper_tables"/"table_04_keyword_network_stats.csv")
        n = load(NEW/"paper_tables"/"table_04_keyword_network_stats.csv")
        key = o.columns[0]
        m = o.merge(n, on=key, how="outer", suffixes=("_옛","_새"))
        m.to_csv(OUT/"network_stats_diff.csv", index=False, encoding="utf-8-sig")
        show("④ 키워드 네트워크 기초 통계 (옛 vs 새)", m)
    except Exception as e:
        print("  (네트워크 통계 비교 건너뜀:", e, ")")

# ── 5) 주제군별 인용 요약 (08 topic_citation_summary) — RQ4 핵심(피인용) ──
def cmp_topic_flow():
    o = load(OLD/"03_citation_topic_flow"/"topic_citation_summary.csv")
    n = load(NEW/"03_citation_topic_flow"/"topic_citation_summary.csv")
    o = o.rename(columns=lambda c: c.replace("_2016_2024", "_period"))
    n = n.rename(columns=lambda c: c.replace("_2016_2025", "_period"))
    keep = ["community_label", "incoming_internal_ai_citations",
            "outgoing_internal_ai_citations", "same_topic_ratio_pct", "cross_topic_ratio_pct"]
    m = o[keep].merge(n[keep], on="community_label", how="outer", suffixes=("_옛", "_새"))
    m["피인용_증감"] = (m["incoming_internal_ai_citations_새"].fillna(0)
                     - m["incoming_internal_ai_citations_옛"].fillna(0)).astype(int)
    m = m.sort_values("incoming_internal_ai_citations_새", ascending=False)
    m.to_csv(OUT/"topic_flow_diff.csv", index=False, encoding="utf-8-sig")
    view = m[["community_label", "incoming_internal_ai_citations_옛",
              "incoming_internal_ai_citations_새", "피인용_증감",
              "same_topic_ratio_pct_옛", "same_topic_ratio_pct_새"]]
    view.columns = ["주제군", "피인용_옛", "피인용_새", "증감", "동일주제%_옛", "동일주제%_새"]
    show("⑤ 주제군별 피인용(들어오는 인용) + 동일주제 비율 (옛 vs 새)", view)


if __name__ == "__main__":
    cmp_community()
    cmp_citation()
    cmp_annual()
    cmp_netstats()
    cmp_topic_flow()
    print(f"\n저장: {OUT}/*_diff.csv")
