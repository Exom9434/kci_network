"""
04.keyword_stats.py
--------------------
키워드 빈도 통계 두 가지 생성

출력:
  results/04_keyword_network/
    keyword_freq_distribution.csv  — 등장횟수별 키워드 갯수 (분포)
    keyword_freq_list.csv          — 키워드별 등장횟수 (전체, 빈도 내림차순)
"""

import importlib.util
import collections
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

# keyword_build의 전처리 함수 로드
spec = importlib.util.spec_from_file_location("kb", BASE / "04.keyword_build.py")
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)

OUT = BASE / "results" / "04_keyword_network"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # AI 논문 로드 및 키워드 전처리 (keyword_build와 동일)
    df = kb.load_ai_papers()
    kw_per_paper, freq = kb.build_kw_lists(df)

    print(f"전처리 후 유니크 키워드 수: {len(freq):,}")
    print(f"전체 키워드 토큰 수: {sum(freq.values()):,}")

    # ── 통계 1: 등장횟수별 키워드 갯수 (분포)
    dist = collections.Counter(freq.values())
    dist_df = pd.DataFrame(
        sorted(dist.items()), columns=["등장횟수(논문수)", "키워드_갯수"]
    )
    dist_df["누적_키워드_갯수"] = dist_df["키워드_갯수"].cumsum()
    dist_df["비율(%)"] = (dist_df["키워드_갯수"] / len(freq) * 100).round(2)
    dist_df.to_csv(OUT / "keyword_freq_distribution.csv",
                   index=False, encoding="utf-8-sig")
    print("\n── 등장횟수별 키워드 갯수 (상위 20개)")
    print(dist_df.head(20).to_string(index=False))
    print(f"\n  ...총 {len(dist_df)}개 등장횟수 구간")
    print(f"[saved] keyword_freq_distribution.csv")

    # ── 통계 2: 키워드별 등장횟수 (전체)
    kw_df = pd.DataFrame(freq.most_common(), columns=["키워드", "등장횟수(논문수)"])
    kw_df.insert(0, "순위", range(1, len(kw_df) + 1))
    kw_df.to_csv(OUT / "keyword_freq_list.csv",
                 index=False, encoding="utf-8-sig")
    print("\n── 키워드별 등장횟수 (상위 30개)")
    print(kw_df.head(30).to_string(index=False))
    print(f"[saved] keyword_freq_list.csv")

    print(f"\n✓ 완료: {OUT}")


if __name__ == "__main__":
    main()
