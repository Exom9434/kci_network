"""
키워드 목록 + 카테고리 정보 결합 스크립트
==========================================
- 00.법률별_키워드_목록.csv  (법률명, 키워드 목록)
- 00.전체_법률_목록(2026.03.16).csv  (법령명, 카테고리, 카테고리 등)
→ 00.법률별_키워드_카테고리.csv 로 저장

사용법:
    python 02.merge_keywords_category.py
"""

import pandas as pd

KEYWORD_CSV = "00.법령_키워드_TFIDF.csv"
LAW_CSV     = "00.전체_법률_목록(2026.03.20)_기타제거.csv"
OUTPUT_CSV  = "00.법률별_키워드_카테고리_TFIDF.csv"

# 법률 목록에서 필요한 컬럼만 추출
LAW_COLS = [
    "법령명",
    "소관부처명",
    "카테고리",
]


def main():
    print("=" * 60)
    print("  키워드 목록 + 카테고리 결합")
    print("=" * 60)

    kw  = pd.read_csv(KEYWORD_CSV,  encoding="utf-8-sig")
    law = pd.read_csv(LAW_CSV,      encoding="utf-8-sig")

    print(f"[로드] 키워드 목록: {len(kw)}개 / 법률 목록: {len(law)}개")

    # 조인
    merged = kw.merge(
        law[LAW_COLS],
        left_on="법률명",
        right_on="법령명",
        how="left",
    ).drop(columns=["법령명"])  # 중복 컬럼 제거

    # 컬럼 순서 정리
    merged = merged[["법률명", "소관부처명", "카테고리", "키워드 목록"]]

    # 결과 요약
    total        = len(merged)
    has_final    = merged["카테고리"].notna().sum()
    no_final     = total - has_final
    has_keywords = (merged["키워드 목록"] != "").sum()

    print(f"\n[결과]")
    print(f"  전체 법률:          {total:>5}개")
    print(f"  카테고리 확정: {has_final:>5}개")
    print(f"  카테고리 미정: {no_final:>5}개 (수동 검토 필요)")
    print(f"  키워드 있음:        {has_keywords:>5}개")
    print(f"  키워드 없음:        {total - has_keywords:>5}개 (정의 조항 없는 법률)")

    print(f"\n[카테고리별 분포]")
    print(merged["카테고리"].value_counts().to_string())

    print(f"\n[샘플 출력]")
    sample = merged[merged["키워드 목록"] != ""].head(5)
    for _, row in sample.iterrows():
        kws = row["키워드 목록"][:40] + "..." if len(str(row["키워드 목록"])) > 40 else row["키워드 목록"]
        print(f"  [{row['카테고리']}] {row['법률명'][:20]} | {kws}")

    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {OUTPUT_CSV}")
    print("완료!")


if __name__ == "__main__":
    main()
