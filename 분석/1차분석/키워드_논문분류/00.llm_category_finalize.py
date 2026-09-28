"""
LLM 분류 결과 최종 집계 스크립트
===================================
- 카테고리_claude / 카테고리_gpt / 카테고리_gemini 중
  2개 이상 일치하는 경우 → 최종_카테고리에 기재
  일치하지 않는 경우 → 공란 (수동 검토 필요)

사용법:
    python 00.llm_category_finalize.py
"""

import pandas as pd
from collections import Counter

INPUT_CSV  = "00.전체_법률_목록(2026.03.16).csv"
OUTPUT_CSV = "00.전체_법률_목록(2026.03.16).csv"

COLS = ["카테고리_claude", "카테고리_gpt", "카테고리_gemini"]


def get_majority(row) -> str:
    """3개 값 중 2개 이상 일치하는 값 반환, 없으면 빈 문자열"""
    votes = [v for v in (row[c] for c in COLS)
             if pd.notna(v) and v not in ("오류", "분류불가", "")]
    if not votes:
        return ""
    most_common, count = Counter(votes).most_common(1)[0]
    return most_common if count >= 2 else ""


def main():
    print("=" * 60)
    print("  LLM 분류 결과 최종 집계")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[로드] {len(df)}개 법률 목록\n")

    df["최종_카테고리"] = df.apply(get_majority, axis=1)

    # 결과 요약
    total      = len(df)
    confirmed  = (df["최종_카테고리"] != "").sum()
    needs_review = total - confirmed

    print(f"  2개 이상 일치 (확정): {confirmed:>5}개 ({confirmed/total*100:.1f}%)")
    print(f"  불일치 (수동 검토):   {needs_review:>5}개 ({needs_review/total*100:.1f}%)")

    print("\n[최종_카테고리 분포]")
    print(df["최종_카테고리"].replace("", "(수동검토)").value_counts().to_string())

    if needs_review > 0:
        print(f"\n[수동 검토 필요 항목 샘플 (최대 10개)]")
        review_df = df[df["최종_카테고리"] == ""][
            ["법령명", "카테고리"] + COLS
        ].head(10)
        print(review_df.to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {OUTPUT_CSV}")
    print("완료!")


if __name__ == "__main__":
    main()
