"""
AI 법학 논문 기본정보 필터링 스크립트
======================================
전체 법학 논문 목록에서 AI 관련 논문(2016~2025)을 추출.

필터 조건:
    - 발행연도: 2016 ~ 2025
    - 제목 또는 초록에 다음 중 하나 이상 포함:
        · "AI"            (대소문자 일치)
        · "인공지능"
        · "Artificial Intelligence"

사용법:
    python 00.filter_ai_papers_v2.py
"""

import pandas as pd

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

INPUT_FILE  = "전체_법학_논문목록_중복제거.csv"
OUTPUT_FILE = "KCI_AI_논문_기본정보_목록_201601_202512.csv"

YEAR_START  = 2016
YEAR_END    = 2025

# 대소문자 구분이 필요한 항목은 regex=True + case=True 로 처리
# "AI"는 대소문자 일치(case=True), 나머지는 포함 여부만 확인
AI_TERMS = ["AI", "인공지능", "Artificial Intelligence"]


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print(f"[로딩] {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"  전체: {len(df):,}편")

    # 발행연도 숫자 변환
    df["발행연도"] = pd.to_numeric(df["발행연도"], errors="coerce")

    # ── 필터 1: 연도 ──────────────────────────
    year_mask = (df["발행연도"] >= YEAR_START) & (df["발행연도"] <= YEAR_END)

    # ── 필터 2: 키워드 ────────────────────────
    # "AI"는 대소문자 구분(case=True), 단어 경계 없이 포함 여부만 체크
    # 나머지는 대소문자 무관
    def make_keyword_mask(col: pd.Series) -> pd.Series:
        col = col.fillna("")
        mask = (
            col.str.contains("AI",                   regex=False, case=True)  |
            col.str.contains("인공지능",              regex=False, case=False) |
            col.str.contains("Artificial Intelligence", regex=False, case=False)
        )
        return mask

    keyword_mask = make_keyword_mask(df["제목"]) | make_keyword_mask(df["초록"])

    # ── 최종 필터 적용 ────────────────────────
    df_filtered = df[year_mask & keyword_mask].copy()

    # ── 결과 저장 ─────────────────────────────
    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ── 요약 출력 ─────────────────────────────
    print(f"\n{'='*45}")
    print(f"  연도 필터  ({YEAR_START}~{YEAR_END}):   {year_mask.sum():>6,}편")
    print(f"  키워드 필터 적용 후:       {len(df_filtered):>6,}편")
    print(f"{'='*45}")

    print(f"\n[연도별 분포]")
    print(df_filtered["발행연도"].value_counts().sort_index().to_string())

    print(f"\n[저장] {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
