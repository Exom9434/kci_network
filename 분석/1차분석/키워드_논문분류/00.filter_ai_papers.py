"""
AI 법학 논문 필터링 스크립트
=====================================
전체 법학 논문 목록에서 인공지능 관련 논문만 추출

[필터링 조건]
- 제목 또는 초록에 아래 키워드 중 하나 이상 포함:
  1. '인공지능'
  2. 'AI'  (단어 경계 기준, 대소문자 정확히 일치)
  3. 'Artificial Intelligence'

사용법:
    python 00.filter_ai_papers.py
"""

import pandas as pd

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

INPUT_CSV  = "전체_법학_논문목록_중복제거.csv"
OUTPUT_CSV = "KCI_AI_논문_기본정보_목록_201601_202512.csv"


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  AI 법학 논문 필터링")
    print("=" * 55)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[로드] {INPUT_CSV}: {len(df)}편")

    title    = df["제목"].fillna("")
    abstract = df["초록"].fillna("") if "초록" in df.columns else pd.Series([""] * len(df))

    keyword_mask = (
        title.str.contains("인공지능", na=False)
        | abstract.str.contains("인공지능", na=False)
        | title.str.contains(r"\bAI", regex=True, na=False)
        | abstract.str.contains(r"\bAI", regex=True, na=False)
        | title.str.contains("Artificial Intelligence", regex=False, na=False)
        | abstract.str.contains("Artificial Intelligence", regex=False, na=False)
    )

    df["발행연도"] = pd.to_numeric(df["발행연도"], errors="coerce")
    year_mask = (df["발행연도"] >= 2016) & (df["발행연도"] <= 2025)

    df_ai = df[keyword_mask & year_mask].copy()

    print(f"[필터링] AI 관련 논문: {len(df_ai)}편 ({len(df_ai)/len(df)*100:.1f}%)")
    print()
    print("[연도별 분포]")
    vc = df_ai["발행연도"].value_counts().sort_index()
    for year, cnt in vc.items():
        print(f"  {year}년  {cnt:>4}편")

    df_ai.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {OUTPUT_CSV}")
    print("완료!")


if __name__ == "__main__":
    main()
