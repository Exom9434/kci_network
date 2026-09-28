"""
수동 검토 대상 논문 추출 스크립트
=====================================
분류 결과에서 기초법(키워드없음) + 기초법(임계값미달) 논문을
수동 검토용 CSV로 추출.

출력 컬럼:
    논문ID, 제목, 발행연도, 학술지명, 분류결과, 매칭_키워드, 수동분류, 비고

정렬 순서: 분류결과(가나다) → 발행연도(최신순)

사용법:
    python 04.export_수동검토.py
"""

import pandas as pd

INPUT_CSV  = "03.논문_카테고리_분류결과.csv"
OUTPUT_CSV = "04.수동검토_기초법.csv"

REVIEW_LABELS = ["기초법(키워드없음)", "기초법(임계값미달)"]


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    수동검토 = df[df["분류결과"].isin(REVIEW_LABELS)].copy()
    수동검토 = 수동검토.sort_values(["분류결과", "발행연도"], ascending=[True, False])
    수동검토["수동분류"] = ""
    수동검토["비고"] = ""

    수동검토.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[완료] {OUTPUT_CSV} — 총 {len(수동검토)}편")
    print(수동검토["분류결과"].value_counts().to_string())


if __name__ == "__main__":
    main()
