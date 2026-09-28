"""
법률 정의 조항 키워드 추출 스크립트
=====================================
- "법령 csv 저장(26.03.16 기준) 중복제거" 폴더의 전체 법률 CSV 처리
- 조문제목 == "정의"인 행의 항/호 컬럼에서 쌍따옴표("") 안 텍스트 추출
- 결과를 00.법률별_키워드_목록.csv로 저장

사용법:
    python 01.extract_definition_keywords.py
"""

import re
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

LAW_DIR    = "법령 csv 저장(26.03.16 기준) 중복제거"
OUTPUT_CSV = "00.법률별_키워드_목록.csv"

# 쌍따옴표 패턴 (일반 " 와 전각 " " 모두 처리)
QUOTE_PATTERN = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]')


def extract_keywords(text: str) -> list[str]:
    """문자열에서 쌍따옴표 안 키워드 추출"""
    if not isinstance(text, str):
        return []
    return QUOTE_PATTERN.findall(text)


def process_law_file(filepath: Path) -> tuple[str, list[str]]:
    """법률 CSV 파일 하나에서 정의 조항 키워드 추출"""
    law_name = filepath.stem  # 파일명 = 법률명

    try:
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip', dtype=str)
    except Exception:
        return law_name, []

    if '조문제목' not in df.columns:
        return law_name, []

    # 정의 조항 필터링
    정의_df = df[df['조문제목'].str.strip() == '정의']
    if 정의_df.empty:
        return law_name, []

    # 항 + 호 컬럼에서 키워드 추출
    keywords = []
    for col in ['항', '호']:
        if col in 정의_df.columns:
            for text in 정의_df[col].dropna():
                keywords.extend(extract_keywords(text))

    # 중복 제거, 순서 유지
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw = kw.strip()
        if kw and kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return law_name, unique_keywords


def main():
    print("=" * 60)
    print("  법률 정의 조항 키워드 추출")
    print("=" * 60)

    csv_files = sorted(Path(LAW_DIR).glob("*.csv"))
    print(f"[대상] {len(csv_files)}개 법률 파일\n")

    results = []
    no_definition = 0

    for filepath in tqdm(csv_files, desc="처리 중", unit="개"):
        law_name, keywords = process_law_file(filepath)

        if not keywords:
            no_definition += 1

        results.append({
            "법률명":    law_name,
            "키워드 목록": ", ".join(keywords),
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 결과 요약
    has_keywords = df_out[df_out["키워드 목록"] != ""]
    print(f"\n[완료]")
    print(f"  키워드 추출 성공: {len(has_keywords)}개")
    print(f"  정의 조항 없음:   {no_definition}개")
    print(f"\n[샘플 출력]")
    print(has_keywords[["법률명", "키워드 목록"]].head(5).to_string(index=False))
    print(f"\n[저장] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
