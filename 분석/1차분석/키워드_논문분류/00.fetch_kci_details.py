"""
KCI 논문 상세정보 및 인용 데이터 수집 스크립트
================================================
KCI OpenAPI를 통해 논문별 키워드·초록·학술분류·인용 정보를 수집.

출력 컬럼:
    source_id, title, pub_year, abstract, keywords, category,
    target_arti_id, ref_type

사용법:
    python 00.fetch_kci_details.py
"""

import os
import re
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

load_dotenv()
API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

INPUT_FILE     = "KCI_AI_논문_기본정보_목록_201601_202512.csv"
OUTPUT_FILE    = "00.KCI_인공_논문_상세_및_인용데이터.csv"
SAVE_INTERVAL  = 50
YEAR_START     = 2016
YEAR_END       = 2025
KEYWORD_FILTER = r"인공지능|AI"


# ─────────────────────────────────────────────
# 🔧 API 호출 및 파싱
# ─────────────────────────────────────────────

def fetch_article_detail_and_refs(article_id: str, title: str, year) -> tuple[list, str]:
    """
    KCI API에서 논문 상세정보(키워드·초록·분류)와 인용 목록을 가져옴.

    Returns:
        (rows, status_message)
        rows: 인용 수만큼의 행 리스트 (인용 없으면 1행)
    """
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": article_id}
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        response.encoding = "utf-8"
        raw_xml = response.text

        if "<total>0</total>" in raw_xml:
            return [], "No Data"

        # ── 키워드 추출 ──────────────────────────────
        # 버그 수정: \]\]Task → \]\]>
        keyword_group = re.search(r"<keyword-group>(.*?)</keyword-group>", raw_xml, re.DOTALL)
        if keyword_group:
            kw_list = re.findall(
                r"<keyword[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</keyword>",
                keyword_group.group(1)
            )
            keywords_str = ", ".join([k.strip() for k in kw_list if k.strip()])
        else:
            keywords_str = ""

        # ── 초록 추출 (regex 방식으로 통일) ──────────
        # article_info.findall() 제거 → re.search 로 대체
        abstract_match = re.search(
            r'<abstract[^>]*lang="original"[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</abstract>',
            raw_xml, re.DOTALL
        )
        if not abstract_match:
            abstract_match = re.search(
                r"<abstract[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</abstract>",
                raw_xml, re.DOTALL
            )
        abstract_text = abstract_match.group(1).strip() if abstract_match else ""

        # ── 학술분류 추출 ────────────────────────────
        category_match = re.search(
            r"<article-categories[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</article-categories>",
            raw_xml, re.DOTALL
        )
        category = category_match.group(1).strip() if category_match else ""

        # ── 인용 정보 추출 ───────────────────────────
        ref_matches = re.findall(
            r"<reference([^>]*)>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</reference>",
            raw_xml, re.DOTALL
        )

        # 모든 행에 공통으로 들어갈 기본 데이터
        base_data = {
            "source_id": article_id,
            "title":     title,
            "pub_year":  year,
            "abstract":  abstract_text,
            "keywords":  keywords_str,
            "category":  category,
        }

        results = []
        if not ref_matches:
            row = base_data.copy()
            row.update({"target_arti_id": "", "ref_type": ""})
            results.append(row)
        else:
            for attrs, _ in ref_matches:
                arti_id_match   = re.search(r'arti-id="([^"]*)"',   attrs)
                type_name_match = re.search(r'type-name="([^"]*)"', attrs)
                row = base_data.copy()
                row.update({
                    "target_arti_id": arti_id_match.group(1)   if arti_id_match   else "",
                    "ref_type":       type_name_match.group(1) if type_name_match else "",
                })
                results.append(row)

        return results, "Success"

    except Exception as e:
        return [], str(e)


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    # 1. 이어하기 — 이미 수집된 ID 로드
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing   = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
            processed_ids = set(df_existing["source_id"].astype(str).unique())
            print(f"[이어하기] 기존 파일 로드: {len(processed_ids)}건 이미 수집됨")
        except Exception:
            pass

    # 2. 대상 로드 및 필터링
    print(f"[로딩] {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df["발행연도"] = pd.to_numeric(df["발행연도"], errors="coerce")

    year_filter    = (df["발행연도"] >= YEAR_START) & (df["발행연도"] <= YEAR_END)
    keyword_filter = df["제목"].str.contains(KEYWORD_FILTER, case=False, na=False)

    df_filtered   = df[year_filter & keyword_filter]
    df_to_process = df_filtered[
        ~df_filtered["논문ID"].astype(str).isin(processed_ids)
    ].copy()

    print(f"\n{'='*50}")
    print(f"  필터: '{KEYWORD_FILTER}' | {YEAR_START}~{YEAR_END}년")
    print(f"  조건 일치:       {len(df_filtered):>6}건")
    print(f"  이미 수집 제외:  {len(processed_ids):>6}건")
    print(f"  최종 수집 대상:  {len(df_to_process):>6}건")
    print(f"{'='*50}\n")

    # 3. 메인 수집 루프
    all_results = []
    try:
        for i, row in enumerate(
            tqdm(df_to_process.itertuples(index=False),
                 total=len(df_to_process), desc="수집 중", unit="건")
        ):
            aid   = str(row.논문ID)
            title = getattr(row, "제목", "")
            year  = getattr(row, "발행연도", "")

            details, status = fetch_article_detail_and_refs(aid, title, year)

            if details:
                all_results.extend(details)

            # 중간 저장
            if (i + 1) % SAVE_INTERVAL == 0:
                pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[중단] 수집 중단됨 — 지금까지 수집한 데이터 저장합니다.")

    # 4. 최종 저장
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"  [완료] {OUTPUT_FILE}")
    print(f"  수집 논문 수: {df_final['source_id'].nunique()}건")
    print(f"  총 행 수 (인용 포함): {len(df_final)}행")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
