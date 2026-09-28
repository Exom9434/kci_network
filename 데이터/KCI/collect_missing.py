"""
누락된 989개 논문 상세 데이터 수집 스크립트
- KCI_AI_논문_기본정보_목록_201601_202512.csv (2014개) 기준
- 00.KCI_AI_논문_상세_및_인용데이터.csv 에 없는 논문만 수집
- 수집 완료 후 기존 파일에 append
"""

import os
import requests
import pandas as pd
import re
import time
from tqdm import tqdm
from dotenv import load_dotenv

# ── 설정 ────────────────────────────────────────────────────────
load_dotenv(__import__("pathlib").Path(__file__).resolve().parents[2] / ".env")
API_KEY = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

INPUT_FILE  = 'KCI_AI_논문_기본정보_목록_201601_202512.csv'   # 2014개 기준
OUTPUT_FILE = '00.KCI_AI_논문_상세_및_인용데이터.csv'          # 기존 파일 (append)
SAVE_INTERVAL = 500   # 행 기준 N개마다 중간 저장
SLEEP_SEC     = 0.7   # API 요청 간격(초)

# ── 1. 누락 논문 ID 추출 ─────────────────────────────────────────
df_basic  = pd.read_csv(INPUT_FILE)
df_exist  = pd.read_csv(OUTPUT_FILE)

basic_ids   = set(df_basic['논문ID'].astype(str))
exist_ids   = set(df_exist['source_id'].astype(str))
missing_ids = basic_ids - exist_ids

print(f"기본정보 전체  : {len(basic_ids):,}개")
print(f"기수집 완료    : {len(exist_ids & basic_ids):,}개")
print(f"수집 대상(누락): {len(missing_ids):,}개")
print()

df_to_collect = df_basic[df_basic['논문ID'].astype(str).isin(missing_ids)].copy()
print("연도별 누락 현황:")
print(df_to_collect['발행연도'].value_counts().sort_index().to_string())
print()

# ── 2. API 수집 함수 ─────────────────────────────────────────────
def fetch_detail_and_refs(article_id, title, year):
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": article_id}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        resp.encoding = 'utf-8'
        xml = resp.text

        if "<total>0</total>" in xml:
            return [], "NoData"

        # 키워드
        kw_group = re.search(r'<keyword-group>(.*?)</keyword-group>', xml, re.DOTALL)
        if kw_group:
            kws = re.findall(r'<keyword[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</keyword>',
                             kw_group.group(1))
            keywords = ", ".join(k.strip() for k in kws if k.strip())
        else:
            keywords = ""

        # 초록
        ab_match = re.search(
            r'<abstract[^>]*lang=["\']original["\'][^>]*>(.*?)</abstract>', xml, re.DOTALL)
        if not ab_match:
            ab_match = re.search(r'<abstract[^>]*>(.*?)</abstract>', xml, re.DOTALL)
        abstract = re.sub(r'<[^>]+>', '', ab_match.group(1)).strip() if ab_match else ""

        # 카테고리
        cat_match = re.search(r'<article-categories[^>]*>(.*?)</article-categories>', xml, re.DOTALL)
        category  = re.sub(r'<[^>]+>', '', cat_match.group(1)).strip() if cat_match else ""

        base = {
            "source_id": article_id,
            "title"    : title,
            "pub_year" : year,
            "abstract" : abstract,
            "keywords" : keywords,
            "category" : category,
        }

        # 인용
        ref_matches = re.findall(
            r'<reference([^>]*)>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</reference>', xml, re.DOTALL)

        if not ref_matches:
            row = base.copy()
            row.update({"target_arti_id": "", "ref_type": ""})
            return [row], "OK_NoRef"

        rows = []
        for attrs, _ in ref_matches:
            arti_id  = re.search(r'arti-id="([^"]*)"', attrs)
            ref_type = re.search(r'type-name="([^"]*)"', attrs)
            row = base.copy()
            row.update({
                "target_arti_id": arti_id.group(1)  if arti_id  else "",
                "ref_type"      : ref_type.group(1) if ref_type else "",
            })
            rows.append(row)
        return rows, "OK"

    except Exception as e:
        return [], str(e)


# ── 3. 수집 루프 ─────────────────────────────────────────────────
buffer   = []
ok_cnt   = 0
fail_cnt = 0

print(f"🚀 수집 시작 (간격 {SLEEP_SEC}초, {SAVE_INTERVAL}행마다 중간저장)\n")

for row in tqdm(df_to_collect.itertuples(index=False), total=len(df_to_collect)):
    aid   = str(row.논문ID)
    title = row.제목
    year  = row.발행연도

    rows, status = fetch_detail_and_refs(aid, title, year)

    if rows:
        buffer.extend(rows)
        ok_cnt += 1
    else:
        fail_cnt += 1
        tqdm.write(f"  ⚠ {aid} : {status}")

    if len(buffer) >= SAVE_INTERVAL:
        pd.DataFrame(buffer).to_csv(
            OUTPUT_FILE, mode='a', index=False, header=False, encoding='utf-8-sig')
        buffer = []

    time.sleep(SLEEP_SEC)

# ── 4. 최종 저장 ─────────────────────────────────────────────────
if buffer:
    pd.DataFrame(buffer).to_csv(
        OUTPUT_FILE, mode='a', index=False, header=False, encoding='utf-8-sig')

print(f"\n{'─'*50}")
print(f"✅ 수집 완료  |  성공: {ok_cnt}개  /  실패(NoData 등): {fail_cnt}개")
print(f"   저장 위치: {OUTPUT_FILE}")

# ── 5. 최종 현황 확인 ────────────────────────────────────────────
df_final = pd.read_csv(OUTPUT_FILE)
print(f"\n📊 최종 파일 현황")
print(f"   전체 행 수   : {len(df_final):,}")
print(f"   고유 논문 수 : {df_final['source_id'].nunique():,}")
print(f"\n연도별 고유 논문:")
print(df_final.drop_duplicates('source_id')['pub_year'].value_counts().sort_index().to_string())
