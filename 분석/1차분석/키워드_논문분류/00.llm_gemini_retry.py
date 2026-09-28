"""
Gemini 분류불가 행 재시도 스크립트
=====================================
- 카테고리_gemini가 '분류불가'인 행만 골라서 Gemini API 재호출
- Claude / GPT 결과는 건드리지 않음

사용법:
    python 00.llm_gemini_retry.py
"""

import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# .env 로드
_env_path = Path(__file__).parent / ".env"
if not _env_path.exists():
    print(f"[오류] .env 파일이 없습니다: {_env_path}")
    sys.exit(1)
load_dotenv(_env_path)

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GEMINI_MODEL":   "gemini-2.5-flash",

    "INPUT_CSV":  "00.전체_법률_목록(2026.03.16).csv",
    "OUTPUT_CSV": "00.전체_법률_목록(2026.03.16).csv",

    "BATCH_SIZE":    50,
    "RETRY_COUNT":   3,
    "RETRY_DELAY":   10,   # 429 발생 시 대기 (초)
    "REQUEST_DELAY": 0.5,  # Tier1 기준 최대 300 RPM → 0.5초면 충분
}

CATEGORIES = [
    "공법", "민사법", "형사법", "경제법", "노동법",
    "사회보장법", "금융법", "의료법", "조세법",
    "데이터법", "지식재산권법", "기초법",
]

SYSTEM_PROMPT = """당신은 대한민국 법령 분류 전문가입니다.
주어진 법령 정보를 바탕으로 아래 카테고리 중 가장 적합한 하나를 선택하세요.

[카테고리 목록]
- 공법: 헌법·행정법·행정조직·공무원·선거·국방·경찰·소방 등
- 민사법: 민법·상법·가족법·부동산·계약·절차법 등
- 형사법: 형법·형사소송법·범죄 처벌 관련법 등
- 경제법: 산업진흥·통상·공정거래·교통·건설·환경·에너지 등
- 노동법: 근로관계·고용·산재·노사 관련법 등
- 사회보장법: 사회복지·건강보험·연금·고령자·장애인 등
- 금융법: 금융·은행·증권·보험·자본시장 등
- 의료법: 의료·약사·보건·식품 등
- 조세법: 소득세·부가세·법인세 등 세법 전반
- 데이터법: 정보통신·개인정보보호·사이버보안·방송·IT 등
- 지식재산권법: 특허·저작권·상표·디자인권 등
- 기초법: 법학이론·비교법·국제법 일반 등

[규칙]
- 반드시 위 목록 중 정확히 하나만 답하세요.
- 다른 설명 없이 카테고리명만 출력하세요.
- 예시 출력: 공법"""


def build_user_prompt(row: pd.Series) -> str:
    return (
        f"법령명: {row['법령명']}\n"
        f"소관부처명: {row['소관부처명']}\n"
        f"분야명_한글: {row['분야명_한글']}\n"
        f"기존 카테고리: {row['카테고리']}"
    )


def validate_category(response) -> str:
    if not response:
        return "분류불가"
    response = str(response).strip()
    for cat in CATEGORIES:
        if cat in response:
            return cat
    return "분류불가"


def call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=CONFIG["GEMINI_API_KEY"])

    for attempt in range(CONFIG["RETRY_COUNT"]):
        try:
            resp = client.models.generate_content(
                model=CONFIG["GEMINI_MODEL"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=50,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        types.SafetySetting(category=c, threshold="BLOCK_NONE")
                        for c in [
                            "HARM_CATEGORY_HARASSMENT",
                            "HARM_CATEGORY_HATE_SPEECH",
                            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "HARM_CATEGORY_DANGEROUS_CONTENT",
                        ]
                    ],
                ),
            )
            return validate_category(resp.text)
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                wait = CONFIG["RETRY_DELAY"] * (attempt + 1)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 60  # rate limit은 1분 대기
                time.sleep(wait)
            else:
                print(f"\n[Gemini 오류] {e}")
                return "오류"


def main():
    print("=" * 60)
    print("  Gemini 분류불가 행 재시도")
    print("=" * 60)

    if not CONFIG["GEMINI_API_KEY"]:
        print("[오류] .env에 GEMINI_API_KEY가 없습니다.")
        sys.exit(1)

    df = pd.read_csv(CONFIG["INPUT_CSV"], encoding="utf-8-sig")

    # 재시도 대상: 분류불가 또는 NaN
    retry_mask = (
        df["카테고리_gemini"].isna() |
        (df["카테고리_gemini"] == "분류불가")
    )
    retry_indices = df[retry_mask].index.tolist()

    print(f"[대상] 전체 {len(df)}개 중 재시도 대상: {len(retry_indices)}개\n")

    if not retry_indices:
        print("재시도할 행이 없습니다.")
        return

    processed = 0
    still_failed = 0

    with tqdm(total=len(retry_indices), desc="Gemini 재시도", unit="행") as pbar:
        for idx in retry_indices:
            row = df.loc[idx]
            prompt = build_user_prompt(row)
            result = call_gemini(prompt)

            df.at[idx, "카테고리_gemini"] = result

            if result == "분류불가":
                still_failed += 1

            processed += 1
            pbar.update(1)

            if processed % CONFIG["BATCH_SIZE"] == 0:
                df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding="utf-8-sig")
                pbar.set_postfix({"저장": f"{processed}행"})

            time.sleep(CONFIG["REQUEST_DELAY"])

    df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding="utf-8-sig")

    print(f"\n[완료] 처리: {processed}개 / 여전히 분류불가: {still_failed}개")
    print(f"[저장] {CONFIG['OUTPUT_CSV']}")
    print("\n카테고리_gemini 최종 분포:")
    print(df["카테고리_gemini"].value_counts().to_string())


if __name__ == "__main__":
    main()
