"""
법률 카테고리 LLM 자동 분류 스크립트
=====================================
- Claude (Anthropic) / GPT (OpenAI) / Gemini (Google) 3개 모델 동시 분류
- 결과를 카테고리_claude, 카테고리_gpt, 카테고리_gemini 칼럼에 저장
- 체크포인트 지원: 중단 후 재시작 시 이미 처리된 행은 건너뜀
- 배치 단위로 저장하여 데이터 유실 방지

사용법:
    pip install anthropic openai google-genai tqdm pandas python-dotenv
    python 00.llm_category_check.py

설정:
    스크립트와 같은 폴더의 .env 파일에 API 키를 입력하세요. (.env.sample 참고)
"""

import os
import time
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (스크립트 위치 기준)
_env_path = Path(__file__).parent / ".env"
if not _env_path.exists():
    print(f"[오류] .env 파일이 없습니다: {_env_path}")
    print("  → .env.sample을 복사하여 .env를 만들고 API 키를 입력하세요.")
    sys.exit(1)
load_dotenv(_env_path)


# ─────────────────────────────────────────────
# ⚙️  CONFIG (여기서 설정 변경)
# ─────────────────────────────────────────────

CONFIG = {
    # API 키 (.env 파일에서 자동 로드)
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "OPENAI_API_KEY":    os.getenv("OPENAI_API_KEY"),
    "GEMINI_API_KEY":    os.getenv("GEMINI_API_KEY"),

    # 모델 (비용 효율적인 모델 기본값)
    "CLAUDE_MODEL":  "claude-haiku-4-5-20251001",
    "GPT_MODEL":     "gpt-4o-mini",
    "GEMINI_MODEL":  "gemini-2.5-flash",

    # 파일 경로
    "INPUT_CSV":  "00.전체_법률_목록(2026.03.16).csv",
    "OUTPUT_CSV": "00.전체_법률_목록(2026.03.16).csv",  # 덮어쓰기 (원본 유지하려면 변경)

    # 처리 설정
    "BATCH_SIZE":     50,     # 몇 행마다 저장할지
    "MAX_WORKERS":    3,      # 3개 API 병렬 호출 스레드 수
    "RETRY_COUNT":    3,      # API 오류 시 재시도 횟수
    "RETRY_DELAY":    2,      # 재시도 대기 초
    "REQUEST_DELAY":  0.3,    # 각 행 처리 후 대기 초 (rate limit 방지)
}

# 분류 가능한 카테고리 목록
CATEGORIES = [
    "공법",        # 헌법, 행정법, 조직법 등
    "민사법",      # 민법, 상법, 절차법 등
    "형사법",      # 형법, 형사소송법 등
    "경제법",      # 산업·통상·에너지·교통 등
    "노동법",      # 근로관계, 고용 등
    "사회보장법",  # 복지, 건강보험, 연금 등
    "금융법",      # 금융·은행·증권·보험 등
    "의료법",      # 의료·약사·보건 등
    "조세법",      # 세법 전반
    "데이터법",    # 정보통신·개인정보·사이버보안 등
    "지식재산권법",# 특허·저작권·상표 등
    "기타",        # 위 카테고리에 해당하지 않는 법령
]

SYSTEM_PROMPT = f"""당신은 대한민국 법령 분류 전문가입니다.
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
- 기타: 위 카테고리에 해당하지 않는 법령

[규칙]
- 반드시 위 목록 중 정확히 하나만 답하세요.
- 다른 설명 없이 카테고리명만 출력하세요.
- 예시 출력: 공법"""


def build_user_prompt(row: pd.Series) -> str:
    """분류 요청 프롬프트 생성"""
    return (
        f"법령명: {row['법령명']}\n"
        f"소관부처명: {row['소관부처명']}\n"
        f"분야명_한글: {row['분야명_한글']}\n"
        f"기존 카테고리: {row['카테고리']}"
    )


def validate_category(response) -> str:
    """응답에서 유효한 카테고리 추출 (None 안전 처리)"""
    if not response:
        return "분류불가"
    response = str(response).strip()
    for cat in CATEGORIES:
        if cat in response:
            return cat
    return "분류불가"


# ─────────────────────────────────────────────
# 🤖 API 호출 함수
# ─────────────────────────────────────────────

def call_claude(prompt: str) -> str:
    """Claude API 호출"""
    import anthropic
    client = anthropic.Anthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])

    for attempt in range(CONFIG["RETRY_COUNT"]):
        try:
            msg = client.messages.create(
                model=CONFIG["CLAUDE_MODEL"],
                max_tokens=20,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return validate_category(msg.content[0].text)
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"] * (attempt + 1))
            else:
                print(f"\n[Claude 오류] {e}")
                return "오류"


def call_gpt(prompt: str) -> str:
    """GPT API 호출"""
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

    for attempt in range(CONFIG["RETRY_COUNT"]):
        try:
            resp = client.chat.completions.create(
                model=CONFIG["GPT_MODEL"],
                max_tokens=20,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return validate_category(resp.choices[0].message.content)
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"] * (attempt + 1))
            else:
                print(f"\n[GPT 오류] {e}")
                return "오류"


def call_gemini(prompt: str) -> str:
    """Gemini API 호출"""
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
                time.sleep(CONFIG["RETRY_DELAY"] * (attempt + 1))
            else:
                print(f"\n[Gemini 오류] {e}")
                return "오류"


# ─────────────────────────────────────────────
# 🚀 메인 실행
# ─────────────────────────────────────────────

def classify_row(row: pd.Series) -> dict:
    """한 행에 대해 3개 API를 병렬 호출"""
    prompt = build_user_prompt(row)

    results = {}
    api_funcs = {
        "카테고리_claude": call_claude,
        "카테고리_gpt":    call_gpt,
        "카테고리_gemini": call_gemini,
    }

    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_key = {
            executor.submit(fn, prompt): key
            for key, fn in api_funcs.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = "오류"
                print(f"\n[{key} 예외] {e}")

    return results


def main():
    print("=" * 60)
    print("  법률 카테고리 LLM 자동 분류")
    print("=" * 60)

    # API 키 검증
    missing = [k for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
               if not CONFIG.get(k)]
    if missing:
        print(f"[오류] .env에 다음 키가 없습니다: {', '.join(missing)}")
        sys.exit(1)
    print(f"[.env] API 키 로드 완료 ({_env_path})")

    # 데이터 로드
    df = pd.read_csv(CONFIG["INPUT_CSV"], encoding="utf-8-sig")
    print(f"[로드] {len(df)}개 법률 목록")

    # 결과 칼럼 초기화 (없으면 추가)
    for col in ["카테고리_claude", "카테고리_gpt", "카테고리_gemini"]:
        if col not in df.columns:
            df[col] = None

    # 처리 대상: 3개 칼럼 중 하나라도 비어있는 행
    todo_mask = (
        df["카테고리_claude"].isna() |
        df["카테고리_gpt"].isna() |
        df["카테고리_gemini"].isna()
    )
    todo_indices = df[todo_mask].index.tolist()

    print(f"[체크포인트] 처리 완료: {len(df) - len(todo_indices)}개 / 남은 행: {len(todo_indices)}개")

    if not todo_indices:
        print("모든 행이 이미 처리되었습니다.")
        return

    # 처리 시작
    processed = 0
    with tqdm(total=len(todo_indices), desc="분류 진행", unit="행") as pbar:
        for idx in todo_indices:
            row = df.loc[idx]

            # 이미 채워진 칼럼은 건너뜀 (부분 완료 행 대응)
            results = {}
            prompt = build_user_prompt(row)

            api_calls = {}
            if pd.isna(df.at[idx, "카테고리_claude"]):
                api_calls["카테고리_claude"] = call_claude
            if pd.isna(df.at[idx, "카테고리_gpt"]):
                api_calls["카테고리_gpt"] = call_gpt
            if pd.isna(df.at[idx, "카테고리_gemini"]):
                api_calls["카테고리_gemini"] = call_gemini

            with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
                future_to_key = {
                    executor.submit(fn, prompt): key
                    for key, fn in api_calls.items()
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        results[key] = "오류"

            # 결과 반영
            for col, val in results.items():
                df.at[idx, col] = val

            processed += 1
            pbar.update(1)

            # 배치 저장
            if processed % CONFIG["BATCH_SIZE"] == 0:
                df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding="utf-8-sig")
                pbar.set_postfix({"저장": f"{processed}행 완료"})

            # rate limit 방지
            time.sleep(CONFIG["REQUEST_DELAY"])

    # 최종 저장
    df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding="utf-8-sig")

    # 결과 요약
    print("\n" + "=" * 60)
    print("  분류 완료 요약")
    print("=" * 60)
    for col in ["카테고리_claude", "카테고리_gpt", "카테고리_gemini"]:
        print(f"\n[{col}]")
        print(df[col].value_counts().to_string())

    print(f"\n[저장] {CONFIG['OUTPUT_CSV']}")
    print("완료!")


if __name__ == "__main__":
    main()
