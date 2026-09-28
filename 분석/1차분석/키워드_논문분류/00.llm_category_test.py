"""
법률 카테고리 LLM 분류 - 테스트 & 비용 추정 스크립트
=====================================================
- 전체 1,706개 중 50개만 샘플링하여 3개 API 테스트
- 실제 토큰 사용량 기록 → 전체 실행 시 예상 비용 추정
- 결과는 별도 CSV로 저장 (원본 파일 수정 없음)

사용법:
    python 00.llm_category_test.py
"""

import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
_env_path = Path(__file__).parent / ".env"
if not _env_path.exists():
    print(f"[오류] .env 파일이 없습니다: {_env_path}")
    print("  → .env.sample을 복사하여 .env를 만들고 API 키를 입력하세요.")
    sys.exit(1)
load_dotenv(_env_path)


# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "OPENAI_API_KEY":    os.getenv("OPENAI_API_KEY"),
    "GEMINI_API_KEY":    os.getenv("GEMINI_API_KEY"),

    "CLAUDE_MODEL":  "claude-haiku-4-5-20251001",
    "GPT_MODEL":     "gpt-4o-mini",
    "GEMINI_MODEL":  "gemini-2.5-flash",

    "INPUT_CSV":     "00.전체_법률_목록(2026.03.16).csv",
    "OUTPUT_CSV":    "00.llm_category_test_결과.csv",   # 원본과 별도 저장

    "SAMPLE_SIZE":   50,      # 테스트할 행 수
    "SAMPLE_SEED":   42,      # 재현 가능한 랜덤 샘플링
    "MAX_WORKERS":   3,
    "RETRY_COUNT":   2,
    "RETRY_DELAY":   2,
    "REQUEST_DELAY": 0.3,
}

# 모델별 가격 (USD / 1M 토큰, 2025년 기준 - 변동 가능)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gemini-2.0-flash":          {"input": 0.10,  "output": 0.40},
}

CATEGORIES = [
    "공법", "민사법", "형사법", "경제법", "노동법",
    "사회보장법", "금융법", "의료법", "조세법",
    "데이터법", "지식재산권법", "기초법",
]

SYSTEM_PROMPT = """당신은 대한민국 법령 분류 전문가입니다.
주어진 법령 정보를 바탕으로 아래 카테고리 중 가장 적합한 하나만 선택하세요.

[카테고리 목록]
- 공법: 국가의 조직·작용 및 공권력 행사 관련 (헌법, 행정법, 행정조직, 공무원, 선거, 국방, 경찰, 소방 등)
- 민사법: 사인 간 권리·의무 관계 (민법, 상법, 가족법, 부동산, 계약, 민사절차 등)
- 형사법: 범죄와 형벌 및 형사절차 (형법, 형사소송법, 처벌법 등)
- 경제법: 산업·시장 규제 및 경제질서 (공정거래, 산업진흥, 통상, 건설, 교통, 환경, 에너지 등)
- 노동법: 근로관계 및 노사관계 (근로기준, 고용, 산재, 노동조합 등)
- 사회보장법: 국민의 생활보장 및 복지 (사회복지, 건강보험, 연금, 고령자, 장애인 등)
- 금융법: 금융거래 및 금융기관 규제 (은행, 증권, 보험, 자본시장 등)
- 의료법: 보건·의료 및 의약 관련 (의료, 약사, 식품, 위생 등)
- 조세법: 조세의 부과·징수 (소득세, 법인세, 부가가치세 등)
- 데이터법: 정보통신 및 데이터 보호 (개인정보보호, 사이버보안, 방송, IT 등)
- 지식재산권법: 창작물 및 산업재산 보호 (특허, 저작권, 상표, 디자인 등)
- 기타: 위 기준으로 명확히 분류되지 않는 경우

[판단 기준]
- 법령의 “주된 목적”과 “핵심 규율 대상”을 기준으로 판단하세요.
- 여러 분야에 걸치는 경우, 가장 중심적인 영역 하나만 선택하세요.

[출력 규칙]
- 반드시 위 목록 중 하나만 선택하세요.
- 카테고리명만 출력하세요. (다른 설명 금지)
- 예시: 공법"""


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


# ─────────────────────────────────────────────
# 🤖 API 호출 (토큰 수 반환 포함)
# ─────────────────────────────────────────────

def call_claude(prompt: str) -> tuple[str, int, int]:
    """(카테고리, input_tokens, output_tokens)"""
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
            return (
                validate_category(msg.content[0].text),
                msg.usage.input_tokens,
                msg.usage.output_tokens,
            )
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"])
            else:
                print(f"\n  [Claude 오류] {e}")
                return ("오류", 0, 0)


def call_gpt(prompt: str) -> tuple[str, int, int]:
    """(카테고리, input_tokens, output_tokens)"""
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
            return (
                validate_category(resp.choices[0].message.content),
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"])
            else:
                print(f"\n  [GPT 오류] {e}")
                return ("오류", 0, 0)


def call_gemini(prompt: str) -> tuple[str, int, int]:
    """(카테고리, input_tokens, output_tokens)"""
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
            usage = resp.usage_metadata
            return (
                validate_category(resp.text),
                usage.prompt_token_count,
                usage.candidates_token_count,
            )
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"])
            else:
                print(f"\n  [Gemini 오류] {e}")
                return ("오류", 0, 0)


# ─────────────────────────────────────────────
# 💰 비용 계산
# ─────────────────────────────────────────────

def estimate_cost(model: str, total_input: int, total_output: int, total_rows: int) -> dict:
    """샘플 토큰 기준으로 전체 실행 비용 추정"""
    price = PRICING.get(model, {"input": 0, "output": 0})
    sample = CONFIG["SAMPLE_SIZE"]

    # 전체 행 기준 추정 토큰
    est_input  = total_input  / sample * total_rows
    est_output = total_output / sample * total_rows

    cost_usd = (est_input * price["input"] + est_output * price["output"]) / 1_000_000
    cost_krw = cost_usd * 1_380  # 환율 대략 적용

    return {
        "샘플_input_tokens":  total_input,
        "샘플_output_tokens": total_output,
        "전체추정_input":     int(est_input),
        "전체추정_output":    int(est_output),
        "예상비용_USD":       round(cost_usd, 4),
        "예상비용_KRW":       int(cost_krw),
    }


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  법률 카테고리 LLM 분류 - 테스트 (50개 샘플)")
    print("=" * 60)

    # API 키 검증
    missing = [k for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
               if not CONFIG.get(k)]
    if missing:
        print(f"[오류] .env에 다음 키가 없습니다: {', '.join(missing)}")
        sys.exit(1)
    print(f"[.env] API 키 로드 완료\n")

    # 데이터 로드 & 샘플링
    df_all = pd.read_csv(CONFIG["INPUT_CSV"], encoding="utf-8-sig")
    total_rows = len(df_all)
    df = df_all.sample(n=CONFIG["SAMPLE_SIZE"], random_state=CONFIG["SAMPLE_SEED"]).copy()
    df = df.reset_index(drop=True)
    print(f"[샘플] 전체 {total_rows}개 중 {CONFIG['SAMPLE_SIZE']}개 무작위 추출 (seed={CONFIG['SAMPLE_SEED']})\n")

    # 결과 칼럼 추가
    for col in ["카테고리_claude", "카테고리_gpt", "카테고리_gemini"]:
        df[col] = None

    # 토큰 카운터
    token_totals = {
        "claude":  {"input": 0, "output": 0},
        "gpt":     {"input": 0, "output": 0},
        "gemini":  {"input": 0, "output": 0},
    }

    api_map = {
        "카테고리_claude":  ("claude",  call_claude),
        "카테고리_gpt":     ("gpt",     call_gpt),
        "카테고리_gemini":  ("gemini",  call_gemini),
    }

    print(f"{'':>4}  {'법령명':30}  {'Claude':10}  {'GPT':10}  {'Gemini':10}")
    print("-" * 72)

    for i, row in df.iterrows():
        prompt = build_user_prompt(row)
        results = {}

        with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            future_to_key = {
                executor.submit(fn, prompt): (col, name)
                for col, (name, fn) in api_map.items()
            }
            for future in as_completed(future_to_key):
                col, name = future_to_key[future]
                category, inp, out = future.result()
                results[col] = category
                token_totals[name]["input"]  += inp
                token_totals[name]["output"] += out

        for col, val in results.items():
            df.at[i, col] = val

        law_name = str(row["법령명"])[:28]
        print(f"  {i+1:>2}.  {law_name:<30}  "
              f"{results.get('카테고리_claude',''):10}  "
              f"{results.get('카테고리_gpt',''):10}  "
              f"{results.get('카테고리_gemini',''):10}")

        time.sleep(CONFIG["REQUEST_DELAY"])

    # 결과 저장
    df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding="utf-8-sig")
    print(f"\n[저장] {CONFIG['OUTPUT_CSV']}")

    # ── 비용 추정 출력 ──────────────────────────────
    print("\n" + "=" * 60)
    print("  💰 비용 추정 (전체 실행 시)")
    print("=" * 60)

    model_map = {
        "claude": CONFIG["CLAUDE_MODEL"],
        "gpt":    CONFIG["GPT_MODEL"],
        "gemini": CONFIG["GEMINI_MODEL"],
    }

    total_usd = 0
    for name, model in model_map.items():
        est = estimate_cost(
            model,
            token_totals[name]["input"],
            token_totals[name]["output"],
            total_rows,
        )
        total_usd += est["예상비용_USD"]
        print(f"\n  [{name.upper()} / {model}]")
        print(f"    샘플 토큰:   input {est['샘플_input_tokens']:,}  /  output {est['샘플_output_tokens']:,}")
        print(f"    전체 추정:   input {est['전체추정_input']:,}  /  output {est['전체추정_output']:,}")
        print(f"    예상 비용:   ${est['예상비용_USD']:.4f}  ≈  ₩{est['예상비용_KRW']:,}")

    print(f"\n  ─────────────────────────────────")
    print(f"  3개 모델 합계 예상:  ${total_usd:.4f}  ≈  ₩{int(total_usd * 1_380):,}")
    print("=" * 60)

    # ── 분류 일치율 ──────────────────────────────
    print("\n  📊 3개 모델 간 분류 일치율")
    print("-" * 40)
    all_match = (
        (df["카테고리_claude"] == df["카테고리_gpt"]) &
        (df["카테고리_gpt"]    == df["카테고리_gemini"])
    ).sum()
    any_two = (
        (df["카테고리_claude"] == df["카테고리_gpt"]) |
        (df["카테고리_gpt"]    == df["카테고리_gemini"]) |
        (df["카테고리_claude"] == df["카테고리_gemini"])
    ).sum()
    print(f"  3개 모두 일치:  {all_match} / {CONFIG['SAMPLE_SIZE']} ({all_match/CONFIG['SAMPLE_SIZE']*100:.1f}%)")
    print(f"  2개 이상 일치:  {any_two} / {CONFIG['SAMPLE_SIZE']} ({any_two/CONFIG['SAMPLE_SIZE']*100:.1f}%)")

    print("\n완료!")


if __name__ == "__main__":
    main()
