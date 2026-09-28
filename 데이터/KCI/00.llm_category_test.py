"""
KCI AI 법학 논문 카테고리 LLM 분류 - 배치 처리 테스트
=====================================================
- 30개 샘플로 3개 모델 × 2가지 모드 배치 테스트
- 모드 A: 제목만  /  모드 B: 제목 + 초록

[배치 처리 방식]  ← finalize와 동일 구조로 검증
  Claude  → Anthropic Message Batches API (50% 할인)
  GPT     → OpenAI Batch API              (50% 할인)
  Gemini  → ThreadPoolExecutor 병렬 처리

custom_id 형식: "{논문ID}__A" / "{논문ID}__B"
  → 한 배치에 모드 A·B 동시 제출 (Claude 60건, GPT 60건)

사용법:
    python 00.llm_category_test.py
"""

import io
import json
import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
_env_path = next(p for p in Path(__file__).resolve().parents if (p / ".env").exists()) / ".env"
if not _env_path.exists():
    print(f"[오류] .env 파일이 없습니다: {_env_path}")
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

    "INPUT_CSV":     "KCI_AI_논문_기본정보_목록_201601_202512.csv",
    "OUTPUT_CSV":    "00.llm_category_test_결과.csv",

    "SAMPLE_SIZE":   30,
    "SAMPLE_SEED":   42,
    "GEMINI_WORKERS": 6,
    "RETRY_COUNT":   2,
    "RETRY_DELAY":   3,
    "BATCH_POLL_SEC": 20,   # 테스트용: 폴링 간격 짧게

    "ABSTRACT_MAX_CHARS": 99999,  # 사실상 무제한
}

# 배치 할인 반영 가격 (USD / 1M 토큰)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.50,  "output": 2.50},   # 50% 할인
    "gpt-4o-mini":               {"input": 0.075, "output": 0.30},   # 50% 할인
    "gemini-2.5-flash":          {"input": 0.10,  "output": 0.40},   # 배치 미지원
}

CATEGORIES = [
    "공법", "민사법", "형사법", "경제법", "노동법",
    "사회보장법", "금융법", "의료법", "조세법",
    "데이터법", "인공지능법", "지식재산권법", "기초법",
    "리걸테크", "자율무기법", "자율주행자동차법", "해외법",
]

# ─────────────────────────────────────────────
# 📋 시스템 프롬프트
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 법학 연구 논문 분류 전문가입니다.
주어진 논문 정보를 바탕으로 아래 카테고리 중 가장 적합한 하나만 선택하세요.

[카테고리 목록]
- 공법: 헌법, 행정법, 공무원, 선거, 국방, 경찰, 지방자치, 행정절차 등 국가 공권력 관련
- 민사법: 민법(계약·불법행위·가족), 상법(회사·보험), 민사소송법 등 사인 간 법률관계
- 형사법: 형법(범죄·형벌), 형사소송법 관련
- 경제법: 공정거래, 소비자보호, 환경법, 에너지, 교통, 산업규제 관련
- 노동법: 근로기준, 고용보험, 산업안전, 노동조합, 노사관계 관련
- 사회보장법: 사회복지, 국민연금, 건강보험, 장애인·노인복지법 관련
- 금융법: 은행, 자본시장법(증권), 핀테크, 가상자산 규제, 금융감독 관련
- 의료법: 의료법, 약사법, 바이오헬스, 식품위생, 보건정책 관련
- 조세법: 소득세, 법인세, 부가가치세, 상속세, 조세 행정 및 심판 관련
- 데이터법: 개인정보, 사이버보안, 정보통신망법, 방송통신 규제 관련
- 인공지능법: AI 거버넌스, 알고리즘, 알고리즘 책임, 자율주행 법제, AI 윤리 및 일반 규제
- 지식재산권법: 특허, 저작권, 상표, 디자인, 영업비밀, 콘텐츠 법제 관련
- 기초법: 법철학, 법사학, 법사회학, 법이론, 비교법, 일반 국제법(조약·국제기구)
- 리걸테크: 법률 AI 서비스, 판결 예측, 법률 자동화 도구, 리걸 데이터베이스 연구
- 자율무기법: AI 무기체계, 군사 AI, 국제인도법상 자동화 병기 관련
- 자율주행자동차법: 자율주행차 관련 법률, 규제, 사고 책임, 교통법규 등
- 해외법: 미국·유럽·중국 등 특정 국가·지역 법제 연구

[분류 우선순위 및 판단 기준]
1. (기술 vs 실체법) AI 기술이 활용되더라도 쟁점이 전통적인 민사책임이면 '민사법', 형사처벌이면 '형사법'으로 분류하세요. '인공지능법'은 AI 전용 법안이나 알고리즘 특유의 거버넌스를 다룰 때 선택합니다.
2. (데이터 vs 지적재산) 데이터 자체의 보호와 프라이버시는 '데이터법', 데이터의 창작성과 권리 관계는 '지식재산권법'으로 분류하세요.

[출력 규칙]
- 반드시 카테고리명만 출력하세요. (예: 민사법)
- 어떤 설명도 덧붙이지 마세요."""


# ─────────────────────────────────────────────
# 🏷️  키워드 예외 처리
# ─────────────────────────────────────────────

KEYWORD_OVERRIDES: dict[str, str] = {
    "리걸테크": "리걸테크",
    "자율무기": "자율무기법",
}

def check_keyword_override(title: str) -> str | None:
    for keyword, category in KEYWORD_OVERRIDES.items():
        if keyword in str(title):
            return category
    return None


# ─────────────────────────────────────────────
# 📝 프롬프트 빌더
# ─────────────────────────────────────────────

def build_prompt_A(row: pd.Series) -> str:
    return f"논문제목: {row['제목']}"

def build_prompt_B(row: pd.Series) -> str:
    abstract = str(row["초록"]) if pd.notna(row["초록"]) else ""
    if len(abstract) > CONFIG["ABSTRACT_MAX_CHARS"]:
        abstract = abstract[:CONFIG["ABSTRACT_MAX_CHARS"]] + "…"
    return f"논문제목: {row['제목']}\n초록: {abstract}"

def validate_category(response) -> str:
    if not response:
        return "분류불가"
    response = str(response).strip()
    for cat in CATEGORIES:
        if cat in response:
            return cat
    return "분류불가"


# ─────────────────────────────────────────────
# 🤖 Claude — Anthropic Batch API
# ─────────────────────────────────────────────

def run_claude_batch(prompts: dict[str, str]) -> tuple[dict[str, str], int, int]:
    """
    prompts: {custom_id: prompt}  (custom_id = "{논문ID}__A" / "__B")
    반환: (결과dict, 총input토큰, 총output토큰)
    """
    import anthropic
    client = anthropic.Anthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])

    print(f"  [Claude 배치] {len(prompts)}건 제출...")
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": cid,
                "params": {
                    "model":      CONFIG["CLAUDE_MODEL"],
                    "max_tokens": 20,
                    "system":     SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            }
            for cid, prompt in prompts.items()
        ]
    )

    # 완료 대기
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        total_req = c.processing + c.succeeded + c.errored + c.canceled + c.expired
        print(f"  [Claude 배치] {batch.processing_status} — 완료 {c.succeeded}/{total_req}")
        if batch.processing_status == "ended":
            break
        time.sleep(CONFIG["BATCH_POLL_SEC"])

    # 결과 수집
    results: dict[str, str] = {}
    total_in, total_out = 0, 0
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            msg = result.result.message
            results[result.custom_id] = validate_category(msg.content[0].text)
            total_in  += msg.usage.input_tokens
            total_out += msg.usage.output_tokens
        else:
            results[result.custom_id] = "오류"

    print(f"  [Claude 배치] 완료 ({len(results)}건, 토큰 in={total_in:,} out={total_out:,})")
    return results, total_in, total_out


# ─────────────────────────────────────────────
# 🤖 GPT — OpenAI Batch API
# ─────────────────────────────────────────────

def run_gpt_batch(prompts: dict[str, str]) -> tuple[dict[str, str], int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

    print(f"  [GPT 배치] {len(prompts)}건 제출...")
    jsonl = "\n".join(
        json.dumps({
            "custom_id": cid,
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":      CONFIG["GPT_MODEL"],
                "max_tokens": 20,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            },
        }, ensure_ascii=False)
        for cid, prompt in prompts.items()
    )
    uploaded = client.files.create(
        file=("batch.jsonl", io.BytesIO(jsonl.encode("utf-8"))),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # 완료 대기
    while True:
        batch = client.batches.retrieve(batch.id)
        c = batch.request_counts
        print(f"  [GPT 배치] {batch.status} — 완료 {c.completed}/{c.total}")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(CONFIG["BATCH_POLL_SEC"])

    if batch.status != "completed":
        print(f"  [GPT 배치] 경고: {batch.status} 상태로 종료")
        return {}, 0, 0

    # 결과 수집
    output = client.files.content(batch.output_file_id)
    results: dict[str, str] = {}
    total_in, total_out = 0, 0
    for line in output.text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        try:
            body    = r["response"]["body"]
            text    = body["choices"][0]["message"]["content"]
            usage   = body.get("usage", {})
            results[r["custom_id"]] = validate_category(text)
            total_in  += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
        except Exception:
            results[r["custom_id"]] = "오류"

    print(f"  [GPT 배치] 완료 ({len(results)}건, 토큰 in={total_in:,} out={total_out:,})")
    return results, total_in, total_out


# ─────────────────────────────────────────────
# 🤖 Gemini — ThreadPoolExecutor 병렬
# ─────────────────────────────────────────────

def _call_gemini_once(cid: str, prompt: str) -> tuple[str, str, int, int]:
    """(custom_id, 카테고리, input_tokens, output_tokens)"""
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
            return (cid, validate_category(resp.text),
                    usage.prompt_token_count, usage.candidates_token_count)
        except Exception:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"])
    return cid, "오류", 0, 0


def run_gemini_parallel(prompts: dict[str, str]) -> tuple[dict[str, str], int, int]:
    print(f"  [Gemini] {len(prompts)}건 병렬 처리 (workers={CONFIG['GEMINI_WORKERS']})...")
    results: dict[str, str] = {}
    total_in, total_out = 0, 0
    with ThreadPoolExecutor(max_workers=CONFIG["GEMINI_WORKERS"]) as executor:
        futures = {
            executor.submit(_call_gemini_once, cid, prompt): cid
            for cid, prompt in prompts.items()
        }
        for future in as_completed(futures):
            cid, category, inp, out = future.result()
            results[cid] = category
            total_in  += inp
            total_out += out
    print(f"  [Gemini] 완료 ({len(results)}건, 토큰 in={total_in:,} out={total_out:,})")
    return results, total_in, total_out


# ─────────────────────────────────────────────
# 💰 비용 출력
# ─────────────────────────────────────────────

def print_cost_section(
    token_data: dict,   # {"claude": (in,out), "gpt": (in,out), "gemini": (in,out)}
    total_papers: int,
    sample_size: int,
):
    KRW = 1_380
    model_map = {
        "claude": CONFIG["CLAUDE_MODEL"],
        "gpt":    CONFIG["GPT_MODEL"],
        "gemini": CONFIG["GEMINI_MODEL"],
    }
    print(f"\n  {'모델':8s}  {'샘플 토큰(in/out)':22s}  {'전체 추정(in/out)':22s}  {'예상 비용':16s}")
    print("  " + "─" * 72)
    total_usd = 0
    for name, model in model_map.items():
        s_in, s_out = token_data[name]
        price = PRICING[model]
        scale = total_papers / sample_size
        e_in  = int(s_in  * scale)
        e_out = int(s_out * scale)
        usd   = (e_in * price["input"] + e_out * price["output"]) / 1_000_000
        total_usd += usd
        discount = " (배치50%↓)" if name != "gemini" else " (병렬)"
        print(f"  {name.upper():8s}  {s_in:>8,}/{s_out:<8,}     "
              f"{e_in:>8,}/{e_out:<8,}     "
              f"${usd:.4f} ≈ ₩{int(usd*KRW):,}{discount}")
    print(f"  {'합계':8s}  {'':22s}  {'':22s}  "
          f"${total_usd:.4f} ≈ ₩{int(total_usd*KRW):,}")
    return total_usd


def print_agreement(prefix: str, df: pd.DataFrame):
    c, g, m = f"{prefix}_claude", f"{prefix}_gpt", f"{prefix}_gemini"
    n = len(df)
    all3 = ((df[c] == df[g]) & (df[g] == df[m])).sum()
    any2 = ((df[c] == df[g]) | (df[g] == df[m]) | (df[c] == df[m])).sum()
    print(f"    3개 모두 일치: {all3:2d}/{n} ({all3/n*100:.0f}%)")
    print(f"    2개 이상 일치: {any2:2d}/{n} ({any2/n*100:.0f}%)")


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  KCI AI 법학 논문 카테고리 — 배치 처리 테스트")
    print("  모드 A: 제목만  /  모드 B: 제목+초록  (배치 API)")
    print("=" * 65)

    missing = [k for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
               if not CONFIG.get(k)]
    if missing:
        print(f"[오류] .env에 다음 키가 없습니다: {', '.join(missing)}")
        sys.exit(1)
    print("[.env] API 키 로드 완료\n")

    # ── 데이터 샘플링 ────────────────────────────
    df_all   = pd.read_csv(CONFIG["INPUT_CSV"], encoding="utf-8-sig")
    total_papers = len(df_all)
    df_valid = df_all[df_all["초록"].notna() & (df_all["초록"].str.strip() != "")]
    df = df_valid.sample(n=CONFIG["SAMPLE_SIZE"], random_state=CONFIG["SAMPLE_SEED"]).copy()
    df = df.reset_index(drop=True)
    print(f"[샘플] {total_papers}편 중 초록 보유 {len(df_valid)}편에서 {CONFIG['SAMPLE_SIZE']}편 추출\n")

    # ── 프롬프트 준비 ────────────────────────────
    # custom_id 형식: "{논문ID}__A" / "{논문ID}__B"
    prompts_all: dict[str, str] = {}
    keyword_rows: set[str] = set()

    for _, row in df.iterrows():
        pid      = str(row["논문ID"])
        override = check_keyword_override(str(row["제목"]))
        if override:
            keyword_rows.add(pid)
        else:
            prompts_all[f"{pid}__A"] = build_prompt_A(row)
            prompts_all[f"{pid}__B"] = build_prompt_B(row)

    llm_count = CONFIG["SAMPLE_SIZE"] - len(keyword_rows)
    print(f"[키워드 확정] {len(keyword_rows)}편")
    print(f"[LLM 배치 대상] {llm_count}편 × 2 모드 = {llm_count*2}건/모델\n")

    # ── 3개 모델 실행 ────────────────────────────
    print("▶ STEP 1/3 — Gemini 병렬")
    gemini_res, g_in, g_out = run_gemini_parallel(prompts_all)

    print("\n▶ STEP 2/3 — Claude 배치")
    claude_res, c_in, c_out = run_claude_batch(prompts_all)

    print("\n▶ STEP 3/3 — GPT 배치")
    gpt_res, p_in, p_out = run_gpt_batch(prompts_all)

    # ── 결과 DataFrame 조립 ──────────────────────
    for prefix in ["A", "B"]:
        for col in ["claude", "gpt", "gemini"]:
            df[f"{prefix}_{col}"] = None

    for _, row in df.iterrows():
        pid = str(row["논문ID"])
        i   = row.name
        override = check_keyword_override(str(row["제목"]))
        if override:
            for prefix in ["A", "B"]:
                for col in ["claude", "gpt", "gemini"]:
                    df.at[i, f"{prefix}_{col}"] = override
        else:
            for prefix in ["A", "B"]:
                cid = f"{pid}__{prefix}"
                df.at[i, f"{prefix}_claude"]  = claude_res.get(cid, "오류")
                df.at[i, f"{prefix}_gpt"]     = gpt_res.get(cid, "오류")
                df.at[i, f"{prefix}_gemini"]  = gemini_res.get(cid, "오류")

    # ── 결과 테이블 출력 ─────────────────────────
    print("\n" + "=" * 65)
    print("  📋 분류 결과")
    print("=" * 65)
    hdr = f"{'':>3}  {'제목':26}  {'── 모드 A ──':36}  {'── 모드 B ──':36}"
    sub = f"{'':>3}  {'':26}  {'Claude':12}{'GPT':12}{'Gemini':12}  {'Claude':12}{'GPT':12}{'Gemini':12}"
    print(hdr)
    print(sub)
    print("─" * len(sub))
    for _, row in df.iterrows():
        i = row.name
        title = str(row["제목"])[:24]
        kw = check_keyword_override(str(row["제목"]))
        if kw:
            print(f"  {i+1:>2}. {title:<26}  {'[키워드]':<12}{kw}")
            continue
        print(f"  {i+1:>2}. {title:<26}  "
              f"{str(df.at[i,'A_claude']):12}{str(df.at[i,'A_gpt']):12}{str(df.at[i,'A_gemini']):12}  "
              f"{str(df.at[i,'B_claude']):12}{str(df.at[i,'B_gpt']):12}{str(df.at[i,'B_gemini']):12}")

    # ── 결과 저장 ────────────────────────────────
    df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding="utf-8-sig")
    print(f"\n[저장] {CONFIG['OUTPUT_CSV']}")

    # ── 비용 추정 ────────────────────────────────
    print("\n" + "=" * 65)
    print("  💰 예상 비용 (배치 50% 할인 반영, 전체 2,014편 기준)")
    print("=" * 65)

    n = llm_count  # 키워드 확정 제외한 실제 LLM 처리 수

    print("\n  ─ 모드 A (제목만) ─")
    token_A = {
        "claude": (c_in // 2, c_out // 2),   # A/B 반씩
        "gpt":    (p_in // 2, p_out // 2),
        "gemini": (g_in // 2, g_out // 2),
    }
    cost_a = print_cost_section(token_A, total_papers, n)

    print("\n  ─ 모드 B (제목+초록) ─")
    token_B = {
        "claude": (c_in - c_in // 2, c_out - c_out // 2),
        "gpt":    (p_in - p_in // 2, p_out - p_out // 2),
        "gemini": (g_in - g_in // 2, g_out - g_out // 2),
    }
    cost_b = print_cost_section(token_B, total_papers, n)

    print(f"\n  초록 추가 비용: ${cost_b - cost_a:.4f} ≈ "
          f"₩{int((cost_b - cost_a)*1_380):,}  "
          f"(+{(cost_b/cost_a - 1)*100:.0f}%)")

    # ── 일치율 ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  📊 모델 간 분류 일치율")
    print("=" * 65)
    print("\n  모드 A (제목만):")
    print_agreement("A", df)
    print("\n  모드 B (제목+초록):")
    print_agreement("B", df)

    # ── 모드 간 차이 ─────────────────────────────
    print(f"\n  🔄 모드 A vs 모드 B 결과 차이")
    print("  " + "─" * 40)
    for model in ["claude", "gpt", "gemini"]:
        diff = (df[f"A_{model}"] != df[f"B_{model}"]).sum()
        n_all = len(df)
        print(f"    {model.upper():8s}: {diff}/{n_all}건 ({diff/n_all*100:.0f}%) 달라짐")

    diff_mask = (
        (df["A_claude"] != df["B_claude"]) |
        (df["A_gpt"]    != df["B_gpt"])    |
        (df["A_gemini"] != df["B_gemini"])
    )
    if diff_mask.any():
        print(f"\n  ── 모드 간 차이 발생 논문 ({diff_mask.sum()}건) ──")
        for _, r in df[diff_mask].iterrows():
            print(f"    · {str(r['제목'])[:38]}")
            print(f"      Claude : A={r['A_claude']} / B={r['B_claude']}")
            print(f"      GPT    : A={r['A_gpt']} / B={r['B_gpt']}")
            print(f"      Gemini : A={r['A_gemini']} / B={r['B_gemini']}")

    print("\n완료!")


if __name__ == "__main__":
    main()
