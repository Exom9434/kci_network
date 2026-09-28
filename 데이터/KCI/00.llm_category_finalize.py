"""
KCI AI 법학 논문 카테고리 LLM 분류 - 배치 처리 버전
=============================================================
- 입력: KCI_AI_논문_기본정보_목록_201601_202512.csv (2,014편)
- 입력 모드: 제목 + 초록
- 확정 기준:
    3/3 일치 → 해당 카테고리 확정
    2:1 일치 → 융복합(A+B) 확정 (프롬프트 순서대로 표기)
    3개 전부 다름 → 수동 검토

[배치 처리 방식]
  Claude  → Anthropic Message Batches API  (50% 할인)
  GPT     → OpenAI Batch API               (50% 할인)
  Gemini  → ThreadPoolExecutor 병렬 처리   (배치 API 미지원)

[출력 파일]
  KCI_AI_논문_카테고리_분류결과.csv   전체 원시 결과 (모델별 응답)
  KCI_AI_논문_카테고리_확정.csv       3/3 일치 확정 + 2:1 융복합 확정
  KCI_AI_논문_카테고리_검토필요.csv   3개 전부 다름 또는 오류 → 수동 검토

[이어하기]
  KCI_AI_분류_배치상태.json 에 배치 ID 저장
  재실행 시 이미 제출된 배치는 재사용, 완료된 결과는 스킵

사용법:
    python 00.llm_category_finalize.py
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

    "INPUT_CSV":      "KCI_AI_논문_기본정보_목록_201601_202512.csv",
    "OUTPUT_RAW":     "KCI_AI_논문_카테고리_분류결과.csv",
    "OUTPUT_OK":      "KCI_AI_논문_카테고리_확정.csv",
    "OUTPUT_REVIEW":  "KCI_AI_논문_카테고리_검토필요.csv",
    "STATUS_FILE":    "KCI_AI_분류_배치상태.json",   # 배치 ID 저장

    "GEMINI_WORKERS":    10,     # Gemini 병렬 호출 수
    "RETRY_COUNT":        2,
    "RETRY_DELAY":        3,
    "BATCH_POLL_SEC":    30,     # 배치 완료 폴링 간격 (초)

    "ABSTRACT_MAX_CHARS": 99999,  # 사실상 무제한
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
# 📝 공통 유틸
# ─────────────────────────────────────────────

def build_prompt(row: pd.Series) -> str:
    abstract = str(row["초록"]) if pd.notna(row.get("초록")) else ""
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


def load_status() -> dict:
    p = Path(CONFIG["STATUS_FILE"])
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def save_status(status: dict):
    Path(CONFIG["STATUS_FILE"]).write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────
# 🤖 Claude — Anthropic Message Batches API
# ─────────────────────────────────────────────

def run_claude_batch(prompts: dict[str, str], status: dict) -> dict[str, str]:
    """
    prompts: {논문ID: prompt_text}
    반환:    {논문ID: 카테고리}
    """
    import anthropic
    client = anthropic.Anthropic(api_key=CONFIG["ANTHROPIC_API_KEY"])

    # ── 배치 제출 (또는 기존 ID 재사용) ────────────
    batch_id = status.get("anthropic_batch_id")
    if not batch_id:
        print(f"  [Claude 배치] {len(prompts)}건 제출 중...")
        batch = client.messages.batches.create(
            requests=[
                {
                    "custom_id": paper_id,
                    "params": {
                        "model": CONFIG["CLAUDE_MODEL"],
                        "max_tokens": 20,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
                for paper_id, prompt in prompts.items()
            ]
        )
        batch_id = batch.id
        status["anthropic_batch_id"] = batch_id
        save_status(status)
        print(f"  [Claude 배치] 제출 완료 — ID: {batch_id}")
    else:
        print(f"  [Claude 배치] 기존 배치 재사용 — ID: {batch_id}")

    # ── 완료 대기 ───────────────────────────────
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [Claude 배치] 상태: {batch.processing_status} "
              f"(완료 {counts.succeeded}/{counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired})")
        if batch.processing_status == "ended":
            break
        time.sleep(CONFIG["BATCH_POLL_SEC"])

    # ── 결과 수집 ───────────────────────────────
    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            results[result.custom_id] = validate_category(
                result.result.message.content[0].text
            )
        else:
            results[result.custom_id] = "오류"

    print(f"  [Claude 배치] 결과 수집 완료 ({len(results)}건)")
    return results


# ─────────────────────────────────────────────
# 🤖 GPT — OpenAI Batch API
# ─────────────────────────────────────────────

def run_gpt_batch(prompts: dict[str, str], status: dict) -> dict[str, str]:
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

    # ── 배치 제출 (또는 기존 ID 재사용) ────────────
    batch_id = status.get("openai_batch_id")
    if not batch_id:
        print(f"  [GPT 배치] {len(prompts)}건 제출 중...")

        # JSONL 생성 및 업로드
        jsonl = "\n".join(
            json.dumps({
                "custom_id": paper_id,
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
            for paper_id, prompt in prompts.items()
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
        batch_id = batch.id
        status["openai_batch_id"] = batch_id
        save_status(status)
        print(f"  [GPT 배치] 제출 완료 — ID: {batch_id}")
    else:
        print(f"  [GPT 배치] 기존 배치 재사용 — ID: {batch_id}")

    # ── 완료 대기 ───────────────────────────────
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  [GPT 배치] 상태: {batch.status} "
              f"(완료 {batch.request_counts.completed}/{batch.request_counts.total})")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(CONFIG["BATCH_POLL_SEC"])

    if batch.status != "completed":
        print(f"  [GPT 배치] 경고: 배치가 {batch.status} 상태로 종료")
        return {}

    # ── 결과 수집 ───────────────────────────────
    output = client.files.content(batch.output_file_id)
    results = {}
    for line in output.text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        try:
            text = r["response"]["body"]["choices"][0]["message"]["content"]
            results[r["custom_id"]] = validate_category(text)
        except Exception:
            results[r["custom_id"]] = "오류"

    print(f"  [GPT 배치] 결과 수집 완료 ({len(results)}건)")
    return results


# ─────────────────────────────────────────────
# 🤖 Gemini — ThreadPoolExecutor 병렬 처리
# ─────────────────────────────────────────────

def _call_gemini_once(paper_id: str, prompt: str) -> tuple[str, str]:
    """(논문ID, 카테고리) 반환"""
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
            return paper_id, validate_category(resp.text)
        except Exception as e:
            if attempt < CONFIG["RETRY_COUNT"] - 1:
                time.sleep(CONFIG["RETRY_DELAY"])
            else:
                return paper_id, "오류"


def run_gemini_parallel(prompts: dict[str, str], status: dict) -> dict[str, str]:
    # 이미 완료된 결과가 있으면 재사용
    if status.get("gemini_done"):
        cached = status.get("gemini_results", {})
        print(f"  [Gemini] 기존 결과 재사용 ({len(cached)}건)")
        return cached

    print(f"  [Gemini] {len(prompts)}건 병렬 처리 중 (workers={CONFIG['GEMINI_WORKERS']})...")
    results = {}
    done = 0
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=CONFIG["GEMINI_WORKERS"]) as executor:
        futures = {
            executor.submit(_call_gemini_once, pid, prompt): pid
            for pid, prompt in prompts.items()
        }
        for future in as_completed(futures):
            paper_id, category = future.result()
            results[paper_id] = category
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  [Gemini] {done}/{total} 완료...")

    # 결과 캐싱
    status["gemini_results"] = results
    status["gemini_done"] = True
    save_status(status)
    print(f"  [Gemini] 완료 ({len(results)}건)")
    return results


# ─────────────────────────────────────────────
# 🔍 판정 로직: 3/3 확정 / 2:1 융복합 / 3개 상이
# ─────────────────────────────────────────────

RESULT_COLS = ["카테고리_claude", "카테고리_gpt", "카테고리_gemini"]
ERROR_VALS  = {"오류", "분류불가", ""}

# CATEGORIES 순서 인덱스 (융복합 표기 순서 결정용)
_CAT_ORDER = {cat: i for i, cat in enumerate(CATEGORIES)}

def get_final_category(row) -> str:
    """
    3/3 일치 → 해당 카테고리
    2:1 일치 → 융복합(다수+소수) — CATEGORIES 순서대로
    3개 전부 다름 또는 오류 포함 → "" (수동 검토)
    """
    from collections import Counter
    votes = [str(row.get(c, "")).strip() for c in RESULT_COLS]

    # 오류값이 하나라도 있으면 수동 검토
    if any(v in ERROR_VALS for v in votes):
        return ""

    unique = set(votes)

    # 3/3 완전 일치
    if len(unique) == 1:
        return votes[0]

    # 2:1 (unique 2개 = 한 값이 2번, 다른 값이 1번)
    if len(unique) == 2:
        counts = Counter(votes)
        # CATEGORIES 순서 기준으로 정렬
        pair = sorted(unique, key=lambda x: _CAT_ORDER.get(x, 9999))
        return f"융복합({pair[0]}+{pair[1]})"

    # 3개 전부 다름
    return ""


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  KCI AI 법학 논문 카테고리 LLM 분류 — 배치 처리")
    print("  Claude·GPT: Batch API(50%↓) | Gemini: 병렬 처리")
    print("=" * 65)

    # API 키 검증
    missing = [k for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
               if not CONFIG.get(k)]
    if missing:
        print(f"[오류] .env에 다음 키가 없습니다: {', '.join(missing)}")
        sys.exit(1)
    print("[.env] API 키 로드 완료\n")

    # ── 데이터 로드 ───────────────────────────────
    df = pd.read_csv(CONFIG["INPUT_CSV"], encoding="utf-8-sig")
    print(f"[입력] {len(df)}편\n")

    # ── 상태 파일 로드 ────────────────────────────
    status = load_status()
    if status:
        print(f"[이어하기] 기존 배치 상태 파일 발견: {CONFIG['STATUS_FILE']}")

    # ── 키워드 예외 처리 ──────────────────────────
    keyword_results = {}
    llm_rows = []
    for _, row in df.iterrows():
        override = check_keyword_override(str(row["제목"]))
        if override:
            keyword_results[str(row["논문ID"])] = override
        else:
            llm_rows.append(row)

    print(f"[키워드 확정] {len(keyword_results)}편")
    print(f"[LLM 분류 대상] {len(llm_rows)}편\n")

    # ── 프롬프트 준비 ─────────────────────────────
    prompts = {str(row["논문ID"]): build_prompt(row) for row in llm_rows}

    # ── 3개 모델 실행 ─────────────────────────────
    print("▶ STEP 1/3 — Gemini 병렬 처리")
    gemini_results = run_gemini_parallel(prompts, status)

    print("\n▶ STEP 2/3 — Claude 배치 제출 & 대기")
    claude_results = run_claude_batch(prompts, status)

    print("\n▶ STEP 3/3 — GPT 배치 제출 & 대기")
    gpt_results = run_gpt_batch(prompts, status)

    # ── 결과 병합 ────────────────────────────────
    print("\n▶ 결과 병합 중...")
    rows_out = []
    for _, row in df.iterrows():
        pid = str(row["논문ID"])
        new_row = row.to_dict()

        if pid in keyword_results:
            cat = keyword_results[pid]
            new_row["카테고리_claude"]  = cat
            new_row["카테고리_gpt"]    = cat
            new_row["카테고리_gemini"] = cat
        else:
            new_row["카테고리_claude"]  = claude_results.get(pid, "오류")
            new_row["카테고리_gpt"]    = gpt_results.get(pid, "오류")
            new_row["카테고리_gemini"] = gemini_results.get(pid, "오류")

        rows_out.append(new_row)

    df_raw = pd.DataFrame(rows_out)
    df_raw.to_csv(CONFIG["OUTPUT_RAW"], index=False, encoding="utf-8-sig")
    print(f"[저장] {CONFIG['OUTPUT_RAW']}")

    # ── 판정 ─────────────────────────────────────
    df_raw["최종_카테고리"] = df_raw.apply(get_final_category, axis=1)
    total       = len(df_raw)
    confirmed   = (df_raw["최종_카테고리"] != "").sum()
    triple_ok   = df_raw["최종_카테고리"].apply(
                      lambda v: v != "" and not v.startswith("융복합")).sum()
    hybrid_ok   = df_raw["최종_카테고리"].apply(
                      lambda v: v.startswith("융복합")).sum()
    need_review = total - confirmed

    # ── 확정 파일 (3/3 + 2:1 융복합) ────────────
    df_ok = df_raw[df_raw["최종_카테고리"] != ""].copy()
    df_ok.to_csv(CONFIG["OUTPUT_OK"], index=False, encoding="utf-8-sig")

    # ── 검토 필요 파일 (3개 전부 다름 또는 오류) ─
    df_review = df_raw[df_raw["최종_카테고리"] == ""].copy()

    def review_note(row) -> str:
        votes = {c: str(row.get(c, "")).strip() for c in RESULT_COLS}
        err_models = [c.replace("카테고리_", "") for c, v in votes.items()
                      if v in ERROR_VALS]
        if err_models:
            return f"오류/분류불가: {', '.join(err_models)}"
        return (f"3개 불일치: {votes['카테고리_claude']} / "
                f"{votes['카테고리_gpt']} / {votes['카테고리_gemini']}")

    df_review = df_review.copy()
    df_review["검토_메모"] = df_review.apply(review_note, axis=1)
    df_review["최종_카테고리"] = ""

    review_cols = ["논문ID", "제목", "발행연도", "초록",
                   "카테고리_claude", "카테고리_gpt", "카테고리_gemini",
                   "검토_메모", "최종_카테고리"]
    review_cols = [c for c in review_cols if c in df_review.columns]
    df_review[review_cols].to_csv(CONFIG["OUTPUT_REVIEW"], index=False, encoding="utf-8-sig")

    # ── 상태 파일 정리 ────────────────────────────
    Path(CONFIG["STATUS_FILE"]).unlink(missing_ok=True)
    print(f"[정리] {CONFIG['STATUS_FILE']} 삭제")

    # ── 결과 요약 ────────────────────────────────
    print("\n" + "=" * 65)
    print("  📊 분류 결과 요약")
    print("=" * 65)
    print(f"  전체 논문:              {total:>5}편")
    print(f"  ✅ 3/3 일치 확정:       {triple_ok:>5}편 ({triple_ok/total*100:.1f}%)")
    print(f"  🔀 2:1 융복합 확정:     {hybrid_ok:>5}편 ({hybrid_ok/total*100:.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"     소계 확정:           {confirmed:>5}편 ({confirmed/total*100:.1f}%)")
    print(f"  ⚠  수동 검토 필요:      {need_review:>5}편 ({need_review/total*100:.1f}%)")

    print(f"\n  [확정 카테고리 분포]")
    cat_dist = df_ok["최종_카테고리"].value_counts()
    max_cnt = max(cat_dist) if len(cat_dist) else 1
    for cat, cnt in cat_dist.items():
        bar = "█" * (cnt * 30 // max_cnt)
        print(f"    {cat:22s}  {cnt:4d}편  {bar}")

    if need_review > 0:
        print(f"\n  [검토 필요 샘플 (최대 10건)]")
        for _, r in df_review.head(10).iterrows():
            print(f"    · {str(r['제목'])[:40]}")
            print(f"      {r['검토_메모']}")

    print(f"\n  저장 파일:")
    print(f"    {CONFIG['OUTPUT_OK']}   ← 확정 ({confirmed}편: 3/3 {triple_ok}편 + 융복합 {hybrid_ok}편)")
    print(f"    {CONFIG['OUTPUT_REVIEW']}  ← 검토 필요 ({need_review}편)")
    print("=" * 65)
    print("완료!")


if __name__ == "__main__":
    main()
