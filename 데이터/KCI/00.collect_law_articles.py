"""
00.collect_law_articles.py  (cell 1 보정판 · v2 '159목록 멤버십' 방식)
====================================================================
KCI articleSearch 의 journal= 는 '필터'가 아니라 '관련도 순위 검색'이다.
  - "IT와 법연구"로 검색하면 앞 몇 페이지만 그 학술지 논문이고,
    뒤 페이지는 다른 '법학 학술지' 논문이 관련도순으로 딸려온다(전부 159목록 안).
  - <total>(예:21,711)은 전체 코퍼스 수치라 페이지 종료 기준으로 못 씀.

[v1의 문제] '검색어와 정확히 같은 학술지'만 취하고 연속 2페이지 0매칭에 종료 →
  공통 토큰 학술지(법학연구·공법연구…)의 흩어진 자기 논문을 놓쳐 248편 누락.

[v2 전략]
  1) 반환 record 의 '실제 journal-name' 이 159목록(VALID) 안이면 전부 수집(올바른 라벨)
  2) 전역(seen) 중복제거 → A학술지 논문이 B검색의 bleed로 잡혀도 채워짐
  3) 한 쿼리에서 '새 논문ID가 STREAK 페이지 연속 0' 이면 종료(합집합이 서로 메꿈)
  4) MAX_PAGE 상한 + 500 재시도 + 학술지 단위 append(--resume)

산출물: 전체_법학_논문목록.csv (학술지명=실제값, 초록 포함, 논문ID 유일)
실행:
    python 00.collect_law_articles.py                  # 전체
    python 00.collect_law_articles.py --limit 5        # 테스트(쿼리 5개)
    python 00.collect_law_articles.py --resume
    python 00.collect_law_articles.py --streak 15 --maxpage 150   # 더 촘촘히
"""
import os, re, time, argparse
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(next(p for p in Path(__file__).resolve().parents if (p / ".env").exists()) / ".env", override=True)
API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

INPUT_FILE  = BASE / "법학 기관 목록_2026-07-23.csv"
OUTPUT_FILE = BASE / "전체_법학_논문목록.csv"
CKPT_FILE   = BASE / "전체_법학_논문목록.checkpoint.csv"
CKPT_TEST   = BASE / "전체_법학_논문목록.checkpoint.test.csv"  # --limit 전용(격리)

DISPLAY   = 100
SLEEP_SEC = 0.25
RETRY     = 4
COLS = ["학술지명", "논문ID", "제목", "저자", "초록", "발행연도", "KCI_URL"]

if not API_KEY or "여기에" in str(API_KEY):
    raise ValueError("KCI_API_KEY 미설정(.env 확인)")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def fetch_page(journal, page, valid):
    """(status, records[159목록 안만], total, n_returned)"""
    params = {"apiCode": "articleSearch", "key": API_KEY,
              "journal": journal, "displayCount": DISPLAY, "page": page}
    last = None
    for attempt in range(RETRY):
        try:
            r = requests.get(BASE_URL, params=params,
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"; time.sleep(1.5*(attempt+1)); continue
            r.raise_for_status()
            root = ET.fromstring(r.content)
            msg = root.find(".//{*}resultMsg")
            if msg is not None and msg.text and "정상" not in msg.text:
                return ("API_MSG:" + msg.text, [], 0, 0)
            tn = root.find(".//{*}total")
            total = int(tn.text) if tn is not None and tn.text else 0

            out, n = [], 0
            for rec in root.findall(".//{*}record"):
                ai = rec.find("{*}articleInfo"); ji = rec.find("{*}journalInfo")
                if ai is None:
                    continue
                n += 1
                real = norm(ji.findtext("{*}journal-name", default="")) if ji is not None else ""
                if real not in valid:                # ← 핵심: 159목록 멤버십
                    continue
                abstract = ""
                ab_nodes = ai.findall(".//{*}abstract")
                for ab in ab_nodes:
                    if ab.get("lang") == "original" and ab.text:
                        abstract = ab.text.strip(); break
                if not abstract and ab_nodes and ab_nodes[0].text:
                    abstract = ab_nodes[0].text.strip()
                out.append({
                    "학술지명": real,
                    "논문ID":  ai.get("article-id", ""),
                    "제목":    (ai.findtext(".//{*}article-title", default="") or "").strip(),
                    "저자":    ", ".join(a.text for a in ai.findall(".//{*}author") if a.text),
                    "초록":    abstract,
                    "발행연도": (ji.findtext("{*}pub-year", default="") if ji is not None else ""),
                    "KCI_URL": ai.findtext("{*}url", default=""),
                })
            return ("OK", out, total, n)
        except ET.ParseError as e:
            last = f"ParseError:{e}"; time.sleep(1.0)
        except Exception as e:
            last = str(e); time.sleep(1.0)
    return (f"FAIL:{last}", [], 0, 0)


def collect_query(journal, valid, seen, streak_max, max_page):
    rows, page, streak = [], 1, 0
    while page <= max_page:
        status, recs, total, n = fetch_page(journal, page, valid)
        if status != "OK":
            tqdm.write(f"  ⚠ [{journal}] p{page}: {status}"); break
        if n == 0:
            break
        new = [r for r in recs if r["논문ID"] and r["논문ID"] not in seen]
        for r in new:
            seen.add(r["논문ID"]); rows.append(r)
        if len(new) == 0:                 # 새 논문 없음(다른 쿼리서 이미 봤거나 소진)
            streak += 1
            if streak >= streak_max:
                break
        else:
            streak = 0
        if page * DISPLAY >= total:
            break
        page += 1
        time.sleep(SLEEP_SEC)
    return rows, page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--streak", type=int, default=10, help="새 논문 0인 페이지 연속 N번이면 종료")
    ap.add_argument("--maxpage", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    valid = set(norm(x) for x in df["학술지한글명"].dropna() if norm(x))
    queries = sorted(valid)
    print(f"159목록 학술지 {len(valid)}종 (검색 시드 {len(queries)}개)")

    is_test = bool(args.limit)
    ckpt = CKPT_TEST if is_test else CKPT_FILE

    # 비-resume(또는 테스트)면 이전 체크포인트를 새로 시작(누적 방지)
    if (not args.resume or is_test) and ckpt.exists():
        ckpt.unlink()
        print(f"이전 체크포인트 초기화: {ckpt.name}")

    done = set(); seen = set()
    if args.resume and not is_test and ckpt.exists():
        prev = pd.read_csv(ckpt, encoding="utf-8-sig", dtype=str)
        done = set(prev["_query"].dropna().unique()) if "_query" in prev else set()
        seen = set(prev["논문ID"].dropna())
        print(f"이어서: 쿼리 {len(done)}개 완료, 기수집 {len(seen):,}편")

    todo = [q for q in queries if q not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"이번 실행 쿼리: {len(todo)}개  (streak={args.streak}, maxpage={args.maxpage})"
          + ("  [테스트: 본파일 미갱신]" if is_test else "") + "\n")

    header = not ckpt.exists()
    ok_pages = 0
    for q in tqdm(todo, desc="쿼리"):
        rows, pages = collect_query(q, valid, seen, args.streak, args.maxpage)
        ok_pages += max(pages - 1, 0)
        df_q = pd.DataFrame(rows, columns=COLS); df_q["_query"] = q
        df_q.to_csv(ckpt, mode="a", index=False, header=header, encoding="utf-8-sig")
        header = False
        tqdm.write(f"  [{q}] 신규 {len(rows)}편 (p{pages-1}) | 누적 {len(seen):,}")

    if ok_pages == 0:
        print("\n❌ 유효 페이지 0 — 모든 요청이 실패(HTTP 500 등). API 상태를 확인하고 재시도하세요.")
        print("   (probe_journal_match.py 로 1~2건 호출이 되는지부터 점검)")
        return

    if is_test:
        cur = pd.read_csv(ckpt, encoding="utf-8-sig", dtype=str).drop_duplicates("논문ID")
        print(f"\n🧪 테스트 완료(본파일 미갱신): 이번 {len(cur):,}편 수집")
        print(pd.to_numeric(cur['발행연도'],errors='coerce').value_counts().sort_index().to_string())
        print("정상으로 보이면 --limit 없이 전체 실행하세요.")
        return

    final = pd.read_csv(ckpt, encoding="utf-8-sig", dtype=str)
    final = final.drop(columns=[c for c in ["_query"] if c in final.columns])
    final = final.drop_duplicates("논문ID")
    final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료: {OUTPUT_FILE.name}")
    print(f"   전체 {len(final):,}편(유일 논문ID) / 학술지 {final['학술지명'].nunique()}종")
    print(f"   연도분포:")
    print(pd.to_numeric(final['발행연도'],errors='coerce').value_counts().sort_index().to_string())
    print(f"   체크포인트({ckpt.name})는 확인 후 삭제")


if __name__ == "__main__":
    main()
