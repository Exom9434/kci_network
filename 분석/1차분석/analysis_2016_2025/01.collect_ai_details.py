"""
01.collect_ai_details.py  (analysis_2016_2025 전용 · 하이브리드)
==============================================================
깨끗하게 재구축한 2016~2025 AI 법학 논문 상세·참고문헌을 이 폴더에 따로 수집한다.
(기존 메인 폴더의 옛 데이터/초안과 분리)

경로 규칙
  - 공유 입력(법학목록·안전망·.env)은 부모 폴더(kci_network)에서 읽음
  - 산출물(상세데이터·AI목록·체크포인트)은 이 폴더(analysis_2016_2025)에 저장

방식(하이브리드): 메인 폴더 01과 동일
  후보 = strict필터(전체_법학_논문목록) ∪ 기존 AI목록 ID들(안전망)
       → articleDetail 수집 → 수집 제목+초록에 strict필터 재적용(2016~2025)
  KCI 퍼지검색 누락(진짜 AI ~91편)은 안전망으로 회복, 오탐(AIIB 등)은 재필터로 제거.

실행(이 폴더 안에서):
    cd analysis_2016_2025
    python 01.collect_ai_details.py --limit 20   # 테스트
    python 01.collect_ai_details.py              # 전체
    python 01.collect_ai_details.py --resume
    python 01.collect_ai_details.py --candidates-only
"""
import os, re, time, argparse, shutil
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

BASE   = Path(__file__).parent          # analysis_2016_2025/
PARENT = BASE.parent                    # kci_network/
load_dotenv(PARENT / ".env", override=True)
API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

# ── 공유 입력(부모 폴더) ──────────────────────────────────────
INPUT_FILE   = PARENT / "전체_법학_논문목록.csv"
SAFETY_LISTS = [
    PARENT / "KCI_AI_논문_기본정보_목록_201601_202512.csv",
    PARENT / "AI_법학_논문목록.csv",
]
# ── 산출물(이 폴더) ───────────────────────────────────────────
AI_LIST_OUT = BASE / "KCI_AI_논문_기본정보_목록_재구축.csv"
DETAIL_OUT  = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
CKPT_FILE   = BASE / "01.collect_ai_details.checkpoint.csv"

YEAR_MIN, YEAR_MAX = 2016, 2025
AI_REGEX = re.compile(r"인공지능|(?<![A-Za-z])AI(?![A-Za-z])")

SLEEP_SEC = 0.5
RETRY     = 4
COLS = ["source_id", "title", "pub_year", "abstract",
        "keywords", "category", "target_arti_id", "ref_type"]

if not API_KEY or "여기에" in str(API_KEY):
    raise ValueError("KCI_API_KEY 미설정(부모 폴더 .env 확인)")


def is_ai(text: str) -> bool:
    return bool(AI_REGEX.search(text or ""))


def fetch_detail_and_refs(article_id, title, year):
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": article_id}
    headers = {"User-Agent": "Mozilla/5.0"}
    last = None
    for attempt in range(RETRY):
        try:
            r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
            if r.status_code >= 500:
                last = f"HTTP {r.status_code}"; time.sleep(1.5*(attempt+1)); continue
            r.encoding = "utf-8"; xml = r.text
            if "<total>0</total>" in xml:
                return [], "NoData"

            kw = re.search(r"<keyword-group>(.*?)</keyword-group>", xml, re.DOTALL)
            if kw:
                kws = re.findall(r"<keyword[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</keyword>", kw.group(1))
                keywords = ", ".join(k.strip() for k in kws if k.strip())
            else:
                keywords = ""
            ab = re.search(r'<abstract[^>]*lang=["\']original["\'][^>]*>(.*?)</abstract>', xml, re.DOTALL)
            if not ab:
                ab = re.search(r"<abstract[^>]*>(.*?)</abstract>", xml, re.DOTALL)
            abstract = re.sub(r"<[^>]+>", "", ab.group(1)).strip() if ab else ""
            cat = re.search(r"<article-categories[^>]*>(.*?)</article-categories>", xml, re.DOTALL)
            category = re.sub(r"<[^>]+>", "", cat.group(1)).strip() if cat else ""

            base = {"source_id": article_id, "title": title, "pub_year": year,
                    "abstract": abstract, "keywords": keywords, "category": category}
            refs = re.findall(r"<reference([^>]*)>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</reference>", xml, re.DOTALL)
            if not refs:
                return [{**base, "target_arti_id": "", "ref_type": ""}], "OK_NoRef"
            rows = []
            for attrs, _ in refs:
                aid_m = re.search(r'arti-id="([^"]*)"', attrs)
                typ_m = re.search(r'type-name="([^"]*)"', attrs)
                rows.append({**base,
                             "target_arti_id": aid_m.group(1) if aid_m else "",
                             "ref_type":       typ_m.group(1) if typ_m else ""})
            return rows, "OK"
        except Exception as e:
            last = str(e); time.sleep(1.0)
    return [], f"FAIL:{last}"


def _load(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_candidates():
    frames = []
    if INPUT_FILE.exists():
        d = _load(INPUT_FILE)
        if "초록" not in d.columns: d["초록"] = ""
        yr = pd.to_numeric(d["발행연도"], errors="coerce")
        txt = d["제목"].fillna("") + "  " + d["초록"].fillna("")
        hit = d[yr.between(YEAR_MIN, YEAR_MAX) & txt.map(is_ai)]
        frames.append(hit[["논문ID", "제목", "발행연도"]])
        print(f"  · 전체_법학_논문목록 strict필터: {len(hit):,}편")
    else:
        print(f"  · (경고) {INPUT_FILE} 없음 — 부모 폴더에서 00 수집 먼저 필요")
    for p in SAFETY_LISTS:
        if p.exists():
            d = _load(p)
            sub = d[[c for c in ["논문ID", "제목", "발행연도"] if c in d.columns]].copy()
            frames.append(sub)
            print(f"  · 안전망 {p.name}: {len(sub):,}편 ID 합침")
    cand = pd.concat(frames, ignore_index=True).dropna(subset=["논문ID"]).drop_duplicates("논문ID")
    print(f"  ⇒ 후보 합집합(유일 ID): {len(cand):,}편")
    return cand.reset_index(drop=True)


def post_filter(detail: pd.DataFrame):
    g = detail.groupby("source_id").agg(
        title=("title", "first"), abstract=("abstract", "first"), pub_year=("pub_year", "first"))
    yr = pd.to_numeric(g["pub_year"], errors="coerce")
    txt = g["title"].fillna("") + "  " + g["abstract"].fillna("")
    keep = g[yr.between(YEAR_MIN, YEAR_MAX) & txt.map(is_ai)].index
    return detail[detail["source_id"].isin(keep)].copy(), set(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--candidates-only", action="store_true")
    args = ap.parse_args()

    print(f"[analysis_2016_2025] 공유입력={PARENT.name}, 산출물={BASE.name}\n후보 구성:")
    cand = build_candidates()
    if args.candidates_only:
        cand.to_csv(BASE / "01.candidates.csv", index=False, encoding="utf-8-sig")
        print(f"후보만 저장: 01.candidates.csv ({len(cand):,})"); return

    done = set()
    if args.resume and CKPT_FILE.exists():
        done = set(_load(CKPT_FILE)["source_id"].unique())
        print(f"이어서: {len(done):,}편 완료 → 건너뜀")

    todo = cand[~cand["논문ID"].astype(str).isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"\n상세·참고문헌 수집 대상: {len(todo):,}편\n")
    if len(todo) == 0 and not done:
        print("대상 없음."); return

    header = not (args.resume and CKPT_FILE.exists())
    buf, ok, noref, fail = [], 0, 0, 0
    for row in tqdm(todo.itertuples(index=False), total=len(todo), desc="상세수집"):
        aid = str(row.논문ID)
        title = getattr(row, "제목", "") if "제목" in cand.columns else ""
        year  = getattr(row, "발행연도", "") if "발행연도" in cand.columns else ""
        rows, status = fetch_detail_and_refs(aid, title, year)
        if rows:
            buf.extend(rows); ok += (status == "OK"); noref += (status == "OK_NoRef")
        else:
            buf.append({**{c: "" for c in COLS}, "source_id": aid, "title": title, "pub_year": year})
            fail += 1; tqdm.write(f"  ⚠ {aid}: {status}")
        if len(buf) >= 200:
            pd.DataFrame(buf)[COLS].to_csv(CKPT_FILE, mode="a", index=False, header=header, encoding="utf-8-sig")
            header, buf = False, []
        time.sleep(SLEEP_SEC)
    if buf:
        pd.DataFrame(buf)[COLS].to_csv(CKPT_FILE, mode="a", index=False, header=header, encoding="utf-8-sig")
    print(f"\n수집 완료 | 참고문헌있음 {ok} / 없음 {noref} / 실패 {fail}")

    raw = _load(CKPT_FILE).drop_duplicates()
    filtered, kept = post_filter(raw)
    dropped = raw["source_id"].nunique() - len(kept)
    print(f"엄격 재필터: 유지 {len(kept):,}편 / 오탐·범위밖 제거 {dropped:,}편")

    filtered.drop_duplicates("source_id")[["source_id", "title", "pub_year"]].to_csv(
        AI_LIST_OUT, index=False, encoding="utf-8-sig")
    if DETAIL_OUT.exists():
        shutil.copy(DETAIL_OUT, DETAIL_OUT.with_suffix(".csv.bak"))
        print("기존 산출물 백업(.bak)")
    filtered.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ 최종: {DETAIL_OUT}")
    print(f"   전체 {len(filtered):,}행 / 고유 논문 {filtered['source_id'].nunique():,}편")
    print(pd.to_numeric(filtered.drop_duplicates('source_id')['pub_year'], errors='coerce')
          .value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
