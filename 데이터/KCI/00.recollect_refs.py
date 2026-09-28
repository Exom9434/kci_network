"""
00.recollect_refs.py
====================
KCI 2025 인용지수 공개 반영 → 참고문헌이 비어있는 논문 재수집(덮어쓰기)

배경:
- 2025년 논문 457편은 3월 수집 당시 KCI가 참고문헌을 아직 파싱하지 않아
  전부 '참고문헌 0개' 빈 행으로만 저장돼 있음(target_arti_id 0% 매칭).
- 인용지수 공개 후 재수집하면 참고문헌이 채워짐.

collect_missing.py 와의 차이:
- collect_missing 은 "상세데이터에 아예 없는 논문"만 받아옴 → 2025는 이미 있어서 0건.
- 이 스크립트는 "이미 있지만 참고문헌이 비어있는 논문"을 재수집해 기존 행을 교체함.

동작:
1) 상세데이터에서 참고문헌 0개인 source_id 추출 (기본: 전체 / --year 로 특정연도만)
2) 원본 백업(.bak) 생성
3) 대상 논문을 articleDetail API로 재수집
4) 대상 source_id의 기존 행 제거 후 새 행으로 교체하여 전체 재작성 (append 아님 → BOM/중복 없음)
5) 중간 체크포인트 저장(.checkpoint.csv)로 중단 시 이어서 실행 가능

실행:
    python 00.recollect_refs.py                # 참고문헌 0개인 논문 전체(599편)
    python 00.recollect_refs.py --year 2025    # 2025년만(457편)
    python 00.recollect_refs.py --limit 20     # 테스트용 20편만
    python 00.recollect_refs.py --resume       # 중단분 이어서
"""
import os, re, time, argparse, shutil
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# ── 설정 ────────────────────────────────────────────────────────
BASE = Path(__file__).parent
load_dotenv(next(p for p in Path(__file__).resolve().parents if (p / ".env").exists()) / ".env", override=True)
API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

DETAIL_FILE = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
CKPT_FILE   = BASE / "00.recollect_refs.checkpoint.csv"
SLEEP_SEC   = 0.7
COLS = ["source_id", "title", "pub_year", "abstract",
        "keywords", "category", "target_arti_id", "ref_type"]

if not API_KEY or "여기에" in str(API_KEY):
    raise ValueError("KCI_API_KEY가 .env에 설정되지 않았습니다.")


# ── 참고문헌 유무 판정 ───────────────────────────────────────────
def paper_has_ref(sub: pd.DataFrame) -> bool:
    t = sub["target_arti_id"].astype(str).str.strip()
    return ((t != "") & (t != "nan")).any()


# ── API 수집 함수 (collect_missing.py 검증본 재사용) ─────────────
def fetch_detail_and_refs(article_id, title, year):
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": article_id}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        resp.encoding = "utf-8"
        xml = resp.text

        if "<total>0</total>" in xml:
            return [], "NoData"

        kw_group = re.search(r"<keyword-group>(.*?)</keyword-group>", xml, re.DOTALL)
        if kw_group:
            kws = re.findall(r"<keyword[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</keyword>",
                             kw_group.group(1))
            keywords = ", ".join(k.strip() for k in kws if k.strip())
        else:
            keywords = ""

        ab_match = re.search(
            r'<abstract[^>]*lang=["\']original["\'][^>]*>(.*?)</abstract>', xml, re.DOTALL)
        if not ab_match:
            ab_match = re.search(r"<abstract[^>]*>(.*?)</abstract>", xml, re.DOTALL)
        abstract = re.sub(r"<[^>]+>", "", ab_match.group(1)).strip() if ab_match else ""

        cat_match = re.search(r"<article-categories[^>]*>(.*?)</article-categories>", xml, re.DOTALL)
        category  = re.sub(r"<[^>]+>", "", cat_match.group(1)).strip() if cat_match else ""

        base = {"source_id": article_id, "title": title, "pub_year": year,
                "abstract": abstract, "keywords": keywords, "category": category}

        ref_matches = re.findall(
            r"<reference([^>]*)>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</reference>", xml, re.DOTALL)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="특정 발행연도만 재수집")
    ap.add_argument("--limit", type=int, default=None, help="테스트용 상위 N편만")
    ap.add_argument("--resume", action="store_true", help="체크포인트 이어서 실행")
    args = ap.parse_args()

    # ── 대상 선정 ─────────────────────────────────────────────
    df = pd.read_csv(DETAIL_FILE, encoding="utf-8-sig", dtype=str, low_memory=False)
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce").astype("Int64")

    meta = (df.sort_values("source_id")
              .groupby("source_id")
              .agg(title=("title", "first"),
                   pub_year=("pub_year", "first")))
    has_ref = df.groupby("source_id").apply(paper_has_ref)
    targets = meta[~meta.index.map(has_ref)].reset_index()

    if args.year is not None:
        targets = targets[targets["pub_year"] == args.year]

    # ── 체크포인트 이어서 ─────────────────────────────────────
    done_ids = set()
    if args.resume and CKPT_FILE.exists():
        prev = pd.read_csv(CKPT_FILE, encoding="utf-8-sig", dtype=str)
        done_ids = set(prev["source_id"].astype(str))
        print(f"체크포인트에서 {len(done_ids)}편 이미 완료 → 건너뜀")

    targets = targets[~targets["source_id"].isin(done_ids)]
    if args.limit:
        targets = targets.head(args.limit)

    print(f"재수집 대상: {len(targets)}편")
    if args.year is None:
        print(targets["pub_year"].value_counts().sort_index().to_string())
    if len(targets) == 0:
        print("대상 없음. 종료.")
        return
    print()

    # ── 수집 루프 (체크포인트에 append) ───────────────────────
    ckpt_header = not (args.resume and CKPT_FILE.exists())
    buffer, ok, noref, fail = [], 0, 0, 0
    for row in tqdm(targets.itertuples(index=False), total=len(targets)):
        aid, title, year = str(row.source_id), row.title, row.pub_year
        rows, status = fetch_detail_and_refs(aid, title, year)
        if rows:
            buffer.extend(rows)
            if status == "OK": ok += 1
            else: noref += 1
        else:
            # 그래도 논문 자체는 유지되도록 빈 행 1개 기록
            buffer.append({**{c: "" for c in COLS},
                           "source_id": aid, "title": title, "pub_year": year})
            fail += 1
            tqdm.write(f"  ⚠ {aid}: {status}")

        if len(buffer) >= 300:
            pd.DataFrame(buffer)[COLS].to_csv(
                CKPT_FILE, mode="a", index=False, header=ckpt_header, encoding="utf-8-sig")
            ckpt_header, buffer = False, []
        time.sleep(SLEEP_SEC)

    if buffer:
        pd.DataFrame(buffer)[COLS].to_csv(
            CKPT_FILE, mode="a", index=False, header=ckpt_header, encoding="utf-8-sig")

    print(f"\n수집 완료 | 참고문헌있음 {ok} / 참고문헌없음 {noref} / 실패 {fail}")

    # ── 병합: 대상 논문의 기존 행 제거 후 신규 행으로 교체 ────
    new = pd.read_csv(CKPT_FILE, encoding="utf-8-sig", dtype=str)
    replaced_ids = set(new["source_id"].astype(str))

    backup = DETAIL_FILE.with_suffix(".csv.bak")
    shutil.copy(DETAIL_FILE, backup)
    print(f"원본 백업: {backup.name}")

    kept = df[~df["source_id"].astype(str).isin(replaced_ids)].copy()
    kept["pub_year"] = kept["pub_year"].astype(str)
    merged = pd.concat([kept[COLS], new[COLS]], ignore_index=True)
    # 전체 재작성(append 아님) → 중간 BOM/중복 없음
    merged.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")

    print(f"\n최종 저장: {DETAIL_FILE.name}")
    print(f"  전체 행: {len(merged):,} / 고유 논문: {merged['source_id'].nunique():,}")
    print(f"  교체된 논문: {len(replaced_ids):,}편")
    print(f"\n체크포인트({CKPT_FILE.name})는 확인 후 삭제하세요.")


if __name__ == "__main__":
    main()
