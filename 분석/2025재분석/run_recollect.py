"""
run_recollect.py  (2025인용_재수집 폴더 전용 · 00.recollect_refs.py 파생본)
=========================================================================
KCI 2025 인용지수 공개 반영 → 참고문헌 0개 논문을 다시 받아 채운다.
★ 원본은 절대 건드리지 않는다. 입력은 읽기 전용, 결과는 이 폴더에만 쓴다.

원본 대비 바뀐 점:
- DETAIL_IN  : 부모 폴더의 원본 CSV = 읽기 전용 입력
- OUT_FILE   : 이 폴더의 새 CSV = 참조가 채워진 병합 결과(원본 미변경)
- CKPT_FILE  : 이 폴더의 체크포인트
- 백업 대신 '원본 무변경' 방식이므로 .bak를 원본 폴더에 만들지 않음

동작:
1) 원본에서 참고문헌 0개인 source_id 추출 (기본: --year 2025 / 생략 시 전체 599편)
2) 대상 논문을 articleDetail API로 재수집 → 체크포인트에 append
3) 원본의 해당 논문 행을 제거하고 새 행으로 교체 → OUT_FILE로 '전체 재작성'
   (원본 파일은 그대로. OUT_FILE만 갱신된 완본이 됨)

실행:
    cd 2025인용_재수집
    python run_recollect.py --year 2025          # 2025년만(457편) ← 권장
    python run_recollect.py                       # 참고문헌 0개 전체(599편)
    python run_recollect.py --year 2025 --limit 20   # 테스트 20편
    python run_recollect.py --year 2025 --resume     # 중단분 이어서
"""
import os, re, time, argparse, requests, pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

HERE   = Path(__file__).parent          # .../2025인용_재수집
PARENT = HERE.parents[1]                # .../kci_network (분석/2025재분석 기준)
load_dotenv(PARENT / ".env")

API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

DETAIL_IN = PARENT / "데이터" / "KCI" / "00.KCI_AI_논문_상세_및_인용데이터.csv"   # 원본(읽기 전용)
OUT_FILE  = HERE   / "00.KCI_AI_논문_상세_및_인용데이터.updated.csv"  # 갱신 완본(이 폴더)
CKPT_FILE = HERE   / "recollect.checkpoint.csv"

SLEEP_SEC = 0.7
COLS = ["source_id", "title", "pub_year", "abstract",
        "keywords", "category", "target_arti_id", "ref_type"]

if not API_KEY or "여기에" in str(API_KEY):
    raise ValueError("KCI_API_KEY 미설정(../.env 확인)")


def paper_has_ref(sub: pd.DataFrame) -> bool:
    t = sub["target_arti_id"].astype(str).str.strip()
    return ((t != "") & (t != "nan")).any()


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

        ab = re.search(r'<abstract[^>]*lang=["\']original["\'][^>]*>(.*?)</abstract>', xml, re.DOTALL)
        if not ab:
            ab = re.search(r"<abstract[^>]*>(.*?)</abstract>", xml, re.DOTALL)
        # CDATA 언랩(원본 스크립트 버그 보완)
        abstract = ""
        if ab:
            inner = ab.group(1)
            cd = re.search(r"<!\[CDATA\[(.*?)\]\]>", inner, re.DOTALL)
            abstract = (cd.group(1) if cd else re.sub(r"<[^>]+>", "", inner)).strip()

        cat = re.search(r"<article-categories[^>]*>(.*?)</article-categories>", xml, re.DOTALL)
        category = re.sub(r"<[^>]+>", "", cat.group(1)).strip() if cat else ""

        base = {"source_id": article_id, "title": title, "pub_year": year,
                "abstract": abstract, "keywords": keywords, "category": category}

        refs = re.findall(r"<reference([^>]*)>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</reference>", xml, re.DOTALL)
        if not refs:
            return [{**base, "target_arti_id": "", "ref_type": ""}], "OK_NoRef"
        rows = []
        for attrs, _ in refs:
            aid  = re.search(r'arti-id="([^"]*)"', attrs)
            typ  = re.search(r'type-name="([^"]*)"', attrs)
            rows.append({**base,
                         "target_arti_id": aid.group(1) if aid else "",
                         "ref_type":       typ.group(1) if typ else ""})
        return rows, "OK"
    except Exception as e:
        return [], str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="특정 발행연도만 재수집")
    ap.add_argument("--limit", type=int, default=None, help="테스트용 상위 N편만")
    ap.add_argument("--resume", action="store_true", help="체크포인트 이어서")
    args = ap.parse_args()

    df = pd.read_csv(DETAIL_IN, encoding="utf-8-sig", dtype=str, low_memory=False)
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce").astype("Int64")

    meta = (df.sort_values("source_id").groupby("source_id")
              .agg(title=("title", "first"), pub_year=("pub_year", "first")))
    has_ref = df.groupby("source_id").apply(paper_has_ref)
    targets = meta[~meta.index.map(has_ref)].reset_index()
    if args.year is not None:
        targets = targets[targets["pub_year"] == args.year]

    done_ids = set()
    if args.resume and CKPT_FILE.exists():
        prev = pd.read_csv(CKPT_FILE, encoding="utf-8-sig", dtype=str)
        done_ids = set(prev["source_id"].astype(str))
        print(f"체크포인트에서 {len(done_ids)}편 완료 → 건너뜀")
    targets = targets[~targets["source_id"].isin(done_ids)]
    if args.limit:
        targets = targets.head(args.limit)

    print(f"원본(읽기전용): {DETAIL_IN.name}")
    print(f"재수집 대상: {len(targets)}편")
    if args.year is None and len(targets):
        print(targets["pub_year"].value_counts().sort_index().to_string())
    if len(targets) == 0:
        print("대상 없음. 종료."); return
    print()

    ckpt_header = not (args.resume and CKPT_FILE.exists())
    buffer, ok, noref, fail = [], 0, 0, 0
    for row in tqdm(targets.itertuples(index=False), total=len(targets)):
        aid, title, year = str(row.source_id), row.title, row.pub_year
        rows, status = fetch_detail_and_refs(aid, title, year)
        if rows:
            buffer.extend(rows)
            ok += (status == "OK"); noref += (status != "OK")
        else:
            buffer.append({**{c: "" for c in COLS},
                           "source_id": aid, "title": title, "pub_year": year})
            fail += 1; tqdm.write(f"  ! {aid}: {status}")
        if len(buffer) >= 300:
            pd.DataFrame(buffer)[COLS].to_csv(
                CKPT_FILE, mode="a", index=False, header=ckpt_header, encoding="utf-8-sig")
            ckpt_header, buffer = False, []
        time.sleep(SLEEP_SEC)
    if buffer:
        pd.DataFrame(buffer)[COLS].to_csv(
            CKPT_FILE, mode="a", index=False, header=ckpt_header, encoding="utf-8-sig")

    print(f"\n수집 완료 | 참고문헌있음 {ok} / 없음 {noref} / 실패 {fail}")

    # 병합: 원본은 그대로 두고, 대상 논문만 교체한 '완본'을 OUT_FILE로 저장
    new = pd.read_csv(CKPT_FILE, encoding="utf-8-sig", dtype=str)
    replaced_ids = set(new["source_id"].astype(str))
    kept = df[~df["source_id"].astype(str).isin(replaced_ids)].copy()
    kept["pub_year"] = kept["pub_year"].astype(str)
    merged = pd.concat([kept[COLS], new[COLS]], ignore_index=True)
    merged.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    # 검증 요약
    t = merged["target_arti_id"].astype(str).str.strip()
    hr = (t != "") & (t != "nan") & (t != "")
    yr = pd.to_numeric(merged["pub_year"], errors="coerce")
    y2025 = merged[yr == 2025]
    t25 = y2025["target_arti_id"].astype(str).str.strip()
    print(f"\n저장(원본 무변경): {OUT_FILE.name}")
    print(f"  전체 행 {len(merged):,} / 고유 논문 {merged['source_id'].nunique():,} / 교체 {len(replaced_ids):,}편")
    print(f"  2025 참조행: {((t25!='')&(t25!='nan')).sum():,} (재수집 전 0)")
    print(f"\n체크포인트({CKPT_FILE.name})는 확인 후 삭제. 원본 파일은 그대로입니다.")


if __name__ == "__main__":
    main()
