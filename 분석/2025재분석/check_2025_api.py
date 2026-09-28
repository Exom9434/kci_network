"""
check_2025_api.py  (2025인용_재수집 폴더 전용)
=============================================
재수집 전 사전 점검: KCI Open API articleDetail이 이제 2025년 논문의
'참고문헌(reference)'을 실제로 내려주는지 확인한다.

- 입력: ../00.KCI_AI_논문_상세_및_인용데이터.csv (원본, 읽기 전용)에서 2025 ID 샘플링
- 동작: 샘플 N편을 API 호출 → reference 개수 집계
- 출력: 화면 요약 + 이 폴더에 debug_*.xml 3건 저장

판정:
  참고문헌 있음 > 0  → 재수집 진행 OK (run_recollect.py 실행)
  전부 0             → 아직 API 미반영 → 재수집해도 빈 껍데기, 대기

실행:
    cd 2025인용_재수집
    python check_2025_api.py
    python check_2025_api.py --n 30   # 샘플 수 조정
"""
import os, re, time, argparse, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

HERE   = Path(__file__).parent          # .../2025인용_재수집
PARENT = HERE.parents[1]                # .../kci_network
load_dotenv(PARENT / ".env")

API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"
DETAIL   = PARENT / "데이터" / "KCI" / "00.KCI_AI_논문_상세_및_인용데이터.csv"   # 원본(읽기 전용)

if not API_KEY or "여기에" in str(API_KEY):
    raise ValueError("KCI_API_KEY 미설정(../.env 확인)")


def check_article(aid):
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": aid}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(BASE_URL, params=params, headers=headers, timeout=25)
        r.encoding = "utf-8"; xml = r.text
        if "<total>0</total>" in xml:
            return {"id": aid, "status": "TOTAL_0", "refs": 0, "cited": 0, "year": "?"}
        refs  = len(re.findall(r"<reference\s", xml))   # <referenceInfo> 제외
        cited = len(re.findall(r"<cited\s", xml))        # <citedInfo> 제외
        yr    = re.search(r"<pub-year[^>]*>(\d{4})", xml)
        return {"id": aid, "status": "OK", "year": yr.group(1) if yr else "?",
                "refs": refs, "cited": cited}
    except Exception as e:
        return {"id": aid, "status": f"ERR:{e}", "refs": 0, "cited": 0, "year": "?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="샘플 논문 수")
    args = ap.parse_args()

    df = pd.read_csv(DETAIL, encoding="utf-8-sig", dtype=str, low_memory=False)
    ids = df[df["pub_year"] == "2025"]["source_id"].dropna().unique()
    sample = list(ids[: args.n])
    print(f"2025년 논문 {len(ids)}편 중 {len(sample)}편 샘플 확인\n")

    for aid in sample[:3]:
        r = requests.get(BASE_URL,
                         params={"apiCode": "articleDetail", "key": API_KEY, "id": aid},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        r.encoding = "utf-8"
        (HERE / f"debug_{aid}.xml").write_text(r.text, encoding="utf-8")
        time.sleep(0.8)

    print(f"{'ID':<20}{'상태':<10}{'연도':<7}{'참고문헌':>8}{'피인용':>8}   판정")
    print("-" * 66)
    results = []
    for aid in sample:
        res = check_article(aid); results.append(res)
        flag = "있음 O" if res["refs"] > 0 else "없음 X"
        print(f"{res['id']:<20}{res['status']:<10}{res['year']:<7}"
              f"{res['refs']:>8}{res['cited']:>8}   {flag}")
        time.sleep(0.8)

    ok    = [r for r in results if r["status"] == "OK"]
    wref  = [r for r in ok if r["refs"] > 0]
    print("\n" + "=" * 66)
    print(f"응답 성공 {len(ok)}/{len(results)} · 참고문헌 있음 {len(wref)}/{len(ok)}")
    if wref:
        print("\nO 2025 참고문헌이 API에 반영됨 → run_recollect.py 실행 권장")
    else:
        print("\nX 아직 2025 참고문헌이 API에 없음 → 재수집해도 빈 껍데기. 대기 권장")


if __name__ == "__main__":
    main()
