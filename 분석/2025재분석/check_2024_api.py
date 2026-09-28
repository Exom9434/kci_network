"""
check_2024_api.py  (2025인용_재수집 폴더 전용 · 대조군 테스트)
=============================================================
목적: 2025가 아니라 '이미 참조가 채워져 있는' 2024년 논문을 같은 방식으로
      지금 API에 다시 물어봐서, 파이프라인 자체는 정상인지 확인하는 대조군.

핵심 아이디어:
  - 2024 논문은 원본 CSV에 참조가 이미 있음(articleDetail로 3월에 수집).
  - 그 논문들을 '지금' 다시 호출해서 API가 참조를 계속 내려주는지 본다.
  - 각 논문에 대해  [CSV 참조수]  vs  [지금 API 참조수]  를 나란히 비교.

판정:
  대부분 API>0 & CSV와 비슷  → 파이프라인 정상 = 2025만 데이터 공백(코드 무죄 재확인)
  2024도 API=0 으로 나옴       → API 응답 방식이 바뀐 것 → 파싱/엔드포인트 재점검 필요

실행:
    cd 2025인용_재수집
    python check_2024_api.py
    python check_2024_api.py --n 30 --year 2023   # 다른 연도도 대조 가능
"""
import os, re, time, argparse, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

HERE   = Path(__file__).parent
PARENT = HERE.parents[1]
load_dotenv(PARENT / ".env")

API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"
DETAIL   = PARENT / "데이터" / "KCI" / "00.KCI_AI_논문_상세_및_인용데이터.csv"   # 원본(읽기 전용)

if not API_KEY or "여기에" in str(API_KEY):
    raise ValueError("KCI_API_KEY 미설정(../.env 확인)")


def api_ref_count(aid):
    """지금 API가 이 논문에 대해 내려주는 참조/피인용 수."""
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": aid}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(BASE_URL, params=params, headers=headers, timeout=25)
        r.encoding = "utf-8"; xml = r.text
        if "<total>0</total>" in xml:
            return "TOTAL_0", 0, 0
        refs  = len(re.findall(r"<reference([^>]*)>", xml))  # 수집코드와 동일 규칙
        cited = len(re.findall(r"<cited\s", xml))
        return "OK", refs, cited
    except Exception as e:
        return f"ERR:{e}", 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="샘플 논문 수")
    ap.add_argument("--year", type=int, default=2024, help="대조군 연도(기본 2024)")
    args = ap.parse_args()

    df = pd.read_csv(DETAIL, encoding="utf-8-sig", dtype=str, low_memory=False)
    df["y"] = pd.to_numeric(df["pub_year"], errors="coerce")

    # 해당 연도에서 'CSV에 참조가 있는' 논문만 골라 샘플(대조 의미가 있으려면)
    t = df["target_arti_id"].astype(str).str.strip()
    df["hr"] = (t != "") & (t != "nan")
    csv_ref = (df[df["y"] == args.year].groupby("source_id")["hr"].sum())
    have = csv_ref[csv_ref > 0]
    sample = list(have.index[: args.n])
    print(f"{args.year}년 논문 중 CSV에 참조 있는 {len(have)}편에서 {len(sample)}편 샘플\n")

    print(f"{'ID':<20}{'상태':<9}{'CSV참조':>8}{'API참조(지금)':>14}{'API피인용':>10}   일치?")
    print("-" * 74)
    rows = []
    for aid in sample:
        csvn = int(have[aid])
        status, apin, cited = api_ref_count(aid)
        mark = "O" if (status == "OK" and apin > 0) else "X"
        print(f"{aid:<20}{status:<9}{csvn:>8}{apin:>14}{cited:>10}   {mark}")
        rows.append((status, csvn, apin))
        time.sleep(0.8)

    ok  = [r for r in rows if r[0] == "OK"]
    pos = [r for r in ok if r[2] > 0]
    print("\n" + "=" * 74)
    print(f"응답 성공 {len(ok)}/{len(rows)} · API가 참조 내려준 논문 {len(pos)}/{len(ok)}")
    if ok and len(pos) >= max(1, int(0.8 * len(ok))):
        print("\nO 2024는 지금도 API가 참조를 정상 반환 → 파이프라인 정상.")
        print("   = 코드/엔드포인트 문제 아님. 2025만 데이터 공백이라는 결론 재확인.")
    elif ok:
        print("\n! 2024인데도 API 참조가 0으로 많이 나옴 → API 응답구조 변경 가능성.")
        print("   debug XML을 저장해 <reference> 태그 형태를 직접 확인해야 함.")
    else:
        print("\n! API 응답 자체가 실패 → 키/네트워크/차단 여부부터 확인.")


if __name__ == "__main__":
    main()
