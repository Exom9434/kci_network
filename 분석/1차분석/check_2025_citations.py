"""
check_2025_citations.py
=======================
2025년 논문 인용 데이터 업데이트 여부 확인
- 기존 데이터셋에서 2025년 논문 ID 샘플링 → KCI API 호출
- 참고문헌(reference) 데이터 존재 여부 출력

실행: python check_2025_citations.py
"""
import os, re, time, requests, pandas as pd
from dotenv import load_dotenv
from pathlib import Path

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

API_KEY  = os.getenv("KCI_API_KEY")
BASE_URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"
DATA     = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
N_SAMPLE = 20   # 확인할 샘플 수

# ── 2025년 논문 ID 샘플링 ──────────────────────────────────────────
df = pd.read_csv(DATA, encoding="utf-8-sig")
ids_2025 = df[df["pub_year"] == 2025]["source_id"].dropna().unique()
sample_ids = list(ids_2025[:N_SAMPLE])
print(f"2025년 논문 {len(ids_2025)}편 중 {len(sample_ids)}편 샘플 확인\n")

# ── API 호출 ──────────────────────────────────────────────────────
def check_article(article_id, verbose=False):
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": article_id}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(BASE_URL, params=params, headers=headers, timeout=20)
        r.encoding = "utf-8"
        xml = r.text

        if verbose:
            print(f"\n[RAW XML - {article_id}]\n{xml[:800]}\n{'─'*60}")

        # 상태 분류
        if "<total>0</total>" in xml:
            return {"id": article_id, "status": "TOTAL_0",
                    "ref_count": 0, "cited_count": 0, "pub_year": "?", "has_ref": False}
        if "<record>" not in xml:
            snippet = xml[:200].replace('\n', ' ')
            return {"id": article_id, "status": "NO_RECORD",
                    "ref_count": 0, "cited_count": 0, "pub_year": "?", "has_ref": False,
                    "snippet": snippet}

        refs  = re.findall(r'<reference\s', xml)   # <referenceInfo> 제외
        cited = re.findall(r'<cited\s',    xml)   # <citedInfo> 제외
        yr    = re.search(r'<pub-year[^>]*>(\d{4})', xml)

        return {
            "id":          article_id,
            "status":      "OK",
            "pub_year":    yr.group(1) if yr else "?",
            "ref_count":   len(refs),
            "cited_count": len(cited),
            "has_ref":     len(refs) > 0,
        }
    except Exception as e:
        return {"id": article_id, "status": "CONN_ERROR", "ref_count": 0,
                "cited_count": 0, "pub_year": "?", "has_ref": False,
                "error": str(e)}

# 처음 3건 raw XML을 파일로 저장
print("[ 처음 3건 raw XML 저장 중... ]")
for aid in sample_ids[:3]:
    params  = {"apiCode": "articleDetail", "key": API_KEY, "id": aid}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(BASE_URL, params=params, headers=headers, timeout=20)
    r.encoding = "utf-8"
    out_path = BASE / f"debug_{aid}.xml"
    out_path.write_text(r.text, encoding="utf-8")
    print(f"  저장: {out_path.name}")
    time.sleep(1.0)
print()

print(f"\n{'ID':<20} {'상태':<16} {'발행연도':<8} {'참고문헌':<10} {'피인용':<8} {'데이터유무'}")
print("-" * 76)

results = []
for aid in sample_ids:
    res = check_article(aid, verbose=False)
    results.append(res)
    flag = "✅ 있음" if res["has_ref"] else "❌ 없음"
    status = res["status"]
    # snippet 있으면 추가 출력
    snippet = res.get("snippet", "")
    print(f"{aid:<20} {status:<16} {res.get('pub_year','-'):<8} "
          f"{res['ref_count']:<10} {res['cited_count']:<8} {flag}")
    if snippet:
        print(f"  └ 응답 앞부분: {snippet[:120]}")
    time.sleep(1.0)

# ── 요약 ──────────────────────────────────────────────────────────
ok    = [r for r in results if r["status"] == "OK"]
w_ref = [r for r in ok if r["has_ref"]]

print(f"\n{'='*70}")
print(f"결과 요약:")
print(f"  응답 성공:         {len(ok)}/{len(results)}")
print(f"  참고문헌 있음:     {len(w_ref)}/{len(ok)}")

if len(w_ref) > 0:
    print(f"\n✅ 2025년 인용 데이터가 업데이트된 것으로 보입니다.")
    print(f"   → 전체 재수집을 진행하는 것을 권장합니다.")
else:
    print(f"\n❌ 아직 2025년 인용 데이터가 없습니다.")
    print(f"   → 추후 재확인이 필요합니다.")
