"""
probe_journal_match.py  (v2 - deep page 진단)
=============================================
journal= 는 정확일치로 확인됨. 그런데 <total>이 비정상적으로 큼.
→ 뒤 페이지가 (A) 다른 학술지 섞임인지 (B) 빈페이지/중복인지 가려낸다.

각 검색어에 대해 여러 페이지를 찍어:
  - 반환 record 수
  - 실제 journal-name 분포 (검색어와 다른 게 나오면 '오염')
  - 1페이지 논문ID와 겹치는 수 (중복이면 API가 같은 걸 반복)

실행: python probe_journal_match.py
"""
import os, re, time, requests
from collections import Counter
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
KEY = os.getenv("KCI_API_KEY")
URL = "https://open.kci.go.kr/po/openapi/openApiSearch.kci"

JOURNAL = "IT와 법연구"
PAGES   = [1, 3, 5, 20, 50, 100]


def get_page(journal, page):
    r = requests.get(URL, params={"apiCode": "articleSearch", "key": KEY,
                                  "journal": journal, "displayCount": 100, "page": page},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.encoding = "utf-8"
    return r.status_code, r.text


def parse(xml):
    names = re.findall(
        r"<journal-name[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</journal-name>", xml, re.DOTALL)
    ids = re.findall(r'article-id="([^"]+)"', xml)
    if not ids:
        ids = re.findall(r"<article-id[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</article-id>", xml)
    total = re.search(r"<total>(\d+)</total>", xml)
    return (int(total.group(1)) if total else -1,
            [n.strip() for n in names], ids)


if __name__ == "__main__":
    if not KEY:
        raise SystemExit("KCI_API_KEY 없음(.env 확인)")
    print(f"검색어 = [{JOURNAL}]\n")
    page1_ids = set()
    for p in PAGES:
        try:
            code, xml = get_page(JOURNAL, p)
            if code != 200:
                print(f"  page {p:>3}: HTTP {code}  (에러 → 여기서 API가 끊김)")
                time.sleep(0.5); continue
            total, names, ids = parse(xml)
            c = Counter(names)
            if p == 1:
                page1_ids = set(ids)
                overlap = "-"
            else:
                overlap = f"{len(set(ids) & page1_ids)}/{len(ids)}"
            other = sum(v for k, v in c.items() if k != JOURNAL.strip())
            print(f"  page {p:>3}: record {len(ids):>3}건 | total={total:,} | "
                  f"다른학술지 {other}건 | 1p와중복 {overlap}")
            if names and other:
                for k, v in c.most_common(4):
                    print(f"           - {v:>3} {k}")
            time.sleep(0.4)
        except Exception as e:
            print(f"  page {p:>3}: 오류 {e}")

    print("""
판정 가이드:
  · 뒤 페이지에 '다른학술지'가 많다  → total=전체, 관련도정렬(오염). 실제 journal-name 필터 필요.
  · 뒤 페이지 record 0건            → 빈페이지. '0건이면 stop' 만으로 해결.
  · 뒤 페이지가 1p와 중복 많음       → API가 반복 반환. '이미 본 ID면 stop' 필요.
""")
