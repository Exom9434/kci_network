"""OpenAlex가 SCImago Law 학술지의 논문을 얼마나 수록하는지 연도별로 확인.
실행: python3 check_coverage.py  → openalex_coverage.csv
"""
import csv, json, os, time, urllib.parse, urllib.request
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from collect_openalex import issns, get, MAILTO

rows = list(csv.DictReader(open("scimago_law_journals_2016_2025.csv", encoding="utf-8-sig")))
all_issn = sorted({i for r in rows for i in issns(r["Issn"])})
tot, absn = Counter(), Counter()
for k in range(0, len(all_issn), 50):
    base = "primary_location.source.issn:" + "|".join(all_issn[k:k + 50]) + ",publication_year:2016-2025,type:article"
    for flt, c in [(base, tot), (base + ",has_abstract:true", absn)]:
        q = urllib.parse.urlencode({"filter": flt, "group_by": "publication_year", "mailto": MAILTO})
        for g in get("https://api.openalex.org/works?" + q)["group_by"]:
            c[int(g["key"])] += g["count"]
        time.sleep(0.2)
    print(f"{k + 50}/{len(all_issn)}")
with open("openalex_coverage.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["year", "openalex_articles", "with_abstract"])
    for y in sorted(tot):
        w.writerow([y, tot[y], absn[y]]); print(y, tot[y], absn[y])
