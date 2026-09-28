"""SCImago Law 학술지(2016·2025 통합)에 실린 2016~2025 인공지능 논문을 OpenAlex에서 수집.
본인 맥 터미널에서 실행 (표준 라이브러리만 사용):
    cd 영문파일 && python3 collect_openalex.py
출력:
    cache/bNNNN.jsonl       1단계 원자료(ISSN 50개 묶음별, 이어받기용)
    openalex_ai_law.jsonl   인공지능 판정 논문 원자료 (참고문헌 목록 포함)
    openalex_ai_law.csv     분석용 표 (초록 복원)
    abstract_coverage.csv   학술지별 초록 보유율
AI 판정은 한국 코퍼스(01.collect_ai_details.py)와 같은 기준:
    제목+초록에 'artificial intelligence' 또는 단어경계 대문자 'AI'.
"""
import csv, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from collections import defaultdict

MAILTO = "silverbreak1@gmail.com"
JOURNALS = "scimago_law_journals_2016_2025.csv"
# 대소문자 구분. 긴 것부터. 한글 옆은 경계로 인정(영문자만 경계 위반).
TERMS = ["GenAI", "genAI", "GPAI", "OpenAI", "AIGC", "AILD", "GAI", "XAI", "xAI", "AIA", "AIs", "AI"]
TERM_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(TERMS) + r")(?![A-Za-z])")
PHRASE_RE = re.compile(r"artificial intelligence|artificially intelligent", re.I)

WEAK = {"AIA", "GAI"}

def all_caps(t):
    # 영문 소문자가 없고 영문 대문자가 10자 이상이면 전부 대문자 제목으로 봄(한글 제목은 해당 없음)
    return not re.search("[a-z]", t) and len(re.findall("[A-Z]", t)) >= 10

def match_terms(title, abstract=""):
    """일치한 용어 집합. 전부 대문자인 제목에서는 약어 규칙을 쓰지 않음(CLAIM 등 오탐 방지)."""
    title, abstract = title or "", abstract or ""
    found = set()
    for part, caps in [(title, all_caps(title)), (abstract, False)]:
        if PHRASE_RE.search(part):
            found.add("artificial intelligence")
        if not caps:
            found.update(TERM_RE.findall(part))
    # 약한 용어는 단독 근거로 쓰지 않음. AIA=America Invents Act, GAI=EU 사법내무(JAI/GAI) 결정·각종 지수 약칭
    return found if found - WEAK else set()

def is_ai(title, abstract=""):
    return bool(match_terms(title, abstract))

def issns(cell):
    out = []
    for s in str(cell).split(","):
        s = s.strip()
        if len(s) == 8:
            out.append(f"{s[:4]}-{s[4:]}")
    return out

def abstract(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idx in inv.items() for i in idx]
    return " ".join(w for _, w in sorted(pos))

def api_key():
    # 환경변수 또는 상위 폴더 .env의 OPENALEX_API_KEY (openalex.org/settings/api 에서 무료 발급)
    if os.environ.get("OPENALEX_API_KEY"):
        return os.environ["OPENALEX_API_KEY"]
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.startswith("OPENALEX_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None

KEY = api_key()

def get(url):
    if KEY:
        url += "&api_key=" + KEY
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                sys.exit("하루 사용량 초과(429). 한국시간 오전 9시(UTC 자정)에 초기화됨. "
                         "받아 둔 묶음은 cache/에 남아 있으니 그대로 다시 실행하면 이어서 받음.")
            print("  retry", attempt + 1, e, file=sys.stderr); time.sleep(2 ** attempt)
        except Exception as e:
            print("  retry", attempt + 1, e, file=sys.stderr); time.sleep(2 ** attempt)
    raise RuntimeError(url)

def fetch(issn_batch):
    # 검색은 넓게(OR), 판정은 로컬에서 엄격하게
    flt = ",".join([
        "primary_location.source.issn:" + "|".join(issn_batch),
        "publication_year:2016-2025",
        "type:article",
        'title_and_abstract.search:"artificial intelligence" OR "artificially intelligent" OR ' + " OR ".join(TERMS),
    ])
    cursor = "*"
    while cursor:
        q = urllib.parse.urlencode({"filter": flt, "per_page": 200, "cursor": cursor, "mailto": MAILTO})
        d = get("https://api.openalex.org/works?" + q)
        yield from d["results"]
        cursor = d["meta"].get("next_cursor")
        time.sleep(0.2)

def download(all_issn):
    """50개 ISSN 묶음마다 cache/bNNNN.jsonl로 저장. 이미 받은 묶음은 건너뜀(이어받기)."""
    os.makedirs("cache", exist_ok=True)
    for k in range(0, len(all_issn), 50):
        path = f"cache/b{k:04d}.jsonl"
        if os.path.exists(path):
            continue
        with open(path + ".part", "w", encoding="utf-8") as f:
            n = 0
            for w in fetch(all_issn[k:k + 50]):
                f.write(json.dumps(w, ensure_ascii=False) + "\n"); n += 1
        os.replace(path + ".part", path)
        print(f"  ISSN {min(k + 50, len(all_issn))}/{len(all_issn)}  1단계 {n}편")

def write_replace(path, fn):
    # 끝까지 쓴 뒤에만 기존 파일을 교체(중단돼도 이전 결과 보존)
    fn(path + ".tmp"); os.replace(path + ".tmp", path)

def main():
    rows = list(csv.DictReader(open(JOURNALS, encoding="utf-8-sig")))
    all_issn = sorted({i for r in rows for i in issns(r["Issn"])})
    print(f"학술지 {len(rows)}종, ISSN {len(all_issn)}개, API 키 {'있음' if KEY else '없음(하루 한도 낮음)'}")
    download(all_issn)
    seen, kept_raw, kept = set(), [], []
    for name in sorted(os.listdir("cache")):
        if not name.endswith(".jsonl"):
            continue
        for line in open(os.path.join("cache", name), encoding="utf-8"):
            w = json.loads(line)
            if w["id"] in seen:
                continue
            seen.add(w["id"])
            ab = abstract(w.get("abstract_inverted_index"))
            terms = match_terms(w.get("title"), ab)
            if not terms:
                continue
            kept_raw.append(w)
            src = (w.get("primary_location") or {}).get("source") or {}
            kept.append({
                "id": w["id"], "doi": w.get("doi"), "year": w.get("publication_year"),
                "title": w.get("title"), "abstract": ab, "matched": "|".join(sorted(terms)), "has_abstract": bool(ab),
                "source": src.get("display_name"), "source_id": src.get("id"),
                "language": w.get("language"),
                "countries": "|".join(sorted({c for a in w.get("authorships", []) for c in a.get("countries", [])})),
                "keywords": "|".join(x.get("display_name", "") for x in w.get("keywords", [])),
                "n_refs": len(w.get("referenced_works", [])), "cited_by": w.get("cited_by_count"),
            })
    print(f"1단계 {len(seen)}편 중 인공지능 판정 {len(kept)}편")

    def w_jsonl(p):
        with open(p, "w", encoding="utf-8") as f:
            for w in kept_raw:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
    def w_csv(p):
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(kept[0])); wr.writeheader(); wr.writerows(kept)
    def w_cov(p):
        cov = defaultdict(lambda: [0, 0])
        for r in kept:
            cov[r["source"]][0] += 1; cov[r["source"]][1] += r["has_abstract"]
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f); wr.writerow(["source", "n", "with_abstract"])
            for s_, (n, a) in sorted(cov.items(), key=lambda x: -x[1][0]):
                wr.writerow([s_, n, a])
    write_replace("openalex_ai_law.jsonl", w_jsonl)
    write_replace("openalex_ai_law.csv", w_csv)
    write_replace("abstract_coverage.csv", w_cov)
    tot = len(kept); ab = sum(r["has_abstract"] for r in kept); rf = sum(r["n_refs"] > 0 for r in kept)
    print(f"완료: {tot}편, 초록 {ab} ({ab/tot:.0%}), 참고문헌 보유 {rf} ({rf/tot:.0%})")

def _check():
    assert issns("17459125, 00111384") == ["1745-9125", "0011-1384"]
    assert abstract({"world": [1], "hello": [0]}) == "hello world"
    assert is_ai("Regulating AI in Europe") and is_ai("artificial intelligence and tort")
    assert not is_ai("Aid and trade") and not is_ai("the ai of")
    assert match_terms("GenAI and GPAI under the AIA") == {"GenAI", "GPAI", "AIA"}
    assert not is_ai("IPR proceedings after the AIA")
    assert not is_ai("Decisione quadro 2002/584/GAI") and is_ai("GAI and GPAI models")
    assert is_ai("When AIs contract") and is_ai("생성형AI의 저작권")
    assert not is_ai("AIIB and AIDS policy") and not is_ai("CLAIMS AGAINST THE STATE")
    assert not is_ai("THE REGULATION OF AI UNDER EU LAW")  # 전부 대문자 제목의 약어는 불인정(초록으로 판정)
    assert is_ai("ARTIFICIAL INTELLIGENCE AND TORT")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 어디서 실행해도 영문파일 폴더 기준
    _check()
    main()
