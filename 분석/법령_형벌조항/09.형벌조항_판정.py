# -*- coding: utf-8 -*-
"""
법률별 형벌조항 유무 판정

입력: 법령 csv 저장(26.03.16 기준) 중복제거/*.csv  (법령 1건 = csv 1개, 탭 구분)
출력: 09.형벌조항_판정결과.csv

판정 기준
  형벌 = 형법 제41조의 형(사형·징역·금고·자격상실·자격정지·벌금·구류·과료·몰수)
  과태료·과징금·이행강제금·범칙금은 행정제재이므로 형벌에서 제외한다.
  실제로 형을 부과하는 문형("~에 처한다", "~에 처할 수 있다", "~을 병과한다")만
  집계하므로, 결격사유("금고 이상의 형을 선고받고")나 타법 인용("「형법」 제10조"),
  "벌칙 적용에서 공무원 의제" 같은 조항은 걸리지 않는다.
"""
import csv, re, os, glob, sys

SRC = "법령 csv 저장(26.03.16 기준) 중복제거"
DST = "09.형벌조항_판정결과.csv"

HYEONG = r"(?:사형|무기징역|무기금고|징역|금고|벌금|구류|과료|몰수)"
PUNISH = re.compile(HYEONG + r"[^.\n]{0,60}?(?:처한다|처할|처하되|병과)")
# 형을 부과하는 게 아니라 형을 가리키기만 하는 문형.
# 연금 급여제한("금고 이상의 형에 처할 범죄"), 법원 관할("벌금에 처할 사건"),
# 고발요건, 수형자 이송요건 등이 여기 걸린다.
REJECT = re.compile(r"형에\s*처할|처할\s*(?:사건|범죄)|선고받|병과된\s*때|집행이\s*종료")

def sentence(text, span):
    i = text.find(span)
    s = text.rfind(".", 0, i) + 1
    e = text.find(".", i + len(span))
    return text[s:e if e > 0 else len(text)]
KINDS  = [("사형","사형"), ("징역","징역"), ("금고","금고"), ("벌금","벌금"),
          ("구류","구류"), ("과료","과료"), ("몰수","몰수")]
YANGBEOL = re.compile(r"양벌규정")
GWATAERYO = re.compile(r"과태료")

def body(r):
    return " ".join(filter(None, [r.get("조문내용"), r.get("항"), r.get("호"), r.get("목")]))

def analyze(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    arts, kinds, yang, gwa = {}, set(), False, False
    for r in rows:
        t, b = (r.get("조문제목") or ""), body(r)
        m = PUNISH.search(b)
        if m and REJECT.search(sentence(b, m.group(0))):
            m = None
        if m:
            no = r.get("조문번호")
            arts.setdefault(no, t)
            for k, label in KINDS:
                if k in m.group(0):
                    kinds.add(label)
        if YANGBEOL.search(t):
            yang = True
        if GWATAERYO.search(t):
            gwa = True
    order = [l for _, l in KINDS if l in kinds]
    return {
        "형벌조항": "Y" if arts else "N",
        "형벌조문수": len(arts),
        "형벌종류": "·".join(order),
        "양벌규정": "Y" if yang else "N",
        "과태료조항": "Y" if gwa else "N",
        "형벌조문번호": ",".join(sorted(arts, key=lambda x: (len(x), x))),
    }

def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    if not files:
        sys.exit("입력 폴더를 찾을 수 없습니다: " + SRC)
    cols = ["법령명", "형벌조항", "형벌조문수", "형벌종류", "양벌규정", "과태료조항", "형벌조문번호"]
    n_y = 0
    with open(DST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in files:
            row = {"법령명": os.path.basename(p)[:-4]}
            row.update(analyze(p))
            n_y += row["형벌조항"] == "Y"
            w.writerow(row)
    print("전체 %d건 / 형벌조항 있음 %d건 / 없음 %d건 → %s"
          % (len(files), n_y, len(files) - n_y, DST))

if __name__ == "__main__":
    main()
