"""
04.keyword_build.py
--------------------
저자 제공 키워드 기반 공동 사용(co-occurrence) 네트워크 구축

전처리 전략: 한글 우선(Korean-first)
  1. 논문 내 키워드를 쉼표로 분리 후 공백 제거
  2. 괄호 표기 정규화  예) "인공지능(AI)" → "인공지능"
  3. SYNONYM_MAP 적용: 영문 키워드를 한글 표준 키워드로 통일
  4. 동일 논문 내 중복 키워드 제거
  5. MIN_KW_FREQ 미만 출현 키워드 제거 (희귀 노드 제거)
  6. STOP_KEYWORDS: 거의 모든 논문에 등장해 변별력 없는 키워드 제외

출력:
  results/04_keyword_network/
    keyword_stats.txt            기초 통계
    top_keywords.csv             상위 빈도 키워드
    keyword_cooccurrence.graphml 공동 사용 네트워크 (전체)
    keyword_network_stats.txt    네트워크 통계
    degree_distribution.png      연결 분포
"""

from pathlib import Path
import re
import unicodedata
import os
import collections
import itertools

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ───────────────────────────── 경로 설정 ──────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
OUT    = BASE / "results" / "04_keyword_network"
OUT.mkdir(parents=True, exist_ok=True)

# ───────────────────────────── 파라미터 ───────────────────────────────
MIN_KW_FREQ   = 5    # 키워드 최소 등장 논문 수 (미만 제거)
MIN_EDGE_COOC = 3    # 엣지 최소 공동 출현 횟수 (미만 제거)

# 변별력 없는 최상위 일반 키워드 제외 목록
STOP_KEYWORDS = {
    "인공지능", "artificial intelligence", "ai",
    "법", "law", "legal",
}

# ───────────────────────── 동의어 표준화 테이블 ───────────────────────
# 키: 소문자 정규화된 영문(또는 혼용) 표현 → 값: 한글 표준 표현
SYNONYM_MAP: dict[str, str] = {
    # --- 인공지능 계열 ---
    "artificial intelligence":              "인공지능",
    "a.i.":                                 "인공지능",
    "generative ai":                        "생성형 인공지능",
    "generative artificial intelligence":   "생성형 인공지능",
    "large language model":                 "대규모 언어 모델",
    "llm":                                  "대규모 언어 모델",
    "chatgpt":                              "챗gpt",
    "chat gpt":                             "챗gpt",
    "machine learning":                     "머신러닝",
    "deep learning":                        "딥러닝",
    "neural network":                       "신경망",
    "natural language processing":          "자연어 처리",
    "nlp":                                  "자연어 처리",
    "computer vision":                      "컴퓨터 비전",
    "explainable ai":                       "설명가능한 인공지능",
    "xai":                                  "설명가능한 인공지능",
    "algorithm":                            "알고리즘",
    "algorithmic decision":                 "알고리즘 의사결정",
    "algorithmic decision-making":          "알고리즘 의사결정",
    "robot":                                "로봇",
    "robotics":                             "로봇공학",
    # --- 데이터·기술 계열 ---
    "big data":                             "빅데이터",
    "data":                                 "데이터",
    "data protection":                      "데이터 보호",
    "data governance":                      "데이터 거버넌스",
    "blockchain":                           "블록체인",
    "metaverse":                            "메타버스",
    "platform":                             "플랫폼",
    "internet of things":                   "사물인터넷",
    "iot":                                  "사물인터넷",
    "cybersecurity":                        "사이버보안",
    "cyber security":                       "사이버보안",
    # --- 지식재산·저작권 계열 ---
    "copyright":                            "저작권",
    "intellectual property":                "지식재산권",
    "patent":                               "특허",
    "trademark":                            "상표",
    "fair use":                             "공정이용",
    "originality":                          "창작성",
    "authorship":                           "저작자",
    # --- 개인정보·프라이버시 계열 ---
    "personal information":                 "개인정보",
    "personal information protection":      "개인정보보호",
    "privacy":                              "프라이버시",
    "gdpr":                                 "gdpr",           # 고유명사 유지
    "right to be forgotten":                "잊혀질 권리",
    # --- 자율주행 계열 ---
    "autonomous vehicle":                   "자율주행자동차",
    "autonomous vehicles":                  "자율주행자동차",
    "self-driving car":                     "자율주행자동차",
    "self-driving":                         "자율주행",
    "autonomous driving":                   "자율주행",
    "driverless car":                       "자율주행자동차",
    # --- 책임·규제 계열 ---
    "liability":                            "책임",
    "civil liability":                      "민사책임",
    "criminal liability":                   "형사책임",
    "product liability":                    "제조물책임",
    "regulation":                           "규제",
    "governance":                           "거버넌스",
    "compliance":                           "컴플라이언스",
    # --- 권리·헌법 계열 ---
    "human rights":                         "인권",
    "fundamental rights":                   "기본권",
    "discrimination":                       "차별",
    "due process":                          "적법절차",
    "right to explanation":                 "설명 요구권",
    # --- 윤리·가치 계열 ---
    "ethics":                               "윤리",
    "ai ethics":                            "인공지능 윤리",
    "fairness":                             "공정성",
    "transparency":                         "투명성",
    "accountability":                       "책임성",
    "explainability":                       "설명가능성",
    "bias":                                 "편향성",
    "trustworthy ai":                       "신뢰할 수 있는 인공지능",
    # --- 산업혁명 계열 ---
    "4th industrial revolution":            "4차 산업혁명",
    "fourth industrial revolution":         "4차 산업혁명",
    "industry 4.0":                         "인더스트리 4.0",
    # --- 의료·헬스케어 계열 ---
    "healthcare":                           "의료",
    "medical ai":                           "의료 인공지능",
    "clinical decision support":            "임상 의사결정 지원",
    # --- 형사·행정법 계열 ---
    "criminal law":                         "형사법",
    "administrative law":                   "행정법",
    "due diligence":                        "주의의무",
    "judicial decision":                    "사법 판단",
    # --- 기타 ---
    "legal personality":                    "법인격",
    "legal personhood":                     "법인격",
    "electronic person":                    "전자인격",
    "contract":                             "계약",
    "smart contract":                       "스마트 계약",

    # ══ 실행 결과 기반 추가 보완 (2026.04.02) ══

    # 생성형 AI 계열 (한글+영문 혼용 표기)
    "생성형 ai":                             "생성형 인공지능",
    "generative a.i.":                      "생성형 인공지능",

    # 4차 산업혁명 변형 통일
    "제4차 산업혁명":                         "4차 산업혁명",
    "4차산업혁명":                            "4차 산업혁명",
    "4th industrial revolution":            "4차 산업혁명",   # 기존 매핑 중복 방지용 (소문자 동일)

    # 제조물책임 띄어쓰기 통일 (법령 정식 표기: 붙여쓰기)
    "제조물 책임":                            "제조물책임",
    "product liability":                    "제조물책임",     # 기존 "제조물 책임" → 통일

    # 개인정보보호법 영문 표기
    "personal information protection act":  "개인정보보호법",
    "pipa":                                 "개인정보보호법",

    # 딥페이크
    "deepfake":                             "딥페이크",
    "deepfakes":                            "딥페이크",
    "deep fake":                            "딥페이크",

    # 자율성
    "autonomy":                             "자율성",

    # 지능정보사회
    "intelligent information society":      "지능정보사회",

    # 머신러닝 동의어
    "기계학습":                              "머신러닝",
    "machine learning":                     "머신러닝",       # 기존 매핑 확인용 (소문자 동일)

    # 저작권법·저작권 침해 영문
    "copyright law":                        "저작권법",
    "copyright infringement":               "저작권 침해",
    "copyright violation":                  "저작권 침해",

    # 리걸테크
    "legal tech":                           "리걸테크",
    "legal technology":                     "리걸테크",
    "legaltech":                            "리걸테크",

    # 추가 빈출 영문 표기
    "liability":                            "책임",           # 기존 매핑 확인용
    "autonomous system":                    "자율 시스템",
    "facial recognition":                   "안면인식",
    "face recognition":                     "안면인식",
    "surveillance":                         "감시",
    "profiling":                            "프로파일링",
    "risk":                                 "위험",
    "safety":                               "안전",
    "due diligence":                        "주의의무",       # 기존 매핑 확인용
    "negligence":                           "과실",
    "tort":                                 "불법행위",
    "tort law":                             "불법행위법",
    "criminal responsibility":              "형사책임",
    "eu ai act":                            "EU AI Act",     # 고유명사 유지
    "artificial intelligence act":         "EU AI Act",
    "ai act":                               "EU AI Act",
    "chatgpt":                              "챗gpt",          # 기존 매핑 확인용
    "gpt":                                  "gpt",            # 고유명사 유지
    "gpt-4":                                "gpt-4",
    "autonomous weapon":                    "자율무기",
    "autonomous weapons":                   "자율무기",
    "lethal autonomous weapon":             "자율무기",
    "lethal autonomous weapons systems":    "자율무기",

    # 개인정보보호법 띄어쓰기 변형
    "개인정보 보호법":                        "개인정보보호법",

    # 헌법
    "constitution":                         "헌법",
    "digital twin":                         "디지털 트윈",
    "fintech":                              "핀테크",
    "financial technology":                 "핀테크",
}

# ───────────────────────────── 폰트 설정 ─────────────────────────────
_font_candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _font_candidates:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 (AI 논문만, 인용 중복 제거)
# ═══════════════════════════════════════════════════════════════════════
def load_ai_papers() -> pd.DataFrame:
    """카테고리_확정 / 검토필요 기준으로 AI 논문 source_id 목록 취득 후 키워드 컬럼 추출."""
    def find(keyword: str) -> Path:
        for f in os.listdir(BASE):
            if keyword in unicodedata.normalize("NFC", f):
                return BASE / f
        raise FileNotFoundError(keyword)

    df_all = pd.read_csv(DATA, encoding="utf-8-sig")
    ai_ids = set(
        pd.read_csv(find("카테고리_확정"), encoding="utf-8-sig")["논문ID"].astype(str)
    ) | set(
        pd.read_csv(find("검토필요"), encoding="utf-8-sig")["논문ID"].astype(str)
    )
    df = (
        df_all[df_all["source_id"].astype(str).isin(ai_ids)]
        .drop_duplicates(subset="source_id")[["source_id", "pub_year", "keywords"]]
        .copy()
    )
    print(f"[load] AI 논문: {len(df)}편  /  키워드 있음: {df['keywords'].notna().sum()}편")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. 키워드 전처리
# ═══════════════════════════════════════════════════════════════════════
_BRACKET_RE = re.compile(r"\s*[\(\（][^)\）]*[\)\）]")   # 괄호 안 내용 제거

def normalize_keyword(raw: str) -> str:
    """단일 키워드 문자열 정규화."""
    kw = raw.strip()
    kw = _BRACKET_RE.sub("", kw).strip()   # "인공지능(AI)" → "인공지능"
    kw_lower = kw.lower()
    # 동의어 매핑 적용 (소문자 비교)
    if kw_lower in SYNONYM_MAP:
        return SYNONYM_MAP[kw_lower]
    return kw


def parse_keywords(kw_str: str) -> list[str]:
    """쉼표 구분 키워드 문자열 → 정제된 키워드 리스트 (중복 제거)."""
    raw_list = [k.strip() for k in kw_str.split(",") if k.strip()]
    normalized = [normalize_keyword(k) for k in raw_list]
    # 빈 문자열·stopword 제거 + 논문 내 중복 제거 (순서 유지)
    seen: set[str] = set()
    result = []
    for kw in normalized:
        if not kw or kw.lower() in STOP_KEYWORDS:
            continue
        kw_key = kw.lower()
        if kw_key not in seen:
            seen.add(kw_key)
            result.append(kw)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. 전체 빈도 계산 → MIN_KW_FREQ 필터
# ═══════════════════════════════════════════════════════════════════════
def build_kw_lists(df: pd.DataFrame) -> tuple[list[list[str]], collections.Counter]:
    """각 논문의 키워드 리스트 + 전체 키워드 빈도 카운터 반환."""
    kw_per_paper: list[list[str]] = []
    freq: collections.Counter = collections.Counter()

    for kw_str in df["keywords"].dropna():
        kws = parse_keywords(kw_str)
        kw_per_paper.append(kws)
        freq.update(kws)

    # 빈도 집계 시 대소문자 통일 (같은 소문자 키워드 중 빈도 가장 높은 표기 채택)
    # → 여기서는 이미 SYNONYM_MAP으로 충분히 통일했으므로 추가 처리 생략
    return kw_per_paper, freq


# ═══════════════════════════════════════════════════════════════════════
# 4. 공동 사용 네트워크 구축
# ═══════════════════════════════════════════════════════════════════════
def build_network(
    kw_per_paper: list[list[str]],
    freq: collections.Counter,
    min_kw_freq: int = MIN_KW_FREQ,
    min_edge: int = MIN_EDGE_COOC,
) -> nx.Graph:
    """
    노드: 키워드 (min_kw_freq 이상 등장)
    엣지: 같은 논문에 함께 등장한 키워드 쌍 (min_edge 이상 공동 출현)
    엣지 가중치: 공동 출현 논문 수
    """
    vocab: set[str] = {kw for kw, c in freq.items() if c >= min_kw_freq}
    print(f"[vocab] MIN_KW_FREQ={min_kw_freq} 적용 후 어휘 크기: {len(vocab)}개")

    edge_counter: collections.Counter = collections.Counter()
    for kws in kw_per_paper:
        filtered = [kw for kw in kws if kw in vocab]
        for a, b in itertools.combinations(sorted(filtered), 2):
            edge_counter[(a, b)] += 1

    G = nx.Graph()
    # 노드 추가 (빈도 속성 포함)
    for kw in vocab:
        G.add_node(kw, freq=freq[kw])
    # 엣지 추가
    for (a, b), w in edge_counter.items():
        if w >= min_edge:
            G.add_edge(a, b, weight=w)

    print(f"[network] 노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}")
    return G


# ═══════════════════════════════════════════════════════════════════════
# 5. 통계 출력 & 저장
# ═══════════════════════════════════════════════════════════════════════
def save_stats(G: nx.Graph, freq: collections.Counter, df: pd.DataFrame) -> None:
    # ── 상위 빈도 키워드 CSV
    top_df = pd.DataFrame(
        freq.most_common(200), columns=["keyword", "paper_count"]
    )
    top_df.to_csv(OUT / "top_keywords.csv", index=False, encoding="utf-8-sig")

    # ── 네트워크 기초 통계
    degrees = dict(G.degree())
    deg_vals = list(degrees.values())
    components = list(nx.connected_components(G))
    giant = max(components, key=len)
    G_giant = G.subgraph(giant)

    lines = [
        "=" * 50,
        "키워드 공동 사용 네트워크 기초 통계",
        "=" * 50,
        f"분석 대상 논문  : {len(df)}편",
        f"키워드 있는 논문: {df['keywords'].notna().sum()}편",
        f"전체 키워드 토큰: {sum(freq.values()):,}개 (중복 포함)",
        f"유니크 키워드   : {len(freq):,}개 (전처리 후)",
        f"",
        f"[파라미터]",
        f"  MIN_KW_FREQ   = {MIN_KW_FREQ}  (노드 최소 출현 논문 수)",
        f"  MIN_EDGE_COOC = {MIN_EDGE_COOC}  (엣지 최소 공동 출현 횟수)",
        f"  STOP_KEYWORDS = {sorted(STOP_KEYWORDS)}",
        f"",
        f"[네트워크]",
        f"  노드 수         : {G.number_of_nodes():,}",
        f"  엣지 수         : {G.number_of_edges():,}",
        f"  평균 연결 정도  : {sum(deg_vals)/len(deg_vals):.2f}",
        f"  최대 연결 정도  : {max(deg_vals)}",
        f"  연결 성분 수    : {len(components)}",
        f"  거대 성분 크기  : {len(giant)} ({len(giant)/G.number_of_nodes()*100:.1f}%)",
        f"  고립 노드 수    : {sum(1 for d in deg_vals if d == 0)}",
        f"",
        f"[상위 연결 키워드 (degree)]",
    ]
    top_deg = sorted(degrees.items(), key=lambda x: -x[1])[:20]
    for kw, d in top_deg:
        lines.append(f"  {d:4d}  {kw}")

    lines += [
        "",
        "[상위 엣지 가중치 (공동 출현 횟수)]",
    ]
    top_edges = sorted(G.edges(data=True), key=lambda e: -e[2]["weight"])[:20]
    for a, b, d in top_edges:
        lines.append(f"  {d['weight']:4d}  {a}  ↔  {b}")

    stat_text = "\n".join(lines)
    print(stat_text)
    (OUT / "keyword_network_stats.txt").write_text(stat_text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# 6. 연결 분포 시각화
# ═══════════════════════════════════════════════════════════════════════
def plot_degree_dist(G: nx.Graph) -> None:
    degrees = sorted([d for _, d in G.degree()], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(degrees)), degrees, width=1.0, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("키워드 순위 (연결 정도 순)")
    axes[0].set_ylabel("연결 정도 (Degree)")
    axes[0].set_title("키워드 네트워크 연결 분포")

    import numpy as np
    from collections import Counter
    cnt = Counter(degrees)
    xs = sorted(cnt.keys())
    ys = [cnt[x] for x in xs]
    axes[1].scatter(xs, ys, s=30, color="steelblue", alpha=0.7)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("연결 정도 (log)")
    axes[1].set_ylabel("빈도 (log)")
    axes[1].set_title("Degree Distribution (log-log)")

    plt.tight_layout()
    plt.savefig(OUT / "degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[saved] degree_distribution.png")


# ═══════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════
def main() -> None:
    df = load_ai_papers()

    print("\n── 키워드 전처리 중...")
    kw_per_paper, freq = build_kw_lists(df)

    # 전처리 후 빈도 기초 확인
    print(f"\n상위 30개 키워드 (전처리 후):")
    for kw, c in freq.most_common(30):
        print(f"  {c:4d}  {kw}")

    print("\n── 공동 사용 네트워크 구축 중...")
    G = build_network(kw_per_paper, freq)

    print("\n── 통계 저장 중...")
    save_stats(G, freq, df)

    print("\n── 연결 분포 시각화 중...")
    plot_degree_dist(G)

    print("\n── GraphML 저장 중...")
    nx.write_graphml(G, OUT / "keyword_cooccurrence.graphml")
    print(f"[saved] keyword_cooccurrence.graphml")

    # 거대 성분만 별도 저장
    giant = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(giant).copy()
    nx.write_graphml(G_giant, OUT / "keyword_giant.graphml")
    print(f"[saved] keyword_giant.graphml  (노드 {G_giant.number_of_nodes()}개)")

    print(f"\n✓ 완료: {OUT}")


if __name__ == "__main__":
    main()
