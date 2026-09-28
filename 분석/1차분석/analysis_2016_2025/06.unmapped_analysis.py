"""
06.unmapped_analysis.py
-----------------------
커뮤니티 매핑에서 제외된 681편 분석 (빈도 합산 기반 논문 할당 로직 포함)
"""

from pathlib import Path
import re, unicodedata, collections, itertools, os
from collections import defaultdict, Counter

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─────────────────────── 경로 및 설정 ───────────────────────
BASE = Path(__file__).parent
DATA = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
OUT  = BASE / "results" / "06_unmapped"
OUT.mkdir(parents=True, exist_ok=True)

MAP_FILE = BASE / "results" / "04_keyword_network" / "community_paper_map.csv"

MIN_KW_FREQ   = 3   
MIN_EDGE_COOC = 2
STOP_KEYWORDS = {"인공지능", "artificial intelligence", "ai", "법", "law", "legal"}

SYNONYM_MAP = {
    "artificial intelligence": "인공지능", "generative ai": "생성형 인공지능",
    "generative artificial intelligence": "생성형 인공지능",
    "large language model": "대규모 언어 모델", "llm": "대규모 언어 모델",
    "chatgpt": "챗gpt", "chat gpt": "챗gpt",
    "machine learning": "머신러닝", "deep learning": "딥러닝",
    "natural language processing": "자연어 처리", "nlp": "자연어 처리",
    "big data": "빅데이터", "data": "데이터", "blockchain": "블록체인",
    "copyright": "저작권", "intellectual property": "지식재산권",
    "personal information": "개인정보", "privacy": "프라이버시",
    "autonomous vehicle": "자율주행자동차", "autonomous vehicles": "자율주행자동차",
    "regulation": "규제", "governance": "거버넌스",
    "liability": "책임", "ethics": "윤리", "ai ethics": "인공지능 윤리",
    "fairness": "공정성", "transparency": "투명성",
    "4th industrial revolution": "4차 산업혁명",
    "fourth industrial revolution": "4차 산업혁명",
    "eu ai act": "EU AI Act", "artificial intelligence act": "EU AI Act",
    "deepfake": "딥페이크", "deepfakes": "딥페이크",
    "생성형 ai": "생성형 인공지능",
    "제4차 산업혁명": "4차 산업혁명", "4차산업혁명": "4차 산업혁명",
    "개인정보 보호법": "개인정보보호법",
    "legal tech": "리걸테크", "legaltech": "리걸테크",
    "기계학습": "머신러닝",
}

_BRACKET_RE = re.compile(r"\s*[\(\（][^)\）]*[\)\）]")

def normalize_kw(raw: str) -> str:
    kw = _BRACKET_RE.sub("", raw.strip()).strip()
    return SYNONYM_MAP.get(kw.lower(), kw)

def parse_keywords(kw_str: str) -> list[str]:
    seen, result = set(), []
    if pd.isna(kw_str): return []
    for raw in str(kw_str).split(","):
        kw = normalize_kw(raw.strip())
        if not kw or kw.lower() in STOP_KEYWORDS:
            continue
        key = kw.lower()
        if key not in seen:
            seen.add(key)
            result.append(kw)
    return result

# ─────────────────────── 폰트 설정 ───────────────────────
for _fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 
            "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

# ─────────────────────── 핵심 분석 함수 ───────────────────────

def load_unmapped():
    df_conf = pd.read_csv(BASE / "KCI_AI_논문_카테고리_확정.csv", encoding="utf-8-sig")
    df_man  = pd.read_csv(BASE / "KCI_AI_논문_카테고리_미확정_수동검토완료.csv", encoding="utf-8-sig")
    df_cat  = pd.concat([df_conf[["논문ID","최종_카테고리"]], df_man[["논문ID","최종_카테고리"]]], ignore_index=True)
    df_cat["논문ID"] = df_cat["논문ID"].astype(str)

    df_map = pd.read_csv(MAP_FILE, encoding="utf-8-sig")
    df_map["논문ID"] = df_map["논문ID"].astype(str)
    mapped_ids = set(df_map["논문ID"])

    df_main = pd.read_csv(DATA, encoding="utf-8-sig").drop_duplicates("source_id")
    df_main["source_id"] = df_main["source_id"].astype(str)
    df_unmapped_ids = df_cat[~df_cat["논문ID"].isin(mapped_ids)]["논문ID"]

    df_um = (
        df_main[df_main["source_id"].isin(df_unmapped_ids)]
        [["source_id","pub_year","keywords"]]
        .merge(df_cat.rename(columns={"논문ID":"source_id"}), on="source_id", how="left")
    )
    return df_um, df_cat, mapped_ids

def plot_category_ratio(df_all_cat, mapped_ids):
    df_all_cat = df_all_cat.copy()
    df_all_cat["mapped"] = df_all_cat["논문ID"].isin(mapped_ids)
    single = df_all_cat[~df_all_cat["최종_카테고리"].str.startswith("융복합", na=False)]
    counts = single.groupby("최종_카테고리").size()
    major = counts[counts >= 10].index

    rows = []
    for cat in major:
        sub = single[single["최종_카테고리"] == cat]
        rows.append({"카테고리": cat, "전체": len(sub), "매핑": sub["mapped"].sum(),
                     "미매핑": len(sub) - sub["mapped"].sum(), "매핑률": sub["mapped"].mean() * 100})
    df_r = pd.DataFrame(rows).sort_values("매핑률")

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(df_r))
    ax.barh(y, df_r["매핑률"], color="#4C72B0", alpha=0.85, label="매핑")
    ax.barh(y, 100 - df_r["매핑률"], left=df_r["매핑률"], color="#DD8452", alpha=0.6, label="미매핑")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c} (n={t})" for c, t in zip(df_r["카테고리"], df_r["전체"])], fontsize=9)
    ax.set_xlabel("비율 (%)")
    ax.set_title("단일 카테고리별 커뮤니티 매핑률 (n≥10)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "unmapped_category_ratio.png", dpi=150)
    plt.close()

def build_unmapped_network(df_um):
    kw_per_paper, freq = [], Counter()
    for kw_str in df_um["keywords"].dropna():
        kws = parse_keywords(kw_str)
        kw_per_paper.append(kws)
        freq.update(kws)

    vocab = {kw for kw, c in freq.items() if c >= MIN_KW_FREQ}
    G = nx.Graph()
    for kw in vocab: G.add_node(kw, freq=freq[kw])
    
    edge_cnt = Counter()
    for kws in kw_per_paper:
        filtered = [k for k in kws if k in vocab]
        for a, b in itertools.combinations(sorted(filtered), 2):
            edge_cnt[(a, b)] += 1
    for (a, b), w in edge_cnt.items():
        if w >= MIN_EDGE_COOC: G.add_edge(a, b, weight=w)
    return G, freq

def leiden_communities(G):
    try:
        import igraph as ig, leidenalg
        mapping = {n: i for i, n in enumerate(G.nodes())}
        ig_G = ig.Graph(len(mapping), [(mapping[u], mapping[v]) for u, v in G.edges()])
        part = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition, seed=42)
        rev = {i: n for n, i in mapping.items()}
        return {rev[idx]: cid for cid, members in enumerate(part) for idx in members}
    except ImportError:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
        return {n: cid for cid, nodes in enumerate(comms) for n in nodes}

def save_community_summary(df_um, G, comm_map, freq):
    """빈도 합산 기준 동점 처리 및 논문 수 할당"""
    paper_assignments = []
    
    for _, row in df_um.iterrows():
        kws = parse_keywords(row.get("keywords", ""))
        valid_kws = [k for k in kws if k in comm_map]
        
        if not valid_kws:
            paper_assignments.append(-1)
            continue
            
        # 커뮤니티별 (매칭 키워드 수, 키워드 빈도 합산) 계산
        scores = defaultdict(lambda: {"count": 0, "f_sum": 0})
        for k in valid_kws:
            cid = comm_map[k]
            scores[cid]["count"] += 1
            scores[cid]["f_sum"] += freq.get(k, 0)
            
        # 정렬: 1순위 키워드 수(desc), 2순위 빈도 합(desc)
        best_cid = sorted(scores.items(), key=lambda x: (x[1]["count"], x[1]["f_sum"]), reverse=True)[0][0]
        paper_assignments.append(best_cid)

    df_um["assigned_community"] = paper_assignments
    p_counts = Counter(paper_assignments)
    
    kw_groups = defaultdict(list)
    for node, cid in comm_map.items(): kw_groups[cid].append(node)

    rows = []
    for cid, members in sorted(kw_groups.items(), key=lambda x: -len(x[1])):
        top_kws = sorted(members, key=lambda k: -freq.get(k, 0))[:6]
        rows.append({
            "커뮤니티": cid,
            "노드 수(키워드)": len(members),
            "논문 수(Paper)": p_counts.get(cid, 0),
            "대표 키워드": ", ".join(top_kws),
        })
    
    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT / "unmapped_communities.csv", index=False, encoding="utf-8-sig")
    print("\n[주변부 커뮤니티 요약]")
    print(df_res.to_string(index=False))
    return df_res

def plot_unmapped_network(G, comm_map, freq):
    if G.number_of_nodes() == 0: return
    giant = max(nx.connected_components(G), key=len)
    Gs = G.subgraph(giant).copy()
    if Gs.number_of_nodes() > 150:
        top = sorted(Gs.degree(), key=lambda x: -x[1])[:150]
        Gs = Gs.subgraph([n for n, _ in top]).copy()

    colors = plt.cm.tab20.colors
    pos = nx.spring_layout(Gs, seed=42, k=1.5)
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(Gs, pos, ax=ax, alpha=0.2, edge_color="#aaaaaa")
    nx.draw_networkx_nodes(Gs, pos, ax=ax, node_size=[max(100, freq.get(n, 1)*30) for n in Gs.nodes()],
                           node_color=[colors[comm_map.get(n, 0) % len(colors)] for n in Gs.nodes()], alpha=0.8)
    labels = {n: n for n, _ in sorted(Gs.degree(), key=lambda x: -x[1])[:30]}
    nx.draw_networkx_labels(Gs, pos, labels=labels, ax=ax, font_size=8)
    ax.axis("off")
    plt.savefig(OUT / "unmapped_network.png", dpi=150)
    plt.close()

def main():
    df_um, df_all_cat, mapped_ids = load_unmapped()
    plot_category_ratio(df_all_cat, mapped_ids)
    
    G, freq = build_unmapped_network(df_um)
    if G.number_of_nodes() > 0:
        comm_map = leiden_communities(G)
        save_community_summary(df_um, G, comm_map, freq)
        plot_unmapped_network(G, comm_map, freq)
        
        # 통계 저장
        stats = f"분석 편수: {len(df_um)}\n노드: {G.number_of_nodes()}\n엣지: {G.number_of_edges()}"
        (OUT / "unmapped_network_stats.txt").write_text(stats, encoding="utf-8")
    
    print(f"\n✓ 분석 완료: {OUT}")

if __name__ == "__main__":
    main()