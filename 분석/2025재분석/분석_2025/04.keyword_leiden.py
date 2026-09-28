"""
04.keyword_leiden.py
---------------------
키워드 공동 사용 네트워크(keyword_giant.graphml)에
Leiden 알고리즘 적용 → 커뮤니티 탐지

주요 출력:
  results/04_keyword_network/
    keyword_communities.csv      — 키워드별 커뮤니티 ID, 상위 키워드 레이블
    community_summary.txt        — 커뮤니티별 요약 (상위 키워드, 논문 수 등)
    community_paper_map.csv      — 논문 → 커뮤니티 매핑 (다중 커뮤니티 가능)
    community_vs_category.csv    — 커뮤니티 × LLM 카테고리 교차표
    community_vs_category.png    — 히트맵 시각화
"""

from pathlib import Path
import os
import unicodedata
import collections
import importlib.util

import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ────────────────────────────── 경로 설정 ─────────────────────────────
BASE = Path(__file__).parent
NET  = BASE / "results" / "04_keyword_network" / "keyword_giant.graphml"
OUT  = BASE / "results" / "04_keyword_network"

# ────────────────── 04.keyword_build.py 전처리 함수 임포트 ────────────
# kw_to_com의 키가 keyword_build의 정규화 결과이므로
# 논문 키워드 조회 시 동일한 전처리를 적용해야 정확히 매칭됨
_kb_spec = importlib.util.spec_from_file_location(
    "keyword_build", BASE / "04.keyword_build.py"
)
_kb = importlib.util.module_from_spec(_kb_spec)
_kb_spec.loader.exec_module(_kb)
parse_keywords = _kb.parse_keywords   # 괄호 제거 + SYNONYM_MAP + STOP 필터

# ────────────────────────────── 폰트 설정 ─────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════
def load_categories() -> pd.DataFrame:
    """카테고리_확정 + 검토필요 CSV를 합쳐 논문ID → 최종_카테고리 매핑 반환."""
    def find(keyword: str) -> Path:
        for f in os.listdir(BASE):
            fn = unicodedata.normalize("NFC", f)
            if keyword in fn:
                return BASE / f
        raise FileNotFoundError(keyword)

    cat = pd.read_csv(find("카테고리_확정"), encoding="utf-8-sig")[["논문ID", "최종_카테고리"]]
    rev = pd.read_csv(find("검토필요"),   encoding="utf-8-sig")[["논문ID", "최종_카테고리"]]
    merged = pd.concat([cat, rev], ignore_index=True).drop_duplicates(subset="논문ID")
    merged["논문ID"] = merged["논문ID"].astype(str)
    print(f"[categories] 논문 {len(merged)}편, 카테고리: {sorted(merged['최종_카테고리'].unique())}")
    return merged


def load_main_keywords(cat_df: pd.DataFrame) -> pd.DataFrame:
    """
    00.KCI_AI_논문_상세_및_인용데이터.csv 에서 source_id + keywords 추출.
    AI 논문 2014편 기준.
    """
    data_path = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
    df_all = pd.read_csv(data_path, encoding="utf-8-sig")
    ai_ids = set(cat_df["논문ID"])
    df = (
        df_all[df_all["source_id"].astype(str).isin(ai_ids)]
        .drop_duplicates(subset="source_id")[["source_id", "keywords"]]
        .copy()
    )
    df["source_id"] = df["source_id"].astype(str)
    print(f"[keywords] 키워드 컬럼 로드: {len(df)}편")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. NetworkX → igraph 변환
# ══════════════════════════════════════════════════════════════════════
def nx_to_igraph(G_nx: nx.Graph) -> tuple[ig.Graph, dict[int, str]]:
    """NetworkX 그래프를 igraph로 변환. idx→node명 매핑 반환."""
    nodes = list(G_nx.nodes())
    idx_map = {node: i for i, node in enumerate(nodes)}

    edges = [(idx_map[u], idx_map[v]) for u, v in G_nx.edges()]
    weights = [G_nx[u][v].get("weight", 1) for u, v in G_nx.edges()]

    G_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    G_ig.vs["name"]  = nodes
    G_ig.vs["freq"]  = [G_nx.nodes[n].get("freq", 1) for n in nodes]
    G_ig.es["weight"] = weights

    idx_to_name = {i: n for i, n in enumerate(nodes)}
    return G_ig, idx_to_name


# ══════════════════════════════════════════════════════════════════════
# 3. Leiden 커뮤니티 탐지
# ══════════════════════════════════════════════════════════════════════
def run_leiden(G_ig: ig.Graph, resolution: float = 1.0) -> list[int]:
    """
    leidenalg.RBConfigurationVertexPartition + 엣지 가중치 사용.
    반환: 각 노드의 커뮤니티 ID 리스트
    """
    partition = leidenalg.find_partition(
        G_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=10,
        seed=42,
    )
    membership = partition.membership
    print(f"[leiden] 커뮤니티 수: {len(set(membership))}  |  modularity: {partition.modularity:.4f}")
    return membership


# ══════════════════════════════════════════════════════════════════════
# 4. 커뮤니티 레이블 생성 (상위 빈도 키워드 3개)
# ══════════════════════════════════════════════════════════════════════
def make_community_labels(
    G_ig: ig.Graph, membership: list[int]
) -> dict[int, str]:
    """커뮤니티 ID → '상위키1 / 상위키2 / 상위키3' 레이블."""
    com_kws: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for idx, cid in enumerate(membership):
        freq = G_ig.vs[idx]["freq"]
        name = G_ig.vs[idx]["name"]
        com_kws[cid].append((freq, name))

    labels: dict[int, str] = {}
    for cid, items in com_kws.items():
        top3 = [kw for _, kw in sorted(items, reverse=True)[:3]]
        labels[cid] = " / ".join(top3)
    return labels


# ══════════════════════════════════════════════════════════════════════
# 5. 논문 → 커뮤니티 매핑
#    (논문의 키워드 중 giant 컴포넌트에 있는 키워드들의 커뮤니티를 집계)
# ══════════════════════════════════════════════════════════════════════
def map_papers_to_communities(
    kw_df: pd.DataFrame,
    kw_to_com: dict[str, int],
    com_labels: dict[int, str],
) -> pd.DataFrame:
    """
    논문별로 보유 키워드가 속한 커뮤니티 분포 계산.
    주 커뮤니티(most common) 1개를 primary_community로 지정.
    """
    rows = []
    for _, row in kw_df.iterrows():
        pid = row["source_id"]
        if pd.isna(row["keywords"]):
            continue
        # keyword_build와 동일한 전처리 적용 (괄호 제거 + SYNONYM_MAP + STOP 필터)
        kws = parse_keywords(str(row["keywords"]))
        com_hits: list[int] = []
        for kw in kws:
            if kw in kw_to_com:
                com_hits.append(kw_to_com[kw])
            elif kw.lower() in kw_to_com:
                com_hits.append(kw_to_com[kw.lower()])
        if not com_hits:
            continue
        counter = collections.Counter(com_hits)
        primary = counter.most_common(1)[0][0]
        rows.append({
            "논문ID":           pid,
            "primary_community": primary,
            "community_label":  com_labels.get(primary, ""),
            "matched_keywords": len(com_hits),
        })

    df_map = pd.DataFrame(rows)
    print(f"[paper_map] 커뮤니티 매핑된 논문: {len(df_map)}편")
    return df_map


# ══════════════════════════════════════════════════════════════════════
# 6. 커뮤니티 × LLM 카테고리 교차 분석
# ══════════════════════════════════════════════════════════════════════
def cross_analysis(
    paper_map: pd.DataFrame,
    cat_df: pd.DataFrame,
    com_labels: dict[int, str],
) -> pd.DataFrame:
    """커뮤니티 × 카테고리 교차표 생성 및 히트맵 저장."""
    merged = paper_map.merge(cat_df, on="논문ID", how="left")
    merged = merged.dropna(subset=["최종_카테고리"])

    # 교차표
    crosstab = pd.crosstab(
        merged["community_label"],
        merged["최종_카테고리"],
        margins=True,
    )
    crosstab.to_csv(OUT / "community_vs_category.csv", encoding="utf-8-sig")
    print(f"[cross] 교차표 저장: {crosstab.shape}")

    # ── 히트맵 (margins 제외)
    ct_raw = pd.crosstab(merged["community_label"], merged["최종_카테고리"])
    # 행 정규화 (각 커뮤니티 내 카테고리 비율)
    ct_norm = ct_raw.div(ct_raw.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(ct_norm) * 0.55 + 2)))

    # 왼쪽: 절대 빈도
    im0 = axes[0].imshow(ct_raw.values, aspect="auto", cmap="Blues")
    axes[0].set_xticks(range(len(ct_raw.columns)))
    axes[0].set_xticklabels(ct_raw.columns, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(ct_raw.index)))
    axes[0].set_yticklabels(ct_raw.index, fontsize=7)
    axes[0].set_title("커뮤니티 × 카테고리 (논문 수)", fontsize=11)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    for r in range(ct_raw.shape[0]):
        for c in range(ct_raw.shape[1]):
            v = ct_raw.values[r, c]
            if v > 0:
                axes[0].text(c, r, str(v), ha="center", va="center", fontsize=6,
                             color="white" if v > ct_raw.values.max() * 0.6 else "black")

    # 오른쪽: 행 정규화 비율
    im1 = axes[1].imshow(ct_norm.values, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(ct_norm.columns)))
    axes[1].set_xticklabels(ct_norm.columns, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(ct_norm.index)))
    axes[1].set_yticklabels(ct_norm.index, fontsize=7)
    axes[1].set_title("커뮤니티 × 카테고리 (비율)", fontsize=11)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUT / "community_vs_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[saved] community_vs_category.png")

    return merged


# ══════════════════════════════════════════════════════════════════════
# 7. 결과 텍스트 저장
# ══════════════════════════════════════════════════════════════════════
def save_summary(
    G_ig: ig.Graph,
    membership: list[int],
    com_labels: dict[int, str],
    merged: pd.DataFrame,
) -> None:
    com_ids = sorted(set(membership))
    lines = [
        "=" * 60,
        "키워드 커뮤니티 탐지 결과 (Leiden)",
        "=" * 60,
        f"총 커뮤니티 수  : {len(com_ids)}",
        f"분석 노드 수    : {G_ig.vcount()}",
        "",
    ]

    for cid in com_ids:
        members = [G_ig.vs[i]["name"] for i, m in enumerate(membership) if m == cid]
        members_with_freq = sorted(
            [(G_ig.vs[i]["freq"], G_ig.vs[i]["name"]) for i, m in enumerate(membership) if m == cid],
            reverse=True
        )
        paper_count = len(merged[merged["primary_community"] == cid]) if "primary_community" in merged.columns else "?"
        cat_dist = ""
        if "primary_community" in merged.columns and "최종_카테고리" in merged.columns:
            sub = merged[merged["primary_community"] == cid]["최종_카테고리"]
            cat_dist = ", ".join(f"{k}:{v}" for k, v in sub.value_counts().head(5).items())

        lines.append(f"── 커뮤니티 {cid}  [{com_labels.get(cid, '')}]")
        lines.append(f"   키워드 수: {len(members)}  |  대표 논문: {paper_count}편")
        lines.append(f"   카테고리 분포: {cat_dist}")
        lines.append(f"   상위 키워드: " + ", ".join(kw for _, kw in members_with_freq[:10]))
        lines.append("")

    text = "\n".join(lines)
    (OUT / "community_summary.txt").write_text(text, encoding="utf-8")
    print(text)


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    # 1. 그래프 로드
    print("── 그래프 로드 중...")
    G_nx = nx.read_graphml(NET)
    print(f"[graph] 노드: {G_nx.number_of_nodes()}  엣지: {G_nx.number_of_edges()}")

    # 2. igraph 변환
    G_ig, idx_to_name = nx_to_igraph(G_nx)

    # 3. Leiden 커뮤니티 탐지
    print("\n── Leiden 커뮤니티 탐지 중...")
    membership = run_leiden(G_ig, resolution=1.2)

    # 4. 커뮤니티 레이블
    com_labels = make_community_labels(G_ig, membership)

    # 5. 키워드 → 커뮤니티 매핑 딕셔너리
    kw_to_com: dict[str, int] = {
        G_ig.vs[i]["name"]: membership[i]
        for i in range(G_ig.vcount())
    }

    # 6. 키워드별 커뮤니티 CSV 저장
    kw_com_df = pd.DataFrame([
        {
            "keyword":         G_ig.vs[i]["name"],
            "freq":            G_ig.vs[i]["freq"],
            "community_id":    membership[i],
            "community_label": com_labels.get(membership[i], ""),
        }
        for i in range(G_ig.vcount())
    ]).sort_values(["community_id", "freq"], ascending=[True, False])
    kw_com_df.to_csv(OUT / "keyword_communities.csv", index=False, encoding="utf-8-sig")
    print(f"[saved] keyword_communities.csv  ({len(kw_com_df)}행)")

    # 7. 카테고리 데이터 로드
    print("\n── 카테고리 데이터 로드 중...")
    cat_df  = load_categories()
    kw_df   = load_main_keywords(cat_df)

    # 8. 논문 → 커뮤니티 매핑
    print("\n── 논문 → 커뮤니티 매핑 중...")
    paper_map = map_papers_to_communities(kw_df, kw_to_com, com_labels)
    paper_map.to_csv(OUT / "community_paper_map.csv", index=False, encoding="utf-8-sig")

    # 9. 커뮤니티 × 카테고리 교차 분석
    print("\n── 교차 분석 중...")
    merged_df = cross_analysis(paper_map, cat_df, com_labels)

    # 10. 요약 저장
    print("\n── 요약 저장 중...")
    save_summary(G_ig, membership, com_labels, merged_df)

    print(f"\n✓ 완료: {OUT}")


if __name__ == "__main__":
    main()
