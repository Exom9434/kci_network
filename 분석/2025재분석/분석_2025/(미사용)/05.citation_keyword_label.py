"""
05.citation_keyword_label.py
-----------------------------
03/03-2 인용 네트워크 커뮤니티에 키워드 기반 레이블 부여 (방법 A)

방법:
  각 인용 커뮤니티 소속 논문들의 키워드를 전처리 후 집계 →
  상위 빈도 키워드 3개를 키워드 레이블로 사용

출력:
  results/03_network/
    community_keyword_label.csv    — 커뮤니티별 키워드 레이블 + LLM 레이블 비교
    community_label_comparison.png — 비교 시각화
  results/03-2_network/
    (동일)
"""

from pathlib import Path
import importlib.util
import collections

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ────────────────────────────── 경로 설정 ─────────────────────────────
BASE  = Path(__file__).parent
DATA  = BASE / "00.KCI_AI_논문_상세_및_인용데이터.csv"
OUT03  = BASE / "results" / "03_network"
OUT032 = BASE / "results" / "03-2_network"

# ────────────────── keyword_build 전처리 함수 임포트 ──────────────────
_spec = importlib.util.spec_from_file_location("kb", BASE / "04.keyword_build.py")
_kb   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kb)
parse_keywords = _kb.parse_keywords

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
# 1. 논문 키워드 사전 구축 (paper_id → 정제된 키워드 리스트)
# ══════════════════════════════════════════════════════════════════════
def build_paper_kw_dict() -> dict[str, list[str]]:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df.drop_duplicates(subset="source_id")[["source_id", "keywords"]]
    result = {}
    for _, row in df.iterrows():
        pid = str(row["source_id"])
        if pd.isna(row["keywords"]):
            result[pid] = []
        else:
            result[pid] = parse_keywords(str(row["keywords"]))
    print(f"[paper_kw] 논문 {len(result):,}편 키워드 로드 완료")
    return result


# ══════════════════════════════════════════════════════════════════════
# 2. 커뮤니티별 키워드 집계 → 상위 키워드 레이블
# ══════════════════════════════════════════════════════════════════════
def make_keyword_labels(
    node_com: pd.DataFrame,
    paper_kw: dict[str, list[str]],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    커뮤니티별로 소속 논문의 키워드를 합산 →
    상위 top_n 키워드를 keyword_label로 반환
    """
    com_freq: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    matched = 0

    for _, row in node_com.iterrows():
        pid = str(row["node_id"])
        cid = row["community_id"]
        kws = paper_kw.get(pid, [])
        if kws:
            com_freq[cid].update(kws)
            matched += 1

    print(f"[label] 키워드 매칭된 논문: {matched}/{len(node_com)}편")

    rows = []
    for cid, counter in sorted(com_freq.items()):
        top_kws = [kw for kw, _ in counter.most_common(top_n)]
        rows.append({
            "community_id":   cid,
            "keyword_label":  " / ".join(top_kws[:3]),   # 레이블용 상위 3개
            "top_keywords":   " / ".join(top_kws),        # 참고용 상위 5개
            "kw_detail":      ", ".join(
                f"{kw}({cnt})" for kw, cnt in counter.most_common(10)
            ),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 3. LLM 레이블 vs 키워드 레이블 비교표 생성
# ══════════════════════════════════════════════════════════════════════
def make_comparison(
    community_summary: pd.DataFrame,
    kw_label_df: pd.DataFrame,
) -> pd.DataFrame:
    # 03은 'size', 03-2는 'size_ai' 컬럼명 사용
    size_col = "size" if "size" in community_summary.columns else "size_ai"
    community_summary = community_summary.rename(columns={size_col: "size"})

    merged = community_summary[
        ["community_id", "label_id", "auto_label", "final_label", "size",
         "top1_category", "top1_ratio"]
    ].merge(kw_label_df, on="community_id", how="left")

    # final_label이 있으면 우선, 없으면 auto_label 사용
    merged["llm_label"] = merged["final_label"].fillna(merged["auto_label"])
    merged = merged[[
        "community_id", "label_id", "size",
        "llm_label", "keyword_label", "top_keywords", "kw_detail",
        "top1_category", "top1_ratio",
    ]]
    return merged


# ══════════════════════════════════════════════════════════════════════
# 4. 비교 시각화
# ══════════════════════════════════════════════════════════════════════
def plot_comparison(comp: pd.DataFrame, out_path: Path, title: str) -> None:
    comp_sorted = comp.sort_values("size", ascending=True).reset_index(drop=True)
    n = len(comp_sorted)

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.55 + 2)))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    y = range(n)
    bar_h = 0.35

    bars = ax.barh([i + bar_h/2 for i in y], comp_sorted["size"],
                   height=bar_h, color="#457B9D", alpha=0.75, label="논문 수")

    ax.set_yticks(list(y))
    ax.set_yticklabels(comp_sorted["label_id"], fontsize=9)
    ax.set_xlabel("논문 수", fontsize=10)
    ax.set_title(title, fontsize=12, pad=14)

    # 각 바 오른쪽에 두 레이블 표시
    for i, (_, row) in enumerate(comp_sorted.iterrows()):
        x = row["size"]
        llm = str(row["llm_label"])[:28]
        kw  = str(row["keyword_label"])[:35]
        ax.text(x + 1, i + bar_h/2 + 0.18, f"LLM: {llm}",
                va="center", fontsize=7, color="#1D3557")
        ax.text(x + 1, i + bar_h/2 - 0.18, f"KW:  {kw}",
                va="center", fontsize=7, color="#E63946")

    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════
# 5. community_summary.csv 에 keyword_label 컬럼 추가 저장
# ══════════════════════════════════════════════════════════════════════
def update_summary(summary_path: Path, comp: pd.DataFrame) -> None:
    orig = pd.read_csv(summary_path, encoding="utf-8-sig")
    kw_cols = comp[["community_id", "keyword_label", "top_keywords"]]
    updated = orig.merge(kw_cols, on="community_id", how="left")
    updated.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[updated] {summary_path.name}  (keyword_label 컬럼 추가)")


# ══════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════
def process(net_name: str, out_dir: Path, paper_kw: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  {net_name} 처리 중")
    print(f"{'='*55}")

    node_com = pd.read_csv(out_dir / "node_community.csv", encoding="utf-8-sig")
    com_sum  = pd.read_csv(out_dir / "community_summary.csv", encoding="utf-8-sig")

    # 03-2는 AI 논문(node_type=ai)만 대상
    if "node_type" in node_com.columns:
        node_com = node_com[node_com["node_type"] == "ai"]
        print(f"[filter] AI 논문만 대상: {len(node_com)}편")

    kw_label_df = make_keyword_labels(node_com, paper_kw)

    comp = make_comparison(com_sum, kw_label_df)
    comp.to_csv(out_dir / "community_keyword_label.csv",
                index=False, encoding="utf-8-sig")
    print(f"[saved] community_keyword_label.csv")

    # 콘솔 출력
    print(f"\n── 레이블 비교표")
    for _, row in comp.iterrows():
        print(f"  {row['label_id']} ({row['size']:3d}편)")
        print(f"    LLM: {row['llm_label']}")
        print(f"    KW:  {row['keyword_label']}")

    plot_comparison(
        comp,
        out_dir / "community_label_comparison.png",
        f"{net_name} 커뮤니티 레이블 비교 (LLM vs 키워드)",
    )

    # community_summary에 keyword_label 컬럼 추가
    update_summary(out_dir / "community_summary.csv", comp)


def main() -> None:
    paper_kw = build_paper_kw_dict()
    process("03 내부 인용 네트워크",  OUT03,  paper_kw)
    process("03-2 확장 인용 네트워크", OUT032, paper_kw)
    print(f"\n✓ 전체 완료")


if __name__ == "__main__":
    main()
