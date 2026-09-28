"""
compare_communities.py
======================
2025 반영 후 '새 커뮤니티'가 예전 커뮤니티와 얼마나 같은지(구성 논문 겹침) 비교.
목적: 예전에 눈으로 붙인 라벨을 새 커뮤니티에 이어붙일 수 있는지 판단.

- 라이덴 community_id 는 실행마다 바뀌므로 번호로는 못 맞춤 → 구성 논문(node_id) 겹침으로 매칭.
- 각 '새 커뮤니티'에 대해 겹침이 가장 큰 '옛 커뮤니티'를 찾아 나란히 표시.
  · shared           : 공유 논문 수
  · pct_of_new       : 새 커뮤니티 중 옛 커뮤니티와 겹치는 비율(라벨 이어붙이기 판단 핵심)
  · jaccard          : 합집합 대비 교집합
  · 판정             : pct_of_new>=0.7 이면 '안정'(예전 라벨 재사용 가능), 아니면 '재검토'

입력(자동): 옛 = ../../results/{03_network,03-2_network}/node_community.csv
            새 = ./results/{03_network,03-2_network}/node_community.csv
출력: ./results/label_compare/{internal,extended}_community_match.csv

실행: (분석_2025 폴더에서)  python compare_communities.py
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
OLD_BASE = HERE / ".." / ".."          # kci_network 루트(원본 results)
OUT = HERE / "results" / "label_compare"; OUT.mkdir(parents=True, exist_ok=True)

def load(p):
    d = pd.read_csv(p, dtype=str)
    d["node_id"] = d["node_id"].astype(str)
    return d

def comm_members(df):
    return {cid: set(g["node_id"]) for cid, g in df.groupby("community_id")}

def comm_label(df):
    # 커뮤니티당 auto_label (동일값이므로 first)
    return df.groupby("community_id")["auto_label"].first().to_dict()

def compare(tag, rel):
    old = load(OLD_BASE / "results" / rel / "node_community.csv")
    new = load(HERE / "results" / rel / "node_community.csv")
    old_m, new_m = comm_members(old), comm_members(new)
    old_l, new_l = comm_label(old), comm_label(new)

    rows = []
    for ncid, nset in sorted(new_m.items(), key=lambda x: -len(x[1])):
        best, best_share = None, -1
        for ocid, oset in old_m.items():
            s = len(nset & oset)
            if s > best_share:
                best, best_share = ocid, s
        oset = old_m[best]
        union = len(nset | oset)
        rows.append({
            "new_cid": ncid, "new_label": new_l.get(ncid, ""), "new_size": len(nset),
            "old_cid": best, "old_label": old_l.get(best, ""), "old_size": len(oset),
            "shared": best_share,
            "pct_of_new": round(best_share / len(nset), 3),
            "jaccard": round(best_share / union, 3) if union else 0,
            "판정": "안정" if best_share / len(nset) >= 0.7 else "재검토",
        })
    res = pd.DataFrame(rows)
    out = OUT / f"{tag}_community_match.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")

    stable = (res["판정"] == "안정").sum()
    print(f"\n===== [{tag}] 새 커뮤니티 {len(res)}개 =====")
    print(f"  안정(예전 라벨 재사용 가능, 겹침≥70%): {stable}/{len(res)}")
    print(res[["new_cid","new_label","new_size","old_label","shared","pct_of_new","판정"]]
          .to_string(index=False))
    print(f"  → 저장: {out}")
    return res

if __name__ == "__main__":
    compare("internal", "03_network")
    compare("extended", "03-2_network")
    print("\n해석: pct_of_new 가 높을수록 그 새 커뮤니티는 예전 것과 사실상 동일 → 예전 라벨 그대로.")
    print("      낮은(재검토) 커뮤니티만 새로 눈으로 확인해 final_label 을 붙이면 됩니다.")
