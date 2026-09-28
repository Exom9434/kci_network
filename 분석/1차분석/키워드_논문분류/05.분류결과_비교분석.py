"""
분류결과 비교 분석 스크립트
============================
세 가지 분류 결과를 나란히 비교합니다.

  A안  : 03.논문_카테고리_분류결과.csv               (정의조항 키워드 + 제목+초록)
  B안  : 03.논문_카테고리_분류결과_tfidf_정의조항.csv (TF-IDF 키워드 + 제목+초록)
  C안  : 03.논문_카테고리_분류결과_tfidf_논문키워드버전.csv (TF-IDF 키워드 + 논문키워드)

사용법:
    python 05.분류결과_비교분석.py
"""

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# ⚙️  파일 경로
# ─────────────────────────────────────────────

FILES = {
    "A안(정의조항+제목초록)":      "03.논문_카테고리_분류결과.csv",
    "B안(TF-IDF+제목초록)":        "03.논문_카테고리_분류결과_tfidf_정의조항.csv",
    "C안(TF-IDF+논문키워드)":      "03.논문_카테고리_분류결과_tfidf_논문키워드버전.csv",
}

# ─────────────────────────────────────────────
# 🔧  유틸
# ─────────────────────────────────────────────

SEP = "=" * 65

def load(label, path):
    if not Path(path).exists():
        print(f"  [스킵] 파일 없음: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"분류결과": label})
    print(f"  {label}: {len(df)}편  ({path})")
    return df[["논문ID", label]]


def category_dist(series, label):
    counts = series.value_counts()
    total  = len(series)
    rows   = []
    for cat, cnt in counts.items():
        rows.append({
            "카테고리": cat,
            f"편수({label})": cnt,
            f"비율({label})": f"{cnt/total*100:.1f}%",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 🚀  메인
# ─────────────────────────────────────────────

def main():
    print(SEP)
    print("  분류결과 비교 분석")
    print(SEP)

    # ── 1. 로드 ───────────────────────────────
    print("\n[파일 로드]")
    dfs = {}
    for label, path in FILES.items():
        df = load(label, path)
        if df is not None:
            dfs[label] = df

    if not dfs:
        print("분석할 파일이 없습니다.")
        return

    # ── 2. 카테고리 분포 비교 ─────────────────
    print(f"\n{SEP}")
    print("  [1] 카테고리 분포 비교")
    print(SEP)

    all_cats = sorted(
        set(cat for label, df in dfs.items()
            for cat in df[label].unique())
    )

    dist_frames = []
    for label, df in dfs.items():
        dist_frames.append(category_dist(df[label], label))

    # 전체 카테고리 기준으로 합치기
    merged = pd.DataFrame({"카테고리": all_cats})
    for frame in dist_frames:
        merged = merged.merge(frame, on="카테고리", how="left")
    merged = merged.fillna("-")

    print(merged.to_string(index=False))

    # ── 3. 기초법 세부 현황 ───────────────────
    print(f"\n{SEP}")
    print("  [2] 기초법 세부 현황")
    print(SEP)

    for label, df in dfs.items():
        total = len(df)
        kw_none  = (df[label] == "기초법(키워드없음)").sum()
        threshold= (df[label] == "기초법(임계값미달)").sum()
        kicho    = (df[label] == "기초법").sum()  # 구분 없는 경우
        sub = kw_none + threshold + kicho
        print(f"  {label}")
        print(f"    기초법(키워드없음): {kw_none}편  ({kw_none/total*100:.1f}%)")
        print(f"    기초법(임계값미달): {threshold}편  ({threshold/total*100:.1f}%)")
        if kicho:
            print(f"    기초법(구분없음):   {kicho}편  ({kicho/total*100:.1f}%)")
        print(f"    소계:              {sub}편  ({sub/total*100:.1f}%)")
        print()

    # ── 4. A안 기준 일치율 ────────────────────
    if len(dfs) < 2:
        return

    base_label = list(dfs.keys())[0]
    base_df    = dfs[base_label].rename(columns={base_label: "A안"})

    print(f"\n{SEP}")
    print(f"  [3] A안 대비 일치율  (기준: {base_label})")
    print(SEP)

    for label, df in list(dfs.items())[1:]:
        merged2 = base_df.merge(df, on="논문ID", how="inner")
        total   = len(merged2)
        match   = (merged2["A안"] == merged2[label]).sum()
        print(f"  A안 vs {label}")
        print(f"    공통 논문: {total}편")
        print(f"    일치:      {match}편  ({match/total*100:.1f}%)")
        print(f"    불일치:    {total-match}편  ({(total-match)/total*100:.1f}%)")
        print()

    # ── 5. A안과 불일치 논문 상세 (상위 20개) ─
    print(f"\n{SEP}")
    print("  [4] A안 vs B안 불일치 논문 샘플 (최대 20개)")
    print(SEP)

    labels = list(dfs.keys())
    if len(labels) >= 2:
        b_label = labels[1]
        b_df    = dfs[b_label]
        merged3 = base_df.merge(b_df, on="논문ID", how="inner")
        diff    = merged3[merged3["A안"] != merged3[b_label]]
        # 제목 포함 위해 원본 A안 CSV에서 조회
        try:
            a_full = pd.read_csv(FILES[base_label], encoding="utf-8-sig")[["논문ID", "제목"]]
            diff   = diff.merge(a_full, on="논문ID", how="left")
            cols   = ["논문ID", "제목", "A안", b_label]
        except Exception:
            cols   = ["논문ID", "A안", b_label]

        print(f"  총 불일치: {len(diff)}편\n")
        pd.set_option("display.max_colwidth", 40)
        print(diff[cols].head(20).to_string(index=False))

    # ── 6. 결과 CSV 저장 ──────────────────────
    print(f"\n{SEP}")
    print("  [5] 3안 통합 비교표 저장")
    print(SEP)

    all_ids = pd.DataFrame({"논문ID": sorted(
        set(id_ for df in dfs.values() for id_ in df["논문ID"])
    )})
    result = all_ids
    for label, df in dfs.items():
        result = result.merge(df, on="논문ID", how="left")

    # 제목 붙이기
    try:
        a_full = pd.read_csv(FILES[base_label], encoding="utf-8-sig")[["논문ID", "제목", "발행연도"]]
        result = a_full.merge(result, on="논문ID", how="right")
    except Exception:
        pass

    out_path = "05.분류결과_3안_비교표.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out_path}  ({len(result)}편)")
    print(SEP)


if __name__ == "__main__":
    main()
