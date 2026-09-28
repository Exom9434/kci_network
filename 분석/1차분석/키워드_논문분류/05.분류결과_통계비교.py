"""
분류결과 통계 비교 스크립트
============================
A/B/C안 분류결과를 통계적으로 비교합니다.

  A안 : 03.논문_카테고리_분류결과.csv               (정의조항 키워드 + 제목+초록)
  B안 : 03.논문_카테고리_분류결과_tfidf_정의조항.csv (TF-IDF 키워드 + 제목+초록)
  C안 : 03.논문_카테고리_분류결과_tfidf_논문키워드버전.csv (TF-IDF 키워드 + 논문키워드)

[출력 통계]
  1. 카테고리 분포 비교표 (편수 + 비율 + 편차)
  2. 안별 Cohen's Kappa (A 기준)
  3. 혼동행렬 — A vs B, A vs C
  4. 카테고리별 일치율 (A 기준 precision/recall)
  5. 논문별 3안 전부 일치 / 2안 일치 / 전부 불일치 현황
  6. 분류결과 불안정 논문 목록 (3안이 전부 다른 경우)

[출력 파일]
  05.통계비교_카테고리분포.csv
  05.통계비교_카테고리일치율.csv
  05.통계비교_혼동행렬_AvsB.csv
  05.통계비교_혼동행렬_AvsC.csv
  05.통계비교_불안정논문.csv

사용법:
    python 05.분류결과_통계비교.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# ⚙️  파일 경로 설정
# ─────────────────────────────────────────────

FILES = {
    "A안": "03.논문_카테고리_분류결과.csv",
    "B안": "03.논문_카테고리_분류결과_tfidf_정의조항.csv",
    "C안": "03.논문_카테고리_분류결과_tfidf_논문키워드버전.csv",
}

SEP  = "=" * 65
SEP2 = "-" * 65


# ─────────────────────────────────────────────
# 🔧  유틸 함수
# ─────────────────────────────────────────────

def load_result(label, path):
    if not Path(path).exists():
        print(f"  [스킵] 파일 없음: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[["논문ID", "제목", "분류결과"]].copy()
    df = df.rename(columns={"분류결과": label, "제목": "제목"})
    print(f"  {label}: {len(df)}편")
    return df


def cohen_kappa(y1, y2):
    """두 분류 결과 간 Cohen's Kappa 계산"""
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return float("nan")

    # 관찰 일치율
    p_o = (y1 == y2).sum() / n

    # 기대 일치율
    p_e = sum(
        (y1 == c).sum() / n * (y2 == c).sum() / n
        for c in labels
    )

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def confusion_matrix_df(y_true, y_pred, labels):
    """혼동행렬을 DataFrame으로 반환 (행=실제, 열=예측)"""
    rows = []
    for true_cat in labels:
        row = {"실제(A안)": true_cat}
        mask_true = y_true == true_cat
        for pred_cat in labels:
            row[pred_cat] = ((y_true == true_cat) & (y_pred == pred_cat)).sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("실제(A안)")


def category_agreement(y_true, y_pred, labels):
    """카테고리별 precision / recall / F1 (A안 기준)"""
    rows = []
    for cat in labels:
        tp = ((y_true == cat) & (y_pred == cat)).sum()
        fp = ((y_true != cat) & (y_pred == cat)).sum()
        fn = ((y_true == cat) & (y_pred != cat)).sum()
        prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1     = (2 * prec * recall / (prec + recall)
                  if not (np.isnan(prec) or np.isnan(recall) or (prec + recall) == 0)
                  else float("nan"))
        rows.append({
            "카테고리":   cat,
            "A안_편수":   (y_true == cat).sum(),
            "예측_편수":  (y_pred == cat).sum(),
            "일치(TP)":  tp,
            "Precision": f"{prec:.3f}"  if not np.isnan(prec)   else "-",
            "Recall":    f"{recall:.3f}" if not np.isnan(recall) else "-",
            "F1":        f"{f1:.3f}"    if not np.isnan(f1)     else "-",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 🚀  메인
# ─────────────────────────────────────────────

def main():
    print(SEP)
    print("  분류결과 통계 비교")
    print(SEP)

    # ── 로드 ──────────────────────────────────
    print("\n[파일 로드]")
    raw = {}
    for label, path in FILES.items():
        df = load_result(label, path)
        if df is not None:
            raw[label] = df

    if len(raw) < 2:
        print("비교할 파일이 부족합니다 (최소 2개 필요).")
        return

    labels_list = list(raw.keys())
    base_label  = labels_list[0]   # A안

    # ── 공통 논문 합치기 ──────────────────────
    merged = raw[base_label][["논문ID", "제목", base_label]]
    for label in labels_list[1:]:
        merged = merged.merge(
            raw[label][["논문ID", label]],
            on="논문ID", how="outer"
        )

    print(f"\n  전체 고유 논문: {len(merged)}편")
    print(f"  모든 안에 공통: {merged.dropna().shape[0]}편")

    # ── 1. 카테고리 분포 비교 ─────────────────
    print(f"\n{SEP}")
    print("  [1] 카테고리 분포 비교")
    print(SEP)

    all_cats = sorted(
        set(cat for label in labels_list
            for cat in raw[label][label].dropna().unique())
    )
    totals = {label: len(raw[label]) for label in labels_list}

    dist_rows = []
    for cat in all_cats:
        row = {"카테고리": cat}
        for label in labels_list:
            cnt = (raw[label][label] == cat).sum()
            total = totals[label]
            row[f"{label}_편수"] = cnt
            row[f"{label}_비율"] = f"{cnt/total*100:.1f}%"
        # A안 대비 B안 편차 (있을 경우)
        if "A안" in labels_list and "B안" in labels_list:
            a_pct = (raw["A안"]["A안"] == cat).sum() / totals["A안"] * 100
            b_pct = (raw["B안"]["B안"] == cat).sum() / totals["B안"] * 100
            row["B-A_편차"] = f"{b_pct - a_pct:+.1f}%p"
        if "A안" in labels_list and "C안" in labels_list:
            a_pct = (raw["A안"]["A안"] == cat).sum() / totals["A안"] * 100
            c_pct = (raw["C안"]["C안"] == cat).sum() / totals["C안"] * 100
            row["C-A_편차"] = f"{c_pct - a_pct:+.1f}%p"
        dist_rows.append(row)

    df_dist = pd.DataFrame(dist_rows)
    print(df_dist.to_string(index=False))
    df_dist.to_csv("05.통계비교_카테고리분포.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → 저장: 05.통계비교_카테고리분포.csv")

    # ── 공통 논문만 추출 (이하 통계는 공통 기준) ─
    common = merged.dropna(subset=[base_label]).copy()
    y_base = common[base_label]

    # ── 2. Cohen's Kappa ──────────────────────
    print(f"\n{SEP}")
    print(f"  [2] Cohen's Kappa  (기준: {base_label})")
    print(SEP)

    for label in labels_list[1:]:
        common2 = merged.dropna(subset=[base_label, label])
        kappa = cohen_kappa(common2[base_label], common2[label])
        n = len(common2)
        match = (common2[base_label] == common2[label]).sum()
        print(f"  {base_label} vs {label}")
        print(f"    공통 논문:  {n}편")
        print(f"    일치율:     {match/n*100:.1f}%  ({match}/{n})")
        print(f"    Kappa:      {kappa:.4f}  ", end="")
        if kappa >= 0.80:   print("(거의 완벽한 일치)")
        elif kappa >= 0.60: print("(상당한 일치)")
        elif kappa >= 0.40: print("(중간 수준 일치)")
        elif kappa >= 0.20: print("(약한 일치)")
        else:               print("(거의 일치 안 함)")
        print()

    # ── 3. 혼동행렬 ───────────────────────────
    print(f"\n{SEP}")
    print(f"  [3] 혼동행렬  (행=A안 실제, 열=비교안 예측)")
    print(SEP)

    for label in labels_list[1:]:
        common2 = merged.dropna(subset=[base_label, label])
        cats    = sorted(set(common2[base_label]) | set(common2[label]))
        cm      = confusion_matrix_df(common2[base_label], common2[label], cats)
        print(f"\n  A안 vs {label}  (n={len(common2)})\n")
        print(cm.to_string())
        cm.to_csv(f"05.통계비교_혼동행렬_Avs{label[0]}.csv", encoding="utf-8-sig")
        print(f"\n  → 저장: 05.통계비교_혼동행렬_Avs{label[0]}.csv")

    # ── 4. 카테고리별 일치율 ──────────────────
    print(f"\n{SEP}")
    print(f"  [4] 카테고리별 일치율  (기준: {base_label})")
    print(SEP)

    agree_out = []
    for label in labels_list[1:]:
        common2 = merged.dropna(subset=[base_label, label])
        cats    = sorted(set(common2[base_label]) | set(common2[label]))
        df_ag   = category_agreement(common2[base_label], common2[label], cats)
        df_ag.insert(0, "비교안", label)
        agree_out.append(df_ag)
        print(f"\n  {base_label} vs {label}\n")
        print(df_ag.to_string(index=False))

    df_agree_all = pd.concat(agree_out, ignore_index=True)
    df_agree_all.to_csv("05.통계비교_카테고리일치율.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → 저장: 05.통계비교_카테고리일치율.csv")

    # ── 5. 논문별 안 일치 현황 ────────────────
    print(f"\n{SEP}")
    print("  [5] 논문별 안 일치 현황")
    print(SEP)

    common_all = merged.dropna()
    if len(labels_list) >= 3:
        all_same  = (common_all[labels_list[0]] == common_all[labels_list[1]]) & \
                    (common_all[labels_list[1]] == common_all[labels_list[2]])
        ab_same   = common_all[labels_list[0]] == common_all[labels_list[1]]
        ac_same   = common_all[labels_list[0]] == common_all[labels_list[2]]
        bc_same   = common_all[labels_list[1]] == common_all[labels_list[2]]
        none_same = ~ab_same & ~ac_same & ~bc_same

        n = len(common_all)
        print(f"  공통 논문 (3안 모두): {n}편\n")
        print(f"  3안 전부 일치:  {all_same.sum()}편  ({all_same.sum()/n*100:.1f}%)")
        print(f"  A=B만 일치:     {(ab_same & ~ac_same).sum()}편")
        print(f"  A=C만 일치:     {(ac_same & ~ab_same).sum()}편")
        print(f"  B=C만 일치:     {(bc_same & ~ab_same & ~ac_same).sum()}편")
        print(f"  3안 전부 불일치:{none_same.sum()}편  ({none_same.sum()/n*100:.1f}%)")
    elif len(labels_list) == 2:
        common2 = merged.dropna(subset=[labels_list[0], labels_list[1]])
        same    = common2[labels_list[0]] == common2[labels_list[1]]
        n = len(common2)
        print(f"  공통 논문: {n}편")
        print(f"  일치:      {same.sum()}편  ({same.sum()/n*100:.1f}%)")
        print(f"  불일치:    {(~same).sum()}편  ({(~same).sum()/n*100:.1f}%)")

    # ── 6. 불안정 논문 목록 ───────────────────
    print(f"\n{SEP}")
    print("  [6] 분류 불안정 논문  (3안이 전부 다른 경우)")
    print(SEP)

    if len(labels_list) >= 3:
        unstable = common_all[none_same][["논문ID", "제목"] + labels_list]
        print(f"  총 {len(unstable)}편\n")
        pd.set_option("display.max_colwidth", 35)
        print(unstable.head(30).to_string(index=False))
        unstable.to_csv("05.통계비교_불안정논문.csv", index=False, encoding="utf-8-sig")
        print(f"\n  → 저장: 05.통계비교_불안정논문.csv")
    else:
        print("  (3안 데이터가 모두 있어야 계산 가능)")

    print(f"\n{SEP}")
    print("  완료")
    print(SEP)


if __name__ == "__main__":
    main()
