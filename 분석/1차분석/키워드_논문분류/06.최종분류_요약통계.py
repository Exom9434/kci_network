"""
05.논문_카테고리_최종분류.csv 기준으로 분류 결과 요약 통계를 CSV로 저장
출력 파일: 06.최종분류_요약통계.csv
"""

import pandas as pd

INPUT_CSV = "05.논문_카테고리_최종분류.csv"
OUTPUT_CSV = "06.최종분류_요약통계.csv"

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
total = len(df)

# 카테고리별 편수 집계
vc = df["최종카테고리"].value_counts().reset_index()
vc.columns = ["카테고리", "편수"]
vc["비율(%)"] = (vc["편수"] / total * 100).round(1)

# 유형 구분 (단독 / 융복합 / 기초법)
def get_type(cat):
    if str(cat).startswith("융복합"):
        return "융복합"
    elif str(cat).startswith("기초법"):
        return "기초법"
    else:
        return "단독"

vc["유형"] = vc["카테고리"].apply(get_type)

# 소계 행 추가
summary_rows = []
for typ in ["단독", "융복합", "기초법"]:
    sub = vc[vc["유형"] == typ]
    summary_rows.append({
        "카테고리": f"[소계] {typ}",
        "편수": sub["편수"].sum(),
        "비율(%)": round(sub["편수"].sum() / total * 100, 1),
        "유형": typ
    })
summary_rows.append({
    "카테고리": "[합계]",
    "편수": total,
    "비율(%)": 100.0,
    "유형": ""
})

result = pd.concat([vc, pd.DataFrame(summary_rows)], ignore_index=True)
result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# 콘솔 출력
print(f"총 논문: {total}편\n")
for typ in ["단독", "융복합", "기초법"]:
    sub = vc[vc["유형"] == typ]
    print(f"=== {typ} ({sub['편수'].sum()}편, {sub['편수'].sum()/total*100:.1f}%) ===")
    print(sub[["카테고리", "편수", "비율(%)"]].to_string(index=False))
    print()

print(f"저장 완료: {OUTPUT_CSV}")
