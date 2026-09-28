"""
메일 수치 검증 스크립트
=====================================
05.논문_카테고리_최종분류.csv의 실제 값과
메일에 기재된 수치를 대조하여 일치 여부를 확인합니다.
"""

import pandas as pd

INPUT_CSV = "05.논문_카테고리_최종분류.csv"

# ─────────────────────────────────────
# 메일에 기재된 수치 (수정 시 여기만 변경)
# ─────────────────────────────────────
MAIL_TOTAL = 1885

MAIL_TABLE = {
    "데이터법":        474,
    "인공지능법":       411,
    "융복합":          404,   # 융복합 전체 합계
    "공법":           133,
    "경제법":          84,
    "형사법":          64,
    "지식재산권법":      54,
    "사회보장법":        54,
    "민사법":          44,
    "금융법":          43,
    "의료법":          38,
    "리걸테크":         17,
    "조세법":          16,
    "노동법":          14,
    "자율무기":          7,
    "자율주행자동차법":     2,
    "기초법(미분류)":    26,   # 기초법 전체 합계
}

# ─────────────────────────────────────
# 실제 데이터 집계
# ─────────────────────────────────────
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
col = [c for c in df.columns if '분류' in c or '카테고리' in c][-1]
vc = df[col].value_counts()

real = {}
for cat, cnt in vc.items():
    if '융복합' in cat:
        real['융복합'] = real.get('융복합', 0) + cnt
    elif '기초법' in cat:
        real['기초법(미분류)'] = real.get('기초법(미분류)', 0) + cnt
    else:
        real[cat] = cnt

real_total = len(df)

# ─────────────────────────────────────
# 검증 출력
# ─────────────────────────────────────
print("=" * 55)
print("  메일 수치 검증")
print("=" * 55)

all_ok = True

# 총 편수
total_ok = real_total == MAIL_TOTAL
status = "✅" if total_ok else f"❌  실제: {real_total}편"
print(f"\n  총 논문 수  메일: {MAIL_TOTAL}편  {status}")
if not total_ok:
    all_ok = False

# 카테고리별
print(f"\n  {'카테고리':<16} {'메일':>6}   {'실제':>6}   결과")
print("  " + "-" * 45)
for cat, mail_cnt in MAIL_TABLE.items():
    real_cnt = real.get(cat, 0)
    ok = real_cnt == mail_cnt
    if not ok:
        all_ok = False
    status = "✅" if ok else f"❌  차이: {real_cnt - mail_cnt:+d}"
    print(f"  {cat:<16} {mail_cnt:>6}편  {real_cnt:>6}편  {status}")

# 메일 표 합계 검증
mail_sum = sum(MAIL_TABLE.values())
real_sum = real_total
print(f"\n  {'표 합계':<16} {mail_sum:>6}편  {real_sum:>6}편  "
      + ("✅" if mail_sum == real_sum else f"❌  차이: {real_sum - mail_sum:+d}"))

print("\n" + "=" * 55)
if all_ok:
    print("  ✅ 모든 수치 일치 — 메일 발송 가능")
else:
    print("  ❌ 불일치 항목 있음 — 메일 수정 필요")
print("=" * 55)
