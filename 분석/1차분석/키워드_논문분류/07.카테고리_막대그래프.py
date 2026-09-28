"""
AI 법학 논문 카테고리 분포 막대그래프 생성
- 07.카테고리_분포_막대그래프.png  : 단일 카테고리 분포 (융복합 제외)
- 07.융복합_세부_막대그래프.png    : 융복합 세부 분류 (5편 이상)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib

INPUT_CSV = "05.논문_카테고리_최종분류.csv"

df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')

# ── 1. 단일 카테고리 분포 (융복합 제외) ───────────────────────────────────────
single = df[~df['최종카테고리'].str.startswith('융복합', na=False)]
counts = single['최종카테고리'].value_counts().sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(counts.index, counts.values, color='steelblue')
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xlabel('논문 수')
ax.set_title('AI 법학 논문 카테고리 분포 (융복합 제외)', fontsize=13, pad=12)
ax.set_xlim(0, counts.max() * 1.12)
plt.tight_layout()
plt.savefig('07.카테고리_분포_막대그래프.png', dpi=150)
plt.close()
print('저장 완료: 07.카테고리_분포_막대그래프.png')

# ── 2. 융복합 세부 분류 (5편 이상) ────────────────────────────────────────────
fusion = df[df['최종카테고리'].str.startswith('융복합', na=False)]
fc = fusion['최종카테고리'].value_counts()
fc = fc[fc >= 5].sort_values()

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(fc.index, fc.values, color='darkorange')
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xlabel('논문 수')
ax.set_title('AI 법학 논문 융복합 세부 분류 (5편 이상)', fontsize=13, pad=12)
ax.set_xlim(0, fc.max() * 1.12)
plt.tight_layout()
plt.savefig('07.융복합_세부_막대그래프.png', dpi=150)
plt.close()
print('저장 완료: 07.융복합_세부_막대그래프.png')
