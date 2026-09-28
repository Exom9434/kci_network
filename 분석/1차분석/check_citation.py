"""
인용 데이터 품질 확인 스크립트
실행: python check_citation.py
"""
import pandas as pd

df = pd.read_csv('00.KCI_AI_논문_상세_및_인용데이터.csv')
papers = df.drop_duplicates(subset='source_id')[['source_id','pub_year']].copy()
ai_ids = set(papers['source_id'].astype(str))

print("=" * 55)
print("① 기본 현황")
print("=" * 55)
print(f"전체 행 수       : {len(df):,}")
print(f"고유 AI 논문 수  : {len(papers):,}")

# 인용 행 vs 인용 없는 행
has_ref  = df[df['target_arti_id'].notna() & (df['target_arti_id'] != '')]
no_ref   = df[df['target_arti_id'].isna()  | (df['target_arti_id'] == '')]
print(f"인용 관계 행     : {len(has_ref):,}")
print(f"인용 없는 행     : {len(no_ref):,}")

# 논문 단위 - 인용 있는 논문 vs 없는 논문
papers_with_ref = set(has_ref['source_id'].astype(str))
print(f"\n인용 데이터 있는 논문 : {len(papers_with_ref):,}개 ({len(papers_with_ref)/len(papers)*100:.1f}%)")
print(f"인용 데이터 없는 논문 : {len(papers) - len(papers_with_ref):,}개")

print()
print("=" * 55)
print("② 논문당 인용 수 분포")
print("=" * 55)
ref_counts = has_ref.groupby('source_id').size()
print(ref_counts.describe().round(1).to_string())
print(f"\n인용 0건 논문 제외 기준 중앙값: {ref_counts.median():.0f}건")

print()
print("=" * 55)
print("③ target_arti_id 유형 분석")
print("=" * 55)
# KCI 논문 ID 형식: ART로 시작
target_ids = has_ref['target_arti_id'].dropna().astype(str)
is_kci     = target_ids.str.startswith('ART')
is_ai_paper = target_ids.isin(ai_ids)

print(f"전체 인용 관계        : {len(target_ids):,}건")
print(f"KCI 논문 (ART*)       : {is_kci.sum():,}건 ({is_kci.mean()*100:.1f}%)")
print(f"  └ 우리 AI논문 2014개 내부 : {is_ai_paper.sum():,}건 ({is_ai_paper.mean()*100:.1f}%)")
print(f"  └ KCI but AI논문 외부    : {(is_kci & ~is_ai_paper).sum():,}건")
print(f"KCI 외부 (비ART*)     : {(~is_kci).sum():,}건 ({(~is_kci).mean()*100:.1f}%)")

print()
print("=" * 55)
print("④ 연도별 인용 데이터 보유 비율")
print("=" * 55)
yr = papers.copy()
yr['has_ref'] = yr['source_id'].astype(str).isin(papers_with_ref)
tbl = yr.groupby('pub_year')['has_ref'].agg(['sum','count'])
tbl.columns = ['인용있음', '전체']
tbl['비율(%)'] = (tbl['인용있음'] / tbl['전체'] * 100).round(1)
print(tbl.to_string())

print()
print("=" * 55)
print("⑤ AI논문 → AI논문 내부 인용 네트워크 규모")
print("=" * 55)
internal = has_ref[has_ref['target_arti_id'].astype(str).isin(ai_ids)]
src_in_net = internal['source_id'].nunique()
tgt_in_net = internal['target_arti_id'].nunique()
print(f"내부 인용 엣지 수    : {len(internal):,}건")
print(f"인용하는 논문 (출발) : {src_in_net:,}개")
print(f"인용받는 논문 (도착) : {tgt_in_net:,}개")
print(f"연결된 총 고유 노드  : {len(set(internal['source_id']) | set(internal['target_arti_id'])):,}개")
print()
print("연도별 내부 인용 엣지 수:")
src_year = papers.set_index('source_id')['pub_year'].to_dict()
internal2 = internal.copy()
internal2['src_year'] = internal2['source_id'].map(src_year)
print(internal2.groupby('src_year').size().to_string())
