"""주제군별 분과 집중도의 시기별 변화와 인용 허브성의 관계 점검 (2026-09-27).
출력: results/11_category_concentration/
  by_period.csv        주제군 x 시기: 최대 분과 점유율, HHI, 융복합 비율, 인공지능법 비율
  ai_law_location.csv  인공지능법 분과(단일+융복합) 논문이 어느 주제군에 있는지, 시기별
  robustness.csv       집중도 정의별 교차인용/자기인용률 상관 (n=10)
분과 = 확정 + 검토필요 CSV의 최종_카테고리 (04.keyword_leiden.py와 동일 경로)
"""
import pandas as pd, numpy as np, os
OUT='results/11_category_concentration'; os.makedirs(OUT,exist_ok=True)
cat=pd.concat([pd.read_csv(f,encoding='utf-8-sig')[['논문ID','최종_카테고리']] for f in
    ['KCI_AI_논문_카테고리_확정.csv','KCI_AI_논문_카테고리_검토필요.csv']]).drop_duplicates('논문ID',keep='last')
yr=pd.read_csv('00.KCI_AI_논문_상세_및_인용데이터.csv')[['source_id','pub_year']].drop_duplicates('source_id').rename(columns={'source_id':'논문ID'})
jr=pd.read_csv('../../../데이터/KCI/KCI_AI_논문_기본정보_목록_201601_202512.csv',encoding='utf-8-sig')[['논문ID','학술지명']]
m=pd.read_csv('results/04_keyword_network/community_paper_map.csv',encoding='utf-8-sig')
x=m.merge(yr,on='논문ID').merge(cat,on='논문ID').merge(jr,on='논문ID').rename(columns={'최종_카테고리':'cat'})
per=lambda y:'2016-18' if y<=2018 else '2019-20' if y<=2020 else '2021-22' if y<=2022 else '2023-25'
x['per']=x.pub_year.map(per); x['fus']=x.cat.str.startswith('융복합')
x['aiS']=x.cat.eq('인공지능법'); x['aiA']=x.cat.str.contains('인공지능법')
big=x.community_label.value_counts(); big=big[big>=20].index
def stats(g):
    si=g[~g.fus]; vc=si.cat.value_counts(); jv=g.학술지명.value_counts()
    return dict(n=len(g),최대분과=vc.index[0] if len(vc) else '',
        점유_전체분모=vc.iloc[0]/len(g) if len(vc) else np.nan, 점유_단일분모=vc.iloc[0]/len(si) if len(si) else np.nan,
        HHI_단일=((vc/len(si))**2).sum() if len(si) else np.nan, HHI_융복합별도=((g.cat.value_counts()/len(g))**2).sum(),
        융복합=g.fus.mean(), 인공지능법_단일=g.aiS.mean(), 인공지능법_포함=g.aiA.mean(),
        학술지_HHI=((jv/len(g))**2).sum(), 학술지_상위3=jv.iloc[:3].sum()/len(g))
rows=[dict(주제군=c,시기=p,**stats(g)) for c in big for p,g in list(x[x.community_label==c].groupby('per'))+[('전체',x[x.community_label==c])]]
bp=pd.DataFrame(rows); bp.round(3).to_csv(f'{OUT}/by_period.csv',index=False,encoding='utf-8-sig')
a=x[x.aiA]; pd.crosstab(a.community_label,a.per,margins=True).to_csv(f'{OUT}/ai_law_location.csv',encoding='utf-8-sig')
s=pd.read_csv('results/03_citation_topic_flow/topic_citation_summary.csv',encoding='utf-8-sig')
s['교차인용_논문당']=s.incoming_from_other_topic_citations/s.paper_count_all
r=bp[bp.시기=='전체'].merge(s[['community_label','교차인용_논문당','same_topic_ratio_pct']],left_on='주제군',right_on='community_label')
rk=lambda a,b:np.corrcoef(a.rank(),b.rank())[0,1]
rob=pd.DataFrame([dict(지표=v,r_교차인용=np.corrcoef(r[v],r.교차인용_논문당)[0,1],rho_교차인용=rk(r[v],r.교차인용_논문당),
    r_자기인용률=np.corrcoef(r[v],r.same_topic_ratio_pct)[0,1],rho_자기인용률=rk(r[v],r.same_topic_ratio_pct))
    for v in ['점유_전체분모','점유_단일분모','HHI_단일','HHI_융복합별도','융복합','학술지_HHI','학술지_상위3']])
rob.round(2).to_csv(f'{OUT}/robustness.csv',index=False,encoding='utf-8-sig'); print(rob.round(2).to_string())
