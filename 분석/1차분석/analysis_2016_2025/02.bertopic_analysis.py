"""
KCI 인공지능 논문 BERTopic 토픽 모델링 스크립트

사전 설치 필요:
    pip install bertopic sentence-transformers umap-learn hdbscan plotly

입력: 00.KCI_AI_논문_상세_및_인용데이터.csv
출력:
    04_topic_overview.html         전체 토픽 키워드 바차트
    05_topics_over_time.html       연도별 토픽 비중 변화
    06_topic_heatmap.html          연도 × 토픽 히트맵
    07_topic_documents_sample.txt  토픽별 대표 논문 목록
"""

import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import re, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
INPUT_FILE      = '00.KCI_AI_논문_상세_및_인용데이터.csv'
OUT_DIR         = 'results/02_bertopic'
os.makedirs(OUT_DIR, exist_ok=True)
# PNG 저장을 위해 kaleido 필요: pip install kaleido

def save_fig(fig, name):
    """HTML + PNG 동시 저장 (논문용 scale=2 고해상도)"""
    fig.write_html(f'{OUT_DIR}/{name}.html')
    fig.write_image(f'{OUT_DIR}/{name}.png', scale=2)
    print(f"  ✅ {OUT_DIR}/{name}.html / .png")
EMBEDDING_MODEL = 'woong0322/ko-legal-sbert-finetuned'   # 한국어 법률 특화 SBERT
# 대안 (다운로드 더 빠름): 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

NR_TOPICS    = 'auto'   # 'auto' 또는 정수(예: 15)로 토픽 수 지정
MIN_TOPIC_SIZE = 10     # 하나의 토픽을 이루는 최소 문서 수 (너무 잘게 쪼개지면 올리기)
TOP_N_WORDS  = 10       # 토픽당 상위 키워드 수

# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════════════
print("▶ 데이터 로드 중...")
df = pd.read_csv(INPUT_FILE)
papers = df.drop_duplicates(subset='source_id')[
    ['source_id', 'title', 'pub_year', 'abstract', 'keywords']].copy()

print(f"  전체 고유 논문: {len(papers):,}개")

# 텍스트 결합: 제목(2배 가중) + 초록 + 키워드
def build_text(row):
    parts = []
    if pd.notna(row['title'])    and str(row['title']).strip():
        parts.append(str(row['title']).strip() + ' ' + str(row['title']).strip())  # 제목 2회
    if pd.notna(row['abstract']) and str(row['abstract']).strip():
        parts.append(str(row['abstract']).strip())
    if pd.notna(row['keywords']) and str(row['keywords']).strip():
        parts.append(str(row['keywords']).strip())
    return ' '.join(parts)

papers['text'] = papers.apply(build_text, axis=1)
papers = papers[papers['text'].str.len() > 30].copy()   # 너무 짧은 것 제외
papers = papers.sort_values('pub_year').reset_index(drop=True)

docs  = papers['text'].tolist()
years = papers['pub_year'].tolist()

print(f"  분석 대상: {len(docs):,}개 (텍스트 있는 것)")
print(f"  연도 범위: {min(years)} ~ {max(years)}")
print()

# ══════════════════════════════════════════════════════════════════
# 2. 임베딩 모델 로드
# ══════════════════════════════════════════════════════════════════
print(f"▶ 임베딩 모델 로드 중: {EMBEDDING_MODEL}")
print("  (첫 실행 시 모델 다운로드로 수 분 소요될 수 있습니다)")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("  모델 로드 완료")
print()

# ══════════════════════════════════════════════════════════════════
# 3. BERTopic 구성 및 학습
# ══════════════════════════════════════════════════════════════════
print("▶ BERTopic 학습 중...")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=MIN_TOPIC_SIZE,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# 한국어 불용어 처리용 CountVectorizer
korean_stopwords = [
    '있다', '이다', '하다', '되다', '이', '가', '을', '를', '의', '에', '은', '는',
    '그', '이', '저', '것', '수', '및', '등', '또한', '따라', '위해', '통해',
    '대한', '관한', '관련', '제', '적', '으로', '에서', '에도', '에서의',
    '본', '본고', '논문', '연구', '분석', '검토', '고찰', '방안', '문제',
    '대해', '대하여', '있는', '있으며', '있음', '있어', '한다', '한다고',
    'the', 'of', 'and', 'in', 'a', 'to', 'is', 'for', 'this', 'that',
    'with', 'on', 'are', 'be', 'as', 'by', 'an', 'it', 'at', 'or',
]

vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=korean_stopwords,
    min_df=3
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    nr_topics=NR_TOPICS,
    top_n_words=TOP_N_WORDS,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)
papers['topic'] = topics

nr_topics_found = len(set(topics)) - (1 if -1 in topics else 0)
outlier_ratio   = topics.count(-1) / len(topics) * 100
print(f"\n  발견된 토픽 수: {nr_topics_found}개")
print(f"  아웃라이어(-1): {topics.count(-1)}개 ({outlier_ratio:.1f}%)")
print()

# ══════════════════════════════════════════════════════════════════
# 4. 토픽 정보 출력 및 저장
# ══════════════════════════════════════════════════════════════════
print("▶ 토픽별 주요 키워드:")
topic_info = topic_model.get_topic_info()
topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(20)
for _, row in topic_info_filtered.iterrows():
    tid   = row['Topic']
    count = row['Count']
    words = topic_model.get_topic(tid)
    kws   = ', '.join([w for w, _ in words[:7]])
    print(f"  토픽 {tid:2d} ({count:3d}편): {kws}")
print()

# ══════════════════════════════════════════════════════════════════
# 5. 시각화 ① — 전체 토픽 바차트
# ══════════════════════════════════════════════════════════════════
print("▶ [1/3] 토픽 개요 차트 저장 중...")
fig_overview = topic_model.visualize_barchart(
    top_n_topics=min(20, nr_topics_found),
    n_words=8,
    title='KCI 인공지능 논문 토픽별 주요 키워드'
)
fig_overview.update_layout(width=1200, height=800, font=dict(size=12))
save_fig(fig_overview, '04_topic_overview')

# ══════════════════════════════════════════════════════════════════
# 6. 시각화 ② — 연도별 토픽 비중 변화 (Topics Over Time)
# ══════════════════════════════════════════════════════════════════
print("▶ [2/3] 연도별 토픽 변화 분석 중...")
timestamps = [str(y) for y in years]

topics_over_time = topic_model.topics_over_time(
    docs,
    timestamps,
    nr_bins=10,          # 연도별로 자동 집계
    global_tuning=True,
    evolution_tuning=True
)

# 상위 토픽만 표시 (아웃라이어 제외, 상위 15개)
top_topics = (topic_info[topic_info['Topic'] != -1]
              .sort_values('Count', ascending=False)
              .head(15)['Topic'].tolist())

fig_time = topic_model.visualize_topics_over_time(
    topics_over_time,
    topics=top_topics,
    title='KCI 인공지능 논문 — 연도별 토픽 비중 변화'
)
fig_time.update_layout(width=1200, height=650, font=dict(size=12))
save_fig(fig_time, '05_topics_over_time')

# ══════════════════════════════════════════════════════════════════
# 7. 시각화 ③ — 연도 × 토픽 히트맵 (직접 계산)
# ══════════════════════════════════════════════════════════════════
print("▶ [3/3] 연도×토픽 히트맵 저장 중...")

papers_valid = papers[papers['topic'] != -1].copy()
topic_labels = {
    tid: f"T{tid}: " + ', '.join([w for w, _ in topic_model.get_topic(tid)[:3]])
    for tid in top_topics
}

heatmap_data = []
for year in sorted(papers_valid['pub_year'].unique()):
    yp = papers_valid[papers_valid['pub_year'] == year]
    n  = len(yp)
    for tid in top_topics:
        cnt = (yp['topic'] == tid).sum()
        heatmap_data.append({
            '연도': str(year),
            '토픽': topic_labels.get(tid, f'T{tid}'),
            '비중(%)': round(cnt / n * 100, 1) if n > 0 else 0
        })

df_heat = pd.DataFrame(heatmap_data)
pivot   = df_heat.pivot(index='토픽', columns='연도', values='비중(%)')

fig_heat = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale='Blues',
    text=pivot.values,
    texttemplate='%{text}',
    textfont={"size": 9},
    hovertemplate='연도: %{x}<br>토픽: %{y}<br>비중: %{z}%<extra></extra>',
    colorbar=dict(title='비중(%)')
))
fig_heat.update_layout(
    title=dict(text='연도별 토픽 비중 히트맵', font=dict(size=17)),
    xaxis_title='발행 연도', yaxis_title='토픽',
    width=1000, height=700,
    font=dict(size=11), margin=dict(l=280, r=60, t=70, b=60)
)
save_fig(fig_heat, '06_topic_heatmap')

# ══════════════════════════════════════════════════════════════════
# 8. 토픽별 대표 논문 저장 (텍스트)
# ══════════════════════════════════════════════════════════════════
print("▶ 토픽별 대표 논문 목록 저장 중...")
with open(f'{OUT_DIR}/07_topic_documents_sample.txt', 'w', encoding='utf-8') as f:
    f.write("KCI 인공지능 논문 — 토픽별 대표 논문\n")
    f.write("=" * 60 + "\n\n")
    for tid in top_topics:
        words  = topic_model.get_topic(tid)
        kws    = ', '.join([w for w, _ in words[:8]])
        subset = papers[papers['topic'] == tid].sort_values('pub_year')
        f.write(f"[토픽 {tid}] ({len(subset)}편)\n")
        f.write(f"주요 키워드: {kws}\n")
        f.write("-" * 40 + "\n")
        for _, row in subset.head(5).iterrows():
            f.write(f"  ({row['pub_year']}) {row['title']}\n")
        f.write("\n")
print(f"  ✅ {OUT_DIR}/07_topic_documents_sample.txt")

# ══════════════════════════════════════════════════════════════════
# 완료
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("✅ BERTopic 분석 완료! 생성된 파일:")
print("   04_topic_overview.html         전체 토픽 바차트")
print("   05_topics_over_time.html       연도별 토픽 추이")
print("   06_topic_heatmap.html          연도×토픽 히트맵")
print("   07_topic_documents_sample.txt  토픽별 대표 논문")
print("=" * 55)

# 모델 저장 (재실행 시 임베딩 시간 절약)
# topic_model.save("bertopic_model")
# print("  모델도 저장됨: bertopic_model/")
