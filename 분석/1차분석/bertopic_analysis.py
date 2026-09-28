"""
KCI 인공지능 논문 BERTopic 토픽 모델링 스크립트

사전 설치:
    pip install bertopic sentence-transformers umap-learn hdbscan plotly scikit-learn

입력: 00.KCI_AI_논문_상세_및_인용데이터.csv
출력:
    04_topic_overview.html          전체 토픽 키워드 바차트
    05_topics_over_time.html        연도별 토픽 비중 변화
    06_topic_heatmap.html           연도 × 토픽 히트맵
    07_topic_documents_sample.txt   토픽별 대표 논문 목록
    topic_assignments.csv           논문별 토픽 배정 결과
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import plotly.express as px

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════
INPUT_FILE = '00.KCI_AI_논문_상세_및_인용데이터.csv'

# 임베딩 모델 선택 (주석 해제해서 바꿔가며 비교 가능)
EMBEDDING_MODEL = 'woong0322/ko-legal-sbert-finetuned'   # 한국어 법률 특화 SBERT
# EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask'         # 한국어 범용 SBERT (대안)
# EMBEDDING_MODEL = 'BM-K/KoSimCSE-roberta'               # 한국어 SimCSE (대안)

NR_TOPICS      = 'auto'  # 'auto' 또는 정수 (ex. 15)
MIN_TOPIC_SIZE = 15      # 토픽 최소 문서 수 — 너무 잘게 쪼개지면 값 올리기
TOP_N_WORDS    = 10      # 토픽당 키워드 수

# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 전처리
# ══════════════════════════════════════════════════════════════════
print("▶ 데이터 로드 중...")
df = pd.read_csv(INPUT_FILE)
papers = (df.drop_duplicates(subset='source_id')
            [['source_id','title','pub_year','abstract','keywords']]
            .copy())

print(f"  전체 고유 논문: {len(papers):,}개")

# 텍스트 결합: 제목(2회 가중) + 초록 + 키워드
def build_text(row):
    parts = []
    if pd.notna(row['title']) and str(row['title']).strip():
        t = str(row['title']).strip()
        parts.append(t + ' ' + t)          # 제목 2회 (가중)
    if pd.notna(row['abstract']) and str(row['abstract']).strip():
        parts.append(str(row['abstract']).strip())
    if pd.notna(row['keywords']) and str(row['keywords']).strip():
        parts.append(str(row['keywords']).strip())
    return ' '.join(parts)

papers['text'] = papers.apply(build_text, axis=1)
papers = papers[papers['text'].str.len() > 30].reset_index(drop=True)
papers = papers.sort_values('pub_year').reset_index(drop=True)

docs  = papers['text'].tolist()
years = papers['pub_year'].tolist()

print(f"  분석 대상: {len(docs):,}개")
print(f"  연도 범위: {min(years)} ~ {max(years)}")
print(f"  임베딩 모델: {EMBEDDING_MODEL}")
print()

# ══════════════════════════════════════════════════════════════════
# 2. 임베딩 모델 로드
# ══════════════════════════════════════════════════════════════════
print("▶ 임베딩 모델 로드 중...")
print("  (첫 실행 시 모델 다운로드로 수 분 소요될 수 있습니다)")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("  완료\n")

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

# 한국어 법률 도메인 불용어
stopwords = [
    # 한국어 일반
    '있다','이다','하다','되다','이','가','을','를','의','에','은','는',
    '그','저','것','수','및','등','또한','따라','위해','통해',
    '대한','관한','관련','으로','에서','에도','에서의',
    '있는','있으며','있음','있어','한다','한다고',
    # 논문 상투어
    '본','본고','논문','연구','분석','검토','고찰','방안','문제',
    '대해','대하여','필요','중요','가능','경우','때문','통하여',
    '제시','제안','검토','목적','결과','내용','방법','기반',
    # 영어 일반
    'the','of','and','in','a','to','is','for','this','that',
    'with','on','are','be','as','by','an','it','at','or','from',
    'but','not','have','has','been','which','were','their',
]

vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=stopwords,
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

topic_info = topic_model.get_topic_info()
nr_found   = len(set(topics)) - (1 if -1 in topics else 0)
outlier_n  = topics.count(-1)
print(f"\n  발견된 토픽 수: {nr_found}개")
print(f"  아웃라이어(-1): {outlier_n}개 ({outlier_n/len(topics)*100:.1f}%)\n")

# 토픽별 키워드 출력
print("▶ 토픽별 주요 키워드:")
for _, row in topic_info[topic_info['Topic'] != -1].head(20).iterrows():
    words = topic_model.get_topic(row['Topic'])
    kws   = ', '.join([w for w, _ in words[:7]])
    print(f"  T{row['Topic']:2d} ({row['Count']:3d}편): {kws}")
print()

# ══════════════════════════════════════════════════════════════════
# 4. 시각화
# ══════════════════════════════════════════════════════════════════
top_topics = (topic_info[topic_info['Topic'] != -1]
              .sort_values('Count', ascending=False)
              .head(min(15, nr_found))['Topic'].tolist())

# ── 4-1. 토픽 바차트
print("▶ [1/3] 토픽 개요 바차트...")
fig1 = topic_model.visualize_barchart(
    top_n_topics=min(20, nr_found), n_words=8,
    title='KCI 인공지능 논문 — 토픽별 주요 키워드'
)
fig1.update_layout(width=1200, height=800, font=dict(size=12))
fig1.write_html('04_topic_overview.html')
print("  ✅ 04_topic_overview.html")

# ── 4-2. Topics over time
print("▶ [2/3] 연도별 토픽 변화...")
timestamps = [str(y) for y in years]
tot = topic_model.topics_over_time(
    docs, timestamps,
    nr_bins=len(set(years)),
    global_tuning=True,
    evolution_tuning=True
)
fig2 = topic_model.visualize_topics_over_time(
    tot, topics=top_topics,
    title='KCI 인공지능 논문 — 연도별 토픽 비중 변화'
)
fig2.update_layout(width=1200, height=650, font=dict(size=12))
fig2.write_html('05_topics_over_time.html')
print("  ✅ 05_topics_over_time.html")

# ── 4-3. 연도 × 토픽 히트맵
print("▶ [3/3] 연도×토픽 히트맵...")
topic_labels = {
    tid: f"T{tid}: " + ', '.join([w for w, _ in topic_model.get_topic(tid)[:3]])
    for tid in top_topics
}
papers_valid = papers[papers['topic'] != -1]
hmap = []
for year in sorted(papers_valid['pub_year'].unique()):
    yp = papers_valid[papers_valid['pub_year'] == year]
    n  = len(yp)
    for tid in top_topics:
        cnt = (yp['topic'] == tid).sum()
        hmap.append({'연도': str(year), '토픽': topic_labels[tid],
                     '비중(%)': round(cnt/n*100, 1) if n > 0 else 0})

df_hmap = pd.DataFrame(hmap)
pivot   = df_hmap.pivot(index='토픽', columns='연도', values='비중(%)')

fig3 = go.Figure(data=go.Heatmap(
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
fig3.update_layout(
    title=dict(text='연도별 토픽 비중 히트맵', font=dict(size=17)),
    xaxis_title='발행 연도', yaxis_title='토픽',
    width=1000, height=700, font=dict(size=11),
    margin=dict(l=300, r=60, t=70, b=60)
)
fig3.write_html('06_topic_heatmap.html')
print("  ✅ 06_topic_heatmap.html")

# ══════════════════════════════════════════════════════════════════
# 5. 토픽별 대표 논문 + CSV 저장
# ══════════════════════════════════════════════════════════════════
print("▶ 토픽별 대표 논문 저장...")
with open('07_topic_documents_sample.txt', 'w', encoding='utf-8') as f:
    f.write(f"KCI 인공지능 논문 BERTopic 결과\n")
    f.write(f"모델: {EMBEDDING_MODEL}\n")
    f.write("=" * 60 + "\n\n")
    for tid in top_topics:
        words  = topic_model.get_topic(tid)
        kws    = ', '.join([w for w, _ in words[:8]])
        subset = papers[papers['topic'] == tid].sort_values('pub_year')
        f.write(f"[토픽 {tid}] {len(subset)}편\n")
        f.write(f"키워드: {kws}\n")
        f.write("-" * 40 + "\n")
        for _, row in subset.head(7).iterrows():
            f.write(f"  ({row['pub_year']}) {row['title']}\n")
        f.write("\n")
print("  ✅ 07_topic_documents_sample.txt")

papers[['source_id','title','pub_year','topic']].to_csv(
    'topic_assignments.csv', index=False, encoding='utf-8-sig')
print("  ✅ topic_assignments.csv")

# ══════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("✅ BERTopic 분석 완료!")
print(f"  사용 모델: {EMBEDDING_MODEL}")
print(f"  발견 토픽: {nr_found}개")
print()
print("  04_topic_overview.html")
print("  05_topics_over_time.html")
print("  06_topic_heatmap.html")
print("  07_topic_documents_sample.txt")
print("  topic_assignments.csv")
print("=" * 55)

# 모델 저장 (재실행 시 임베딩 시간 절약 — 필요 시 주석 해제)
# topic_model.save("bertopic_model_ko_legal")
# print("  모델 저장: bertopic_model_ko_legal/")
