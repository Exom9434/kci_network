"""
TF-IDF 법령 키워드 추출 + 카테고리 매핑 통합 스크립트
=======================================================
법령 텍스트에서 TF-IDF로 키워드를 추출하고,
기존 카테고리 매핑(00.법률별_키워드_카테고리.csv)과 병합하여
03.classify_papers.py가 바로 읽을 수 있는 스키마로 출력.

[입력]
- 법령_통합텍스트(26.03.16)/*.txt  : 법령별 전문 텍스트
- 00.법률별_키워드_카테고리.csv     : 법률명 → 카테고리 매핑

[출력]
- 법령_키워드_TFIDF.csv
  컬럼: 법률명 | 카테고리 | 키워드 목록
  → build_category_keywords()와 호환 (카테고리, 키워드 목록 컬럼 사용)

[설정]
- TOP_N     : 법령별 추출 키워드 수
- MAX_DF    : 너무 흔한 단어 제거 기준 (전체 문서의 X% 이상 등장)
- MIN_DF    : 너무 드문 단어 제거 기준 (최소 N개 문서 이상 등장)

사용법:
    python 00.tfidf_keywords_with_category.py
"""

import os
import glob
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab
from tqdm import tqdm

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────

INPUT_DIR    = "법령_통합텍스트(26.03.16)"
MAPPING_CSV  = "00.법률별_키워드_카테고리.csv"
OUTPUT_CSV   = "법령_키워드_TFIDF_v2.csv"
CACHE_FILE   = "법령_토큰화_캐시.pkl"

TOP_N   = 30    # 법령별 상위 키워드 수 (20→30: 소수 카테고리 키워드 확보)
MAX_DF  = 0.30  # 전체 문서의 30% 이상 등장 시 제외 (0.8→0.3: 행정 일반 명사 적극 제거)
MIN_DF  = 2     # 최소 2개 문서 이상 등장해야 포함

# 행정 일반 명사 스탑워드 (공법 편향 완화)
STOP_WORDS = {
    "위원회", "정부", "국가", "행정", "관리", "계획", "사업", "기관",
    "규정", "조항", "법률", "대통령", "장관", "시행", "적용", "이하",
    "경우", "규정", "해당", "관련", "이상", "이내", "기준", "조치",
    "방법", "절차", "협의", "업무", "운영", "지원", "설치", "구성",
    "의결", "심의", "보고", "제출", "고시", "공고", "통보", "통지",
}


# ─────────────────────────────────────────────
# 🚀 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TF-IDF 법령 키워드 추출 + 카테고리 매핑")
    print("=" * 60)

    # ── 1. 카테고리 매핑 로드 ─────────────────
    print(f"\n[매핑] {MAPPING_CSV} 로드 중...")
    df_map = pd.read_csv(MAPPING_CSV, encoding="utf-8-sig")
    # 법률명 → 카테고리 딕셔너리
    law_to_cat = dict(zip(df_map["법률명"].str.strip(), df_map["카테고리"].str.strip()))
    print(f"  법률 {len(law_to_cat)}개 카테고리 매핑 완료")

    # ── 2. 법령 텍스트 로드 & 형태소 분석 ──────
    file_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not file_paths:
        print(f"[에러] {INPUT_DIR} 에 .txt 파일 없음")
        return

    # 캐시 로드 (있으면 형태소 분석 스킵)
    if os.path.exists(CACHE_FILE):
        print(f"\n[캐시] {CACHE_FILE} 로드 중...")
        with open(CACHE_FILE, "rb") as f:
            law_names, documents_raw = pickle.load(f)
        print(f"  {len(law_names)}개 법령 캐시 로드 완료 (형태소 분석 스킵)")
        # 스탑워드는 캐시 후에도 적용 (캐시는 필터 전 토큰 목록으로 저장)
        documents = [
            " ".join(t for t in doc_tokens if t not in STOP_WORDS)
            for doc_tokens in documents_raw
        ]
    else:
        print(f"\n[텍스트] {len(file_paths)}개 법령 파일 로드 및 명사 추출 중...")
        tagger    = MeCab.Tagger()
        law_names = []
        documents_raw = []   # 스탑워드 적용 전 토큰 목록 (캐시 저장용)
        documents = []

        for path in tqdm(file_paths, desc="형태소 분석", unit="개"):
            law_name = os.path.basename(path).replace(".txt", "").strip()
            law_names.append(law_name)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # 명사(NN*) 추출, 2글자 이상
            parsed = tagger.parse(content)
            nouns  = []
            for line in parsed.splitlines():
                if "\t" not in line or line == "EOS":
                    continue
                surface, feature = line.split("\t", 1)
                pos = feature.split(",")[0]
                if pos.startswith("NN") and len(surface) >= 2:
                    nouns.append(surface)
            documents_raw.append(nouns)
            # 스탑워드 제거 후 문자열로
            documents.append(" ".join(t for t in nouns if t not in STOP_WORDS))

        # 캐시 저장
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((law_names, documents_raw), f)
        print(f"\n[캐시 저장] {CACHE_FILE}")

    # ── 3. TF-IDF 계산 ────────────────────────
    print("\n[TF-IDF] 계산 중...")
    vectorizer = TfidfVectorizer(max_df=MAX_DF, min_df=MIN_DF, use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  어휘 수: {len(feature_names)}개")

    # ── 4. 법령별 상위 키워드 추출 ───────────
    print(f"\n[추출] 법령별 상위 {TOP_N}개 키워드 추출 중...")
    results = []
    no_category = []

    for i, law_name in enumerate(law_names):
        row        = tfidf_matrix.getrow(i).toarray()[0]
        top_idx    = row.argsort()[-TOP_N:][::-1]
        keywords   = [feature_names[idx] for idx in top_idx if row[idx] > 0]
        keyword_str = ", ".join(keywords)

        category = law_to_cat.get(law_name)
        if category is None:
            no_category.append(law_name)
            continue  # 카테고리 없는 법률(기타 등) 제외

        results.append({
            "법률명":    law_name,
            "카테고리":  category,
            "키워드 목록": keyword_str,
        })

    # ── 5. 결과 저장 ──────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ── 6. 요약 출력 ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  [완료] {OUTPUT_CSV}")
    print(f"  분류된 법률:     {len(results)}개")
    print(f"  카테고리 미매핑: {len(no_category)}개 (제외)")
    print(f"\n[카테고리별 법률 수]")
    print(df_out["카테고리"].value_counts().to_string())

    if no_category:
        print(f"\n[카테고리 미매핑 법률 샘플 (최대 10개)]")
        for name in no_category[:10]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
