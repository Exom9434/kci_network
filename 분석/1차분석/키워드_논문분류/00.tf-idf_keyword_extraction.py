import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Mecab
import re

def main():
    # 1. 설정
    input_dir = "법령_통합텍스트(26.03.16)"
    output_file = "법령_키워드_TFIDF.csv"
    top_n = 20  # 각 법령별 추출할 키워드 개수
    
    # Mecab 초기화
    mecab = Mecab()
    
    # 2. 파일 리스트 확보
    file_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    if not file_paths:
        print(f"Error: No .txt files found in {input_dir}")
        return

    documents = []
    law_names = []
    
    print(f"Reading {len(file_paths)} files and tokenizing...")
    
    # 3. 데이터 로드 및 토큰화 (명사 추출)
    for path in file_paths:
        law_name = os.path.basename(path).replace(".txt", "")
        law_names.append(law_name)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 전처리: 한글만 남기기 (선택사항)
            # content = re.sub(r'[^가-힣\s]', ' ', content)
            
            # 명사 추출 및 한 글자 단어 제외
            nouns = mecab.nouns(content)
            nouns = [n for n in nouns if len(n) > 1]
            
            # TF-IDF를 위해 공백으로 구분된 문자열로 결합
            documents.append(" ".join(nouns))
            
    # 4. TF-IDF 계산
    print("Calculating TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_df=0.8,       # 너무 흔한 단어 제외 (80% 이상의 문서에 등장)
        min_df=2,         # 너무 드문 단어 제외 (최소 2개 이상의 문서에 등장)
        use_idf=True
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # 5. 법령별 상위 키워드 추출
    print(f"Extracting top {top_n} keywords per law...")
    results = []
    
    for i, law_name in enumerate(law_names):
        # 해당 문서의 TF-IDF 벡터 가져오기
        row = tfidf_matrix.getrow(i).toarray()[0]
        
        # 점수 높은 순으로 인덱스 정렬
        top_indices = row.argsort()[-top_n:][::-1]
        
        # 실제 키워드와 점수 매핑 (점수가 0보다 큰 것만)
        keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        
        # 리스트를 쉼표로 구분된 문자열로 변환
        keyword_str = ", ".join(keywords)
        
        results.append({
            "법률명": law_name,
            "키워드 목록": keyword_str
        })
        
    # 6. 결과 저장
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Success! Result saved to {output_file}")

if __name__ == "__main__":
    main()
