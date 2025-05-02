import pandas as pd
import glob
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime
from ai_models.database.db_config import execute_query, execute_values_query, execute_transaction

def create_news_sentiment_table():
    """뉴스 감성 분석 결과 테이블 생성"""
    queries = [
        ("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            pub_date TIMESTAMPTZ NOT NULL,
            finbert_positive DECIMAL(5,4),
            finbert_negative DECIMAL(5,4),
            finbert_neutral DECIMAL(5,4),
            finbert_sentiment VARCHAR(10),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """, None),
        ("CREATE INDEX IF NOT EXISTS idx_news_sentiment_date ON news_sentiment (pub_date DESC);", None)
    ]
    execute_transaction(queries)
    print("News sentiment table created successfully!")

# 가장 최근의 뉴스 파일 찾기
news_files = glob.glob('ai_models/data/lg_news_api.xlsx')
if not news_files:
    raise FileNotFoundError("뉴스 파일을 찾을 수 없습니다. 파일명을 확인하세요.")
    
latest_file = max(news_files, key=os.path.getctime)
print(f"사용할 파일: {latest_file}")

# FinBERT 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# 데이터 로드
news_df = pd.read_excel(latest_file)
print(f"데이터프레임 컬럼: {news_df.columns.tolist()}")

# 컬럼명 확인 및 수정
title_column = None
pubdate_column = None
for col in news_df.columns:
    if col.lower() == 'title':
        title_column = col
    elif col.lower() == 'date':
        pubdate_column = col

if not title_column:
    raise KeyError("제목 컬럼을 찾을 수 없습니다. 컬럼명을 확인하세요.")
if not pubdate_column:
    raise KeyError("발행일 컬럼을 찾을 수 없습니다. 컬럼명을 확인하세요.")

# 감성 분석 함수
def get_finbert_sentiment(text):
    if pd.isna(text) or text == '':
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'sentiment': 'neutral'}
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 긍정(0), 부정(1), 중립(2) 클래스 확률
    positive = predictions[0][0].item()
    negative = predictions[0][1].item()
    neutral = predictions[0][2].item()
    
    # 가장 높은 확률의 감성 반환
    sentiment_labels = ['positive', 'negative', 'neutral']
    sentiment = sentiment_labels[np.argmax([positive, negative, neutral])]
    
    return {
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'sentiment': sentiment
    }

def process_news_sentiment():
    """뉴스 데이터를 처리하고 데이터베이스에 저장"""
    # 뉴스 제목에 대한 감성 분석
    news_df['finbert_scores'] = news_df[title_column].apply(get_finbert_sentiment)
    news_df['finbert_positive'] = news_df['finbert_scores'].apply(lambda x: x['positive'])
    news_df['finbert_negative'] = news_df['finbert_scores'].apply(lambda x: x['negative'])
    news_df['finbert_neutral'] = news_df['finbert_scores'].apply(lambda x: x['neutral'])
    news_df['finbert_sentiment'] = news_df['finbert_scores'].apply(lambda x: x['sentiment'])
    
    # 날짜 형식 변환
    news_df[pubdate_column] = pd.to_datetime(news_df[pubdate_column])
    
    # 데이터베이스에 저장할 데이터 준비
    data = [(
        row[title_column],
        row[pubdate_column],
        row['finbert_positive'],
        row['finbert_negative'],
        row['finbert_neutral'],
        row['finbert_sentiment']
    ) for _, row in news_df.iterrows()]
    
    # 트랜잭션으로 데이터 업데이트
    queries = [
        ("DELETE FROM news_sentiment;", None),
        ("""
        INSERT INTO news_sentiment (
            title, pub_date, finbert_positive, finbert_negative,
            finbert_neutral, finbert_sentiment
        ) VALUES %s
        """, data)
    ]
    execute_transaction(queries)
    
    print(f"✅ {len(data)}개의 뉴스 감성 분석 결과가 데이터베이스에 저장되었습니다.")
    
    # Excel 파일로도 백업 저장
    result_df = pd.DataFrame({
        'Title': news_df[title_column],
        'PubDate': news_df[pubdate_column],
        'finbert_positive': news_df['finbert_positive'],
        'finbert_negative': news_df['finbert_negative'],
        'finbert_neutral': news_df['finbert_neutral'],
        'finbert_sentiment': news_df['finbert_sentiment']
    })
    
    output_file = 'ai_models/data/lg_news_finbert_sentiment.xlsx'
    result_df.to_excel(output_file, index=False)
    print(f"감성 분석 결과가 {output_file}에도 백업 저장되었습니다.")

if __name__ == "__main__":
    print("📢 뉴스 감성 분석을 시작합니다...")
    create_news_sentiment_table()
    process_news_sentiment()
    print("✅ 감성 분석 완료!")
