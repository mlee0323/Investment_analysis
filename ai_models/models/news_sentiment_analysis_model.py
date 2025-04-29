import pandas as pd
import glob
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

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

# 뉴스 제목에 대한 감성 분석
news_df['finbert_scores'] = news_df[title_column].apply(get_finbert_sentiment)
news_df['finbert_positive'] = news_df['finbert_scores'].apply(lambda x: x['positive'])
news_df['finbert_negative'] = news_df['finbert_scores'].apply(lambda x: x['negative'])
news_df['finbert_neutral'] = news_df['finbert_scores'].apply(lambda x: x['neutral'])
news_df['finbert_sentiment'] = news_df['finbert_scores'].apply(lambda x: x['sentiment'])

# 결과 저장 (원본 컬럼 유지)
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
print(f"감성 분석 결과가 {output_file}에 저장되었습니다.")
