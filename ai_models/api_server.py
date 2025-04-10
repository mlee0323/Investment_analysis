from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Any
import json
import pickle
from datetime import datetime

app = FastAPI()

# 모델과 스케일러 로드
try:
    # 모델 로드
    model = tf.keras.models.load_model('/content/drive/MyDrive/stock_prediction_model.h5')
    
    # 메타데이터 로드
    with open('/content/drive/MyDrive/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # 스케일러 로드
    with open('/content/drive/MyDrive/stock_prediction_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("모델과 스케일러 로드 완료")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    raise

class StockData(BaseModel):
    symbol: str
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence: float
    last_updated: str

@app.get("/stocks")
async def get_supported_stocks():
    """지원하는 주식 목록 반환"""
    return {"supported_stocks": ["LG전자"]}

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(data: StockData):
    """주식 가격 예측"""
    try:
        if data.symbol != "LG전자":
            raise HTTPException(status_code=400, detail="지원하지 않는 주식입니다.")
        
        # 데이터 전처리
        df = pd.DataFrame(data.data)
        
        # 필요한 특성 선택 및 정렬
        features = ['현재가', '시가', '고가', '저가', '거래량']
        df = df[features]
        
        # 데이터 스케일링
        scaled_data = scaler.transform(df)
        
        # 예측을 위한 형태로 변환
        X = np.array([scaled_data])
        
        # 예측 수행
        predictions = model.predict(X)
        
        # 예측값을 원래 스케일로 변환
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # 신뢰도 계산 (예시: 예측값의 표준편차를 기반으로)
        confidence = 1.0 - (np.std(predictions) / np.mean(predictions))
        confidence = max(0.0, min(1.0, confidence))  # 0~1 사이로 제한
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            confidence=confidence,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)