from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import json
import os
from typing import List, Dict, Optional
from enum import Enum

app = FastAPI(title="Stock Prediction API")

# 지원하는 종목 정의
class StockSymbol(str, Enum):
    LG = "LG전자"
    SAMSUNG = "삼성전자"
    SK = "SK하이닉스"
    # 추가 종목들...

# 모델과 스케일러 로드
models = {}  # 종목별 모델 저장
scalers = {}  # 종목별 스케일러 저장
model_metadata = {}  # 종목별 메타데이터 저장

class StockData(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    sentiment: float

class PredictionRequest(BaseModel):
    stock_symbol: StockSymbol
    stock_data: List[StockData]

class PredictionResponse(BaseModel):
    stock_symbol: str
    predictions: List[float]
    dates: List[str]
    confidence: List[float]  # 예측 신뢰도 추가

def load_models():
    global models, scalers, model_metadata
    
    try:
        # 각 종목별로 모델과 스케일러 로드
        for symbol in StockSymbol:
            model_path = f'models/{symbol.value}_model.h5'
            metadata_path = f'models/{symbol.value}_metadata.json'
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                # 모델 로드
                models[symbol] = tf.keras.models.load_model(model_path)
                
                # 메타데이터 로드
                with open(metadata_path, 'r') as f:
                    model_metadata[symbol] = json.load(f)
                
                # 스케일러 초기화
                scaler = RobustScaler()
                scaler.scale_ = np.array(model_metadata[symbol]['price_scaler_params']['scale_'])
                scaler.center_ = np.array(model_metadata[symbol]['price_scaler_params']['center_'])
                scalers[symbol] = scaler
                
                print(f"Model loaded successfully for {symbol.value}")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_models()

def preprocess_data(stock_data: List[StockData], symbol: StockSymbol) -> np.ndarray:
    # 데이터를 numpy 배열로 변환
    data = []
    for item in stock_data:
        data.append([
            item.open,
            item.high,
            item.low,
            item.close,
            item.volume,
            item.sentiment
        ])
    
    data = np.array(data)
    
    # 스케일링
    scaled_data = scalers[symbol].transform(data)
    
    # 모델 입력 형태로 변환
    return scaled_data.reshape(1, -1, 6)

@app.get("/stocks")
async def get_supported_stocks():
    """지원하는 종목 목록 반환"""
    return {"supported_stocks": [symbol.value for symbol in StockSymbol]}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        symbol = request.stock_symbol
        
        # 종목 모델이 있는지 확인
        if symbol not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for stock symbol: {symbol.value}"
            )
        
        # 데이터 전처리
        input_data = preprocess_data(request.stock_data, symbol)
        
        # 예측 수행
        predictions = models[symbol].predict(input_data)
        
        # 예측값 역스케일링
        scaled_predictions = scalers[symbol].inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # 예측 신뢰도 계산 (표준편차 기반)
        confidence = np.std(predictions, axis=0).tolist()
        
        # 예측 날짜 생성
        last_date = request.stock_data[-1].date
        dates = [last_date]
        
        # 나머지 날짜 생성 (주말 제외)
        for i in range(1, 5):
            next_date = np.datetime64(last_date) + np.timedelta64(i, 'D')
            while np.is_busday(next_date) == False:
                next_date += np.timedelta64(1, 'D')
            dates.append(str(next_date))
        
        return PredictionResponse(
            stock_symbol=symbol.value,
            predictions=scaled_predictions.tolist(),
            dates=dates,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 