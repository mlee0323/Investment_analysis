from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta
import os

app = FastAPI()

# 현재 디렉토리 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class PredictionResponse(BaseModel):
    dates: List[str]
    actual_prices: List[float]
    predicted_prices: List[float]
    error_rates: List[float]
    confidence: float
    last_updated: str

@app.get("/")
async def root():
    """API 루트 경로"""
    return {
        "message": "주식 가격 예측 API",
        "endpoints": {
            "/predictions/lg": "LG전자 주가 예측 결과 조회",
            "/docs": "API 문서"
        }
    }

@app.get("/predictions/lg", response_model=PredictionResponse)
async def get_lg_predictions():
    """LG전자 주가 예측 결과 반환"""
    try:
        # 실제 예측 결과 데이터
        dates = ["2025-03-21", "2025-03-24", "2025-03-25", "2025-03-26", "2025-03-27"]
        actual_prices = [69700, 67500, 67200, 66800, 65700]
        predicted_prices = [71938, 68423, 88125, 61137, 70939]
        error_rates = [3.21, 1.37, 31.14, -8.48, 7.97]
        
        return PredictionResponse(
            dates=dates,
            actual_prices=actual_prices,
            predicted_prices=predicted_prices,
            error_rates=error_rates,
            confidence=0.85,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)