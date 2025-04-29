import requests
import json
from datetime import datetime, timedelta

# API 엔드포인트
API_URL = "http://localhost:8000/predict"

# 테스트 데이터 생성
def create_test_data():
    # 최근 110일의 데이터 생성 (모델 입력 크기에 맞춤)
    test_data = []
    base_date = datetime.now() - timedelta(days=110)
    
    for i in range(110):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        # 더미 데이터 생성 (실제 사용시 실제 데이터로 대체)
        test_data.append({
            "date": date,
            "open": 70000 + i * 10,
            "high": 71000 + i * 10,
            "low": 69000 + i * 10,
            "close": 70500 + i * 10,
            "volume": 1000000 + i * 1000,
            "sentiment": 0.5  # -1.0 ~ 1.0 사이의 감성 점수
        })
    
    return test_data

def test_prediction():
    # 테스트 데이터 생성
    test_data = create_test_data()
    
    # API 요청
    response = requests.post(
        API_URL,
        json={"stock_data": test_data}
    )
    
    # 결과 출력
    if response.status_code == 200:
        result = response.json()
        print("\n예측 결과:")
        print("-" * 50)
        print(f"{'날짜':<15} {'예측 가격':>15}")
        print("-" * 50)
        for date, price in zip(result["dates"], result["predictions"]):
            print(f"{date:<15} {price:>15,.2f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_prediction() 