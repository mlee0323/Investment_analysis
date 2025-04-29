import requests
import json

# API 엔드포인트
url = "http://127.0.0.1:8000/predict"

# 테스트 데이터
test_data = {
    "symbol": "LG전자",
    "data": [
        {
            "현재가": 100000,
            "시가": 99000,
            "고가": 101000,
            "저가": 98000,
            "거래량": 500000
        }
    ]
}

# API 호출
response = requests.post(url, json=test_data)

# 결과 출력
if response.status_code == 200:
    result = response.json()
    print("\n예측 결과:")
    print(f"예측 가격: {result['predictions'][0]:,.0f}원")
    print(f"신뢰도: {result['confidence']*100:.1f}%")
    print(f"업데이트 시간: {result['last_updated']}")
else:
    print("에러 발생:", response.json()) 