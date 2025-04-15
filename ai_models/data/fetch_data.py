import requests

# API 엔드포인트 및 파라미터 설정
url = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockItemList"
params = {
    "serviceKey": "",  # 공공데이터포털에서 발급받은 인증키
    "resultType": "json"          # 결과 형식 설정 (json 또는 xml)
}

# API 호출
response = requests.get(url, params=params)

# 응답 데이터 출력
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
