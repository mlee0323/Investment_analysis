import requests
import pandas as pd
import urllib3

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_stock_items():
    print("KOSPI 200 종목 정보 가져오는 중...")
    
    # API 엔드포인트
    url = "http://apis.data.go.kr/1160100/service/GetKrxListedInfoService/getItemInfo"
    
    # API 키 설정
    service_key = "Br3pycEHLqE+tbM3H74ZHJhDxUwtJrwoAER9rltjFnMMV6Aibf4zOOomChkZIgiwYwvX3BuGWHvHWlCFXWy04A=="
    
    # 파라미터 설정
    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "idxIndMid": "1028",  # KOSPI 200 지수 코드
        "numOfRows": "200"    # KOSPI 200 종목 모두 가져오기
    }

    try:
        # API 호출
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        if 'response' in data and 'body' in data['response']:
            items = data['response']['body']['items']['item']
            
            # 데이터프레임 생성
            df = pd.DataFrame(items)
            
            # 실제로 받아오는 컬럼 확인
            print("\n받아온 컬럼 목록:")
            print(df.columns.tolist())
            
            # 필요한 컬럼만 선택 (실제로 존재하는 컬럼만)
            columns = ['srtnCd', 'itmsNm', 'isinCd', 'corpNm']
            df = df[columns]
            
            # 컬럼명 한글로 변경
            column_names = {
                'srtnCd': '종목코드',
                'itmsNm': '종목명',
                'isinCd': 'ISIN코드',
                'corpNm': '기업명'
            }
            df = df.rename(columns=column_names)
            
            # 결과 저장
            output_file = 'kospi200_items.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ KOSPI 200 종목 정보가 {output_file}에 저장되었습니다.")
            print(f"- 총 {len(df)}개의 종목 정보 저장됨")
            
            return df
            
        else:
            print("❌ API 응답에 필요한 데이터가 없습니다.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 중 오류 발생: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    fetch_stock_items()
