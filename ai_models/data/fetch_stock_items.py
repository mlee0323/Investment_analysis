import requests
import pandas as pd
import urllib3

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_stock_items():
    print("KOSPI 200 종목 및 관련 기업 정보 가져오는 중...")
    
    # API 엔드포인트
    url = "http://apis.data.go.kr/1160100/service/GetKrxListedInfoService/getItemInfo"
    
    # API 키 설정
    service_key = "Br3pycEHLqE+tbM3H74ZHJhDxUwtJrwoAER9rltjFnMMV6Aibf4zOOomChkZIgiwYwvX3BuGWHvHWlCFXWy04A=="
    
    # 파라미터 설정
    params = {
        "serviceKey": service_key,
        "resultType": "json",
        "mrktCtg": "KOSPI",  # KOSPI 전체
        "numOfRows": "3000"  # KOSPI 전체 종목 수보다 충분히 큰 수
    }

    try:
        # API 호출
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        if 'response' in data and 'body' in data['response']:
            items = data['response']['body']['items']['item']
            
            # 단일 아이템인 경우 리스트로 변환
            if isinstance(items, dict):
                items = [items]
            
            # 데이터프레임 생성
            df = pd.DataFrame(items)
            
            # 사용 가능한 컬럼 확인
            print("\n[사용 가능한 컬럼]")
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
            
            # 데이터 검증
            print(f"\n[데이터 검증]")
            print(f"- 전체 종목 수: {len(df)}개")
            
            # KOSPI 200 종목 선택 (처음 200개)
            kospi200 = df.head(200)
            
            # 관련 기업 필터링 (전체 종목에서 검색)
            related_companies = []
            major_groups = ['삼성', 'SK', 'LG', '현대', 'NAVER', '카카오', '네이버']
            
            print("\n[관련 기업 검색]")
            for group in major_groups:
                group_companies = df[df['기업명'].str.contains(group, na=False)]
                if not group_companies.empty:
                    print(f"- {group} 그룹: {len(group_companies)}개 기업 발견")
                    related_companies.extend(group_companies.to_dict('records'))
            
            # 중복 제거
            related_companies = pd.DataFrame(related_companies)
            related_companies = related_companies.drop_duplicates(subset=['종목코드'])
            
            # KOSPI 200에 포함된 관련 기업 제외
            related_companies = related_companies[~related_companies['종목코드'].isin(kospi200['종목코드'])]
            
            # KOSPI 200과 관련 기업 합치기
            final_df = pd.concat([kospi200, related_companies])
            
            # 결과 저장
            output_file = 'kospi200_and_related.csv'
            final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ KOSPI 200 및 관련 기업 정보가 {output_file}에 저장되었습니다.")
            print(f"- KOSPI 200 종목 수: {len(kospi200)}개")
            print(f"- 관련 기업 수: {len(related_companies)}개")
            print(f"- 총 {len(final_df)}개의 종목 정보 저장됨")
            
            return final_df
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
