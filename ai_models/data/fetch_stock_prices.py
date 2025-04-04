import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def get_date_range():
    """
    고정된 종료일(20250321)과 그로부터 500일 전의 시작일을 계산하여 반환
    """
    end_date = datetime.strptime('20250321', '%Y%m%d')
    # 500 거래일 (약 2년 정도지만 정확히 500일로 설정)
    start_date = end_date - timedelta(days=500)
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

def fetch_stock_data(stock_code, stock_name, api_key, base_url, begin_date, end_date):
    """
    날짜 범위를 지정하여 한 번에 주식 데이터 가져오기
    """
    try:
        params = {
            'serviceKey': api_key,
            'resultType': 'json',
            'itmsNm': stock_name,
            'numOfRows': '1000',  # 최대 row 수
            'pageNo': '1',
            'beginBasDt': begin_date,
            'endBasDt': end_date
        }
        
        response = requests.get(base_url + "/getStockPriceInfo", 
                                params=params, 
                                verify=False)  # SSL 검증 비활성화
        
        response.raise_for_status()
        
        data = response.json()
        
        if (data.get('response', {}).get('body', {}).get('totalCount', 0) > 0 and 
            'items' in data['response']['body']):
            items = data['response']['body']['items'].get('item', [])
            
            if not isinstance(items, list):
                items = [items]
            
            df = pd.DataFrame(items)
            
            # 필요한 컬럼만 선택
            columns = ['basDt', 'srtnCd', 'itmsNm', 'clpr', 'vs', 'fltRt', 'mkp', 'hipr', 'lopr', 'trqu', 'mrktTotAmt']
            df = df[columns]
            
            column_names = {
                'basDt': '기준일자',
                'srtnCd': '종목코드',
                'itmsNm': '종목명',
                'clpr': '현재가',
                'vs': '전일대비',
                'fltRt': '등락률',
                'mkp': '시가',
                'hipr': '고가',
                'lopr': '저가',
                'trqu': '거래량',
                'mrktTotAmt': '시가총액'
            }
            df = df.rename(columns=column_names)
            
            numeric_columns = ['현재가', '전일대비', '등락률', '거래량', '시가', '고가', '저가', '시가총액']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.drop_duplicates(subset=['기준일자', '종목코드'])
            
            # 페이징 처리 - API가 1000개 이상의 결과를 반환할 경우
            total_count = data['response']['body']['totalCount']
            if total_count > 1000:
                total_pages = (total_count + 999) // 1000  # 올림 나눗셈
                additional_dfs = []
                
                for page in range(2, total_pages + 1):
                    params['pageNo'] = str(page)
                    
                    try:
                        page_response = requests.get(base_url + "/getStockPriceInfo", 
                                                   params=params, 
                                                   verify=False)
                        page_response.raise_for_status()
                        page_data = page_response.json()
                        
                        if (page_data.get('response', {}).get('body', {}).get('totalCount', 0) > 0 and 
                            'items' in page_data['response']['body']):
                            page_items = page_data['response']['body']['items'].get('item', [])
                            
                            if not isinstance(page_items, list):
                                page_items = [page_items]
                            
                            page_df = pd.DataFrame(page_items)
                            page_df = page_df[columns]
                            page_df = page_df.rename(columns=column_names)
                            
                            for col in numeric_columns:
                                page_df[col] = pd.to_numeric(page_df[col], errors='coerce')
                            
                            additional_dfs.append(page_df)
                    except Exception as e:
                        print(f"추가 페이지 가져오기 오류: {e}")
                
                if additional_dfs:
                    additional_df = pd.concat(additional_dfs, ignore_index=True)
                    df = pd.concat([df, additional_df], ignore_index=True)
                    df = df.drop_duplicates(subset=['기준일자', '종목코드'])
            
            return df, total_count
        
        return None, 0
    
    except requests.exceptions.RequestException as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        return None, 0
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        return None, 0

def fetch_stock_prices():
    # 디코딩된 API 키
    api_key = "Br3pycEHLqE+tbM3H74ZHJhDxUwtJrwoAER9rltjFnMMV6Aibf4zOOomChkZIgiwYwvX3BuGWHvHWlCFXWy04A=="
    base_url = "http://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
    
    # requests의 SSL 경고 비활성화
    requests.packages.urllib3.disable_warnings(
        requests.packages.urllib3.exceptions.InsecureRequestWarning
    )
    
    # 시작일과 종료일 설정
    begin_date, end_date = get_date_range()
    print(f"📆 조회 기간: {begin_date} ~ {end_date}")
    
    try:
        kospi200_df = pd.read_csv('kospi200_and_related.csv')
        kospi200_df['종목코드'] = kospi200_df['종목코드'].astype(str).str.replace('A', '', regex=False)
    except Exception as e:
        print(f"KOSPI 종목 리스트 파일을 읽을 수 없습니다: {e}")
        return None
    
    # 종목별 데이터를 담을 리스트
    all_stock_data = []
    
    # 종목별 진행 상황을 추적하기 위한 딕셔너리
    total_api_calls = 0
    total_records = 0
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {}
        
        for _, row in kospi200_df.iterrows():
            stock_code = row['종목코드']
            stock_name = row['종목명']
            future = executor.submit(fetch_stock_data, stock_code, stock_name, 
                                    api_key, base_url, begin_date, end_date)
            future_to_stock[future] = (stock_code, stock_name)
        
        for i, future in enumerate(as_completed(future_to_stock)):
            stock_code, stock_name = future_to_stock[future]
            try:
                df, total_count = future.result()
                total_api_calls += 1  # 기본 API 호출 1회
                
                # 페이징이 있었다면 추가 API 호출 횟수 계산
                if total_count > 1000:
                    total_pages = (total_count + 999) // 1000
                    total_api_calls += (total_pages - 1)
                
                if df is not None and not df.empty:
                    # 종목명 일관성 유지
                    df['종목명'] = stock_name
                    all_stock_data.append(df)
                    total_records += len(df)
                    success_count += 1
                    print(f"✅ ({i+1}/{len(kospi200_df)}) {stock_name}({stock_code}) - {len(df)}개 데이터 수집 완료")
                else:
                    fail_count += 1
                    print(f"❌ ({i+1}/{len(kospi200_df)}) {stock_name}({stock_code}) - 데이터 없음")
            except Exception as e:
                fail_count += 1
                print(f"⚠️ ({i+1}/{len(kospi200_df)}) {stock_name}({stock_code}) - 오류: {e}")
    
    if all_stock_data:
        # 모든 주가 데이터 합치기
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        # 데이터 전처리
        combined_df['기준일자'] = pd.to_datetime(combined_df['기준일자'], format='%Y%m%d')
        
        # 하나의 파일로 저장 (종목별로 정렬)
        combined_df = combined_df.sort_values(['종목코드', '기준일자'])
        
        # 파일명에 저장 날짜와 기간 포함
        save_date = datetime.now().strftime('%Y%m%d')
        filename = f'kospi200_stock_prices_{begin_date}_{end_date}_{save_date}.csv'
        combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 모든 데이터가 '{filename}' 파일로 저장되었습니다.")
        print(f"📊 통계:")
        print(f"   - 총 API 호출 횟수: {total_api_calls}회")
        print(f"   - 성공한 종목 수: {success_count}/{len(kospi200_df)}")
        print(f"   - 수집된 데이터 레코드 수: {total_records}개")
        
        return combined_df
    else:
        print("❌ 저장할 데이터가 없습니다.")
        return None

if __name__ == "__main__":
    print("📢 KOSPI200 주가 데이터를 가져오는 중...")
    stock_data = fetch_stock_prices()
    if stock_data is not None:
        print(f"✅ 모든 데이터 수집 및 저장 완료!")
    else:
        print("❌ 데이터 수집 실패!")