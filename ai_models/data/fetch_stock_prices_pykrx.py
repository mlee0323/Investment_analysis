from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta
import os

def get_date_range():
    """
    고정된 종료일(20250321)과 그로부터 500일 전의 시작일을 계산하여 반환
    """
    end_date = datetime.strptime('20250321', '%Y%m%d')
    start_date = end_date - timedelta(days=500)
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

def clean_stock_code(stock_code):
    """
    종목코드에서 'A' 접두사 제거
    """
    return stock_code.replace('A', '')

def fetch_stock_data(stock_code, start_date, end_date):
    """
    PyKrx를 사용하여 주식 데이터 가져오기
    """
    try:
        # 종목코드 정리
        clean_code = clean_stock_code(stock_code)
        
        # 일별 OHLCV 데이터 가져오기
        df = stock.get_market_ohlcv_by_date(start_date, end_date, clean_code)
        
        # 데이터가 없는 경우
        if df.empty:
            return None, 0
            
        # 컬럼명 한글로 변경
        column_names = {
            '시가': '시가',
            '고가': '고가',
            '저가': '저가',
            '종가': '현재가',
            '거래량': '거래량',
            '거래대금': '거래대금'
        }
        df = df.rename(columns=column_names)
        
        # 종목코드와 종목명 추가
        df['종목코드'] = stock_code  # 원래 종목코드 사용
        df['종목명'] = stock.get_market_ticker_name(clean_code)
        
        # 날짜를 인덱스에서 컬럼으로 변경
        df = df.reset_index()
        df = df.rename(columns={'날짜': '기준일자'})
        
        # 시가총액 데이터 가져오기
        market_cap = stock.get_market_cap_by_date(start_date, end_date, clean_code)
        if not market_cap.empty:
            df['시가총액'] = market_cap['시가총액']
        
        # 외국인/기관 보유량 데이터 가져오기
        foreign_holding = stock.get_exhaustion_rates_of_foreign_investment_by_ticker(clean_code, start_date, end_date)
        if not foreign_holding.empty:
            df['외국인보유량'] = foreign_holding['외국인보유량']
            df['외국인보유비율'] = foreign_holding['외국인보유비율']
        
        return df, len(df)
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        return None, 0

def fetch_stock_prices():
    # 시작일과 종료일 설정
    start_date, end_date = get_date_range()
    print(f"📆 조회 기간: {start_date} ~ {end_date}")
    
    try:
        # KOSPI200 종목 리스트 가져오기
        kospi200_df = pd.read_csv('kospi200_and_related.csv')
        kospi200_df['종목코드'] = kospi200_df['종목코드'].astype(str)
    except Exception as e:
        print(f"KOSPI 종목 리스트 파일을 읽을 수 없습니다: {e}")
        return None
    
    # 종목별 데이터를 담을 리스트
    all_stock_data = []
    success_count = 0
    fail_count = 0
    
    for idx, row in kospi200_df.iterrows():
        stock_code = row['종목코드']
        stock_name = row['종목명']
        
        print(f"🔄 ({idx+1}/{len(kospi200_df)}) {stock_name}({stock_code}) 데이터 수집 중...")
        
        df, count = fetch_stock_data(stock_code, start_date, end_date)
        
        if df is not None and not df.empty:
            all_stock_data.append(df)
            success_count += 1
            print(f"✅ {stock_name}({stock_code}) - {count}개 데이터 수집 완료")
        else:
            fail_count += 1
            print(f"❌ {stock_name}({stock_code}) - 데이터 없음")
    
    if all_stock_data:
        # 모든 주가 데이터 합치기
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        # 데이터 전처리
        combined_df['기준일자'] = pd.to_datetime(combined_df['기준일자'])
        
        # 하나의 파일로 저장 (종목별로 정렬)
        combined_df = combined_df.sort_values(['종목코드', '기준일자'])
        
        # 파일명에 저장 날짜와 기간 포함
        save_date = datetime.now().strftime('%Y%m%d')
        filename = f'kospi200_stock_prices_pykrx_{start_date}_{end_date}_{save_date}.csv'
        combined_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 모든 데이터가 '{filename}' 파일로 저장되었습니다.")
        print(f"📊 통계:")
        print(f"   - 성공한 종목 수: {success_count}/{len(kospi200_df)}")
        print(f"   - 실패한 종목 수: {fail_count}")
        print(f"   - 수집된 데이터 레코드 수: {len(combined_df)}개")
        
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