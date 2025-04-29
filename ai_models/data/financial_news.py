import requests
import datetime
import pandas as pd
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import time

def parse_relative_date(relative_date):
    """상대적 날짜 문자열을 실제 날짜로 변환"""
    today = datetime.datetime.now()
    
    if '시간 전' in relative_date:
        hours = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(hours=hours)
    elif '일 전' in relative_date:
        days = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(days=days)
    elif '주 전' in relative_date:
        weeks = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(weeks=weeks)
    elif '개월 전' in relative_date:
        months = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(days=months*30)
    elif '년 전' in relative_date:
        years = int(re.search(r'\d+', relative_date).group())
        return today - datetime.timedelta(days=years*365)
    else:
        return today

def fetch_news_google(query, start_date, end_date):
    """Google News에서 뉴스 가져오기"""
    base_url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "ko",
        "gl": "KR",
        "ceid": "KR:ko"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print("Failed to fetch news from Google")
        return []
    
    root = ET.fromstring(response.content)
    news_items = []
    
    for item in root.findall(".//item"):
        try:
            title = item.find("title").text
            link = item.find("link").text
            pub_date = item.find("pubDate").text
            
            if 'Z' in pub_date:
                pub_date = datetime.datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            else:
                pub_date = parse_relative_date(pub_date)
            
            news_items.append({
                "Title": title,
                "Link": link,
                "Date": pub_date.strftime('%Y-%m-%d'),
                "Source": "Google News"
            })
        except Exception as e:
            print(f"Error processing Google news item: {e}")
            continue
    
    return news_items

def fetch_news_naver(query, start_date, end_date):
    """네이버 뉴스에서 뉴스 가져오기"""
    base_url = "https://search.naver.com/search.naver"
    params = {
        "where": "news",
        "query": query,
        "sort": "1",  # 최신순
        "ds": start_date.strftime('%Y.%m.%d'),
        "de": end_date.strftime('%Y.%m.%d')
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch news from Naver")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = []
    
    for item in soup.select('.news_wrap'):
        try:
            title = item.select_one('.news_tit').text
            link = item.select_one('.news_tit')['href']
            date_text = item.select_one('.info').text.split(' ')[0]
            
            # 날짜 파싱 개선
            try:
                if '전' in date_text:
                    date = parse_relative_date(date_text)
                else:
                    # 다양한 날짜 형식 처리
                    date_formats = [
                        '%Y.%m.%d',
                        '%Y-%m-%d',
                        '%Y/%m/%d',
                        '%Y년 %m월 %d일',
                        '%m/%d/%Y',
                        '%d/%m/%Y'
                    ]
                    
                    date = None
                    for fmt in date_formats:
                        try:
                            date = datetime.datetime.strptime(date_text, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if date is None:
                        date = today
                
                news_items.append({
                    "Title": title,
                    "Link": link,
                    "Date": date.strftime('%Y-%m-%d'),
                    "Source": "Naver News"
                })
            except Exception as e:
                print(f"Error parsing date '{date_text}': {e}")
                continue
                
        except Exception as e:
            print(f"Error processing Naver news item: {e}")
            continue
    
    return news_items

def fetch_news_lg():
    """LG 및 계열사 관련 뉴스 수집"""
    # LG 계열사 목록
    affiliates = [
        "LG", "LG전자", "LG화학", "LG이노텍", "LG디스플레이",
        "LG유플러스", "LG하우시스", "LG생활건강", "LG에너지솔루션"
    ]
    
    # 확장된 검색 쿼리
    base_query = """
    ({company})+(주가|실적|투자|분기|반기|연간|매출|이익|수익|성장|전망|전략|혁신|R&D|연구|개발|투자|M&A|인수|합병|협력|제휴|계약|수주|공급|출시|신제품|기술|특허)
    -스포츠 -야구 -축구 -농구 -배구 -골프 -테니스
    """
    
    end_date = datetime.datetime(2025, 3, 21)
    start_date = end_date - datetime.timedelta(days=180)
    
    print(f"Fetching news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # 모든 계열사의 뉴스 수집
    all_news = []
    for company in affiliates:
        print(f"\nFetching news for {company}...")
        query = base_query.format(company=company)
        
        # Google News
        print("Fetching from Google News...")
        google_news = fetch_news_google(query, start_date, end_date)
        for item in google_news:
            item['Company'] = company
        all_news.extend(google_news)
        
        # Naver News
        print("Fetching from Naver News...")
        naver_news = fetch_news_naver(query, start_date, end_date)
        for item in naver_news:
            item['Company'] = company
        all_news.extend(naver_news)
        
        # API 호출 간격 조절
        time.sleep(1)
    
    # 중복 제거
    seen_titles = set()
    unique_news = []
    for item in all_news:
        if item['Title'] not in seen_titles:
            seen_titles.add(item['Title'])
            unique_news.append(item)
    
    print(f"\nTotal {len(unique_news)}개의 뉴스 fetched.")
    return unique_news

def save_to_excel(news_items, filename="ai_models/data/lg_affiliates_news.xlsx"):
    if not news_items:
        print("No news items to save")
        return
        
    df = pd.DataFrame(news_items)
    
    # 날짜 파싱 개선
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        # 파싱되지 않은 날짜를 오늘 날짜로 대체
        df['Date'] = df['Date'].fillna(pd.Timestamp('today').normalize())
    except Exception as e:
        print(f"Error parsing dates: {e}")
        df['Date'] = pd.Timestamp('today').normalize()
    
    # 날짜순 정렬
    df = df.sort_values(['Company', 'Date'])
    
    # 중복 제거
    df = df.drop_duplicates(subset=['Title'])
    
    # Excel 파일로 저장
    with pd.ExcelWriter(filename) as writer:
        # 전체 뉴스
        df.to_excel(writer, sheet_name='All News', index=False)
        
        # 계열사별로 분리하여 저장
        for company in df['Company'].unique():
            company_news = df[df['Company'] == company]
            company_news.to_excel(writer, sheet_name=company, index=False)
    
    print(f"{len(df)}개의 뉴스를 {filename}에 저장했습니다.")

if __name__ == "__main__":
    news = fetch_news_lg()
    save_to_excel(news)