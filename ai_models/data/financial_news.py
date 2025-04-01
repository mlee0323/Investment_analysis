import requests
import datetime
import pandas as pd
import xml.etree.ElementTree as ET
import re

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

def fetch_news_lg():
    query = "LG"
    end_date = datetime.datetime(2025, 3, 21)
    start_date = end_date - datetime.timedelta(days=90)
    
    print(f"Fetching news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    url = f"https://news.google.com/rss/search?q={query}+주가|실적|투자&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to fetch news")
        return []
    
    root = ET.fromstring(response.content)
    news_items = []
    
    for item in root.findall(".//item"):
        try:
            title = item.find("title").text
            link = item.find("link").text
            pub_date = item.find("pubDate").text
            
            # 날짜 형식 변환
            if 'Z' in pub_date:  # ISO 형식인 경우
                pub_date = datetime.datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
            else:  # 상대적 날짜인 경우
                pub_date = parse_relative_date(pub_date)
            
            # LG 관련 필터링
            if "LG" in title and not any(x in title for x in ['LG전자', 'LG화학', 'LG이노텍']):
                news_items.append({
                    "Title": title,
                    "Link": link,
                    "Date": pub_date.strftime('%Y-%m-%d')
                })
        except Exception as e:
            print(f"Error processing news item: {e}")
            continue
    
    print(f"{len(news_items)}개의 뉴스 fetched.")
    return news_items

def save_to_excel(news_items, filename="lg_news_api.xlsx"):
    if not news_items:
        print("No news items to save")
        return
        
    df = pd.DataFrame(news_items)
    
    # 날짜순 정렬
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 중복 제거
    df = df.drop_duplicates(subset=['Title'])
    
    df.to_excel(filename, index=False)
    print(f"{len(df)}개의 뉴스를 {filename}에 저장했습니다.")

if __name__ == "__main__":
    news = fetch_news_lg()
    save_to_excel(news)