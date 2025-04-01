import requests
import datetime
import pandas as pd
import xml.etree.ElementTree as ET

def fetch_news_lg():
    query = "LG"
    end_date = datetime.datetime(2025, 3, 21)
    start_date = end_date - datetime.timedelta(days=60)
    
    print(f"Fetching news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to fetch news")
        return []
    
    root = ET.fromstring(response.content)
    news_items = []
    
    for item in root.findall(".//item"):
        title = item.find("title").text
        link = item.find("link").text
        pub_date = item.find("pubDate").text
        news_items.append({"title": title, "link": link, "pub_date": pub_date})
    
    print(f"Fetched {len(news_items)} news articles.")
    return news_items

def save_to_excel(news_items, filename="lg_news.xlsx"):
    df = pd.DataFrame(news_items)
    df.to_excel(filename, index=False)
    print(f"Saved {len(news_items)} news articles to {filename}")

if __name__ == "__main__":
    news = fetch_news_lg()
    save_to_excel(news)