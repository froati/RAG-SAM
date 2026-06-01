import requests
from bs4 import BeautifulSoup
import urllib.parse

def get_samsung_news(limit=5, keywords=None):
    """
    구글 뉴스 RSS 피드를 사용하여 '삼성전자' 관련 최신 뉴스를 가져옵니다.
    keywords 인자가 제공되면 해당 키워드 중 하나라도 제목이나 요약에 포함된 뉴스만 필터링합니다.
    """
    query = urllib.parse.quote("삼성전자")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        try:
            soup = BeautifulSoup(response.content, 'xml')
        except:
            soup = BeautifulSoup(response.content, 'html.parser')
            
        items = soup.find_all('item')
        
        news_list = []
        for item in items:
            if len(news_list) >= limit:
                break
                
            title = item.title.text
            link = item.link.text
            
            # 키워드 필터링 적용
            if keywords:
                # 키워드 중 하나라도 제목에 포함되어 있는지 확인
                is_relevant = any(kw.lower() in title.lower() for kw in keywords)
                if not is_relevant:
                    continue

            if " - " in title:
                title_parts = title.rsplit(" - ", 1)
                title = title_parts[0]
                press = title_parts[1]
            else:
                press = item.source.text if item.source else "구글 뉴스"
            
            summary = "" 
            
            news_list.append({
                "title": title,
                "link": link,
                "press": press,
                "summary": summary
            })
            
        return news_list
        
    except Exception as e:
        print(f"뉴스 수집 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

    news = get_samsung_news()
    if not news:
        print("뉴스를 가져오지 못했습니다.")
    for n in news:
        print(f"[{n['press']}] {n['title']}")
        print(f"링크: {n['link']}")
        print("-" * 30)
