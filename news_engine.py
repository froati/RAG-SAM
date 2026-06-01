import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import urllib.parse
import warnings
import re

# XML 파싱 관련 경고 무시
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def get_samsung_news(limit=5, keywords=None):
    """
    구글 뉴스 RSS 피드를 사용하여 '삼성전자' 관련 최신 뉴스를 가져옵니다.
    keywords 인자가 제공되면 해당 키워드 중 하나라도 제목에 포함된 뉴스만 필터링합니다.
    """
    query = urllib.parse.quote("삼성전자")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # lxml이 없는 경우를 대비해 html.parser 사용
        soup = BeautifulSoup(response.content, 'html.parser')
            
        items = soup.find_all('item')
        
        # 전체 텍스트에서 링크를 찾기 위한 정규식 (html.parser의 한계 극복)
        raw_content = response.text
        # <item>...</item> 단위로 분리하여 링크 추출
        raw_items = re.findall(r'<item>(.*?)</item>', raw_content, re.DOTALL)
        
        news_list = []
        for i, item in enumerate(items):
            if len(news_list) >= limit:
                break
                
            title = item.title.text if item.title else "제목 없음"
            
            # 키워드 필터링 적용
            if keywords:
                is_relevant = any(kw.lower() in title.lower() for kw in keywords)
                if not is_relevant:
                    continue

            # 정규식으로 해당 아이템의 실제 링크 추출 (가장 확실한 방법)
            link = "#"
            if i < len(raw_items):
                link_match = re.search(r'<link>(.*?)</link>', raw_items[i], re.DOTALL)
                if link_match:
                    link = link_match.group(1).strip()
            
            # 정규식 실패 시 BeautifulSoup fallback
            if link == "#" or not link:
                link_tag = item.find('link')
                if link_tag:
                    link = link_tag.next_sibling.strip() if link_tag.next_sibling else link_tag.text
            
            # 여전히 비어있으면 전체 텍스트 검색
            if not link or link == "#":
                link_search = re.search(r'https?://[^\s<>"]+', str(item))
                if link_search: link = link_search.group(0)

            # 언론사 정보 분리
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

def analyze_sentiment(news_list, llm):
    """
    뉴스 리스트를 바탕으로 시장 심리(긍정, 중립, 부정) 비율을 분석합니다.
    """
    if not news_list:
        return {"긍정": 33, "중립": 34, "부정": 33}
    
    news_titles = "\n".join([f"- {n['title']}" for n in news_list])
    
    prompt = f"""
    아래 뉴스 제목들을 분석하여 **삼성전자(Samsung Electronics)**의 투자 심리에 미치는 영향만 긍정, 중립, 부정 비율(%)로 응답하세요.
    
    **주의사항:**
    1. 타사(SK하이닉스 등)의 악재나 전망은 삼성전자에게 상대적 호재 또는 중립으로 해석해야 합니다. 
    2. 반드시 '삼성전자'의 관점에서만 이익과 손해를 따지세요.
    3. 응답 형식: '긍정: 숫자, 중립: 숫자, 부정: 숫자' (합계 100)
    
    뉴스 제목:
    {news_titles}
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # 숫자만 추출 (간단한 파싱)
        import re
        numbers = re.findall(r'\d+', content)
        if len(numbers) >= 3:
            return {
                "긍정": int(numbers[0]),
                "중립": int(numbers[1]),
                "부정": int(numbers[2])
            }
    except Exception as e:
        print(f"심리 분석 중 오류 발생: {e}")
    
    return {"긍정": 0, "중립": 100, "부정": 0} # 오류 시 기본값: 중립 100%

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
