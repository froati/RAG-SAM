import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import urllib.parse
import warnings
import re

# XML 파싱 관련 경고 무시
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def get_samsung_news(limit=8, keywords=None):
    """
    구글 뉴스 RSS 피드를 사용하여 '삼성전자' 관련 최신 뉴스를 가져옵니다.
    투자 관련성 필터링 로직이 강화되었습니다.
    """
    query = urllib.parse.quote("삼성전자")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    
    # 투자와 관련된 핵심 키워드 (가중치 부여용)
    investment_keywords = [
        "실적", "주가", "반도체", "HBM", "영업이익", "매출", "배당", "인수", "합병", 
        "신제품", "갤럭시", "파운드리", "공시", "투자", "전망", "목표가", "계약"
    ]
    
    # 제외할 노이즈 키워드
    blacklist_keywords = ["동호회", "인사발령", "스포츠", "연예", "포토", "게시판", "이벤트"]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        items = soup.find_all('item')
        
        raw_content = response.text
        raw_items = re.findall(r'<item>(.*?)</item>', raw_content, re.DOTALL)
        
        news_list = []
        for i, item in enumerate(items):
            if len(news_list) >= limit:
                break
                
            title = item.title.text if item.title else "제목 없음"
            
            # 1. 블랙리스트 필터링 (노이즈 제거)
            if any(bk in title for bk in blacklist_keywords):
                continue
                
            # 2. 투자 관련성 체크 (투자 키워드가 하나라도 포함되면 우선순위)
            is_investment_related = any(ik in title for ik in investment_keywords)
            
            # 사용자 지정 키워드 필터링이 있는 경우 적용
            if keywords:
                if not any(kw.lower() in title.lower() for kw in keywords):
                    continue

            # 링크 추출 로직 (기존 유지)
            link = "#"
            if i < len(raw_items):
                link_match = re.search(r'<link>(.*?)</link>', raw_items[i], re.DOTALL)
                if link_match:
                    link = link_match.group(1).strip()
            
            if link == "#" or not link:
                link_tag = item.find('link')
                if link_tag:
                    link = link_tag.next_sibling.strip() if link_tag.next_sibling else link_tag.text

            # 언론사 정보 분리
            if " - " in title:
                title_parts = title.rsplit(" - ", 1)
                title = title_parts[0]
                press = title_parts[1]
            else:
                press = item.source.text if item.source else "구글 뉴스"
            
            news_list.append({
                "title": title,
                "link": link,
                "press": press,
                "summary": "",
                "is_relevant": is_investment_related
            })
            
        return news_list
        
    except Exception as e:
        print(f"뉴스 수집 중 오류 발생: {e}")
        return []

def analyze_sentiment(news_list, llm):
    """
    뉴스 리스트를 바탕으로 시장 심리를 분석합니다. 
    투자 무관 뉴스에 대한 무시 로직이 강화된 프롬프트를 사용합니다.
    """
    if not news_list:
        return {"긍정": 33, "중립": 34, "부정": 33}
    
    news_titles = "\n".join([f"- {n['title']}" for n in news_list])
    
    prompt = f"""
    아래 삼성전자 관련 뉴스 제목들을 분석하여 투자 심리 비율(%)을 응답하세요.
    
    **엄격한 분석 지침:**
    1. **투자 관련성 검증**: 주가, 실적, 산업 동향, 경영 전략과 관련 없는 뉴스(인사발령, 단순 행사, 사회 공헌 등)는 분석에서 제외하거나 '중립'으로 처리하세요.
    2. **삼성전자 중심 해석**: 타사 소식은 삼성전자의 경쟁 우위나 시장 점유율 관점에서만 해석하세요.
    3. **수치화**: 오직 투자자에게 실질적인 영향을 줄 소식만 긍정/부정으로 분류하세요. 모호하면 '중립' 비중을 높이세요.
    
    응답 형식: '긍정: 숫자, 중립: 숫자, 부정: 숫자' (합계 100)
    
    뉴스 제목:
    {news_titles}
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        
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
    
    return {"긍정": 0, "중립": 100, "부정": 0}

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

    news = get_samsung_news()
    if not news:
        print("뉴스를 가져오지 못했습니다.")
    for n in news:
        print(f"[{n['press']}] {n['title']} (투자관련: {n['is_relevant']})")
        print(f"링크: {n['link']}")
        print("-" * 30)
