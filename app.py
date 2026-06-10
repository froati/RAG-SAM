import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import re
import yfinance as yf
from rag_engine import InvestmentRAGEngine
from news_engine import get_samsung_news, analyze_sentiment

# streamlit run app.py
# --- Page Config ---
st.set_page_config(
    page_title="삼성전자 스마트 투자 보조 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize RAG Engine ---
@st.cache_resource
def load_rag_engine():
    # 'data' 폴더에서 모든 PDF 파일 목록을 가져옵니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return None

    pdf_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith(".pdf")
    ]

    # 파일 목록 정렬 (최신순 또는 이름순 등 일관성 유지)
    pdf_files.sort()

    if not pdf_files:
        print("'data' 폴더에 PDF 파일이 없습니다.")
        return None

    return InvestmentRAGEngine(pdf_files, index_path="faiss_index")


try:
    rag_engine = load_rag_engine()
    if rag_engine and rag_engine.vector_db:
        st.sidebar.success(f"문서 학습 완료 ({len(rag_engine.pdf_paths)}개 파일)")
        for p in rag_engine.pdf_paths:
            st.sidebar.caption(f"{os.path.basename(p)}")
    elif rag_engine and not rag_engine.vector_db:
        st.sidebar.warning("엔진은 생성되었으나 학습된 데이터(Vector DB)가 없습니다.")
    else:
        st.sidebar.error("RAG 엔진 객체 생성 실패 (파일 없음)")
except Exception as e:
    st.error(f"RAG 엔진 로드 중 치명적 오류 발생: {e}")
    st.info(" .env 파일에 OPENAI_API_KEY가 올바른지, 혹은 터미널 로그를 확인해주세요.")
    rag_engine = None

# --- Custom CSS (Notion Style & Layout) ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .notion-text {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #37352f;
    }
    .source-accordion {
        background-color: #f7f6f3;
        border-radius: 5px;
        padding: 10px;
    }
    .badge-safe { background-color: #e2fceb; color: #216e39; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warning { background-color: #fff9db; color: #856404; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-danger { background-color: #ffe3e3; color: #cf222e; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    
    /* 뉴스 그리드 레이아웃 */
    .news-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    .news-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 18px;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .news-press-badge {
        background-color: #f1f3f4;
        color: #5f6368;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .news-title:hover {
        text-decoration: underline;
        color: #1a73e8;
    }
    .news-summary {
        font-size: 0.88em;
        color: #3c4043;
        line-height: 1.5;
        margin-top: 8px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* 투자 전략 스타일 개선 (이모티콘 제거) */
    .strategy-grid {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-top: 15px;
    }
    .strategy-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #2ecc71;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .strategy-card.short-term { border-left-color: #3498db; }
    .strategy-card.mid-term { border-left-color: #2ecc71; }
    .strategy-card.long-term { border-left-color: #9b59b6; }
    
    .strategy-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        font-weight: 800;
        font-size: 1.1em;
        color: #202124;
    }
    .strategy-content {
        color: #4a4a4a;
        line-height: 1.6;
        font-size: 0.95em;
    }

    /* 채팅 스타일 커스텀 */
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Fetching ---
@st.cache_data(ttl=600)
def fetch_news_and_strategy():
    # 1. PDF 보고서에서 핵심 키워드 추출
    report_keywords = rag_engine.get_report_keywords() if rag_engine else []
    
    # 2. 키워드 기반 뉴스 필터링
    news = get_samsung_news(limit=6, keywords=report_keywords)
    
    if len(news) < 4:
        extra_news = get_samsung_news(limit=6)
        for n in extra_news:
            if n['title'] not in [exist['title'] for exist in news]:
                news.append(n)
            if len(news) >= 6: break

    # 3. 심리 분석 (LLM 사용)
    sentiment = analyze_sentiment(news[:5], rag_engine.llm) if rag_engine else {"긍정": 33, "중립": 34, "부정": 33}
    
    # 4. 투자 전략 생성
    strategy = rag_engine.get_investment_strategy(news[:3]) if rag_engine else "엔진이 로드되지 않았습니다."
    
    return news, strategy, sentiment

@st.cache_data(ttl=3600)
def fetch_stock_data():
    ticker = "005930.KS"
    try:
        stock = yf.Ticker(ticker)
        # 1. 시세 데이터 가져오기 (가장 필수적)
        df = stock.history(period="6mo")
        if df.empty:
            raise ValueError("주가 데이터를 가져올 수 없습니다.")
            
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 2. 지표 데이터 가져오기 (Rate Limit이 자주 발생하는 부분)
        try:
            info = stock.info
            current_price = info.get("currentPrice")
            if current_price is None and not df.empty:
                current_price = df['Close'].iloc[-1]
            
            metrics = {
                "CurrentPrice": current_price or 0,
                "PER": info.get("trailingPE") or 15.0,
                "PBR": info.get("priceToBook") or 1.2,
                "ROE": (info.get("returnOnEquity") or 0.12) * 100,
                "DividendYield": (info.get("dividendYield") or 0.02) * 100
            }
        except Exception:
            # info 호출 실패 시 최소한의 데이터만 유지
            metrics = {
                "CurrentPrice": df['Close'].iloc[-1] if not df.empty else 0,
                "PER": 15.0, "PBR": 1.2, "ROE": 12.0, "DividendYield": 2.0
            }
        return df, metrics
        
    except Exception as e:
        # 완전히 실패했을 경우 빈 데이터프레임과 기본 지표 반환
        print(f"yfinance 에러: {e}")
        return pd.DataFrame(), {
            "CurrentPrice": 0, "PER": 0, "PBR": 0, "ROE": 0, "DividendYield": 0
        }

def create_stock_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name="MA60", line=dict(color='blue', width=1)))
    fig.update_layout(
        title="삼성전자 주가 추이 (최근 6개월)",
        yaxis_title="가격 (원)",
        yaxis=dict(tickformat=",.0f"),
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white"
    )
    return fig

def create_gauge_chart(value, title, min_val, max_val):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.4], 'color': "#e2fceb"},
                {'range': [(max_val-min_val)*0.4, (max_val-min_val)*0.7], 'color': "#fff9db"},
                {'range': [(max_val-min_val)*0.7, max_val], 'color': "#ffe3e3"}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# 데이터 로드
news_data, strategy_text, sentiment_ratios = fetch_news_and_strategy()

# --- Sidebar: Dashboard ---
with st.sidebar:
    st.header("실시간 대시보드")
    
    try:
        stock_df, metrics = fetch_stock_data()
        
        if not stock_df.empty:
            diff = stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[-2]
            pct = (diff / stock_df['Close'].iloc[-2]) * 100
            st.metric("현재가", f"{int(metrics['CurrentPrice']):,}원", f"{pct:.2f}%")
        else:
            st.metric("현재가", f"{int(metrics['CurrentPrice']):,}원", "데이터 없음")
        
        st.subheader("재무 건전성")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**PER**: {metrics['PER']:.2f}")
            st.plotly_chart(create_gauge_chart(metrics["PER"], "", 0, 30), width="stretch")
        with col2:
            st.markdown(f"**PBR**: {metrics['PBR']:.2f}")
            st.plotly_chart(create_gauge_chart(metrics["PBR"], "", 0, 3), width="stretch")
            
        st.subheader("시장 심리 (Sentiment)")
        sentiment_df = pd.DataFrame({
            "Sentiment": ["긍정", "중립", "부정"],
            "Ratio": [sentiment_ratios["긍정"], sentiment_ratios["중립"], sentiment_ratios["부정"]]
        })
        fig_donut = px.pie(sentiment_df, values='Ratio', names='Sentiment', hole=.4,
                     color_discrete_sequence=['#2ecc71', '#95a5a6', '#e74c3c'])
        fig_donut.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_donut, width="stretch")
        
        st.subheader("시장 동향 점수")
        trend_data = pd.DataFrame({
            "Date": pd.date_range(end=datetime.now(), periods=10),
            "Score": [70, 72, 68, 75, 80, 78, 82, 85, 83, 88] # 예시 데이터
        })
        fig_line = px.line(trend_data, x="Date", y="Score")
        fig_line.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_line, width="stretch")
    except Exception as e:
        st.error(f"대시보드 로드 실패: {e}")

# --- Main Area: AI Assistant ---
st.title("삼성전자 AI 투자 비서")
st.markdown("---")

# 실시간 주가 차트 표시
if 'stock_df' in locals() and not stock_df.empty:
    st.plotly_chart(create_stock_chart(stock_df), width="stretch")
    st.markdown("<br>", unsafe_allow_html=True)

# 2. AI 채팅 영역 (고정 높이 컨테이너)
st.subheader("AI 분석 채팅")
chat_container = st.container(height=500, border=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages inside the scrollable container
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="notion-text">{message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                with st.expander("출처 확인하기"):
                    for source in message["sources"]:
                        st.markdown(f"- [{source['title']}]({source['link']})")

# User Input
if prompt := st.chat_input("삼성전자의 최근 배당 정책에 대해 알려줘"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_text = st.status("정보를 분석 중입니다...", expanded=True)
            
            if rag_engine:
                # 대화 기록과 뉴스 데이터 전달
                response_text, sources, contexts = rag_engine.process_query(
                    prompt, 
                    chat_history=st.session_state.messages[-6:],
                    news_list=news_data[:3]  # 상위 3개 뉴스 활용
                )
                status_text.update(label="분석 완료!", state="complete", expanded=False)
                
                st.markdown(f'<div class="notion-text">{response_text}</div>', unsafe_allow_html=True)
                
                # 출처 표시
                if sources:
                    with st.expander("출처 확인하기"):
                        for src in sources:
                            if src['page'] == "URL":
                                st.markdown(f"- **{src['title']}**")
                            else:
                                filename = os.path.basename(src['title'])
                                st.markdown(f"- **{filename}** (p.{src['page']})")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "sources": [{"title": s['title'], "link": "#"} for s in sources]
                })
                st.rerun()
            else:
                st.error("RAG 엔진이 로드되지 않았습니다.")

# --- Bottom Section: News & Investment Strategy ---
st.markdown("---")

# 1. 최근 주요 뉴스 (그리드 레이아웃)
st.subheader("최근 주요 뉴스")
news_html = '<div class="news-grid">'
for item in news_data:
    press = item.get('press', '구글 뉴스')
    link = item.get('link', '#')
    title = item.get('title', '제목 없음')
    card = f'<div class="news-card"><div><div class="news-press-badge">{press}</div><a href="{link}" target="_blank" class="news-title">{title}</a></div></div>'
    news_html += card
news_html += '</div>'
st.markdown(news_html.replace('\n', ''), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. AI 추천 투자 전략 (카드 레이아웃)
st.subheader("AI 추천 투자 전략")

# 전략 텍스트 파싱 및 렌더링 로직
import re
strategies = re.split(r'\n?\d\.\s', strategy_text)
strategies = [s.strip() for s in strategies if s.strip()]

strategy_html = '<div class="strategy-grid">'
classes = ["short-term", "mid-term", "long-term"]

for i, strategy in enumerate(strategies):
    if i >= 3: break
    
    parts = strategy.split(":", 1)
    if len(parts) == 2:
        title = parts[0].strip()
        content = parts[1].strip()
    else:
        title = f"전략 {i+1}"
        content = strategy
    
    card = f'<div class="strategy-card {classes[i]}"><div class="strategy-header">{title}</div><div class="strategy-content">{content}</div></div>'
    strategy_html += card
strategy_html += '</div>'
st.markdown(strategy_html.replace('\n', ''), unsafe_allow_html=True)
