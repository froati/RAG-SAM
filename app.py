import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import yfinance as yf
from rag_engine import InvestmentRAGEngine
from news_engine import get_samsung_news

# streamlit run app.py
# --- Page Config ---
st.set_page_config(
    page_title="삼성전자 스마트 투자 보조 시스템",
    page_icon="📈",
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
        print("⚠️ 'data' 폴더에 PDF 파일이 없습니다.")
        return None

    return InvestmentRAGEngine(pdf_files, index_path="faiss_index")


try:
    rag_engine = load_rag_engine()
    if rag_engine and rag_engine.vector_db:
        st.sidebar.success(f"✅ 문서 학습 완료 ({len(rag_engine.pdf_paths)}개 파일)")
        for p in rag_engine.pdf_paths:
            st.sidebar.caption(f"📄 {os.path.basename(p)}")
    elif rag_engine and not rag_engine.vector_db:
        st.sidebar.warning("⚠️ 엔진은 생성되었으나 학습된 데이터(Vector DB)가 없습니다.")
    else:
        st.sidebar.error("❌ RAG 엔진 객체 생성 실패 (파일 없음)")
except Exception as e:
    st.error(f"🚨 RAG 엔진 로드 중 치명적 오류 발생: {e}")
    st.info("💡 .env 파일에 OPENAI_API_KEY가 올바른지, 혹은 터미널 로그를 확인해주세요.")
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
    
    /* 뉴스 카드 스타일 */
    .news-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .news-title {
        font-weight: bold;
        color: #1a73e8;
        text-decoration: none;
        font-size: 1.1em;
    }
    .news-meta {
        color: #70757a;
        font-size: 0.85em;
        margin-bottom: 5px;
    }
    
    /* 투자 전략 스타일 */
    .strategy-container {
        background-color: #f8f9fa;
        border-left: 5px solid #2ecc71;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 20px;
    }
    .strategy-item {
        margin-bottom: 15px;
    }
    .strategy-label {
        font-weight: bold;
        color: #2c3e50;
        display: block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_news_and_strategy():
    # 1. PDF 보고서에서 핵심 키워드 추출
    report_keywords = rag_engine.get_report_keywords() if rag_engine else []
    
    # 2. 키워드 기반 뉴스 필터링 (보고서 관련 내용 위주)
    news = get_samsung_news(limit=5, keywords=report_keywords) # 분석을 위해 5개 수집
    
    # 3. 만약 키워드 필터링 결과가 너무 적으면 일반 뉴스로 보완
    if len(news) < 3:
        extra_news = get_samsung_news(limit=5)
        for n in extra_news:
            if n['title'] not in [exist['title'] for exist in news]:
                news.append(n)
            if len(news) >= 5: break

    strategy = rag_engine.get_investment_strategy(news[:3]) if rag_engine else "엔진이 로드되지 않았습니다."
    return news, strategy

def calculate_sentiment(news_list):
    """
    뉴스 제목의 키워드를 분석하여 간단한 감성 점수를 도출합니다.
    """
    pos_words = ["상승", "돌파", "성공", "수주", "실적발표", "최고", "확대", "성장", "강세", "긍정"]
    neg_words = ["하락", "감소", "우려", "부진", "위기", "둔화", "최저", "축소", "약세", "부정"]
    
    pos_count = 0
    neg_count = 0
    
    for n in news_list:
        title = n['title']
        pos_count += sum(1 for word in pos_words if word in title)
        neg_count += sum(1 for word in neg_words if word in title)
    
    total = pos_count + neg_count
    if total == 0:
        return 50, 30, 20 # 기본값 (중립 위주)
    
    pos_ratio = int((pos_count / total) * 100)
    neg_ratio = int((neg_count / total) * 100)
    neu_ratio = 100 - pos_ratio - neg_ratio
    
    # 너무 극단적인 경우 보정
    if pos_ratio > 80: pos_ratio, neu_ratio = 80, 10
    if neg_ratio > 80: neg_ratio, neu_ratio = 80, 10
    
    return pos_ratio, neu_ratio, neg_ratio

def calculate_trend_score(df):
    """
    주가 데이터를 기반으로 최근 시장 동향 점수를 산출합니다.
    """
    # 최근 10일 데이터
    recent = df.tail(10).copy()
    
    # 1. 이동평균선 대비 위치 (MA20 위면 가점)
    ma_signal = 1 if recent['Close'].iloc[-1] > recent['MA20'].iloc[-1] else 0
    
    # 2. 최근 5일 수익률
    return_5d = (recent['Close'].iloc[-1] - recent['Close'].iloc[-5]) / recent['Close'].iloc[-5]
    
    # 3. 변동성 및 추세 결합 (0~100점 사이 산출)
    base_score = 70 # 기본 점수
    trend_adj = (return_5d * 500) + (ma_signal * 10)
    
    # 날짜별 점수 생성 (시각화용)
    scores = []
    for i in range(len(recent)):
        daily_score = base_score + (i * 1.5) + (trend_adj if i > 5 else 0)
        scores.append(min(max(daily_score, 10), 98)) # 10~98점 사이 제한
        
    return pd.DataFrame({"Date": recent.index, "Score": scores})

@st.cache_data(ttl=3600)
def fetch_stock_data():
    ticker = "005930.KS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    info = stock.info
    # yfinance 정보가 불확실할 경우 최근 주가 기반으로 PER 추정 또는 기본값
    metrics = {
        "CurrentPrice": info.get("currentPrice", df['Close'].iloc[-1] if not df.empty else 0),
        "PER": info.get("trailingPE") or 15.0,
        "PBR": info.get("priceToBook") or 1.2,
        "ROE": (info.get("returnOnEquity") or 0.12) * 100,
        "DividendYield": (info.get("dividendYield") or 0.02) * 100
    }
    return df, metrics

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
        yaxis=dict(tickformat=",.0f"), # 천 단위 콤마 추가
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

# --- Sidebar: Dashboard ---
with st.sidebar:
    st.header("📊 실시간 대시보드")
    
    try:
        stock_df, metrics = fetch_stock_data()
        
        diff = stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[-2]
        pct = (diff / stock_df['Close'].iloc[-2]) * 100
        st.metric("현재가", f"{int(metrics['CurrentPrice']):,}원", f"{pct:.2f}%")
        
        st.subheader("재무 건전성")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**PER**: {metrics['PER']:.2f}")
            st.plotly_chart(create_gauge_chart(metrics["PER"], "", 0, 30), width="stretch")
        with col2:
            st.markdown(f"**PBR**: {metrics['PBR']:.2f}")
            st.plotly_chart(create_gauge_chart(metrics["PBR"], "", 0, 3), width="stretch")
            
        st.subheader("시장 심리 (Sentiment)")
        sentiment_data = pd.DataFrame({
            "Sentiment": ["긍정", "중립", "부정"],
            "Ratio": [65, 20, 15]
        })
        fig_donut = px.pie(sentiment_data, values='Ratio', names='Sentiment', hole=.4,
                     color_discrete_sequence=['#2ecc71', '#95a5a6', '#e74c3c'])
        fig_donut.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_donut, width="stretch")
        
        st.subheader("시장 동향 점수")
        trend_data = pd.DataFrame({
            "Date": pd.date_range(start="2026-05-01", periods=10),
            "Score": [70, 72, 68, 75, 80, 78, 82, 85, 83, 88]
        })
        fig_line = px.line(trend_data, x="Date", y="Score")
        fig_line.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_line, width="stretch")
    except Exception as e:
        st.error(f"대시보드 로드 실패: {e}")

# --- Main Area: AI Assistant ---
st.title("🤖 삼성전자 AI 투자 비서")
st.markdown("---")

# 실시간 주가 차트 표시
if 'stock_df' in locals():
    st.plotly_chart(create_stock_chart(stock_df), width="stretch")
    st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="notion-text">{message["content"]}</div>', unsafe_allow_html=True)
        if "sources" in message:
            with st.expander("📚 출처 확인하기 (Source Accordion)"):
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['link']})")

# User Input
if prompt := st.chat_input("삼성전자의 최근 배당 정책에 대해 알려줘"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Self-RAG Process Visualization
        status_text = st.status("🔍 정보를 분석 중입니다...", expanded=True)
        time.sleep(0.5)
        status_text.write("1. 관련 문서 검색 중 (Knowledge Base: 삼성전자 보고서)")
        
        if rag_engine:
            # 대화 기록 전달 (최대 3쌍/6개 메시지)
            response_text, sources = rag_engine.process_query(prompt, chat_history=st.session_state.messages[-6:])
            status_text.write("2. 답변 생성 및 자가 검증 중 (Gemini Self-Reflection)")
            time.sleep(0.5)
            status_text.update(label="✅ 분석 완료!", state="complete", expanded=False)
            
            # Display response
            message_placeholder.markdown(f'<div class="notion-text">{response_text}</div>', unsafe_allow_html=True)
            
            # Display sources
            if sources:
                with st.expander("📚 출처 확인하기 (Source Accordion)"):
                    for src in sources:
                        filename = os.path.basename(src['title'])
                        st.markdown(f"- **{filename}** (p.{src['page']})")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": [{"title": os.path.basename(s['title']), "link": "#"} for s in sources]
            })
        else:
            status_text.update(label="❌ RAG 엔진 미로드", state="error", expanded=True)
            st.error("RAG 엔진이 설정되지 않았습니다. API 키와 파일 경로를 확인해주세요.")

# --- Bottom Section: News & Investment Strategy ---
st.markdown("---")

news_data, strategy_text = fetch_news_and_strategy()

# 1. 최근 주요 뉴스 (상단)
st.subheader("📰 최근 주요 뉴스")
for item in news_data:
    st.markdown(f"""
    <div class="news-card">
        <div class="news-meta">{item['press']}</div>
        <a href="{item['link']}" target="_blank" style="text-decoration: none;">
            <div class="news-title">{item['title']}</div>
        </a>
        <div class="notion-text" style="font-size: 0.9em; margin-top: 5px;">{item['summary']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. AI 추천 투자 전략 (하단)
st.subheader("💡 AI 추천 투자 전략")
# 전략 텍스트 포맷팅 (개행 및 강조)
formatted_strategy = strategy_text.replace("1.", "### 1.").replace("2.", "### 2.").replace("3.", "### 3.")
st.markdown(f"""
<div class="strategy-container">
    <div class="notion-text">{formatted_strategy}</div>
</div>
""", unsafe_allow_html=True)
