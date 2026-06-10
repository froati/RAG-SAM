# 삼성전자 스마트 투자 보조 시스템 설계서 (System Design)

본 시스템은 **랭체인(LangChain) 프레임워크**를 기반으로 조율되며, 금융 데이터의 실시간성과 공시 자료의 정확성을 결합한 하이브리드 RAG 시스템입니다.

## 1. 핵심 아키텍처
*   **프레임워크**: LangChain을 활용한 전체 워크플로우 제어 및 컴포넌트 관리.
*   **데이터 연동**: `yfinance` API(주가/재무 지표) 및 웹 크롤러(실시간 뉴스)를 통한 실시간 데이터 파이프라인 구축.
*   **인덱싱**: OpenAI Embeddings (`text-embedding-3-small`)와 FAISS 벡터 DB를 활용한 고속 시맨틱 검색.
*   **쿼리 처리**: LLM Zero-shot 기반의 쿼리 라우팅을 통해 보고서, 뉴스, 하이브리드 모드로 지능적 분기.

## 2. 데이터 처리 전략
*   **금융 특화 청킹(Financial Specialized Chunking)**: 
    *   공시 보고서(PDF)의 구조적 특성을 고려하여 섹션 및 표(Table) 단위의 의미론적 분할 적용.
    *   단순 길이 기반 분할이 아닌, 금융 용어 및 수치 데이터의 맥락을 보존하는 세퍼레이터 활용.
*   **하이브리드 검색**: 키워드(BM25)와 의미(Vector) 검색의 앙상블을 통한 정보 추출 정밀도 향상.
*   **리랭킹(Re-ranking)**: Cross-Encoder 모델을 활용하여 검색 결과의 연관성 최적화.

## 3. 기술 스택
*   **LLM**: OpenAI GPT-4o-mini
*   **Vector DB**: FAISS
*   **Data**: yfinance, BeautifulSoup (Google News RSS)
*   **Language**: Python 3.10+
