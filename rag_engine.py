import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List

# 같은 폴더에 있는 .env 로드
load_dotenv()

class RetrievalResponse(BaseModel):
    Reasoning: str = Field(description="검색의 필요유무를 추론하는 과정")
    Retrieve: Literal['Yes', 'No'] = Field(description="검색 필요유무")

class RelevanceResponse(BaseModel):
    Reasoning: str = Field(description="연관문서의 관련성 평가 추론과정")
    ISREL: Literal['Relevant', 'Irrelevant'] = Field(description="관련성 평가 결과")

class GenerationResponse(BaseModel):
    response: str = Field(description="생성된 답변")

class SupportResponse(BaseModel):
    Reasoning: str = Field(description="답변이 문서에 근거하는지 평가")
    ISSUP: Literal['Fully supported', 'Partially supported', 'No support'] = Field(description="지원 평가 결과")

class InvestmentRAGEngine:
    def __init__(self, pdf_paths: List[str], index_path: str = "faiss_index"):
        self.pdf_paths = pdf_paths
        self.index_path = os.path.join(os.path.dirname(__file__), index_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) # 온도를 조금 더 낮추어 정확성 강화
        self.vector_db = self._setup_vector_db()

    def _setup_vector_db(self):
        # 1. 이미 저장된 인덱스가 있는지 확인
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            try:
                print(f"기존 인덱스 로드 중: {self.index_path}")
                return FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"인덱스 로드 실패 ({e}). 인덱스를 다시 생성합니다.")

        # 2. 저장된 인덱스가 없거나 로드 실패 시 PDF 파싱 및 생성
        print("새로운 인덱스 생성 중 (고도화된 PDF 분석)...")

        all_docs = []
        # 청크 크기를 유지하되, 표의 맥락을 위해 오버랩을 300으로 상향
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        for path in self.pdf_paths:
            if os.path.exists(path):
                # PyMuPDFLoader 사용 (PyPDF보다 레이아웃 보존력이 좋음)
                loader = PyMuPDFLoader(path)
                docs = loader.load_and_split(text_splitter)
                all_docs.extend(docs)

        if all_docs:
            batch_size = 100
            db = FAISS.from_documents(all_docs[:batch_size], self.embeddings)
            
            for i in range(batch_size, len(all_docs), batch_size):
                batch_docs = all_docs[i:i+batch_size]
                db.add_documents(batch_docs)
                print(f"배치 처리 중... ({i + len(batch_docs)}/{len(all_docs)})")

            # 생성 후 로컬에 저장
            db.save_local(self.index_path)
            print(f"인덱스 저장 완료: {self.index_path}")
            return db
        return None


    def process_query(self, query: str, chat_history: List[dict] = None, news_list: List[dict] = None):
        # 1. Query Routing (질문 의도 분류)
        routing_prompt = PromptTemplate.from_template("""
        사용자의 질문을 분석하여 '보고서 중심(REPORT)', '최신 뉴스 중심(NEWS)', 또는 '종합 분석(HYBRID)' 중 하나로 분류하세요.
        - REPORT: 과거 재무 수치, 배당 기록, 사업의 구성 등 고정된 사실 확인
        - NEWS: 최근 주가 소식, 신제품 공개, 법인 이전 등 최신 트렌드 확인
        - HYBRID: 보고서 데이터와 최신 뉴스를 결합한 전략적 분석이나 전망
        
        질문: {query}
        분류(REPORT/NEWS/HYBRID):""")
        
        routing_chain = routing_prompt | self.llm
        routing_result = routing_chain.invoke({"query": query}).content.strip().upper()
        
        # 2. Retrieval Strategy (의도에 따른 검색 강화)
        if not self.vector_db:
            return "분석할 문서가 없습니다.", [], []

        # 수치 데이터 누락 방지를 위해 검색 개수(k)를 상향 조정
        if "REPORT" in routing_result:
            pdf_k, news_limit = 10, 1 # 기존 8에서 10으로 상향
        elif "NEWS" in routing_result:
            pdf_k, news_limit = 4, 6 # 기존 3, 5에서 상향
        else:
            pdf_k, news_limit = 7, 5 # 기존 6, 4에서 상향

        docs = self.vector_db.similarity_search(query, k=pdf_k)
        pdf_context = "\n".join([f"[보고서] {doc.page_content}" for doc in docs])
        
        current_news = news_list[:news_limit] if news_list else []
        news_context = ""
        if current_news:
            news_context = "\n[최신 뉴스 정보]\n" + "\n".join([f"- {n['title']}: {n.get('summary', '')}" for n in current_news])

        # 3. History Handling
        history_text = ""
        if chat_history:
            history_text = "\n[이전 대화 기록]\n"
            for msg in chat_history[-4:]:
                role = "사용자" if msg["role"] == "user" else "비서"
                history_text += f"{role}: {msg['content']}\n"

        # 4. Prompt Engineering (Advanced Grounding for Tables)
        prompt_template = """
        당신은 삼성전자 투자 전문 비서입니다. 제공된 [보고서 컨텍스트]와 [최신 뉴스 정보]만을 근거로 답변하세요.
        
        {history_section}
        
        [지시사항]
        1. **표 데이터 해석**: 보고서 컨텍스트에 숫자가 나열된 부분은 재무제표의 '표'일 가능성이 높습니다. 행과 열의 관계를 논리적으로 유추하여 정확한 수치를 인용하세요.
        2. **수치 정확성 및 단위**: '억원', '백만원', '조원' 등 단위를 반드시 확인하고, 필요하다면 수치를 합산하거나 비교하여 답변하세요.
        3. **불확실성 명시**: 만약 문서의 텍스트가 깨져서 수치가 불분명하다면, "문서상의 텍스트 추출 문제로 수치가 불분명하나, 인접한 문맥상 ~로 추정됩니다"와 같이 솔직하게 답하세요.
        4. **엄격한 근거 기반**: 제공된 정보에 없는 내용을 절대 지어내지 마세요.
        5. **출처 명시**: 답변 내에서 "보고서 p.{page_info}에 따르면"과 같이 상세히 밝히세요.
        
        질문: {query}
        
        [보고서 컨텍스트]
        {pdf_context}
        
        {news_context}
        
        전문가 답변:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        # 페이지 정보를 프롬프트에 동적으로 넣기 위해 간단한 가공
        page_info = ", ".join(set([str(doc.metadata.get('page', '알수없음')) for doc in docs]))
        
        response = chain.invoke({
            "query": query, 
            "pdf_context": pdf_context,
            "news_context": news_context,
            "history_section": history_text,
            "page_info": page_info
        })
        
        sources = [{"title": doc.metadata.get('source', '보고서'), "page": doc.metadata.get('page', 0)} for doc in docs]
        if current_news:
            for n in current_news:
                sources.append({"title": f"뉴스: {n['title']}", "page": "URL"})
        
        combined_contexts = [doc.page_content for doc in docs]
        if current_news:
            combined_contexts.extend([f"뉴스: {n['title']} - {n.get('summary', '')}" for n in current_news])
        
        return response.content, sources, combined_contexts

    def get_report_keywords(self):
        """
        PDF 보고서 내용 중 핵심 키워드를 추출합니다.
        뉴스 필터링 시 사용됩니다.
        """
        if not self.vector_db:
            return ["반도체", "스마트폰", "디스플레이", "배당", "실적"]

        # 보고서의 주요 전망이나 사업 부문을 알 수 있는 쿼리 실행
        docs = self.vector_db.similarity_search("삼성전자의 주요 사업 부문 및 핵심 기술", k=5)
        context = "\n".join([doc.page_content[:200] for doc in docs])
        
        prompt = PromptTemplate.from_template("""
        제공된 삼성전자 보고서 컨텍스트를 분석하여, 현재 삼성전자의 투자 가치와 가장 밀접한 핵심 키워드 5개를 단어 형태로만 추출하세요.
        (예: HBM, 갤럭시, 파운드리, 특별배당, 영업이익)
        
        컨텍스트: {context}
        결과(쉼표로 구분):""")
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        
        keywords = [kw.strip() for kw in response.content.split(',')]
        return keywords

    def get_investment_strategy(self, news_list: List[dict]):
        """
        뉴스 데이터와 보고서 데이터를 종합하여 투자 전략을 생성합니다.
        """
        if not self.vector_db:
            return "분석할 문서가 없습니다."

        # 최신 재무 현황 파악을 위한 검색
        docs = self.vector_db.similarity_search("삼성전자 실적 및 향후 전망", k=3)
        report_context = "\n".join([doc.page_content for doc in docs])
        
        news_context = "\n".join([f"- {n['title']}: {n['summary']}" for n in news_list])

        prompt = PromptTemplate.from_template("""
        당신은 삼성전자 전문 투자 전략가입니다. 아래 제공된 [공시 보고서 정보]와 [최신 뉴스]를 바탕으로 투자 전략을 수립하세요.
        
        [공시 보고서 정보]
        {report_context}
        
        [최신 뉴스]
        {news_context}
        
        [지시사항]
        - 1개월(단기), 3개월(중기), 6개월(장기) 관점에서 각각 투자 전략을 작성하세요.
        - 각 기간별로 핵심 전략을 명확한 문장으로 설명하세요.
        - 보고서의 정밀한 수치와 뉴스의 최신 트렌드를 적절히 결합하세요.
        - 전문적이면서도 투자자가 이해하기 쉬운 톤을 유지하세요.
        
        [출력 형식]
        1. 1개월(단기) 전략: [전략 내용]
        2. 3개월(중기) 전략: [전략 내용]
        3. 6개월(장기) 전략: [전략 내용]
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({
            "report_context": report_context,
            "news_context": news_context
        })
        
        return response.content
