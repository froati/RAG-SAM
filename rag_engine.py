import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
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
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
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
                print(f"⚠️ 인덱스 로드 실패 ({e}). 인덱스를 다시 생성합니다.")

        # 2. 저장된 인덱스가 없거나 로드 실패 시 PDF 파싱 및 생성
        print("새로운 인덱스 생성 중 (PDF 분석)...")

        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for path in self.pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                docs = loader.load_and_split(text_splitter)
                all_docs.extend(docs)

        if all_docs:
            # 한 번에 요청할 때 토큰 제한(max_tokens_per_request)을 피하기 위해 배치 처리
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


    def process_query(self, query: str, chat_history: List[dict] = None):
        # 1. Retrieval Decision
        if not self.vector_db:
            return "분석할 문서가 없습니다.", []

        # 2. Retrieve
        docs = self.vector_db.similarity_search(query, k=5)
        context = "\n".join([f"[문서 {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
        
        # 3. 특정 질문(주가 변동성) 처리 로직 및 프롬프트 구성
        is_volatility_request = "이거 관련해서 주가 변동성 알려줘" in query.replace(" ", "")
        
        history_text = ""
        if is_volatility_request and chat_history:
            history_text = "\n[이전 대화 기록]\n"
            for msg in chat_history[-6:]: # 최대 3쌍(6개 메시지)
                role = "사용자" if msg["role"] == "user" else "비서"
                history_text += f"{role}: {msg['content']}\n"

        prompt_template = """
        당신은 삼성전자 투자 전문 비서입니다. 제공된 [컨텍스트]와 필요 시 [이전 대화 기록]을 바탕으로 답변하세요.
        
        {history_section}
        
        [지시사항]
        - 만약 사용자가 '주가 변동성'에 대해 물었다면, 이전 대화 맥락에서 언급된 사업적 이슈(예: 실적, 신제품, 계약 등)가 실제 삼성전자의 주가 변동성이나 투자 심리에 어떤 영향을 줄 수 있는지 분석하여 답변하세요.
        - 답변은 상세하고 전문적이어야 하며, 수치 데이터가 있다면 적극 활용하세요.
        - '💡 관련 투자 정보' 섹션을 포함하여 풍부한 정보를 제공하세요.
        
        질문: {query}
        [컨텍스트]
        {context}
        
        전문가 답변:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm
        
        response = chain.invoke({
            "query": query, 
            "context": context, 
            "history_section": history_text
        })
        
        sources = [{"title": doc.metadata.get('source', '알 수 없음'), "page": doc.metadata.get('page', 0)} for doc in docs]
        
        return response.content, sources

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
