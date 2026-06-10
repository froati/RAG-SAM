import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# --- Robust Imports for LangChain Retrievers ---
try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None

try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        ContextualCompressionRetriever = None

try:
    from langchain.retrievers.document_compressors import FlashrankRerank
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.document_compressors.flashrank import FlashrankRerank
    except ImportError:
        FlashrankRerank = None
# -----------------------------------------------

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List

# Load .env
load_dotenv()

# Pydantic 2.x class definition fix for FlashrankRerank
if FlashrankRerank:
    try:
        FlashrankRerank.model_rebuild()
    except Exception:
        pass

class RetrievalResponse(BaseModel):
    Reasoning: str = Field(description="Reasoning for retrieval necessity")
    Retrieve: Literal['Yes', 'No'] = Field(description="Whether retrieval is needed")

class RelevanceResponse(BaseModel):
    Reasoning: str = Field(description="Reasoning for document relevance")
    ISREL: Literal['Relevant', 'Irrelevant'] = Field(description="Relevance result")

class GenerationResponse(BaseModel):
    response: str = Field(description="Generated response")

class SupportResponse(BaseModel):
    Reasoning: str = Field(description="Reasoning for support from documents")
    ISSUP: Literal['Fully supported', 'Partially supported', 'No support'] = Field(description="Support result")

class InvestmentRAGEngine:
    def __init__(self, pdf_paths: List[str], index_path: str = "faiss_index"):
        self.pdf_paths = pdf_paths
        if not os.path.isabs(index_path):
            self.index_path = os.path.join(os.path.dirname(__file__), index_path)
        else:
            self.index_path = index_path
            
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Setup retrievers
        self.vector_db, self.bm25_retriever = self._setup_retrievers()
        
        # Initialize Reranker
        if FlashrankRerank:
            try:
                self.reranker = FlashrankRerank()
            except Exception as e:
                print(f"⚠️ Reranker initialization failed: {e}")
                self.reranker = None
        else:
            self.reranker = None

    def _setup_retrievers(self):
        all_docs = []
        # 금융 특화 청킹: 재무제표 레이아웃 보존 및 수치 연관성을 위한 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=[
                "\n\n\n",   # 대단원/페이지 구분
                "\n\n",     # 문단 구분
                "\n",       # 행 구분
                ". ",       # 문장 구분
                " ",        # 단어 구분
                ""
            ],
            keep_separator=True
        )

        for path in self.pdf_paths:
            if os.path.exists(path):
                try:
                    filename = os.path.basename(path)
                    # 메타데이터 추출 (Faithfulness 향상을 위한 시점 정보 주입)
                    metadata = {"source": filename}
                    if "2026.05.15" in filename: # 2026 1Q
                        metadata.update({"year": 2026, "quarter": 1, "type": "분기"})
                    elif "2026.03.10" in filename: # 2025 Full year
                        metadata.update({"year": 2025, "quarter": 4, "type": "사업"})
                    elif "2025.08.14" in filename: # 2025 1H
                        metadata.update({"year": 2025, "quarter": 2, "type": "반기"})

                    # PyMuPDFLoader를 사용하여 레이아웃 보존하며 텍스트 추출
                    loader = PyMuPDFLoader(path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update(metadata)
                    
                    split_docs = text_splitter.split_documents(docs)
                    all_docs.extend(split_docs)
                except Exception as e:
                    print(f"⚠️ Error loading {path}: {e}")

        if not all_docs:
            return None, None

        db = None
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            try:
                print(f"✅ Loading existing FAISS index: {self.index_path}")
                db = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"⚠️ Index load failed: {e}")

        if db is None:
            print("🚀 Creating new FAISS index...")
            batch_size = 100
            db = FAISS.from_documents(all_docs[:batch_size], self.embeddings)
            for i in range(batch_size, len(all_docs), batch_size):
                batch_docs = all_docs[i:i+batch_size]
                db.add_documents(batch_docs)
            os.makedirs(self.index_path, exist_ok=True)
            db.save_local(self.index_path)

        print("✅ Setting up BM25 retriever...")
        bm25 = BM25Retriever.from_documents(all_docs)

        return db, bm25

    def _expand_query(self, query: str) -> List[str]:
        """질문을 검색에 유리한 여러 형태로 확장합니다. (Context Recall 향상)"""
        prompt = PromptTemplate.from_template("""
        [Query Expansion]
        사용자의 질문을 보고, 삼성전자 보고서에서 정확한 정보를 찾기 위한 검색어 2개를 생성하세요.
        - 질문에 '최근'이 포함되면 '2026년 1분기' 또는 '제58기 1분기'를 키워드에 포함시키세요.
        - '작년'이 포함되면 '2025년' 또는 '제57기'를 포함시키세요.
        - 수치 데이터(매출, 영업이익) 요청 시 해당 항목을 명시하세요.
        결과는 반드시 파이썬 리스트 형식으로만 출력하세요. (예: ["2026년 1분기 매출액", "삼성전자 DX 부문 실적"])
        
        질문: {query}
        검색어 리스트:""")
        
        chain = prompt | self.llm
        try:
            res = chain.invoke({"query": query}).content
            import ast
            expanded = ast.literal_eval(res)
            if isinstance(expanded, list):
                return expanded
        except:
            pass
        return [query]

    def process_query(self, query: str, chat_history: List[dict] = None, news_list: List[dict] = None):
        if not self.vector_db or not self.bm25_retriever:
            return "분석할 문서가 없습니다.", [], []

        # 1. LLM Zero-shot 기반의 Semantic Classification (지능형 쿼리 라우팅)
        routing_prompt = PromptTemplate.from_template("""
        [LLM Zero-shot Semantic Classification]
        사용자의 질문 의도를 실시간 분류하여 정보 소스를 정밀 분기하세요.
        
        - REPORT: 과거 재무 수치, 배당 기록, 사업부 구성 등 기업 공시(공시 보고서) 중심의 사실 확인이 필요한 경우
        - NEWS: 최근 주가 소식, 신제품 공개, 산업 동향 등 최신 시장 정보가 필요한 경우
        - HYBRID: 보고서의 수치와 최신 뉴스의 트렌드를 결합한 종합적 분석이 필요한 경우
        
        질문: {query}
        분류 결과(REPORT/NEWS/HYBRID):""")

        routing_chain = routing_prompt | self.llm
        routing_result = routing_chain.invoke({"query": query}).content.strip().upper()

        if "REPORT" in routing_result:
            pdf_k, news_limit = 6, 1
        elif "NEWS" in routing_result:
            pdf_k, news_limit = 2, 6
        else:
            pdf_k, news_limit = 4, 3

        # 2. Query Expansion (REPORT 또는 HYBRID일 때 적용)
        search_queries = [query]
        if "REPORT" in routing_result or "HYBRID" in routing_result:
            search_queries.extend(self._expand_query(query))

        # 3. Retrieval & Reranking
        faiss_retriever = self.vector_db.as_retriever(search_kwargs={"k": 15})
        self.bm25_retriever.k = 15

        if EnsembleRetriever:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, self.bm25_retriever],
                weights=[0.6, 0.4] # 벡터 검색 비중을 약간 높임
            )
        else:
            ensemble_retriever = faiss_retriever

        # 확장된 쿼리들에 대해 중복 없이 문서 수집
        all_retrieved_docs = []
        for q in search_queries:
            all_retrieved_docs.extend(ensemble_retriever.invoke(q))
        
        # 중복 제거
        seen_contents = set()
        unique_docs = []
        for doc in all_retrieved_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)

        if self.reranker:
            # Flashrank를 통한 고성능 리랭킹 (Context Precision 향상)
            docs = self.reranker.compress_documents(unique_docs, query)[:pdf_k]
        else:
            docs = unique_docs[:pdf_k]
        
        pdf_context = ""
        for doc in docs:
            y = doc.metadata.get('year', '?')
            q = doc.metadata.get('quarter', '?')
            t = doc.metadata.get('type', '보고서')
            pdf_context += f"[{y}년 {q}Q {t}] {doc.page_content}\n\n"

        current_news = news_list[:news_limit] if news_list else []
        news_context = ""
        if current_news:
            news_context = "\n[최신 뉴스 정보]\n" + "\n".join([f"- {n['title']}: {n.get('summary', '')}" for n in current_news])

        # 4. History
        history_text = ""
        if chat_history:
            history_text = "\n[이전 대화 기록]\n"
            for msg in chat_history[-3:]:
                role = "사용자" if msg["role"] == "user" else "비서"
                history_text += f"{role}: {msg['content']}\n"

        # 5. Generation
        prompt_template = """
        [역할]
        당신은 삼성전자 투자 전문 분석가입니다. 현재 시점은 2026년 6월 10일입니다.
        제공된 [보고서 컨텍스트]와 [최신 뉴스 정보]를 바탕으로 질문에 대해 **정확한 수치와 상세한 근거**를 포함하여 답변하세요.

        {history_section}

        [지시사항]
        1. **최신성 우선**: 질문에서 '현재', '최근', '지금' 등을 언급하면 반드시 2026년 1분기(제58기 1분기) 데이터를 최우선적으로 사용하세요. (Faithfulness)
        2. **시점 매칭**: 
           - 2026년 1분기 = 제58기 1분기
           - 2025년 (연간) = 제57기
           - 2025년 반기 = 제57기 반기
        3. **핵심 답변 우선**: 질문에 대한 직접적인 정답(수치 등)을 첫 문장에 배치하세요.
        4. **불필요한 문구 절대 금지**: 인사말, 투자 주의사항, 일반적인 시장 전망 등은 절대 포함하지 마세요.
        5. **정보 부재 시 대응**: 문서에 정보가 없다면 "제시된 문서에서 관련 정보를 찾을 수 없습니다."라고만 답변하세요.
        6. **출처 명시**: 답변 문장 끝에 (보고서 p.{{page_info}}) 형식으로만 표기하세요. (Grounding)

        질문: {query}

        [보고서 컨텍스트]
        {pdf_context}

        {news_context}

        전문가 답변:"""
        
        pages = set()
        for doc in docs:
            p = doc.metadata.get('page')
            if p is not None:
                pages.add(str(p + 1))
        page_info = ", ".join(sorted(list(pages))) if pages else "미표기"

        prompt = PromptTemplate.from_template(prompt_template.replace("{{page_info}}", page_info))
        chain = prompt | self.llm

        response = chain.invoke({
            "query": query,
            "pdf_context": pdf_context,
            "news_context": news_context,
            "history_section": history_text
        })

        sources = [{"title": doc.metadata.get('source', '보고서'), "page": doc.metadata.get('page', 0)} for doc in docs]
        if current_news:
            for n in current_news:
                sources.append({"title": f"뉴스: {n['title']}", "page": "URL"})

        combined_contexts = [doc.page_content for doc in docs]
        if current_news:
            combined_contexts.extend([f"뉴스: {n['title']} - {n.get('summary', '')}" for n in current_news])   

        return response.content, sources, combined_contexts

        
        pages = set()
        for doc in docs:
            p = doc.metadata.get('page')
            if p is not None:
                pages.add(str(p + 1))
        page_info = ", ".join(sorted(list(pages))) if pages else "미표기"

        prompt = PromptTemplate.from_template(prompt_template.replace("{{page_info}}", page_info))
        chain = prompt | self.llm

        response = chain.invoke({
            "query": query,
            "pdf_context": pdf_context,
            "news_context": news_context,
            "history_section": history_text
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
        if not self.vector_db:
            return ["반도체", "스마트폰", "디스플레이", "배당", "실적"]

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
        if not self.vector_db:
            return "분석할 문서가 없습니다."

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
        - 보고서의 확정된 수치와 뉴스의 최신 트렌드를 적절히 결합하세요.
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
