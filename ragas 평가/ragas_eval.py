import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 상위 디렉토리의 모듈을 가져오기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_engine import InvestmentRAGEngine

def parse_ragas_data(file_path):
    """ragas_file.txt에서 질문과 모범답변을 파싱합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = []
    # 질문과 모범답변 추출 (정규표현식 대신 간단한 문자열 분할 사용)
    sections = content.split('---')
    for section in sections:
        lines = section.strip().split('\n')
        for i in range(len(lines)):
            if lines[i].strip().startswith(tuple(f"{j}." for j in range(1, 11))):
                # 질문 행 찾기
                q_line = lines[i].strip()
                question = q_line.split('질문:')[1].strip() if '질문:' in q_line else q_line.split('.', 1)[1].strip()
                
                # 다음 행에서 모범답변 찾기
                if i + 1 < len(lines) and '모범답변:' in lines[i+1]:
                    answer = lines[i+1].split('모범답변:')[1].strip()
                    data.append({"question": question, "ground_truth": answer})
    return data

def run_evaluation():
    print("🚀 RAGAS 평가를 시작합니다...")
    
    # 1. 데이터 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "ragas_file.txt")
    eval_sets = parse_ragas_data(data_file)
    
    if not eval_sets:
        print("❌ 평가용 데이터를 찾을 수 없습니다.")
        return

    # 2. RAG 엔진 초기화
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    engine = InvestmentRAGEngine(pdf_files, index_path="faiss_index")

    # 3. 답변 및 컨텍스트 수집
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(eval_sets):
        print(f"[{i+1}/{len(eval_sets)}] 질문 처리 중: {item['question'][:30]}...")
        # engine.process_query returns response, sources, contexts
        response, sources, retrieved_contexts = engine.process_query(item['question'])
        
        questions.append(item['question'])
        answers.append(response)
        contexts.append(retrieved_contexts)
        ground_truths.append(item['ground_truth'])

    # 4. RAGAS 데이터셋 생성
    ds_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(ds_dict)

    # 5. 평가 실행 (OpenAI 모델 사용)
    # RAGAS 0.2.0+ 에서는 Langchain 객체를 전용 Wrapper로 감싸야 합니다.
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        ContextPrecision(llm=eval_llm),
        ContextRecall(llm=eval_llm),
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    # 6. 결과 출력 및 저장
    print("\n📊 --- 평가 결과 ---")
    df = result.to_pandas()
    print(df)
    
    # 숫자형 컬럼만 평균 계산
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        summary = numeric_df.mean()
        print("\n📈 --- 평균 점수 ---")
        print(summary)

    output_path = os.path.join(current_dir, "evaluation_results.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 결과가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    run_evaluation()
