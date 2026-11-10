"""3단계: 검색과 답변 — 요약문으로 검색하고, 원본으로 답변을 생성합니다.

MultiVectorRetriever
  - vectorstore(Chroma): 요약문 임베딩 (검색용, 짧고 키워드가 정돈됨)
  - docstore(InMemoryStore): 원본 텍스트 / 표 / 이미지(base64)
  - 요약문과 원본은 같은 doc_id 로 연결
"""
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"   # 한국어 지원, 저렴
ANSWER_MODEL = "gpt-5-mini"                  # 이미지 입력 가능 + 그래프 수치 읽기가 gpt-4o-mini보다 정확
ID_KEY = "doc_id"

SYSTEM_PROMPT = """당신은 「아동·청소년 삶의 질 2025」 보도자료를 근거로 답하는 분석가입니다.
아래에 검색된 원본 자료(텍스트, 표, 그래프 이미지)만 근거로 질문에 한국어로 답하세요.
- 수치는 자료에 적힌 값 그대로 쓰고, 단위(%, 점, 건)를 붙이세요.
- 그래프 이미지에서 읽은 값이면 그렇다고 밝히세요.
- 자료에서 찾을 수 없으면 추측하지 말고 "자료에서 확인할 수 없습니다"라고 답하세요.
- 두 값을 비교할 때는 어느 쪽이 크고 작은지 숫자로 다시 확인한 뒤 '높다/낮다'를 쓰세요.
- 2~4문장의 평문으로 간결하게 답하세요."""


def create_multi_vector_retriever(
    text_summaries, texts, table_summaries, tables,
    image_summaries, images, image_names, k=5,
):
    """요약문은 Chroma에, 원본은 docstore에 넣어 MultiVectorRetriever를 만듭니다."""
    vectorstore = Chroma(
        collection_name=f"mm_rag_{uuid.uuid4().hex[:8]}",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
        search_kwargs={"k": k},
    )

    def add(summaries, originals, kind, sources):
        if not summaries:
            return
        ids = [str(uuid.uuid4()) for _ in originals]
        retriever.vectorstore.add_documents([
            Document(page_content=s, metadata={ID_KEY: ids[i], "type": kind, "source": sources[i]})
            for i, s in enumerate(summaries)
        ])
        retriever.docstore.mset([
            (ids[i], Document(page_content=o, metadata={"type": kind, "source": sources[i]}))
            for i, o in enumerate(originals)
        ])

    add(text_summaries, texts, "text", [f"text-{i}" for i in range(len(texts))])
    add(table_summaries, tables, "table", [f"table-{i}" for i in range(len(tables))])
    add(image_summaries, images, "image", list(image_names))
    return retriever


def create_text_retriever(texts, k=5):
    """비교용(Before): 본문 텍스트 청크만 그대로 임베딩한 일반 텍스트 RAG 검색기"""
    vectorstore = Chroma(
        collection_name=f"text_rag_{uuid.uuid4().hex[:8]}",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    vectorstore.add_documents([
        Document(page_content=t, metadata={"type": "text", "source": f"text-{i}"})
        for i, t in enumerate(texts)
    ])
    return vectorstore.as_retriever(search_kwargs={"k": k})


def _build_messages(inputs):
    """검색된 원본을 텍스트 부분과 이미지 부분으로 나눠 하나의 멀티모달 메시지로 만듭니다."""
    docs, question = inputs["docs"], inputs["question"]
    text_parts = [d.page_content for d in docs if d.metadata.get("type") != "image"]
    image_docs = [d for d in docs if d.metadata.get("type") == "image"]

    content = [{
        "type": "text",
        "text": f"{SYSTEM_PROMPT}\n\n질문: {question}\n\n[텍스트·표 자료]\n" + "\n\n---\n\n".join(text_parts),
    }]
    for d in image_docs:
        content.append({"type": "text", "text": f"[그래프 이미지: {d.metadata.get('source')}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d.page_content}"}})
    return [HumanMessage(content=content)]


def build_rag_chain(retriever, model_name=ANSWER_MODEL):
    """질문 → {docs, question, answer}. 어떤 원본이 근거였는지 함께 돌려줍니다."""
    if model_name.startswith("gpt-5"):   # 추론 모델: temperature 고정, 추론량만 조절
        model = ChatOpenAI(model=model_name, reasoning_effort="low")
    else:
        model = ChatOpenAI(model=model_name, temperature=0, max_tokens=1024)
    answer =RunnableLambda(_build_messages) | model | StrOutputParser()
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer)


def source_types(docs):
    """근거 원본 종류 요약. 예: 'image(figure-5-242.jpg), text(text-1)'"""
    return ", ".join(f"{d.metadata.get('type')}({d.metadata.get('source')})" for d in docs)
