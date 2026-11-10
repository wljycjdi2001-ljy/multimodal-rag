"""3단계: 검색과 답변 — 요약문으로 검색하고, 원본으로 답변을 생성합니다.

MultiVectorRetriever
  - vectorstore(Chroma)   : 요약문 (검색용)
  - docstore(InMemoryStore): 원본 텍스트 / 표 / 이미지(base64)
  - 요약문과 원본은 같은 doc_id 로 연결
"""
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_multi_vector_retriever(text_summaries, texts, table_summaries, tables,
                                  image_summaries, images):
    """요약문은 벡터DB에, 원본은 docstore에 넣어 MultiVectorRetriever를 만듭니다."""
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 5},          # 질문과 가까운 요약 5개를 찾음
    )

    def add_documents(retriever, doc_summaries, doc_contents):
        # 원본마다 고유 번호(doc_id)를 붙입니다
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        # 요약문 → 벡터DB (doc_id 를 metadata 에 기록)
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        # 원본 → docstore (같은 doc_id 로 저장)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


def create_text_retriever(texts):
    """비교용(Before): 본문 텍스트만 그대로 벡터DB에 넣은 일반 텍스트 RAG 검색기"""
    vectorstore = Chroma.from_texts(
        texts,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="text_rag",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def doc_type(doc):
    """검색된 원본이 이미지 / 표 / 텍스트 중 무엇인지 알려 줍니다."""
    if isinstance(doc, Document):
        doc = doc.page_content
    if doc.startswith("/9j/"):          # JPEG 이미지를 base64로 바꾸면 항상 "/9j/" 로 시작
        return "image"
    if "쪽 표]" in doc[:10]:            # extract.read_table_text 가 붙인 "[N쪽 표]" 표시
        return "table"
    return "text"


def split_image_text_types(docs):
    """검색된 원본을 이미지(base64)와 텍스트(본문·표)로 나눕니다."""
    images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if doc_type(doc) == "image":
            images.append(doc)
        else:
            texts.append(doc)
    return {"images": images, "texts": texts}


def img_prompt_func(data_dict):
    """질문 + 텍스트·표 원본 + 이미지 원본을 하나의 메시지로 묶습니다."""
    formatted_texts = "\n\n---\n\n".join(data_dict["context"]["texts"])

    text_message = {
        "type": "text",
        "text": (
            "당신은 「아동·청소년 삶의 질 2025」 보도자료를 근거로 답하는 분석가입니다.\n"
            "아래 자료(텍스트, 표, 그래프 이미지)만 근거로 질문에 한국어로 답하세요.\n"
            "- 수치는 자료에 적힌 값 그대로 쓰고, 단위(%, 점, 건)를 붙이세요.\n"
            "- 그래프 이미지에서 읽은 값이면 그렇다고 밝히세요.\n"
            "- 자료에서 찾을 수 없으면 추측하지 말고 \"자료에서 확인할 수 없습니다\"라고 답하세요.\n"
            "- 두 값을 비교할 때는 어느 쪽이 크고 작은지 숫자로 다시 확인한 뒤 '높다/낮다'를 쓰세요.\n"
            "- 2~4문장의 평문으로 간결하게 답하세요.\n\n"
            f"질문: {data_dict['question']}\n\n"
            f"[텍스트·표 자료]\n{formatted_texts}"
        ),
    }
    messages = [text_message]

    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """질문 → 검색 → 원본을 텍스트/이미지로 나눔 → 프롬프트 → 모델 → 답변"""
    # gpt-5-mini 는 추론 모델이라 temperature 를 바꿀 수 없어서, 추론량만 낮게 둡니다
    model = ChatOpenAI(model="gpt-5-mini", reasoning_effort="low")

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    return chain
