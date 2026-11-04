"""PDF 파싱: UnstructuredLoader로 PDF를 조각(chunk)으로 분리합니다."""
from langchain_unstructured import UnstructuredLoader


def extract_chunks(file_path, strategy="hi_res", max_characters=1000):
    loader = UnstructuredLoader(
        file_path=file_path,
        partition_via_api=False,         # False = 로컬에서 처리 (API 키 불필요)
        strategy=strategy,               # fast(빠름) vs hi_res(정확/느림)
        chunking_strategy="by_title",    # 제목 단위로 문서를 잘라줌 (RAG에 유리)
        max_characters=max_characters,   # 조각 하나의 최대 글자 수
    )
    return loader.load()
