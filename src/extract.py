"""1단계: 데이터 추출 — PDF를 텍스트 / 표 / 이미지로 분리합니다.

- 위치(레이아웃) 인식: unstructured hi_res (표·그림 영역 탐지, 그림은 figures/ 에 저장)
- 표 내용: 표 영역 안의 글자를 PDF 텍스트 레이어에서 PyMuPDF로 읽음
  (Tesseract 한국어 데이터가 없으면 표 OCR 결과가 깨지기 때문)
- 본문: 그림 내부 OCR 잡음(Image 요소)을 빼고 제목 단위로 다시 묶음
"""
import re
from pathlib import Path

import pymupdf
from langchain_text_splitters import CharacterTextSplitter
from PIL import Image as PILImage
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(pdf_path, image_dir):
    """PDF에서 요소(본문/표/그림)를 추출하고, 그림은 image_dir 에 .jpg 로 저장합니다."""
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                 # 레이아웃 모델로 표·그림 위치까지 인식
        extract_images_in_pdf=True,        # 그래프·그림을 이미지 파일로 저장
        infer_table_structure=True,        # 표 영역을 Table 요소로 분리
        extract_image_block_output_dir=str(image_dir),
    )


def _table_text_from_pdf(doc, element):
    """표 영역 좌표를 PDF 좌표로 바꿔, 그 안의 글자를 텍스트 레이어에서 읽습니다."""
    meta = element.metadata
    if not (meta.coordinates and meta.page_number):
        return str(element)
    page = doc[meta.page_number - 1]
    sx = page.rect.width / meta.coordinates.system.width
    sy = page.rect.height / meta.coordinates.system.height
    xs = [p[0] * sx for p in meta.coordinates.points]
    ys = [p[1] * sy for p in meta.coordinates.points]
    # 표 위쪽의 제목 "< ... >" 과 단위 줄까지 포함하도록 위로 40pt 넓힘
    clip = pymupdf.Rect(min(xs) - 5, min(ys) - 40, max(xs) + 5, max(ys) + 5)
    text = page.get_text("text", clip=clip, sort=True)
    return f"[{meta.page_number}쪽 표]\n" + re.sub(r"[ \t]+", " ", text).strip()


def _is_real_table(text):
    """숫자가 거의 없는 영역(담당자 연락처 등)은 표가 아니라고 보고 제외합니다."""
    return len(re.findall(r"\d+\.\d", text)) >= 4


def categorize_elements(raw_pdf_elements, pdf_path):
    """요소를 텍스트 블록과 표로 나눕니다.

    texts : 제목 단위로 묶은 본문 (그림 내부 OCR 잡음 제외)
    tables: PDF 텍스트 레이어에서 읽은 표 내용
    """
    doc = pymupdf.open(str(pdf_path))
    tables = []
    for el in raw_pdf_elements:
        if type(el).__name__ == "Table":
            text = _table_text_from_pdf(doc, el)
            if _is_real_table(text):
                tables.append(text)

    body = [el for el in raw_pdf_elements if type(el).__name__ not in ("Image", "Table")]
    chunks = chunk_by_title(
        body,
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    texts = [str(c) for c in chunks if str(c).strip()]
    return texts, tables


def figure_captions(raw_pdf_elements, pdf_path):
    """그림마다 바로 위에 인쇄된 제목 '< ... >'을 찾아 {파일명: 제목} 으로 돌려줍니다.

    잘라낸 그래프 이미지에는 제목이 빠져 있는 경우가 많아,
    이미지 요약 때 제목을 함께 알려 주면 요약(=검색 키)이 훨씬 정확해집니다.
    """
    doc = pymupdf.open(str(pdf_path))
    captions = {}
    for el in raw_pdf_elements:
        meta = el.metadata
        path = getattr(meta, "image_path", None)
        if type(el).__name__ != "Image" or not path or not meta.coordinates:
            continue
        page = doc[meta.page_number - 1]
        sx = page.rect.width / meta.coordinates.system.width
        sy = page.rect.height / meta.coordinates.system.height
        xs = [p[0] * sx for p in meta.coordinates.points]
        ys = [p[1] * sy for p in meta.coordinates.points]
        found = []
        for up in (60, 120):     # 그림 위쪽 60pt → 없으면 120pt까지 넓혀서 찾음
            clip = pymupdf.Rect(min(xs) - 30, min(ys) - up, max(xs) + 30, min(ys) + 5)
            text = page.get_text("text", clip=clip, sort=True)
            found = re.findall(r"<\s*([^<>]{3,80}?)\s*>", text.replace("\n", " "))
            if found:
                break
        if found:
            captions[Path(path).name] = re.sub(r"\s+", " ", found[-1]).strip()
    return captions


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트 블록을 이어 붙인 뒤 토큰 기준으로 다시 자릅니다."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))


def select_figures(image_dir, min_kb=20, max_aspect=5.0, skip_pages=(1,)):
    """검색에 쓸 그림만 고릅니다.

    - 1쪽(표지)의 로고·글자 조각 제외
    - 20KB 미만의 아주 작은 조각 제외
    - 가로로 너무 긴 띠 모양 조각(가로/세로 > 5) 제외
    파일명은 figure-<쪽>-<번호>.jpg 형식입니다.
    """
    selected = []
    for p in sorted(Path(image_dir).glob("figure-*.jpg")):
        page = int(p.stem.split("-")[1])
        if page in skip_pages or p.stat().st_size < min_kb * 1024:
            continue
        with PILImage.open(p) as im:
            w, h = im.size
        if w / h > max_aspect:
            continue
        selected.append(p)
    return sorted(selected, key=lambda p: (int(p.stem.split("-")[1]), int(p.stem.split("-")[2])))
