"""1단계: 데이터 추출 — PDF를 텍스트 / 표 / 이미지로 나눕니다.

수업 실습(01_데이터추출)과 같은 흐름에, 이 PDF에 맞게 3가지를 더했습니다.
  1) 표 글자는 PDF에서 직접 읽기   (한국어 OCR이 안 돼서 표 글자가 깨지기 때문)
  2) 그래프 위에 적힌 제목 찾기     (잘라낸 그래프 이미지에는 제목이 없기 때문)
  3) 쓸모없는 작은 이미지 빼기     (로고, 글자 조각 등)
"""
import re
from pathlib import Path

import pymupdf
from langchain_text_splitters import CharacterTextSplitter
from PIL import Image
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf


def extract_pdf_elements(path, fname, image_dir):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    path     : PDF가 있는 폴더 (끝에 경로 구분자 포함)
    fname    : PDF 파일 이름
    image_dir: 추출한 그림을 저장할 폴더 (figures/)
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                 # 정밀 추출 (표·그림 위치까지 인식)
        extract_images_in_pdf=True,        # 그림을 이미지 파일로 저장
        infer_table_structure=True,        # 표 구조 추론
        extract_image_block_output_dir=str(image_dir),
        # 실습과 달리 chunking_strategy 는 여기서 쓰지 않습니다.
        # → 그래프 안의 깨진 OCR 글자를 먼저 빼고, categorize_elements 에서 묶습니다.
    )


def read_table_text(pdf_path, table):
    """표 영역 안의 글자를 PDF에서 직접 읽어 옵니다."""
    doc = pymupdf.open(pdf_path)
    page = doc[table.metadata.page_number - 1]

    # unstructured 좌표(이미지 픽셀) → PDF 좌표(pt) 로 바꾸는 비율
    scale = page.rect.width / table.metadata.coordinates.system.width
    points = table.metadata.coordinates.points       # [왼쪽위, 왼쪽아래, 오른쪽아래, 오른쪽위]
    x0, y0 = points[0]
    x1, y1 = points[2]

    # 표 위에 있는 제목 "< ... >" 까지 읽으려고 위쪽으로 40pt 더 넓게 잡습니다
    area = pymupdf.Rect(x0 * scale - 5, y0 * scale - 40, x1 * scale + 5, y1 * scale + 5)
    text = page.get_text("text", clip=area, sort=True)
    text = re.sub(r"[ \t]+", " ", text).strip()      # 여러 칸 띄어쓰기를 한 칸으로
    return f"[{table.metadata.page_number}쪽 표]\n{text}"


def categorize_elements(raw_pdf_elements, pdf_path):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.

    raw_pdf_elements: extract_pdf_elements() 의 결과
    pdf_path        : 표 글자를 읽을 PDF 경로
    """
    tables = []
    body = []

    for element in raw_pdf_elements:
        if "Table" in str(type(element)):
            table_text = read_table_text(str(pdf_path), element)
            # 소수점 숫자가 4개 미만이면 표가 아님 (담당자 연락처 상자 등)
            if len(re.findall(r"\d+\.\d", table_text)) >= 4:
                tables.append(table_text)
        elif "Image" in str(type(element)):
            continue                     # 그래프 안의 OCR 글자는 깨져 있어서 버림
        else:
            body.append(element)

    # 실습의 chunking_strategy="by_title" 과 같은 설정으로 제목 단위로 묶기
    chunks = chunk_by_title(
        body,
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    texts = [str(chunk) for chunk in chunks]
    return texts, tables


def split_texts(texts):
    """텍스트 블록을 이어 붙인 뒤 2000토큰 단위로 다시 자릅니다. (실습과 동일)"""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200,
    )
    joined_texts = " ".join(texts)
    return text_splitter.split_text(joined_texts)


def select_figures(figures_dir):
    """figures/ 에서 검색에 쓸 그래프만 고릅니다. 파일 이름: figure-<쪽>-<번호>.jpg"""
    selected = []
    for f in Path(figures_dir).iterdir():
        if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        page = int(f.stem.split("-")[1])
        size_kb = f.stat().st_size // 1024
        with Image.open(f) as img:
            width, height = img.size

        if page == 1:                # 1쪽 표지의 로고
            continue
        if size_kb < 20:             # 20KB 미만의 아주 작은 조각
            continue
        if width / height > 5:       # 가로로 너무 긴 띠 모양 조각
            continue
        selected.append(f)

    # 번호 순서(= 쪽 순서)로 정렬
    return sorted(selected, key=lambda f: int(f.stem.split("-")[2]))


def find_captions(raw_pdf_elements, pdf_path):
    """그래프마다 바로 위에 적힌 제목 "< ... >" 을 찾아 {파일 이름: 제목} 으로 돌려줍니다."""
    doc = pymupdf.open(str(pdf_path))
    captions = {}

    for element in raw_pdf_elements:
        if "Image" not in str(type(element)):
            continue
        image_path = element.metadata.image_path
        if not image_path:
            continue

        page = doc[element.metadata.page_number - 1]
        scale = page.rect.width / element.metadata.coordinates.system.width
        points = element.metadata.coordinates.points
        x0, y0 = points[0]
        x1, y1 = points[2]

        # 그래프 위쪽 60pt 안에서 찾고, 없으면 120pt 까지 넓혀서 다시 찾기
        for up in [60, 120]:
            area = pymupdf.Rect(x0 * scale - 30, y0 * scale - up, x1 * scale + 30, y0 * scale + 5)
            text = page.get_text("text", clip=area, sort=True).replace("\n", " ")
            found = re.findall(r"<\s*([^<>]{3,80}?)\s*>", text)
            if found:
                captions[Path(image_path).name] = " ".join(found[-1].split())   # 가장 가까운 제목
                break

    return captions
