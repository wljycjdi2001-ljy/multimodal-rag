"""2단계: 요약 생성 — 텍스트·표·이미지를 '검색용 요약문'으로 바꿉니다.

요약문은 벡터DB에 임베딩되어 검색에만 쓰이고,
답변을 만들 때는 요약이 아니라 원본(텍스트·표·이미지)을 LLM에 넘깁니다.
"""
import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SUMMARY_MODEL = "gpt-4o-mini"   # 텍스트·표 요약 + 이미지 요약(비전) 모두 지원, 저렴

TEXT_PROMPT = """당신은 통계 보도자료의 표와 텍스트를 검색용으로 요약하는 도우미입니다.
이 요약문은 임베딩되어 원본 표/텍스트를 검색하는 데 쓰입니다.
- 반드시 한국어로, 마크다운 없이 3~5문장의 평문으로 쓰세요.
- 지표 이름, 대상 연령, 연도, 수치(%, 점, 건), 증감 방향을 빠짐없이 담으세요.
- 원문에 없는 내용은 추측하지 마세요.

요약할 표 또는 텍스트: {element}"""

IMAGE_PROMPT = """You are an assistant tasked with summarizing charts from a Korean
statistics press release for retrieval. These summaries will be embedded and used
to retrieve the raw image.
Describe: the chart title, chart type, axes/units, every category or year shown,
the key numeric values written on the chart, and the overall trend or ranking.
Do not invent numbers that are not visible.
The caption printed above this chart in the document is: "{caption}".
Start the summary with that caption.
The summary should be written in Korean (Hangul), as plain text in 4-6 sentences."""


def generate_text_summaries(texts, tables, model_name=SUMMARY_MODEL):
    """텍스트 청크와 표를 각각 요약합니다."""
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT)
    model = ChatOpenAI(model=model_name, temperature=0)
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = chain.batch(texts, {"max_concurrency": 5}) if texts else []
    table_summaries = chain.batch(tables, {"max_concurrency": 5}) if tables else []
    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지 파일 → base64 문자열"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_message(img_base64, caption):
    return [HumanMessage(content=[
        {"type": "text", "text": IMAGE_PROMPT.format(caption=caption or "unknown")},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
    ])]


def image_summarize(img_base64, caption="", model_name=SUMMARY_MODEL):
    """비전 모델로 이미지 한 장을 요약합니다. caption: 문서에 인쇄된 그래프 제목"""
    chat = ChatOpenAI(model=model_name, temperature=0, max_tokens=1024)
    return chat.invoke(_image_message(img_base64, caption)).content


def generate_img_summaries(image_paths, captions=None, model_name=SUMMARY_MODEL):
    """이미지 목록 → (base64 목록, 요약 목록). 5장씩 동시에 요청합니다.

    captions: {파일명: 그래프 제목} (extract.figure_captions 결과)
    """
    captions = captions or {}
    img_base64_list = [encode_image(p) for p in image_paths]
    chat = ChatOpenAI(model=model_name, temperature=0, max_tokens=1024)
    requests = [
        _image_message(b64, captions.get(Path(p).name, ""))
        for p, b64 in zip(image_paths, img_base64_list)
    ]
    responses = chat.batch(requests, {"max_concurrency": 5})
    image_summaries = [r.content for r in responses]
    return img_base64_list, image_summaries
