"""2단계: 요약 생성 — 텍스트·표·이미지를 '검색용 요약문'으로 바꿉니다.

수업 실습(02_요약생성)과 같은 함수 구조입니다.
다른 점: 프롬프트를 통계 자료에 맞게 고쳤고, 이미지 요약 때 그래프 제목을 함께 알려 줍니다.

요약문은 검색에만 쓰이고, 답변을 만들 때는 원본(텍스트·표·이미지)을 씁니다.
"""
import base64

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 요약본을 생성합니다.

    texts          : 텍스트 리스트
    tables         : 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부
    """
    prompt_text = """당신은 통계 보도자료의 표와 텍스트를 검색용으로 요약하는 도우미입니다.
이 요약문은 임베딩되어 원본 표/텍스트를 검색하는 데 쓰입니다.
- 반드시 한국어로, 마크다운 없이 3~5문장의 평문으로 쓰세요.
- 지표 이름, 대상 연령, 연도, 수치(%, 점, 건), 증감 방향을 빠짐없이 담으세요.
- 원문에 없는 내용은 추측하지 마세요.

요약할 표 또는 텍스트: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # 컨베이어 벨트: 입력 → 프롬프트 → 모델 → 글자만 꺼내기
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지를 base64 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """이미지 요약 생성 (VLM 호출)"""
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)

    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ])
    ])
    return msg.content


def generate_img_summaries(image_files, captions):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성

    image_files: 요약할 이미지 파일 목록 (extract.select_figures 결과)
    captions   : {파일 이름: 그래프 제목} (extract.find_captions 결과)
    """
    img_base64_list = []
    image_summaries = []

    prompt_kor = """You are an assistant tasked with summarizing charts from a Korean
statistics press release for retrieval. These summaries will be embedded and used
to retrieve the raw image.
Describe: the chart title, chart type, axes/units, every category or year shown,
the key numeric values written on the chart, and the overall trend or ranking.
Do not invent numbers that are not visible.
The caption printed above this chart in the document is: "{caption}".
Start the summary with that caption.
The summary should be written in Korean (Hangul), as plain text in 4-6 sentences."""

    for img_file in image_files:
        caption = captions.get(img_file.name, "unknown")
        base64_image = encode_image(img_file)
        img_base64_list.append(base64_image)
        image_summaries.append(image_summarize(base64_image, prompt_kor.format(caption=caption)))

    return img_base64_list, image_summaries
