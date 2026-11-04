# Multimodal RAG

PDF 속 텍스트, 표, 이미지를 함께 검색하는 멀티모달 RAG 실습 프로젝트입니다.

## 설치

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

버전을 그대로 재현하려면 `pip install -r requirements.lock.txt`를 사용하세요.

`hi_res` 전략은 Poppler와 Tesseract가 필요합니다. (pip 패키지가 아니므로 따로 설치하고 PATH에 등록)

```
winget install UB-Mannheim.TesseractOCR
winget install oschwartz10612.Poppler
```

## 환경 변수

`.env.example`을 `.env`로 복사한 뒤 키를 채웁니다.

## 구조

- `src/extract.py` PDF 파싱
- `src/summarize.py` 요약 체인
- `src/retriever.py` 다중 벡터 검색기
- `notebooks/Multi_modal_RAG.ipynb` 실습 노트북
