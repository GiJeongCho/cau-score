from fastapi import APIRouter, status, Request
from v1.main import (
    PosTypesRequest, get_pos_types,
    ErrorRateRequest, calculate_error_rate,
    SentenceErrorRateRequest, calculate_sentence_error_rate,
    WordClassificationRequest, classify_words_in_sentence,
    WordFamilyRequest, analyze_sentences,
    countrepeatedwordsRequest, count_repeated_words,
    CalculateTTROverlapRequest, calculate_ttr_and_type_overlap
)

router_v1 = APIRouter(
    prefix="/v1",
    tags=["score"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"}
    },
)

@router_v1.post("/pos-types", summary="POS 유형 빈도 판단 [입력된 문장에서 사용된 품사의 종류와 수를 반환]")
async def pos_types_endpoint(req: PosTypesRequest, request: Request):
    print(f"Request received from {request.client.host}")
    return await get_pos_types(req)

@router_v1.post("/error-rate", summary="오류 및 비오류 단어 비율 계산 [맞는 문장과 틀린 문장을 비교하여 오류 및 비오류 단어의 수 반환]")
async def error_rate_endpoint(req: ErrorRateRequest):
    return await calculate_error_rate(req)

@router_v1.post("/sentence-error-rate", summary="문장 오류 비율 계산 [맞은 문장 수와 틀린 문장 수를 반환]")
async def sentence_error_rate_endpoint(req: SentenceErrorRateRequest):
    return await calculate_sentence_error_rate(req)

@router_v1.post("/word-classification", summary="단어 분류 정보 [초low, 초high, 중등, 고등, 기타로 단어를 분류]")
async def word_classification_endpoint(req: WordClassificationRequest):
    return await classify_words_in_sentence(req)

@router_v1.post("/analyze-sentences", summary="문장 분석 [총 문장 수, 관련 단어 미포함 문장 수 계산]")
async def analyze_sentences_endpoint(req: WordFamilyRequest):
    return await analyze_sentences(req)

@router_v1.post("/count-repeated-words", summary="반복 발화된 단어 빈도 계산 [입력된 문장에서 반복 발화된 단어 빈도와 단어 반환]")
async def count_repeated_words_endpoint(req: countrepeatedwordsRequest):
    return await count_repeated_words(req)

@router_v1.post("/ttr-overlap", summary="TTR 및 어휘 유형 겹침 계산 [응답 텍스트와 예시 답안의 TTR 및 겹치는 어휘 유형 수 계산]")
async def ttr_overlap_endpoint(req: CalculateTTROverlapRequest):
    return await calculate_ttr_and_type_overlap(req)
