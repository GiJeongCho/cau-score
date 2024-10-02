from pydantic import BaseModel
from typing import List, Dict, Any
import spacy
import json
import re
from collections import defaultdict
import os

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

resource_dir = os.getenv('RESOURCE_DIR', 'v1/resources')

# JSON 데이터 로드 (단어 레벨용)
# JSON 데이터 로드 (단어 레벨용)
with open(f'{resource_dir}/resources/final_corrected_combined_word_levels.json', 'r', encoding='utf-8') as f:
    word_levels = json.load(f)

# TXT 데이터 로드 (워드 패밀리 체크용)
def load_word_families(file):
    word_families = []
    with open(f'{resource_dir}/resources/{file}', 'r') as f:
        for line in f:
            word_families.append(set(line.strip().split()))
    return word_families

word_families = load_word_families('merged_word_family_no_duplicates.txt')

# 요청 데이터 모델 정의
class PosTypesRequest(BaseModel):
    sentences: str

class ErrorRateRequest(BaseModel):
    correct_sentence: str
    incorrect_sentence: str

class SentenceErrorRateRequest(BaseModel):
    correct_sentences: str
    student_sentences: str

class WordClassificationRequest(BaseModel):
    sentence: str

class WordFamilyRequest(BaseModel):
    input_sentences: str
    script_sentences: str

class countrepeatedwordsRequest(BaseModel):
    input_sentences : str
    
class CalculateTTROverlapRequest(BaseModel):
    student_response : str
    script_reference : str

def calculate_lcs(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L

def backtrack_lcs(L, X, Y):
    i, j = len(X), len(Y)
    lcs = []

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs[::-1]

    
# 딕셔너리 빌드 함수
def build_lookup_dict(correct_sentences, word_families):
    """
    correct_sentences의 단어들을 워드 패밀리에서 찾아 딕셔너리 형태로 리턴
    """
    correct_words = set()
    
    # correct_sentences를 문장 단위로 분리
    sentences = re.split(r'[.!?]', correct_sentences)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # correct_sentences의 각 문장에 대해 단어 추출
    for sentence in sentences:
        words = set(re.findall(r'\b\w+\b', sentence.lower()))
        correct_words.update(words)

    lookup_dict = defaultdict(set)

    # 워드 패밀리와 correct_words 비교
    for word in correct_words:
        for family in word_families:
            if word in family:
                lookup_dict[word] = family
                break

    return lookup_dict, len(correct_words)  # correct_sentences의 전체 단어 수와 lookup_dict 리턴

## 주요 함수 정의
async def count_repeated_words(request:countrepeatedwordsRequest):
    check_word = []  # 반복된 원래 단어를 저장할 리스트
    """
    입력된 문장에서 워드 패밀리를 이용해 반복 발화된 단어의 빈도와 문장 수를 계산하는 함수
    
    :param input_sentences: 여러 문장이 하나의 문자열로 들어옴 (한 줄의 문자열)
    :param word_families: 워드 패밀리 (유사 단어 집합)
    :return: repeated_word_count (반복 발화된 단어의 빈도), total_sentences (문장의 수)
    """

    # 입력된 문장을 문장 단위로 분리
    sentence_list = re.split(r'[.!?]', request.input_sentences)
    sentence_list = [sentence.strip() for sentence in sentence_list if sentence.strip()]

    total_sentences = len(sentence_list)  # 총 문장 수

    # 단어 빈도 저장을 위한 딕셔너리
    word_count = defaultdict(int)

    # 워드 패밀리에서 단어 검색을 위한 딕셔너리 생성
    lookup_dict = defaultdict(set)
    for family in word_families:
        for word in family:
            lookup_dict[word] = family

    matched_words = set()  # 중복 방지를 위해 사용
    repeated_word_count = 0  # 반복된 단어의 빈도

    # 각 문장에서 단어를 추출하고 빈도를 계산
    for sentence in sentence_list:
        # spaCy를 이용해 문장을 토큰화하고 표제어를 추출
        doc = nlp(sentence.lower())
        words_in_sentence = [(token.text, token.lemma_) for token in doc if token.is_alpha]  # 원래 단어와 표제어 추출

        for original_word, lemma in words_in_sentence:
            # 워드 패밀리에서 유사 단어 처리
            if lemma in lookup_dict:
                family_words = lookup_dict[lemma]

                # 유사 단어가 이미 나온 경우 반복 발화로 처리
                if family_words & matched_words:
                    check_word.append(original_word)  # 원래 단어로 기록
                    repeated_word_count += 1
                else:
                    matched_words.update(family_words)  # 새로운 단어 추가

            # 일반 단어 처리
            word_count[lemma] += 1

    return {
        "data": {"check_repeated_word": check_word},
        "result": {"repeated_word_count": repeated_word_count, "total_sentences": total_sentences}
    }

# pos 유형빈도 판단
async def get_pos_types(request: PosTypesRequest): 
    """
    입력된 문장에서 사용된 품사의 종류와 수를 반환하는 함수
    :param sentences: 분석할 문장들 (str)
    :return: 각 문장의 단어별 품사 정보와 각 문장의 품사 수
    """
    sentences = re.split(r'[.!?]', request.sentences)  # 문장을 문단으로 분리
    data = {}

    pos_count = {}  # 모든 문장의 품사를 카운트하기 위한 딕셔너리

    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()  # 문장 양 끝의 공백 제거
        if not sentence:  # 빈 문장은 무시
            continue
        doc = nlp(sentence)
        pos_info = {token.text: token.pos_ for token in doc}
        sentence_key = f"{index}"
        data[sentence_key] = pos_info

        # 각 품사의 빈도를 전체적으로 계산
        for pos in pos_info.values():
            if pos in pos_count:
                pos_count[pos] += 1
            else:
                pos_count[pos] = 1

    return {"data": data, "result": pos_count}

# 오류 및 비오류 단어 비율 정보. => 오류 단어 수, 비오류 단어 수
async def calculate_error_rate(request: ErrorRateRequest): 
    """
    입력된 문장에서 맞은 단어 수와 틀린 단어 수를 계산하는 함수

    :param correct_sentence: 교정된 문장들 (한 줄의 문자열)
    :param incorrect_sentence: 교정 전 문장들 (한 줄의 문자열)
    :return: correct_words_count (맞은 단어 수), error_words_count (틀린 단어 수)
    """

    # correct_sentence와 incorrect_sentence를 문장 단위로 분리
    correct_sentences_list = re.split(r'[.!?]', request.correct_sentence)
    correct_sentences_list = [sentence.strip() for sentence in correct_sentences_list if sentence.strip()]  # 공백 제거 및 빈 문장 필터

    incorrect_sentences_list = re.split(r'[.!?]', request.incorrect_sentence)
    incorrect_sentences_list = [sentence.strip() for sentence in incorrect_sentences_list if sentence.strip()]  # 공백 제거 및 빈 문장 필터

    correct_word_list = []  # 맞은 단어 리스트
    error_word_list = []  # 틀린 단어 리스트

    # 각 문장을 비교하여 단어 오류 수 계산
    for correct_sentence, incorrect_sentence in zip(correct_sentences_list, incorrect_sentences_list):
        correct_words = correct_sentence.split()
        incorrect_words = incorrect_sentence.split()

        # 두 문장을 비교하여 맞은 단어와 틀린 단어를 구분
        for c_word, i_word in zip(correct_words, incorrect_words):
            if c_word == i_word:
                correct_word_list.append(i_word)  # 맞은 단어 리스트
            else:
                error_word_list.append(i_word)  # 틀린 단어 리스트

    correct_words_count = len(correct_word_list)
    error_words_count = len(error_word_list)

    return {"data":{"correct_word_list": correct_word_list,"error_word_list": error_word_list},
    "result":{"correct_words_count": correct_words_count,"error_words_count": error_words_count,}}

async def calculate_sentence_error_rate(request: SentenceErrorRateRequest): 
    correct_list = re.split(r'[.!?]', request.correct_sentences)
    correct_list = [sentence.strip() for sentence in correct_list if sentence.strip()]

    student_list = re.split(r'[.!?]', request.student_sentences)
    student_list = [sentence.strip() for sentence in student_list if sentence.strip()]

    L = calculate_lcs(correct_list, student_list)
    lcs_sentences = backtrack_lcs(L, correct_list, student_list)
    lcs_set = set(lcs_sentences)

    if len(student_list) <= len(correct_list):
        e_sentences = [s for s in correct_list if s not in lcs_set]
        e_words = [[word for word in sentence.split()] for sentence in e_sentences]

    else:
        e_sentences = [s for s in student_list if s not in lcs_set]
        e_words = [[word for word in sentence.split()] for sentence in e_sentences]

    # 처리 로직 수정
    matched_sentence_count = len(lcs_sentences)
    additional_correct_count = len(correct_list) - len(lcs_sentences) if len(correct_list) > len(student_list) else 0
    error_sentence_count = len(student_list) - matched_sentence_count + additional_correct_count
    total_sentences = max(len(correct_list), len(student_list))

    return {
        "data": {
            "err_sentences": e_sentences,
            "err_words": e_words
        },
        "result": {
            "matched_sentence_count": matched_sentence_count,
            "error_sentence_count": error_sentence_count,
            "total_sentences": total_sentences
        }
    }

# 워드레벨 엔진 => 초 중 고 단어 빈도. 산출, 기본 어휘 유형 빈도 => 초low, 초 high , 중, 고 단어의 산출 빈도 만 필요
async def classify_words_in_sentence(request: WordClassificationRequest):
    """
    주어진 문장에서 단어를 분류하여 각 레벨별로 빈도를 계산하는 함수
    
    :param sentence: 분석할 문장 (str)
    :return: 각 레벨별 단어 빈도를 포함하는 딕셔너리 (dict)
    """
    # JSON 파일에서 단어 레벨 데이터 불러오기


    # 문장을 단어로 분리
    words_in_sentence = [word.strip().lower() for word in request.sentence.split()]

    # 빈도를 저장할 딕셔너리 초기화
    word_classification = {
        "el_low": 0,
        "el_high": 0,
        "middle": 0,
        "high": 0,
        "other": 0
    }
    word_classification2 = {
        "el_low": [],
        "el_high": [],
        "middle": [],
        "high": [],
        "other": []
    }

    # 단어의 빈도를 계산하고 레벨에 따라 분류
    for word in words_in_sentence:
        if word in word_levels['el_low']:
            word_classification['el_low'] += 1
            word_classification2['el_low'].append(word)
            
        elif word in word_levels['el_high']:
            word_classification['el_high'] += 1
            word_classification2['el_high'].append(word)
            
        elif word in word_levels['middle']:
            word_classification['middle'] += 1
            word_classification2['middle'].append(word)
            
        elif word in word_levels['high']:
            word_classification['high'] += 1
            word_classification2['high'].append(word)
            
        else:
            word_classification['other'] += 1
            word_classification2['other'].append(word)

    return {"data":word_classification2,
        "result":word_classification}

# 응답 어휘 유형 수 및 예시 답안 어휘 유형 수 계산 함수
async def calculate_ttr_and_type_overlap(request:CalculateTTROverlapRequest):
    count_word_scripts = []
    count_word_student = []
    """
    응답 텍스트와 예시 답안에서 어휘 유형의 TTR(Type-Token Ratio) 및 겹치는 비율을 계산

    :param student_response: 분석할 학생 응답 텍스트 (str)
    :param script_reference: 예시 답안 텍스트 (str)
    :param word_families: 워드 패밀리 리스트 (유사 단어 집합)
    :return: TTR 값 (float), 응답 어휘 유형 수 (int), 예시 답안 어휘 유형 수 (int)
    """

    # 학생 응답 텍스트와 예시 답안 텍스트를 spaCy로 처리하여 토큰화
    response_doc = nlp(request.student_response)
    reference_doc = nlp(request.script_reference)

    # Tokens: 전체 단어 수 (토큰 수) -> 문장 부호, 줄바꿈 등을 제외
    response_tokens = [token.text for token in response_doc if token.is_alpha]
    reference_tokens = [token.text for token in reference_doc if token.is_alpha]

    # Types: 고유한 단어 수 (어휘 유형)
    response_types = set(response_tokens)
    reference_types = set(reference_tokens)

    # 워드 패밀리를 사용한 단어 유형 수 계산
    lookup_dict = defaultdict(set)
    for family in word_families:
        for word in family:
            lookup_dict[word] = family

    # 스크립트에서 워드 패밀리 안에 포함되는 단어 수 계산
    matched_words_in_reference = set()
    for word in reference_types:
        if word in lookup_dict:
            count_word_scripts.append(word)
            matched_words_in_reference.update(lookup_dict[word])
            
    # 응답자의 맞춘 어휘 유형 수 계산
    for word in response_types:
        if word in matched_words_in_reference:
            count_word_student.append(word)

    return {
        "data": {
            "students_vocabulary_type": count_word_student,
            "scripts_vocabulary_type": count_word_scripts,
            "types": response_types,
            "tokens": response_tokens
        },
        "result": {
            "students_vocabulary_type": len(count_word_student),
            "scripts_vocabulary_type": len(count_word_scripts),
            "number_of_types": len(response_types),
            "number_of_tokens": len(response_tokens)
        }
    }
    
# 문장 분석 함수 # "내용"부분 전부 컨트롤.  워드 패밀리에 단어들이 포함된는 지 보는 메인 함수 
async def analyze_sentences(request: WordFamilyRequest):
    check_word = []
    """script_sentences
    input_sentences와 script_sentences 단어들을 기반으로 관련 단어들을 찾아 분석하는 함수

    :param input_sentences: 분석할 단일 문자열 (str)
    :param script_sentences: 올바른 문장들의 문자열 (str)
    :param word_families: 워드 패밀리 데이터 (list of set)
    :return: script_sentences 단어 수, 관련 단어 수, 관련 단어가 포함된 문장 수, 관련 단어가 미포함된 문장 수
    """

    # input_sentences를 문장 단위로 분리
    input_sentences_list = re.split(r'[.!?]', request.input_sentences)
    input_sentences_list = [sentence.strip() for sentence in input_sentences_list if sentence.strip()]  # 공백 제거 및 빈 문장 필터

    # script_sentences 단어 추출 및 딕셔너리 빌드
    lookup_dict, correct_word_count = build_lookup_dict(request.script_sentences, word_families)

    related_word_count = 0
    sentences_with_related_words = 0
    sentences_without_related_words = 0
    matched_words = set()  # 중복 방지를 위한 집합

    # 각 문장에 대해 단어 분석
    for sentence in input_sentences_list:
        words = set(re.findall(r'\b\w+\b', sentence.lower()))  # 문장에서 단어 추출
        found_related = False

        # 각 단어가 워드 패밀리에 포함되는지 확인
        for word, family in lookup_dict.items():
            if words & family and word not in matched_words:
                check_word.append(word)
                related_word_count += 1
                matched_words.add(word)
                found_related = True

        # 관련 단어가 포함된 문장과 미포함된 문장을 카운팅
        if found_related:
            sentences_with_related_words += 1
        else:
            sentences_without_related_words += 1

    # 결과 반환: script_sentences 단어 수, 관련 단어 수, 관련 단어가 포함된 문장 수, 관련 단어가 미포함된 문장 수
    return {"data":{"repects_word(serch_word_family)":check_word},
            "result":{"correct_word_count":correct_word_count, "related_word_count":related_word_count, "sentences_with_related_words":sentences_with_related_words, "sentences_without_related_words":sentences_without_related_words}}
