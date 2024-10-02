# Base image
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 소스 코드 복사
COPY ./src /app/src
COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock

# Poetry 설치
RUN pip install poetry
RUN poetry install 

# uvicorn과 spacy가 설치되지 않으면 수동 설치
RUN pip install spacy uvicorn fastapi
RUN python -m spacy download en_core_web_sm


# 환경 변수 설정
ENV PYTHONPATH=/app/src
ENV RESOURCE_DIR=/app/src/v1

# 포트 노출
EXPOSE 80

# 헬스체크 설정
HEALTHCHECK --start-period=60s CMD curl -f http://localhost/v1/score/docs || exit 1

# 애플리케이션 실행
ENTRYPOINT ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
