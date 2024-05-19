FROM python:3.12-slim

WORKDIR /code

RUN pip install scikit-learn==1.4.2
RUN apt-get update && \
    apt install -y python3-dev
RUN pip install --upgrade pip
RUN pip install poetry

ADD pyproject.toml .

RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-interaction --no-ansi

EXPOSE 8000

COPY . .