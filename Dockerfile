FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY models models
COPY api api

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
