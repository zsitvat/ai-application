# syntax=docker/dockerfile:1

FROM python:3.11-slim-buster

WORKDIR /app

RUN pip install --upgrade pip

RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends build-essential 

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5000

WORKDIR /app/src

CMD ["python3","app.py"]