FROM python:3.12-slim

RUN set -ex \
    && apt update \
    && apt install -y apt-transport-https curl software-properties-common build-essential \
    && apt-get update

WORKDIR /tmp

COPY ./requirements.txt /opt/requirements.txt
RUN pip3 --no-cache-dir install -r /opt/requirements.txt


ENV PYTHONPATH="/app:/app"

WORKDIR /app
