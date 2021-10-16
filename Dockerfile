FROM python:3.8.9-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt update \
 && apt install -y git-lfs gcc \
 && git lfs install \
 && git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --branch main --single-branch /model/paraphrase-multilingual-mpnet-base-v2 \
 && rm -rf /model/paraphrase-multilingual-mpnet-base-v2/.git \
 && chown -R ${NB_USER}: /model \
 && apt remove --purge -y git && apt autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
COPY ./requirements-dev.txt /app/requirements-dev.txt
RUN pip install --disable-pip-version-check --no-cache-dir -r requirements-dev.txt

COPY src /app/src

RUN echo 'alias l="ls -lA --color --group-directories-first"' >> ~/.bashrc
CMD ["bash"]
