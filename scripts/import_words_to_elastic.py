#!/usr/bin/env python3

import os
import time
import logging
import numpy as np
from zipfile import Path
from typing import List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import NotFoundError

from src.embedding_similarity import get_bert_embedding

logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.WARNING)

INDEX_NAME = "bert_embeddings"
INDEX_CONTENT = {
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
    },
    "mappings": {
        "dynamic": "true",
        "_source": {"enabled": "true"},
        "properties": {
            "title": {"type": "text"},
            "language": {"type": "text"},
            "bert_vector": {"type": "dense_vector", "dims": 768},
        }
    }
}
WORDS = {
    "fr_22k": Path("dictionnaires/french_22k_words.zip", "french_22k_words.txt"),
    "fr_336k": Path("dictionnaires/french_336k_words.zip", "french_336k_words.txt"),
}
BATCH_SIZE = 1000
SEARCH_SIZE = 10


def main():
    elastic = Elasticsearch(os.getenv("ELASTIC_CX_STRING"), verify_certs=False, timeout=120)
    print("Available index :\n" + elastic.cat.indices())
    reset_elastic(elastic)

    # search_by_component(elastic)
    
    while True:
        manual_query(elastic)


def reset_elastic(elastic: Elasticsearch):
    try:
        _ = elastic.cat.indices(index=INDEX_NAME, format="json")
        index_exists = True
    except NotFoundError:
        index_exists = False

    if index_exists:
        reindex = input(f"Drop and create elastic index {INDEX_NAME!r} ? [y/N] ")
        reindex = "n" if len(reindex) == 0 else reindex
        if reindex.lower()[0] != "y":
            return

    print(f"Creating the {INDEX_NAME!r} index.")
    elastic.indices.delete(index=INDEX_NAME, ignore=[404])
    elastic.indices.create(index=INDEX_NAME, body=INDEX_CONTENT)
    
    index_data(elastic, WORDS["fr_22k"])


def index_data(elastic: Elasticsearch, file_path: Path):
    texts = [
        word.strip()
        for word in file_path.read_text().split("\n")
    ]

    for start in range(0, len(texts), BATCH_SIZE):
        end = min(len(texts), start + BATCH_SIZE)
        index_batch(elastic, texts[start: end])
        print(f"Indexed {end} documents.")

    elastic.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")


def index_batch(elastic: Elasticsearch, texts: List[str]):
    bert_vectors = compute_embeddings(texts)

    requests = [
        {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "title": text,
            "bert_vector": vector,
        }
        for text, vector in zip(texts, bert_vectors)
    ]
    bulk(elastic, requests)


##### SEARCHING #####

def manual_query(elastic):
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = compute_embeddings([query])[0]
    embedding_time = time.time() - embedding_start
    print(f"embedding time: {embedding_time * 1000:.2f} ms")

    return handle_query(elastic, query_vector)


def handle_query(elastic, query_vector: np.array):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "Math.abs(cosineSimilarity(params.query_vector, doc['bert_vector']))",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    response = elastic.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "bert_vector"]}
        }
    )
    search_time = time.time() - search_start

    print(response["hits"]["total"]["value"], "total hits.")
    print(f"search time: {search_time * 1000:.2f} ms")
    for hit in response["hits"]["hits"]:
        print("  id: {0}, name: {1:<30}, score: {2}".format(hit["_id"], hit["_source"]["title"], hit["_score"]))


def search_by_component(elastic: Elasticsearch):
    """Try to explain each component of BERT"""
    for i in range(10):
        vector = np.zeros(768)
        vector[i] = 1

        print("Search on component", i)
        handle_query(elastic, vector)


##### EMBEDDING #####

def compute_embeddings(text: List[str]):
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    bert_embedding = get_bert_embedding(model_name=model_name)
    sentence_embeddings = bert_embedding.run_model(text)
    return sentence_embeddings.detach().numpy()


if __name__ == '__main__':
    main()
