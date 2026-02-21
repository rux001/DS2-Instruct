"""BM25-based document retrieval for keyword expansion."""

import json
import os
import re
import random
from rank_bm25 import BM25Okapi

WIKIPEDIA_DATA_DIR = "wikipedia_data"


def tokenize(text):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def load_documents_from_dir(data_dir=WIKIPEDIA_DATA_DIR):
    docs = []
    for fname in os.listdir(data_dir):
        if fname == "metadata.json":
            continue
        path = os.path.join(data_dir, fname)
        if fname.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    text = _extract_text(json.loads(line))
                    if text and len(text) > 50:
                        docs.append(text)
        elif fname.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = _extract_text(item)
                    if text and len(text) > 50:
                        docs.append(text)
    return docs


def _extract_text(obj):
    return obj.get("text", obj.get("content", obj.get("webpage", "")))


def load_keywords(keywords_file):
    with open(keywords_file) as f:
        data = json.load(f)
    return data.get("keywords", data.get("all_keywords", []))


def build_query(task_description, keywords, num_keywords=10):
    sampled = random.sample(keywords, min(num_keywords, len(keywords)))
    return " ".join([task_description] + sampled)


def retrieve_documents(docs, query, top_k=1000):
    tok_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tok_docs)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:top_k]]
