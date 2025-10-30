import os
from sentence_transformers import SentenceTransformer

def load_model_from_path(path: str):
    return SentenceTransformer(path)

def load_model_small(CONFIG: dict):
    path = (CONFIG.get("embeddings", {}) or {}).get("model_path") or os.environ.get("EMB_SMALL_PATH") or "intfloat/multilingual-e5-small"
    return SentenceTransformer(path)

def load_model_large(CONFIG: dict):
    path = (CONFIG.get("embeddings", {}) or {}).get("model_path_large") or os.environ.get("EMB_LARGE_PATH") or "intfloat/multilingual-e5-large"
    return SentenceTransformer(path)

def embed_query(model, text: str):
    return model.encode("query: " + text, normalize_embeddings=True).tolist()

def embed_passage(model, text: str):
    return model.encode("passage: " + text, normalize_embeddings=True).tolist()
