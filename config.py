import os
from dotenv import load_dotenv

# load .env jika ada
load_dotenv()

def _env(key, default=None, cast=str):
    val = os.getenv(key, default)
    if cast is int:
        try:
            return int(val)
        except Exception:
            return int(default) if default is not None else None
    return val

CONFIG = {
    "doc_api":{
        "host": _env("DOC_API_HOST"),
        "port": _env("DOC_API_PORT", 5100, int),
        "base_url": _env("DOC_API_BASE_URL", "http://localhost:5100")
    },
    "api": {
        "host": _env("API_HOST"),
        "port": _env("API_PORT", 5000, int)
    },
    "embeddings": {
        "model_path": _env("EMB_MODEL_PATH"),
        "model_path_large": _env("EMB_LARGE_PATH")
    },
    "qdrant": {
        "host": _env("QDRANT_HOST"),
        "port": _env("QDRANT_PORT", 6333, int)
    },
    # LLM Config Lama (OpenAI-compatible)
    # "llm": {
    #     "base_url": _env("LLM_BASE_URL"),
    #     "api_key": _env("LLM_API_KEY"),
    #     "model": _env("LLM_MODEL"),
    #     "timeout_sec": _env("LLM_TIMEOUT_SEC", 60, int)
    # },
    
    # LLM Config Baru (Gemini)
    "llm": {
        "base_url": _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models"),
        "api_key": _env("LLM_API_KEY", "AIzaSyAR9HwRVob0xiPoI8CfRcYQ9hpXNsDc0fM"),
        "model": _env("LLM_MODEL", "gemini-2.5-flash-lite"),
        "timeout_sec": _env("LLM_TIMEOUT_SEC", 60, int),
        "provider": "gemini"  # gemini or openai
    },
    "db": {
        "host": _env("DB_HOST"),
        "port": _env("DB_PORT", 3306, int),
        "database": _env("DB_DATABASE"),
        "username": _env("DB_USERNAME"),
        "password": _env("DB_PASSWORD")
    },
    "ocr": {
        "engine": _env("OCR_ENGINE", "paddle"),
        "lang": _env("OCR_LANG", "id")
    },
    "rag": {
        "use_post_summary": _env("USE_POST_SUMMARY", "false").lower() == "true",
        "post_summary_top_k": _env("POST_SUMMARY_TOP_K", 2, int)
    }


}
