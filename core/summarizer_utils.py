import os
import requests
import logging
from config import CONFIG

logger = logging.getLogger("summarizer_utils")
logger.setLevel(logging.INFO)

# ambil config LLM dari file utama
LLM_URL = CONFIG["llm"]["base_url"]
LLM_MODEL = CONFIG["llm"]["model"]
LLM_API_KEY = CONFIG["llm"]["api_key"]
TIMEOUT = CONFIG["llm"]["timeout_sec"]

HEADERS = {
    "Authorization": f"Bearer {LLM_API_KEY}",
    "Content-Type": "application/json"
}

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Ringkas teks dengan LLM yang terhubung via konfigurasi CONFIG["llm"].
    """
    try:
        # batasi panjang input agar tidak boros token
        snippet = text.strip()[:4000]
        prompt = (
            f"Ringkas teks berikut menjadi maksimal {max_sentences} kalimat yang padat, jelas, "
            f"dan tetap mempertahankan konteks penting.\n\nTeks:\n{snippet}"
        )

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Anda adalah asisten yang ahli dalam meringkas dokumen panjang menjadi versi singkat yang mudah dipahami."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "top_p": 0.7,
            "max_tokens": 400
        }

        logger.info(f"[SUMMARIZER] Mengirim teks ({len(snippet)} chars) ke LLM model '{LLM_MODEL}'")

        resp = requests.post(LLM_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"[SUMMARIZER] HTTP {resp.status_code}: {resp.text[:200]}")
            return snippet[:350] + "..."

        data = resp.json()
        summary = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not summary:
            logger.warning("[SUMMARIZER] Tidak ada konten balikan dari LLM, pakai fallback.")
            summary = snippet[:350] + "..."

        return summary

    except Exception as e:
        logger.error(f"[SUMMARIZER] Gagal meringkas: {e}")
        return text.strip()[:350] + "..."
