import json, re, requests, logging, traceback
from core.utils import hard_filter_local
from core.db import get_variable
from config import CONFIG

logger = logging.getLogger(__name__)
logger.propagate = False

LLM_URL = CONFIG["llm"]["base_url"]
LLM_HEADERS = {
    "Authorization": f"Bearer {CONFIG['llm']['api_key']}",
    "Content-Type": "application/json"
}
LLM_MODEL = CONFIG["llm"]["model"]


def _extract_json(text: str):
    if not text:
        return None
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            logger.warning(f"[JSON PARSE] ❌ Tidak ditemukan JSON pada teks: {text[:100]}...")
            return None
        return json.loads(m.group(0))
    except Exception as e:
        logger.exception(f"[JSON PARSE] Gagal parsing JSON: {e}")
        return None

def ai_pre_filter(question: str):
    try:
        hard = hard_filter_local(question)
        if not hard["valid"]:
            logger.info(f"[HARD FILTER] ❌ {hard['reason']}")
            return hard

        prompt_db = get_variable("prompt_pre_filter_rag")
        if prompt_db:
            logger.info(f"[DB PROMPT] ✅ prompt_pre_filter_rag ditemukan ({len(prompt_db)} chars)")
        else:
            logger.warning("[DB PROMPT] ⚠️ prompt_pre_filter_rag TIDAK ditemukan di DB, pakai default")

        system_prompt = prompt_db or """
Anda adalah AI filter untuk pertanyaan terkait Pemerintah Kota Medan.

Petunjuk:
1. Balas HANYA dalam format JSON berikut:
   {"valid": true/false, "reason": "<penjelasan>", "clean_question": "<pertanyaan yang sudah dibersihkan>"}

2. Mark valid jika dan hanya jika pertanyaan membahas:
   - Dinas/instansi di bawah Pemko Medan
   - Layanan publik di Medan (KTP, SIM, pajak daerah, fasilitas kesehatan, pendidikan, dll)
   - Izin usaha/lingkungan/keramaian yang dikeluarkan Pemko Medan
   - Fasilitas umum milik Pemko Medan (taman, jalan, RSUD, dll)
   - Kebijakan atau program Pemerintah Kota Medan

3. Mark tidak valid jika:
   - Membahas daerah di luar Medan
   - Membahas figur publik non-pemerintah (selebriti, influencer, dll)
   - Membahas topik pribadi, gosip, atau tidak relevan
   - Pertanyaan tidak jelas/terlalu pendek

4. Bersihkan ejaan & tanda baca, jangan ubah maksud pertanyaan.
JANGAN BERIKAN PENJELASAN DI LUAR JSON.
"""

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": question.strip()}
            ],
            "temperature": 0.0,
            "top_p": 0.6
        }

        resp = requests.post(
            LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"]
        )

        # --- HANDLE RESPON
        if resp.status_code != 200:
            logger.error(f"[AI-FILTER] ❌ HTTP {resp.status_code}: {resp.text}")
            return {"valid": True, "reason": f"LLM error {resp.status_code}", "clean_question": question}

        try:
            raw_json = resp.json()
        except Exception:
            logger.error(f"[AI-FILTER] ❌ Response bukan JSON valid: {resp.text}")
            return {"valid": True, "reason": "Invalid JSON response from LLM", "clean_question": question}

        content = raw_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            logger.warning(f"[AI-FILTER] ⚠️ Tidak ada 'content' di response LLM: {raw_json}")
            return {"valid": True, "reason": "Empty response from LLM", "clean_question": question}

        parsed = _extract_json(content) or {
            "valid": True,
            "reason": "AI tidak mengembalikan JSON",
            "clean_question": question
        }

        logger.info(f"[AI-FILTER] ✅ Valid={parsed.get('valid')} | Reason={parsed.get('reason')}")
        return parsed

    except Exception as e:
        logger.exception(f"[AI-FILTER] ⚠️ Exception: {e}")
        return {"valid": True, "reason": f"Fallback error AI Filter: {e}", "clean_question": question}


# ==========================================================
# 🔹 AI CHECK RELEVANCE (POST)
# ==========================================================
def ai_check_relevance(user_q: str, rag_q: str):
    try:
        prompt_db = get_variable("prompt_relevance_rag")
        if prompt_db:
            logger.info(f"[DB PROMPT] ✅ prompt_relevance_rag ditemukan ({len(prompt_db)} chars)")
        else:
            logger.warning("[DB PROMPT] ⚠️ prompt_relevance_rag TIDAK ditemukan di DB, pakai default")

        system_prompt = prompt_db or """
Tugas Anda mengevaluasi apakah hasil pencarian RAG sesuai dengan maksud
pertanyaan pengguna.
Balas hanya JSON:
{"relevant": true/false, "reason": "...", "reformulated_question": "..."}

Kriteria:
✅ Relevan jika topik masih berkaitan dengan layanan publik, fasilitas, dokumen, kebijakan, atau prosedur administratif di Indonesia, termasuk yang dijalankan oleh instansi pusat maupun pemerintah daerah, selama konteksnya masih informatif bagi masyarakat Medan.
❌ Tidak relevan jika membahas kota lain, konteks umum vs spesifik, membahas hal pribadi, gosip, opini pribadi.
Jika tidak relevan, ubah pertanyaan jadi versi singkat berbentuk tanya
maks. 12 kata.
"""

        user_prompt = f"User: {user_q}\nRAG Result: {rag_q}"

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.1,
            "top_p": 0.5
        }

        resp = requests.post(
            LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"]
        )

        if resp.status_code != 200:
            logger.error(f"[AI-POST] ❌ HTTP {resp.status_code}: {resp.text}")
            return {"relevant": True, "reason": f"LLM error {resp.status_code}", "reformulated_question": ""}

        try:
            raw_json = resp.json()
        except Exception:
            logger.error(f"[AI-POST] ❌ Invalid JSON: {resp.text}")
            return {"relevant": True, "reason": "Invalid JSON response from LLM", "reformulated_question": ""}

        content = raw_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            logger.warning(f"[AI-POST] ⚠️ Tidak ada 'content' di response LLM: {raw_json}")
            return {"relevant": True, "reason": "Empty response from LLM", "reformulated_question": ""}

        parsed = _extract_json(content) or {"relevant": True, "reason": "-", "reformulated_question": ""}

        reform = parsed.get("reformulated_question", "").strip()
        if len(reform.split()) > 12:
            parsed["reformulated_question"] = " ".join(reform.split()[:12]) + "..."

        logger.info(f"[AI-POST] ✅ Relevant={parsed.get('relevant')} | Reason={parsed.get('reason')}")
        return parsed

    except Exception as e:
        logger.exception(f"[AI-POST] ⚠️ Exception: {e}")
        return {"relevant": True, "reason": f"AI relevance check failed: {e}", "reformulated_question": ""}
