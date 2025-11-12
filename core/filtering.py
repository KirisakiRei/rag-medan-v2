import json, re, requests, logging, traceback
from requests.exceptions import ConnectionError, Timeout
from core.utils import hard_filter_local
from core.db import get_variable
from config import CONFIG
from core.prompts import PROMPT_PRE_FILTER_USULAN, PROMPT_PRE_FILTER_RAG, PROMPT_RELEVANCE_RAG, PROMPT_RELEVANCE_USULAN

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

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
            logger.warning(f"\n[JSON PARSE] ❌ Tidak ditemukan JSON pada teks:\n{text[:100]}...\n{'-'*60}")
            return None
        return json.loads(m.group(0))
    except Exception as e:
        logger.exception(f"\n[JSON PARSE] ❌ Gagal parsing JSON: {e}\n{'-'*60}")
        return None

def ai_pre_filter(question: str):
    try:
        logger.info(f"\n{'='*60}\n[AI-FILTER] Memulai Pre Filter Pertanyaan\n{'-'*60}")
        hard = hard_filter_local(question)
        if not hard["valid"]:
            logger.info(f"[HARD FILTER] Ditolak | Reason: {hard['reason']}\n{'='*60}")
            return hard

        prompt_db = get_variable("prompt_pre_filter_rag")
        if prompt_db:
            logger.info(f"[DB PROMPT] prompt_pre_filter_rag ditemukan ({len(prompt_db)} chars)")
        else:
            logger.warning("[DB PROMPT] prompt_pre_filter_rag TIDAK ditemukan di DB, pakai default")

        system_prompt = prompt_db or PROMPT_PRE_FILTER_RAG

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": question.strip()}
            ],
            "temperature": 0.0,
            "top_p": 0.6
        }

        logger.info(f"[AI-FILTER] Mengirim request ke LLM...\n{'-'*60}")
        resp = requests.post(
            LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"]
        )

        if resp.status_code != 200:
            logger.error(f"[AI-FILTER] HTTP {resp.status_code}: {resp.text}\n{'='*60}")
            return {"valid": True, "reason": f"LLM error {resp.status_code}", "clean_question": question}

        try:
            raw_json = resp.json()
        except Exception:
            logger.error(f"[AI-FILTER] ❌ Response bukan JSON valid:\n{resp.text[:200]}\n{'='*60}")
            return {"valid": True, "reason": "Invalid JSON response from LLM", "clean_question": question}

        content = raw_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            logger.warning(f"[AI-FILTER] Tidak ada 'content' di response LLM: {raw_json}\n{'='*60}")
            return {"valid": True, "reason": "Empty response from LLM", "clean_question": question}

        parsed = _extract_json(content) or {
            "valid": True,
            "reason": "AI tidak mengembalikan JSON",
            "clean_question": question
        }

        logger.info(
            f"[AI-FILTER] Hasil Filter:\n"
            f"    Valid : {parsed.get('valid')}\n"
            f"    Reason: {parsed.get('reason')}\n"
            f"    Clean : {parsed.get('clean_question')}\n{'='*60}"
        )
        return parsed

    except (ConnectionError, Timeout) as e:
        logger.error(f"[AI-FILTER] Koneksi ke LLM gagal: {e}")
        if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
            logger.warning("[AI-FILTER] DNS Error: Tidak dapat resolve domain LLM (cek koneksi)")
        return {"valid": True, "reason": f"LLM connection error (fallback active): {e}", "clean_question": question}

    except Exception as e:
        logger.exception(f"[AI-FILTER] Exception: {e}\n{'='*60}")
        return {"valid": True, "reason": f"Fallback error AI Filter: {e}", "clean_question": question}

def ai_check_relevance(user_q: str, rag_q: str):
    try:
        logger.info(f"\n{'='*60}\n[AI-POST] Memulai Relevance Check\n{'-'*60}")
        prompt_db = get_variable("prompt_relevance_rag")
        if prompt_db:
            logger.info(f"[DB PROMPT] prompt_relevance_rag ditemukan ({len(prompt_db)} chars)")
        else:
            logger.warning("[DB PROMPT] prompt_relevance_rag TIDAK ditemukan di DB, pakai default")

        system_prompt = prompt_db or PROMPT_RELEVANCE_RAG

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

        logger.info(f"[AI-POST] Mengirim request ke LLM...\n{'-'*60}")
        resp = requests.post(
            LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"]
        )

        if resp.status_code != 200:
            logger.error(f"[AI-POST]  HTTP {resp.status_code}: {resp.text}\n{'='*60}")
            return {"relevant": True, "reason": f"LLM error {resp.status_code}", "reformulated_question": ""}

        try:
            raw_json = resp.json()
        except Exception:
            logger.error(f"[AI-POST] Invalid JSON:\n{resp.text[:200]}\n{'='*60}")
            return {"relevant": True, "reason": "Invalid JSON response from LLM", "reformulated_question": ""}

        content = raw_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            logger.warning(f"[AI-POST] Tidak ada 'content' di response LLM: {raw_json}\n{'='*60}")
            return {"relevant": True, "reason": "Empty response from LLM", "reformulated_question": ""}

        parsed = _extract_json(content)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"[AI-POST] Gagal parsing JSON dari konten:\n{content[:150]}...\n{'='*60}")
            parsed = {"relevant": True, "reason": "AI relevance check failed (invalid JSON)", "reformulated_question": ""}

        reform = (parsed.get("reformulated_question") or "").strip()
        if len(reform.split()) > 12:
            parsed["reformulated_question"] = " ".join(reform.split()[:12]) + "..."

        logger.info(
            f"[AI-POST] Hasil Relevance Check:\n"
            f"    Relevant: {parsed.get('relevant')}\n"
            f"    Reason  : {parsed.get('reason')}\n"
            f"    Reform  : {parsed.get('reformulated_question')}\n{'='*60}"
        )
        return parsed

    except (ConnectionError, Timeout) as e:
        logger.error(f"[AI-POST] Koneksi ke LLM gagal: {e}\n{'='*60}")
        return {"relevant": True, "reason": f"LLM connection error: {e}", "reformulated_question": ""}

    except Exception as e:
        logger.exception(f"[AI-POST] Exception (final fallback): {e}\n{'='*60}")
        return {"relevant": True, "reason": f"AI relevance check failed: {e}", "reformulated_question": ""}


def ai_pre_filter_usulan(user_input: str):
    try:
        logger.info("\n" + "=" * 60)
        logger.info("[AI-PRE FILTER-USULAN] Memulai reformulasi usulan")
        logger.info(f"Input user : {user_input}")
        logger.info("-" * 60)

        prompt_db = get_variable("prompt_pre_filter_usulan")
        system_prompt = prompt_db or PROMPT_PRE_FILTER_USULAN

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_input.strip()}
            ],
            "temperature": 0.2,
            "top_p": 0.6
        }

        if "session" not in globals():
            globals()["session"] = requests.Session()

        resp = requests.post(LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"])
        if resp.status_code != 200:
            logger.warning(f"[AI-REFORM-USULAN] LLM HTTP {resp.status_code}")
            return {"clean_request": user_input}

        raw = resp.json()
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_json(content)
        clean = (parsed or {}).get("clean_request", user_input)

        logger.info(f"[AI-REFORM-USULAN] ✅ Hasil Reformulasi: {clean}\n" + "=" * 60)
        return {"clean_request": clean}

    except Exception as e:
        logger.error(f"[AI-REFORM-USULAN] Error: {e}")
        return {"clean_request": user_input}


def ai_relevance_usulan(user_input: str, top_result: str):
    try:
        logger.info("\n" + "=" * 60)
        logger.info("[AI-TOPIC-USULAN] Memulai pengecekan relevansi topik")
        logger.info(f"Pertanyaan User : {user_input}")
        logger.info(f"Topik Hasil RAG : {top_result}")
        logger.info("-" * 60)

        prompt_db = get_variable("prompt_relevance_usulan")
        system_prompt = prompt_db or PROMPT_RELEVANCE_USULAN

        user_prompt = f"""
            Pertanyaan pengguna: "{user_input}"
            Topik hasil RAG: "{top_result}"
            """

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.0,
            "top_p": 0.5
        }

        resp = requests.post(
            LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"]
        )

        if resp.status_code != 200:
            logger.warning(f"[AI-RELEVANCE-USULAN] ❌ LLM HTTP {resp.status_code}")
            return {"relevant": True, "reason": "LLM error, skip relevance check"}

        raw = resp.json()
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_json(content)

        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"[AI-RELEVANCE-USULAN] ⚠️ Gagal parsing JSON:\n{content[:200]}")
            parsed = {"relevant": True, "reason": "Fallback: invalid JSON"}

        relevant = parsed.get("relevant", True)
        reason = parsed.get("reason", "-")

        logger.info(
            f"[AI-TOPIC-USULAN] ✅ Hasil Cek Relevansi:\n"
            f"    Relevant : {relevant}\n"
            f"    Reason   : {reason}\n"
            + "=" * 60
        )

        return parsed

    except Exception as e:
        logger.error(f"[AI-TOPIC-USULAN] ⚠️ Error: {e}")
        return {"relevant": True, "reason": f"Fallback (error: {e})"}
