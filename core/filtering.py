import json, re, requests, logging, traceback
from requests.exceptions import ConnectionError, Timeout
from core.utils import hard_filter_local
from core.db import get_variable
from config import CONFIG
from core.prompts import PROMPT_PRE_FILTER_USULAN, PROMPT_PRE_FILTER_RAG, PROMPT_RELEVANCE_RAG, PROMPT_RELEVANCE_USULAN

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

LLM_BASE_URL = CONFIG["llm"]["base_url"]
LLM_API_KEY = CONFIG["llm"]["api_key"]
LLM_MODEL = CONFIG["llm"]["model"]
LLM_PROVIDER = CONFIG["llm"].get("provider", "gemini")


def _call_gemini_llm(system_prompt: str, user_message: str, temperature: float = 0.0, max_tokens: int = 256):
    """
    Helper function untuk memanggil Gemini API.
    
    Args:
        system_prompt: Instruksi sistem/prompt untuk AI
        user_message: Pesan/pertanyaan user
        temperature: Suhu generasi (0.0 - 1.0)
        max_tokens: Maksimal token output
        
    Returns:
        str: Response text dari Gemini, atau None jika error
    """
    try:
        # Gemini API endpoint
        url = f"{LLM_BASE_URL}/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
        
        # Gemini request format
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt.strip()},
                        {"text": user_message.strip()}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": 1,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, headers=headers, json=payload, timeout=CONFIG["llm"]["timeout_sec"])
        
        if response.status_code != 200:
            logger.error(f"[GEMINI] HTTP {response.status_code}: {response.text}")
            return None
            
        response_data = response.json()
        
        # Parse Gemini response structure
        candidates = response_data.get("candidates", [])
        if not candidates:
            logger.warning(f"[GEMINI] No candidates in response: {response_data}")
            return None
            
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        if not parts:
            logger.warning(f"[GEMINI] No parts in response: {response_data}")
            return None
            
        text_response = parts[0].get("text", "").strip()
        return text_response
        
    except Exception as e:
        logger.error(f"[GEMINI] Error calling API: {e}")
        return None

def _extract_json(text: str):
    if not text:
        return None
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            logger.warning(f"\n[JSON PARSE] ❌ Tidak ditemukan JSON pada teks:\n{text[:100]}...\n{'-'*60}")
            return None
        return json.loads(json_match.group(0))
    except Exception as e:
        logger.exception(f"\n[JSON PARSE] ❌ Gagal parsing JSON: {e}\n{'-'*60}")
        return None

def ai_pre_filter(question: str):
    try:
        logger.info(f"\n{'='*60}\n[AI-FILTER] Memulai Pre Filter Pertanyaan\n{'-'*60}")
        hard_filter_result = hard_filter_local(question)
        if not hard_filter_result["valid"]:
            logger.info(f"[HARD FILTER] Ditolak | Reason: {hard_filter_result['reason']}\n{'='*60}")
            return hard_filter_result

        prompt_from_db = get_variable("prompt_pre_filter_rag")
        if prompt_from_db:
            logger.info(f"[DB PROMPT] prompt_pre_filter_rag ditemukan ({len(prompt_from_db)} chars)")
        else:
            logger.warning("[DB PROMPT] prompt_pre_filter_rag TIDAK ditemukan di DB, pakai default")

        system_prompt = prompt_from_db or PROMPT_PRE_FILTER_RAG

        logger.info(f"[AI-FILTER] Mengirim request ke Gemini LLM...\n{'-'*60}")
        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=question,
            temperature=0.0,
            max_tokens=256
        )

        if not llm_content:
            logger.error(f"[AI-FILTER] Gemini API error atau empty response\n{'='*60}")
            return {"valid": True, "reason": "LLM error", "clean_question": question}

        parsed_result = _extract_json(llm_content) or {
            "valid": True,
            "reason": "AI tidak mengembalikan JSON",
            "clean_question": question
        }

        logger.info(
            f"[AI-FILTER] Hasil Filter:\n"
            f"    Valid : {parsed_result.get('valid')}\n"
            f"    Reason: {parsed_result.get('reason')}\n"
            f"    Clean : {parsed_result.get('clean_question')}\n{'='*60}"
        )
        return parsed_result

    except (ConnectionError, Timeout) as e:
        logger.error(f"[AI-FILTER] Koneksi ke LLM gagal: {e}")
        if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
            logger.warning("[AI-FILTER] DNS Error: Tidak dapat resolve domain LLM (cek koneksi)")
        return {"valid": True, "reason": f"LLM connection error (fallback active): {e}", "clean_question": question}

    except Exception as e:
        logger.exception(f"[AI-FILTER] Exception: {e}\n{'='*60}")
        return {"valid": True, "reason": f"Fallback error AI Filter: {e}", "clean_question": question}

def ai_check_relevance(user_question: str, rag_question: str):
    try:
        logger.info(f"\n{'='*60}\n[AI-POST] Memulai Relevance Check\n{'-'*60}")
        prompt_from_db = get_variable("prompt_relevance_rag")
        if prompt_from_db:
            logger.info(f"[DB PROMPT] prompt_relevance_rag ditemukan ({len(prompt_from_db)} chars)")
        else:
            logger.warning("[DB PROMPT] prompt_relevance_rag TIDAK ditemukan di DB, pakai default")

        system_prompt = prompt_from_db or PROMPT_RELEVANCE_RAG

        user_prompt = f"User: {user_question}\nRAG Result: {rag_question}"
        
        logger.info(f"[AI-POST] Mengirim request ke Gemini LLM...\n{'-'*60}")
        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.1,
            max_tokens=256
        )

        if not llm_content:
            logger.error(f"[AI-POST] Gemini API error atau empty response\n{'='*60}")
            return {"relevant": True, "reason": "LLM error", "reformulated_question": ""}

        parsed_result = _extract_json(llm_content)
        if not parsed_result or not isinstance(parsed_result, dict):
            logger.warning(f"[AI-POST] Gagal parsing JSON dari konten:\n{llm_content[:150]}...\n{'='*60}")
            parsed_result = {"relevant": True, "reason": "AI relevance check failed (invalid JSON)", "reformulated_question": ""}

        reformulated_text = (parsed_result.get("reformulated_question") or "").strip()
        if len(reformulated_text.split()) > 12:
            parsed_result["reformulated_question"] = " ".join(reformulated_text.split()[:12]) + "..."

        logger.info(
            f"[AI-POST] Hasil Relevance Check:\n"
            f"    Relevant: {parsed_result.get('relevant')}\n"
            f"    Reason  : {parsed_result.get('reason')}\n"
            f"    Reform  : {parsed_result.get('reformulated_question')}\n{'='*60}"
        )
        return parsed_result

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

        prompt_from_db = get_variable("prompt_pre_filter_usulan")
        system_prompt = prompt_from_db or PROMPT_PRE_FILTER_USULAN

        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=user_input,
            temperature=0.2,
            max_tokens=256
        )

        if not llm_content:
            logger.warning(f"[AI-REFORM-USULAN] Gemini API error atau empty response")
            return {"clean_request": user_input}

        parsed_result = _extract_json(llm_content)
        clean_request = (parsed_result or {}).get("clean_request", user_input)

        logger.info(f"[AI-REFORM-USULAN] ✅ Hasil Reformulasi: {clean_request}\n" + "=" * 60)
        return {"clean_request": clean_request}

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

        prompt_from_db = get_variable("prompt_relevance_usulan")
        system_prompt = prompt_from_db or PROMPT_RELEVANCE_USULAN

        user_prompt = f"""
            Pertanyaan pengguna: "{user_input}"
            Topik hasil RAG: "{top_result}"
            """

        llm_content = _call_gemini_llm(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.0,
            max_tokens=256
        )

        if not llm_content:
            logger.warning(f"[AI-RELEVANCE-USULAN] ❌ Gemini API error atau empty response")
            return {"relevant": True, "reason": "LLM error, skip relevance check"}

        parsed_result = _extract_json(llm_content)

        if not parsed_result or not isinstance(parsed_result, dict):
            logger.warning(f"[AI-RELEVANCE-USULAN] ⚠️ Gagal parsing JSON:\n{llm_content[:200]}")
            parsed_result = {"relevant": True, "reason": "Fallback: invalid JSON"}

        is_relevant = parsed_result.get("relevant", True)
        reason = parsed_result.get("reason", "-")

        logger.info(
            f"[AI-TOPIC-USULAN] ✅ Hasil Cek Relevansi:\n"
            f"    Relevant : {is_relevant}\n"
            f"    Reason   : {reason}\n"
            + "=" * 60
        )

        return parsed_result

    except Exception as e:
        logger.error(f"[AI-TOPIC-USULAN] ⚠️ Error: {e}")
        return {"relevant": True, "reason": f"Fallback (error: {e})"}
