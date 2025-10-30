
import re, json, logging, requests

logger = logging.getLogger(__name__)

def ai_check_relevance(user_q: str, rag_q: str):
    try:
        url = "https://dekallm.cloudeka.ai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6FaPtqd1W5aj0z_-AbsKBA",
            "Content-Type": "application/json"
        }
        system_prompt = """
Tugas Anda mengevaluasi apakah hasil pencarian RAG sesuai dengan maksud
pertanyaan pengguna.
Balas hanya JSON:
{"relevant": true/false, "reason": "...", "reformulated_question": "..."}

Kriteria:
✅ Relevan jika topik sama (layanan publik, fasilitas, dokumen, kebijakan).
❌ Tidak relevan jika membahas jabatan/instansi berbeda,
   kota lain, atau konteks umum vs spesifik.
Jika tidak relevan, ubah pertanyaan jadi versi singkat berbentuk tanya
maks. 12 kata.
""".strip()

        user_prompt = f"User: {user_q}\nRAG Result: {rag_q}"
        payload = {
            "model": "meta/llama-4-maverick-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.1,
            "top_p": 0.5
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {
            "relevant": True, "reason": "-", "reformulated_question": ""
        }

        reform = parsed.get("reformulated_question", "").strip()
        if len(reform.split()) > 12:
            parsed["reformulated_question"] = " ".join(reform.split()[:12]) + "..."

        logger.info(f"[AI RELEVANCE] ✅ Relevant: {parsed['relevant']} | Reason: {parsed['reason']}")
        return parsed

    except Exception as e:
        logger.error(f"[AI-Post] {e}")
        return {"relevant": True, "reason": "AI relevance check failed", "reformulated_question": ""}
