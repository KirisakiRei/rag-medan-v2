from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging, time, sys, traceback
import requests

from config import CONFIG
from core.utils import (
    detect_category,
    normalize_text,
    clean_location_terms,
    keyword_overlap,
)
from core.filtering import ai_pre_filter, ai_check_relevance


app = Flask(__name__)

@app.route("/")
def home():
    return "Server Flask App is running!"

model = SentenceTransformer(CONFIG["embeddings"]["model_path"])
qdrant = QdrantClient(
    host=CONFIG["qdrant"]["host"],
    port=CONFIG["qdrant"]["port"]
)


LOG_FILE = "./logs/rag-medan.log"
SUMMARY_FILE = "./logs/rag-summary.log"

root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8")
        ]
    )
else:
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(logging.StreamHandler(sys.stdout))
    root_logger.addHandler(logging.FileHandler(LOG_FILE, encoding="utf-8"))


logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
app.logger.handlers = logger.handlers
app.logger.propagate = False


rag_summary_logger = logging.getLogger("rag.summary")
rag_summary_logger.setLevel(logging.INFO)
summary_handler = logging.FileHandler(SUMMARY_FILE, encoding="utf-8")
summary_handler.setFormatter(logging.Formatter("%(asctime)s [INFO]: %(message)s"))
rag_summary_logger.addHandler(summary_handler)
rag_summary_logger.propagate = False

@app.route("/health", methods=["GET"])
def health_check():
    try:
        _ = model.encode("health check").tolist()
        model_status = True
    except Exception as e:
        logger.error(f"[HEALTH] Model error: {e}", exc_info=True)
        model_status = False

    try:
        qdrant.get_collections()
        qdrant_status = True
    except Exception as e:
        logger.error(f"[HEALTH] Qdrant error: {e}", exc_info=True)
        qdrant_status = False

    overall_status = model_status and qdrant_status
    return jsonify({
        "status": "healthy" if overall_status else "unhealthy",
        "components": {
            "flask": True,
            "embedding_model": model_status,
            "qdrant": qdrant_status
        }
    }), 200 if overall_status else 500

def error_response(t, msg, detail=None, code=500):
    payload = {"status": "error", "error": {"type": t, "message": msg}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code

@app.route("/api/search", methods=["POST"])
def search():
    try:
        t0 = time.time()
        data = request.json or {}
        user_q = (data.get("question") or "").strip()
        wa = data.get("wa_number", "unknown")

        if not user_q:
            return jsonify({"status": "error", "message": "Field 'question' wajib diisi"}), 400

        # ---------------- AI FILTER (PRE) ----------------
        t_pre = time.time()
        pre = ai_pre_filter(user_q)
        t_pre_time = time.time() - t_pre
        logger.debug(f"[AI-FILTER] Done in {t_pre_time:.3f}s | Result: {pre}")

        if not pre.get("valid", True):
            total_time = time.time() - t0
            return jsonify({
                "status": "low_confidence",
                "message": pre.get("reason", "Pertanyaan tidak relevan"),
                "data": {
                    "similar_questions": [],
                    "metadata": {
                        "wa_number": wa,
                        "original_question": user_q,
                        "final_question": "-",
                        "category": "-",
                        "ai_reason": pre.get("reason", "-"),
                        "ai_reformulated": "-",
                        "final_score_top": "-"
                    }
                },
                "timing": {
                    "ai_domain_sec": round(t_pre_time, 3),
                    "ai_relevance_sec": 0.0,
                    "embedding_sec": 0.0,
                    "qdrant_sec": 0.0,
                    "total_sec": round(total_time, 3)
                }
            }), 200

        # ---------------- EMBEDDING ----------------
        question = normalize_text(clean_location_terms(pre.get("clean_question", user_q)))
        category = detect_category(question)
        cat_id = category["id"] if category else None

        t_emb = time.time()
        qvec = model.encode("query: " + question).tolist()
        emb_time = time.time() - t_emb

        # ---------------- QDRANT SEARCH ----------------
        t_qd = time.time()
        filt = models.Filter(must=[
            models.FieldCondition(
                key="category_id",
                match=models.MatchValue(value=cat_id)
            )
        ]) if cat_id else None

        dense_hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=qvec,
            limit=5,
            query_filter=filt
        )
        if len(dense_hits) < 3:
            dense_hits = qdrant.search(
                collection_name="knowledge_bank",
                query_vector=qvec,
                limit=5
            )
        qd_time = time.time() - t_qd

        # ---------------- AI RELEVANCE CHECK ----------------
        t_post = time.time()
        relevance = {}
        if dense_hits:
            relevance = ai_check_relevance(user_q, dense_hits[0].payload["question"])
        t_post_time = time.time() - t_post
        logger.debug(f"[AI-RELEVANCE] Done in {t_post_time:.3f}s | Result: {relevance}")

        # ---------------- SCORING ----------------
        results, rejected = [], []
        for h in dense_hits:
            dense = float(h.score)
            overlap = keyword_overlap(question, h.payload["question"])
            final_score = round((0.65 * dense) + (0.35 * overlap), 3)

            note, accepted = "-", False
            if dense >= 0.90:
                accepted, note = True, "auto_accepted_by_dense"
            elif 0.86 <= dense <= 0.89 and overlap >= 0.25:
                accepted, note = True, "accepted_by_overlap"

            item = {
                "question": h.payload["question"],
                "answer_id": h.payload.get("answer_id"),
                "category_id": h.payload.get("category_id"),
                "dense_score": dense,
                "overlap_score": overlap,
                "final_score": final_score,
                "note": note
            }
            (results if accepted else rejected).append(item)

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        rejected = sorted(rejected, key=lambda x: x["final_score"], reverse=True)

        total_time = time.time() - t0
        payload = {
            "status": "success" if results else "low_confidence",
            "message": "Hasil ditemukan" if results else "Tidak ada hasil cukup relevan",
            "data": {
                "similar_questions": results if results else rejected,
                "metadata": {
                    "wa_number": wa,
                    "original_question": user_q,
                    "final_question": question,
                    "category": (category["name"] if category else "Global"),
                    "ai_reason": relevance.get("reason", "-") if relevance else "-",
                    "ai_reformulated": relevance.get("reformulated_question", "-") if relevance else "-",
                    "final_score_top": (results[0]["final_score"] if results else "-")
                }
            },
            "timing": {
                "ai_domain_sec": round(t_pre_time, 3),
                "ai_relevance_sec": round(t_post_time, 3),
                "embedding_sec": round(emb_time, 3),
                "qdrant_sec": round(qd_time, 3),
                "total_sec": round(total_time, 3)
            }
        }
        # ---------------- DOC FALLBACK (non-breaking) ----------------
        try:
            if payload.get("status") == "low_confidence":
                t_doc = time.time()
                doc_resp = requests.post("http://127.0.0.1:5100/api/doc-search",
                                         json={"query": user_q, "limit": 5}, timeout=7)
                if doc_resp.status_code == 200:
                    doc_data = doc_resp.json()
                    payload["doc_fallback"] = doc_data
                    payload.setdefault("timing", {})
                    payload["timing"]["doc_fallback_sec"] = round(time.time() - t_doc, 3)
                    payload["data"]["metadata"]["source"] = "document_bank"
        except Exception as _doc_e:
            logger.warning(f"[DOC-FALLBACK] {_doc_e}")

        try:
            summary_lines = [
                "\n" + "=" * 60,
                "[RAG SESSION]",
                f"Pertanyaan User: {user_q}",
                f"Hasil Pre Filter: Valid={pre.get('valid')} | Reason: {pre.get('reason')}",
                f"Hasil Pencarian RAG: {len(dense_hits)} kandidat ditemukan",
                f"Hasil Post Proses (Relevance): {relevance.get('relevant', '-')} | Reason: {relevance.get('reason', '-')}"
            ]

            for idx, r in enumerate(results[:2], start=1):
                summary_lines.append(
                    f"{idx}. {r['question']} | Dense={r['dense_score']:.3f} | Overlap={r['overlap_score']:.3f} | Final={r['final_score']:.3f}"
                )

            answer_ids = [r.get("answer_id") for r in results[:2] if r.get("answer_id")]
            summary_lines.append(f"Answer ID yang dikembalikan: {answer_ids}")
            summary_lines.append(
                f"Total waktu proses: {total_time:.3f} detik "
                f"(Pre={t_pre_time:.3f}s | Emb={emb_time:.3f}s | Qdrant={qd_time:.3f}s | Post={t_post_time:.3f}s)"
            )
            summary_lines.append("=" * 60 + "\n")

            summary_block = "\n".join(summary_lines)
            rag_summary_logger.info(summary_block)
            logger.info(summary_block)
        except Exception as e:
            logger.warning(f"[LOGGING ERROR] Gagal mencetak ringkasan RAG: {e}")

        return jsonify(payload), 200

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][search] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal", detail=str(e))

@app.route("/api/sync", methods=["POST"])
def sync_data():
    try:
        data = request.json
        if not data or "action" not in data:
            return error_response("ValidationError", "Field 'action' wajib diisi", code=400)
        action = data["action"]
        content = data.get("content")

        if action == "bulk_sync":
            if not isinstance(content, list):
                return error_response("ValidationError", "Content harus berupa list", code=400)
            points = []
            for item in content:
                vector = model.encode("passage: " + item["question"]).tolist()
                point_id = str(item["id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": point_id,
                        "question": item["question"],
                        "answer_id": item["answer_id"],
                        "category_id": item.get("category_id")
                    }
                })
            qdrant.upsert(collection_name="knowledge_bank", points=points)
            qdrant.create_payload_index(
                collection_name="knowledge_bank",
                field_name="question",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                ))
            return jsonify({
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil",
                "total_synced": len(points)
            })
        elif action == "add":
            point_id = str(content["id"])
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": point_id,
                        "question": content["question"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id")
                    }
                }]
            )
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan", "id": point_id})
        elif action == "update":
            point_id = str(content["id"])
            vector = model.encode("passage: " + content["question"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "mysql_id": point_id,
                        "question": content["question"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id")
                    }
                }]
            )
            return jsonify({"status": "success", "message": "Data berhasil diperbarui"})
        elif action == "delete":
            point_id = str(content["id"])
            qdrant.delete(
                collection_name="knowledge_bank",
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True
            )
            return jsonify({"status": "success", "message": "Data berhasil dihapus"})
        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][sync_data] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal saat sinkronisasi", detail=str(e))


if __name__ == "__main__":
    app.run(host=CONFIG["api"]["host"], port=CONFIG["api"]["port"], debug=True)
