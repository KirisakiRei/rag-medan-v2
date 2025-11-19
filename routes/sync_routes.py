from flask import Blueprint, request, jsonify
import logging, traceback
from qdrant_client.http import models

sync_bp = Blueprint("sync_bp", __name__)
logger = logging.getLogger("app")


def error_response(t, msg, detail=None, code=500):
    """Helper untuk membuat error response."""
    payload = {"status": "error", "error": {"type": t, "message": msg}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


@sync_bp.route("/api/sync", methods=["POST"])
def sync_data():
    """Endpoint untuk sinkronisasi data knowledge_bank."""
    try:
        # Import model dan qdrant dari app context
        from app import model, qdrant
        
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
                vector = model.encode("passage: " + item["question_rag_name"]).tolist()
                point_id = str(item["question_rag_id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": item["question_id"],
                        "answer_id": item["answer_id"],
                        "category_id": item["category_id"],
                        "question": item["question"],
                        "question_rag_name": item["question_rag_name"]
                    }
                })
            
            qdrant.upsert(collection_name="knowledge_bank", points=points)
            qdrant.create_payload_index(
                collection_name="knowledge_bank",
                field_name="question_rag_name",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                ))
            
            logger.info(f"[SYNC-DATA] Sinkronisasi {len(points)} data ke Knowledge Bank berhasil")
            return jsonify({
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil",
                "total_synced": len(points)
            })
        
        elif action == "add":
            point_id = str(content["question_rag_id"])
            vector = model.encode("passage: " + content["question_rag_name"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": content["question_id"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id"),
                        "question": content["question"],
                        "question_rag_name": content["question_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-DATA] Data berhasil ditambahkan ke Knowledge Bank: ID={point_id}")
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan", "id": point_id})
        
        elif action == "update":
            point_id = str(content["question_rag_id"])
            vector = model.encode("passage: " + content["question_rag_name"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": content["question_id"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id"),
                        "question": content["question"],
                        "question_rag_name": content["question_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-DATA] Data berhasil Diperbarui di Knowledge Bank: ID={point_id}")
            return jsonify({"status": "success", "message": "Data berhasil diperbarui"})
        
        elif action == "delete":
            point_id = str(content["question_rag_id"])
            qdrant.delete(
                collection_name="knowledge_bank",
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"[SYNC-DATA] Data dihapus : ID={point_id}")
            return jsonify({"status": "success", "message": "Data berhasil dihapus"})
        
        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][sync_data] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal saat sinkronisasi", detail=str(e))
