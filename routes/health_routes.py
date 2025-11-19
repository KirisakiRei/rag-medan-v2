from flask import Blueprint, jsonify
import logging

health_bp = Blueprint("health_bp", __name__)
logger = logging.getLogger("app")

@health_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint untuk monitoring status sistem."""
    try:
        # Import model dan qdrant dari app context
        from app import model, qdrant
        
        # Check embedding model
        try:
            _ = model.encode("health check").tolist()
            model_status = True
        except Exception as e:
            logger.error(f"[HEALTH] Model error: {e}", exc_info=True)
            model_status = False

        # Check Qdrant connection
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
        
    except Exception as e:
        logger.error(f"[HEALTH] Unexpected error: {e}", exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
