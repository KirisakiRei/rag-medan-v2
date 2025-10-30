# Optional Flask Blueprint for /api/search.
# Not auto-used by app.py to avoid breaking existing logic.
# If you want to use it, import and register blueprint in app.py.
from flask import Blueprint, request, jsonify
import time
from core.utils import normalize_text, clean_location_terms, keyword_overlap, detect_category
from core.filtering import ai_pre_filter, ai_check_relevance
from qdrant_client.http import models
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import CONFIG

search_bp = Blueprint("search_bp", __name__)

# NOTE: this replicates the current behavior in app.py but it's opt-in.
@search_bp.route("/api/search", methods=["POST"])
def search_endpoint():
    return jsonify({"status":"disabled","message":"Blueprint disabled by default. Use app.py's built-in route."}), 200
