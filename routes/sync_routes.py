# Optional Flask Blueprint for /api/sync mirroring app.py behavior.
from flask import Blueprint, jsonify
sync_bp = Blueprint("sync_bp", __name__)

@sync_bp.route("/api/sync", methods=["POST"])
def sync_endpoint():
    return jsonify({"status":"disabled","message":"Blueprint disabled by default. Use app.py's built-in route."}), 200
