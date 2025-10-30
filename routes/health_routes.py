from flask import Blueprint, jsonify
health_bp = Blueprint("health_bp", __name__)

@health_bp.route("/health", methods=["GET"])
def health_endpoint():
    return jsonify({"status":"ok","note":"Blueprint not registered by default."}), 200
