from flask import Flask
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging, sys, os
from config import CONFIG

# Import blueprints
from routes.health_routes import health_bp
from routes.search_routes import search_bp
from routes.sync_routes import sync_bp
from routes.usulan_routes import usulan_bp

app = Flask(__name__)

@app.route("/")
def home():
    return "Server Flask App is running!"

# ============================================================
# 🔹 Initialize Model & Qdrant (Global Variables)
# ============================================================
model = SentenceTransformer(CONFIG["embeddings"]["model_path"])
qdrant = QdrantClient(
    host=CONFIG["qdrant"]["host"],
    port=CONFIG["qdrant"]["port"]
)

# ============================================================
# 🔹 Setup Logging
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag-medan.log")
SUMMARY_FILE = os.path.join(LOG_DIR, "rag-summary.log")

MASTER_PID = os.getppid()  
CURRENT_PID = os.getpid()

if not logging.getLogger("app").handlers:
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    rag_summary_logger = logging.getLogger("rag.summary")
    rag_summary_logger.setLevel(logging.INFO)
    rag_summary_logger.propagate = False

    summary_handler = logging.FileHandler(SUMMARY_FILE, encoding="utf-8")
    summary_handler.setFormatter(logging.Formatter("%(asctime)s [INFO]: %(message)s"))
    rag_summary_logger.addHandler(summary_handler)

    # Suppress noisy libraries
    for noisy in ["werkzeug", "httpx", "httpcore", "qdrant_client", "urllib3"]:
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers.clear()
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = True

    logger.info(
        f"✅ Logging initialized (PID={CURRENT_PID}, Master={MASTER_PID}) — app log to rag-medan.log, summary log to rag-summary.log"
    )
else:
    logger = logging.getLogger("app")
    rag_summary_logger = logging.getLogger("rag.summary")


# ============================================================
# 🔹 Register Blueprints
# ============================================================
app.register_blueprint(health_bp)
app.register_blueprint(search_bp)
app.register_blueprint(sync_bp)
app.register_blueprint(usulan_bp)

logger.info("✅ All blueprints registered successfully")
logger.info("   - /health → health_routes.py")
logger.info("   - /api/search → search_routes.py")
logger.info("   - /api/sync → sync_routes.py")
logger.info("   - /api/sync-usulan & /api/search-usulan → usulan_routes.py")


# ============================================================
# 🚀 Run Server
# ============================================================
if __name__ == "__main__":
    app.run(
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        debug=False,
        use_reloader=False
    )
