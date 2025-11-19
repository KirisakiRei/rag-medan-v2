# ============================================================
# Routes Package — Organized endpoint handlers
# ============================================================
# Flask Blueprints (for app.py):
from .health_routes import health_bp
from .search_routes import search_bp
from .sync_routes import sync_bp
from .usulan_routes import usulan_bp

# FastAPI Routers (for doc_app.py):
from .doc_sync_routes import doc_sync_router
from .doc_search_routes import doc_search_router

__all__ = [
    "health_bp",
    "search_bp",
    "sync_bp",
    "usulan_bp",
    "doc_sync_router",
    "doc_search_router",
]
