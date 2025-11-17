import os
import fitz                       # PyMuPDF (super stable)
from paddleocr import PaddleOCR
from docx import Document
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
logger = logging.getLogger("ocr_utils")


# ============================================================
# 🔹 OCR Engine (aman & ringan)
# ============================================================
ocr_engine = PaddleOCR(
    lang="id",
    use_angle_cls=True,
    use_gpu=False,            # anti crash
    rec_batch_num=1           # kecil = stabil
)


# ============================================================
# 🔹 OCR helper untuk bytes gambar
# ============================================================
def _ocr_image_bytes(img_bytes: bytes) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp.flush()
        path = tmp.name

    try:
        res = ocr_engine.ocr(path)
        if res and len(res) > 0:
            return "\n".join([line[1][0] for line in res[0]])
        return ""
    except Exception as e:
        logger.warning(f"[OCR] Gagal OCR image: {e}")
        return ""
    finally:
        try:
            os.remove(path)
        except:
            pass


# ============================================================
# 🔹 Clean text halaman
# ============================================================
def _clean_page_text(t: str) -> str:
    import re
    if not t:
        return ""
    t = re.sub(r"^\s*\d{1,4}\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ============================================================
# 🔹 Render PDF → gambar → OCR (AMAN)
# ============================================================
def _safe_render_page(page, dpi=120):
    """
    Render halaman PDF menjadi PNG dengan kontrol ukuran & DPI aman.
    """
    try:
        pix = page.get_pixmap(dpi=dpi)

        # Hindari gambar terlalu besar (bitmap > 12MP)
        if pix.height * pix.width > 12_000_000:
            scale = (12_000_000 / (pix.height * pix.width)) ** 0.5
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)

        return pix.tobytes("png")

    except Exception as e:
        logger.error(f"[PDF] Render halaman gagal: {e}")
        return None


# ============================================================
# 🔹 Extract PDF (hybrid: text + OCR) — ANTI CRASH
# ============================================================
def _extract_pdf_pages(pdf_path: str, dpi: int = 120) -> Dict[int, str]:
    """
    PRODUCTION SAFE MODE:
    - Tidak pakai multithreading untuk render PDF (menghindari crash poppler/mupdf)
    - OCR bisa multi-thread
    """

    logger.info(f"[PDF] Membuka PDF: {pdf_path}")
    pages: Dict[int, str] = {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"PDF tidak dapat dibuka: {e}")

    # STEP 1 — Render halaman & ambil text native
    render_results = []   # (page_num, text_native, png_bytes)

    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]

        # Ambil text PDF native (kalau ada)
        text = page.get_text("text").strip()
        if text:
            render_results.append((page_num, text, None))
            continue

        # Render aman
        png_bytes = _safe_render_page(page, dpi=dpi)
        render_results.append((page_num, None, png_bytes))

    # STEP 2 — OCR halaman yang tidak punya text native (PAKE THREAD)
    def ocr_worker(data):
        page_num, native_text, img_bytes = data
        if native_text:
            return page_num, _clean_page_text(native_text)

        if img_bytes is None:
            return page_num, ""

        text = _ocr_image_bytes(img_bytes)
        return page_num, _clean_page_text(text)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(ocr_worker, item) for item in render_results]
        for fu in as_completed(futures):
            p, t = fu.result()
            pages[p] = t

    return dict(sorted(pages.items()))


# ============================================================
# 🔹 Extract text from file (PDF, DOCX, IMG)
# ============================================================
def extract_text_from_file(file_path: str, lang: str = "id", return_pages: bool = False):
    """
    Return:
      - dict halaman → teks (return_pages=True)
      - full string gabungan
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        pages = _extract_pdf_pages(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        res = ocr_engine.ocr(file_path)
        text = ""
        if res and len(res) > 0:
            text = "\n".join([line[1][0] for line in res[0]])
        pages = {1: _clean_page_text(text)}
    elif ext == ".docx":
        doc = Document(file_path)
        text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        pages = {1: _clean_page_text(text)}
    else:
        raise ValueError(f"Format file {ext} belum didukung.")

    if return_pages:
        return pages

    full_text = "\n\n".join(pages.values()).strip()
    return full_text
