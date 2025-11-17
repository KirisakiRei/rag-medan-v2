import os
from typing import Dict, Tuple
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from docx import Document
import logging
import tempfile

logger = logging.getLogger("ocr_utils")

# ============================================================
# 🔹 Inisialisasi engine OCR (versi lama PaddleOCR 3.3.1)
#    ❗ Hanya parameter yang didukung: lang, use_angle_cls, show_log
# ============================================================
ocr_engine = PaddleOCR(
    lang="id",
    use_angle_cls=True,
    show_log=False,
)

# ============================================================
# 🔹 Utility OCR untuk gambar (single-thread)
# ============================================================
def _ocr_image_bytes(img_bytes: bytes) -> str:
    """OCR dari bytes gambar (utility internal, tanpa multi-thread)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp.flush()
            tmp_path = tmp.name

        res = ocr_engine.ocr(tmp_path)
        if res and len(res) > 0:
            return "\n".join([line[1][0] for line in res[0]])
        return ""
    except Exception as e:
        logger.warning(f"[OCR] Gagal OCR image: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ============================================================
# 🔹 Utility pembersih teks halaman
# ============================================================
def _clean_page_text(t: str) -> str:
    """Bersihkan header/footer sederhana, spasi dobel, nomor halaman, dsb."""
    import re
    if not t:
        return ""
    # hapus nomor halaman yang berdiri sendiri, misal "12"
    t = re.sub(r"^\s*\d{1,4}\s*$", "", t, flags=re.MULTILINE)
    # rapikan spasi
    t = re.sub(r"[ \t]+", " ", t)
    # gabungkan newline berlebih
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ============================================================
# 🔹 Ekstraksi PDF per halaman (HYBRID, TANPA THREAD)
# ============================================================
def _extract_pdf_pages(pdf_path: str, dpi: int = 180) -> Dict[int, str]:
    """
    Ekstrak teks PDF per halaman dengan hybrid:
    - Jika halaman punya teks vector → get_text
    - Jika halaman image-scan → render → PaddleOCR
    Diproses SEKUENSIAL (no multi-thread) demi stabilitas.
    """
    pages: Dict[int, str] = {}

    logger.info(f"[PDF] Membuka PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_num in range(1, total_pages + 1):
        page = doc[page_num - 1]

        # 1) Coba pakai teks bawaan PDF dulu
        try:
            text = page.get_text("text") or ""
        except Exception as e:
            logger.warning(f"[PDF] Gagal get_text di halaman {page_num}: {e}")
            text = ""

        text = text.strip()
        if text:
            pages[page_num] = _clean_page_text(text)
            continue

        # 2) Kalau tidak ada teks → OCR dari bitmap
        try:
            pix = page.get_pixmap(dpi=dpi)
            bytes_ = pix.tobytes("png")
            ocr_text = _ocr_image_bytes(bytes_)
            pages[page_num] = _clean_page_text(ocr_text)
        except Exception as e:
            logger.warning(f"[PDF] Gagal render/OCR halaman {page_num}: {e}")
            pages[page_num] = ""

    return dict(sorted(pages.items(), key=lambda x: x[0]))


# ============================================================
# 🔹 Fungsi utama — Ekstraksi teks dari file
# ============================================================
def extract_text_from_file(file_path: str, lang: str = "id", return_pages: bool = False):
    """
    Ekstraksi teks dari file PDF, DOCX, atau gambar.
    Jika return_pages=True → kembalikan dict {page_number: text}
    Jika False → return string gabungan seluruh halaman.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages = {}

    if ext == ".pdf":
        pages = _extract_pdf_pages(file_path, dpi=180)

    elif ext in [".jpg", ".jpeg", ".png"]:
        try:
            res = ocr_engine.ocr(file_path)
        except Exception as e:
            logger.warning(f"[OCR] Gagal OCR image file {file_path}: {e}")
            res = None

        text = ""
        if res and len(res) > 0:
            text = "\n".join([line[1][0] for line in res[0]])
        pages[1] = _clean_page_text(text)

    elif ext == ".docx":
        doc = Document(file_path)
        text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        pages[1] = _clean_page_text(text)

    else:
        raise ValueError(f"Format file {ext} belum didukung untuk OCR.")

    if return_pages:
        return dict(sorted(pages.items(), key=lambda x: x[0]))

    full_text = "\n\n".join(pages.values()).strip()
    return full_text
