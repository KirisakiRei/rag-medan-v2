import os
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from docx import Document
from PIL import Image

# Inisialisasi engine OCR (gunakan cache agar tidak reload tiap kali)
ocr_engine = PaddleOCR(lang='id', use_angle_cls=True)

def extract_text_from_file(file_path: str, lang: str = "id", return_pages: bool = False):
    """
    Ekstraksi teks dari file PDF, DOCX, atau gambar.
    Jika return_pages=True, maka return (full_text, total_pages)
    Jika False, hanya return full_text saja.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages = {}  # simpan hasil OCR per halaman

    if ext == ".pdf":
        # 🔹 OCR tiap halaman PDF
        pdf_doc = fitz.open(file_path)
        for page_num, page in enumerate(pdf_doc, start=1):
            text = ""
            images = page.get_images(full=True)

            if not images:
                # jika PDF berbasis teks, ambil langsung
                text = page.get_text("text")
            else:
                # render halaman ke image lalu OCR
                pix = page.get_pixmap(dpi=200)
                img_path = f"/tmp/page_{page_num}.png"
                pix.save(img_path)
                try:
                    ocr_result = ocr_engine.ocr(img_path)
                    if ocr_result and len(ocr_result) > 0:
                        # gabungkan semua baris teks dari halaman itu
                        text = "\n".join([line[1][0] for line in ocr_result[0]])
                finally:
                    if os.path.exists(img_path):
                        os.remove(img_path)

            pages[page_num] = text.strip()

    elif ext in [".jpg", ".jpeg", ".png"]:
        # 🔹 OCR single image
        ocr_result = ocr_engine.ocr(file_path)
        text = ""
        if ocr_result and len(ocr_result) > 0:
            text = "\n".join([line[1][0] for line in ocr_result[0]])
        pages[1] = text.strip()

    elif ext == ".docx":
        # 🔹 Baca DOCX langsung (tanpa OCR)
        doc = Document(file_path)
        text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        pages[1] = text.strip()

    else:
        raise ValueError(f"Format file {ext} belum didukung untuk OCR.")

    # 🔹 Gabungkan semua halaman
    full_text = "\n\n".join(pages.values()).strip()

    # 🔹 Return sesuai opsi
    if return_pages:
        return full_text, len(pages)
    return full_text
