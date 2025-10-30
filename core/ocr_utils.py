import os
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from docx import Document
from PIL import Image

ocr_engine = PaddleOCR(lang='id', use_angle_cls=True, show_log=False)

def extract_text_from_file(file_path: str, lang: str = "id"):
    """
    Ekstraksi teks per halaman dari file PDF, DOCX, atau gambar.
    Return dict: {page_number: text}
    """
    ext = os.path.splitext(file_path)[1].lower()
    results = {}

    if ext == ".pdf":
        # 🔹 OCR tiap halaman PDF
        pdf_doc = fitz.open(file_path)
        for page_num, page in enumerate(pdf_doc, start=1):
            images = page.get_images(full=True)
            text = ""

            if not images:
                # jika tidak ada gambar, ambil text langsung (PDF berbasis text)
                text = page.get_text("text")
            else:
                # render ke image lalu OCR
                pix = page.get_pixmap(dpi=200)
                img_path = f"/tmp/page_{page_num}.png"
                pix.save(img_path)
                ocr_result = ocr_engine.ocr(img_path, cls=True)
                if ocr_result:
                    text = "\n".join([line[1][0] for line in ocr_result[0]])
                os.remove(img_path)
            results[page_num] = text.strip()

    elif ext in [".jpg", ".jpeg", ".png"]:
        # 🔹 OCR single image
        ocr_result = ocr_engine.ocr(file_path, cls=True)
        if ocr_result:
            text = "\n".join([line[1][0] for line in ocr_result[0]])
            results[1] = text.strip()

    elif ext == ".docx":
        # 🔹 Baca DOCX
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        results[1] = text.strip()

    else:
        raise ValueError(f"Format file {ext} belum didukung.")

    return results
