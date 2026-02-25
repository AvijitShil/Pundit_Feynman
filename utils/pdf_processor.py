import base64
import fitz  # PyMuPDF


def process_pdf_to_base64(pdf_path: str, dpi: int = 150) -> list[str]:
    """
    Converts each page of a PDF into a base64-encoded JPEG string.
    Preserves full RGB color (important for color-coded graphs in papers).
    """
    try:
        doc = fitz.open(pdf_path)
        base64_images = []

        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("jpeg")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            base64_images.append(img_b64)

        doc.close()
        print(f"Extracted {len(base64_images)} pages at {dpi} DPI (color preserved)")
        return base64_images
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise e
