"""
ocr.py
------
Handles: Medical Report Image -> OCR -> Extracted Text

We use Tesseract OCR (via pytesseract), a free, well-known, local OCR engine.
OCR = Optical Character Recognition: it looks at an image and reads the
text characters it can see, turning pixels into a text string.

NOTE: You must install the Tesseract program itself (not just the Python
package) on your computer. See README.md for install instructions.
"""

from PIL import Image
import pytesseract


def extract_text_from_image(image_file) -> str:
    """
    image_file: a file-like object (e.g. from Streamlit's file_uploader)
    Returns: extracted text as a string.
    """
    image = Image.open(image_file)

    # Simple preprocessing: convert to grayscale, which usually improves
    # OCR accuracy on scanned documents / photos of reports.
    image = image.convert("L")

    text = pytesseract.image_to_string(image)
    return text.strip()
