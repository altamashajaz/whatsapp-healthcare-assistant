from __future__ import annotations

from dataclasses import dataclass
import io

import fitz
import pdfplumber
import pytesseract
from PIL import Image


@dataclass
class ExtractionResult:
    doc_type: str
    raw_text: str
    ocr_confidence: float = 0.0


def _ocr_with_confidence(image: Image.Image) -> tuple[str, float]:
    """Run OCR and return extracted text plus average OCR confidence."""
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
    )

    confidences = []
    words = []

    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            continue

        if confidence >= 0:
            words.append(text)
            confidences.append(confidence)

    extracted_text = " ".join(words).strip()

    if not confidences:
        return extracted_text, 0.0

    return extracted_text, sum(confidences) / len(confidences)


def extract_document(file_bytes: bytes, mime_type: str) -> ExtractionResult:
    """
    Extract local OCR/PDF text.

    Image understanding itself is handled by OpenAI Vision.
    OCR is retained as supporting text extraction and confidence metadata.
    """
    try:
        if mime_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_bytes))

            text, confidence = _ocr_with_confidence(image)

            return ExtractionResult(
                doc_type="image",
                raw_text=text,
                ocr_confidence=confidence,
            )

        if mime_type == "application/pdf":
            parts: list[str] = []

            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""

                        if page_text.strip():
                            parts.append(page_text)

            except Exception:
                pass

            # If the PDF is scanned, render pages and OCR them.
            if not parts:
                pdf = fitz.open(
                    stream=file_bytes,
                    filetype="pdf",
                )

                for page in pdf:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(1.5, 1.5)
                    )

                    image = Image.open(
                        io.BytesIO(
                            pix.tobytes("png")
                        )
                    )

                    text = pytesseract.image_to_string(
                        image
                    ).strip()

                    if text:
                        parts.append(text)

                pdf.close()

            return ExtractionResult(
                doc_type="pdf",
                raw_text="\n\n".join(parts).strip(),
                ocr_confidence=0.0,
            )

        return ExtractionResult(
            doc_type="unreadable",
            raw_text="",
            ocr_confidence=0.0,
        )

    except Exception:
        return ExtractionResult(
            doc_type="unreadable",
            raw_text="",
            ocr_confidence=0.0,
        )