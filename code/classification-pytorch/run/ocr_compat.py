from typing import Any, List, Tuple

import numpy as np


def create_paddle_ocr(lang: str = "ch", use_gpu: bool = False, show_log: bool = False):
    """Create a PaddleOCR engine with compatibility for PaddleOCR 2.x and 3.x."""
    # PaddleOCR 2.x imports albumentations, which imports torch lazily. On some
    # Windows environments, importing torch first avoids DLL lookup failures.
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    from paddleocr import PaddleOCR

    legacy_kwargs = {
        "use_angle_cls": True,
        "lang": lang,
        "use_gpu": use_gpu,
        "show_log": show_log,
    }
    try:
        return PaddleOCR(**legacy_kwargs)
    except (TypeError, ValueError):
        modern_kwargs = {
            "lang": lang,
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "device": "gpu" if use_gpu else "cpu",
        }
        try:
            return PaddleOCR(**modern_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR could not be initialized. For the reproducible package, "
                "install the pinned OCR dependencies from requirements.txt "
                "(paddleocr>=2.7,<3.0 and paddlepaddle>=2.6,<3.0)."
            ) from exc


def _is_legacy_line(line: Any) -> bool:
    """Return whether one OCR line already follows the PaddleOCR 2.x format."""
    if not isinstance(line, (list, tuple)) or not line:
        return False
    first = line[0]
    return isinstance(first, (list, tuple)) and len(first) == 2


def _normalize_modern_result(result: Any) -> List[List[Tuple[Any, Tuple[str, float]]]]:
    """Convert PaddleOCR 3.x result dictionaries into the 2.x line format."""
    if not isinstance(result, list):
        result = [result]

    normalized: List[Tuple[Any, Tuple[str, float]]] = []
    for page in result:
        data = page
        if hasattr(page, "json") and callable(page.json):
            try:
                data = page.json
            except Exception:
                data = page
        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        boxes = data.get("rec_polys") or data.get("dt_polys") or data.get("rec_boxes") or []

        for idx, text in enumerate(texts):
            if not text:
                continue
            score = float(scores[idx]) if idx < len(scores) and scores[idx] is not None else 0.0
            if idx < len(boxes):
                box = boxes[idx]
            else:
                box = np.array([[0, idx], [1, idx], [1, idx + 1], [0, idx + 1]], dtype=np.float32)
            normalized.append((box, (str(text), score)))

    return [normalized]


def ocr_detect_text(ocr_engine, image) -> List[List[Tuple[Any, Tuple[str, float]]]]:
    """Return OCR detections in PaddleOCR 2.x line format."""
    try:
        result = ocr_engine.ocr(image, cls=True)
    except (TypeError, ValueError):
        try:
            result = ocr_engine.ocr(image)
        except (TypeError, ValueError, AttributeError):
            result = ocr_engine.predict(image)

    if not result:
        return []

    if isinstance(result, list) and result and _is_legacy_line(result[0]):
        return result
    if (
        isinstance(result, list)
        and result
        and isinstance(result[0], (list, tuple))
        and (not result[0] or _is_legacy_line(result[0]))
    ):
        return result
    return _normalize_modern_result(result)


def ocr_text(ocr_engine, image) -> str:
    """Run OCR and concatenate detected text in reading order."""
    items = []
    for line in ocr_detect_text(ocr_engine, image):
        for box, (text, _score) in line:
            if not text:
                continue
            box_np = np.array(box, dtype=np.float32)
            cx = float(np.mean(box_np[:, 0]))
            cy = float(np.mean(box_np[:, 1]))
            items.append((cy, cx, str(text).strip()))
    items.sort(key=lambda x: (x[0], x[1]))
    return "".join([text for _, _, text in items]).strip()
