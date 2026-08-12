"""Conservative OpenCV candidate extraction for photographed diagrams.

This is a real local detector for standard boxes, diamonds, ellipses and direct
connectors.  It deliberately marks uncertain labels and arrow directions for
human review instead of presenting heuristics as semantic certainty.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageOps

from .diagram import (
    DiagramEdge,
    DiagramGraph,
    DiagramNode,
    EdgeGeometry,
    NodeKind,
    Point,
    Rect,
    deterministic_id,
)


class DiagramRecognitionError(RuntimeError):
    """Raised when local diagram candidate extraction cannot run."""


def _rect_distance(point: tuple[float, float], rect: Rect) -> float:
    x, y = point
    dx = max(rect.x - x, 0.0, x - (rect.x + rect.width))
    dy = max(rect.y - y, 0.0, y - (rect.y + rect.height))
    return (dx * dx + dy * dy) ** 0.5


def _ocr_label(image: Image.Image, languages: str) -> tuple[str, float | None]:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        return "", None
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "node.png"
        image.save(path, format="PNG")
        completed = subprocess.run(
            [
                tesseract,
                str(path),
                "stdout",
                "-l",
                languages,
                "--psm",
                "6",
                "tsv",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        return "", None
    lines = completed.stdout.splitlines()
    if not lines:
        return "", None
    header = lines[0].split("\t")
    if "text" not in header or "conf" not in header:
        return "", None
    text_index, confidence_index = header.index("text"), header.index("conf")
    words: list[str] = []
    confidences: list[float] = []
    for line in lines[1:]:
        columns = line.split("\t")
        if len(columns) <= max(text_index, confidence_index):
            continue
        word = columns[text_index].strip()
        if not word:
            continue
        words.append(word)
        confidence = float(columns[confidence_index])
        if confidence >= 0:
            confidences.append(confidence / 100)
    return " ".join(words), (sum(confidences) / len(confidences) if confidences else None)


def _crossed_out(gray: object, rect: Rect) -> bool:
    import cv2
    import numpy as np

    values = gray[
        int(rect.y) : int(rect.y + rect.height),
        int(rect.x) : int(rect.x + rect.width),
    ]
    edges = cv2.Canny(values, 50, 150)
    minimum = int(0.65 * min(rect.width, rect.height))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20, minLineLength=max(15, minimum), maxLineGap=8
    )
    if lines is None:
        return False
    positive = negative = False
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        angle = (float(math.degrees(math.atan2(y1 - y0, x1 - x0))) + 180) % 180
        positive |= 25 <= angle <= 70
        negative |= 110 <= angle <= 155
    return positive and negative


def recognize_diagram(
    image_path: Path,
    *,
    source_path: Path | None = None,
    content_kind: str = "flowchart",
    languages: str = "chi_sim+eng",
) -> DiagramGraph:
    """Extract review-required semantic candidates from one diagram photo."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise DiagramRecognitionError("diagram recognition requires OpenCV") from error

    image = Path(image_path).expanduser().resolve()
    source = Path(source_path).expanduser().resolve() if source_path else image
    with Image.open(image) as opened:
        oriented = ImageOps.exif_transpose(opened).convert("RGB")
    rgb = np.asarray(oriented)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    edges = cv2.Canny(blurred, 45, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[Rect, NodeKind, float]] = []
    image_area = float(width * height)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if not 0.0025 * image_area <= area <= 0.32 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        x, y, box_width, box_height = cv2.boundingRect(approximation)
        if box_width < 45 or box_height < 28:
            continue
        rect = Rect(float(x), float(y), float(box_width), float(box_height))
        if any(
            abs(rect.center.x - old.center.x) < 0.04 * width
            and abs(rect.center.y - old.center.y) < 0.04 * height
            and abs(rect.width * rect.height - old.width * old.height)
            / max(rect.width * rect.height, old.width * old.height)
            < 0.45
            for old, _, _ in candidates
        ):
            continue
        circularity = 4 * math.pi * area / max(perimeter * perimeter, 1.0)
        if len(approximation) == 4:
            points = approximation.reshape(4, 2)
            top_index = int(points[:, 1].argmin())
            top = points[top_index]
            diamond = abs(float(top[0]) - (x + box_width / 2)) < box_width * 0.25
            kind = NodeKind.DECISION if diamond and box_width / box_height < 3 else NodeKind.PROCESS
            confidence = 0.82
        elif len(approximation) >= 6 and circularity >= 0.45:
            kind = NodeKind.START_END
            confidence = 0.72
        else:
            continue
        candidates.append((rect, kind, confidence))

    nodes: list[DiagramNode] = []
    for index, (rect, kind, confidence) in enumerate(
        sorted(candidates, key=lambda item: (item[0].y, item[0].x))
    ):
        padding = 5
        crop = oriented.crop(
            (
                max(0, int(rect.x) + padding),
                max(0, int(rect.y) + padding),
                min(width, int(rect.x + rect.width) - padding),
                min(height, int(rect.y + rect.height) - padding),
            )
        )
        label, label_confidence = _ocr_label(crop, languages)
        flags: list[str] = []
        if not label:
            label = f"未识别节点 {index + 1}"
            flags.append("missing_label")
        elif label_confidence is None or label_confidence < 0.72:
            flags.append("uncertain_label")
        latex = label if re.search(r"[=+*/^_]|\\[A-Za-z]+", label) else None
        if latex is not None:
            flags.append("latex_review_required")
        crossed_out = _crossed_out(gray, rect)
        nodes.append(
            DiagramNode(
                id=deterministic_id("node", index, label),
                label=label,
                geometry=rect,
                kind=kind,
                latex=latex,
                confidence=min(confidence, label_confidence or confidence),
                provenance={
                    "source": str(source),
                    "bbox": rect.to_dict(),
                    "shape_algorithm": "opencv_contour_polygon",
                    "label_engine": "tesseract",
                    "label_confidence": label_confidence,
                },
                excluded=crossed_out,
                crossed_out=crossed_out,
                review_flags=tuple(flags),
            )
        )

    if not nodes:
        raise DiagramRecognitionError(
            "no standard diagram nodes were detected; retake the photo or select document"
        )

    hough = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=max(20, int(min(width, height) * 0.025)),
        minLineLength=max(35, int(min(width, height) * 0.06)),
        maxLineGap=max(12, int(min(width, height) * 0.025)),
    )
    edge_pairs: dict[tuple[str, str], DiagramEdge] = {}
    threshold = 0.075 * min(width, height)
    if hough is not None:
        for line_index, (x0, y0, x1, y1) in enumerate(hough.reshape(-1, 4)):
            start = min(nodes, key=lambda node: _rect_distance((x0, y0), node.geometry))
            end = min(nodes, key=lambda node: _rect_distance((x1, y1), node.geometry))
            if start.id == end.id:
                continue
            if _rect_distance((x0, y0), start.geometry) > threshold:
                continue
            if _rect_distance((x1, y1), end.geometry) > threshold:
                continue
            source_node, target_node = sorted((start, end), key=lambda node: (node.y, node.x))
            pair = (source_node.id, target_node.id)
            if pair in edge_pairs:
                continue
            edge_pairs[pair] = DiagramEdge(
                id=deterministic_id("edge", line_index, "->".join(pair)),
                source=source_node.id,
                target=target_node.id,
                geometry=EdgeGeometry(
                    (source_node.geometry.center, target_node.geometry.center)
                ),
                confidence=0.55,
                provenance={
                    "source": str(source),
                    "line": [int(x0), int(y0), int(x1), int(y1)],
                    "algorithm": "opencv_hough_endpoint_assignment",
                },
                review_flags=("uncertain_direction",),
            )

    connected = {edge.source for edge in edge_pairs.values()} | {
        edge.target for edge in edge_pairs.values()
    }
    nodes = [
        replace(
            node,
            review_flags=node.review_flags + (("isolated_node",) if node.id not in connected else ()),
        )
        for node in nodes
    ]
    return DiagramGraph(
        nodes=tuple(nodes),
        edges=tuple(edge_pairs.values()),
        metadata={
            "content_kind": content_kind,
            "recognizer": "opencv_standard_shapes_v1",
            "source": str(source),
            "accuracy": "unverified_real_handwriting",
        },
    )


__all__ = ["DiagramRecognitionError", "recognize_diagram"]
