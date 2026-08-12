"""Conservative routing of photographed pages and diagrams.

Routing is deliberately evidence based.  It uses inexpensive Pillow
projections and, when available, OpenCV contours and line geometry; it never
claims a diagram kind when the strongest candidates are too close.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps


class ContentKind(str, Enum):
    DOCUMENT = "document"
    FLOWCHART = "flowchart"
    MINDMAP = "mindmap"
    ARCHITECTURE = "architecture"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class RouteDecision:
    """A routing result with the evidence needed for a human confirmation."""

    kind: ContentKind
    confidence: float
    evidence: Mapping[str, Any]
    confirmation_required: bool
    requested_kind: ContentKind | None = None

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, ContentKind) else ContentKind(self.kind)
        object.__setattr__(self, "kind", kind)
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("route confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if self.requested_kind is not None and not isinstance(self.requested_kind, ContentKind):
            object.__setattr__(self, "requested_kind", ContentKind(self.requested_kind))
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def route(self) -> ContentKind:
        return self.kind

    @property
    def content_kind(self) -> ContentKind:
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
            "confirmation_required": self.confirmation_required,
            "requested_kind": self.requested_kind.value if self.requested_kind else None,
        }


def _requested_kind(value: str | ContentKind | None) -> ContentKind | None:
    if value is None:
        return None
    if isinstance(value, ContentKind):
        return value
    candidate = str(value).strip().lower()
    if candidate == "auto":
        return None
    return ContentKind(candidate)


def _load_gray(image: Path | str | Image.Image) -> tuple[Image.Image, str | None]:
    if isinstance(image, Image.Image):
        return ImageOps.grayscale(ImageOps.exif_transpose(image.copy())), None
    path = Path(image).expanduser().resolve()
    with Image.open(path) as opened:
        return ImageOps.grayscale(ImageOps.exif_transpose(opened)), str(path)


def _runs(values: Any, threshold: float, minimum: int = 2) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if float(value) >= threshold:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    if start is not None and len(values) - start >= minimum:
        runs.append((start, len(values)))
    return runs


def _segment_distance(point: tuple[float, float], first: Any, second: Any) -> float:
    px, py = point
    x1, y1 = float(first[0]), float(first[1])
    x2, y2 = float(second[0]), float(second[1])
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist(point, (x1, y1))
    projection = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    return math.dist(point, (x1 + projection * dx, y1 + projection * dy))


def _line_intersection(first: Any, second: Any) -> tuple[float, float] | None:
    x1, y1, x2, y2 = map(float, first)
    x3, y3, x4, y4 = map(float, second)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    determinant_a = x1 * y2 - y1 * x2
    determinant_b = x3 * y4 - y3 * x4
    px = (determinant_a * (x3 - x4) - (x1 - x2) * determinant_b) / denominator
    py = (determinant_a * (y3 - y4) - (y1 - y2) * determinant_b) / denominator
    if _segment_distance((px, py), (x1, y1), (x2, y2)) > 0.08 * max(abs(x2 - x1), abs(y2 - y1), 1.0):
        return None
    if _segment_distance((px, py), (x3, y3), (x4, y4)) > 0.08 * max(abs(x4 - x3), abs(y4 - y3), 1.0):
        return None
    return px, py


def _junction_features(lines: list[tuple[float, float, float, float]], width: int, height: int) -> dict[str, float | int]:
    if not lines:
        return {
            "branch_junction_count": 0,
            "branch_max_lines": 0,
            "radial_concentration": 0.0,
            "radial_direction_diversity": 0.0,
        }
    points: list[tuple[float, float]] = []
    for line in lines:
        points.extend(((line[0], line[1]), (line[2], line[3])))
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1 :]:
            intersection = _line_intersection(first, second)
            if intersection is not None:
                points.append(intersection)
    tolerance = max(8.0, 0.035 * min(width, height))
    clusters: list[list[tuple[float, float]]] = []
    for point in points:
        for cluster in clusters:
            center = (
                sum(item[0] for item in cluster) / len(cluster),
                sum(item[1] for item in cluster) / len(cluster),
            )
            if math.dist(point, center) <= tolerance:
                cluster.append(point)
                break
        else:
            clusters.append([point])

    junctions: list[tuple[int, float]] = []
    for cluster in clusters:
        center = (
            sum(item[0] for item in cluster) / len(cluster),
            sum(item[1] for item in cluster) / len(cluster),
        )
        near_lines = [
            line
            for line in lines
            if _segment_distance(center, (line[0], line[1]), (line[2], line[3])) <= tolerance
        ]
        if len(near_lines) < 3:
            continue
        angles = []
        for line in near_lines:
            angle = math.degrees(math.atan2(line[3] - line[1], line[2] - line[0])) % 180.0
            angles.append(angle)
        bins = {int(angle // 18) for angle in angles}
        junctions.append((len(near_lines), len(bins) / max(1, len(near_lines))))
    if not junctions:
        return {
            "branch_junction_count": 0,
            "branch_max_lines": 0,
            "radial_concentration": 0.0,
            "radial_direction_diversity": 0.0,
        }
    max_lines, diversity = max(junctions, key=lambda item: item[0])
    return {
        "branch_junction_count": len(junctions),
        "branch_max_lines": max_lines,
        "radial_concentration": max_lines / max(1, len(lines)),
        "radial_direction_diversity": diversity,
    }


def _features_with_opencv(gray: Image.Image) -> tuple[dict[str, Any], str]:
    import cv2
    import numpy as np

    values = np.asarray(gray, dtype=np.uint8)
    height, width = values.shape[:2]
    scale = min(1.0, 1600.0 / max(width, height))
    if scale < 1.0:
        values = cv2.resize(values, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = values.shape[:2]
    _, binary = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    ink_ratio = float(np.count_nonzero(binary) / max(1, binary.size))
    row_ratio = np.count_nonzero(binary, axis=1) / max(1, width)
    text_rows = _runs(row_ratio, max(0.006, min(0.05, float(np.percentile(row_ratio, 70)))), 2)
    col_ratio = np.count_nonzero(binary, axis=0) / max(1, height)
    edges = cv2.Canny(values, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_count = 0
    triangle_count = 0
    circle_count = 0
    rectangle_centers: list[tuple[float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.002 * width * height:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            moments = cv2.moments(approximation)
            center = (
                float(moments["m10"] / moments["m00"]) if moments["m00"] else 0.0,
                float(moments["m01"] / moments["m00"]) if moments["m00"] else 0.0,
            )
            if not any(
                math.dist(center, (old_center_x, old_center_y)) < 0.04 * min(width, height)
                and abs(area - old_area) / max(area, old_area) < 0.45
                for old_center_x, old_center_y, old_area in rectangle_centers
            ):
                rectangle_centers.append((center[0], center[1], area))
                rectangle_count += 1
        elif len(approximation) == 3:
            triangle_count += 1
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.35 and len(approximation) >= 6:
            circle_count += 1

    minimum_line = max(18, int(0.09 * min(width, height)))
    hough = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(15, int(0.035 * min(width, height))),
        minLineLength=minimum_line,
        maxLineGap=max(6, int(0.025 * min(width, height))),
    )
    lines: list[tuple[float, float, float, float]] = []
    horizontal = 0
    vertical = 0
    line_lengths: list[float] = []
    if hough is not None:
        for item in hough.reshape(-1, 4):
            line = tuple(float(value) for value in item)
            length = math.dist((line[0], line[1]), (line[2], line[3]))
            if length < minimum_line:
                continue
            lines.append(line)
            line_lengths.append(length)
            angle = math.degrees(math.atan2(line[3] - line[1], line[2] - line[0])) % 180.0
            if angle <= 15 or angle >= 165:
                horizontal += 1
            if 75 <= angle <= 105:
                vertical += 1
    line_count = len(lines)
    long_line_count = sum(length >= 0.16 * max(width, height) for length in line_lengths)
    rectilinear_ratio = (horizontal + vertical) / max(1, line_count)
    junctions = _junction_features(lines, width, height)
    components_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    component_count = int(sum(size >= 8 for size in component_sizes))
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    return {
        "width": int(width),
        "height": int(height),
        "ink_ratio": ink_ratio,
        "edge_density": edge_density,
        "text_band_count": len(text_rows),
        "text_band_density": len(text_rows) / max(1, height),
        "vertical_band_count": len(_runs(col_ratio, max(0.01, float(np.percentile(col_ratio, 80))), 2)),
        "component_count": component_count,
        "rectangle_count": rectangle_count,
        "closed_shape_count": rectangle_count + circle_count,
        "triangle_count": triangle_count,
        "circle_count": circle_count,
        "line_count": line_count,
        "long_line_count": long_line_count,
        "horizontal_line_count": horizontal,
        "vertical_line_count": vertical,
        "rectilinear_ratio": rectilinear_ratio,
        **junctions,
    }, "opencv_otsu_contours_hough_projections"


def _features_with_pillow(gray: Image.Image) -> tuple[dict[str, Any], str]:
    values = list(gray.getdata())
    width, height = gray.size
    dark = [value < 180 for value in values]
    ink_ratio = sum(dark) / max(1, len(dark))
    rows = [
        sum(dark[row * width : (row + 1) * width]) / max(1, width)
        for row in range(height)
    ]
    return {
        "width": width,
        "height": height,
        "ink_ratio": ink_ratio,
        "edge_density": 0.0,
        "text_band_count": len(_runs(rows, 0.01, 2)),
        "text_band_density": len(_runs(rows, 0.01, 2)) / max(1, height),
        "vertical_band_count": 0,
        "component_count": 0,
        "rectangle_count": 0,
        "closed_shape_count": 0,
        "triangle_count": 0,
        "circle_count": 0,
        "line_count": 0,
        "long_line_count": 0,
        "horizontal_line_count": 0,
        "vertical_line_count": 0,
        "rectilinear_ratio": 0.0,
        "branch_junction_count": 0,
        "branch_max_lines": 0,
        "radial_concentration": 0.0,
        "radial_direction_diversity": 0.0,
    }, "pillow_projection_baseline"


def _scores(features: Mapping[str, Any]) -> dict[ContentKind, float]:
    shape = min(1.0, float(features["closed_shape_count"]) / 5.0)
    boxes = min(1.0, float(features["rectangle_count"]) / 4.0)
    lines = min(1.0, float(features["long_line_count"]) / 8.0)
    connectors = min(1.0, float(features["line_count"]) / 8.0)
    arrows = min(1.0, float(features["triangle_count"]) / 3.0)
    branch = min(1.0, float(features["branch_max_lines"]) / 5.0) * min(
        1.0, float(features["radial_direction_diversity"]) * 2.5
    )
    junctions = min(1.0, float(features["branch_junction_count"]) / 4.0)
    circles = min(1.0, float(features["circle_count"]) / 2.0)
    rectilinear = float(features["rectilinear_ratio"])
    text_bands = min(1.0, float(features["text_band_count"]) / 8.0)
    components = min(1.0, float(features["component_count"]) / 28.0)
    ink = float(features["ink_ratio"])
    text_signal = text_bands * (0.55 + 0.45 * components)
    document = 0.08 + 0.56 * text_signal + 0.16 * max(0.0, 1.0 - shape) + 0.10 * max(0.0, 1.0 - lines) + 0.10 * min(1.0, ink * 8)
    flowchart = 0.56 * boxes + 0.20 * connectors + 0.20 * arrows + 0.04 * max(0.0, 1.0 - branch)
    mindmap = branch * (
        0.35
        + 0.45 * circles
        + 0.20 * min(1.0, float(features["radial_concentration"]) * 2.0)
    ) + 0.10 * float(features["radial_direction_diversity"])
    architecture_shape_gate = min(
        1.0, max(0.0, float(features["rectangle_count"]) - 2.0) / 3.0
    )
    architecture = (
        0.43 * boxes * architecture_shape_gate
        + 0.30 * lines * rectilinear
        + 0.20 * junctions
        + 0.07 * max(0.0, 1.0 - arrows)
    )
    return {
        ContentKind.DOCUMENT: min(1.0, document),
        ContentKind.FLOWCHART: min(1.0, flowchart),
        ContentKind.MINDMAP: min(1.0, mindmap),
        ContentKind.ARCHITECTURE: min(1.0, architecture),
    }


def route_content(
    image: Path | str | Image.Image,
    requested_kind: str | ContentKind | None = None,
) -> RouteDecision:
    """Route a page to document/diagram kind, or require confirmation.

    ``requested_kind=None`` and ``requested_kind="auto"`` invoke the same
    conservative detector.  Any explicit supported kind is honoured with
    confidence ``1.0`` and recorded as an operator request.
    """

    requested = _requested_kind(requested_kind)
    if requested is not None:
        return RouteDecision(
            kind=requested,
            confidence=1.0,
            confirmation_required=requested is ContentKind.UNCERTAIN,
            requested_kind=requested,
            evidence={
                "algorithm": "explicit_operator_request",
                "requested_kind": requested.value,
                "confidence_basis": "operator_selected",
            },
        )

    gray, source = _load_gray(image)
    try:
        features, algorithm = _features_with_opencv(gray)
    except ImportError:
        features, algorithm = _features_with_pillow(gray)
    scores = _scores(features)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_kind, best_score = ordered[0]
    second_score = ordered[1][1]
    margin = best_score - second_score
    # Diagram photos are intentionally held to a higher bar than a simple
    # printed-page guess.  Ties and weak evidence remain uncertain.
    minimum_score = 0.58
    minimum_margin = 0.13
    document_score = scores[ContentKind.DOCUMENT]
    document_exception = (
        document_score >= 0.45
        and float(features["text_band_count"]) >= 3
        and float(features["closed_shape_count"]) == 0
        and float(features["radial_direction_diversity"]) < 0.30
    )
    flowchart_exception = (
        scores[ContentKind.FLOWCHART] >= 0.58
        and 2 <= float(features["rectangle_count"]) <= 3
        and float(features["circle_count"]) == 0
        and scores[ContentKind.ARCHITECTURE] < 0.65
        and scores[ContentKind.MINDMAP] < 0.60
    )
    certain = (
        (best_score >= minimum_score and margin >= minimum_margin)
        or document_exception
        or flowchart_exception
    )
    if document_exception:
        confidence = min(0.99, 0.56 + 0.40 * document_score)
        kind = ContentKind.DOCUMENT
    elif flowchart_exception:
        confidence = min(0.99, 0.56 + 0.40 * scores[ContentKind.FLOWCHART])
        kind = ContentKind.FLOWCHART
    elif certain:
        confidence = min(0.99, 0.56 + 0.40 * best_score + 0.08 * min(1.0, margin))
        kind = best_kind
    else:
        kind = ContentKind.UNCERTAIN
        confidence = min(0.59, max(0.20, 0.42 + 0.25 * best_score))
    selected_score = scores.get(kind, best_score)
    evidence = {
        "source": source,
        "algorithm": algorithm,
        "features": dict(features),
        "scores": {kind.value: round(score, 6) for kind, score in scores.items()},
        "selected_kind": kind.value,
        "selected_score": round(float(selected_score), 6),
        "runner_up_score": round(float(second_score), 6),
        "score_margin": round(float(margin), 6),
        "thresholds": {
            "minimum_score": minimum_score,
            "minimum_margin": minimum_margin,
        },
        "decision": "selected" if certain else "ambiguous_or_weak",
        "decision_reason": (
            "document_text_distribution"
            if document_exception
            else "flowchart_shape_connector_pattern"
            if flowchart_exception
            else "score_margin_and_thresholds"
            if certain
            else "score_margin_or_threshold_below_gate"
        ),
    }
    return RouteDecision(
        kind=kind,
        confidence=confidence,
        confirmation_required=not certain,
        evidence=evidence,
    )


def classify_content(
    image: Path | str | Image.Image,
    requested_kind: str | ContentKind | None = None,
) -> RouteDecision:
    """Alias for :func:`route_content`."""

    return route_content(image, requested_kind=requested_kind)


def route(
    image: Path | str | Image.Image,
    requested_kind: str | ContentKind | None = None,
) -> RouteDecision:
    """Short alias for :func:`route_content`."""

    return route_content(image, requested_kind=requested_kind)


__all__ = [
    "ContentKind",
    "RouteDecision",
    "classify_content",
    "route",
    "route_content",
]
