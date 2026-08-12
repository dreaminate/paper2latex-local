"""Local photo preparation for OCR and diagram recognition.

The source image is never written in place.  Pillow provides the baseline
orientation, grayscale, and contrast handling; when OpenCV is installed it
also supplies a conservative page-boundary detector and illumination
normalisation.  A crop is applied only when a large, plausible quadrilateral
has enough evidence to be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


class PhotoError(ValueError):
    """Raised when a photo cannot be prepared without changing its source."""


@dataclass(frozen=True)
class PhotoPreprocessResult:
    """A derived OCR image together with machine-readable evidence."""

    source: Path
    output: Path
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    rectified: bool
    corners: tuple[tuple[float, float], ...] | None
    transform_matrix: tuple[tuple[float, float, float], ...] | None
    algorithm: str
    quality: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def source_path(self) -> Path:
        return self.source

    @property
    def output_path(self) -> Path:
        return self.output

    @property
    def evidence(self) -> dict[str, Any]:
        """Return quality and transform evidence under one stable key."""

        return {
            "quality": self.quality,
            "provenance": self.provenance,
            "rectified": self.rectified,
            "corners": self.corners,
            "transform_matrix": self.transform_matrix,
            "algorithm": self.algorithm,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "source_size": list(self.source_size),
            "output_size": list(self.output_size),
            "rectified": self.rectified,
            "corners": [list(point) for point in self.corners]
            if self.corners is not None
            else None,
            "transform_matrix": [list(row) for row in self.transform_matrix]
            if self.transform_matrix is not None
            else None,
            "algorithm": self.algorithm,
            "quality": self.quality,
            "provenance": self.provenance,
        }


# A short alias is useful to callers that do not need to distinguish this
# result from a generic preprocessing result.
PhotoResult = PhotoPreprocessResult


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_corners(points: Any) -> list[tuple[float, float]]:
    values = [(float(point[0]), float(point[1])) for point in points]
    center_x = sum(point[0] for point in values) / 4
    center_y = sum(point[1] for point in values) / 4
    values.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    # The order from atan2 starts at the left-most/lower half depending on the
    # quadrilateral.  Rotate to the top-left and enforce clockwise winding.
    start = min(range(4), key=lambda index: values[index][0] + values[index][1])
    values = values[start:] + values[:start]
    cross = (
        (values[1][0] - values[0][0]) * (values[2][1] - values[1][1])
        - (values[1][1] - values[0][1]) * (values[2][0] - values[1][0])
    )
    if cross < 0:
        values = [values[0], values[3], values[2], values[1]]
    return values


def _quadrilateral_metrics(points: list[tuple[float, float]], width: int, height: int) -> dict[str, float]:
    sides = [
        math.dist(points[index], points[(index + 1) % 4])
        for index in range(4)
    ]
    area = abs(
        sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
    ) / 2
    long_side = max(sides)
    short_side = min(sides)
    side_balance = short_side / max(long_side, 1.0)
    area_ratio = area / max(1.0, float(width * height))
    border_margin = min(
        point[0] for point in points
    ) / max(1, width)
    border_margin = min(
        border_margin,
        min(point[1] for point in points) / max(1, height),
        (width - max(point[0] for point in points)) / max(1, width),
        (height - max(point[1] for point in points)) / max(1, height),
    )
    # A rectangle-like quadrilateral with a useful margin receives a high
    # score.  Border-touching rectangles are deliberately rejected below.
    confidence = min(
        1.0,
        0.35 + 0.45 * min(1.0, area_ratio / 0.75) + 0.2 * side_balance,
    )
    return {
        "area_ratio": float(area_ratio),
        "side_balance": float(side_balance),
        "border_margin": float(border_margin),
        "confidence": float(confidence),
    }


def _detect_page_boundary(image: Image.Image, confidence_threshold: float) -> dict[str, Any]:
    """Find a page quadrilateral without inventing a crop on weak evidence."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return {
            "state": "not_available",
            "algorithm": "pillow_no_opencv",
            "confidence": 0.0,
            "corners": None,
            "transform_matrix": None,
        }

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, 1600.0 / max(width, height))
    if scale < 1.0:
        sample = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        sample = rgb
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    sample_height, sample_width = gray.shape[:2]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        corners = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(corners) != 4 or not cv2.isContourConvex(corners):
            continue
        area = float(cv2.contourArea(corners))
        area_ratio = area / max(1.0, float(sample_width * sample_height))
        if area_ratio < 0.20 or area_ratio > 0.985:
            continue
        ordered = _ordered_corners(corners.reshape(4, 2))
        metrics = _quadrilateral_metrics(ordered, sample_width, sample_height)
        if metrics["side_balance"] < 0.18:
            continue
        # A candidate touching all four image edges is usually the image frame,
        # not a page boundary.  Require a visible margin on at least one side.
        if metrics["border_margin"] < 0.008 and area_ratio > 0.92:
            continue
        candidates.append(
            {
                "corners_sample": ordered,
                "metrics": metrics,
                "contour_area": area,
            }
        )

    if not candidates:
        return {
            "state": "not_confident",
            "algorithm": "opencv_canny_contour_quadrilateral",
            "confidence": 0.0,
            "corners": None,
            "transform_matrix": None,
            "candidate_count": 0,
        }

    # Prefer the largest plausible page boundary.  Confidence breaks ties but
    # must not cause a small inner box to replace the page itself.
    candidate = max(
        candidates,
        key=lambda item: (
            item["metrics"]["area_ratio"],
            item["metrics"]["confidence"],
        ),
    )
    metrics = candidate["metrics"]
    confidence = float(metrics["confidence"])
    corners = [
        (point[0] / scale, point[1] / scale)
        for point in candidate["corners_sample"]
    ]
    if confidence < confidence_threshold or metrics["border_margin"] < 0.012:
        return {
            "state": "not_confident",
            "algorithm": "opencv_canny_contour_quadrilateral",
            "confidence": confidence,
            "corners": corners,
            "transform_matrix": None,
            "candidate_count": len(candidates),
            "metrics": metrics,
        }

    top_left, top_right, bottom_right, bottom_left = corners
    target_width = max(
        1,
        int(round(max(math.dist(top_left, top_right), math.dist(bottom_left, bottom_right)))),
    )
    target_height = max(
        1,
        int(round(max(math.dist(top_left, bottom_left), math.dist(top_right, bottom_right)))),
    )
    matrix = cv2.getPerspectiveTransform(
        np.asarray(corners, dtype=np.float32),
        np.asarray(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype=np.float32,
        ),
    )
    return {
        "state": "rectify",
        "algorithm": "opencv_canny_contour_quadrilateral",
        "confidence": confidence,
        "corners": corners,
        "transform_matrix": matrix.tolist(),
        "target_size": (target_width, target_height),
        "candidate_count": len(candidates),
        "metrics": metrics,
    }


def _normalise_grayscale(image: Image.Image) -> tuple[Image.Image, dict[str, float], str]:
    """Flatten broad illumination gradients and improve OCR contrast."""

    gray = ImageOps.grayscale(image)
    try:
        import cv2
        import numpy as np
    except ImportError:
        enhanced = ImageEnhance.Contrast(ImageOps.autocontrast(gray, cutoff=1)).enhance(1.15)
        values = np.asarray(enhanced) if "np" in locals() else None
        if values is None:
            histogram = enhanced.histogram()
            total = max(1, sum(histogram))
            mean = sum(index * count for index, count in enumerate(histogram)) / total
            spread = 0.0
            return enhanced, {"luminance_mean": float(mean), "luminance_stddev": spread}, "pillow_grayscale_autocontrast"
        return enhanced, {
            "luminance_mean": float(values.mean()),
            "luminance_stddev": float(values.std()),
        }, "pillow_grayscale_autocontrast"

    values = np.asarray(gray, dtype=np.uint8)
    sigma = max(3.0, min(values.shape[:2]) / 28.0)
    background = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
    flattened = cv2.divide(values, np.maximum(background, 1), scale=220)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)
    normalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flattened)
    output = Image.fromarray(normalised, mode="L")
    return output, {
        "luminance_mean": float(normalised.mean()),
        "luminance_stddev": float(normalised.std()),
    }, "opencv_illumination_division_clahe"


def preprocess_photo(
    source: Path,
    output: Path,
    *,
    confidence_threshold: float = 0.72,
) -> PhotoPreprocessResult:
    """Prepare one photo and write a derived grayscale PNG.

    ``source`` is read only.  The output directory is created when needed and
    the original is left untouched, even when the requested output suffix is
    the same as the source suffix.
    """

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise PhotoError("output must differ from source so the source stays untouched")
    if not source_path.is_file():
        raise PhotoError(f"source photo is not a file: {source_path}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise PhotoError("confidence_threshold must be between 0 and 1")

    source_hash = _sha256(source_path)
    try:
        with Image.open(source_path) as opened:
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
            source_size = oriented.size
            exif_transposed = opened.size != oriented.size or bool(
                opened.getexif().get(274)
            )
            boundary = _detect_page_boundary(oriented, confidence_threshold)
            working = oriented
            transform_matrix: tuple[tuple[float, float, float], ...] | None = None
            corners: tuple[tuple[float, float], ...] | None = None
            rectified = boundary["state"] == "rectify"
            if rectified:
                import cv2
                import numpy as np

                rgb = np.asarray(oriented.convert("RGB"))
                target_width, target_height = boundary["target_size"]
                warped = cv2.warpPerspective(
                    rgb,
                    np.asarray(boundary["transform_matrix"], dtype=np.float64),
                    (target_width, target_height),
                    borderMode=cv2.BORDER_REPLICATE,
                )
                working = Image.fromarray(warped, mode="RGB")
                corners = tuple(tuple(point) for point in boundary["corners"])
                transform_matrix = tuple(
                    tuple(float(value) for value in row)
                    for row in boundary["transform_matrix"]
                )
            cleaned, luminance, illumination_algorithm = _normalise_grayscale(working)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned.save(output_path, format="PNG", optimize=True)
    except (OSError, ValueError) as error:
        raise PhotoError(f"cannot preprocess photo {source_path}: {error}") from error

    with Image.open(output_path) as saved:
        output_size = saved.size
        histogram = saved.histogram()
        total = max(1, sum(histogram))
        dark_ratio = sum(histogram[:35]) / total
        bright_ratio = sum(histogram[245:]) / total

    page_boundary_quality = {
        key: value
        for key, value in boundary.items()
        if key in {"state", "confidence", "algorithm", "candidate_count", "metrics"}
    }
    page_boundary_quality.update(
        {
            "corners": [list(point) for point in corners] if corners else None,
            "transform_matrix": [list(row) for row in transform_matrix]
            if transform_matrix
            else None,
        }
    )
    quality = {
        "state": "passed" if boundary["state"] == "rectify" else "warning",
        "source_size": list(source_size),
        "output_size": list(output_size),
        "metrics": {
            **luminance,
            "dark_ratio": float(dark_ratio),
            "bright_ratio": float(bright_ratio),
        },
        "page_boundary": page_boundary_quality,
    }
    algorithm = f"exif_transpose+{boundary['algorithm']}+{illumination_algorithm}"
    provenance = {
        "source": str(source_path),
        "source_sha256": source_hash,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "exif_transposed": exif_transposed,
        "algorithm": algorithm,
        "page_boundary": {
            "corners": [list(point) for point in corners] if corners else None,
            "transform_matrix": [list(row) for row in transform_matrix]
            if transform_matrix
            else None,
            "confidence": float(boundary["confidence"]),
            "state": boundary["state"],
        },
    }
    return PhotoPreprocessResult(
        source=source_path,
        output=output_path,
        source_size=source_size,
        output_size=output_size,
        rectified=rectified,
        corners=corners,
        transform_matrix=transform_matrix,
        algorithm=algorithm,
        quality=quality,
        provenance=provenance,
    )


def preprocess_image(source: Path, output: Path, **kwargs: Any) -> PhotoPreprocessResult:
    """Alias for callers that use image-oriented naming."""

    return preprocess_photo(source, output, **kwargs)


def clean_photo(source: Path, output: Path, **kwargs: Any) -> PhotoPreprocessResult:
    """Alias retained for pipeline adapters that call the derived image clean."""

    return preprocess_photo(source, output, **kwargs)


__all__ = [
    "PhotoError",
    "PhotoPreprocessResult",
    "PhotoResult",
    "clean_photo",
    "preprocess_image",
    "preprocess_photo",
]
