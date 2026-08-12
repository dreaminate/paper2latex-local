"""End-to-end local conversion pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Protocol

from PIL import Image, ImageFilter, ImageOps, ImageStat
from pypdf import PdfReader

from .render import RenderError, render_markdown_pdf, render_searchable_pdf
from .photo import PhotoError, preprocess_photo
from .task import MAX_PAGES, MODES, SUPPORTED_SUFFIXES, TaskError

CONVERSION_MODES = MODES | {"auto"}
CONVERSION_SUFFIXES = SUPPORTED_SUFFIXES | {".pdf"}


@dataclass(frozen=True)
class FormulaRegion:
    """A formula bounding box in normalized coordinates or source pixels."""

    bbox: tuple[float, float, float, float]
    latex: str | None = None


@dataclass(frozen=True)
class PageRecognition:
    """Recognition result returned by an OCR engine for one cleaned page."""

    markdown: str
    confidence: float | None = None
    formula_count: int = 0
    warnings: tuple[str, ...] = ()
    suggested_mode: str | None = None
    mode_confidence: float | None = None
    formula_regions: tuple[FormulaRegion, ...] = ()


class RecognitionEngine(Protocol):
    """Public adapter seam for local OCR engines."""

    engine_id: str
    engine_version: str
    formula_capable: bool

    def recognize(
        self, pages: list[Path], work_dir: Path
    ) -> dict[Path, PageRecognition]: ...


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    status: str
    page_count: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SourcePage:
    source: Path
    source_page: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(inputs: Iterable[Path]) -> list[Path]:
    provided = [Path(item).expanduser().resolve() for item in inputs]
    if not provided:
        raise TaskError("at least one input image or PDF is required")

    def natural_key(path: Path) -> list[object]:
        return [
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        ]

    sources: list[Path] = []
    for item in provided:
        if item.is_dir():
            children = sorted(
                (
                    child.resolve()
                    for child in item.iterdir()
                    if child.is_file() and child.suffix.lower() in CONVERSION_SUFFIXES
                ),
                key=natural_key,
            )
            if not children:
                raise TaskError(
                    f"input directory contains no supported images or PDFs: {item}"
                )
            sources.extend(children)
        elif item.is_file():
            sources.append(item)
        else:
            raise TaskError(f"input does not exist: {item}")

    if len(set(sources)) != len(sources):
        raise TaskError("duplicate input paths are not allowed")
    for source in sources:
        if source.suffix.lower() not in CONVERSION_SUFFIXES:
            allowed = ", ".join(sorted(CONVERSION_SUFFIXES))
            raise TaskError(
                f"unsupported input suffix {source.suffix!r}; expected {allowed}"
            )
    return sources


def resolve_source_files(inputs: Iterable[Path]) -> list[Path]:
    """Resolve files and naturally ordered directory children without converting."""

    return _source_files(inputs)


def _expand_source_pages(sources: list[Path]) -> list[SourcePage]:
    pages: list[SourcePage] = []
    for source in sources:
        if source.suffix.lower() != ".pdf":
            pages.append(SourcePage(source=source, source_page=None))
            continue
        try:
            page_count = len(PdfReader(source).pages)
        except Exception as error:
            raise TaskError(f"cannot read PDF {source}: {error}") from error
        if page_count == 0:
            raise TaskError(f"PDF has no pages: {source}")
        pages.extend(
            SourcePage(source=source, source_page=page_number)
            for page_number in range(1, page_count + 1)
        )
    if len(pages) > MAX_PAGES:
        raise TaskError(f"a conversion accepts at most {MAX_PAGES} pages")
    return pages


def _image_quality(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            grayscale = ImageOps.grayscale(image)
            grayscale.thumbnail((1200, 1200))
            luminance = ImageStat.Stat(grayscale)
            edge_stat = ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES))
            histogram = grayscale.histogram()
    except (OSError, ValueError) as error:
        raise TaskError(f"cannot decode image {path}: {error}") from error

    pixel_count = max(1, sum(histogram))
    bright_ratio = sum(histogram[245:]) / pixel_count
    dark_ratio = sum(histogram[:35]) / pixel_count
    luminance_stddev = float(luminance.stddev[0])
    edge_stddev = float(edge_stat.stddev[0])
    issues: list[str] = []
    warnings: list[str] = []
    if min(width, height) < 1000:
        issues.append("resolution_below_1000px")
    if luminance_stddev < 3.0 or edge_stddev < 2.0:
        issues.append("low_detail_or_blur")
    if dark_ratio > 0.6:
        issues.append("image_too_dark")
    if bright_ratio > 0.995:
        issues.append("overexposed_or_blank")
    elif bright_ratio > 0.45:
        warnings.append("possible_glare_or_overexposure")
    if dark_ratio > 0.2:
        warnings.append("possible_heavy_shadow")

    perspective = _perspective_analysis(path)
    if perspective.get("state") == "warning":
        warnings.append("possible_perspective_skew")
    return {
        "state": "passed" if not issues and not warnings else "warning",
        "width": width,
        "height": height,
        "metrics": {
            "luminance_stddev": round(luminance_stddev, 3),
            "edge_stddev": round(edge_stddev, 3),
            "bright_ratio": round(bright_ratio, 6),
            "dark_ratio": round(dark_ratio, 6),
        },
        "perspective": perspective,
        "issues": issues,
        "warnings": warnings,
    }


def _perspective_analysis(path: Path) -> dict[str, object]:
    try:
        import cv2
    except ImportError:
        return {"state": "not_available", "reason": "opencv_not_installed"}

    image = cv2.imread(str(path))
    if image is None:
        return {"state": "not_evaluated", "reason": "opencv_decode_failed"}
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(image.shape[0] * image.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) != 4 or cv2.contourArea(corners) < page_area * 0.2:
            continue
        points = [point[0].tolist() for point in corners]
        top_left = min(points, key=lambda point: point[0] + point[1])
        bottom_right = max(points, key=lambda point: point[0] + point[1])
        top_right = max(points, key=lambda point: point[0] - point[1])
        bottom_left = min(points, key=lambda point: point[0] - point[1])
        horizontal = max(
            abs(top_right[1] - top_left[1]),
            abs(bottom_right[1] - bottom_left[1]),
        ) / max(1, image.shape[0])
        vertical = max(
            abs(bottom_left[0] - top_left[0]),
            abs(bottom_right[0] - top_right[0]),
        ) / max(1, image.shape[1])
        skew = max(horizontal, vertical)
        return {
            "state": "warning" if skew > 0.08 else "passed",
            "normalized_skew": round(float(skew), 4),
            "page_boundary_detected": True,
        }
    return {"state": "uncertain", "page_boundary_detected": False}


def _prepare_image(source: Path, destination: Path) -> dict[str, object]:
    try:
        prepared = preprocess_photo(source, destination)
    except (OSError, PhotoError, ValueError) as error:
        raise TaskError(f"cannot decode image {source}: {error}") from error
    quality = _image_quality(destination)
    quality["preprocessing"] = prepared.to_dict()
    return quality


def _rasterize_pdf_page(source: Path, page_number: int, destination: Path) -> None:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise TaskError("PDF input requires Poppler's pdftoppm command")
    output_base = destination.with_suffix("")
    command = [
        pdftoppm,
        "-png",
        "-r",
        "300",
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-singlefile",
        str(source),
        str(output_base),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not destination.is_file():
        raise TaskError(
            f"pdftoppm failed for {source} page {page_number} with exit "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )


def _tesseract_mode_evidence(
    page_image: Path, *, formula_count: int
) -> tuple[str, float | None, str]:
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        return "uncertain", None, "unavailable"
    completed = subprocess.run(
        [
            tesseract,
            str(page_image),
            "stdout",
            "-l",
            "chi_sim+eng",
            "--psm",
            "3",
            "tsv",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "uncertain", None, f"tesseract_exit_{completed.returncode}"
    lines = completed.stdout.splitlines()
    if not lines:
        return "uncertain", None, "no_tsv_rows"
    headers = lines[0].split("\t")
    try:
        confidence_index = headers.index("conf")
        text_index = headers.index("text")
    except ValueError:
        return "uncertain", None, "invalid_tsv_header"
    confidences: list[float] = []
    words = 0
    for line in lines[1:]:
        columns = line.split("\t")
        if len(columns) <= max(confidence_index, text_index):
            continue
        text = columns[text_index].strip()
        if not text:
            continue
        try:
            confidence = float(columns[confidence_index])
        except ValueError:
            continue
        if confidence >= 0:
            words += 1
            confidences.append(confidence / 100)
    if not confidences:
        return "uncertain", None, "no_confident_words"
    average = sum(confidences) / len(confidences)
    if words >= 3 and average >= 0.7:
        return "printed", round(average, 4), "tesseract_printed_evidence"
    if formula_count > 0 and average <= 0.35:
        return "handwritten", 0.72, "tesseract_low_print_confidence_with_formula"
    return "uncertain", round(average, 4), "tesseract_ambiguous"


def _crop_formula_regions(
    page_image: Path,
    regions: tuple[FormulaRegion, ...],
    *,
    page_number: int,
    formula_dir: Path,
    root: Path,
) -> list[dict[str, object]]:
    if not regions:
        return []
    crops: list[dict[str, object]] = []
    with Image.open(page_image) as image:
        width, height = image.size
        for formula_number, region in enumerate(regions, start=1):
            x0, y0, x1, y1 = region.bbox
            if max(abs(value) for value in region.bbox) <= 1.0:
                x0, x1 = x0 * width, x1 * width
                y0, y1 = y0 * height, y1 * height
            box = (
                max(0, min(width, int(round(x0)))),
                max(0, min(height, int(round(y0)))),
                max(0, min(width, int(round(x1)))),
                max(0, min(height, int(round(y1)))),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            destination = (
                formula_dir
                / f"page-{page_number:03d}-formula-{formula_number:03d}.png"
            )
            image.crop(box).save(destination, format="PNG", optimize=True)
            crops.append(
                {
                    "path": destination.relative_to(root).as_posix(),
                    "bbox": list(region.bbox),
                    "latex": region.latex,
                    "review": "required",
                }
            )
    return crops


def anchor_formula_crops(
    markdown: str, crops: list[dict[str, object]]
) -> str:
    """Add invisible stable anchors around formula text in reading order."""

    cursor = 0
    math_pattern = re.compile(
        r"\$\$(?P<display>.*?)\$\$|"
        r"(?<!\$)\$(?!\$)(?P<inline>.*?)(?<!\$)\$(?!\$)",
        re.S,
    )
    for formula_number, crop in enumerate(crops, start=1):
        path = str(crop["path"])
        identifier = Path(path).stem or f"formula-{formula_number:03d}"
        crop["id"] = identifier
        anchor = str(crop.get("markdown_anchor", identifier))
        crop["markdown_anchor"] = anchor
        start = f"<!-- paper2latex-formula:{anchor}:start -->"
        end = f"<!-- paper2latex-formula:{anchor}:end -->"
        if start in markdown and end in markdown:
            crop["markdown_mapping"] = "anchored"
            cursor = markdown.index(end, markdown.index(start)) + len(end)
            continue
        latex = crop.get("mineru_latex", crop.get("latex"))
        if not isinstance(latex, str) or not latex:
            crop["markdown_mapping"] = "manual_required"
            crop["review_flags"] = ["manual_markdown_placement_required"]
            continue
        expected = latex.strip()
        if expected.startswith("$$") and expected.endswith("$$"):
            expected = expected[2:-2].strip()
        elif expected.startswith("$") and expected.endswith("$"):
            expected = expected[1:-1].strip()
        match = next(
            (
                candidate
                for candidate in math_pattern.finditer(markdown, cursor)
                if (candidate.group("display") or candidate.group("inline") or "").strip()
                == expected
            ),
            None,
        )
        if match is None:
            crop["markdown_mapping"] = "manual_required"
            crop["review_flags"] = ["manual_markdown_placement_required"]
            continue
        token = match.group(0)
        delimiter = "$$" if token.startswith("$$") else "$"
        markdown = (
            markdown[: match.start()]
            + start
            + token
            + end
            + markdown[match.end() :]
        )
        crop["markdown_mapping"] = "anchored"
        crop["markdown_delimiter"] = delimiter
        cursor = match.start() + len(start) + len(token) + len(end)
    return markdown


def convert_document(
    *,
    inputs: Iterable[Path],
    output_dir: Path,
    name: str,
    mode: str,
    engine: RecognitionEngine,
    force_quality: bool = False,
    generate_pdfs: bool = True,
) -> ConversionResult:
    """Convert input images into a provenance-backed review package."""

    if mode not in CONVERSION_MODES:
        raise TaskError(f"mode must be one of: {', '.join(sorted(CONVERSION_MODES))}")
    sources = _source_files(inputs)
    source_pages = _expand_source_pages(sources)
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise TaskError(f"output already exists: {root}")

    original_dir = root / "original"
    cleaned_dir = root / "cleaned"
    formula_dir = root / "formula-crops"
    exports_dir = root / "exports"
    engine_dir = exports_dir / engine.engine_id
    for directory in (
        original_dir,
        cleaned_dir,
        formula_dir,
        engine_dir,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    original_paths: dict[Path, Path] = {}
    for source_number, source in enumerate(sources, start=1):
        original = original_dir / f"{source_number:03d}-{source.name}"
        shutil.copy2(source, original)
        original_paths[source] = original

    pages: list[dict[str, object]] = []
    cleaned_paths: list[Path] = []
    for page_number, source_page in enumerate(source_pages, start=1):
        source = source_page.source
        original = original_paths[source]
        cleaned = cleaned_dir / f"page-{page_number:03d}.png"
        if source_page.source_page is None:
            quality = _prepare_image(source, cleaned)
        else:
            _rasterize_pdf_page(source, source_page.source_page, cleaned)
            quality = _image_quality(cleaned)
        quality_issues = quality.get("issues")
        if not isinstance(quality_issues, list):
            raise TaskError(f"quality analyzer returned invalid issues for page {page_number}")
        if quality_issues and not force_quality:
            # root did not exist on entry and every child was created by this call.
            shutil.rmtree(root)
            raise TaskError(
                f"page {page_number} failed the quality gate: "
                f"{', '.join(str(issue) for issue in quality_issues)}; "
                "retake it or pass --force-quality"
            )
        if quality_issues and force_quality:
            quality = dict(quality)
            quality["state"] = "forced"
        cleaned_paths.append(cleaned)
        pages.append(
            {
                "page": page_number,
                "mode": mode if mode != "auto" else "pending",
                "mode_confidence": 1.0 if mode != "auto" else None,
                "mode_confirmation_required": False,
                "source_name": source.name,
                "source_page": source_page.source_page,
                "original": original.relative_to(root).as_posix(),
                "cleaned": cleaned.relative_to(root).as_posix(),
                "sha256": _sha256(original),
                "quality_gate": quality,
                "cleanup": "completed",
                "text_ocr": "pending",
                "formula_detection": "pending",
                "formula_ocr": "pending",
                "human_review": "required",
            }
        )
        quality_warnings = quality.get("warnings", [])
        if isinstance(quality_warnings, list):
            warnings_for_page = [str(item) for item in quality_warnings]
            pages[-1]["quality_warnings"] = warnings_for_page

    recognized = engine.recognize(cleaned_paths, engine_dir)
    sections = [f"# {name.strip() or 'Untitled conversion'}"]
    warnings: list[str] = [
        f"page {page['page']}: {warning}"
        for page in pages
        for warning in page.get("quality_warnings", [])
    ]
    for page, cleaned in zip(pages, cleaned_paths, strict=True):
        result = recognized.get(cleaned)
        if result is None:
            page["text_ocr"] = "failed"
            page["formula_detection"] = "failed"
            page["formula_ocr"] = "failed"
            page["warnings"] = ["engine_returned_no_result"]
            sections.extend(
                [
                    f"## Page {page['page']}",
                    "> [!WARNING] OCR failed for this page. Retake or rerun it.",
                ]
            )
            warnings.append(f"page {page['page']}: engine returned no result")
            continue

        page["engine"] = {
            "id": engine.engine_id,
            "version": engine.engine_version,
        }
        page["confidence"] = result.confidence
        page_number = page.get("page")
        if not isinstance(page_number, int):
            raise TaskError("internal page record is missing an integer page number")
        formula_crops = _crop_formula_regions(
            cleaned,
            result.formula_regions,
            page_number=page_number,
            formula_dir=formula_dir,
            root=root,
        )
        page_markdown = anchor_formula_crops(result.markdown.strip(), formula_crops)
        page["formula_crops"] = formula_crops
        page["formula_count"] = max(result.formula_count, len(formula_crops))
        if mode == "auto":
            if result.suggested_mode in MODES:
                suggested_mode = result.suggested_mode
                mode_confidence = result.mode_confidence
                mode_source = "engine"
            else:
                suggested_mode, mode_confidence, mode_source = (
                    _tesseract_mode_evidence(
                        cleaned,
                        formula_count=int(page["formula_count"]),
                    )
                )
            page["mode"] = suggested_mode
            page["mode_confidence"] = mode_confidence
            page["mode_source"] = mode_source
            page["mode_confirmation_required"] = (
                suggested_mode == "uncertain"
                or mode_confidence is None
                or mode_confidence < 0.7
            )
            if page["mode_confirmation_required"]:
                warnings.append(f"page {page['page']}: mode confirmation required")
        page["text_ocr"] = "completed"
        page["formula_detection"] = (
            "completed" if engine.formula_capable else "unsupported"
        )
        page["formula_ocr"] = (
            "completed" if engine.formula_capable else "unsupported"
        )
        page["warnings"] = list(result.warnings)
        warnings.extend(f"page {page['page']}: {item}" for item in result.warnings)
        sections.extend([f"## Page {page['page']}", page_markdown])

    formula_jobs: list[tuple[dict[str, object], dict[str, object], Path]] = []
    if engine.engine_id == "mineru":
        for page in pages:
            crops = page.get("formula_crops", [])
            if not isinstance(crops, list):
                continue
            for crop in crops:
                if isinstance(crop, dict) and isinstance(crop.get("path"), str):
                    formula_jobs.append((page, crop, root / str(crop["path"])))
    if formula_jobs:
        from .unimernet_engine import UniMERNetEngine, UniMERNetError

        unimernet = UniMERNetEngine()
        try:
            predictions = unimernet.recognize(
                [job[2] for job in formula_jobs],
                log_path=exports_dir / "unimernet.log",
            )
        except UniMERNetError as error:
            warnings.append(f"UniMERNet refinement: {error}")
        else:
            for (page, crop, _), prediction in zip(
                formula_jobs, predictions, strict=True
            ):
                mineru_latex = crop.get("latex")
                crop["mineru_latex"] = mineru_latex
                crop["latex"] = prediction.latex
                crop["recognition_engine"] = {
                    "id": unimernet.engine_id,
                    "version": unimernet.engine_version,
                }
                crop["review"] = "required"
                page["formula_refinement"] = "unimernet_completed"
            warnings.append(
                f"UniMERNet proposed {len(predictions)} formula crop replacement(s); "
                "Markdown remains on MinerU values until human formula review"
            )

    failed_page_count = sum(page["text_ocr"] == "failed" for page in pages)
    if failed_page_count == 0:
        conversion_state = "completed"
    elif failed_page_count == len(pages):
        conversion_state = "failed"
    else:
        conversion_state = "partial_failed"

    markdown_path = root / "document.md"
    document_markdown = "\n\n".join(sections).rstrip() + "\n"
    markdown_path.write_text(document_markdown, encoding="utf-8")

    outputs = {
        "markdown": "document.md",
        "rendered_pdf": "not_requested" if not generate_pdfs else "not_generated",
        "searchable_pdf": "not_requested" if not generate_pdfs else "not_generated",
    }
    if generate_pdfs:
        rendered_pdf = root / "document.pdf"
        try:
            render_markdown_pdf(
                markdown_path,
                rendered_pdf,
                log_path=exports_dir / "markdown-pdf.log",
            )
            outputs["rendered_pdf"] = rendered_pdf.relative_to(root).as_posix()
        except RenderError as error:
            warnings.append(f"rendered PDF: {error}")
            outputs["rendered_pdf"] = "failed"

        searchable_pdf = root / "searchable.pdf"
        try:
            render_searchable_pdf(
                cleaned_paths,
                searchable_pdf,
                work_dir=exports_dir / "searchable-pages",
            )
            outputs["searchable_pdf"] = searchable_pdf.relative_to(root).as_posix()
        except RenderError as error:
            warnings.append(f"searchable PDF: {error}")
            outputs["searchable_pdf"] = "failed"

    created_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": 2,
        "task_id": root.name,
        "name": name.strip() or "Untitled conversion",
        "created_at": created_at,
        "status": "needs_review",
        "conversion_state": conversion_state,
        "mode": mode,
        "page_count": len(pages),
        "privacy": {"processing": "local_only", "cloud_upload": False},
        "outputs": outputs,
        "pages": pages,
        "document_markdown": document_markdown,
        "warnings": warnings,
    }
    (root / "review.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ConversionResult(
        output_dir=root,
        status="needs_review",
        page_count=len(pages),
        warnings=tuple(warnings),
    )
