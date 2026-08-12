"""Discovery and adapters for local OCR engines."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .pipeline import FormulaRegion, PageRecognition, RecognitionEngine


class EngineError(RuntimeError):
    """Raised when an OCR engine cannot run without violating the contract."""


def _first_version_line(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    text = (completed.stdout or completed.stderr).strip()
    return text.splitlines()[0] if text else "unknown"


class TesseractEngine:
    """Printed-text fallback that deliberately makes no LaTeX claim."""

    engine_id = "tesseract"
    formula_capable = False

    def __init__(self, *, languages: str = "chi_sim+eng") -> None:
        command = shutil.which("tesseract")
        if command is None:
            raise EngineError("Tesseract is not installed or not on PATH")
        self.command = command
        self.languages = languages
        self.engine_version = _first_version_line([command, "--version"])

    def _run(self, page: Path, output_format: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                self.command,
                str(page),
                "stdout",
                "-l",
                self.languages,
                "--psm",
                "3",
                output_format,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def recognize(
        self, pages: list[Path], work_dir: Path
    ) -> dict[Path, PageRecognition]:
        work_dir.mkdir(parents=True, exist_ok=True)
        results: dict[Path, PageRecognition] = {}
        logs: list[str] = []
        for page in pages:
            text_result = self._run(page, "txt")
            tsv_result = self._run(page, "tsv")
            logs.append(
                f"page: {page}\n"
                f"text_exit: {text_result.returncode}\n{text_result.stderr}\n"
                f"tsv_exit: {tsv_result.returncode}\n{tsv_result.stderr}\n"
            )
            if text_result.returncode != 0:
                continue

            confidences: list[float] = []
            if tsv_result.returncode == 0:
                lines = tsv_result.stdout.splitlines()
                if lines:
                    headers = lines[0].split("\t")
                    try:
                        confidence_index = headers.index("conf")
                    except ValueError:
                        confidence_index = -1
                    if confidence_index >= 0:
                        for line in lines[1:]:
                            columns = line.split("\t")
                            if len(columns) <= confidence_index:
                                continue
                            try:
                                value = float(columns[confidence_index])
                            except ValueError:
                                continue
                            if value >= 0:
                                confidences.append(value / 100)

            confidence = (
                sum(confidences) / len(confidences) if confidences else None
            )
            markdown = text_result.stdout.strip()
            if not markdown:
                markdown = "> [!WARNING] Tesseract found no printed text on this page."
            if confidence is not None and confidence >= 0.72 and len(markdown.split()) >= 2:
                suggested_mode = "printed"
                mode_confidence = min(0.99, confidence)
            elif confidence is not None and confidence <= 0.35 and markdown:
                suggested_mode = "handwritten"
                mode_confidence = 0.55
            else:
                suggested_mode = None
                mode_confidence = None
            results[page] = PageRecognition(
                markdown=markdown,
                confidence=confidence,
                formula_count=0,
                warnings=("formula_recognition_not_supported",),
                suggested_mode=suggested_mode,
                mode_confidence=mode_confidence,
            )

        (work_dir / "tesseract-ocr.log").write_text(
            "\n".join(logs), encoding="utf-8"
        )
        return results


def _mineru_command_and_python() -> tuple[str, str] | None:
    command = shutil.which("mineru")
    if command is None:
        return None
    try:
        first_line = Path(command).read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeDecodeError, IndexError):
        return None
    if not first_line.startswith("#!"):
        return None
    python = first_line[2:].strip()
    if not Path(python).is_file():
        return None
    return command, python


def _mineru_config_path() -> Path:
    configured = os.getenv("MINERU_TOOLS_CONFIG_JSON", "mineru.json")
    path = Path(configured).expanduser()
    return path if path.is_absolute() else Path.home() / path


def _mineru_model_root(backend: str) -> Path | None:
    config_path = _mineru_config_path()
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = "pipeline" if backend == "pipeline" else "vlm"
    configured = data.get("models-dir", {}).get(key)
    if not isinstance(configured, str) or not configured.strip():
        return None
    root = Path(configured).expanduser()
    if not root.is_dir():
        return None
    required = (
        ("config.json", "model.safetensors")
        if backend != "pipeline"
        else ("models",)
    )
    return root if all((root / name).exists() for name in required) else None


def _mineru_safe_command(python: str, arguments: list[str]) -> list[str]:
    # Importing OpenCV first makes the process use one existing OpenMP runtime.
    # This avoids MinerU's observed duplicate-runtime abort without unsafe flags.
    launcher = (
        "import cv2, sys; "
        "from mineru.cli.client import main; "
        "sys.argv[0] = 'mineru'; "
        "main()"
    )
    return [python, "-c", launcher, *arguments]


def _walk_json_objects(value: object):
    """Yield every object in a nested MinerU JSON result."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_objects(child)


def parse_mineru_formula_regions(
    work_dir: Path, page_stem: str
) -> tuple[FormulaRegion, ...]:
    """Read formula boxes and LaTeX from MinerU VLM content-list artifacts.

    MinerU output layouts vary by backend and version, so this parser searches
    nested content-list JSON rather than assuming one fixed directory shape.
    VLM content-list boxes use the documented 0-1000 page coordinate space;
    convert them to the pipeline's normalized 0-1 crop contract.
    Malformed files and malformed individual regions are ignored; the Markdown
    remains the primary recognition result and review.json records available
    crops separately.
    """

    candidates = sorted(work_dir.rglob(f"{page_stem}_content_list*.json"))
    region_order: list[tuple[float, float, float, float]] = []
    latex_by_bbox: dict[tuple[float, float, float, float], str | None] = {}
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in _walk_json_objects(data):
            kind = str(
                item.get("type")
                or item.get("content_type")
                or item.get("category")
                or ""
            ).casefold()
            if "equation" not in kind and "formula" not in kind:
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                normalized_bbox: tuple[float, float, float, float] = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            except (TypeError, ValueError):
                continue
            if max(normalized_bbox) > 1.0:
                if min(normalized_bbox) < 0.0 or max(normalized_bbox) > 1000.0:
                    continue
                normalized_bbox = (
                    normalized_bbox[0] / 1000.0,
                    normalized_bbox[1] / 1000.0,
                    normalized_bbox[2] / 1000.0,
                    normalized_bbox[3] / 1000.0,
                )
            if (
                min(normalized_bbox) < 0.0
                or max(normalized_bbox) > 1.0
                or normalized_bbox[2] <= normalized_bbox[0]
                or normalized_bbox[3] <= normalized_bbox[1]
            ):
                continue
            content = item.get("content")
            nested_content = content if isinstance(content, dict) else {}
            latex_value = (
                item.get("latex")
                or item.get("text")
                or nested_content.get("math_content")
                or nested_content.get("latex")
                or content
            )
            latex = latex_value.strip() if isinstance(latex_value, str) else None
            latex = latex or None
            if normalized_bbox not in latex_by_bbox:
                region_order.append(normalized_bbox)
                latex_by_bbox[normalized_bbox] = latex
            elif latex_by_bbox[normalized_bbox] is None and latex is not None:
                latex_by_bbox[normalized_bbox] = latex
    reading_order = sorted(region_order, key=lambda bbox: (bbox[1], bbox[0]))
    return tuple(
        FormulaRegion(bbox=bbox, latex=latex_by_bbox[bbox])
        for bbox in reading_order
    )


def rewrite_mineru_markdown_assets(
    markdown: str,
    *,
    markdown_path: Path,
    work_dir: Path,
    page_stem: str,
) -> tuple[str, tuple[str, ...]]:
    """Copy MinerU image assets into a stable task path and rewrite Markdown."""

    asset_dir = work_dir / "assets" / page_stem
    warnings: list[str] = []
    used_names: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        label, value = match.group(1), match.group(2).strip()
        if value.startswith(("http://", "https://", "data:")):
            return match.group(0)
        source = Path(value)
        if not source.is_absolute():
            source = (markdown_path.parent / source).resolve()
        if not source.is_file():
            warnings.append(f"missing_markdown_asset:{value}")
            return match.group(0)
        asset_dir.mkdir(parents=True, exist_ok=True)
        name = source.name
        if name in used_names:
            name = f"{source.stem}-{len(used_names) + 1}{source.suffix}"
        used_names.add(name)
        destination = asset_dir / name
        shutil.copy2(source, destination)
        stable = f"exports/mineru/assets/{page_stem}/{name}"
        return f"![{label}]({stable})"

    rewritten = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, markdown)
    return rewritten, tuple(warnings)


class MinerUEngine:
    """Formula-capable MinerU adapter with an explicit local-model gate."""

    engine_id = "mineru"
    formula_capable = True

    def __init__(self, *, backend: str = "vlm-engine", effort: str = "high") -> None:
        located = _mineru_command_and_python()
        if located is None:
            raise EngineError("MinerU is not installed or its Python entry point is invalid")
        self.command, self.python = located
        self.backend = backend
        self.effort = effort
        self.model_root = _mineru_model_root(backend)
        version_command = _mineru_safe_command(self.python, ["--version"])
        self.engine_version = _first_version_line(version_command)

    @property
    def model_ready(self) -> bool:
        return self.model_root is not None

    def recognize(
        self, pages: list[Path], work_dir: Path
    ) -> dict[Path, PageRecognition]:
        if not self.model_ready:
            model_type = "pipeline" if self.backend == "pipeline" else "vlm"
            raise EngineError(
                f"MinerU {model_type} models are not configured locally. "
                "Refusing an implicit model download; run the reviewed model-install step first."
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        input_dir = pages[0].parent
        arguments = [
            "--path",
            str(input_dir),
            "--output",
            str(work_dir),
            "--backend",
            self.backend,
            "--method",
            "ocr",
            "--formula",
            "true",
            "--table",
            "true",
        ]
        if self.backend.startswith("hybrid"):
            arguments.extend(["--effort", self.effort])
        command = _mineru_safe_command(self.python, arguments)
        environment = dict(os.environ)
        environment["MINERU_MODEL_SOURCE"] = "local"
        environment["MINERU_TOOLS_CONFIG_JSON"] = str(_mineru_config_path())
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        (work_dir / "mineru.log").write_text(
            f"command: {' '.join(command)}\n"
            f"exit_code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n",
            encoding="utf-8",
        )

        results: dict[Path, PageRecognition] = {}
        for page in pages:
            candidates = sorted(work_dir.rglob(f"{page.stem}.md"))
            if not candidates:
                continue
            markdown = candidates[-1].read_text(encoding="utf-8").strip()
            markdown, asset_warnings = rewrite_mineru_markdown_assets(
                markdown,
                markdown_path=candidates[-1],
                work_dir=work_dir,
                page_stem=page.stem,
            )
            formula_regions = parse_mineru_formula_regions(work_dir, page.stem)
            markdown_formula_count = len(
                re.findall(r"\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", markdown, re.S)
            )
            formula_count = max(markdown_formula_count, len(formula_regions))
            warnings = asset_warnings
            if completed.returncode != 0:
                warnings += (f"mineru_batch_exit_{completed.returncode}",)
            results[page] = PageRecognition(
                markdown=markdown,
                confidence=None,
                formula_count=formula_count,
                warnings=warnings,
                formula_regions=formula_regions,
            )
        return results


def select_engine(name: str, *, languages: str = "chi_sim+eng") -> RecognitionEngine:
    """Select an integrated local engine without triggering downloads."""

    if name == "tesseract":
        return TesseractEngine(languages=languages)
    if name == "mineru":
        mineru = MinerUEngine()
        if not mineru.model_ready:
            raise EngineError(
                "MinerU VLM models are not configured locally. Refusing an implicit "
                "model download; run the reviewed model-install step first."
            )
        return mineru
    if name != "auto":
        raise EngineError(f"unknown engine: {name}")
    try:
        mineru = MinerUEngine()
    except EngineError:
        mineru = None
    if mineru is not None and mineru.model_ready:
        return mineru
    return TesseractEngine(languages=languages)


ENGINE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "mineru",
        "role": "formula_capable_document_to_markdown",
        "code_license": "LicenseRef-MinerU-Open-Source-License",
        "python_package": "mineru",
        "command": "mineru",
        "integration": "integrated",
    },
    {
        "id": "tesseract",
        "role": "printed_text_fallback_and_searchable_pdf",
        "code_license": "Apache-2.0",
        "python_package": None,
        "command": "tesseract",
        "integration": "integrated",
    },
    {
        "id": "pix2text",
        "role": "printed_or_mixed_page_to_markdown",
        "code_license": "MIT",
        "python_package": "pix2text",
        "command": None,
        "integration": "candidate",
    },
    {
        "id": "unimernet",
        "role": "handwritten_formula_image_to_latex",
        "code_license": "Apache-2.0",
        "python_package": "unimernet",
        "command": "unimernet_gui",
        "integration": "direct_formula_refinement",
    },
    {
        "id": "ocrmypdf",
        "role": "searchable_pdfa",
        "code_license": "MPL-2.0",
        "python_package": "ocrmypdf",
        "command": "ocrmypdf",
        "integration": "candidate_tesseract_fallback_integrated",
    },
)


def discover_engines() -> list[dict[str, Any]]:
    """Return integration, executable, package, and local-model readiness."""

    results: list[dict[str, Any]] = []
    for candidate in ENGINE_CANDIDATES:
        package = candidate["python_package"]
        command = candidate["command"]
        item = dict(candidate)
        item["package_available"] = bool(
            package and importlib.util.find_spec(package) is not None
        )
        item["command_available"] = bool(command and shutil.which(command))
        item["model_weights_bundled"] = False
        if candidate["id"] == "mineru":
            item["vlm_model_ready"] = _mineru_model_root("vlm-engine") is not None
            item["pipeline_model_ready"] = _mineru_model_root("pipeline") is not None
            item["safe_launcher_available"] = _mineru_command_and_python() is not None
        results.append(item)
    return results
