"""Create immutable-input task packages for later OCR processing."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

MAX_PAGES = 50
MODES = frozenset({"printed", "handwritten"})
SUPPORTED_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp"}
)


class TaskError(ValueError):
    """Raised when a task cannot be created without violating its contract."""


def _safe_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    slug = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE)
    slug = slug.strip("-._")
    return slug or "task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(inputs: Iterable[Path]) -> list[Path]:
    paths = [Path(item).expanduser().resolve() for item in inputs]
    if not paths:
        raise TaskError("at least one input photo is required")
    if len(paths) > MAX_PAGES:
        raise TaskError(f"a task accepts at most {MAX_PAGES} photos")
    if len(set(paths)) != len(paths):
        raise TaskError("duplicate input paths are not allowed")

    for path in paths:
        if not path.is_file():
            raise TaskError(f"input is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise TaskError(f"unsupported input suffix {path.suffix!r}; expected {allowed}")
    return paths


def create_task(
    *,
    name: str,
    mode: str,
    inputs: Iterable[Path],
    output_root: Path,
    now: datetime | None = None,
) -> Path:
    """Create a new task package and return its absolute path.

    The function copies source files but does not decode or modify them. Image
    validity and photo-quality checks remain explicit `not_run` stages.
    """

    if mode not in MODES:
        raise TaskError(f"mode must be one of: {', '.join(sorted(MODES))}")
    source_paths = _validate_inputs(inputs)
    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    task_id = f"{created_at:%Y%m%dT%H%M%SZ}-{_safe_slug(name)}"
    root = Path(output_root).expanduser().resolve()
    task_dir = root / task_id
    if task_dir.exists():
        raise TaskError(f"task already exists: {task_dir}")

    original_dir = task_dir / "original"
    cleaned_dir = task_dir / "cleaned"
    formula_dir = task_dir / "formula-crops"
    exports_dir = task_dir / "exports"
    for directory in (original_dir, cleaned_dir, formula_dir, exports_dir):
        directory.mkdir(parents=True, exist_ok=False)

    pages: list[dict[str, Any]] = []
    for index, source in enumerate(source_paths, start=1):
        copied_name = f"{index:03d}-{_safe_slug(source.stem)}{source.suffix.lower()}"
        destination = original_dir / copied_name
        shutil.copy2(source, destination)
        pages.append(
            {
                "page": index,
                "mode": mode,
                "source_name": source.name,
                "original": destination.relative_to(task_dir).as_posix(),
                "sha256": _sha256(destination),
                "quality_gate": "not_run",
                "cleanup": "not_run",
                "text_ocr": "not_run",
                "formula_detection": "not_run",
                "formula_ocr": "not_run",
                "human_review": "not_run",
            }
        )

    document = (
        f"# {name.strip() or 'Untitled task'}\n\n"
        f"<!-- task_id: {task_id}; mode: {mode}; OCR has not run -->\n\n"
        "_Recognition output will be written here after review._\n"
    )
    (task_dir / "document.md").write_text(document, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "task_id": task_id,
        "name": name.strip() or "Untitled task",
        "mode": mode,
        "created_at": created_at.isoformat(),
        "page_count": len(pages),
        "privacy": {"processing": "local_only", "cloud_upload": False},
        "outputs": {
            "markdown": "document.md",
            "rendered_pdf": "not_generated",
            "searchable_pdf": "not_generated",
        },
        "pages": pages,
    }
    (task_dir / "review.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return task_dir


def load_status(task_dir: Path) -> dict[str, Any]:
    """Read and minimally validate a task manifest."""

    root = Path(task_dir).expanduser().resolve()
    manifest_path = next(
        (
            candidate
            for candidate in (
                root / "review.final.json",
                root / "review.current.json",
                root / "review.json",
                root / "diagram-review.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise TaskError(f"review manifest not found in: {root}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data.get("graph", data.get("diagram")), dict):
        required = {"schema_version", "status", "outputs"}
        data.setdefault("task_id", root.name)
    else:
        required = {"schema_version", "task_id", "mode", "page_count", "pages"}
    missing = required.difference(data)
    if missing:
        raise TaskError(f"review manifest is missing: {', '.join(sorted(missing))}")
    return data
