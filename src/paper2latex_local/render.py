"""PDF renderers backed by mature local command-line tools."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader, PdfWriter


class RenderError(RuntimeError):
    """Raised when a requested PDF artifact cannot be validated."""


def _require_command(name: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise RenderError(f"required command is not available: {name}")
    return command


def _validate_pdf(path: Path, *, expected_pages: int | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RenderError(f"PDF was not generated: {path}")
    try:
        page_count = len(PdfReader(path).pages)
    except Exception as error:  # pypdf exposes several parse exceptions.
        raise RenderError(f"generated PDF is invalid: {path}: {error}") from error
    if page_count < 1:
        raise RenderError(f"generated PDF has no pages: {path}")
    if expected_pages is not None and page_count != expected_pages:
        raise RenderError(
            f"generated PDF has {page_count} pages; expected {expected_pages}: {path}"
        )


def render_markdown_pdf(
    markdown_path: Path,
    output_pdf: Path,
    *,
    log_path: Path,
    working_dir: Path | None = None,
) -> None:
    """Render Markdown and TeX math through Pandoc and XeLaTeX."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pandoc = _require_command("pandoc")
    xelatex = _require_command("xelatex")
    command = [
        pandoc,
        str(markdown_path),
        "--from=markdown+tex_math_dollars",
        f"--pdf-engine={xelatex}",
        "--variable=geometry:margin=1in",
        "--variable=CJKmainfont=Songti SC",
        "--variable=mainfont=Arial Unicode MS",
        "--variable=mathfont=STIX Two Math",
        "--output",
        str(output_pdf),
    ]
    completed = subprocess.run(
        command,
        cwd=working_dir or output_pdf.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        f"command: {' '.join(command)}\n"
        f"exit_code: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RenderError(
            f"Pandoc/XeLaTeX failed with exit {completed.returncode}; see {log_path}"
        )
    fatal_warnings = (
        "Missing character:",
        "Could not fetch resource",
        "File not found",
    )
    if any(marker in completed.stderr for marker in fatal_warnings):
        raise RenderError(
            f"Pandoc/XeLaTeX completed with content-loss warnings; see {log_path}"
        )
    _validate_pdf(output_pdf)


def render_searchable_pdf(
    pages: list[Path],
    output_pdf: Path,
    *,
    work_dir: Path,
    languages: str = "chi_sim+eng",
) -> None:
    """Create page-faithful searchable PDFs with Tesseract and merge them."""

    tesseract = _require_command("tesseract")
    work_dir.mkdir(parents=True, exist_ok=True)
    page_pdfs: list[Path] = []
    log_parts: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        output_base = work_dir / f"page-{page_number:03d}"
        command = [
            tesseract,
            str(page),
            str(output_base),
            "-l",
            languages,
            "--psm",
            "3",
            "pdf",
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        log_parts.append(
            f"command: {' '.join(command)}\n"
            f"exit_code: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
        )
        generated = output_base.with_suffix(".pdf")
        if completed.returncode != 0 or not generated.is_file():
            (work_dir / "tesseract.log").write_text(
                "\n".join(log_parts), encoding="utf-8"
            )
            raise RenderError(
                f"Tesseract PDF generation failed for page {page_number} with "
                f"exit {completed.returncode}; see {work_dir / 'tesseract.log'}"
            )
        page_pdfs.append(generated)

    (work_dir / "tesseract.log").write_text(
        "\n".join(log_parts), encoding="utf-8"
    )
    writer = PdfWriter()
    for page_pdf in page_pdfs:
        writer.append(page_pdf)
    with output_pdf.open("wb") as handle:
        writer.write(handle)
    _validate_pdf(output_pdf, expected_pages=len(pages))
