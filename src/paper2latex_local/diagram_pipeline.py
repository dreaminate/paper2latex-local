"""Diagram task packaging from explicit graph candidates.

This module intentionally separates low-level photo routing from semantic graph
recognition.  The current vertical slice accepts a deterministic graph fixture
or a future recognizer's canonical JSON and emits reviewable editable formats.
It does not label OpenCV shape candidates as accurately recognized handwriting.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .diagram import DiagramExportError, DiagramGraph, parse_diagram_fixture
from .diagram_export import (
    export_drawio,
    export_mermaid,
    export_pdf,
    export_svg,
)


@dataclass(frozen=True)
class DiagramConversionResult:
    output_dir: Path
    status: str
    graph_path: Path
    clean_drawio: Path


def _drawio_command() -> str | None:
    installed = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")
    if installed.is_file():
        return str(installed)
    return shutil.which("drawio")


def _export_native_pdf(drawio: Path, output: Path) -> Path:
    command = _drawio_command()
    if command is None:
        raise DiagramExportError("native draw.io PDF export was requested but draw.io is unavailable")
    completed = subprocess.run(
        [command, "--export", "--format", "pdf", "--output", str(output), str(drawio)],
        text=True,
        capture_output=True,
        check=False,
    )
    output.with_suffix(".pdf.log").write_text(
        f"command: {command} --export --format pdf --output {output} {drawio}\n"
        f"exit_code: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output.is_file():
        raise DiagramExportError(
            f"draw.io PDF export failed with exit {completed.returncode}; "
            f"see {output.with_suffix('.pdf.log')}"
        )
    if len(PdfReader(output).pages) < 1:
        raise DiagramExportError(f"draw.io generated an empty PDF: {output}")
    return output


def _export_layouts(
    graph: DiagramGraph,
    output_dir: Path,
    *,
    final: bool,
) -> dict[str, str]:
    diagram_dir = output_dir / "diagram"
    diagram_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    native_pdf = _drawio_command() is not None and os.getenv(
        "PAPER2LATEX_USE_DRAWIO_PDF", "1"
    ) == "1"
    for layout in ("faithful", "clean"):
        drawio = export_drawio(
            graph,
            diagram_dir / f"{layout}.drawio",
            layout=layout,
            final=final,
        )
        mermaid = export_mermaid(
            graph,
            diagram_dir / f"{layout}.mmd",
            layout=layout,
            final=final,
        )
        svg = export_svg(
            graph,
            diagram_dir / f"{layout}.svg",
            layout=layout,
            final=final,
        )
        pdf_path = diagram_dir / f"{layout}.pdf"
        if native_pdf:
            pdf = _export_native_pdf(drawio, pdf_path)
        else:
            pdf = export_pdf(
                graph,
                pdf_path,
                layout=layout,
                final=final,
            )
        outputs.update(
            {
                f"{layout}_drawio": drawio.relative_to(output_dir).as_posix(),
                f"{layout}_mermaid": mermaid.relative_to(output_dir).as_posix(),
                f"{layout}_svg": svg.relative_to(output_dir).as_posix(),
                f"{layout}_pdf": pdf.relative_to(output_dir).as_posix(),
            }
        )
    return outputs


def convert_diagram_fixture(
    *,
    fixture: str | Path | dict[str, object],
    output_dir: Path,
    original: Path | None = None,
    cleaned: Path | None = None,
    content_kind: str = "flowchart",
    recognition_scope: str = "explicit_canonical_graph_fixture",
    final: bool = False,
) -> DiagramConversionResult:
    """Create a complete editable diagram package from canonical graph JSON."""

    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise ValueError(f"output already exists: {root}")
    root.mkdir(parents=True)
    graph = parse_diagram_fixture(fixture)
    graph_path = root / "diagram.json"
    graph_path.write_text(graph.to_json(), encoding="utf-8")
    outputs = _export_layouts(graph, root, final=final)
    original_value = None
    if original is not None:
        source = Path(original).expanduser().resolve()
        original_dir = root / "original"
        original_dir.mkdir()
        destination = original_dir / source.name
        destination.write_bytes(source.read_bytes())
        original_value = destination.relative_to(root).as_posix()
    cleaned_value = None
    if cleaned is not None:
        source = Path(cleaned).expanduser().resolve()
        cleaned_dir = root / "cleaned"
        cleaned_dir.mkdir()
        destination = cleaned_dir / "page-001.png"
        destination.write_bytes(source.read_bytes())
        cleaned_value = destination.relative_to(root).as_posix()
    review = {
        "schema_version": 3,
        "task_id": root.name,
        "status": "finalized" if final else "needs_review",
        "conversion_state": "completed",
        "content_kind": content_kind,
        "recognition_scope": recognition_scope,
        "original": original_value,
        "cleaned": cleaned_value,
        "graph": graph.to_dict(),
        "outputs": outputs,
        "human_review": "confirmed" if final else "required",
        "warnings": (
            []
            if final
            else [
                "semantic_graph_recognition_requires_human_review",
                "real_handwritten_diagram_accuracy_unverified",
            ]
        ),
    }
    (root / "diagram-review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DiagramConversionResult(
        output_dir=root,
        status=str(review["status"]),
        graph_path=graph_path,
        clean_drawio=root / outputs["clean_drawio"],
    )


def finalize_diagram_package(output_dir: Path) -> DiagramConversionResult:
    """Regenerate final exports from the review server's confirmed graph state."""

    root = Path(output_dir).expanduser().resolve()
    final_path = root / "review.final.json"
    if not final_path.is_file():
        raise ValueError(f"final review snapshot not found: {final_path}")
    review = json.loads(final_path.read_text(encoding="utf-8"))
    graph_value = review.get("graph", review.get("diagram", review))
    graph = DiagramGraph.from_dict(graph_value)
    outputs = _export_layouts(graph, root, final=True)
    review["outputs"] = outputs
    review["status"] = "finalized"
    review["human_review"] = "confirmed"
    (root / "diagram-review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    graph_path = root / "diagram.json"
    graph_path.write_text(graph.to_json(), encoding="utf-8")
    return DiagramConversionResult(
        output_dir=root,
        status="finalized",
        graph_path=graph_path,
        clean_drawio=root / outputs["clean_drawio"],
    )


__all__ = [
    "DiagramConversionResult",
    "convert_diagram_fixture",
    "finalize_diagram_package",
]
