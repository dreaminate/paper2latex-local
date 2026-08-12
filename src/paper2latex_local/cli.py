"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Sequence

from .engines import EngineError, discover_engines, select_engine
from .content import ContentKind, RouteDecision, route_content
from .diagram_pipeline import convert_diagram_fixture, finalize_diagram_package
from .diagram_vision import DiagramRecognitionError, recognize_diagram
from .pipeline import CONVERSION_MODES, convert_document, resolve_source_files
from .photo import preprocess_photo
from .review_server import ReviewServer, unresolved_items
from .task import MODES, TaskError, create_task, load_status
from .unimernet_engine import unimernet_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper2latex",
        description="Prepare local-first OCR task packages without uploading photos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-task", help="create a task from phone photos")
    init.add_argument("inputs", nargs="+", type=Path)
    init.add_argument("--name", required=True)
    init.add_argument("--mode", required=True, choices=sorted(MODES))
    init.add_argument("--output-root", type=Path, default=Path("tasks"))

    convert = subparsers.add_parser(
        "convert", help="run local OCR and generate Markdown/PDF review artifacts"
    )
    convert.add_argument("inputs", nargs="+", type=Path)
    convert.add_argument("--name")
    convert.add_argument("--mode", default="auto", choices=sorted(CONVERSION_MODES))
    convert.add_argument("--engine", default="auto", choices=("auto", "mineru", "tesseract"))
    convert.add_argument("--language", default="chi_sim+eng")
    convert.add_argument(
        "--content-type",
        default="auto",
        choices=("auto", "document", "flowchart", "mindmap", "architecture"),
    )
    convert.add_argument(
        "--diagram-fixture",
        type=Path,
        help="canonical diagram JSON from a recognizer or synthetic acceptance fixture",
    )
    convert.add_argument("--output", type=Path)
    convert.add_argument("--force-quality", action="store_true")
    convert.add_argument("--no-pdf", action="store_true")

    status = subparsers.add_parser("status", help="show manifest-backed task status")
    status.add_argument("task_dir", type=Path)

    subparsers.add_parser(
        "engines", help="show local OCR engine, adapter, and model readiness"
    )
    review = subparsers.add_parser(
        "review", help="serve the local side-by-side human review interface"
    )
    review.add_argument("task_dir", type=Path)
    review.add_argument("--port", type=int, default=0)
    review.add_argument("--no-browser", action="store_true")

    finalize = subparsers.add_parser(
        "finalize-diagram", help="export the confirmed review snapshot and open clean.drawio"
    )
    finalize.add_argument("task_dir", type=Path)
    finalize.add_argument("--no-open", action="store_true")
    return parser


def _status_summary(data: dict[str, object]) -> dict[str, object]:
    graph = data.get("graph", data.get("diagram"))
    if isinstance(graph, dict):
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        return {
            "task_id": data.get("task_id"),
            "status": data.get("status"),
            "conversion_state": data.get("conversion_state", "not_run"),
            "content_kind": data.get("content_kind"),
            "node_count": len(nodes) if isinstance(nodes, list) else 0,
            "edge_count": len(edges) if isinstance(edges, list) else 0,
            "unresolved_count": len(unresolved_items(data)),
            "human_review": data.get("human_review"),
            "outputs": data.get("outputs", {}),
        }
    pages = data["pages"]
    assert isinstance(pages, list)

    def stage_state(page: dict[str, object], stage: str) -> str:
        value = page.get(stage)
        if isinstance(value, dict):
            return str(value.get("state", "unknown"))
        return str(value)

    stages = (
        "quality_gate",
        "cleanup",
        "text_ocr",
        "formula_detection",
        "formula_ocr",
        "human_review",
    )
    stage_counts = {
        stage: {
            state: sum(1 for page in pages if stage_state(page, stage) == state)
            for state in sorted({stage_state(page, stage) for page in pages})
        }
        for stage in stages
    }
    return {
        "task_id": data.get("task_id"),
        "status": data.get("status"),
        "conversion_state": data.get("conversion_state", "not_run"),
        "mode": data["mode"],
        "page_count": data["page_count"],
        "stages": stage_counts,
        "outputs": data.get("outputs", {}),
    }


def _conversion_routes(
    inputs: Sequence[Path],
    *,
    requested_kind: str | None,
) -> tuple[list[Path], list[RouteDecision]]:
    sources = resolve_source_files(inputs)
    if requested_kind is not None:
        return sources, [route_content(source, requested_kind=requested_kind) for source in sources]
    if any(source.suffix.lower() == ".pdf" for source in sources):
        if len(sources) != 1 or sources[0].suffix.lower() != ".pdf":
            raise TaskError(
                "auto content routing cannot mix PDFs and images; choose explicit "
                "--content-type document or convert them separately"
            )
        return sources, [route_content(sources[0], requested_kind="document")]
    return sources, [route_content(source) for source in sources]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-task":
            created = create_task(
                name=args.name,
                mode=args.mode,
                inputs=args.inputs,
                output_root=args.output_root,
            )
            print(created)
            return 0
        if args.command == "convert":
            first_input = args.inputs[0].expanduser().resolve()
            default_name = first_input.stem if first_input.is_file() else first_input.name
            name = args.name or default_name
            output = args.output
            if output is None:
                output = first_input.parent / f"{default_name}-ocr"
            requested_kind = None if args.content_type == "auto" else args.content_type
            if requested_kind is None and args.mode != "auto":
                requested_kind = "document"
            sources, routes = _conversion_routes(
                args.inputs,
                requested_kind=requested_kind,
            )
            uncertain = [
                source.name
                for source, route in zip(sources, routes, strict=True)
                if route.confirmation_required
            ]
            if uncertain:
                raise TaskError(
                    "content route is uncertain for "
                    + ", ".join(uncertain)
                    + "; choose --content-type document, flowchart, mindmap, or architecture"
                )
            kinds = {route.kind for route in routes}
            if len(kinds) != 1:
                evidence = ", ".join(
                    f"{source.name}={route.kind.value}"
                    for source, route in zip(sources, routes, strict=True)
                )
                raise TaskError(
                    "mixed content routes in one conversion are not combined; "
                    f"convert each image independently ({evidence})"
                )
            route = routes[0]
            if route.kind is not ContentKind.DOCUMENT:
                if len(sources) != 1:
                    raise TaskError(
                        "diagram conversion accepts exactly one image; convert folder "
                        "images independently"
                    )
                first_input = sources[0]
                if args.diagram_fixture is not None:
                    result = convert_diagram_fixture(
                        fixture=args.diagram_fixture,
                        output_dir=output,
                        original=first_input,
                        content_kind=route.kind.value,
                    )
                else:
                    with tempfile.TemporaryDirectory() as directory:
                        cleaned = Path(directory) / "page-001.png"
                        prepared = preprocess_photo(first_input, cleaned)
                        graph_source = recognize_diagram(
                            cleaned,
                            source_path=first_input,
                            content_kind=route.kind.value,
                            languages=args.language,
                        ).to_dict()
                        preprocess_evidence = prepared.to_dict()
                        preprocess_evidence["output"] = "cleaned/page-001.png"
                        preprocess_evidence["provenance"]["output"] = (
                            "cleaned/page-001.png"
                        )
                        graph_source["metadata"]["photo_preprocess"] = (
                            preprocess_evidence
                        )
                        result = convert_diagram_fixture(
                            fixture=graph_source,
                            output_dir=output,
                            original=first_input,
                            cleaned=cleaned,
                            content_kind=route.kind.value,
                            recognition_scope="opencv_standard_shapes_v1",
                        )
                print(
                    json.dumps(
                        {
                            "status": result.status,
                            "output_dir": str(result.output_dir),
                            "content_route": route.to_dict(),
                            "clean_drawio": str(result.clean_drawio),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            selected = select_engine(args.engine, languages=args.language)
            result = convert_document(
                inputs=args.inputs,
                output_dir=output,
                name=name,
                mode=args.mode,
                engine=selected,
                force_quality=args.force_quality,
                generate_pdfs=not args.no_pdf,
            )
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "output_dir": str(result.output_dir),
                        "page_count": result.page_count,
                        "warnings": list(result.warnings),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "status":
            print(
                json.dumps(
                    _status_summary(load_status(args.task_dir)),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "engines":
            print(
                json.dumps(
                    {"document_engines": discover_engines(), "unimernet": unimernet_status()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "review":
            server = ReviewServer(args.task_dir, port=args.port)
            print(server.url, flush=True)
            if not args.no_browser:
                webbrowser.open(server.url)
            server.serve_forever()
            return 0
        if args.command == "finalize-diagram":
            result = finalize_diagram_package(args.task_dir)
            if not args.no_open and sys.platform == "darwin":
                subprocess.run(
                    ["open", "-a", "draw.io", str(result.clean_drawio)],
                    check=False,
                )
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "output_dir": str(result.output_dir),
                        "clean_drawio": str(result.clean_drawio),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
    except (
        DiagramRecognitionError,
        EngineError,
        OSError,
        TaskError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2
