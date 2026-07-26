"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .engines import discover_engines
from .task import MODES, TaskError, create_task, load_status


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

    status = subparsers.add_parser("status", help="show manifest-backed task status")
    status.add_argument("task_dir", type=Path)

    subparsers.add_parser(
        "engines", help="show candidate engine availability; not integration status"
    )
    return parser


def _status_summary(data: dict[str, object]) -> dict[str, object]:
    pages = data["pages"]
    assert isinstance(pages, list)
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
            state: sum(1 for page in pages if page.get(stage) == state)
            for state in sorted({str(page.get(stage)) for page in pages})
        }
        for stage in stages
    }
    return {
        "task_id": data["task_id"],
        "mode": data["mode"],
        "page_count": data["page_count"],
        "stages": stage_counts,
        "outputs": data.get("outputs", {}),
    }


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
            print(json.dumps(discover_engines(), ensure_ascii=False, indent=2))
            return 0
    except (OSError, TaskError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2
