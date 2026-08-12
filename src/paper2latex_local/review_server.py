"""Localhost-only human review for OCR documents and editable diagrams."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, urlparse


HOST = "127.0.0.1"


class ReviewError(ValueError):
    """Raised when a task cannot enter or complete review."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"review state must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _normalise_crossed_out(value: Any) -> Any:
    """Default crossed-out recognition candidates to excluded, but restorable."""

    if isinstance(value, list):
        return [_normalise_crossed_out(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _normalise_crossed_out(item) for key, item in value.items()}
    if result.get("crossed_out") is True and "excluded" not in result:
        result["excluded"] = True
    return result


def unresolved_items(state: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return explicit unresolved markers; never infer acceptance from output files."""

    unresolved: list[dict[str, str]] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{location}/{index}")
            return
        if not isinstance(value, dict):
            return
        identifier = str(value.get("id", value.get("page", location)))
        flags = value.get("review_flags", ())
        if isinstance(flags, list):
            unresolved.extend(
                {"path": location, "id": identifier, "reason": str(flag)}
                for flag in flags
                if str(flag).strip()
            )
        if value.get("mode_confirmation_required") is True:
            unresolved.append(
                {"path": location, "id": identifier, "reason": "mode_confirmation_required"}
            )
        if (
            "page" in value
            and "mode_confirmation_required" in value
            and value.get("mode") not in {"printed", "handwritten"}
        ):
            unresolved.append(
                {"path": location, "id": identifier, "reason": "valid_page_mode_required"}
            )
        if value.get("review") in {"required", "unresolved"}:
            unresolved.append(
                {"path": location, "id": identifier, "reason": str(value["review"])}
            )
        for key, item in value.items():
            if key not in {"review_flags"}:
                visit(item, f"{location}/{key}")

    visit(dict(state), "")
    unique = {
        (item["path"], item["id"], item["reason"]): item for item in unresolved
    }
    return [unique[key] for key in sorted(unique)]


def _path_values(value: Any) -> set[str]:
    candidates: set[str] = set()
    if isinstance(value, list):
        for item in value:
            candidates.update(_path_values(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "original",
                "cleaned",
                "path",
                "faithful_drawio",
                "clean_drawio",
                "faithful_svg",
                "clean_svg",
                "faithful_pdf",
                "clean_pdf",
                "markdown",
                "rendered_pdf",
                "searchable_pdf",
            } and isinstance(item, str):
                candidates.add(item)
            candidates.update(_path_values(item))
    return candidates


def _records_by_id(value: Mapping[str, Any], collection: str) -> dict[str, dict[str, Any]]:
    records = value.get(collection, [])
    return {
        str(item["id"]): item
        for item in records
        if isinstance(item, dict) and "id" in item
    }


def _page_formula_paths(value: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    pages: set[str] = set()
    formulas: set[str] = set()
    for page in value.get("pages", []):
        if not isinstance(page, dict):
            continue
        pages.add(str(page.get("page")))
        formulas.update(
            str(crop.get("path"))
            for crop in page.get("formula_crops", [])
            if isinstance(crop, dict) and crop.get("path")
        )
    return pages, formulas


def _formula_crops(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        crop
        for page in value.get("pages", [])
        if isinstance(page, dict)
        for crop in page.get("formula_crops", [])
        if isinstance(crop, dict)
    ]


def _math_bodies(markdown: str) -> list[str]:
    pattern = re.compile(
        r"\$\$(?P<display>.*?)\$\$|"
        r"(?<!\$)\$(?!\$)(?P<inline>.*?)(?<!\$)\$(?!\$)",
        re.S,
    )
    return [
        (match.group("display") or match.group("inline") or "").strip()
        for match in pattern.finditer(markdown)
    ]


def _unanchored_math_body_counts(markdown: str) -> Counter[str]:
    anchor_pattern = re.compile(
        r"<!-- paper2latex-formula:(?P<anchor>[^>\n]+):start -->.*?"
        r"<!-- paper2latex-formula:(?P=anchor):end -->",
        re.S,
    )
    return Counter(_math_bodies(anchor_pattern.sub("", markdown)))


class ReviewStore:
    """Three-snapshot store: immutable initial, mutable current, final confirmed."""

    def __init__(self, task_dir: Path) -> None:
        self.root = Path(task_dir).expanduser().resolve()
        if not self.root.is_dir():
            raise ReviewError(f"review task directory does not exist: {self.root}")
        self.initial_path = self.root / "review.initial.json"
        self.current_path = self.root / "review.current.json"
        self.final_path = self.root / "review.final.json"
        self.document_initial_path = self.root / "document.initial.md"
        document = self.root / "document.md"
        if document.is_file() and not self.document_initial_path.is_file():
            self.document_initial_path.write_bytes(document.read_bytes())
        if not self.initial_path.is_file():
            source = next(
                (
                    candidate
                    for candidate in (
                        self.root / "diagram-review.json",
                        self.root / "review.json",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            if source is None:
                raise ReviewError(f"review.json not found in {self.root}")
            initial = _normalise_crossed_out(_read_json(source))
            initial.setdefault("task_id", self.root.name)
            if document.is_file():
                from .pipeline import anchor_formula_crops

                markdown = anchor_formula_crops(
                    document.read_text(encoding="utf-8"),
                    _formula_crops(initial),
                )
                initial["document_markdown"] = markdown
                _write_text(document, markdown)
            _write_json(self.initial_path, initial)
        if not self.current_path.is_file():
            _write_json(self.current_path, self.initial)
        else:
            current = self.current
            changed = False
            expected_task_id = str(self.initial.get("task_id", self.root.name))
            if self.initial.get("task_id") is None and current.get("task_id") != expected_task_id:
                current["task_id"] = expected_task_id
                changed = True
            if document.is_file() and "document_markdown" not in current:
                from .pipeline import anchor_formula_crops

                markdown = anchor_formula_crops(
                    document.read_text(encoding="utf-8"),
                    _formula_crops(current),
                )
                current["document_markdown"] = markdown
                _write_text(document, markdown)
                changed = True
            if changed:
                _write_json(self.current_path, current)

    @property
    def initial(self) -> dict[str, Any]:
        return _read_json(self.initial_path)

    @property
    def current(self) -> dict[str, Any]:
        return _read_json(self.current_path)

    @property
    def final(self) -> dict[str, Any] | None:
        return _read_json(self.final_path) if self.final_path.is_file() else None

    def save(
        self,
        state: Mapping[str, Any],
        *,
        export_diagram: bool = True,
        allow_final_snapshot: bool = False,
    ) -> dict[str, Any]:
        if self.final_path.exists() and not allow_final_snapshot:
            raise ReviewError(
                "review is finalized; an explicit reopen workflow is required before saving"
            )
        current = _normalise_crossed_out(deepcopy(dict(state)))
        current["status"] = "needs_review"
        current["human_review"] = "required"
        for page in current.get("pages", []):
            if isinstance(page, dict):
                page["human_review"] = "required"
        self.validate_integrity(current)
        markdown = None
        if self.document_initial_path.is_file():
            markdown_value = current.get("document_markdown")
            if not isinstance(markdown_value, str):
                raise ReviewError("review state removed document_markdown")
            markdown = markdown_value
            manual_formula_tokens: list[tuple[str, str]] = []
            for crop in _formula_crops(current):
                path = str(crop.get("path", crop.get("id", "formula")))
                latex = crop.get("latex")
                mapping = crop.get("markdown_mapping")
                if mapping == "anchored":
                    if crop.get("review") != "confirmed":
                        continue
                    if not isinstance(latex, str) or not latex:
                        raise ReviewError(f"formula {path} has no reviewed LaTeX")
                    anchor = str(crop.get("markdown_anchor", crop.get("id", "")))
                    start = f"<!-- paper2latex-formula:{anchor}:start -->"
                    end = f"<!-- paper2latex-formula:{anchor}:end -->"
                    pattern = re.compile(
                        re.escape(start) + r".*?" + re.escape(end), re.S
                    )
                    body = latex.strip()
                    if body.startswith("$$") and body.endswith("$$"):
                        body = body[2:-2].strip()
                    elif body.startswith("$") and body.endswith("$"):
                        body = body[1:-1].strip()
                    delimiter = str(crop.get("markdown_delimiter", "$$"))
                    replacement = start + delimiter + body + delimiter + end
                    markdown, count = pattern.subn(
                        lambda _: replacement, markdown
                    )
                    if count != 1:
                        raise ReviewError(
                            f"formula {path} lost its stable Markdown anchor"
                        )
                elif crop.get("review") == "confirmed":
                    if mapping != "manual_confirmed":
                        raise ReviewError(
                            f"formula {path} requires manual Markdown placement"
                        )
                    body = latex.strip() if isinstance(latex, str) else ""
                    if body.startswith("$$") and body.endswith("$$"):
                        body = body[2:-2].strip()
                    elif body.startswith("$") and body.endswith("$"):
                        body = body[1:-1].strip()
                    manual_formula_tokens.append((path, body))
            available_tokens = _unanchored_math_body_counts(markdown)
            for path, body in manual_formula_tokens:
                if not body or available_tokens[body] == 0:
                    raise ReviewError(
                        f"formula {path} is not present as a distinct complete math token "
                        "in document_markdown"
                    )
                available_tokens[body] -= 1
            current["document_markdown"] = markdown
        graph_value = current.get("graph", current.get("diagram"))
        if isinstance(graph_value, dict) and export_diagram:
            from .diagram import DiagramGraph
            from .diagram_pipeline import _export_layouts

            _export_layouts(DiagramGraph.from_dict(graph_value), self.root, final=False)
        if markdown is not None:
            from .render import RenderError, render_markdown_pdf

            preview_pdf = self.root / "document.preview.pdf"
            with tempfile.TemporaryDirectory(
                dir=self.root,
                prefix=".review-preview.",
            ) as directory:
                staged_markdown = Path(directory) / "document.md"
                staged_pdf = Path(directory) / "document.preview.pdf"
                staged_log = Path(directory) / "review-preview-pdf.log"
                staged_markdown.write_text(markdown, encoding="utf-8")
                try:
                    render_markdown_pdf(
                        staged_markdown,
                        staged_pdf,
                        log_path=staged_log,
                        working_dir=self.root,
                    )
                except RenderError as error:
                    raise ReviewError(str(error)) from error
                (self.root / "exports").mkdir(exist_ok=True)
                _write_text(
                    self.root / "exports/review-preview-pdf.log",
                    staged_log.read_text(encoding="utf-8"),
                )
                os.replace(staged_pdf, preview_pdf)
            _write_text(self.root / "document.md", markdown)
            current.setdefault("outputs", {})["review_preview_pdf"] = (
                preview_pdf.relative_to(self.root).as_posix()
            )
        _write_json(self.current_path, current)
        if self.final_path.exists():
            self.final_path.unlink()
        return current

    def validate_integrity(self, state: Mapping[str, Any]) -> None:
        """Reject client states that delete or replace recognition candidates."""

        initial = self.initial
        initial_graph = initial.get("graph", initial.get("diagram"))
        current_graph = state.get("graph", state.get("diagram"))
        if isinstance(initial_graph, dict) != isinstance(current_graph, dict):
            raise ReviewError("review state changed task kind")
        expected_task_id = str(initial.get("task_id", self.root.name))
        if state.get("task_id") != expected_task_id:
            raise ReviewError("review state changed immutable task_id")
        for key in ("schema_version", "content_kind", "mode"):
            if key in initial and state.get(key) != initial.get(key):
                raise ReviewError(f"review state changed immutable {key}")
        initial_graph = initial.get("graph", initial.get("diagram"))
        if isinstance(initial_graph, dict):
            if not isinstance(current_graph, dict):
                raise ReviewError("review state removed the canonical graph")
            for collection in ("nodes", "edges"):
                expected = set(_records_by_id(initial_graph, collection))
                actual = set(_records_by_id(current_graph, collection))
                if actual != expected:
                    raise ReviewError(
                        f"review state changed {collection} identity; "
                        "use excluded=true instead of deleting candidates"
                    )
        initial_pages, initial_formulas = _page_formula_paths(initial)
        if initial_pages or initial_formulas:
            current_pages, current_formulas = _page_formula_paths(state)
            if current_pages != initial_pages:
                raise ReviewError("review state changed page identity")
            if current_formulas != initial_formulas:
                raise ReviewError("review state changed formula crop identity")
            initial_by_page = {
                str(page.get("page")): page
                for page in initial.get("pages", [])
                if isinstance(page, dict)
            }
            current_by_page = {
                str(page.get("page")): page
                for page in state.get("pages", [])
                if isinstance(page, dict)
            }
            for page, initial_page in initial_by_page.items():
                current_page = current_by_page[page]
                for key in ("original", "cleaned", "sha256", "source_name", "source_page"):
                    if key in initial_page and current_page.get(key) != initial_page.get(key):
                        raise ReviewError(
                            f"review state changed immutable page {page} {key}"
                        )

    def finalize(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        current = (
            self.save(
                state,
                export_diagram=False,
                allow_final_snapshot=False,
            )
            if state is not None
            else self.current
        )
        self.validate_integrity(current)
        unresolved = unresolved_items(current)
        if unresolved:
            raise ReviewError(json.dumps(unresolved, ensure_ascii=False))
        graph_value = current.get("graph", current.get("diagram"))
        if isinstance(graph_value, dict):
            from .diagram import DiagramError, DiagramGraph

            try:
                DiagramGraph.from_dict(graph_value).assert_exportable(final=True)
            except DiagramError as error:
                raise ReviewError(str(error)) from error
        final = deepcopy(current)
        final["status"] = "finalized"
        final["human_review"] = "confirmed"
        for page in final.get("pages", []):
            if isinstance(page, dict):
                page["human_review"] = "confirmed"
        diagram_graph = None
        if isinstance(graph_value, dict):
            from .diagram_pipeline import _export_layouts

            diagram_graph = DiagramGraph.from_dict(graph_value)
            final["outputs"] = _export_layouts(
                diagram_graph, self.root, final=True
            )
        markdown_path = self.root / "document.md"
        if markdown_path.is_file():
            from .render import RenderError, render_markdown_pdf

            try:
                render_markdown_pdf(
                    markdown_path,
                    self.root / "document.pdf",
                    log_path=self.root / "exports/final-markdown-pdf.log",
                )
            except RenderError as error:
                raise ReviewError(str(error)) from error
        _write_json(self.final_path, final)
        _write_json(self.current_path, final)
        if diagram_graph is not None:
            _write_json(self.root / "diagram-review.json", final)
            _write_text(self.root / "diagram.json", diagram_graph.to_json())
        return final

    def declared_paths(self) -> dict[str, str]:
        values = _path_values(self.initial) | _path_values(self.current)
        known = (
            "document.md",
            "document.pdf",
            "document.preview.pdf",
            "searchable.pdf",
            "diagram/faithful.drawio",
            "diagram/clean.drawio",
            "diagram/faithful.svg",
            "diagram/clean.svg",
            "diagram/faithful.pdf",
            "diagram/clean.pdf",
        )
        values.update(value for value in known if (self.root / value).is_file())
        resolved: dict[str, str] = {}
        for value in sorted(values):
            relative = Path(value)
            if relative.is_absolute():
                continue
            candidate = (self.root / relative).resolve()
            if candidate.is_file() and candidate.is_relative_to(self.root):
                resolved[value] = f"/api/file?path={quote(value)}"
        return resolved

    def file_path(self, relative_value: str) -> Path:
        if relative_value not in self.declared_paths():
            raise ReviewError("path is not declared by the review task")
        candidate = (self.root / relative_value).resolve()
        if not candidate.is_relative_to(self.root):
            raise ReviewError("path escapes the review task")
        return candidate

    def payload(self) -> dict[str, Any]:
        current = self.current
        return {
            "initial": self.initial,
            "current": current,
            "final": self.final,
            "status": "finalized" if self.final is not None else "needs_review",
            "unresolved": unresolved_items(current),
            "paths": self.declared_paths(),
        }


def _handler(store: ReviewStore) -> type[BaseHTTPRequestHandler]:
    html_path = Path(__file__).with_name("review.html")

    class Handler(BaseHTTPRequestHandler):
        server_version = "paper2latex-review/1"

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ReviewError("request body must be a JSON object")
            state = value.get("state", value)
            if not isinstance(state, dict):
                raise ReviewError("review state must be a JSON object")
            return state

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/review.html"}:
                body = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/review":
                self._json(store.payload())
                return
            if parsed.path == "/api/paths":
                self._json(store.declared_paths())
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                relative_value = query.get("path", [""])[0]
                try:
                    path = store.file_path(relative_value)
                except ReviewError as error:
                    self._json({"error": str(error)}, HTTPStatus.FORBIDDEN)
                    return
                body = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"/api/review", "/api/review/save"}:
                    current = store.save(self._body())
                    self._json(
                        {
                            "status": "saved",
                            "current": current,
                            "unresolved": unresolved_items(current),
                        }
                    )
                    return
                if parsed.path == "/api/review/finalize":
                    state = self._body()
                    final = store.finalize(state)
                    clean_drawio = None
                    if (
                        sys.platform == "darwin"
                        and (store.root / "diagram-review.json").is_file()
                    ):
                        clean_drawio = str(store.root / "diagram/clean.drawio")
                        subprocess.run(
                            ["open", "-a", "draw.io", clean_drawio],
                            check=False,
                        )
                    self._json(
                        {
                            "status": "finalized",
                            "final": final,
                            "clean_drawio": clean_drawio,
                        }
                    )
                    return
            except (json.JSONDecodeError, ReviewError) as error:
                status = (
                    HTTPStatus.CONFLICT
                    if parsed.path == "/api/review/finalize"
                    else HTTPStatus.BAD_REQUEST
                )
                self._json({"error": str(error)}, status)
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


class ReviewServer:
    """Background-capable HTTP server that can only bind to localhost."""

    def __init__(self, task_dir: Path, port: int = 0) -> None:
        self.store = ReviewStore(task_dir)
        self.httpd = ThreadingHTTPServer((HOST, port), _handler(self.store))
        self.thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    def start(self) -> "ReviewServer":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def __enter__(self) -> "ReviewServer":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = [
    "HOST",
    "ReviewError",
    "ReviewServer",
    "ReviewStore",
    "unresolved_items",
]
