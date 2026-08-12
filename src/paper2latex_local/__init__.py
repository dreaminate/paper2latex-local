"""Local-first paper-to-LaTeX conversion."""

from .pipeline import FormulaRegion, PageRecognition, convert_document
from .content import ContentKind, RouteDecision, route_content
from .diagram import DiagramEdge, DiagramGraph, DiagramNode
from .diagram_pipeline import convert_diagram_fixture, finalize_diagram_package
from .photo import PhotoPreprocessResult, preprocess_photo
from .task import MAX_PAGES, TaskError, create_task, load_status

__all__ = [
    "MAX_PAGES",
    "ContentKind",
    "RouteDecision",
    "DiagramEdge",
    "DiagramGraph",
    "DiagramNode",
    "FormulaRegion",
    "PageRecognition",
    "TaskError",
    "convert_document",
    "convert_diagram_fixture",
    "finalize_diagram_package",
    "PhotoPreprocessResult",
    "preprocess_photo",
    "route_content",
    "create_task",
    "load_status",
]
__version__ = "0.3.0"
