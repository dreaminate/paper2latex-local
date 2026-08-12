"""Canonical, local-first diagram graphs and deterministic layouts.

This module deliberately exposes a synthetic JSON fixture parser rather than
claiming general handwritten-diagram recognition.  A future OCR engine can
produce the same :class:`DiagramGraph` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class DiagramError(ValueError):
    """Base error for invalid graph data or export requests."""


class DiagramExportError(DiagramError):
    """Raised when a graph cannot be exported under the requested gate."""


class NodeKind(str, Enum):
    PROCESS = "process"
    DECISION = "decision"
    START_END = "start_end"
    DOCUMENT = "document"
    GROUP = "group"
    NOTE = "note"
    UNKNOWN = "unknown"


class EdgeKind(str, Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    ASSOCIATION = "association"


def _enum_value(value: Any, enum_type: type[Enum], default: Enum) -> str:
    if value is None:
        return str(default.value)
    if isinstance(value, enum_type):
        return str(value.value)
    candidate = str(value).strip().lower()
    try:
        return str(enum_type(candidate).value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise DiagramError(f"unknown {enum_type.__name__}: {value!r}; expected {allowed}") from error


@dataclass(frozen=True)
class Point:
    """A two-dimensional point in diagram coordinates."""

    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Sequence[float]) -> "Point":
        if isinstance(data, Mapping):
            return cls(float(data["x"]), float(data["y"]))
        return cls(float(data[0]), float(data[1]))


@dataclass(frozen=True)
class Rect:
    """Node rectangle geometry (origin at top-left)."""

    x: float = 0.0
    y: float = 0.0
    width: float = 120.0
    height: float = 60.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise DiagramError("node geometry width and height must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Sequence[float] | None) -> "Rect":
        if data is None:
            return cls()
        if isinstance(data, Mapping):
            return cls(
                x=float(data.get("x", 0.0)),
                y=float(data.get("y", 0.0)),
                width=float(data.get("width", 120.0)),
                height=float(data.get("height", 60.0)),
            )
        values = list(data)
        if len(values) != 4:
            raise DiagramError("rectangle geometry must contain x, y, width, height")
        return cls(*(float(value) for value in values))

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)


# Concise public alias for callers that do not need to distinguish node
# rectangles from edge geometry.
Geometry = Rect


@dataclass(frozen=True)
class EdgeGeometry:
    """Optional bend points and label position for an edge."""

    points: tuple[Point, ...] = ()
    label_position: Point | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"points": [point.to_dict() for point in self.points]}
        if self.label_position is not None:
            data["label_position"] = self.label_position.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "EdgeGeometry":
        if data is None:
            return cls()
        if isinstance(data, Mapping):
            points = tuple(Point.from_dict(item) for item in data.get("points", ()))
            label = data.get("label_position")
            return cls(points=points, label_position=Point.from_dict(label) if label else None)
        return cls(points=tuple(Point.from_dict(item) for item in data))


def _provenance_value(value: Any) -> Any:
    """Make provenance deterministic and JSON serialisable without guessing."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _provenance_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_provenance_value(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class DiagramNode:
    id: str
    label: str
    geometry: Rect = field(default_factory=Rect)
    kind: NodeKind = NodeKind.PROCESS
    latex: str | None = None
    confidence: float = 1.0
    provenance: Any = None
    excluded: bool = False
    crossed_out: bool = False
    review_flags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise DiagramError("node id must not be empty")
        if not isinstance(self.geometry, Rect):
            object.__setattr__(self, "geometry", Rect.from_dict(self.geometry))
        object.__setattr__(self, "kind", NodeKind(_enum_value(self.kind, NodeKind, NodeKind.PROCESS)))
        object.__setattr__(self, "review_flags", tuple(str(flag) for flag in self.review_flags))
        object.__setattr__(self, "provenance", _provenance_value(self.provenance))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise DiagramError("node confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "metadata", _provenance_value(self.metadata) or {})

    @property
    def unresolved(self) -> bool:
        return bool(self.review_flags)

    @property
    def review_required(self) -> bool:
        return self.unresolved

    @property
    def editable_latex(self) -> str | None:
        return self.latex

    @property
    def x(self) -> float:
        return self.geometry.x

    @property
    def y(self) -> float:
        return self.geometry.y

    @property
    def width(self) -> float:
        return self.geometry.width

    @property
    def height(self) -> float:
        return self.geometry.height

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "geometry": self.geometry.to_dict(),
            "kind": self.kind.value,
            "latex": self.latex,
            "confidence": self.confidence,
            "provenance": _provenance_value(self.provenance),
            "excluded": bool(self.excluded),
            "crossed_out": bool(self.crossed_out),
            "review_flags": list(self.review_flags),
            "metadata": _provenance_value(self.metadata),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int = 0) -> "DiagramNode":
        label = str(data.get("label", data.get("text", "")))
        node_id = str(data.get("id") or deterministic_id("node", index, label))
        geometry_data = data.get("geometry")
        if geometry_data is None:
            geometry_data = {key: data[key] for key in ("x", "y", "width", "height") if key in data}
        return cls(
            id=node_id,
            label=label,
            geometry=Rect.from_dict(geometry_data),
            kind=data.get("kind", NodeKind.PROCESS),
            latex=data.get("latex", data.get("editable_latex")),
            confidence=float(data.get("confidence", 1.0)),
            provenance=data.get("provenance"),
            excluded=bool(data.get("excluded", False)),
            crossed_out=bool(data.get("crossed_out", data.get("crossedOut", False))),
            review_flags=tuple(data.get("review_flags", data.get("reviewFlags", ()))),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class DiagramEdge:
    id: str
    source: str
    target: str
    label: str = ""
    geometry: EdgeGeometry = field(default_factory=EdgeGeometry)
    kind: EdgeKind = EdgeKind.DIRECTED
    latex: str | None = None
    confidence: float = 1.0
    provenance: Any = None
    excluded: bool = False
    crossed_out: bool = False
    review_flags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise DiagramError("edge id must not be empty")
        if not str(self.source).strip() or not str(self.target).strip():
            raise DiagramError("edge endpoints must not be empty")
        if not isinstance(self.geometry, EdgeGeometry):
            object.__setattr__(self, "geometry", EdgeGeometry.from_dict(self.geometry))
        object.__setattr__(self, "kind", EdgeKind(_enum_value(self.kind, EdgeKind, EdgeKind.DIRECTED)))
        object.__setattr__(self, "review_flags", tuple(str(flag) for flag in self.review_flags))
        object.__setattr__(self, "provenance", _provenance_value(self.provenance))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise DiagramError("edge confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "metadata", _provenance_value(self.metadata) or {})

    @property
    def unresolved(self) -> bool:
        return bool(self.review_flags)

    @property
    def review_required(self) -> bool:
        return self.unresolved

    @property
    def editable_latex(self) -> str | None:
        return self.latex

    @property
    def points(self) -> tuple[Point, ...]:
        return self.geometry.points

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "geometry": self.geometry.to_dict(),
            "kind": self.kind.value,
            "latex": self.latex,
            "confidence": self.confidence,
            "provenance": _provenance_value(self.provenance),
            "excluded": bool(self.excluded),
            "crossed_out": bool(self.crossed_out),
            "review_flags": list(self.review_flags),
            "metadata": _provenance_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int = 0) -> "DiagramEdge":
        source = str(data.get("source", data.get("from", "")))
        target = str(data.get("target", data.get("to", "")))
        label = str(data.get("label", data.get("text", "")))
        edge_id = str(data.get("id") or deterministic_id("edge", index, f"{source}->{target}:{label}"))
        geometry_data = data.get("geometry", data.get("points"))
        return cls(
            id=edge_id,
            source=source,
            target=target,
            label=label,
            geometry=EdgeGeometry.from_dict(geometry_data),
            kind=data.get("kind", EdgeKind.DIRECTED),
            latex=data.get("latex", data.get("editable_latex")),
            confidence=float(data.get("confidence", 1.0)),
            provenance=data.get("provenance"),
            excluded=bool(data.get("excluded", False)),
            crossed_out=bool(data.get("crossed_out", data.get("crossedOut", False))),
            review_flags=tuple(data.get("review_flags", data.get("reviewFlags", ()))),
            metadata=data.get("metadata", {}),
        )


def deterministic_id(kind: str, index: int, label: str = "") -> str:
    """Return a stable ID for fixture records that do not supply one."""

    digest = hashlib.sha1(f"{kind}\0{index}\0{label}".encode("utf-8")).hexdigest()[:10]
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", str(kind)).strip("-_") or "item"
    return f"{prefix}-{index + 1:03d}-{digest}"


@dataclass(frozen=True)
class DiagramGraph:
    nodes: tuple[DiagramNode, ...] = ()
    edges: tuple[DiagramEdge, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "metadata", _provenance_value(self.metadata) or {})
        self.validate()

    def validate(self, *, include_excluded: bool = True) -> "DiagramGraph":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise DiagramError("node IDs must be unique")
        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise DiagramError("edge IDs must be unique")
        node_set = set(node_ids)
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise DiagramError(
                    f"edge {edge.id!r} has unknown endpoint(s): {edge.source!r}, {edge.target!r}"
                )
        return self

    @property
    def unresolved(self) -> bool:
        return any(node.unresolved for node in self.nodes) or any(edge.unresolved for edge in self.edges)

    def active_nodes(self, *, include_excluded: bool = False) -> tuple[DiagramNode, ...]:
        if include_excluded:
            return self.nodes
        return tuple(node for node in self.nodes if not (node.excluded or node.crossed_out))

    def active_edges(self, *, include_excluded: bool = False) -> tuple[DiagramEdge, ...]:
        active_ids = {node.id for node in self.active_nodes(include_excluded=include_excluded)}
        return tuple(
            edge
            for edge in self.edges
            if (include_excluded or not (edge.excluded or edge.crossed_out))
            and edge.source in active_ids
            and edge.target in active_ids
        )

    def assert_exportable(self, *, final: bool = False) -> None:
        self.validate()
        if not final:
            return
        unresolved_nodes = [node.id for node in self.nodes if node.unresolved]
        unresolved_edges = [edge.id for edge in self.edges if edge.unresolved]
        if unresolved_nodes or unresolved_edges:
            details = []
            if unresolved_nodes:
                details.append(f"nodes={','.join(unresolved_nodes)}")
            if unresolved_edges:
                details.append(f"edges={','.join(unresolved_edges)}")
            raise DiagramExportError("final export rejected unresolved review markers (" + "; ".join(details) + ")")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "metadata": _provenance_value(self.metadata),
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.id)],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda item: item.id)],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagramGraph":
        raw_nodes = tuple(data.get("nodes", ()))
        nodes = tuple(DiagramNode.from_dict(item, index=index) for index, item in enumerate(raw_nodes))
        node_ids_by_index = {index: node.id for index, node in enumerate(nodes)}
        raw_edges: list[Mapping[str, Any]] = []
        for item in data.get("edges", ()):
            edge_data = dict(item)
            for endpoint in ("source", "target", "from", "to"):
                value = edge_data.get(endpoint)
                if isinstance(value, int) and value in node_ids_by_index:
                    edge_data[endpoint] = node_ids_by_index[value]
            raw_edges.append(edge_data)
        edges = tuple(DiagramEdge.from_dict(item, index=index) for index, item in enumerate(raw_edges))
        return cls(
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {}),
            schema_version=int(data.get("schema_version", 1)),
        )

    @classmethod
    def from_json(cls, value: str | bytes | Path) -> "DiagramGraph":
        if isinstance(value, Path):
            return cls.from_dict(json.loads(value.read_text(encoding="utf-8")))
        if isinstance(value, str) and not value.lstrip().startswith(("{", "[")) and Path(value).is_file():
            return cls.from_dict(json.loads(Path(value).read_text(encoding="utf-8")))
        return cls.from_dict(json.loads(value))


def parse_diagram_fixture(fixture: str | Path | Mapping[str, Any]) -> DiagramGraph:
    """Parse an explicit JSON diagram fixture deterministically.

    ``fixture`` may be a JSON object, a JSON string, or a path to a JSON file.
    This is a synthetic extraction path for tests and local iteration; it does
    not perform handwriting recognition.
    """

    if isinstance(fixture, Mapping):
        data = fixture
    elif isinstance(fixture, Path) or (
        isinstance(fixture, str)
        and not fixture.lstrip().startswith(("{", "["))
        and Path(fixture).is_file()
    ):
        data = json.loads(Path(fixture).read_text(encoding="utf-8"))
    else:
        data = json.loads(str(fixture))
    if not isinstance(data, Mapping):
        raise DiagramError("diagram fixture must contain a JSON object")
    return DiagramGraph.from_dict(data)


extract_diagram = parse_diagram_fixture
extract_synthetic_diagram = parse_diagram_fixture


def _copy_graph(graph: DiagramGraph, nodes: Iterable[DiagramNode], edges: Iterable[DiagramEdge]) -> DiagramGraph:
    return DiagramGraph(tuple(nodes), tuple(edges), metadata=graph.metadata, schema_version=graph.schema_version)


def faithful_layout(graph: DiagramGraph, *, include_excluded: bool = True) -> DiagramGraph:
    """Keep supplied geometry exactly; only filter crossed-out records when asked."""

    nodes = graph.active_nodes(include_excluded=include_excluded)
    node_ids = {node.id for node in nodes}
    edges = tuple(edge for edge in graph.edges if edge.source in node_ids and edge.target in node_ids and (include_excluded or not (edge.excluded or edge.crossed_out)))
    return _copy_graph(graph, nodes, edges)


def _clean_levels(graph: DiagramGraph, node_ids: set[str]) -> dict[str, int]:
    incoming: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in graph.edges:
        if edge.source in node_ids and edge.target in node_ids and edge.kind is not EdgeKind.UNDIRECTED:
            incoming[edge.target].append(edge.source)
    levels: dict[str, int] = {}
    remaining = set(node_ids)
    while remaining:
        ready = sorted(node_id for node_id in remaining if all(parent in levels for parent in incoming[node_id]))
        if not ready:
            # Break cycles predictably at the lexicographically smallest node.
            ready = [min(remaining)]
        for node_id in ready:
            levels[node_id] = max((levels[parent] + 1 for parent in incoming[node_id] if parent in levels), default=0)
            remaining.remove(node_id)
    return levels


def clean_layout(
    graph: DiagramGraph,
    *,
    include_excluded: bool = False,
    horizontal_gap: float = 80.0,
    vertical_gap: float = 80.0,
) -> DiagramGraph:
    """Place nodes in deterministic top-to-bottom levels for readable diagrams."""

    nodes = graph.active_nodes(include_excluded=include_excluded)
    node_ids = {node.id for node in nodes}
    levels = _clean_levels(graph, node_ids)
    by_level: dict[int, list[DiagramNode]] = {}
    for node in nodes:
        by_level.setdefault(levels[node.id], []).append(node)
    placed: dict[str, DiagramNode] = {}
    for level in sorted(by_level):
        column_x = 40.0
        for node in sorted(by_level[level], key=lambda item: item.id):
            placed[node.id] = replace(
                node,
                geometry=Rect(column_x, 40.0 + level * (60.0 + vertical_gap), node.width, node.height),
            )
            column_x += node.width + horizontal_gap
    # The supplied bend points no longer describe clean geometry; regenerate
    # straight center-to-center paths for deterministic exporters.
    edges: list[DiagramEdge] = []
    for edge in graph.edges:
        if edge.source not in placed or edge.target not in placed:
            continue
        source = placed[edge.source].geometry.center
        target = placed[edge.target].geometry.center
        edges.append(replace(edge, geometry=EdgeGeometry(points=(source, target))))
    return _copy_graph(graph, (placed[node_id] for node_id in sorted(placed)), edges)


def layout_graph(graph: DiagramGraph, mode: str = "faithful", *, include_excluded: bool | None = None) -> DiagramGraph:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"faithful", "source", "original"}:
        return faithful_layout(graph, include_excluded=True if include_excluded is None else include_excluded)
    if normalized in {"clean", "top_to_bottom", "topdown"}:
        return clean_layout(graph, include_excluded=False if include_excluded is None else include_excluded)
    raise DiagramError("layout mode must be 'faithful' or 'clean'")


__all__ = [
    "DiagramError",
    "DiagramExportError",
    "NodeKind",
    "EdgeKind",
    "Point",
    "Rect",
    "Geometry",
    "EdgeGeometry",
    "DiagramNode",
    "DiagramEdge",
    "DiagramGraph",
    "deterministic_id",
    "parse_diagram_fixture",
    "extract_diagram",
    "extract_synthetic_diagram",
    "faithful_layout",
    "clean_layout",
    "layout_graph",
]
