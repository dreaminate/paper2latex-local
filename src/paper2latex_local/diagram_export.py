"""Deterministic editable diagram exporters.

The renderers here are intentionally dependency-free.  They operate on the
canonical graph in :mod:`paper2latex_local.diagram`; no Graphviz, Mermaid CLI,
or handwriting model is invoked.
"""

from __future__ import annotations

from dataclasses import replace
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from .diagram import (
    DiagramEdge,
    DiagramExportError,
    DiagramGraph,
    DiagramNode,
    EdgeKind,
    EdgeGeometry,
    NodeKind,
    Point,
    Rect,
    layout_graph,
)


def _prepared_graph(
    graph: DiagramGraph,
    *,
    layout: str,
    include_excluded: bool,
    final: bool,
) -> DiagramGraph:
    graph.assert_exportable(final=final)
    return layout_graph(graph, layout, include_excluded=include_excluded)


def _edge_points(edge: DiagramEdge, nodes: dict[str, DiagramNode]) -> tuple[Point, ...]:
    if edge.geometry.points:
        return edge.geometry.points
    return (nodes[edge.source].geometry.center, nodes[edge.target].geometry.center)


def _metadata_attrs(item: DiagramNode | DiagramEdge, *, draft: bool) -> dict[str, str]:
    attrs = {
        "data-confidence": f"{item.confidence:.6g}",
        "data-excluded": "1" if item.excluded else "0",
        "data-crossed-out": "1" if item.crossed_out else "0",
    }
    if item.latex is not None:
        attrs["data-latex"] = item.latex
        attrs["data-label"] = item.label
    if item.provenance is not None:
        attrs["data-provenance"] = json.dumps(item.provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if item.review_flags:
        attrs["data-review-flags"] = json.dumps(list(item.review_flags), ensure_ascii=False, separators=(",", ":"))
    if item.metadata:
        attrs["data-metadata"] = json.dumps(item.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if draft and item.unresolved:
        attrs["data-review-required"] = "1"
    return attrs


def _node_style(node: DiagramNode) -> str:
    base = "whiteSpace=wrap;html=1;"
    if node.unresolved:
        base += "strokeColor=#b7791f;dashed=1;"
    if node.kind is NodeKind.DECISION:
        return base + "shape=rhombus;"
    if node.kind is NodeKind.START_END:
        return base + "rounded=1;arcSize=50;"
    if node.kind is NodeKind.DOCUMENT:
        return base + "shape=document;"
    if node.kind is NodeKind.NOTE:
        return base + "shape=note;"
    if node.kind is NodeKind.GROUP:
        return base + "rounded=1;dashed=1;"
    return base + "rounded=1;"


def _latex_display(value: str) -> str:
    body = value.strip()
    if body.startswith("$$") and body.endswith("$$"):
        body = body[2:-2].strip()
    elif body.startswith("$") and body.endswith("$"):
        body = body[1:-1].strip()
    return f"$${body}$$"


def _display_label(item: DiagramNode | DiagramEdge) -> str:
    if item.latex is None:
        return item.label
    formula = _latex_display(item.latex)
    body = formula[2:-2].strip()
    label = item.label.strip()
    return formula if label.strip("$").strip() == body else f"{label}\n{formula}"


def to_drawio_xml(
    graph: DiagramGraph,
    *,
    layout: str = "faithful",
    include_excluded: bool = False,
    final: bool = False,
) -> str:
    """Return uncompressed draw.io XML containing real vertex/edge cells."""

    prepared = _prepared_graph(graph, layout=layout, include_excluded=include_excluded, final=final)
    nodes = {node.id: node for node in prepared.nodes}
    root = ET.Element(
        "mxfile",
        {
            "host": "paper2latex-local",
            "compressed": "false",
            "version": "1.0",
        },
    )
    diagram = ET.SubElement(root, "diagram", {"id": "diagram-1", "name": "Diagram"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1200",
            "dy": "800",
            "grid": "1",
            "gridSize": "10",
            "page": "1",
            "math": "1" if any(item.latex for item in (*prepared.nodes, *prepared.edges)) else "0",
        },
    )
    model_root = ET.SubElement(model, "root")
    ET.SubElement(model_root, "mxCell", {"id": "0"})
    ET.SubElement(model_root, "mxCell", {"id": "1", "parent": "0"})

    draft = not final
    for node in sorted(prepared.nodes, key=lambda item: item.id):
        value = _display_label(node)
        attrs = {
            "id": node.id,
            "value": value,
            "style": _node_style(node),
            "vertex": "1",
            "parent": "1",
        }
        attrs.update(_metadata_attrs(node, draft=draft))
        cell = ET.SubElement(model_root, "mxCell", attrs)
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": _fmt(node.x),
                "y": _fmt(node.y),
                "width": _fmt(node.width),
                "height": _fmt(node.height),
                "as": "geometry",
            },
        )

    for edge in sorted(prepared.edges, key=lambda item: item.id):
        value = _display_label(edge)
        attrs = {
            "id": edge.id,
            "value": value,
            "style": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;",
            "edge": "1",
            "parent": "1",
            "source": edge.source,
            "target": edge.target,
        }
        if draft and edge.unresolved:
            attrs["style"] += "strokeColor=#b7791f;dashed=1;"
        if edge.kind is EdgeKind.UNDIRECTED:
            attrs["style"] += "endArrow=none;startArrow=none;"
        elif edge.kind is EdgeKind.ASSOCIATION:
            attrs["style"] += "dashed=1;"
        attrs.update(_metadata_attrs(edge, draft=draft))
        cell = ET.SubElement(model_root, "mxCell", attrs)
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        points = _edge_points(edge, nodes)
        if len(points) > 2:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for point in points[1:-1]:
                ET.SubElement(array, "mxPoint", {"x": _fmt(point.x), "y": _fmt(point.y)})
        if edge.geometry.label_position is not None:
            geometry.set("x", _fmt(edge.geometry.label_position.x))
            geometry.set("y", _fmt(edge.geometry.label_position.y))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def export_drawio(
    graph: DiagramGraph,
    output: str | Path,
    *,
    layout: str = "faithful",
    include_excluded: bool = False,
    final: bool = False,
) -> Path:
    path = Path(output)
    path.write_text(
        to_drawio_xml(graph, layout=layout, include_excluded=include_excluded, final=final),
        encoding="utf-8",
    )
    return path


export_drawio_xml = export_drawio
drawio_xml = to_drawio_xml


def _mermaid_escape(label: str) -> str:
    escaped = html.escape(str(label), quote=True)
    escaped = escaped.replace("|", "&#124;").replace("\r", "").replace("\n", "<br/>")
    return escaped


def _mermaid_ids(nodes: Iterable[DiagramNode]) -> dict[str, str]:
    used: set[str] = set()
    result: dict[str, str] = {}
    for node in sorted(nodes, key=lambda item: item.id):
        candidate = "n_" + re.sub(r"[^A-Za-z0-9_]", "_", node.id)
        if not candidate[2:] or candidate[2].isdigit():
            candidate = "n_id_" + candidate[2:]
        base = candidate
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        result[node.id] = candidate
    return result


def _mermaid_shape(node: DiagramNode, label: str) -> str:
    if node.kind is NodeKind.DECISION:
        return "{" + label + "}"
    if node.kind is NodeKind.START_END:
        return "([" + label + "])"
    if node.kind is NodeKind.DOCUMENT:
        return "[/" + label + "/]"
    if node.kind is NodeKind.NOTE:
        return "[\"" + label + "\"]"
    return "[\"" + label + "\"]"


def to_mermaid(
    graph: DiagramGraph,
    *,
    layout: str = "clean",
    include_excluded: bool = False,
    final: bool = False,
) -> str:
    prepared = _prepared_graph(graph, layout=layout, include_excluded=include_excluded, final=final)
    ids = _mermaid_ids(prepared.nodes)
    lines = ["flowchart TD"]
    for node in sorted(prepared.nodes, key=lambda item: item.id):
        label = _mermaid_escape(_display_label(node))
        lines.append(f"    {ids[node.id]}{_mermaid_shape(node, label)}")
    for edge in sorted(prepared.edges, key=lambda item: item.id):
        left, right = ids[edge.source], ids[edge.target]
        if edge.kind is EdgeKind.UNDIRECTED:
            connector = " --- "
        elif edge.kind is EdgeKind.ASSOCIATION:
            connector = " -.-> "
        else:
            connector = " --> "
        label = _mermaid_escape(_display_label(edge))
        if label:
            if edge.kind is EdgeKind.UNDIRECTED:
                connector = " ---|" + label + "| "
            elif edge.kind is EdgeKind.ASSOCIATION:
                connector = " -.->|" + label + "| "
            else:
                connector = " -->|" + label + "| "
        lines.append(f"    {left}{connector}{right}")
    return "\n".join(lines) + "\n"


def export_mermaid(
    graph: DiagramGraph,
    output: str | Path,
    *,
    layout: str = "clean",
    include_excluded: bool = False,
    final: bool = False,
) -> Path:
    path = Path(output)
    path.write_text(to_mermaid(graph, layout=layout, include_excluded=include_excluded, final=final), encoding="utf-8")
    return path


mermaid = to_mermaid


def _fmt(value: float) -> str:
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _bounds(graph: DiagramGraph) -> tuple[float, float, float, float]:
    if not graph.nodes:
        return (0.0, 0.0, 240.0, 120.0)
    x0 = min(node.x for node in graph.nodes)
    y0 = min(node.y for node in graph.nodes)
    x1 = max(node.x + node.width for node in graph.nodes)
    y1 = max(node.y + node.height for node in graph.nodes)
    return x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0)


def _svg_label(parent: ET.Element, x: float, y: float, text: str, *, size: int = 13, anchor: str = "middle") -> None:
    text_element = ET.SubElement(parent, "text", {"x": _fmt(x), "y": _fmt(y), "text-anchor": anchor, "font-family": "sans-serif", "font-size": str(size), "dominant-baseline": "middle"})
    # Keep output deterministic and valid without guessing font metrics.
    text_element.text = text


def to_svg(
    graph: DiagramGraph,
    *,
    layout: str = "faithful",
    include_excluded: bool = False,
    final: bool = False,
) -> str:
    prepared = _prepared_graph(graph, layout=layout, include_excluded=include_excluded, final=final)
    x0, y0, width, height = _bounds(prepared)
    margin = 30.0
    view_width, view_height = width + margin * 2, height + margin * 2
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "version": "1.1", "width": _fmt(view_width), "height": _fmt(view_height), "viewBox": f"0 0 {_fmt(view_width)} {_fmt(view_height)}"})
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(defs, "marker", {"id": "arrow", "markerWidth": "8", "markerHeight": "8", "refX": "7", "refY": "4", "orient": "auto", "markerUnits": "strokeWidth"})
    ET.SubElement(marker, "path", {"d": "M0,0 L8,4 L0,8 z", "fill": "#334155"})
    ET.SubElement(svg, "rect", {"x": "0", "y": "0", "width": _fmt(view_width), "height": _fmt(view_height), "fill": "#ffffff"})
    metadata = ET.SubElement(svg, "metadata")
    metadata.text = json.dumps(prepared.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    nodes = {node.id: node for node in prepared.nodes}
    edge_group = ET.SubElement(svg, "g", {"id": "edges", "fill": "none", "stroke": "#334155", "stroke-width": "1.5"})
    for edge in sorted(prepared.edges, key=lambda item: item.id):
        points = _edge_points(edge, nodes)
        coords = [(point.x - x0 + margin, point.y - y0 + margin) for point in points]
        if len(coords) == 2:
            tag = "line"
            attrs = {"x1": _fmt(coords[0][0]), "y1": _fmt(coords[0][1]), "x2": _fmt(coords[1][0]), "y2": _fmt(coords[1][1])}
        else:
            tag = "polyline"
            attrs = {"points": " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in coords)}
        attrs["id"] = edge.id
        if edge.kind is not EdgeKind.UNDIRECTED:
            attrs["marker-end"] = "url(#arrow)"
        ET.SubElement(edge_group, tag, attrs)
        if edge.label:
            mid = coords[len(coords) // 2]
            _svg_label(edge_group, mid[0], mid[1] - 8, ("⚠ " if not final and edge.unresolved else "") + edge.label, size=11)
    node_group = ET.SubElement(svg, "g", {"id": "nodes", "font-family": "sans-serif"})
    for node in sorted(prepared.nodes, key=lambda item: item.id):
        x, y = node.x - x0 + margin, node.y - y0 + margin
        attrs = {"id": node.id, "x": _fmt(x), "y": _fmt(y), "width": _fmt(node.width), "height": _fmt(node.height), "fill": "#f8fafc", "stroke": "#334155", "stroke-width": "1.5"}
        if node.unresolved and not final:
            attrs.update({"stroke": "#b7791f", "stroke-dasharray": "5 3"})
        if node.kind is NodeKind.DECISION:
            points = [(x + node.width / 2, y), (x + node.width, y + node.height / 2), (x + node.width / 2, y + node.height), (x, y + node.height / 2)]
            ET.SubElement(node_group, "polygon", {**attrs, "points": " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in points)})
        elif node.kind is NodeKind.START_END:
            ET.SubElement(node_group, "rect", {**attrs, "rx": _fmt(node.height / 2), "ry": _fmt(node.height / 2)})
        else:
            ET.SubElement(node_group, "rect", attrs)
        _svg_label(node_group, x + node.width / 2, y + node.height / 2, _display_label(node))
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="unicode", xml_declaration=True) + "\n"


def export_svg(
    graph: DiagramGraph,
    output: str | Path,
    *,
    layout: str = "faithful",
    include_excluded: bool = False,
    final: bool = False,
) -> Path:
    path = Path(output)
    path.write_text(to_svg(graph, layout=layout, include_excluded=include_excluded, final=final), encoding="utf-8")
    return path


svg = to_svg


def _pdf_text(value: str) -> bytes:
    # Helvetica's built-in WinAnsi encoding cannot represent arbitrary Unicode;
    # replacement keeps the PDF syntactically valid and leaves the editable
    # labels available in the graph/XML/SVG outputs.
    safe = str(value).encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\r", " ").replace("\n", " ").encode("latin-1")


def _pdf_num(value: float) -> str:
    return _fmt(value)


def to_pdf_bytes(
    graph: DiagramGraph,
    *,
    layout: str = "faithful",
    include_excluded: bool = False,
    final: bool = False,
) -> bytes:
    """Render a small valid PDF directly from vector geometry."""

    prepared = _prepared_graph(graph, layout=layout, include_excluded=include_excluded, final=final)
    x0, y0, width, height = _bounds(prepared)
    margin = 30.0
    page_width, page_height = width + margin * 2, height + margin * 2
    nodes = {node.id: node for node in prepared.nodes}
    commands: list[str] = ["q", "1 1 1 rg", f"0 0 {_pdf_num(page_width)} {_pdf_num(page_height)} re", "f", "0 0 0 RG", "1 w"]
    for edge in sorted(prepared.edges, key=lambda item: item.id):
        points = _edge_points(edge, nodes)
        first = points[0]
        commands.append(f"{_pdf_num(first.x - x0 + margin)} {_pdf_num(page_height - (first.y - y0 + margin))} m")
        for point in points[1:]:
            commands.append(f"{_pdf_num(point.x - x0 + margin)} {_pdf_num(page_height - (point.y - y0 + margin))} l")
        commands.append("S")
    for node in sorted(prepared.nodes, key=lambda item: item.id):
        x, y = node.x - x0 + margin, page_height - (node.y - y0 + margin + node.height)
        commands.extend(["0.97 0.98 1 rg", f"{_pdf_num(x)} {_pdf_num(y)} {_pdf_num(node.width)} {_pdf_num(node.height)} re", "B", "0 0 0 rg", "BT", "/F1 10 Tf", f"{_pdf_num(x + 6)} {_pdf_num(y + node.height / 2)} Td", f"({_pdf_text(_display_label(node)).decode('latin-1')}) Tj", "ET"])
    commands.append("Q")
    stream = ("\n".join(commands) + "\n").encode("latin-1", "replace")
    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_pdf_num(page_width)} {_pdf_num(page_height)}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode("ascii"),
        4: b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * 6
    for number in range(1, 6):
        offsets[number] = len(output)
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(bodies[number])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for number in range(1, 6):
        output.extend(f"{offsets[number]:010d} 00000 n \n".encode("ascii"))
    output.extend(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n")
    output.extend(f"{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(output)


def export_pdf(
    graph: DiagramGraph,
    output: str | Path,
    *,
    layout: str = "faithful",
    include_excluded: bool = False,
    final: bool = False,
) -> Path:
    path = Path(output)
    path.write_bytes(to_pdf_bytes(graph, layout=layout, include_excluded=include_excluded, final=final))
    return path


pdf = to_pdf_bytes


__all__ = [
    "to_drawio_xml",
    "drawio_xml",
    "export_drawio",
    "export_drawio_xml",
    "to_mermaid",
    "mermaid",
    "export_mermaid",
    "to_svg",
    "svg",
    "export_svg",
    "to_pdf_bytes",
    "pdf",
    "export_pdf",
    "DiagramExportError",
]
