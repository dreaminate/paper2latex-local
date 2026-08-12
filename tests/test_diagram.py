from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from lxml import etree as lxml_etree
from pypdf import PdfReader

from paper2latex_local.diagram import (
    DiagramEdge,
    DiagramExportError,
    DiagramGraph,
    DiagramNode,
    EdgeGeometry,
    NodeKind,
    Point,
    Rect,
    clean_layout,
    parse_diagram_fixture,
)
from paper2latex_local.diagram_export import (
    export_drawio,
    export_mermaid,
    export_pdf,
    export_svg,
    to_drawio_xml,
    to_mermaid,
)


def sample_graph() -> DiagramGraph:
    return DiagramGraph(
        nodes=(
            DiagramNode(
                "start",
                "Start | now",
                Rect(15, 20, 100, 45),
                kind=NodeKind.START_END,
                latex=r"$x_0$",
                provenance={"fixture": "sample"},
            ),
            DiagramNode("decision", 'Choose "yes"', Rect(250, 180, 130, 60), kind=NodeKind.DECISION),
            DiagramNode("crossed", "Ignore", Rect(500, 30, 100, 40), crossed_out=True),
        ),
        edges=(
            DiagramEdge(
                "start-to-decision",
                "start",
                "decision",
                "go <now>",
                geometry=EdgeGeometry((Point(115, 42), Point(250, 210))),
            ),
            DiagramEdge("crossed-edge", "start", "crossed", crossed_out=True),
        ),
        metadata={"source": "synthetic"},
    )


class DiagramModelTests(unittest.TestCase):
    def test_json_round_trip_and_stable_order(self) -> None:
        graph = sample_graph()
        encoded = graph.to_json()
        self.assertEqual(encoded, graph.to_json())
        decoded = DiagramGraph.from_json(encoded)
        self.assertEqual(decoded.to_dict(), graph.to_dict())
        self.assertEqual(list(json.loads(encoded)), ["edges", "metadata", "nodes", "schema_version"])

    def test_fixture_parser_assigns_deterministic_ids(self) -> None:
        fixture = {"nodes": [{"label": "A"}, {"label": "B"}], "edges": [{"source": "node-001-", "target": "node-002-"}]}
        # IDs can be supplied explicitly when referring to endpoints; absent
        # IDs receive deterministic IDs from record index and label.
        first = parse_diagram_fixture({"nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}], "edges": [{"source": "a", "target": "b"}]})
        second = parse_diagram_fixture(first.to_json())
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.edges[0].source, "a")
        self.assertEqual(fixture["nodes"][0]["label"], "A")

    def test_validation_rejects_duplicate_ids_and_unknown_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "node IDs"):
            DiagramGraph((DiagramNode("same", "a"), DiagramNode("same", "b")), ())
        with self.assertRaisesRegex(ValueError, "globally unique"):
            DiagramGraph(
                (DiagramNode("same", "a"), DiagramNode("target", "b")),
                (DiagramEdge("same", "same", "target"),),
            )
        with self.assertRaisesRegex(ValueError, "reserved"):
            DiagramGraph((DiagramNode("0", "a"),), ())
        with self.assertRaisesRegex(ValueError, "reserved"):
            DiagramGraph(
                (DiagramNode("source", "a"), DiagramNode("target", "b")),
                (DiagramEdge("1", "source", "target"),),
            )
        with self.assertRaisesRegex(ValueError, "unknown endpoint"):
            DiagramGraph((DiagramNode("a", "a"),), (DiagramEdge("e", "a", "missing"),))

    def test_clean_layout_is_deterministic_and_differs_from_faithful(self) -> None:
        graph = sample_graph()
        clean = clean_layout(graph)
        self.assertNotEqual(
            [(node.id, node.geometry.to_dict()) for node in clean.nodes],
            [(node.id, node.geometry.to_dict()) for node in graph.nodes if not node.crossed_out],
        )
        start = next(node for node in clean.nodes if node.id == "start")
        decision = next(node for node in clean.nodes if node.id == "decision")
        self.assertEqual(start.y, 40.0)
        self.assertGreater(decision.y, start.y)
        self.assertEqual(clean.to_json(), clean_layout(graph).to_json())

    def test_crossed_out_excluded_by_default_and_restorable(self) -> None:
        graph = sample_graph()
        draft = to_drawio_xml(graph)
        restored = to_drawio_xml(graph, include_excluded=True)
        self.assertNotIn('id="crossed"', draft)
        self.assertIn('id="crossed"', restored)

    def test_drawio_has_true_cells_endpoints_and_latex_metadata(self) -> None:
        root = ET.fromstring(to_drawio_xml(sample_graph(), include_excluded=True))
        objects = root.findall(".//object")
        vertices = [obj for obj in objects if obj.find("mxCell").get("vertex") == "1"]
        edges = [obj for obj in objects if obj.find("mxCell").get("edge") == "1"]
        self.assertEqual({obj.get("id") for obj in vertices}, {"start", "decision", "crossed"})
        self.assertEqual(len(edges), 2)
        directed = next(obj for obj in edges if obj.get("id") == "start-to-decision").find("mxCell")
        self.assertEqual(directed.get("source"), "start")
        self.assertEqual(directed.get("target"), "decision")
        formula = next(obj for obj in vertices if obj.get("id") == "start")
        self.assertEqual(formula.get("data-latex"), r"$x_0$")
        self.assertIn(r"$$x_0$$", formula.get("label"))
        self.assertEqual(formula.get("data-label"), "Start | now")
        self.assertEqual(root.find(".//mxGraphModel").get("math"), "1")
        for cell in root.findall(".//mxCell"):
            self.assertFalse(any(name.startswith("data-") for name in cell.attrib))
            if cell.get("vertex") == "1" or cell.get("edge") == "1":
                self.assertIsNone(cell.get("id"))

    def test_drawio_validates_against_official_mxfile_schema(self) -> None:
        schema_path = Path(__file__).parent / "fixtures/drawio-mxfile.xsd"
        schema = lxml_etree.XMLSchema(lxml_etree.parse(str(schema_path)))
        document = lxml_etree.fromstring(to_drawio_xml(sample_graph(), include_excluded=True).encode())
        schema.assertValid(document)

    def test_mermaid_escapes_labels(self) -> None:
        output = to_mermaid(sample_graph())
        self.assertIn("flowchart TD", output)
        self.assertIn("&quot;yes&quot;", output)
        self.assertIn("&#124;", output)
        self.assertIn("&lt;now&gt;", output)
        self.assertNotIn('id="crossed"', output)

    def test_svg_and_pdf_are_parsable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            svg_path = export_svg(sample_graph(), root / "diagram.svg")
            pdf_path = export_pdf(sample_graph(), root / "diagram.pdf")
            self.assertEqual(ET.parse(svg_path).getroot().tag, "{http://www.w3.org/2000/svg}svg")
            self.assertEqual(len(PdfReader(pdf_path).pages), 1)

    def test_final_export_rejects_review_markers(self) -> None:
        graph = DiagramGraph(
            (DiagramNode("a", "Needs review", review_flags=("uncertain-label",)),),
            (),
        )
        with self.assertRaises(DiagramExportError):
            to_drawio_xml(graph, final=True)
        draft = to_drawio_xml(graph)
        self.assertIn("data-review-required=\"1\"", draft)
        self.assertIn("strokeColor=#b7791f", draft)


if __name__ == "__main__":
    unittest.main()
