from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from pypdf import PdfReader

from paper2latex_local.diagram_pipeline import convert_diagram_fixture


class DiagramPipelineTests(unittest.TestCase):
    def test_fixture_generates_both_editable_layouts_and_review_gate(self) -> None:
        fixture = {
            "nodes": [
                {"id": "start", "label": "开始", "kind": "start_end", "geometry": [20, 20, 100, 50]},
                {"id": "calc", "label": "计算", "latex": r"x^2+1", "geometry": [250, 220, 120, 60]},
            ],
            "edges": [{"id": "e", "source": "start", "target": "calc"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cleaned = root / "cleaned.png"
            cleaned.write_bytes(b"derived image")
            output = root / "task"
            with patch("paper2latex_local.diagram_pipeline._drawio_command", return_value=None):
                result = convert_diagram_fixture(
                    fixture=fixture,
                    output_dir=output,
                    cleaned=cleaned,
                    recognition_scope="opencv_standard_shapes_v1",
                )
            self.assertEqual(result.status, "needs_review")
            for layout in ("faithful", "clean"):
                self.assertEqual(ET.parse(output / f"diagram/{layout}.drawio").getroot().tag, "mxfile")
                self.assertEqual(len(PdfReader(output / f"diagram/{layout}.pdf").pages), 1)
                self.assertTrue((output / f"diagram/{layout}.mmd").is_file())
                self.assertTrue((output / f"diagram/{layout}.svg").is_file())
            review = json.loads((output / "diagram-review.json").read_text(encoding="utf-8"))
            self.assertEqual(review["task_id"], output.name)
            self.assertEqual(review["recognition_scope"], "opencv_standard_shapes_v1")
            self.assertEqual((output / review["cleaned"]).read_bytes(), b"derived image")
            self.assertIn("real_handwritten_diagram_accuracy_unverified", review["warnings"])
            faithful = (output / "diagram/faithful.drawio").read_text(encoding="utf-8")
            clean = (output / "diagram/clean.drawio").read_text(encoding="utf-8")
            self.assertNotEqual(faithful, clean)


if __name__ == "__main__":
    unittest.main()
