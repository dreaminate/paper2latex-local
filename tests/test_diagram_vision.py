from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from paper2latex_local.diagram_vision import recognize_diagram


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional")
class DiagramVisionTests(unittest.TestCase):
    def test_synthetic_boxes_and_connector_become_review_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.png"
            image = Image.new("RGB", (900, 900), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((260, 90, 640, 250), outline="black", width=9)
            draw.rectangle((260, 590, 640, 750), outline="black", width=9)
            draw.line((450, 250, 450, 590), fill="black", width=8)
            image.save(path)
            with patch(
                "paper2latex_local.diagram_vision._ocr_label",
                side_effect=(("开始", 0.9), ("结束", 0.9)),
            ):
                graph = recognize_diagram(path)
            self.assertEqual(len(graph.nodes), 2)
            self.assertGreaterEqual(len(graph.edges), 1)
            self.assertTrue(any("uncertain_direction" in edge.review_flags for edge in graph.edges))
            self.assertEqual(graph.metadata["accuracy"], "unverified_real_handwriting")


if __name__ == "__main__":
    unittest.main()
