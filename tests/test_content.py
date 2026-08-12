from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from paper2latex_local.content import ContentKind, route_content


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (900, 700), "white")
    return image, ImageDraw.Draw(image)


def flowchart_fixture() -> Image.Image:
    image, draw = _canvas()
    draw.rectangle((120, 70, 360, 165), outline="black", width=7)
    draw.rectangle((120, 280, 360, 375), outline="black", width=7)
    draw.polygon(((540, 280), (685, 325), (540, 370), (395, 325)), outline="black", width=7)
    draw.line((240, 165, 240, 280), fill="black", width=7)
    draw.line((360, 327, 395, 327), fill="black", width=7)
    draw.line((685, 325, 790, 325), fill="black", width=7)
    draw.polygon(((790, 325), (762, 309), (762, 341)), fill="black")
    return image


def mindmap_fixture() -> Image.Image:
    image, draw = _canvas()
    center = (440, 350)
    draw.ellipse((365, 285, 515, 415), outline="black", width=8)
    for endpoint in ((140, 120), (180, 350), (160, 580), (740, 130), (760, 570)):
        draw.line((*center, *endpoint), fill="black", width=8)
        x, y = endpoint
        draw.ellipse((x - 52, y - 32, x + 52, y + 32), outline="black", width=7)
    return image


def architecture_fixture() -> Image.Image:
    image, draw = _canvas()
    boxes = ((70, 100, 250, 195), (350, 100, 540, 195), (650, 100, 840, 195), (350, 450, 540, 545), (650, 450, 840, 545))
    for box in boxes:
        draw.rectangle(box, outline="black", width=7)
    for start, end in (((250, 148), (350, 148)), ((540, 148), (650, 148)), ((445, 195), (445, 450)), ((540, 498), (650, 498))):
        draw.line((*start, *end), fill="black", width=7)
    draw.line((445, 300, 760, 300), fill="black", width=7)
    draw.line((760, 300, 760, 450), fill="black", width=7)
    return image


def document_fixture() -> Image.Image:
    image, draw = _canvas()
    font = ImageFont.load_default(size=30)
    for index, text in enumerate(("A theorem about functions and limits", "Assume x is positive and let y equal x squared.", "The proof follows by applying the definition.", "Therefore the result holds for every value.")):
        draw.text((70, 100 + index * 115), text, fill="black", font=font)
    return image


class ContentRoutingTests(unittest.TestCase):
    def test_explicit_requests_are_honoured(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        for requested in ("document", "flowchart", "mindmap", "architecture"):
            decision = route_content(image, requested_kind=requested)
            self.assertEqual(decision.kind.value, requested)
            self.assertEqual(decision.confidence, 1.0)
            self.assertFalse(decision.confirmation_required)
        self.assertEqual(route_content(image, requested_kind=ContentKind.FLOWCHART).kind, ContentKind.FLOWCHART)

    def test_auto_and_none_are_equivalent(self) -> None:
        image = document_fixture()
        self.assertEqual(route_content(image).to_dict(), route_content(image, "auto").to_dict())

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional")
    def test_obvious_diagram_kinds(self) -> None:
        self.assertEqual(route_content(flowchart_fixture()).kind, ContentKind.FLOWCHART)
        self.assertEqual(route_content(mindmap_fixture()).kind, ContentKind.MINDMAP)
        self.assertEqual(route_content(architecture_fixture()).kind, ContentKind.ARCHITECTURE)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional")
    def test_document_and_ambiguous_are_conservative(self) -> None:
        document = route_content(document_fixture())
        self.assertEqual(document.kind, ContentKind.DOCUMENT)
        blank = route_content(Image.new("RGB", (900, 700), "white"))
        self.assertEqual(blank.kind, ContentKind.UNCERTAIN)
        self.assertTrue(blank.confirmation_required)

    def test_path_input_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            document_fixture().save(path)
            decision = route_content(path)
            self.assertIn(decision.kind, tuple(ContentKind))


if __name__ == "__main__":
    unittest.main()
