from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from paper2latex_local.photo import PhotoError, preprocess_photo


def page_fixture() -> Image.Image:
    image = Image.new("RGB", (900, 650), (35, 35, 35))
    draw = ImageDraw.Draw(image)
    draw.polygon(((160, 80), (760, 40), (820, 550), (120, 605)), fill="white", outline="black")
    draw.line((240, 170, 650, 140), fill="black", width=8)
    draw.line((230, 290, 670, 270), fill="black", width=8)
    return image


class PhotoPreprocessingTests(unittest.TestCase):
    def test_orientation_and_grayscale_output_preserve_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            output = root / "cleaned.png"
            image = Image.new("RGB", (300, 200), "white")
            image.putpixel((30, 40), (0, 0, 0))
            exif = image.getexif()
            exif[274] = 6
            image.save(source, exif=exif)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            result = preprocess_photo(source, output)
            self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(result.source_size, (200, 300))
            self.assertTrue(output.is_file())
            with Image.open(output) as cleaned:
                self.assertEqual(cleaned.mode, "L")
                self.assertEqual(cleaned.size, result.output_size)
            self.assertIn("transform_matrix", result.provenance["page_boundary"])
            self.assertIn("algorithm", result.provenance)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional")
    def test_confident_page_boundary_is_rectified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.jpg"
            output = root / "cleaned.png"
            page_fixture().save(source)
            result = preprocess_photo(source, output)
            self.assertTrue(result.rectified)
            self.assertIsNotNone(result.corners)
            self.assertIsNotNone(result.transform_matrix)
            self.assertEqual(result.provenance["page_boundary"]["state"], "rectify")
            self.assertNotEqual(result.output_size, result.source_size)

    def test_weak_boundary_does_not_invent_a_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plain.jpg"
            output = root / "cleaned.png"
            Image.new("RGB", (900, 650), (170, 170, 170)).save(source)
            result = preprocess_photo(source, output)
            self.assertFalse(result.rectified)
            self.assertIsNone(result.corners)
            self.assertEqual(result.output_size, result.source_size)
            self.assertEqual(result.provenance["page_boundary"]["state"], "not_confident")

    def test_refuses_in_place_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (100, 100), "white").save(source)
            with self.assertRaises(PhotoError):
                preprocess_photo(source, source)


if __name__ == "__main__":
    unittest.main()
