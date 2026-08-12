from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from paper2latex_local.cli import main
from paper2latex_local.content import ContentKind, RouteDecision


class CliTests(unittest.TestCase):
    def test_auto_routes_each_folder_image_and_rejects_mixed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photos"
            source.mkdir()
            for name in ("one.png", "two.png"):
                Image.new("RGB", (100, 100), "white").save(source / name)

            routes = [
                RouteDecision(ContentKind.DOCUMENT, 0.9, {}, False),
                RouteDecision(ContentKind.FLOWCHART, 0.9, {}, False),
            ]
            stderr = io.StringIO()
            with patch("paper2latex_local.cli.route_content", side_effect=routes):
                with redirect_stderr(stderr):
                    code = main(["convert", str(source), "--no-pdf"])

            self.assertEqual(code, 2)
            self.assertIn("mixed content routes", stderr.getvalue())
            self.assertEqual(stderr.getvalue().count("one.png=document"), 1)
            self.assertEqual(stderr.getvalue().count("two.png=flowchart"), 1)

    def test_diagram_fixture_route_generates_editable_drawio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "diagram.png"
            Image.new("RGB", (1200, 900), "white").save(source)
            fixture = root / "graph.json"
            fixture.write_text(
                json.dumps(
                    {
                        "nodes": [
                            {"id": "a", "label": "开始"},
                            {"id": "b", "label": "结束"},
                        ],
                        "edges": [{"id": "e", "source": "a", "target": "b"}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "diagram-ocr"
            stdout = io.StringIO()
            with patch("paper2latex_local.diagram_pipeline._drawio_command", return_value=None):
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "convert",
                            str(source),
                            "--content-type",
                            "flowchart",
                            "--diagram-fixture",
                            str(fixture),
                            "--output",
                            str(output),
                        ]
                    )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["content_route"]["kind"], "flowchart")
            self.assertTrue((output / "diagram/clean.drawio").is_file())
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["status", str(output)]), 0)
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["content_kind"], "flowchart")
            self.assertEqual(status["node_count"], 2)

    def test_diagram_route_refuses_to_silently_ignore_an_extra_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one.png"
            second = root / "two.png"
            Image.new("RGB", (100, 100), "white").save(first)
            Image.new("RGB", (100, 100), "white").save(second)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "convert",
                        str(first),
                        str(second),
                        "--content-type",
                        "flowchart",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("exactly one image", stderr.getvalue())

    def test_init_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "page.jpg"
            photo.write_bytes(b"photo")
            output = root / "tasks"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "init-task",
                        "--name",
                        "Demo",
                        "--mode",
                        "printed",
                        "--output-root",
                        str(output),
                        str(photo),
                    ]
                )
            self.assertEqual(code, 0)
            created = Path(stdout.getvalue().strip())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["status", str(created)])
            self.assertEqual(code, 0)
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["page_count"], 1)
            self.assertEqual(status["stages"]["text_ocr"], {"not_run": 1})

    def test_reports_error_for_missing_input(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main(
                [
                    "init-task",
                    "--name",
                    "Missing",
                    "--mode",
                    "printed",
                    "missing.jpg",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("input is not a file", stderr.getvalue())

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract is not installed")
    def test_convert_command_runs_a_real_local_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            image = Image.new("RGB", (1600, 2200), "white")
            ImageDraw.Draw(image).text((120, 180), "Local OCR", fill="black")
            image.save(source)
            output = root / "page-ocr"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "convert",
                        str(source),
                        "--name",
                        "Local OCR",
                        "--mode",
                        "printed",
                        "--engine",
                        "tesseract",
                        "--output",
                        str(output),
                        "--force-quality",
                        "--no-pdf",
                    ]
                )

            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "needs_review")
            self.assertEqual(Path(summary["output_dir"]), output.resolve())
            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pages"][0]["engine"]["id"], "tesseract")
            self.assertEqual(manifest["pages"][0]["formula_ocr"], "unsupported")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["status", str(output)])
            self.assertEqual(code, 0)
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["task_id"], output.name)
            self.assertEqual(status["conversion_state"], "completed")
            self.assertEqual(status["stages"]["quality_gate"], {"forced": 1})


if __name__ == "__main__":
    unittest.main()
