from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter

from paper2latex_local.pipeline import (
    FormulaRegion,
    PageRecognition,
    anchor_formula_crops,
    convert_document,
)
from paper2latex_local.task import TaskError
from paper2latex_local.review_server import ReviewStore
from paper2latex_local.unimernet_engine import FormulaPrediction


def fake_render_pdf(
    markdown: Path,
    output: Path,
    *,
    log_path: Path,
    working_dir: Path | None = None,
) -> None:
    output.write_bytes(b"%PDF-test")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("exit_code: 0\n", encoding="utf-8")


class FakeFormulaEngine:
    engine_id = "fake-formula"
    engine_version = "1.0"
    formula_capable = True

    def recognize(self, pages: list[Path], work_dir: Path) -> dict[Path, PageRecognition]:
        del work_dir
        return {
            page: PageRecognition(
                markdown="The result is $x^2 + y^2 = z^2$.",
                confidence=0.98,
                formula_count=1,
            )
            for page in pages
        }


class FakeHandwritingEngine(FakeFormulaEngine):
    def recognize(self, pages: list[Path], work_dir: Path) -> dict[Path, PageRecognition]:
        del work_dir
        return {
            page: PageRecognition(
                markdown="$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$",
                confidence=0.91,
                formula_count=1,
                suggested_mode="handwritten",
                mode_confidence=0.94,
                formula_regions=(
                    FormulaRegion(
                        bbox=(0.1, 0.1, 0.9, 0.3),
                        latex="\\int_0^1 x^2\\,dx = \\frac{1}{3}",
                    ),
                ),
            )
            for page in pages
        }


class FakePartialEngine(FakeFormulaEngine):
    def recognize(self, pages: list[Path], work_dir: Path) -> dict[Path, PageRecognition]:
        return super().recognize(pages[:1], work_dir)


class FakeMinerUEngine(FakeFormulaEngine):
    engine_id = "mineru"

    def recognize(self, pages: list[Path], work_dir: Path) -> dict[Path, PageRecognition]:
        del work_dir
        return {
            page: PageRecognition(
                markdown="First $x$; second $x$.",
                confidence=0.9,
                formula_count=2,
                formula_regions=(FormulaRegion((0.1, 0.1, 0.3, 0.2), "x"),),
            )
            for page in pages
        }


class ConversionPipelineTests(unittest.TestCase):
    def test_formula_anchors_wrap_math_tokens_not_matching_prose(self) -> None:
        crops = [{"path": "formula-crops/one.png", "latex": "x"}]
        result = anchor_formula_crops("Let x vary; formula is $x$.", crops)
        self.assertIn("Let x vary", result)
        self.assertIn("start -->$x$<!-- paper2latex", result)
        self.assertEqual(crops[0]["markdown_mapping"], "anchored")
        self.assertEqual(crops[0]["markdown_delimiter"], "$")

    def test_unimernet_proposal_updates_only_its_anchored_formula_after_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            Image.new("RGB", (1600, 2200), "white").save(source)
            output = root / "page-ocr"
            prediction = FormulaPrediction(
                source=output / "formula-crops/page-001-formula-001.png",
                latex="y",
            )
            with patch(
                "paper2latex_local.unimernet_engine.UniMERNetEngine.recognize",
                return_value=[prediction],
            ):
                convert_document(
                    inputs=[source],
                    output_dir=output,
                    name="Anchored",
                    mode="handwritten",
                    engine=FakeMinerUEngine(),
                    force_quality=True,
                    generate_pdfs=False,
                )

            draft = (output / "document.md").read_text()
            self.assertEqual(draft.count("$x$"), 2)
            self.assertNotIn("$y$", draft)
            store = ReviewStore(output)
            current = store.current
            crop = current["pages"][0]["formula_crops"][0]
            self.assertEqual(crop["mineru_latex"], "x")
            self.assertEqual(crop["latex"], "y")
            crop["review"] = "confirmed"
            with patch(
                "paper2latex_local.render.render_markdown_pdf",
                side_effect=fake_render_pdf,
            ):
                store.finalize(current)
            final = (output / "document.md").read_text()
            self.assertEqual(final.count("$y$"), 1)
            self.assertEqual(final.count("$x$"), 1)

    def test_conversion_writes_latex_markdown_and_manifest_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            Image.new("RGB", (1600, 2200), "white").save(source)
            output = root / "page-ocr"

            result = convert_document(
                inputs=[source],
                output_dir=output,
                name="Pythagoras",
                mode="printed",
                engine=FakeFormulaEngine(),
                force_quality=True,
                generate_pdfs=False,
            )

            self.assertEqual(result.status, "needs_review")
            markdown = (output / "document.md").read_text(encoding="utf-8")
            self.assertIn("$x^2 + y^2 = z^2$", markdown)

            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["outputs"]["markdown"], "document.md")
            self.assertEqual(manifest["pages"][0]["engine"]["id"], "fake-formula")
            self.assertEqual(manifest["pages"][0]["formula_count"], 1)
            self.assertEqual(manifest["pages"][0]["text_ocr"], "completed")
            self.assertEqual(manifest["pages"][0]["formula_ocr"], "completed")
            self.assertEqual(manifest["pages"][0]["quality_gate"]["state"], "forced")
            self.assertIn(
                "low_detail_or_blur",
                manifest["pages"][0]["quality_gate"]["issues"],
            )

    def test_directory_input_uses_natural_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "photos"
            source_dir.mkdir()
            for name in ("page-10.png", "page-2.png", "page-1.png"):
                Image.new("RGB", (1600, 2200), "white").save(source_dir / name)

            output = root / "photos-ocr"
            convert_document(
                inputs=[source_dir],
                output_dir=output,
                name="Ordered pages",
                mode="printed",
                engine=FakeFormulaEngine(),
                force_quality=True,
                generate_pdfs=False,
            )

            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [page["source_name"] for page in manifest["pages"]],
                ["page-1.png", "page-2.png", "page-10.png"],
            )

    def test_auto_mode_records_a_confident_per_page_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.png"
            Image.new("RGB", (1600, 2200), "white").save(source)
            output = root / "notes-ocr"

            convert_document(
                inputs=[source],
                output_dir=output,
                name="Handwritten notes",
                mode="auto",
                engine=FakeHandwritingEngine(),
                force_quality=True,
                generate_pdfs=False,
            )

            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "auto")
            self.assertEqual(manifest["pages"][0]["mode"], "handwritten")
            self.assertEqual(manifest["pages"][0]["mode_confidence"], 0.94)
            self.assertFalse(manifest["pages"][0]["mode_confirmation_required"])
            crops = manifest["pages"][0]["formula_crops"]
            self.assertEqual(len(crops), 1)
            self.assertTrue((output / crops[0]["path"]).is_file())
            self.assertEqual(crops[0]["latex"], "\\int_0^1 x^2\\,dx = \\frac{1}{3}")

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract is not installed")
    def test_auto_mode_uses_printed_text_evidence_when_engine_has_no_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "printed.png"
            image = Image.new("RGB", (1600, 2200), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=72)
            for line_number in range(8):
                draw.text(
                    (100, 140 + line_number * 180),
                    "Printed mathematics theorem and proof",
                    fill="black",
                    font=font,
                )
            image.save(source)
            output = root / "printed-ocr"

            convert_document(
                inputs=[source],
                output_dir=output,
                name="Printed",
                mode="auto",
                engine=FakeFormulaEngine(),
                force_quality=True,
                generate_pdfs=False,
            )

            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pages"][0]["mode"], "printed")
            self.assertGreaterEqual(manifest["pages"][0]["mode_confidence"], 0.7)
            self.assertFalse(manifest["pages"][0]["mode_confirmation_required"])

    def test_quality_gate_stops_before_leaving_an_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "too-small.png"
            Image.new("RGB", (200, 200), "white").save(source)
            output = root / "too-small-ocr"

            with self.assertRaisesRegex(TaskError, "quality gate"):
                convert_document(
                    inputs=[source],
                    output_dir=output,
                    name="Too small",
                    mode="printed",
                    engine=FakeFormulaEngine(),
                    generate_pdfs=False,
                )

            self.assertFalse(output.exists())

    def test_partial_engine_failure_keeps_success_and_marks_failed_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for page_number in (1, 2):
                source = root / f"page-{page_number}.png"
                Image.new("RGB", (1600, 2200), "white").save(source)
                sources.append(source)
            output = root / "partial-ocr"

            result = convert_document(
                inputs=sources,
                output_dir=output,
                name="Partial",
                mode="printed",
                engine=FakePartialEngine(),
                force_quality=True,
                generate_pdfs=False,
            )

            self.assertEqual(result.status, "needs_review")
            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["conversion_state"], "partial_failed")
            self.assertEqual(manifest["pages"][0]["text_ocr"], "completed")
            self.assertEqual(manifest["pages"][1]["text_ocr"], "failed")
            markdown = (output / "document.md").read_text(encoding="utf-8")
            self.assertIn("$x^2 + y^2 = z^2$", markdown)
            self.assertIn("OCR failed for this page", markdown)

    @unittest.skipUnless(shutil.which("pdftoppm"), "Poppler is not installed")
    def test_pdf_input_is_rasterized_and_counted_by_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "two-pages.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            writer.add_blank_page(width=612, height=792)
            with source.open("wb") as handle:
                writer.write(handle)

            output = root / "two-pages-ocr"
            convert_document(
                inputs=[source],
                output_dir=output,
                name="Two pages",
                mode="printed",
                engine=FakeFormulaEngine(),
                force_quality=True,
                generate_pdfs=False,
            )

            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["page_count"], 2)
            self.assertEqual([page["source_page"] for page in manifest["pages"]], [1, 2])
            self.assertEqual(len(list((output / "original").glob("*.pdf"))), 1)
            self.assertEqual(len(list((output / "cleaned").glob("*.png"))), 2)

    @unittest.skipUnless(
        all(shutil.which(command) for command in ("pandoc", "xelatex", "tesseract")),
        "PDF integration tools are not installed",
    )
    def test_default_conversion_generates_two_valid_pdf_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "theorem.png"
            image = Image.new("RGB", (1600, 2200), "white")
            ImageDraw.Draw(image).text((120, 180), "Pythagorean theorem", fill="black")
            image.save(source)
            output = root / "theorem-ocr"

            convert_document(
                inputs=[source],
                output_dir=output,
                name="Pythagoras",
                mode="printed",
                engine=FakeFormulaEngine(),
                force_quality=True,
            )

            rendered = output / "document.pdf"
            searchable = output / "searchable.pdf"
            self.assertGreaterEqual(len(PdfReader(rendered).pages), 1)
            self.assertEqual(len(PdfReader(searchable).pages), 1)

            manifest = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["outputs"]["rendered_pdf"], "document.pdf")
            self.assertEqual(manifest["outputs"]["searchable_pdf"], "searchable.pdf")


if __name__ == "__main__":
    unittest.main()
