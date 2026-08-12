from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2latex_local.engines import (
    EngineError,
    parse_mineru_formula_regions,
    rewrite_mineru_markdown_assets,
    select_engine,
)


class EngineSelectionTests(unittest.TestCase):
    def test_mineru_markdown_assets_are_copied_to_stable_task_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown_dir = root / "nested"
            markdown_dir.mkdir()
            image_dir = markdown_dir / "images"
            image_dir.mkdir()
            (image_dir / "formula.png").write_bytes(b"png")
            markdown_path = markdown_dir / "page.md"
            rewritten, warnings = rewrite_mineru_markdown_assets(
                "![formula](images/formula.png)",
                markdown_path=markdown_path,
                work_dir=root / "exports/mineru",
                page_stem="page-001",
            )
            self.assertEqual(warnings, ())
            self.assertIn(
                "exports/mineru/assets/page-001/formula.png", rewritten
            )
            self.assertTrue(
                (root / "exports/mineru/assets/page-001/formula.png").is_file()
            )

    def test_empty_pipeline_model_path_is_not_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "mineru.json"
            config.write_text(
                json.dumps({"models-dir": {"pipeline": "", "vlm": ""}}),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"MINERU_TOOLS_CONFIG_JSON": str(config)},
                clear=False,
            ):
                from paper2latex_local.engines import _mineru_model_root

                self.assertIsNone(_mineru_model_root("pipeline"))
                self.assertIsNone(_mineru_model_root("vlm-engine"))

    def test_nested_v2_math_content_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "page-001_content_list_v2.json").write_text(
                json.dumps(
                    [
                        {
                            "type": "formula",
                            "bbox": [100, 200, 900, 400],
                            "content": {"math_content": r"\\frac{1}{x}"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            regions = parse_mineru_formula_regions(root, "page-001")
            self.assertEqual(regions[0].latex, r"\\frac{1}{x}")

    def test_mineru_content_list_formula_boxes_become_crop_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_list = root / "page-001_content_list.json"
            content_list.write_text(
                json.dumps(
                    [
                        {"type": "text", "bbox": [100, 100, 900, 200], "text": "hello"},
                        {
                            "type": "interline_equation",
                            "bbox": [200, 300, 800, 500],
                            "text": "x^2+y^2=z^2",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (root / "page-001_content_list_v2.json").write_text(
                json.dumps(
                    [
                        {
                            "type": "formula",
                            "bbox": [200, 300, 800, 500],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            regions = parse_mineru_formula_regions(root, "page-001")

            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].bbox, (0.2, 0.3, 0.8, 0.5))
            self.assertEqual(regions[0].latex, "x^2+y^2=z^2")

    @unittest.skipUnless(shutil.which("mineru"), "MinerU is not installed")
    def test_explicit_mineru_refuses_an_implicit_model_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_config = Path(directory) / "mineru.json"
            with patch.dict(
                os.environ,
                {"MINERU_TOOLS_CONFIG_JSON": str(missing_config)},
                clear=False,
            ):
                with self.assertRaisesRegex(EngineError, "models are not configured"):
                    select_engine("mineru")


if __name__ == "__main__":
    unittest.main()
