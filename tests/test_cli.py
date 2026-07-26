from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from paper2latex_local.cli import main


class CliTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
