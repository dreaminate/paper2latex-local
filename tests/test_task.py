from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from paper2latex_local.task import MAX_PAGES, TaskError, create_task, load_status


class CreateTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "tasks"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _photo(self, name: str, content: bytes = b"photo-bytes") -> Path:
        path = self.inputs / name
        path.write_bytes(content)
        return path

    def test_creates_manifest_and_preserves_hash(self) -> None:
        first = self._photo("page one.JPG", b"first")
        second = self._photo("公式.png", b"second")
        created = create_task(
            name="Calculus Notes",
            mode="handwritten",
            inputs=[first, second],
            output_root=self.output,
            now=datetime(2026, 7, 26, 15, 0, tzinfo=UTC),
        )

        self.assertEqual(created.name, "20260726T150000Z-calculus-notes")
        data = load_status(created)
        self.assertEqual(data["page_count"], 2)
        self.assertEqual(data["mode"], "handwritten")
        self.assertEqual(data["pages"][0]["quality_gate"], "not_run")
        self.assertEqual(
            data["pages"][0]["sha256"], hashlib.sha256(b"first").hexdigest()
        )
        self.assertTrue((created / data["pages"][0]["original"]).is_file())
        self.assertIn("OCR has not run", (created / "document.md").read_text())

    def test_rejects_more_than_maximum_pages(self) -> None:
        photos = [self._photo(f"{index}.jpg") for index in range(MAX_PAGES + 1)]
        with self.assertRaisesRegex(TaskError, "at most 50"):
            create_task(
                name="too many",
                mode="printed",
                inputs=photos,
                output_root=self.output,
            )

    def test_rejects_duplicate_input(self) -> None:
        photo = self._photo("same.jpg")
        with self.assertRaisesRegex(TaskError, "duplicate"):
            create_task(
                name="duplicate",
                mode="printed",
                inputs=[photo, photo],
                output_root=self.output,
            )

    def test_rejects_unsupported_suffix(self) -> None:
        text = self._photo("notes.txt")
        with self.assertRaisesRegex(TaskError, "unsupported"):
            create_task(
                name="bad input",
                mode="printed",
                inputs=[text],
                output_root=self.output,
            )

    def test_refuses_to_overwrite_existing_task(self) -> None:
        photo = self._photo("page.jpg")
        now = datetime(2026, 7, 26, 15, 0, tzinfo=UTC)
        create_task(
            name="same",
            mode="printed",
            inputs=[photo],
            output_root=self.output,
            now=now,
        )
        with self.assertRaisesRegex(TaskError, "already exists"):
            create_task(
                name="same",
                mode="printed",
                inputs=[photo],
                output_root=self.output,
                now=now,
            )


if __name__ == "__main__":
    unittest.main()
