from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from paper2latex_local.render import RenderError
from paper2latex_local.review_server import HOST, ReviewError, ReviewServer, ReviewStore
from paper2latex_local.task import load_status
import paper2latex_local.review_server as review_server_module


def request_json(url: str, *, method: str = "GET", value: object | None = None):
    data = None if value is None else json.dumps(value).encode("utf-8")
    request = Request(url, method=method, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


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


class ReviewServerTests(unittest.TestCase):
    def make_task(self, root: Path) -> None:
        (root / "original").mkdir()
        (root / "diagram").mkdir()
        Image.new("RGB", (60, 60), "white").save(root / "original/page.png")
        (root / "diagram/clean.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        (root / "diagram/faithful.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        (root / "review.json").write_text(
            json.dumps(
                {
                    "status": "needs_review",
                    "original": "original/page.png",
                    "graph": {
                        "nodes": [
                            {
                                "id": "a",
                                "label": "maybe",
                                "crossed_out": True,
                                "review_flags": ["uncertain_label"],
                            }
                        ],
                        "edges": [],
                    },
                    "outputs": {
                        "clean_svg": "diagram/clean.svg",
                        "faithful_svg": "diagram/faithful.svg",
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_local_review_save_finalize_and_immutable_initial(self) -> None:
        self.assertEqual(HOST, "127.0.0.1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_task(root)
            with ReviewServer(root) as server:
                _, first = request_json(server.url + "api/review")
                self.assertTrue(first["current"]["graph"]["nodes"][0]["excluded"])
                original_initial = (root / "review.initial.json").read_bytes()
                with self.assertRaises(HTTPError) as caught:
                    request_json(
                        server.url + "api/review/finalize",
                        method="POST",
                        value={"state": first["current"]},
                    )
                self.assertEqual(caught.exception.code, 409)
                state = first["current"]
                state["graph"]["nodes"][0]["label"] = "confirmed"
                state["graph"]["nodes"][0]["review_flags"] = []
                state["graph"]["nodes"][0]["excluded"] = False
                _, saved = request_json(
                    server.url + "api/review/save",
                    method="POST",
                    value={"state": state},
                )
                self.assertEqual(saved["status"], "saved")
                self.assertTrue((root / "diagram/clean.svg").is_file())
                self.assertFalse((root / "review.final.json").exists())
                _, finalized = request_json(
                    server.url + "api/review/finalize",
                    method="POST",
                    value={"state": state},
                )
                self.assertEqual(finalized["status"], "finalized")
                self.assertTrue((root / "review.final.json").is_file())
                self.assertEqual(original_initial, (root / "review.initial.json").read_bytes())
                self.assertFalse((root / "review.history.json").exists())

    def test_file_endpoint_refuses_undeclared_and_traversal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_task(root)
            with ReviewServer(root) as server:
                with urlopen(server.url + "api/file?path=original/page.png") as response:
                    self.assertEqual(response.status, 200)
                for value in ("../secret", "review.json"):
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(server.url + "api/file?path=" + value)
                    self.assertEqual(caught.exception.code, 403)

    def test_formula_edit_rewrites_markdown_and_finalizes_only_after_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("Result: $$x^2$$\n", encoding="utf-8")
            (root / "review.json").write_text(
                json.dumps(
                    {
                        "status": "needs_review",
                        "pages": [
                            {
                                "page": 1,
                                "formula_crops": [
                                    {
                                        "path": "formula-crops/one.png",
                                        "latex": "x^2",
                                        "review": "required",
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            state = store.current
            crop = state["pages"][0]["formula_crops"][0]
            crop["latex"] = "x^3"
            crop["review"] = "confirmed"

            with patch(
                "paper2latex_local.render.render_markdown_pdf",
                side_effect=fake_render_pdf,
            ) as render_pdf:
                final = store.finalize(state)

            self.assertEqual((root / "document.initial.md").read_text(), "Result: $$x^2$$\n")
            document = (root / "document.md").read_text()
            self.assertIn("x^3", document)
            self.assertIn("paper2latex-formula", document)
            self.assertEqual(render_pdf.call_count, 2)
            self.assertEqual(final["status"], "finalized")
            self.assertTrue((root / "review.final.json").is_file())

    def test_saving_unreviewed_proposal_preserves_mineru_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("Result: $x$\n", encoding="utf-8")
            (root / "review.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page": 1,
                                "formula_crops": [
                                    {
                                        "path": "formula-crops/one.png",
                                        "mineru_latex": "x",
                                        "latex": "y",
                                        "review": "required",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            store.save(store.current)
            document = (root / "document.md").read_text()
            self.assertIn("$x$", document)
            self.assertNotIn("$y$", document)

    def test_pdf_failure_does_not_create_a_final_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("Ready\n", encoding="utf-8")
            (root / "review.json").write_text(
                json.dumps({"status": "needs_review", "pages": []}),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            with patch(
                "paper2latex_local.render.render_markdown_pdf",
                side_effect=RenderError("render failed"),
            ):
                with self.assertRaisesRegex(ReviewError, "render failed"):
                    store.finalize()

            self.assertFalse((root / "review.final.json").exists())
            self.assertEqual(store.current["status"], "needs_review")

    def test_review_ui_can_restore_a_crossed_out_candidate(self) -> None:
        html = Path(
            "src/paper2latex_local/review.html"
        ).read_text(encoding="utf-8")
        self.assertIn("node.crossed_out = false", html)
        self.assertIn("公式原图", html)
        self.assertIn("MinerU：", html)
        self.assertIn("document-markdown", html)
        self.assertIn("class=\"reverse\"", html)
        self.assertLess(html.index("document.preview.pdf"), html.index("document.pdf"))
        self.assertIn('<option value="">请选择</option>', html)
        self.assertNotIn("node.mode = row.querySelector('.mode').value", html)
        self.assertIn("!['printed', 'handwritten'].includes(page.mode)", html)
        self.assertIn("请先选择打印体或手写体", html)

    def test_document_save_renders_current_draft_preview_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("$x$\n", encoding="utf-8")
            (root / "review.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page": 1,
                                "formula_crops": [
                                    {
                                        "path": "formula-crops/one.png",
                                        "latex": "x",
                                        "review": "required",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            current = store.current
            crop = current["pages"][0]["formula_crops"][0]
            crop["latex"] = "y"
            crop["review"] = "confirmed"

            def render(
                markdown: Path,
                output: Path,
                *,
                log_path: Path,
                working_dir: Path | None = None,
            ) -> None:
                self.assertIn("$y$", markdown.read_text(encoding="utf-8"))
                output.write_bytes(b"%PDF-current-draft")
                log_path.write_text("exit_code: 0\n", encoding="utf-8")

            with patch("paper2latex_local.render.render_markdown_pdf", side_effect=render):
                saved = store.save(current)

            self.assertEqual(saved["outputs"]["review_preview_pdf"], "document.preview.pdf")
            self.assertEqual((root / "document.preview.pdf").read_bytes(), b"%PDF-current-draft")
            self.assertIn("$y$", (root / "document.md").read_text(encoding="utf-8"))

    def test_preview_render_failure_preserves_document_and_current_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("$x$\n", encoding="utf-8")
            (root / "review.json").write_text(
                json.dumps({"pages": [{"page": 1, "formula_crops": []}]}),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            before_document = (root / "document.md").read_bytes()
            before_current = (root / "review.current.json").read_bytes()
            changed = store.current
            changed["document_markdown"] = "$y$\n"
            with patch(
                "paper2latex_local.render.render_markdown_pdf",
                side_effect=RenderError("preview render failed"),
            ):
                with self.assertRaisesRegex(ReviewError, "preview render failed"):
                    store.save(changed)
            self.assertEqual((root / "document.md").read_bytes(), before_document)
            self.assertEqual((root / "review.current.json").read_bytes(), before_current)

    def test_preview_render_resolves_relative_assets_from_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "exports/mineru/assets/table.png"
            asset.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(asset)
            (root / "document.md").write_text(
                "![table](exports/mineru/assets/table.png)\n",
                encoding="utf-8",
            )
            (root / "review.json").write_text(
                json.dumps({"pages": [{"page": 1, "formula_crops": []}]}),
                encoding="utf-8",
            )
            store = ReviewStore(root)

            def render(
                markdown: Path,
                output: Path,
                *,
                log_path: Path,
                working_dir: Path | None = None,
            ) -> None:
                self.assertEqual(working_dir, root.resolve())
                self.assertTrue((working_dir / "exports/mineru/assets/table.png").is_file())
                output.write_bytes(b"%PDF-with-asset")
                log_path.write_text("exit_code: 0\n", encoding="utf-8")

            with patch("paper2latex_local.render.render_markdown_pdf", side_effect=render):
                store.save(store.current)
            self.assertTrue((root / "document.preview.pdf").is_file())

    def test_duplicate_formula_edit_is_not_applied_globally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("$x$ and $x$\n", encoding="utf-8")
            state = {
                "status": "needs_review",
                "pages": [
                    {
                        "page": 1,
                        "formula_crops": [
                            {
                                "path": "formula-crops/one.png",
                                "mineru_latex": "x",
                                "latex": "y",
                                "review": "confirmed",
                            }
                        ],
                    }
                ],
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            current = store.current
            crop = current["pages"][0]["formula_crops"][0]
            crop["latex"] = "y"
            crop["review"] = "confirmed"
            store.save(current)
            document = (root / "document.md").read_text()
            self.assertIn("-->$y$<!--", document)
            self.assertTrue(document.endswith(" and $x$\n"))

    def test_unmapped_formula_requires_explicit_manual_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("No formula here.\n", encoding="utf-8")
            state = {
                "pages": [
                    {
                        "page": 1,
                        "formula_crops": [
                            {
                                "path": "formula-crops/one.png",
                                "latex": "y",
                                "review": "required",
                            }
                        ],
                    }
                ]
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            current = store.current
            crop = current["pages"][0]["formula_crops"][0]
            self.assertEqual(crop["markdown_mapping"], "manual_required")
            crop["review_flags"] = []
            crop["review"] = "confirmed"
            with self.assertRaisesRegex(ReviewError, "manual Markdown placement"):
                store.save(current)
            crop["markdown_mapping"] = "manual_confirmed"
            current["document_markdown"] += "\n$$y$$\n"
            store.save(current)
            self.assertIn("$$y$$", (root / "document.md").read_text())

    def test_manual_formula_must_be_inside_a_math_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("Plain prose has x.\n", encoding="utf-8")
            state = {
                "pages": [
                    {
                        "page": 1,
                        "formula_crops": [
                            {
                                "path": "formula-crops/one.png",
                                "latex": "x",
                                "review": "required",
                            }
                        ],
                    }
                ]
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            current = store.current
            crop = current["pages"][0]["formula_crops"][0]
            crop["review_flags"] = []
            crop["review"] = "confirmed"
            crop["markdown_mapping"] = "manual_confirmed"
            with self.assertRaisesRegex(ReviewError, "complete math token"):
                store.save(current)

    def test_manual_formula_crops_require_distinct_math_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("No formulas yet.\n", encoding="utf-8")
            state = {
                "pages": [
                    {
                        "page": 1,
                        "formula_crops": [
                            {
                                "path": f"formula-crops/{name}.png",
                                "review": "required",
                            }
                            for name in ("one", "two")
                        ],
                    }
                ]
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            current = store.current
            for crop in current["pages"][0]["formula_crops"]:
                crop["latex"] = "x"
                crop["review_flags"] = []
                crop["review"] = "confirmed"
                crop["markdown_mapping"] = "manual_confirmed"
            current["document_markdown"] += "\n$x$\n"

            with self.assertRaisesRegex(ReviewError, "distinct complete math token"):
                store.save(current)

            current["document_markdown"] += "$x$\n"
            store.save(current)
            self.assertEqual((root / "document.md").read_text().count("$x$"), 2)

    def test_uncertain_page_mode_cannot_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {
                "pages": [
                    {
                        "page": 1,
                        "mode": "uncertain",
                        "mode_confirmation_required": True,
                    }
                ]
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            current = store.current
            current["pages"][0]["mode_confirmation_required"] = False
            with self.assertRaisesRegex(ReviewError, "valid_page_mode_required"):
                store.finalize(current)
            current["pages"][0]["mode"] = "handwritten"
            store.finalize(current)
            self.assertEqual(store.final["pages"][0]["mode"], "handwritten")

    def test_finalize_rejects_client_state_that_deletes_the_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_task(root)
            store = ReviewStore(root)
            with self.assertRaisesRegex(ReviewError, "changed task kind"):
                store.finalize({})
            self.assertFalse((root / "review.final.json").exists())

    def test_save_rejects_candidate_deletion_without_destroying_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_task(root)
            store = ReviewStore(root)
            original = store.current
            with self.assertRaisesRegex(ReviewError, "changed task kind"):
                store.save({})
            self.assertEqual(store.current, original)

    def test_document_review_cannot_add_a_graph_or_change_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {
                "schema_version": 2,
                "task_id": "doc",
                "mode": "printed",
                "pages": [{"page": 1, "sha256": "fixed", "formula_crops": []}],
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            changed = store.current
            changed["graph"] = {"nodes": [], "edges": []}
            with self.assertRaisesRegex(ReviewError, "changed task kind"):
                store.save(changed)
            changed = store.current
            changed["pages"][0]["sha256"] = "changed"
            with self.assertRaisesRegex(ReviewError, "immutable page 1 sha256"):
                store.save(changed)

    def test_diagram_task_id_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_task(root)
            review = json.loads((root / "review.json").read_text(encoding="utf-8"))
            review["schema_version"] = 3
            review["task_id"] = root.name
            review["content_kind"] = "flowchart"
            (root / "review.json").write_text(json.dumps(review), encoding="utf-8")
            store = ReviewStore(root)
            changed = store.current
            changed["task_id"] = "another-task"
            with self.assertRaisesRegex(ReviewError, "immutable task_id"):
                store.save(changed, export_diagram=False)

    def test_legacy_review_without_task_id_gets_server_owned_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_task(root)
            store = ReviewStore(root)
            self.assertEqual(store.initial["task_id"], root.name)
            changed = store.current
            changed["task_id"] = "forged"
            with self.assertRaisesRegex(ReviewError, "immutable task_id"):
                store.save(changed, export_diagram=False)

            legacy_initial = store.initial
            legacy_initial.pop("task_id")
            (root / "review.initial.json").write_text(
                json.dumps(legacy_initial), encoding="utf-8"
            )
            legacy_current = store.current
            legacy_current.pop("task_id")
            (root / "review.current.json").write_text(
                json.dumps(legacy_current), encoding="utf-8"
            )
            migrated = ReviewStore(root)
            self.assertEqual(migrated.current["task_id"], root.name)
            forged = migrated.current
            forged["task_id"] = "forged"
            with self.assertRaisesRegex(ReviewError, "immutable task_id"):
                migrated.save(forged, export_diagram=False)

    def test_save_cannot_forge_a_finalized_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "task_id": "doc",
                        "status": "needs_review",
                        "mode": "printed",
                        "page_count": 1,
                        "pages": [{"page": 1, "human_review": "required"}],
                    }
                ),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            forged = store.current
            forged["status"] = "finalized"
            forged["human_review"] = "confirmed"
            forged["pages"][0]["human_review"] = "confirmed"
            saved = store.save(forged)
            self.assertEqual(saved["status"], "needs_review")
            self.assertEqual(saved["human_review"], "required")
            self.assertEqual(saved["pages"][0]["human_review"], "required")
            self.assertFalse((root / "review.final.json").exists())
            self.assertEqual(load_status(root)["status"], "needs_review")

    def test_finalized_review_requires_explicit_reopen_before_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review.json").write_text(
                json.dumps({"status": "needs_review", "pages": []}),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            store.finalize()
            with self.assertRaisesRegex(ReviewError, "explicit reopen"):
                store.save(store.current)
            self.assertTrue((root / "review.final.json").is_file())
            self.assertEqual(store.current["status"], "finalized")

    def test_finalize_rejects_deleted_formula_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("$x$\n", encoding="utf-8")
            state = {
                "pages": [
                    {
                        "page": 1,
                        "formula_crops": [
                            {
                                "path": "formula-crops/one.png",
                                "latex": "x",
                                "review": "required",
                            }
                        ],
                    }
                ]
            }
            (root / "review.json").write_text(json.dumps(state), encoding="utf-8")
            store = ReviewStore(root)
            deleted = store.current
            deleted["pages"][0]["formula_crops"] = []
            with self.assertRaisesRegex(ReviewError, "formula crop identity"):
                store.finalize(deleted)

    def test_effective_status_uses_final_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document.md").write_text("Ready\n", encoding="utf-8")
            (root / "review.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "task_id": "doc",
                        "status": "needs_review",
                        "mode": "printed",
                        "page_count": 1,
                        "pages": [{"page": 1, "human_review": "required"}],
                    }
                ),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            with patch(
                "paper2latex_local.render.render_markdown_pdf",
                side_effect=fake_render_pdf,
            ):
                store.finalize()
            status = load_status(root)
            self.assertEqual(status["status"], "finalized")
            self.assertEqual(status["pages"][0]["human_review"], "confirmed")

    def test_final_snapshot_failure_never_marks_current_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "review.json").write_text(
                json.dumps({"status": "needs_review", "pages": []}),
                encoding="utf-8",
            )
            store = ReviewStore(root)
            real_write = review_server_module._write_json

            def fail_final(path: Path, value: object) -> None:
                if path.name == "review.final.json":
                    raise OSError("disk full")
                real_write(path, value)

            with patch(
                "paper2latex_local.review_server._write_json",
                side_effect=fail_final,
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    store.finalize()
            self.assertEqual(store.current["status"], "needs_review")
            self.assertFalse((root / "review.final.json").exists())


if __name__ == "__main__":
    unittest.main()
