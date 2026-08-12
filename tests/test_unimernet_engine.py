from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2latex_local.unimernet_engine import UniMERNetEngine, UniMERNetError


class UniMERNetEngineTests(unittest.TestCase):
    def make_engine(self, root: Path) -> UniMERNetEngine:
        python = root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        model = root / "model"
        model.mkdir()
        for filename in (
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.pth",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (model / filename).write_text("x", encoding="utf-8")
        return UniMERNetEngine(python=python, model_dir=model)

    def test_adapter_is_offline_and_returns_latex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = self.make_engine(root)
            image = root / "formula.png"
            image.write_bytes(b"png")
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    {
                        "device": "mps",
                        "predictions": [
                            {"source": str(image.resolve()), "latex": r"x^2+1"}
                        ],
                    }
                ),
                stderr="",
            )
            with patch("subprocess.run", return_value=completed) as run:
                prediction = engine.recognize(
                    [image], log_path=root / "unimernet.log"
                )[0]
            self.assertEqual(prediction.latex, r"x^2+1")
            environment = run.call_args.kwargs["env"]
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(environment["NO_ALBUMENTATIONS_UPDATE"], "1")

    def test_missing_model_refuses_implicit_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = UniMERNetEngine(
                python=root / "missing-python", model_dir=root / "missing-model"
            )
            with self.assertRaisesRegex(UniMERNetError, "implicit download"):
                engine.recognize(
                    [root / "formula.png"], log_path=root / "unimernet.log"
                )


if __name__ == "__main__":
    unittest.main()
