"""Direct local UniMERNet adapter for formula crops.

The adapter runs an isolated Python interpreter because the project's document
OCR environment and UniMERNet's PyTorch/OpenCV stack have incompatible native
runtime requirements on this Mac.  All model paths are explicit; inference is
never allowed to download weights implicitly.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


UNIMERNET_PACKAGE_VERSION = "0.2.3"
UNIMERNET_MODEL_ID = "wanderkid/unimernet_base"
UNIMERNET_MODEL_REVISION = "af898d48ebb1765cd3511d88f5d5f7c92279c731"
UNIMERNET_MODEL_BYTES = 1_300_760_949
UNIMERNET_MODEL_SHA256 = "16cd0891233cfee3c11215a7b87306f160f7e7f3f52091a6253751c149a8c180"
DEFAULT_UNIMERNET_PYTHON = Path(
    os.environ.get(
        "PAPER2LATEX_UNIMERNET_PYTHON",
        Path.home() / ".local/share/paper2latex-local/unimernet-venv/bin/python",
    )
)
DEFAULT_UNIMERNET_MODEL = Path(
    os.environ.get(
        "PAPER2LATEX_UNIMERNET_MODEL",
        Path.home()
        / ".cache/huggingface/hub/models--wanderkid--unimernet_base"
        / "snapshots/af898d48ebb1765cd3511d88f5d5f7c92279c731",
    )
)


class UniMERNetError(RuntimeError):
    """Raised when direct formula recognition cannot run locally."""


@dataclass(frozen=True)
class FormulaPrediction:
    source: Path
    latex: str


def _runner_source() -> str:
    return r'''import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from unimernet.models.unimernet.unimernet import UniMERModel
from unimernet.processors.formula_processor import FormulaImageEvalProcessor


def main():
    request = json.loads(sys.stdin.read())
    model_dir = Path(request["model_dir"])
    checkpoint = model_dir / "pytorch_model.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"UniMERNet checkpoint is missing: {checkpoint}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = OmegaConf.create({
        "model_name": "unimernet",
        "model_config": {
            "model_name": str(model_dir),
            "max_seq_len": int(request.get("max_seq_len", 1536)),
        },
        "tokenizer_name": "nougat",
        "tokenizer_config": {"path": str(model_dir)},
        "load_pretrained": True,
        "load_finetuned": False,
        "pretrained": str(checkpoint),
    })
    model = UniMERModel.from_config(cfg).to(device).eval()
    processor = FormulaImageEvalProcessor([192, 672])
    predictions = []
    for value in request["images"]:
        path = Path(value)
        with Image.open(path) as image:
            tensor = processor(image.convert("RGB")).unsqueeze(0).to(device)
        latex = model.generate(
            {"image": tensor}, temperature=0.0, do_sample=False
        )["pred_str"][0]
        predictions.append({"source": str(path), "latex": latex.strip()})
    print(json.dumps({"device": str(device), "predictions": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


class UniMERNetEngine:
    """Recognize already-detected formula images as editable LaTeX."""

    engine_id = "unimernet"
    engine_version = UNIMERNET_PACKAGE_VERSION

    def __init__(
        self,
        *,
        python: Path = DEFAULT_UNIMERNET_PYTHON,
        model_dir: Path = DEFAULT_UNIMERNET_MODEL,
    ) -> None:
        self.python = Path(python).expanduser().absolute()
        self.model_dir = Path(model_dir).expanduser().resolve()

    @property
    def model_ready(self) -> bool:
        required = (
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.pth",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        return self.python.is_file() and all(
            (self.model_dir / filename).is_file() for filename in required
        )

    def recognize(self, images: list[Path], *, log_path: Path) -> list[FormulaPrediction]:
        if not images:
            return []
        if not self.model_ready:
            raise UniMERNetError(
                "UniMERNet 0.2.3 or the pinned Base model is not installed locally; "
                "refusing an implicit download"
            )
        resolved = [Path(image).expanduser().resolve() for image in images]
        request = json.dumps(
            {
                "model_dir": str(self.model_dir),
                "images": [str(image) for image in resolved],
                "max_seq_len": 1536,
            },
            ensure_ascii=False,
        )
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
        completed = subprocess.run(
            [self.python, "-c", _runner_source()],
            input=request,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"exit_code: {completed.returncode}\n"
            f"model_id: {UNIMERNET_MODEL_ID}\n"
            f"model_revision: {UNIMERNET_MODEL_REVISION}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise UniMERNetError(
                f"UniMERNet exited with {completed.returncode}; see {log_path}"
            )
        payload = json.loads(completed.stdout.splitlines()[-1])
        return [
            FormulaPrediction(Path(item["source"]), str(item["latex"]))
            for item in payload["predictions"]
        ]


def unimernet_status() -> dict[str, object]:
    engine = UniMERNetEngine()
    checkpoint = engine.model_dir / "pytorch_model.pth"
    return {
        "id": engine.engine_id,
        "package_version": engine.engine_version,
        "python": str(engine.python),
        "model_id": UNIMERNET_MODEL_ID,
        "model_revision": UNIMERNET_MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_bytes": UNIMERNET_MODEL_BYTES,
        "model_sha256": UNIMERNET_MODEL_SHA256,
        "model_dir": str(engine.model_dir),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "model_ready": engine.model_ready,
        "implicit_downloads": False,
    }


__all__ = [
    "FormulaPrediction",
    "UniMERNetEngine",
    "UniMERNetError",
    "unimernet_status",
]
