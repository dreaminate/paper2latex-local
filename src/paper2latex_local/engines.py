"""Discovery metadata for optional OCR engines.

Discovery is not integration. An installed package or command may still be
incompatible with the host or with this project's future adapter.
"""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any


ENGINE_CANDIDATES = (
    {
        "id": "pix2text",
        "role": "printed_or_mixed_page_to_markdown",
        "code_license": "MIT",
        "python_package": "pix2text",
        "command": None,
        "integration": "not_integrated",
    },
    {
        "id": "unimernet",
        "role": "handwritten_formula_image_to_latex",
        "code_license": "Apache-2.0",
        "python_package": "unimernet",
        "command": "unimernet_gui",
        "integration": "not_integrated",
    },
    {
        "id": "mineru",
        "role": "printed_complex_document_to_markdown",
        "code_license": "custom_apache_based",
        "python_package": "mineru",
        "command": "mineru",
        "integration": "not_integrated",
    },
    {
        "id": "ocrmypdf",
        "role": "searchable_pdfa",
        "code_license": "MPL-2.0",
        "python_package": "ocrmypdf",
        "command": "ocrmypdf",
        "integration": "not_integrated",
    },
)


def discover_engines() -> list[dict[str, Any]]:
    """Return candidate metadata plus local package/command availability."""

    results: list[dict[str, Any]] = []
    for candidate in ENGINE_CANDIDATES:
        package = candidate["python_package"]
        command = candidate["command"]
        item = dict(candidate)
        item["package_available"] = importlib.util.find_spec(package) is not None
        item["command_available"] = bool(command and shutil.which(command))
        item["model_weights_bundled"] = False
        results.append(item)
    return results
