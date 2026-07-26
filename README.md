# Paper2LaTeX Local

Local-first scaffolding for turning phone photos of paper into reviewable
Markdown with LaTeX formulas and, later, searchable PDF exports.

> Status: foundation release. The task package and provenance layer work.
> OCR, photo quality scoring, perspective correction, and PDF rendering are
> deliberately marked `not_run` until real engines and tests are integrated.

## Why this repository exists

The target workflow handles two kinds of documents:

- pages containing printed text;
- pages containing handwriting, often dominated by numbers and mathematics.

Each page belongs to one mode. A batch contains at most 50 phone photos.
Recognition should run locally, prioritize accuracy, preserve the originals,
and require visual review of generated LaTeX.

## What works now

- create a self-contained task folder from 1–50 photos;
- preserve source filenames and SHA-256 hashes;
- keep printed and handwritten modes explicit;
- generate the initial Markdown document and review manifest;
- report whether candidate external OCR engines are available;
- refuse unsupported inputs, duplicate inputs, and accidental overwrites.

## Quick start

Python 3.11 or newer is required. The foundation release has no runtime
dependencies.

```bash
python -m pip install -e .
paper2latex init-task \
  --name "calculus-notes" \
  --mode handwritten \
  photo-01.jpg photo-02.jpg
```

The default output is:

```text
tasks/20260726T150000Z-calculus-notes/
├── original/
├── cleaned/
├── formula-crops/
├── exports/
├── document.md
└── review.json
```

Inspect the manifest:

```bash
paper2latex status tasks/20260726T150000Z-calculus-notes
paper2latex engines
```

## Planned engine boundary

The project will integrate engines through adapters instead of copying their
code:

| Role | Candidate | Current state |
| --- | --- | --- |
| Printed/mixed page to Markdown | Pix2Text or MinerU | not integrated |
| Handwritten formula image to LaTeX | UniMERNet | not integrated |
| Searchable archival PDF | OCRmyPDF | not integrated |

Code licenses are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Model-weight licenses must be checked separately before bundling or
redistribution.

## Truth boundary

This repository does **not** currently claim:

- reliable handwritten formula recognition;
- automatic formula-region detection;
- photo blur, glare, shadow, or missing-corner detection;
- perspective correction;
- Markdown-to-PDF or searchable-PDF generation;
- measured accuracy on real user photos.

Those capabilities become complete only after implementation and sample-based
acceptance tests.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## License

Project code is available under the [MIT License](LICENSE). External engines
and model weights keep their own licenses.
