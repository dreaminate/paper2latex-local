# Architecture and evidence boundary

## Implemented pipeline

```text
images / image folders / PDFs
  -> natural ordering and 50-page bound
  -> immutable originals + SHA-256
  -> orientation normalization / 300 DPI PDF rasterization
  -> quality gate and optional perspective evidence
  -> page routing: printed | handwritten | uncertain
  -> local OCR adapter
  -> formula crops when the adapter supplies bounding boxes
  -> Markdown with explicit failed-page markers
  -> Pandoc/XeLaTeX rendered PDF
  -> Tesseract searchable PDF
  -> review.json + human review
```

An image first receives a content decision:

```text
phone photo
  -> document     -> MinerU/Tesseract -> Markdown/LaTeX/PDF
  -> flowchart    \
  -> mindmap       -> canonical DiagramGraph -> faithful + clean projections
  -> architecture /
  -> uncertain    -> stop and ask for an explicit route
```

`DiagramGraph` is the only semantic source of truth. draw.io XML, Mermaid, SVG,
and PDF are projections and are regenerated from that graph. Faithful/clean
geometry is represented by draw.io, SVG, and PDF; Mermaid is a structural
projection because its source syntax does not encode arbitrary node positions. OpenCV extracts
conservative shape/connector candidates; it is not treated as a general
handwriting-understanding model. Low-confidence labels, directions, isolated
nodes, and formula-like labels carry explicit review flags.

The pipeline distinguishes three states:

- `conversion_state: completed`: every page produced OCR output;
- `conversion_state: partial_failed`: successful pages were retained and one or
  more pages are explicitly missing;
- `status: needs_review`: a person still has to confirm page modes and formulas.

These fields are deliberately separate. Machine completion is not user acceptance.

## Task package

`review.json` schema version 2 is the processing source of truth. Each page
records its original and cleaned path, source PDF page if applicable, SHA-256,
quality metrics, route and confidence, engine and version, OCR stages, formula
crops, warnings, and human-review state.

Original inputs are copied once and never modified. Derived images belong in
`cleaned/`; formula crops belong in `formula-crops/`; engine logs and intermediate
PDF pages belong in `exports/`.

Diagram task packages use schema version 3 and three review snapshots:

- `review.initial.json`: immutable initial recognition;
- `review.current.json`: the latest edited state, overwritten on save;
- `review.final.json`: created only after unresolved flags are cleared.

There is deliberately no intermediate revision history. The HTTP review server
binds only to `127.0.0.1` and serves only paths declared inside its task root.
It preserves candidate identities on every save, requires unresolved items to be
cleared, and writes the final snapshot before publishing finalized current state.
Document formula edits use comments outside complete math tokens as stable
anchors. Unmapped crops remain `manual_required` until the reviewer places the
LaTeX in the editable Markdown and confirms that placement.

The earlier `init-task` command remains available for packaging without OCR. The
`convert` command is the end-to-end path and never treats an initialized task as
a recognition result.

## Engine boundary

| Project | Role | License evidence | Integration |
| --- | --- | --- | --- |
| [MinerU](https://github.com/opendatalab/MinerU) | Layout, text, formulas, tables, Markdown | `LicenseRef-MinerU-Open-Source-License` in installed package | integrated external adapter; local model required |
| [UniMERNet](https://github.com/opendatalab/UniMERNet) | Mathematical expression recognition | Apache-2.0 repository/model card | direct isolated adapter integrated for MinerU formula crops |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Printed-text fallback and searchable PDF | Apache-2.0 | integrated external command |
| [Pix2Text](https://github.com/breezedeus/Pix2Text) | Layout, text, formulas, Markdown | MIT | candidate, not integrated |
| [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) | Searchable PDF/A | MPL-2.0 | candidate; local Tesseract renderer is integrated |

The repository adapts commands and public APIs; it does not copy external engine
code. A code license does not establish the license of every model weight. No
model weight is bundled.

## Diagram prior-art decision

The implementation reuses installed OpenCV for candidate geometry and the
documented uncompressed draw.io `mxGraphModel` format for editable cells. It
does not add NetworkX, the Python Graphviz wrapper, or a Mermaid Python package.
Graphviz `dot` and Mermaid CLI were not available on the target Mac. The project
contains dependency-free SVG/PDF fallback renderers for tests and restricted
environments. The `/ocr转换` Skill sets `PAPER2LATEX_USE_DRAWIO_PDF=1`, so this
Mac's production workflow uses installed draw.io Desktop for Unicode- and
math-preserving diagram PDF export and opens the confirmed clean layout.

## MinerU safety boundary

The installed MinerU 3.4.4 entry point aborts on this Mac when it imports its
OpenMP runtimes in the default order. The adapter launches the same installed
entry point through its declared Python interpreter after importing OpenCV. This
startup order was verified with `mineru --version` and `mineru --help` behavior.
It does not set the unsupported `KMP_DUPLICATE_LIB_OK` environment variable.

Recognition always sets `MINERU_MODEL_SOURCE=local` and requires a configured,
existing local model path. Missing models therefore fail before implicit network
downloads. Model installation is a separate, confirmed operation.

## Acceptance path

Before handwriting or formula recognition can be called user-accepted, evaluate
a private sample set containing:

1. clear printed Chinese and English;
2. skewed phone photos with shadows and glare;
3. handwritten digits and Latin/Greek symbols;
4. fractions, roots, powers, subscripts, sums, integrals, and matrices;
5. long equations and multi-line derivations.

Compare both LaTeX source and rendered structure against the source crop. Record
per-page and per-formula corrections in `review.json`; do not replace uncertain
content with an unmarked guess.
