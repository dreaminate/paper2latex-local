# Architecture and evidence boundary

## Intended pipeline

```text
phone photos
  -> quality gate
  -> orientation / crop / perspective cleanup
  -> page routing: printed | handwritten
  -> text and formula-region detection
  -> OCR / formula-to-LaTeX adapters
  -> side-by-side human review
  -> Markdown + rendered PDF + searchable archival PDF
```

The foundation release implements only task creation, provenance, explicit
page mode, status reporting, and adapter discovery.

## Task package

`review.json` is the machine-readable source of processing truth. Every stage
starts as `not_run` and must be changed only by the stage that produced the
corresponding artifact.

Original photos are immutable inputs. Derived images belong in `cleaned/` and
formula crops belong in `formula-crops/`. Human-readable exports belong in
`exports/`.

## Candidate prior art

| Project | Intended use | Code license checked | Integration state |
| --- | --- | --- | --- |
| [Pix2Text](https://github.com/breezedeus/Pix2Text) | Page layout, text, formulas, Markdown | MIT | candidate |
| [UniMERNet](https://github.com/opendatalab/UniMERNet) | Formula image to LaTeX, including handwritten evaluation data | Apache-2.0 | candidate |
| [MinerU](https://github.com/opendatalab/MinerU) | Printed and complex document parsing | custom license based on Apache-2.0 | candidate, license review required |
| [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) | Searchable PDF/A output | MPL-2.0 | candidate external command |

The code license of a repository does not automatically establish the license
of every model weight it downloads. No model weight is bundled in this
repository.

## Acceptance path

Before an OCR adapter can be called supported, test it on a private,
user-approved sample set containing:

1. clear printed Chinese and English;
2. skewed phone photos with shadows;
3. handwritten digits and Latin/Greek symbols;
4. fractions, roots, superscripts, subscripts, sums, integrals, and matrices;
5. long equations and multi-line derivations.

For formulas, compare both the generated LaTeX source and rendered structure.
A visually wrong formula is a failure even when token-level similarity looks
high.
