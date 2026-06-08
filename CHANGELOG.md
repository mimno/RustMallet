# Changelog

## Unreleased

### Added
- **Post-training inference** (`transform()` on new documents). After `fit()`,
  `LatentDirichletAllocation.transform()` accepts a new list of raw text strings
  and returns topic distributions inferred with fixed topic-word probabilities
  (Gibbs sampling). New `n_inference_iter` parameter (default 50) controls the
  number of sampling iterations per document.
- `TopicModel.infer_strings(docs, n_iter, seed)` in the low-level API. Tokenises
  raw text using the same regex as corpus loading, maps tokens to the training
  vocabulary, and runs fixed-phi Gibbs inference.

### Changed
- Python bindings now require the native PyO3 extension. The previous subprocess
  fallback (shelling out to the `preprocess` and `train` CLI binaries) has been
  removed. If the extension has not been built, import raises `ImportError` with
  clear build instructions.
- `LatentDirichletAllocation` no longer accepts a `workdir` parameter (was only
  used by the subprocess path).

### Removed
- `pyrmallet/_binaries.py` and `pyrmallet/_io.py` (only used by the subprocess path).
- `python/rust_mallet/` package (old prototype superseded by `pyrmallet/`).

---

## 0.1.0 — initial release

### Added
- `preprocess` CLI: tokenise text files into a binary corpus format. Supports
  plain-text, id-prefixed, and TSV input formats. Unicode-aware tokeniser;
  configurable stoplist, document-frequency filtering, and token regex.
- `analyze` CLI: suggest stopwords from a preprocessed corpus using document-
  frequency, length, numeric, and non-alphabetic heuristics.
- `train` CLI: sparse Gibbs LDA (SparseLDA three-bucket scheme) with asymmetric
  hyperparameter optimization (Minka fixed-point updates for alpha and beta).
  Output averaged over multiple post-training samples.
- `show` CLI: display top words per topic and per-document topic summaries from
  the `train` output files.
- `pyrmallet` Python package with a `LatentDirichletAllocation` class matching
  the scikit-learn estimator API. Takes raw text strings; tokenisation and
  vocabulary building happen inside Rust via PyO3.
- Low-level `pyrmallet._rust_mallet` extension: `Corpus` (from strings, TSV, or
  binary file) and `TopicModel` with `infer()` for fixed-phi Gibbs inference.
- Release workflow producing pre-built binaries for Linux (x86-64, aarch64),
  macOS (Apple Silicon), and Windows (x86-64).
- Example data: 5000 CS/NLP arXiv abstracts (`examples/cs_cl.tsv`) and an
  English stoplist (`examples/english-stoplist.txt`).
