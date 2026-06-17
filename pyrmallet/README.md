# pyrmallet

Python bindings for [RustMallet](https://github.com/mimno/RustMallet) — a fast Rust implementation of the sparse Gibbs sampling LDA algorithm from [MALLET](https://mallet.cs.umass.edu/), following the SparseLDA scheme of Yao, Mimno and McCallum (KDD 2009).

Built with [PyO3](https://pyo3.rs) and [maturin](https://maturin.rs). There are two layers: a sklearn-compatible `LatentDirichletAllocation` class and a lower-level `_rust_mallet` extension module.

## Install

```bash
pip install pyrmallet
```

## sklearn-compatible API

`LatentDirichletAllocation` follows the scikit-learn estimator interface. It takes a list of raw text strings — tokenization and vocabulary building happen inside Rust.

```python
from pyrmallet import LatentDirichletAllocation

docs = ["the quick brown fox ...", "machine learning models ...", ...]

lda = LatentDirichletAllocation(n_components=20, max_iter=1000)
lda.fit(docs)

lda.components_               # ndarray [n_topics, n_vocab], rows sum to ~1
lda.doc_topic_distributions_  # ndarray [n_docs, n_topics]
lda.feature_names_in_         # vocabulary array
lda.n_features_in_            # vocabulary size
```

`fit_transform()` is also available and returns `doc_topic_distributions_` directly.

**Inferring topic distributions for new documents**

After `fit()`, call `transform()` with any list of raw text strings. Tokens not seen during training are silently ignored.

```python
new_docs = ["natural language processing tasks ...", "deep reinforcement learning ..."]
theta = lda.transform(new_docs)  # ndarray [n_new_docs, n_topics]
```

The number of Gibbs iterations used for inference is controlled by `n_inference_iter` (default 50).

**Constructor parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 10 | Number of topics |
| `max_iter` | 1000 | Gibbs sampling iterations |
| `burn_in` | 200 | Iterations before hyperparameter optimization |
| `optimize_interval` | 50 | Optimize alpha/beta every N iterations; 0 to disable |
| `num_samples` | 5 | Samples averaged for final estimates |
| `sample_interval` | 25 | Iterations between samples |
| `doc_topic_prior` | `n_components` | Initial symmetric alpha sum |
| `topic_word_prior` | 0.01 | Initial beta per word |
| `random_state` | 42 | Random seed |
| `n_inference_iter` | 50 | Gibbs iterations per document during `transform()` |
| `stopwords` | None | List of words to exclude, or path to a stoplist file |
| `min_doc_freq` | 1 | Drop words appearing in fewer than N documents |
| `max_doc_fraction` | 1.0 | Drop words appearing in more than this fraction of documents |
| `verbose` | False | Print log-likelihood progress during training |

## Low-level API

`pyrmallet._rust_mallet` exposes `Corpus` and `TopicModel` objects directly.

```python
from pyrmallet import _rust_mallet as rm

# Build a corpus directly from strings (no file I/O)
stopwords = rm.load_stopwords("examples/english-stoplist.txt")
corpus = rm.Corpus.from_strings(
    docs,
    stopwords=stopwords,
    min_doc_freq=2,
)

# Or load from a file
corpus = rm.Corpus.from_text_file("docs.txt", stopwords=stopwords)
corpus = rm.Corpus.from_tsv_file(
    "docs.tsv", id_column=0, text_column=1,
    stopwords=stopwords,
)

# Save/load a preprocessed corpus
corpus.save("corpus.corp")
corpus = rm.Corpus.load("corpus.corp")

# Train
model = rm.train(corpus, num_topics=20, iterations=1000, verbose=True)

# Inspect results
model.top_words(n=10)       # List[List[str]], one word list per topic
model.topic_word_matrix()   # List[List[float]], shape [num_topics][num_types]
model.doc_topic_matrix()    # List[List[float]], shape [num_docs][num_topics]
model.log_likelihood(corpus)

# Infer topic distributions for new raw-text documents (fixed-phi Gibbs)
theta = model.infer_strings(new_docs, n_iter=50)  # List[List[float]], shape [n_docs][num_topics]

# Or infer from a pre-built count matrix (columns indexed by training vocabulary)
theta = model.infer(count_matrix, n_iter=50)      # List[List[float]]
```

## Building from source

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain. From the repo root:

```bash
PATH="$HOME/.cargo/bin:$PATH" uv run --with maturin maturin develop
```

See the [RustMallet README](https://github.com/mimno/RustMallet#readme) for the full project, including the standalone CLI tools.
