# RustMallet

A Rust implementation of the sparse Gibbs sampling LDA algorithm from [MALLET](https://mallet.cs.umass.edu/), following the SparseLDA scheme of Yao, Mimno and McCallum (KDD 2009).

## Quick Start

**Install:** Download a pre-built binary for your platform from the [Releases](https://github.com/mimno/RustMallet/releases) page and put the binaries somewhere on your PATH. Or [build from source](#building).

**Try the included example** (5000 CS/NLP paper abstracts):

```bash
preprocess --input examples/cs_cl.tsv --format tsv --no-label \
    --stoplist examples/english-stoplist.txt --min-doc-freq 5 \
    --output corpus.corp

train --corpus corpus.corp --num-topics 20 --iterations 500

show
```

Sample output:
```
Topic  0: bias  language  biases  social  analysis  findings  impact  fairness
Topic  1: alignment  preference  reward  reinforcement  feedback  policy  human  optimization
Topic  4: retrieval  context  rag  documents  retrieval-augmented  generation  long  queries
Topic  5: safety  attacks  adversarial  privacy  robustness  attack  llm  unlearning
Topic 12: multimodal  visual  image  video  understanding  mllms  multi-modal  textual
Topic 17: speech  recognition  asr  audio  automatic  neural  word  error
```

**Your own data:**

```bash
# Step 1: convert your text to corpus format
preprocess --input mydocs.txt --output corpus.corp

# Step 2: find stopword candidates and review them
analyze --corpus corpus.corp > stopwords.txt

# Step 3: reprocess with the stoplist and frequency filtering
preprocess --input mydocs.txt --output corpus.corp \
    --stoplist stopwords.txt --min-doc-freq 2

# Step 4: train
train --corpus corpus.corp --num-topics 20

# Step 5: view results
show
```

**Choosing the number of topics:**

| Corpus size | Suggested starting point |
|-------------|--------------------------|
| < 500 docs  | 10–20 topics |
| 500–5000    | 20–50 topics |
| 5000–50000  | 50–150 topics |
| > 50000     | 100–300 topics |

Start lower than you think you need. If topics seem too broad or mixed, increase the number. If they look nearly identical, decrease it.

---

## Tools

Four command-line tools form a pipeline:

```
preprocess → analyze → preprocess (with stoplist) → train → show
```

### `preprocess`

Reads a text file, tokenizes it, and writes a binary corpus file used by the other tools.

```
preprocess --input <file> --output <file> [options]
```

**Input formats**

| Flag | Behavior |
|------|----------|
| *(default)* | One document per line, whitespace-tokenized |
| `--id-field` | First whitespace token on each line is the document name |
| `--format tsv` | Tab-delimited columns (default: col 0 = id, col 1 = label, col 2 = text) |

**TSV column options** (used with `--format tsv`)

| Flag | Default | Description |
|------|---------|-------------|
| `--id-column <n>` | 0 | Column index for document name |
| `--label-column <n>` | 1 | Column index for document label |
| `--no-label` | — | Disable label column |
| `--text-column <n>` | 2 | Column index for document text |

**Tokenization**

Tokens are extracted using a regular expression. The default pattern requires a Unicode letter at both ends (minimum length 2) and allows letters or a small set of non-breaking punctuation in the middle:

```
\p{L}[-'\u{2019}.\u{00B7}\p{L}]*\p{L}
```

The permitted interior characters beyond letters are:

| Character | Code | Rationale |
|-----------|------|-----------|
| `-` | U+002D | Hyphen-minus: compound words (well-known, fine-tuning, German, Scandinavian) |
| `'` | U+0027 | Apostrophe: English contractions (don't), French elision (l'homme) |
| `'` | U+2019 | Typographic/smart apostrophe: same role as U+0027, common in modern text |
| `.` | U+002E | Period: abbreviations and initials (U.S.A, e.g., Lic.) |
| `·` | U+00B7 | Middle dot: Catalan geminated-L (col·legi), Welsh, some other scripts |

Em-dash (U+2014), en-dash (U+2013), and all other punctuation break tokens.

| Flag | Default | Description |
|------|---------|-------------|
| `--token-regex <pattern>` | `\p{L}[-'\u{2019}.\u{00B7}\p{L}]*\p{L}` | Regex for token extraction |

**Filtering**

| Flag | Default | Description |
|------|---------|-------------|
| `--stoplist <file>` | — | File with one stopword per line; `#` lines are comments |
| `--min-doc-freq <n>` | 1 | Drop words appearing in fewer than N documents |
| `--max-doc-fraction <f>` | 1.0 | Drop words appearing in more than this fraction of documents |

**Examples**

```bash
# Plain text, one document per line
preprocess --input docs.txt --output corpus.corp

# MALLET-style TSV: id TAB label TAB text
preprocess --input docs.tsv --output corpus.corp --format tsv

# Apply a stoplist and prune rare words
preprocess --input docs.txt --output corpus.corp \
    --stoplist stopwords.txt --min-doc-freq 2

# Custom tokenizer: letters only, no interior punctuation
preprocess --input docs.txt --output corpus.corp \
    --token-regex '\p{L}\p{L}+'
```

---

### `analyze`

Reads a binary corpus file and suggests stopwords using several heuristics. The analysis report goes to **stderr**; the suggested word list goes to **stdout** (one word per line), so stdout can be redirected directly to a stoplist file.

```
analyze --corpus <file> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-doc-fraction <f>` | 0.10 | Flag words appearing in more than this fraction of documents |
| `--max-word-length <n>` | 4 | Flag words shorter than this many characters |
| `--min-doc-freq <n>` | 2 | Report count of words appearing in fewer than N documents |
| `--num-candidates <n>` | 50 | Max words shown per heuristic in the report |
| `--output-stoplist <file>` | — | Also write the word list to this file |

**Heuristics**

1. **High document frequency** — words with IDF below the threshold derived from `--max-doc-fraction`
2. **Short tokens** — words shorter than `--max-word-length` (typically function words)
3. **Numeric tokens** — words composed entirely of digits
4. **Non-alphabetic tokens** — words with fewer than 50% alphabetic characters

Rare words (below `--min-doc-freq`) are reported but not added to the suggested list; remove them at the `preprocess` step with `--min-doc-freq`.

**Examples**

```bash
# Redirect word list to a ready-to-use stoplist file
analyze --corpus corpus.corp > stopwords.txt

# Tighten the threshold (flag words in >5% of docs instead of >10%)
analyze --corpus corpus.corp --max-doc-fraction 0.05 > stopwords.txt

# Write report to a file, word list to another file
analyze --corpus corpus.corp --output-stoplist stopwords.txt 2>report.txt
```

---

### `train`

Reads a binary corpus file and runs LDA using sparse Gibbs sampling. Progress output matches MALLET's format. After the main training loop, hyperparameters are optimized and final distributions are estimated by averaging over multiple samples.

```
train --corpus <file> [options]
```

**Sampling**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-topics <k>` | 10 | Number of topics |
| `--iterations <n>` | 1000 | Number of Gibbs sampling iterations |
| `--burn-in <n>` | 200 | Iterations before hyperparameter optimization begins |
| `--seed <n>` | 42 | Random seed |

**Hyperparameter optimization**

Alpha (document-topic prior) and beta (topic-word prior) are optimized automatically using Minka's fixed-point updates after the burn-in period.

| Flag | Default | Description |
|------|---------|-------------|
| `--optimize-interval <n>` | 50 | Optimize every N iterations after burn-in; set 0 to disable |
| `--alpha-sum <f>` | num-topics | Initial symmetric alpha sum |
| `--beta <f>` | 0.01 | Initial beta per word |

**Output estimation**

Rather than reading distributions from a single final sample, the output files are estimated by averaging over multiple evenly-spaced samples collected after training. This reduces the effect of sampling noise.

| Flag | Default | Description |
|------|---------|-------------|
| `--num-samples <n>` | 5 | Number of samples to average |
| `--sample-interval <n>` | 25 | Gibbs iterations between samples |

**Output**

| Flag | Default | Description |
|------|---------|-------------|
| `--topic-word <file>` | `topic_word.tsv` | Topic-word probability output |
| `--doc-topic <file>` | `doc_topic.tsv` | Document-topic probability output |
| `--show-topics-interval <n>` | 50 | Print top words every N iterations |
| `--words-per-topic <n>` | 7 | Words shown per topic in progress output |

**Output files**

`topic_word.tsv` — one row per (topic, word) pair:
```
topic   word        probability
0       government  0.04832411
0       election    0.03901234
...
```

`doc_topic.tsv` — one row per document, with a label column when labels were present in the corpus:
```
doc         label     topic_0     topic_1     topic_2
doc001      politics  0.01234568  0.97530864  0.01234568
doc002      science   0.97530864  0.01234568  0.01234568
...
```

**Progress output** (stderr, matching MALLET format)

```
Mallet LDA: 20 topics, 5 topic bits, 11111 topic mask
max tokens: 312
total tokens: 847392
Hyperparameter optimization every 50 iterations after burn-in (200)
<10> LL/token: -7.23451
...
[O] alpha_sum=4.21318  beta=0.02341
0   0.21132  government election vote president congress democracy policy
...
<1000> LL/token: -6.12345

Total time: 3 minutes 42 seconds

Collecting 5 samples (25 iterations apart)...
  sample 5/5
```

**Examples**

```bash
# Basic training with defaults
train --corpus corpus.corp --num-topics 20 --iterations 1000

# Disable hyperparameter optimization
train --corpus corpus.corp --num-topics 20 --optimize-interval 0

# More samples for smoother estimates
train --corpus corpus.corp --num-topics 20 --num-samples 10 --sample-interval 50

# Custom output paths
train --corpus corpus.corp --num-topics 20 \
    --topic-word phi.tsv --doc-topic theta.tsv
```

---

### `show`

Reads the output files from `train` and displays results in human-readable form. By default reads `topic_word.tsv` from the current directory.

```
show [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--topic-word <file>` | `topic_word.tsv` | Topic-word probability file |
| `--doc-topic <file>` | `doc_topic.tsv` | Document-topic probability file |
| `--words <n>` | 10 | Words to show per topic |
| `--doc-topics <n>` | 0 | Top topics per document (0 = off) |
| `--threshold <f>` | 0.1 | Minimum probability to show for document topics |

**Examples**

```bash
# Show top 10 words per topic
show

# Show more words
show --words 20

# Show which topics each document is about (top 3, at least 10% probability)
show --doc-topics 3 --threshold 0.10
```

---

## Python bindings

The package includes Python bindings built with [PyO3](https://pyo3.rs) and [maturin](https://maturin.rs). There are two layers: a sklearn-compatible `LatentDirichletAllocation` class and a lower-level `_rust_mallet` extension module.

### Building

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain. From the repo root:

```bash
PATH="$HOME/.cargo/bin:$PATH" uv run --with maturin maturin develop
```

This compiles the native extension (`pyrmallet/_rust_mallet.abi3.so`) and installs the package into uv's virtual environment. The build always uses release optimizations (`profile = "release"` is set in `pyproject.toml`).

### sklearn-compatible API

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

### Low-level API

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

---

## Building

**CLI tools**

```bash
cargo build --release
```

Requires [Rust](https://rustup.rs). Binaries are written to `target/release/`.

**Python bindings**

```bash
PATH="$HOME/.cargo/bin:$PATH" uv run --with maturin maturin develop
```

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain. See [Python bindings](#python-bindings) for details.

---

## Example data

`examples/cs_cl.tsv` contains 5000 computer science and computational linguistics paper abstracts from arXiv (2024). Columns: arXiv ID, date, title+abstract. Use it as a ready-made test corpus:

```bash
preprocess --input examples/cs_cl.tsv --format tsv --no-label \
    --stoplist examples/english-stoplist.txt --min-doc-freq 5 \
    --output corpus.corp
train --corpus corpus.corp --num-topics 20
show
```

`examples/english-stoplist.txt` is a starter English stoplist. Edit it to remove any content words relevant to your corpus before running `preprocess`.

---

## Corpus file format

The binary `.corp` file produced by `preprocess` has the following layout. The magic bytes are `CRP2`.

```
[4 bytes]  magic "CRP2"
[u32]      number of word types
[u32]      number of documents

for each word type:
  [u16 + bytes]  word (UTF-8)
  [u32]          document frequency
  [u32]          total frequency

for each document:
  [u16 + bytes]  document name (UTF-8)
  [u16 + bytes]  document label (UTF-8, empty string if none)
  [u32]          number of tokens
  [u32 × n]      token IDs
```

All integers are little-endian.

---

## Algorithm

### Gibbs sampling

The sampler is a direct port of MALLET's `WorkerRunnable.sampleTopicsForOneDoc`, implementing the three-bucket decomposition of the LDA conditional:

```
P(z=t | rest) ∝  α_t · β / (n_t + βW)                    [smoothing-only]
              +  β · n_{t,d} / (n_t + βW)                 [topic-beta]
              +  (α_t + n_{t,d}) · n_{w,t} / (n_t + βW)  [topic-term]
```

The smoothing-only bucket is word-invariant and accumulated once per document. The topic-beta bucket iterates only over topics present in the document. The topic-term bucket only iterates over topics where the current word has a non-zero count. Word-topic counts are stored as sorted packed integers `(count << topic_bits) | topic` in descending order, so the most probable topics appear first and the scan terminates at the first zero entry.

### Hyperparameter optimization

Alpha and beta are updated using Minka's fixed-point method. Each update requires one pass over the current topic assignments to build count histograms, then evaluates the digamma function at each distinct count value.

**Alpha** (asymmetric, per-topic): for each topic *t*,

```
α_t ← α_t · Σ_d (Ψ(n_{t,d} + α_t) − Ψ(α_t))
           / Σ_d (Ψ(N_d + α_sum) − Ψ(α_sum))
```

**Beta** (symmetric): treating each topic as one observation over *W* word types,

```
β_sum ← β_sum · Σ_c h_w[c] · (Ψ(c + β) − Ψ(β))
               / W · Σ_s h_t[s] · (Ψ(s + β_sum) − Ψ(β_sum))
```

where *h_w[c]* is the number of (word, topic) pairs with count *c*, and *h_t[s]* is the number of topics with *s* total tokens.

### Output estimation

After training, the sampler runs for `num_samples × sample_interval` additional iterations. At each `sample_interval` boundary the smoothed distributions φ_{w,t} and θ_{t,d} are recorded, then averaged across all samples. This reduces the variance of the point estimate relative to reading from a single final state.

---

## References

- Yao, L., Mimno, D., & McCallum, A. (2009). Efficient methods for topic model inference on streaming document collections. *KDD*.
- Blei, D., Ng, A., & Jordan, M. (2003). Latent Dirichlet allocation. *JMLR*.
- Minka, T. (2000). Estimating a Dirichlet distribution. Technical report, MIT.
