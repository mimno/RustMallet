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

## Building

```bash
cargo build --release
```

Requires [Rust](https://rustup.rs). Binaries are written to `target/release/`.

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
