use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashSet;
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{corpus as corp, model, optimize, output, sampler};

#[pyclass]
pub struct Corpus {
    pub(crate) inner: corp::Corpus,
}

#[pymethods]
impl Corpus {
    /// Load from a plain text file (one document per line).
    #[staticmethod]
    #[pyo3(signature = (path, *, stopwords=None, min_doc_freq=1, max_doc_fraction=1.0, id_field=false, token_regex=None))]
    fn from_text_file(
        path: String,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        id_field: bool,
        token_regex: Option<String>,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        let opts = corp::LoadOptions {
            format: corp::InputFormat::Plain { id_field },
            token_regex: token_regex.unwrap_or_else(|| corp::DEFAULT_TOKEN_REGEX.to_string()),
            stopwords: stop,
            min_doc_freq,
            max_doc_fraction,
        };
        corp::load_text_file(Path::new(&path), &opts)
            .map(|inner| Corpus { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load from a tab-separated file (e.g. MALLET's id TAB label TAB text layout).
    #[staticmethod]
    #[pyo3(signature = (path, *, id_column=0, label_column=None, text_column=1, stopwords=None, min_doc_freq=1, max_doc_fraction=1.0, token_regex=None))]
    fn from_tsv_file(
        path: String,
        id_column: usize,
        label_column: Option<usize>,
        text_column: usize,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        token_regex: Option<String>,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        let opts = corp::LoadOptions {
            format: corp::InputFormat::Tsv { id_column, label_column, text_column },
            token_regex: token_regex.unwrap_or_else(|| corp::DEFAULT_TOKEN_REGEX.to_string()),
            stopwords: stop,
            min_doc_freq,
            max_doc_fraction,
        };
        corp::load_text_file(Path::new(&path), &opts)
            .map(|inner| Corpus { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a preprocessed binary corpus file produced by `preprocess` or `corpus.save()`.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        corp::load_corpus(Path::new(&path))
            .map(|inner| Corpus { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Save corpus to a binary file that can be reloaded with `Corpus.load()`.
    fn save(&self, path: String) -> PyResult<()> {
        corp::save_corpus(&self.inner, Path::new(&path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn num_docs(&self) -> usize { self.inner.num_docs() }

    #[getter]
    fn num_types(&self) -> usize { self.inner.num_types() }

    #[getter]
    fn total_tokens(&self) -> usize { self.inner.total_tokens() }

    /// Ordered vocabulary list: vocabulary[word_id] == word string.
    #[getter]
    fn vocabulary(&self) -> Vec<String> { self.inner.id_to_word.clone() }

    #[getter]
    fn doc_names(&self) -> Vec<String> { self.inner.doc_names.clone() }

    #[getter]
    fn doc_labels(&self) -> Vec<String> { self.inner.doc_labels.clone() }

    /// Build a Corpus directly from a list of text strings (no file I/O).
    ///
    /// Args:
    ///   docs: List of raw document text strings.
    ///   doc_ids: Optional list of document names; defaults to "doc_0", "doc_1", ...
    ///   stopwords: Words to exclude during tokenisation.
    ///   min_doc_freq: Drop words appearing in fewer than N documents.
    ///   max_doc_fraction: Drop words appearing in more than this fraction of documents.
    ///   token_regex: Override the default tokenisation regex.
    #[staticmethod]
    #[pyo3(signature = (docs, doc_ids=None, stopwords=None, min_doc_freq=1, max_doc_fraction=1.0, token_regex=None))]
    fn from_strings(
        docs: Vec<String>,
        doc_ids: Option<Vec<String>>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        token_regex: Option<String>,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.to_lowercase())
            .collect();
        let opts = corp::LoadOptions {
            format: corp::InputFormat::Plain { id_field: false },
            token_regex: token_regex.unwrap_or_else(|| corp::DEFAULT_TOKEN_REGEX.to_string()),
            stopwords: stop,
            min_doc_freq,
            max_doc_fraction,
        };

        let inputs: Vec<(String, String, String)> = docs
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let name = doc_ids.as_ref()
                    .and_then(|ids| ids.get(i))
                    .cloned()
                    .unwrap_or_else(|| format!("doc_{}", i));
                (name, String::new(), text)
            })
            .collect();

        corp::load_from_strings(&inputs, &opts)
            .map(|inner| Corpus { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Build a Corpus from a document-term count matrix.
    ///
    /// Args:
    ///   x: List[List[float]] of shape [n_docs, n_words], where x[i][j] is the
    ///      count of word j in document i.
    ///   vocabulary: optional list of word strings (length must equal n_words).
    ///               Defaults to "0", "1", ... if omitted.
    #[staticmethod]
    #[pyo3(signature = (x, vocabulary=None))]
    fn from_dtm(x: Vec<Vec<f64>>, vocabulary: Option<Vec<String>>) -> PyResult<Self> {
        if x.is_empty() {
            return Err(PyValueError::new_err("X must not be empty"));
        }
        let n_words = x[0].len();

        let id_to_word = match vocabulary {
            Some(v) if v.len() != n_words => {
                return Err(PyValueError::new_err(format!(
                    "vocabulary length ({}) must equal number of columns in X ({})",
                    v.len(), n_words
                )));
            }
            Some(v) => v,
            None => (0..n_words).map(|i| i.to_string()).collect(),
        };

        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut doc_names: Vec<String> = Vec::new();
        let mut doc_labels: Vec<String> = Vec::new();
        let mut total_freqs = vec![0u32; n_words];
        let mut doc_freqs = vec![0u32; n_words];

        for (i, row) in x.iter().enumerate() {
            if row.len() != n_words {
                return Err(PyValueError::new_err(format!(
                    "row {} has {} columns, expected {}", i, row.len(), n_words
                )));
            }
            let mut tokens: Vec<u32> = Vec::new();
            for (word_id, &count) in row.iter().enumerate() {
                let c = count.round() as u32;
                for _ in 0..c {
                    tokens.push(word_id as u32);
                }
                if c > 0 {
                    total_freqs[word_id] += c;
                    doc_freqs[word_id] += 1;
                }
            }
            if !tokens.is_empty() {
                docs.push(tokens);
                doc_names.push(format!("doc_{}", i));
                doc_labels.push(String::new());
            }
        }

        Ok(Corpus {
            inner: corp::Corpus { id_to_word, docs, doc_names, doc_labels, doc_freqs, total_freqs },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Corpus(num_docs={}, num_types={}, total_tokens={})",
            self.inner.num_docs(),
            self.inner.num_types(),
            self.inner.total_tokens(),
        )
    }
}

#[pyclass]
pub struct TopicModel {
    model: model::TopicModel,
    vocabulary: Vec<String>,
    doc_names: Vec<String>,
    doc_labels: Vec<String>,
    /// phi[word_id][topic] — averaged topic-word probabilities
    phi: Vec<Vec<f64>>,
    /// theta[doc_idx][topic] — averaged document-topic probabilities
    theta: Vec<Vec<f64>>,
}

#[pymethods]
impl TopicModel {
    #[getter]
    fn num_topics(&self) -> usize { self.model.num_topics }

    #[getter]
    fn num_types(&self) -> usize { self.model.num_types }

    #[getter]
    fn num_docs(&self) -> usize { self.theta.len() }

    /// Ordered vocabulary list: vocabulary[word_id] == word string.
    #[getter]
    fn vocabulary(&self) -> Vec<String> { self.vocabulary.clone() }

    #[getter]
    fn doc_names(&self) -> Vec<String> { self.doc_names.clone() }

    #[getter]
    fn doc_labels(&self) -> Vec<String> { self.doc_labels.clone() }

    /// Per-topic alpha values (asymmetric Dirichlet).
    #[getter]
    fn alpha(&self) -> Vec<f64> { self.model.alpha.clone() }

    /// Symmetric word prior beta.
    #[getter]
    fn beta(&self) -> f64 { self.model.beta }

    /// Return the top `n` words for each topic as a list of word lists.
    ///
    /// Returns: List[List[str]], one inner list per topic.
    #[pyo3(signature = (n=10))]
    fn top_words(&self, n: usize) -> Vec<Vec<String>> {
        let num_topics = self.model.num_topics;
        let num_types = self.model.num_types;
        (0..num_topics).map(|topic| {
            let mut word_probs: Vec<(f64, usize)> = (0..num_types)
                .map(|word_id| (self.phi[word_id][topic], word_id))
                .collect();
            word_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            word_probs.iter().take(n)
                .map(|(_, id)| self.vocabulary[*id].clone())
                .collect()
        }).collect()
    }

    /// Full topic-word probability matrix.
    ///
    /// Returns: List[List[float]] of shape [num_topics][num_types].
    /// Each row sums to approximately 1.0.
    fn topic_word_matrix(&self) -> Vec<Vec<f64>> {
        let num_topics = self.model.num_topics;
        let num_types = self.model.num_types;
        let mut result = vec![vec![0.0f64; num_types]; num_topics];
        // word_id in outer loop: each phi[word_id] row is loaded once and
        // read sequentially, avoiding num_types cache misses per topic.
        for word_id in 0..num_types {
            for topic in 0..num_topics {
                result[topic][word_id] = self.phi[word_id][topic];
            }
        }
        result
    }

    /// Full document-topic probability matrix.
    ///
    /// Returns: List[List[float]] of shape [num_docs][num_topics].
    /// Each row sums to approximately 1.0.
    fn doc_topic_matrix(&self) -> Vec<Vec<f64>> {
        self.theta.clone()
    }

    /// Topic distribution for a single document.
    ///
    /// Returns: List[float] of length num_topics.
    fn get_doc_topics(&self, doc_idx: usize) -> PyResult<Vec<f64>> {
        if doc_idx >= self.theta.len() {
            return Err(PyValueError::new_err(format!(
                "doc_idx {} out of range ({} documents)",
                doc_idx, self.theta.len()
            )));
        }
        Ok(self.theta[doc_idx].clone())
    }

    /// Compute model log-likelihood on a corpus.
    fn log_likelihood(&self, corpus: &Corpus) -> f64 {
        output::model_log_likelihood(&self.model, &corpus.inner)
    }

    /// Infer topic distributions for documents in a count matrix.
    ///
    /// Runs Gibbs sampling for `n_iter` iterations with the learned
    /// topic-word distribution held fixed.
    ///
    /// Args:
    ///   x: List[List[float]] of shape [n_docs, n_words] (count matrix).
    ///   n_iter: Gibbs sampling iterations per document.
    ///   seed: random seed.
    ///
    /// Returns: List[List[float]] of shape [n_docs, num_topics].
    fn infer(&self, x: Vec<Vec<f64>>, n_iter: usize, seed: u64) -> PyResult<Vec<Vec<f64>>> {
        let num_topics = self.model.num_topics;
        let alpha = &self.model.alpha;
        let alpha_sum = self.model.alpha_sum;
        let n_words = self.model.num_types;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut results = Vec::with_capacity(x.len());

        for row in &x {
            let mut tokens: Vec<usize> = Vec::new();
            for (word_id, &count) in row.iter().enumerate() {
                if word_id >= n_words { continue; }
                let c = count.round() as usize;
                for _ in 0..c {
                    tokens.push(word_id);
                }
            }

            if tokens.is_empty() {
                // Return prior for empty documents.
                results.push(alpha.iter().map(|&a| a / alpha_sum).collect());
                continue;
            }

            let mut assignments: Vec<usize> = tokens.iter()
                .map(|_| rng.gen_range(0..num_topics))
                .collect();
            let mut counts = vec![0u32; num_topics];
            for &t in &assignments { counts[t] += 1; }

            for _ in 0..n_iter {
                for pos in 0..tokens.len() {
                    let word_id = tokens[pos];
                    let old_topic = assignments[pos];
                    counts[old_topic] -= 1;

                    let mut total = 0.0f64;
                    let mut probs = vec![0.0f64; num_topics];
                    for t in 0..num_topics {
                        let p = (counts[t] as f64 + alpha[t]) * self.phi[word_id][t];
                        probs[t] = p;
                        total += p;
                    }

                    let new_topic = if total > 0.0 {
                        let mut sample = rng.gen::<f64>() * total;
                        let mut chosen = num_topics - 1;
                        for t in 0..num_topics {
                            sample -= probs[t];
                            if sample <= 0.0 { chosen = t; break; }
                        }
                        chosen
                    } else {
                        rng.gen_range(0..num_topics)
                    };

                    assignments[pos] = new_topic;
                    counts[new_topic] += 1;
                }
            }

            let denom = tokens.len() as f64 + alpha_sum;
            results.push(
                (0..num_topics).map(|t| (counts[t] as f64 + alpha[t]) / denom).collect()
            );
        }

        Ok(results)
    }

    fn __repr__(&self) -> String {
        format!(
            "TopicModel(num_topics={}, num_types={}, num_docs={})",
            self.model.num_topics,
            self.model.num_types,
            self.theta.len(),
        )
    }
}

/// Train an LDA topic model on `corpus` and return a fitted TopicModel.
///
/// Args:
///   corpus: A Corpus object produced by Corpus.from_text_file(), Corpus.from_tsv_file(),
///           or Corpus.load().
///   num_topics: Number of topics (default 10).
///   iterations: Number of Gibbs sampling iterations (default 1000).
///   burn_in: Iterations before hyperparameter optimization begins (default 200).
///   optimize_interval: Optimize alpha and beta every N iterations after burn-in
///                      (default 50; set 0 to disable).
///   num_samples: Samples to average for final probability estimates (default 5).
///   sample_interval: Gibbs iterations between samples (default 25).
///   alpha_sum: Initial symmetric Dirichlet alpha sum (default num_topics).
///   beta: Initial Dirichlet beta per word (default 0.01).
///   seed: Random seed for reproducibility (default 42).
///   verbose: Print log-likelihood progress every 50 iterations (default False).
#[pyfunction]
#[pyo3(signature = (
    corpus,
    num_topics=10,
    iterations=1000,
    burn_in=200,
    optimize_interval=50,
    num_samples=5,
    sample_interval=25,
    alpha_sum=None,
    beta=0.01,
    seed=42,
    verbose=false,
))]
pub fn train(
    corpus: &Corpus,
    num_topics: usize,
    iterations: usize,
    burn_in: usize,
    optimize_interval: usize,
    num_samples: usize,
    sample_interval: usize,
    alpha_sum: Option<f64>,
    beta: f64,
    seed: u64,
    verbose: bool,
) -> PyResult<TopicModel> {
    let c = &corpus.inner;

    if c.num_docs() == 0 {
        return Err(PyValueError::new_err("corpus contains no documents"));
    }
    if num_samples == 0 {
        return Err(PyValueError::new_err("num_samples must be >= 1"));
    }

    let alpha_sum = alpha_sum.unwrap_or(num_topics as f64);
    let mut m = model::TopicModel::new(num_topics, alpha_sum, beta, c.num_types());

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    m.initialize(c, &mut rng);

    for iter in 1..=iterations {
        sampler::run_iteration(&mut m, c, &mut rng);

        if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
            optimize::optimize_alpha(&mut m, c);
            optimize::optimize_beta(&mut m);
        }

        if verbose && iter % 50 == 0 {
            let ll = output::model_log_likelihood(&m, c);
            eprintln!("<{}> LL/token: {:.5}", iter, ll / c.total_tokens() as f64);
        }
    }

    let mut acc_phi   = vec![vec![0.0f64; num_topics]; m.num_types];
    let mut acc_theta = vec![vec![0.0f64; num_topics]; c.num_docs()];
    let mut counts    = vec![0u32; num_topics];

    for _ in 0..num_samples {
        for _ in 0..sample_interval {
            sampler::run_iteration(&mut m, c, &mut rng);
        }

        for word_id in 0..m.num_types {
            for topic in 0..num_topics {
                let count = m.get_type_topic_count(word_id, topic);
                let denom = m.tokens_per_topic[topic] as f64 + m.beta_sum;
                acc_phi[word_id][topic] += (count as f64 + m.beta) / denom;
            }
        }

        for doc_idx in 0..c.num_docs() {
            for t in 0..num_topics { counts[t] = 0; }
            for &t in &m.doc_topics[doc_idx] { counts[t as usize] += 1; }
            let denom = c.docs[doc_idx].len() as f64 + m.alpha_sum;
            for t in 0..num_topics {
                acc_theta[doc_idx][t] += (counts[t] as f64 + m.alpha[t]) / denom;
            }
        }
    }

    let n = num_samples as f64;
    for row in acc_phi.iter_mut()   { for v in row.iter_mut() { *v /= n; } }
    for row in acc_theta.iter_mut() { for v in row.iter_mut() { *v /= n; } }

    Ok(TopicModel {
        model: m,
        vocabulary: c.id_to_word.clone(),
        doc_names: c.doc_names.clone(),
        doc_labels: c.doc_labels.clone(),
        phi: acc_phi,
        theta: acc_theta,
    })
}

/// Load a stopword list from a file (one word per line, # comments ignored).
///
/// Returns: List[str]
#[pyfunction]
pub fn load_stopwords(path: String) -> PyResult<Vec<String>> {
    corp::load_stoplist(Path::new(&path))
        .map(|set| set.into_iter().collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Corpus>()?;
    m.add_class::<TopicModel>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(load_stopwords, m)?)?;
    Ok(())
}
