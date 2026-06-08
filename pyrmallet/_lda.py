import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from ._binaries import find_binary
from ._io import parse_doc_topic_tsv, parse_topic_word_tsv, write_stoplist, write_text_file

try:
    from . import _rust_mallet as _rm
    _NATIVE = True
except ImportError:
    _rm = None
    _NATIVE = False


class LatentDirichletAllocation:
    """LDA via RustMallet's sparse Gibbs sampler, matching sklearn's API.

    Parameters
    ----------
    n_components : int
        Number of topics (--num-topics).
    doc_topic_prior : float or None
        Symmetric Dirichlet alpha sum (--alpha-sum). Defaults to n_components.
    topic_word_prior : float
        Dirichlet beta per word (--beta). Default 0.01.
    max_iter : int
        Gibbs sampling iterations (--iterations). Default 1000.
    random_state : int
        Random seed (--seed). Default 42.
    verbose : bool
        Echo RustMallet stderr to sys.stderr during fit(). Default False.
    burn_in : int
        Iterations before hyperparameter optimization (--burn-in). Default 200.
    optimize_interval : int
        Optimize alpha/beta every N iterations after burn-in. Set 0 to disable.
    num_samples : int
        Samples averaged for final distribution estimates (--num-samples).
    sample_interval : int
        Gibbs iterations between samples (--sample-interval).
    stopwords : list of str, or path str, or None
        Stopword list passed to preprocess. If a list, written to a temp file.
        If a str, treated as a path to an existing stoplist file.
    min_doc_freq : int
        Drop words appearing in fewer than N documents (--min-doc-freq).
    max_doc_fraction : float
        Drop words appearing in more than this fraction of documents.
    workdir : str or None
        Directory for intermediate files. None uses a temp dir (cleaned up
        after fit). Set to a path to inspect intermediate files.
        Ignored when the native PyO3 extension is available.

    Attributes (set after fit)
    --------------------------
    components_ : ndarray of shape (n_topics, n_vocab)
        Topic-word probability distributions.
    doc_topic_distributions_ : ndarray of shape (n_docs, n_topics)
        Document-topic distributions for the training corpus.
    feature_names_in_ : ndarray of str, shape (n_vocab,)
        Vocabulary in corpus order.
    n_features_in_ : int
        Vocabulary size.
    """

    def __init__(
        self,
        n_components: int = 10,
        doc_topic_prior: Optional[float] = None,
        topic_word_prior: float = 0.01,
        max_iter: int = 1000,
        random_state: int = 42,
        *,
        verbose: bool = False,
        burn_in: int = 200,
        optimize_interval: int = 50,
        num_samples: int = 5,
        sample_interval: int = 25,
        stopwords=None,
        min_doc_freq: int = 1,
        max_doc_fraction: float = 1.0,
        workdir: Optional[str] = None,
    ):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.burn_in = burn_in
        self.optimize_interval = optimize_interval
        self.num_samples = num_samples
        self.sample_interval = sample_interval
        self.stopwords = stopwords
        self.min_doc_freq = min_doc_freq
        self.max_doc_fraction = max_doc_fraction
        self.workdir = workdir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "LatentDirichletAllocation":
        """Fit LDA on a corpus of raw text documents.

        Parameters
        ----------
        X : list of str
            One document per element.
        y : ignored

        Returns
        -------
        self
        """
        if not isinstance(X, list) or not all(isinstance(d, str) for d in X):
            raise ValueError(
                "X must be a list of strings (one document per element). "
                "pyrmallet does not accept document-term matrices; pass raw text."
            )
        if len(X) == 0:
            raise ValueError("X must contain at least one document.")

        if _NATIVE:
            return self._fit_native(X)
        return self._fit_subprocess(X)

    def transform(self, X) -> np.ndarray:
        """Return document-topic distributions.

        Only the training corpus distributions are available. This method
        returns the cached doc_topic_distributions_ regardless of X.
        Inference on new (unseen) documents is not supported; call fit() on
        the expanded corpus instead.

        Parameters
        ----------
        X : ignored (must have same length as training corpus)

        Returns
        -------
        ndarray of shape (n_docs, n_topics)
        """
        self._check_fitted()
        if not isinstance(X, list) or len(X) != len(self.doc_topic_distributions_):
            raise NotImplementedError(
                "pyrmallet does not support inference on unseen documents. "
                "To get distributions for new documents, call fit() on the full "
                "dataset (training + new documents)."
            )
        return self.doc_topic_distributions_

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and return document-topic distributions for X.

        Parameters
        ----------
        X : list of str
        y : ignored

        Returns
        -------
        ndarray of shape (n_docs, n_topics)
        """
        return self.fit(X, y).doc_topic_distributions_

    def score(self, X, y=None) -> float:
        raise NotImplementedError(
            "pyrmallet does not expose log-likelihood after training. "
            "Use verbose=True to see LL/token progress during fit()."
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_components": self.n_components,
            "doc_topic_prior": self.doc_topic_prior,
            "topic_word_prior": self.topic_word_prior,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "burn_in": self.burn_in,
            "optimize_interval": self.optimize_interval,
            "num_samples": self.num_samples,
            "sample_interval": self.sample_interval,
            "stopwords": self.stopwords,
            "min_doc_freq": self.min_doc_freq,
            "max_doc_fraction": self.max_doc_fraction,
            "workdir": self.workdir,
        }

    def set_params(self, **params) -> "LatentDirichletAllocation":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}'.")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError(
                "This LatentDirichletAllocation instance is not fitted yet. "
                "Call fit() before transform()."
            )

    def _fit_native(self, X: list) -> "LatentDirichletAllocation":
        """Fit using the native PyO3 extension — no subprocesses or disk I/O."""
        stop_list: List[str] = []
        if isinstance(self.stopwords, (list, tuple, set)):
            stop_list = list(self.stopwords)
        elif isinstance(self.stopwords, str):
            stop_list = _rm.load_stopwords(self.stopwords)

        doc_ids = [f"doc_{i}" for i in range(len(X))]
        corpus = _rm.Corpus.from_strings(
            X,
            doc_ids=doc_ids,
            stopwords=stop_list,
            min_doc_freq=self.min_doc_freq,
            max_doc_fraction=self.max_doc_fraction,
        )

        alpha_sum = (
            float(self.doc_topic_prior)
            if self.doc_topic_prior is not None
            else float(self.n_components)
        )
        model = _rm.train(
            corpus,
            num_topics=self.n_components,
            iterations=self.max_iter,
            burn_in=self.burn_in,
            optimize_interval=self.optimize_interval,
            num_samples=self.num_samples,
            sample_interval=self.sample_interval,
            alpha_sum=alpha_sum,
            beta=self.topic_word_prior,
            seed=self.random_state,
            verbose=self.verbose,
        )

        self.feature_names_in_ = np.array(model.vocabulary)
        self.n_features_in_ = model.num_types
        self.components_ = np.array(model.topic_word_matrix())
        self.doc_topic_distributions_ = np.array(model.doc_topic_matrix())
        return self

    def _fit_subprocess(self, X: list) -> "LatentDirichletAllocation":
        """Fit by shelling out to the preprocess and train CLI binaries."""
        preprocess_bin = find_binary("preprocess")
        train_bin = find_binary("train")

        tmpdir = None
        try:
            if self.workdir is not None:
                work = Path(self.workdir)
                work.mkdir(parents=True, exist_ok=True)
            else:
                tmpdir = tempfile.mkdtemp(prefix="pyrmallet_")
                work = Path(tmpdir)

            input_path = work / "input.txt"
            corpus_path = work / "corpus.corp"
            tw_path = work / "topic_word.tsv"
            dt_path = work / "doc_topic.tsv"

            doc_ids = [f"doc_{i}" for i in range(len(X))]
            write_text_file(X, input_path, doc_ids)

            pre_cmd = [
                preprocess_bin,
                "--input", str(input_path),
                "--output", str(corpus_path),
                "--format", "tsv",
                "--id-column", "0",
                "--no-label",
                "--text-column", "1",
                "--min-doc-freq", str(self.min_doc_freq),
                "--max-doc-fraction", str(self.max_doc_fraction),
            ]

            if self.stopwords is not None:
                if isinstance(self.stopwords, (list, tuple, set)):
                    sl_path = work / "stopwords.txt"
                    write_stoplist(self.stopwords, sl_path)
                    pre_cmd += ["--stoplist", str(sl_path)]
                else:
                    pre_cmd += ["--stoplist", str(self.stopwords)]

            self._run("preprocess", pre_cmd)

            alpha_sum = (
                float(self.doc_topic_prior)
                if self.doc_topic_prior is not None
                else float(self.n_components)
            )
            train_cmd = [
                train_bin,
                "--corpus", str(corpus_path),
                "--num-topics", str(self.n_components),
                "--iterations", str(self.max_iter),
                "--burn-in", str(self.burn_in),
                "--optimize-interval", str(self.optimize_interval),
                "--num-samples", str(self.num_samples),
                "--sample-interval", str(self.sample_interval),
                "--alpha-sum", str(alpha_sum),
                "--beta", str(self.topic_word_prior),
                "--seed", str(self.random_state),
                "--topic-word", str(tw_path),
                "--doc-topic", str(dt_path),
            ]

            self._run("train", train_cmd)

            vocabulary, components = parse_topic_word_tsv(tw_path)
            _, _, doc_topic = parse_doc_topic_tsv(dt_path)

            self.feature_names_in_ = np.array(vocabulary)
            self.n_features_in_ = len(vocabulary)
            self.components_ = components
            self.doc_topic_distributions_ = doc_topic

        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)

        return self

    def _run(self, name: str, cmd: list) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if self.verbose and result.stderr:
            sys.stderr.write(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"RustMallet '{name}' failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )
