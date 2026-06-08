from typing import List, Optional

import numpy as np

try:
    from . import _rust_mallet as _rm
except ImportError as e:
    raise ImportError(
        "pyrmallet native extension not found. "
        "Build it with:\n"
        '  PATH="$HOME/.cargo/bin:$PATH" uv run --with maturin maturin develop'
    ) from e


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
        Echo RustMallet progress to stderr during fit(). Default False.
    burn_in : int
        Iterations before hyperparameter optimization (--burn-in). Default 200.
    optimize_interval : int
        Optimize alpha/beta every N iterations after burn-in. Set 0 to disable.
    num_samples : int
        Samples averaged for final distribution estimates (--num-samples).
    sample_interval : int
        Gibbs iterations between samples (--sample-interval).
    n_inference_iter : int
        Gibbs iterations per document during transform(). Default 50.
    stopwords : list of str, path str, or None
        Stopword list. A list is used directly; a str is treated as a path.
    min_doc_freq : int
        Drop words appearing in fewer than N documents.
    max_doc_fraction : float
        Drop words appearing in more than this fraction of documents.

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
        n_inference_iter: int = 50,
        stopwords=None,
        min_doc_freq: int = 1,
        max_doc_fraction: float = 1.0,
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
        self.n_inference_iter = n_inference_iter
        self.stopwords = stopwords
        self.min_doc_freq = min_doc_freq
        self.max_doc_fraction = max_doc_fraction

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
        self._model = _rm.train(
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

        self.feature_names_in_ = np.array(self._model.vocabulary)
        self.n_features_in_ = self._model.num_types
        self.components_ = np.array(self._model.topic_word_matrix())
        self.doc_topic_distributions_ = np.array(self._model.doc_topic_matrix())
        return self

    def transform(self, X) -> np.ndarray:
        """Infer topic distributions for new documents.

        Tokenises each document using the same regex and vocabulary as
        training; words not seen during training are ignored.

        Parameters
        ----------
        X : list of str
            Raw text documents (need not overlap with the training corpus).

        Returns
        -------
        ndarray of shape (n_docs, n_topics)
        """
        self._check_fitted()
        if not isinstance(X, list) or not all(isinstance(d, str) for d in X):
            raise ValueError("X must be a list of strings.")
        return np.array(
            self._model.infer_strings(
                X,
                n_iter=self.n_inference_iter,
                seed=self.random_state,
            )
        )

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
            "n_inference_iter": self.n_inference_iter,
            "stopwords": self.stopwords,
            "min_doc_freq": self.min_doc_freq,
            "max_doc_fraction": self.max_doc_fraction,
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
        if not hasattr(self, "_model"):
            raise RuntimeError(
                "This LatentDirichletAllocation instance is not fitted yet. "
                "Call fit() before transform()."
            )
