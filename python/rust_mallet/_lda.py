try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _bases = (BaseEstimator, TransformerMixin)
except ImportError:
    _bases = (object,)


def _to_list_of_lists(X):
    """Convert X (numpy array, scipy sparse, or list of lists) to List[List[float]]."""
    if hasattr(X, 'toarray'):
        X = X.toarray()
    if hasattr(X, 'tolist'):
        return X.tolist()
    return [list(row) for row in X]


class LDA(*_bases):
    """
    Latent Dirichlet Allocation with a scikit-learn compatible interface.

    Wraps RustMallet's fast collapsed Gibbs sampler. Accepts the same
    document-term count matrices produced by sklearn's CountVectorizer.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.
    n_iter : int, default=1000
        Number of Gibbs sampling iterations for training.
    burn_in : int, default=200
        Iterations before hyperparameter optimization begins.
    optimize_interval : int, default=50
        Optimize alpha and beta every N iterations after burn-in.
        Set 0 to disable.
    n_samples : int, default=5
        Number of Gibbs samples averaged for the final estimates.
    sample_interval : int, default=25
        Iterations between samples.
    alpha_sum : float or None, default=None
        Initial symmetric Dirichlet alpha sum. Defaults to n_components.
    beta : float, default=0.01
        Symmetric word prior (Dirichlet beta per word type).
    random_state : int, default=42
        Random seed for reproducibility.
    n_inference_iter : int, default=100
        Gibbs iterations when inferring topics for new documents in
        transform(). More iterations give more stable estimates.

    Attributes (set after fit)
    --------------------------
    components_ : list of list of float, shape [n_components, n_features_in_]
        Topic-word probability matrix; each row sums to ~1.0.
        (Unlike sklearn's LDA, these are normalized probabilities.)
    doc_topic_prior_ : list of float, length n_components
        Per-topic alpha values after hyperparameter optimization.
    topic_word_prior_ : float
        Beta value after hyperparameter optimization.
    n_features_in_ : int
        Vocabulary size seen during fit.

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from rust_mallet import LDA
    >>> docs = ["the cat sat", "the dog ran", "cats and dogs"]
    >>> vec = CountVectorizer()
    >>> X = vec.fit_transform(docs)
    >>> lda = LDA(n_components=2, n_iter=100, n_samples=2)
    >>> lda.fit(X)
    LDA(...)
    >>> lda.transform(X)  # shape [n_docs, n_components]
    [...]
    """

    def __init__(
        self,
        n_components=10,
        *,
        n_iter=1000,
        burn_in=200,
        optimize_interval=50,
        n_samples=5,
        sample_interval=25,
        alpha_sum=None,
        beta=0.01,
        random_state=42,
        n_inference_iter=100,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.optimize_interval = optimize_interval
        self.n_samples = n_samples
        self.sample_interval = sample_interval
        self.alpha_sum = alpha_sum
        self.beta = beta
        self.random_state = random_state
        self.n_inference_iter = n_inference_iter

    def fit(self, X, y=None):
        """
        Fit LDA model on document-term matrix X.

        Parameters
        ----------
        X : array-like of shape [n_docs, n_words]
            Document-term count matrix. Accepts numpy arrays, scipy sparse
            matrices (CSR/CSC), or Python lists of lists.
        y : ignored

        Returns
        -------
        self
        """
        from ._rust_mallet import Corpus, train

        corpus = Corpus.from_dtm(_to_list_of_lists(X))
        self._model = train(
            corpus,
            num_topics=self.n_components,
            iterations=self.n_iter,
            burn_in=self.burn_in,
            optimize_interval=self.optimize_interval,
            num_samples=self.n_samples,
            sample_interval=self.sample_interval,
            alpha_sum=self.alpha_sum,
            beta=self.beta,
            seed=self.random_state,
        )
        self.components_ = self._model.topic_word_matrix()
        self.doc_topic_prior_ = self._model.alpha
        self.topic_word_prior_ = self._model.beta
        self.n_features_in_ = self._model.num_types
        return self

    def transform(self, X):
        """
        Infer topic distributions for documents in X.

        Runs Gibbs sampling for n_inference_iter iterations with the
        learned topic-word distribution held fixed.

        Parameters
        ----------
        X : array-like of shape [n_docs, n_words]
            Document-term count matrix (same vocabulary as training).

        Returns
        -------
        list of list of float, shape [n_docs, n_components]
            Document-topic probability matrix; each row sums to ~1.0.
        """
        self._check_fitted()
        return self._model.infer(
            _to_list_of_lists(X), self.n_inference_iter, self.random_state
        )

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit and return the training document-topic matrix.

        Returns the stored training theta (the averaged sample from
        training, which is more accurate than re-running inference).

        Returns
        -------
        list of list of float, shape [n_docs, n_components]
        """
        self.fit(X, y)
        return self._model.doc_topic_matrix()

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'n_iter': self.n_iter,
            'burn_in': self.burn_in,
            'optimize_interval': self.optimize_interval,
            'n_samples': self.n_samples,
            'sample_interval': self.sample_interval,
            'alpha_sum': self.alpha_sum,
            'beta': self.beta,
            'random_state': self.random_state,
            'n_inference_iter': self.n_inference_iter,
        }

    def set_params(self, **params):
        valid = self.get_params()
        for k, v in params.items():
            if k not in valid:
                raise ValueError(
                    f"Invalid parameter {k!r} for LDA. "
                    f"Valid parameters are: {sorted(valid)}"
                )
            setattr(self, k, v)
        return self

    def _check_fitted(self):
        if not hasattr(self, '_model'):
            raise RuntimeError(
                "This LDA instance is not fitted yet. Call fit() first."
            )

    def __repr__(self):
        params = ', '.join(f'{k}={v!r}' for k, v in self.get_params().items())
        return f'LDA({params})'
