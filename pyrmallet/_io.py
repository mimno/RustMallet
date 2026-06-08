from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def write_text_file(
    docs: List[str],
    path: Path,
    doc_ids: Optional[List[str]] = None,
) -> None:
    if doc_ids is None:
        doc_ids = [f"doc_{i}" for i in range(len(docs))]
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, text in zip(doc_ids, docs):
            f.write(f"{doc_id}\t{text}\n")


def write_stoplist(words: Iterable[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")


def parse_topic_word_tsv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read topic_word.tsv and return (vocabulary, components).

    components has shape (n_topics, n_vocab); columns are ordered by first
    appearance of each word (which is vocabulary order from the corpus).
    """
    word_to_idx: dict = {}
    entries: list = []
    n_topics = 0

    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            topic = int(parts[0])
            word = parts[1]
            prob = float(parts[2])
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
            entries.append((topic, word_to_idx[word], prob))
            if topic + 1 > n_topics:
                n_topics = topic + 1

    n_vocab = len(word_to_idx)
    components = np.zeros((n_topics, n_vocab), dtype=np.float64)
    for topic, word_idx, prob in entries:
        components[topic, word_idx] = prob

    vocabulary = [""] * n_vocab
    for word, idx in word_to_idx.items():
        vocabulary[idx] = word

    return vocabulary, components


def parse_doc_topic_tsv(
    path: Path,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Read doc_topic.tsv and return (doc_ids, labels, matrix).

    matrix has shape (n_docs, n_topics). labels is an empty list when the
    corpus had no label column.
    """
    doc_ids: List[str] = []
    labels: List[str] = []
    rows: list = []

    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        has_label = len(header) >= 2 and not header[1].startswith("topic_")
        topic_start = 2 if has_label else 1
        n_topics = len(header) - topic_start

        for line in f:
            parts = line.rstrip("\n").split("\t")
            doc_ids.append(parts[0])
            labels.append(parts[1] if has_label else "")
            rows.append([float(p) for p in parts[topic_start : topic_start + n_topics]])

    return doc_ids, labels, np.array(rows, dtype=np.float64)
