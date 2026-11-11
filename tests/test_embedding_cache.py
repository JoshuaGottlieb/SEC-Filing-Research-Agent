"""The passage-embedding cache: reuse, persistence, exact vectors, and alignment unchanged."""

from __future__ import annotations

import time
import zlib
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest

from sec_agent.compare.align import Passage, PeriodRoll, align_comparison, align_passages
from sec_agent.compare.embedding_cache import (
    LOOKUP_GROUP,
    CachedEmbedder,
    embedding_cache_path,
    passage_key,
)
from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.index import BgeEmbedder
from sec_agent.schemas import ComparisonType

MODEL = "fake-model"


class RecordingEmbedder:
    """Word counts hashed into float32 vectors (exact in float32); records every call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.queries: list[str] = []

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = np.zeros(64, dtype=np.float32)
            for word, n in Counter(text.lower().split()).items():
                vector[zlib.crc32(word.encode()) % 64] += n
            vector /= max(float(np.linalg.norm(vector)), 1.0)
            vectors.append(vector.tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "embeddings.sqlite"


def test_a_repeat_reads_the_stored_vectors_without_the_model(cache_path):
    inner = RecordingEmbedder()
    cached = CachedEmbedder(inner, MODEL, cache_path)
    first = cached.embed_passages(["cocoa costs rose", "tariffs fell"])
    again = cached.embed_passages(["tariffs fell", "cocoa costs rose"])
    assert inner.calls == [["cocoa costs rose", "tariffs fell"]]
    assert again == [first[1], first[0]]
    assert (cached.hits, cached.misses) == (2, 2)


def test_only_missing_texts_reach_the_model_once_each_in_order(cache_path):
    inner = RecordingEmbedder()
    cached = CachedEmbedder(inner, MODEL, cache_path)
    cached.embed_passages(["b"])
    vectors = cached.embed_passages(["c", "b", "a", "c"])
    assert inner.calls == [["b"], ["c", "a"]]
    assert vectors[0] == vectors[3]
    assert (cached.hits, cached.misses) == (1, 3)


def test_vectors_are_returned_exactly_as_the_model_gave_them(cache_path):
    inner = RecordingEmbedder()
    texts = ["net sales fell 10.2%", "gross margin rose 593 bps"]
    expected = RecordingEmbedder().embed_passages(texts)
    cached = CachedEmbedder(inner, MODEL, cache_path)
    assert cached.embed_passages(texts) == expected  # fresh
    assert cached.embed_passages(texts) == expected  # stored


def test_the_cache_survives_a_new_process(cache_path):
    CachedEmbedder(RecordingEmbedder(), MODEL, cache_path).embed_passages(["kept"])
    inner = RecordingEmbedder()
    reopened = CachedEmbedder(inner, MODEL, cache_path)
    reopened.embed_passages(["kept"])
    assert inner.calls == []


def test_another_model_never_reads_these_vectors(cache_path):
    CachedEmbedder(RecordingEmbedder(), MODEL, cache_path).embed_passages(["same text"])
    inner = RecordingEmbedder()
    CachedEmbedder(inner, "other-model", cache_path).embed_passages(["same text"])
    assert inner.calls == [["same text"]]
    assert passage_key(MODEL, "same text") != passage_key("other-model", "same text")


def test_more_texts_than_one_lookup_group_are_all_found(cache_path):
    texts = [f"passage number {n}" for n in range(LOOKUP_GROUP * 2 + 7)]
    inner = RecordingEmbedder()
    cached = CachedEmbedder(inner, MODEL, cache_path)
    first = cached.embed_passages(texts)
    assert cached.embed_passages(texts) == first
    assert len(inner.calls) == 1


def test_a_wrong_number_of_vectors_is_refused_and_nothing_is_stored(cache_path):
    class Short(RecordingEmbedder):
        def embed_passages(self, texts):
            return super().embed_passages(texts)[:-1]

    cached = CachedEmbedder(Short(), MODEL, cache_path)
    with pytest.raises(ValueError, match="asked for 2 vectors, got 1"):
        cached.embed_passages(["one", "two"])
    inner = RecordingEmbedder()
    CachedEmbedder(inner, MODEL, cache_path).embed_passages(["one"])
    assert inner.calls == [["one"]]


def test_queries_pass_through_and_are_not_stored(cache_path):
    inner = RecordingEmbedder()
    cached = CachedEmbedder(inner, MODEL, cache_path)
    assert cached.embed_query("freight") == [1.0, 0.0]
    cached.embed_query("freight")
    assert inner.queries == ["freight", "freight"]
    assert (cached.hits, cached.misses) == (0, 0)


def test_alignment_through_the_cache_matches_alignment_without_it(cache_path):
    def passages(prefix: str, texts: list[str]) -> tuple[Passage, ...]:
        return tuple(
            Passage(text=text, paragraph_ids=(f"{prefix}:{n}",)) for n, text in enumerate(texts)
        )

    current = passages("c", ["Cocoa costs rose sharply.", "We acquired LesserEvil.", "Same."])
    prior = passages("p", ["Cocoa costs rose.", "Tariffs were low.", "Same."])
    roll = PeriodRoll(year_shifts={1}, prior_month="December", current_month="December")
    plain = align_passages(current, prior, RecordingEmbedder(), roll)
    cached = CachedEmbedder(RecordingEmbedder(), MODEL, cache_path)
    assert align_passages(current, prior, cached, roll) == plain
    assert align_passages(current, prior, cached, roll) == plain  # stored vectors


def test_the_shared_cache_lives_in_the_data_directory(settings):
    assert embedding_cache_path(settings) == settings.data_dir / "embedding_cache.sqlite"


@pytest.mark.corpus
def test_real_hershey_quarter_aligns_the_same_through_the_cache(tmp_path):
    settings = get_settings()
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    comparison = corpus.comparison("HSY", ComparisonType.SEQ_10Q)
    model = BgeEmbedder()
    cached = CachedEmbedder(model, "BAAI/bge-base-en-v1.5", tmp_path / "embeddings.sqlite")
    with FactStore(settings.duckdb_path, read_only=True) as store:
        plain = align_comparison(comparison, settings.parsed_filings_dir, model, store)
        cold = align_comparison(comparison, settings.parsed_filings_dir, cached, store)
        embedded = cached.misses
        started = time.perf_counter()
        warm = align_comparison(comparison, settings.parsed_filings_dir, cached, store)
        warm_seconds = time.perf_counter() - started
    assert cold == plain and warm == plain
    assert embedded > 100 and cached.misses == embedded  # the warm run embedded nothing
    assert warm_seconds < 10  # the model's run took about 30 s; reading takes well under 10
