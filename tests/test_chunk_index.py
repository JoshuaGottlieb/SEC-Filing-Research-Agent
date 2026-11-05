"""The vector index: it builds once, reloads, stays in step with the store, and filters."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore, SourcedChunk
from sec_agent.ingest.index import (
    BATCH_SIZE,
    BgeEmbedder,
    ChunkFilter,
    ChunkIndex,
    build_index,
    chunk_fingerprint,
)
from sec_agent.schemas import Chunk, ComparisonType, FiscalPeriod, FormType, Section

DIMENSIONS = 256


class FakeEmbedder:
    """Hashes words into a vector, so shared words mean nearby vectors, and records its calls."""

    def __init__(self) -> None:
        self.passages: list[str] = []
        self.queries: list[str] = []

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.passages += texts
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._embed(text)


def chunk(
    number: int,
    text: str = "freight costs rose",
    *,
    ticker: str = "HSY",
    accession: str = "hsy-q2-2026",
    form: FormType = FormType.FORM_10Q,
    year: int = 2026,
    fiscal_period: str = "Q2",
    section: Section = Section.MDNA,
) -> Chunk:
    return Chunk(
        chunk_id=f"{accession}:{section.value}:{number:04d}:0",
        paragraph_id=f"{accession}:{section.value}:{number:04d}",
        ticker=ticker,
        cik="0000047111",
        form=form,
        period=FiscalPeriod(
            fiscal_year=year, fiscal_period=fiscal_period, period_end=date(year, 6, 30)
        ),
        section=section,
        accession=accession,
        text=text,
        token_count=len(text.split()),
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def index(settings, embedder):
    with ChunkIndex(settings.chroma_dir, embedder) as opened:
        yield opened


def ids(hits) -> list[str]:
    return [hit.chunk_id for hit in hits]


def test_a_build_adds_every_chunk(index):
    summary = index.sync([chunk(1), chunk(2)])
    assert (summary.added, summary.updated, summary.removed, summary.unchanged) == (2, 0, 0, 0)
    assert index.count() == 2


def test_a_second_build_embeds_nothing(index, embedder):
    chunks = [chunk(1), chunk(2)]
    index.sync(chunks)
    embedder.passages.clear()
    summary = index.sync(chunks)
    assert embedder.passages == []
    assert (summary.added, summary.updated, summary.removed, summary.unchanged) == (0, 0, 0, 2)


def test_the_index_reloads_from_disk_without_embedding_again(settings):
    chunks = [chunk(1, "cocoa prices"), chunk(2, "freight costs")]
    with ChunkIndex(settings.chroma_dir, FakeEmbedder()) as first:
        first.sync(chunks)
    reloaded_embedder = FakeEmbedder()
    with ChunkIndex(settings.chroma_dir, reloaded_embedder) as reloaded:
        assert reloaded.count() == 2
        assert reloaded.sync(chunks).unchanged == 2
        assert reloaded_embedder.passages == []
        hits = reloaded.search("freight costs", ChunkFilter(ticker="HSY"), 1)
        assert ids(hits) == [chunks[1].chunk_id]


def test_a_chunk_whose_text_changed_is_embedded_again(index, embedder):
    index.sync([chunk(1, "old text"), chunk(2, "kept text")])
    embedder.passages.clear()
    summary = index.sync([chunk(1, "new wording"), chunk(2, "kept text")])
    assert (summary.added, summary.updated, summary.unchanged) == (0, 1, 1)
    assert embedder.passages == ["new wording"]
    # The search finds the new text under the old id, and there is still one entry for it.
    assert index.count() == 2
    assert ids(index.search("new wording", ChunkFilter(ticker="HSY"), 1)) == [chunk(1).chunk_id]


def test_a_chunk_whose_metadata_changed_is_written_again(index):
    index.sync([chunk(1, section=Section.MDNA)])
    moved = chunk(1, section=Section.MDNA).model_copy(update={"ticker": "MAT"})
    assert index.sync([moved]).updated == 1
    assert ids(index.search("freight", ChunkFilter(ticker="MAT"), 5)) == [moved.chunk_id]
    assert index.search("freight", ChunkFilter(ticker="HSY"), 5) == []


def test_a_chunk_gone_from_the_store_leaves_the_index(index):
    index.sync([chunk(1), chunk(2)])
    summary = index.sync([chunk(2)])
    assert summary.removed == 1
    assert index.count() == 1
    assert ids(index.search("freight", ChunkFilter(ticker="HSY"), 5)) == [chunk(2).chunk_id]


def test_more_chunks_than_one_batch_all_arrive(index):
    chunks = [chunk(number, f"topic{number} text") for number in range(BATCH_SIZE * 2 + 3)]
    assert index.sync(chunks).added == len(chunks)
    assert index.count() == len(chunks)
    assert index.sync([]).removed == len(chunks)
    assert index.count() == 0


def test_an_interrupted_build_keeps_its_progress(settings):
    chunks = [chunk(number, f"topic{number} text") for number in range(BATCH_SIZE + 5)]

    class FailsOnSecondBatch(FakeEmbedder):
        def embed_passages(self, texts):
            if self.passages:
                raise RuntimeError("interrupted")
            return super().embed_passages(texts)

    with ChunkIndex(settings.chroma_dir, FailsOnSecondBatch()) as index:
        with pytest.raises(RuntimeError, match="interrupted"):
            index.sync(chunks)
        assert index.count() == BATCH_SIZE
    resumed = FakeEmbedder()
    with ChunkIndex(settings.chroma_dir, resumed) as index:
        summary = index.sync(chunks)
        assert (summary.added, summary.unchanged) == (5, BATCH_SIZE)
        assert len(resumed.passages) == 5


def test_build_index_reads_the_store(settings, embedder):
    stored = [chunk(1), chunk(2, accession="mat-q2-2026", ticker="MAT")]
    with FactStore(settings.duckdb_path) as store, ChunkIndex(settings.chroma_dir, embedder) as idx:
        for item in stored:
            store.replace_chunks(
                item.accession, [SourcedChunk(chunk=item, paragraph_ids=(item.paragraph_id,))]
            )
        assert build_index(store, idx).added == 2
        # Chunks cleared from the store, as a reload does, leave the index on the next build.
        store.remove_filing("mat-q2-2026")
        assert build_index(store, idx).removed == 1
        assert idx.count() == 1


def test_store_returns_every_filings_chunks_or_one(settings):
    first, second = chunk(1), chunk(2, accession="mat-q2-2026", ticker="MAT")
    with FactStore(settings.duckdb_path) as store:
        for item in (first, second):
            store.replace_chunks(
                item.accession, [SourcedChunk(chunk=item, paragraph_ids=(item.paragraph_id,))]
            )
        assert {item.chunk_id for item in store.chunks()} == {first.chunk_id, second.chunk_id}
        assert store.chunks("mat-q2-2026") == [second]


# --- Filters ---


@pytest.fixture
def mixed(index) -> dict[str, Chunk]:
    """The same words under different companies, periods, forms and sections."""
    chunks = {
        "hsy_q2_mdna": chunk(1),
        "hsy_q2_risk": chunk(2, section=Section.RISK_FACTORS),
        "hsy_q1_mdna": chunk(3, accession="hsy-q1-2026", fiscal_period="Q1"),
        "hsy_fy_mdna": chunk(
            4, accession="hsy-fy-2025", form=FormType.FORM_10K, year=2025, fiscal_period="FY"
        ),
        "mat_q2_mdna": chunk(5, accession="mat-q2-2026", ticker="MAT"),
    }
    index.sync(list(chunks.values()))
    return chunks


def test_a_company_filter_excludes_other_companies(index, mixed):
    hits = ids(index.search("freight costs", ChunkFilter(ticker="MAT"), 10))
    assert hits == [mixed["mat_q2_mdna"].chunk_id]


def test_a_period_filter_excludes_other_periods(index, mixed):
    hits = ids(index.search("freight costs", ChunkFilter(ticker="HSY", period="Q1 2026"), 10))
    assert hits == [mixed["hsy_q1_mdna"].chunk_id]


def test_a_section_filter_excludes_other_sections(index, mixed):
    where = ChunkFilter(accession="hsy-q2-2026", section=Section.RISK_FACTORS)
    assert ids(index.search("freight costs", where, 10)) == [mixed["hsy_q2_risk"].chunk_id]


def test_a_form_filter_excludes_other_forms(index, mixed):
    where = ChunkFilter(ticker="HSY", form=FormType.FORM_10K)
    assert ids(index.search("freight costs", where, 10)) == [mixed["hsy_fy_mdna"].chunk_id]


def test_an_accession_filter_keeps_one_filings_sections(index, mixed):
    hits = set(ids(index.search("freight costs", ChunkFilter(accession="hsy-q2-2026"), 10)))
    assert hits == {mixed["hsy_q2_mdna"].chunk_id, mixed["hsy_q2_risk"].chunk_id}


def test_a_filter_matching_nothing_returns_nothing(index, mixed):
    assert index.search("freight costs", ChunkFilter(ticker="SWK"), 10) == []


def test_a_search_with_no_filter_is_refused():
    with pytest.raises(ValueError, match="limited"):
        ChunkFilter()


def test_results_come_nearest_first_with_cosine_distance(index):
    close, far = chunk(1, "freight costs rose"), chunk(2, "cocoa prices fell sharply")
    index.sync([far, close])
    hits = index.search("freight costs rose", ChunkFilter(ticker="HSY"), 5)
    assert ids(hits) == [close.chunk_id, far.chunk_id]
    # The same text points the same way, and text sharing no words is about one apart.
    assert hits[0].distance == pytest.approx(0.0, abs=1e-5)
    assert hits[1].distance == pytest.approx(1.0, abs=0.01)


def test_top_k_limits_the_results(index):
    index.sync([chunk(number) for number in range(6)])
    assert len(index.search("freight", ChunkFilter(ticker="HSY"), 2)) == 2


def test_the_fingerprint_follows_the_whole_chunk():
    assert chunk_fingerprint(chunk(1)) == chunk_fingerprint(chunk(1))
    assert chunk_fingerprint(chunk(1, "other")) != chunk_fingerprint(chunk(1))
    assert chunk_fingerprint(chunk(1, section=Section.MDNA)) != chunk_fingerprint(
        chunk(1, section=Section.BUSINESS).model_copy(update={"chunk_id": chunk(1).chunk_id})
    )


# --- The real model and the real index ---


@pytest.fixture(scope="module")
def real_embedder() -> BgeEmbedder:
    return BgeEmbedder()


@pytest.mark.corpus
def test_real_embeddings_are_normalized_and_768_wide(real_embedder):
    vectors = np.array(real_embedder.embed_passages(["Gross margin fell.", "Cocoa costs rose."]))
    assert vectors.shape == (2, 768)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
    assert np.linalg.norm(real_embedder.embed_query("gross margin")) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.corpus
def test_a_query_is_embedded_with_the_instruction_and_a_passage_without(real_embedder):
    text = "gross margin"
    assert real_embedder.embed_query(text) != real_embedder.embed_passages([text])[0]


@pytest.mark.corpus
def test_the_real_index_matches_the_store_and_filters(real_embedder):
    settings = get_settings()
    with (
        FactStore(settings.duckdb_path) as store,
        ChunkIndex(settings.chroma_dir, real_embedder) as index,
    ):
        if index.count() == 0:
            pytest.skip("The index has not been built.")
        chunks = store.chunks()
        # Checking without embedding: a stale or missing chunk would make this a real build.
        assert index.count() == len(chunks)
        stored = index._collection.get(include=["metadatas"])
        recorded = dict(zip(stored["ids"], stored["metadatas"], strict=True))
        for item in chunks:
            assert recorded[item.chunk_id]["fingerprint"] == chunk_fingerprint(item)

        corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
        current = corpus.comparison("HSY", ComparisonType.YOY_10Q).current
        where = ChunkFilter(accession=current.accession, section=Section.MDNA)
        hits = index.search("gross margin decline", where, 20)
        assert len(hits) == 20
        for hit in hits:
            found = store.get_chunk(hit.chunk_id)
            assert found is not None
            assert (found.ticker, found.accession, found.section) == (
                "HSY",
                current.accession,
                Section.MDNA,
            )
