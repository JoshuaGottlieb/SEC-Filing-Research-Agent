"""The chunker: size limits, section boundaries, parent ids, tables, and the chunk store."""

from __future__ import annotations

from datetime import date

import pytest

from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.contracts import ChunkLookup
from sec_agent.edgar.corpus import Corpus, CorpusCompany, CorpusFiling
from sec_agent.finance.fact_store import FactStore, SourcedChunk
from sec_agent.ingest.audit import SENTENCE_SPLIT, TableAudit
from sec_agent.ingest.chunker import (
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP_TOKENS,
    chunk_corpus,
    chunk_filing,
    chunk_section,
    chunk_table,
    load_token_counter,
)
from sec_agent.ingest.pipeline import (
    FilingRejectedError,
    ParsedFiling,
    ParserName,
    load_validated_filing,
    parsed_filing_path,
    save_parsed_filing,
)
from sec_agent.ingest.sections import Paragraph, ParsedSection, SectionStatus
from sec_agent.schemas import FormType, Section

ACCESSION = "acc-1"
PINNED = CorpusFiling(
    form=FormType.FORM_10K,
    fiscal_year=2025,
    fiscal_period="FY",
    period_end=date(2025, 12, 31),
    filed_date=date(2026, 2, 1),
    accession=ACCESSION,
    primary_document="doc.htm",
)
COMPANY = CorpusCompany(ticker="AAA", company="A", cik="0000000001", filings=(PINNED,))
FILING = COMPANY.to_filing(PINNED)


def count(text: str) -> int:
    """A stand-in for the real tokenizer, so unit tests need no download: one word, one token."""
    return len(text.split())


def sentence(tag: object, words: int = 20) -> str:
    """A sentence of exactly `words` words, recognizable by its tag."""
    return " ".join([f"s{tag}", *["word"] * (words - 2), "end."])


def paragraph(number: int, text: str, section: Section = Section.MDNA) -> Paragraph:
    return Paragraph(paragraph_id=f"{ACCESSION}:{section.value}:{number:04d}", text=text)


def paragraphs(sizes: list[int], section: Section = Section.MDNA) -> list[Paragraph]:
    """One paragraph per size, built from 20-word sentences, each with a unique tag."""
    result = []
    for number, size in enumerate(sizes):
        result.append(
            paragraph(
                number,
                " ".join(sentence(f"{number}x{i}") for i in range(size // 20)),
                section,
            )
        )
    return result


def parsed_filing(
    sections: dict[Section, ParsedSection], problems: tuple[str, ...] = ()
) -> ParsedFiling:
    return ParsedFiling(
        accession=ACCESSION,
        form=FormType.FORM_10K,
        parser=ParserName.EDGARTOOLS,
        layout="edgartools",
        sections=sections,
        coverage=1.0,
        table_audit=TableAudit(tables_total=0, tables_removed=0, prose_rows_preserved=0),
        problems=problems,
    )


def section_of(section: Section, paragraph_list: list[Paragraph]) -> ParsedSection:
    return ParsedSection(section=section, status=SectionStatus.CONTENT, paragraphs=paragraph_list)


def table(
    number: int = 1,
    *,
    text: str = "Title\nLabel | 2025\nRow | 1",
    title: str | None = "Title",
    section: str | None = "mdna",
) -> dict:
    return {
        "table_id": f"{ACCESSION}:t{number}",
        "section": section,
        "title": title,
        "text": text,
    }


def test_chunks_stay_within_the_size_range():
    chunks = chunk_section(FILING, Section.MDNA, paragraphs([60] * 60), count)
    sizes = [item.chunk.token_count for item in chunks]
    assert len(sizes) > 2
    assert max(sizes) <= MAX_TOKENS
    assert min(sizes[:-1]) >= MIN_TOKENS


def test_token_count_is_the_size_of_the_stored_text():
    for item in chunk_section(FILING, Section.MDNA, paragraphs([60] * 30), count):
        assert item.chunk.token_count == count(item.chunk.text)


def test_short_paragraphs_join_and_keep_their_boundaries():
    chunks = chunk_section(FILING, Section.MDNA, paragraphs([100, 100, 100]), count)
    assert len(chunks) == 1
    assert chunks[0].chunk.text.count("\n\n") == 2


def test_a_section_shorter_than_the_minimum_is_still_one_chunk():
    chunks = chunk_section(FILING, Section.MDNA, paragraphs([40]), count)
    assert [item.chunk.token_count for item in chunks] == [40]


def test_an_empty_section_makes_no_chunks():
    assert chunk_section(FILING, Section.MDNA, [], count) == []


def test_a_small_leftover_joins_the_chunk_before_it():
    # 460 fills one chunk, and the last 20 would otherwise be left as a tiny chunk of its own.
    chunks = chunk_section(FILING, Section.MDNA, paragraphs([460, 20]), count)
    assert len(chunks) == 1
    assert chunks[0].chunk.token_count == 480


def test_a_long_paragraph_splits_only_at_sentence_boundaries():
    long = paragraphs([1400])
    chunks = chunk_section(FILING, Section.MDNA, long, count)
    assert len(chunks) > 2
    assert max(item.chunk.token_count for item in chunks) <= MAX_TOKENS
    whole = set(SENTENCE_SPLIT.split(long[0].text))
    for item in chunks:
        assert set(SENTENCE_SPLIT.split(item.chunk.text)) <= whole


def test_a_small_chunk_fills_itself_from_a_long_paragraph():
    # 100 then 800: ending at the paragraph boundary would leave a chunk of only 100.
    chunks = chunk_section(FILING, Section.MDNA, paragraphs([100, 800]), count)
    assert chunks[0].chunk.token_count >= MIN_TOKENS


def test_a_sentence_over_the_limit_is_cut_at_the_limit_and_nothing_is_lost():
    words = [f"w{i}" for i in range(1300)]
    chunks = chunk_section(FILING, Section.MDNA, [paragraph(0, " ".join(words))], count)
    assert len(chunks) > 2
    assert max(item.chunk.token_count for item in chunks) <= MAX_TOKENS
    assert {word for item in chunks for word in item.chunk.text.split()} == set(words)


def test_one_huge_word_is_still_split():
    def by_character(text: str) -> int:
        return len(text.replace(" ", "").replace("\n", ""))

    chunks = chunk_section(FILING, Section.MDNA, [paragraph(0, "x" * 1300)], by_character)
    assert max(item.chunk.token_count for item in chunks) <= MAX_TOKENS
    assert "".join(item.chunk.text for item in chunks).count("x") >= 1300


def test_consecutive_chunks_overlap_by_the_budget_or_less():
    chunks = chunk_section(FILING, Section.MDNA, paragraphs([100] * 12), count)
    assert len(chunks) > 1
    for before, after in zip(chunks, chunks[1:], strict=False):
        first_sentence = SENTENCE_SPLIT.split(after.chunk.text.split("\n\n")[0])[0]
        assert first_sentence in before.chunk.text
        shared = next(
            n
            for n in range(OVERLAP_TOKENS, 0, -1)
            if before.chunk.text.split()[-n:] == after.chunk.text.split()[:n]
        )
        assert 0 < shared <= OVERLAP_TOKENS


def test_overlap_falls_back_to_closing_words_when_the_last_sentence_is_long():
    long_sentences = " ".join(sentence(i, words=200) for i in range(5))
    chunks = chunk_section(FILING, Section.MDNA, [paragraph(0, long_sentences)], count)
    assert len(chunks) > 1
    closing = chunks[0].chunk.text.split()[-OVERLAP_TOKENS:]
    assert chunks[1].chunk.text.split()[:OVERLAP_TOKENS] == closing


def test_every_sentence_lands_in_a_chunk():
    source = paragraphs([60, 1400, 20, 340, 100, 700])
    chunks = chunk_section(FILING, Section.MDNA, source, count)
    combined = "\n".join(item.chunk.text for item in chunks)
    for item in source:
        for text in SENTENCE_SPLIT.split(item.text):
            assert text in combined


def test_each_chunk_records_its_parent_paragraphs():
    source = paragraphs([200, 200, 1400, 200])
    by_id = {item.paragraph_id: item.text for item in source}
    chunks = chunk_section(FILING, Section.MDNA, source, count)
    assert chunks[0].chunk.paragraph_id == source[0].paragraph_id
    for item in chunks:
        assert item.chunk.paragraph_id in item.paragraph_ids
        sentences = (
            s for block in item.chunk.text.split("\n\n") for s in SENTENCE_SPLIT.split(block)
        )
        known = {s for pid in item.paragraph_ids for s in SENTENCE_SPLIT.split(by_id[pid])}
        assert set(sentences) <= known


def test_the_parent_is_where_new_text_begins_not_the_overlap():
    source = paragraphs([480, 480])
    chunks = chunk_section(FILING, Section.MDNA, source, count)
    assert chunks[1].chunk.paragraph_id == source[1].paragraph_id
    # The overlap repeats paragraph 0, so that paragraph is still listed as a source.
    assert source[0].paragraph_id in chunks[1].paragraph_ids


def test_chunk_ids_are_unique_and_deterministic():
    source = paragraphs([1400, 200, 1400])
    first = chunk_section(FILING, Section.MDNA, source, count)
    ids = [item.chunk.chunk_id for item in first]
    assert len(ids) == len(set(ids))
    assert ids == [
        item.chunk.chunk_id for item in chunk_section(FILING, Section.MDNA, source, count)
    ]
    assert ids[0] == f"{source[0].paragraph_id}:0"
    assert ids[1] == f"{source[0].paragraph_id}:1"


def test_a_chunk_never_straddles_two_sections():
    risk = paragraphs([300, 300], Section.RISK_FACTORS)
    mdna = paragraphs([300, 300], Section.MDNA)
    parsed = parsed_filing(
        {
            Section.RISK_FACTORS: section_of(Section.RISK_FACTORS, risk),
            Section.MDNA: section_of(Section.MDNA, mdna),
        }
    )
    chunks = chunk_filing(FILING, parsed, [], count)
    assert {item.chunk.section for item in chunks} == {Section.RISK_FACTORS, Section.MDNA}
    for item in chunks:
        assert all(f":{item.chunk.section.value}:" in pid for pid in item.paragraph_ids)


def test_chunk_carries_the_filing_metadata():
    chunk = chunk_section(FILING, Section.MDNA, paragraphs([60]), count)[0].chunk
    assert (chunk.ticker, chunk.cik, chunk.form, chunk.accession) == (
        "AAA",
        "0000000001",
        FormType.FORM_10K,
        ACCESSION,
    )
    assert chunk.period == FILING.period
    assert chunk.section is Section.MDNA


@pytest.mark.parametrize(
    "status", [SectionStatus.NO_MATERIAL_CHANGES, SectionStatus.CROSS_REFERENCE]
)
def test_a_statement_section_is_chunked_so_it_can_be_cited(status):
    statement = paragraph(0, "There have been no material changes.", Section.RISK_FACTORS)
    parsed = parsed_filing(
        {
            Section.RISK_FACTORS: ParsedSection(
                section=Section.RISK_FACTORS, status=status, paragraphs=(statement,)
            )
        }
    )
    chunks = chunk_filing(FILING, parsed, [], count)
    assert [item.chunk.text for item in chunks] == [statement.text]


def test_a_missing_section_makes_no_chunks():
    parsed = parsed_filing(
        {Section.BUSINESS: ParsedSection(section=Section.BUSINESS, status=SectionStatus.MISSING)}
    )
    assert chunk_filing(FILING, parsed, [], count) == []


def test_a_table_keeps_its_id_section_and_title():
    chunks = chunk_table(FILING, table(7), count)
    assert len(chunks) == 1
    chunk = chunks[0].chunk
    assert chunk.chunk_id == f"{ACCESSION}:t7:0"
    assert chunk.paragraph_id == f"{ACCESSION}:t7"
    assert chunks[0].paragraph_ids == (f"{ACCESSION}:t7",)
    assert chunk.section is Section.MDNA
    assert chunk.text.startswith("Title\n")


def test_a_table_outside_the_sections_is_filed_as_other():
    chunk = chunk_table(FILING, table(section=None), count)[0].chunk
    assert chunk.section is Section.OTHER


def test_an_unlocated_non_gaap_reconciliation_is_filed_under_mdna():
    text = "Second Quarter\n(Millions) | GAAP | Non-GAAP Adjustments | Non-GAAP\nSales | 1 | 2 | 3"
    chunk = chunk_table(FILING, table(text=text, section=None), count)[0].chunk
    assert chunk.section is Section.MDNA


def test_a_long_table_splits_between_rows_and_repeats_its_header():
    rows = [f"Row{i} | a | b | c | d | e | f | g | h | i" for i in range(100)]
    text = "\n".join(["Revenue by region", "Region | 2025 | 2024", *rows])
    chunks = chunk_table(FILING, table(text=text, title="Revenue by region"), count)
    assert len(chunks) > 1
    seen = []
    for item in chunks:
        assert item.chunk.token_count <= MAX_TOKENS
        lines = item.chunk.text.split("\n")
        assert lines[:2] == ["Revenue by region", "Region | 2025 | 2024"]
        seen += lines[2:]
    assert set(seen) == set(rows)
    # Ids stay unique, and each part still points at the one table.
    assert len({item.chunk.chunk_id for item in chunks}) == len(chunks)
    assert {item.chunk.paragraph_id for item in chunks} == {f"{ACCESSION}:t1"}


def test_a_table_with_no_title_repeats_only_its_column_labels():
    rows = [f"Row{i} | a | b | c | d | e | f | g | h | i" for i in range(100)]
    text = "\n".join(["Region | 2025 | 2024", *rows])
    chunks = chunk_table(FILING, table(text=text, title=None), count)
    assert len(chunks) > 1
    assert all(item.chunk.text.startswith("Region | 2025 | 2024\n") for item in chunks)


def test_a_header_too_long_to_repeat_is_left_out():
    labels = " | ".join(f"label{i}" for i in range(150))
    rows = [f"Row{i} | a | b | c | d | e | f | g | h | i" for i in range(100)]
    text = "\n".join(["Title", labels, *rows])
    chunks = chunk_table(FILING, table(text=text), count)
    assert chunks[1].chunk.text.startswith("Title\n")
    assert labels not in chunks[1].chunk.text
    assert max(item.chunk.token_count for item in chunks) <= MAX_TOKENS


def test_a_table_row_over_the_limit_is_cut():
    row = " ".join(f"w{i}" for i in range(1300))
    chunks = chunk_table(FILING, table(text=f"Title\nLabels\n{row}"), count)
    assert max(item.chunk.token_count for item in chunks) <= MAX_TOKENS
    combined = {word for item in chunks for word in item.chunk.text.split()}
    assert {f"w{i}" for i in range(1300)} <= combined


def test_chunk_filing_includes_tables_after_the_text():
    parsed = parsed_filing({Section.MDNA: section_of(Section.MDNA, paragraphs([60]))})
    chunks = chunk_filing(FILING, parsed, [table(3)], count)
    assert [item.chunk.paragraph_id.endswith(":t3") for item in chunks] == [False, True]


def test_chunk_filing_refuses_a_rejected_filing():
    parsed = parsed_filing({}, problems=("bad",))
    with pytest.raises(FilingRejectedError):
        chunk_filing(FILING, parsed, [], count)


def test_chunk_filing_refuses_another_filings_parse():
    other = parsed_filing({}).model_copy(update={"accession": "acc-2"})
    with pytest.raises(ValueError, match="acc-2"):
        chunk_filing(FILING, other, [], count)


# --- The chunk store ---


def sourced(number: int, accession: str = ACCESSION, paragraph_ids: tuple[str, ...] = ()):
    chunk = chunk_section(FILING, Section.MDNA, paragraphs([60]), count)[0].chunk.model_copy(
        update={
            "chunk_id": f"{accession}:mdna:{number:04d}:0",
            "paragraph_id": f"{accession}:mdna:{number:04d}",
            "accession": accession,
        }
    )
    return SourcedChunk(chunk=chunk, paragraph_ids=paragraph_ids or (chunk.paragraph_id,))


def test_store_round_trips_a_chunk_and_satisfies_chunk_lookup(settings):
    original = chunk_section(FILING, Section.MDNA, paragraphs([60]), count)[0]
    with FactStore(settings.duckdb_path) as store:
        assert isinstance(store, ChunkLookup)
        store.replace_chunks(ACCESSION, [original])
        assert store.get_chunk(original.chunk.chunk_id) == original.chunk
        assert store.get_chunk("no-such-chunk") is None


def test_store_survives_being_reopened(settings):
    original = sourced(1)
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks(ACCESSION, [original])
    with FactStore(settings.duckdb_path) as store:
        assert store.get_chunk(original.chunk.chunk_id) == original.chunk


def test_rechunking_does_not_duplicate_or_orphan(settings):
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks(ACCESSION, [sourced(1), sourced(2)])
        store.replace_chunks(ACCESSION, [sourced(2), sourced(3)])
        assert store.count("chunks") == 2
        # Chunk 1 is gone rather than orphaned, and the new order is kept.
        assert [chunk.paragraph_id[-4:] for chunk in store.chunks(ACCESSION)] == ["0002", "0003"]


def test_rechunking_one_filing_leaves_others_alone(settings):
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks("acc-2", [sourced(1, "acc-2")])
        store.replace_chunks(ACCESSION, [sourced(1)])
        store.replace_chunks(ACCESSION, [])
        assert store.count("chunks") == 1
        assert store.chunks(ACCESSION) == []
        assert len(store.chunks("acc-2")) == 1


def test_store_refuses_another_filings_chunks(settings):
    with FactStore(settings.duckdb_path) as store, pytest.raises(ValueError, match="acc-1"):
        store.replace_chunks(ACCESSION, [sourced(1, "acc-2")])


def test_a_failed_replacement_keeps_the_old_chunks(settings):
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks(ACCESSION, [sourced(1)])
        # Two chunks with one id break the primary key, so the whole replacement must undo.
        with pytest.raises(Exception, match="(?i)constraint|duplicate"):
            store.replace_chunks(ACCESSION, [sourced(2), sourced(2)])
        assert [chunk.paragraph_id[-4:] for chunk in store.chunks(ACCESSION)] == ["0001"]


def test_every_chunk_holding_a_paragraph_can_be_found(settings):
    source = paragraphs([480, 480])
    chunks = chunk_section(FILING, Section.MDNA, source, count)
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks(ACCESSION, chunks)
        # Paragraph 0 is in the first chunk and, through the overlap, in the second.
        assert store.chunk_ids_for_paragraph(source[0].paragraph_id) == tuple(
            item.chunk.chunk_id for item in chunks[:2]
        )
        # Paragraph 1 starts in the second chunk and runs on into the third.
        assert store.chunk_ids_for_paragraph(source[1].paragraph_id) == tuple(
            item.chunk.chunk_id for item in chunks[1:]
        )
        assert store.chunk_ids_for_paragraph("nothing") == ()


def test_removing_a_filing_removes_its_chunks(settings):
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks(ACCESSION, [sourced(1)])
        store.remove_filing(ACCESSION)
        assert store.count("chunks") == 0


def test_reloading_a_filings_facts_clears_its_stale_chunks(settings):
    html = "<html><body><p>Nothing tagged here.</p></body></html>"
    parsed = parsed_filing({})
    with FactStore(settings.duckdb_path) as store:
        store.replace_chunks(ACCESSION, [sourced(1)])
        store.load_filing(COMPANY, PINNED, html, parsed)
        assert store.count("chunks") == 0


# --- The whole corpus ---


def write_parsed(settings, parsed: ParsedFiling) -> None:
    save_parsed_filing(
        parsed, parsed_filing_path(settings.parsed_filings_dir, COMPANY.ticker, ACCESSION)
    )


def test_chunk_corpus_builds_and_rebuilds_the_store(settings):
    corpus = Corpus(companies=(COMPANY,))
    write_parsed(
        settings, parsed_filing({Section.MDNA: section_of(Section.MDNA, paragraphs([60]))})
    )
    with FactStore(settings.duckdb_path) as store:
        store.load_filing(COMPANY, PINNED, "<html><body></body></html>", parsed_filing({}))
        assert chunk_corpus(corpus, settings.parsed_filings_dir, store, count) == {ACCESSION: 1}
        assert chunk_corpus(corpus, settings.parsed_filings_dir, store, count) == {ACCESSION: 1}
        assert store.count("chunks") == 1


def test_chunk_corpus_refuses_a_rejected_record(settings):
    corpus = Corpus(companies=(COMPANY,))
    write_parsed(settings, parsed_filing({}, problems=("bad",)))
    with FactStore(settings.duckdb_path) as store:
        store.load_filing(COMPANY, PINNED, "<html><body></body></html>", parsed_filing({}))
        with pytest.raises(FilingRejectedError):
            chunk_corpus(corpus, settings.parsed_filings_dir, store, count)
        assert store.count("chunks") == 0


def test_chunk_corpus_refuses_a_filing_with_no_facts_loaded(settings):
    # Its unparsed tables would look like an empty list, and their figures would be lost.
    corpus = Corpus(companies=(COMPANY,))
    write_parsed(settings, parsed_filing({}))
    with (
        FactStore(settings.duckdb_path) as store,
        pytest.raises(FilingRejectedError, match="facts"),
    ):
        chunk_corpus(corpus, settings.parsed_filings_dir, store, count)


# --- The real tokenizer and the real filings ---


@pytest.fixture(scope="module")
def real_count():
    return load_token_counter()


@pytest.mark.corpus
def test_real_tokenizer_does_not_truncate(real_count):
    # A truncating counter would report about 512 here and hide every oversized chunk.
    assert real_count("word " * 2000) == 2000


@pytest.mark.corpus
def test_all_pinned_filings_chunk_within_limits(real_count):
    settings = get_settings()
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    filings = [(company, pinned) for company in corpus.companies for pinned in company.filings]
    assert len(filings) == 15
    with FactStore(settings.duckdb_path) as store:
        for company, pinned in filings:
            parsed = load_validated_filing(
                parsed_filing_path(settings.parsed_filings_dir, company.ticker, pinned.accession)
            )
            tables = store.unparsed_tables(pinned.accession)
            chunks = chunk_filing(company.to_filing(pinned), parsed, tables, real_count)
            ids = [item.chunk.chunk_id for item in chunks]
            assert len(ids) == len(set(ids)), pinned.accession
            assert max(item.chunk.token_count for item in chunks) <= MAX_TOKENS, pinned.accession

            # Every table reaches a chunk, and every chunk's sources belong to its section.
            parents = {item.chunk.paragraph_id for item in chunks}
            assert {t["table_id"] for t in tables} <= parents, pinned.accession
            for item in chunks:
                if item.chunk.paragraph_id not in {t["table_id"] for t in tables}:
                    for pid in item.paragraph_ids:
                        assert f":{item.chunk.section.value}:" in pid, item.chunk.chunk_id

            # Every paragraph with text is somewhere, so nothing was dropped in splitting.
            sources = {pid for item in chunks for pid in item.paragraph_ids}
            for content in parsed.sections.values():
                assert {p.paragraph_id for p in content.paragraphs} <= sources, pinned.accession

            # Below the minimum is reserved for short sections, tables and the odd leftover.
            text_chunks = [item for item in chunks if ":t" not in item.chunk.paragraph_id]
            small = [item for item in text_chunks if item.chunk.token_count < MIN_TOKENS]
            assert len(small) <= max(8, len(text_chunks) // 20), pinned.accession
