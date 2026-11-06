"""Splits validated filings into the passages retrieval searches.

A chunk joins whole paragraphs where it can. A paragraph too large to fit is split at sentence
boundaries, and a single sentence too large to fit is cut at the token limit. Consecutive chunks
overlap so a passage cut at a boundary can still be found whole. Each section is chunked on its
own, so a chunk never straddles two Items. Tables that did not parse cleanly become chunks too,
so every figure can be reached either as a fact or as text, and each part of such a table states
the scale the filing gives for it.

Every section with text is chunked, including a 10-Q's "no material changes" or cross-reference
statement: those are real answers a search should be able to cite. Change detection does not
read them from here. It reads `ParsedSection.status`, so it can report "no material changes"
rather than "removed".

Headings are left out of the chunk text on purpose. In the Hershey parses the running page
header "THE HERSHEY COMPANY" replaces the real heading on over a third of the notes paragraphs,
so adding headings would repeat that noise throughout the chunks.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore, SourcedChunk
from sec_agent.ingest.audit import SENTENCE_SPLIT
from sec_agent.ingest.pipeline import (
    FilingRejectedError,
    ParsedFiling,
    load_validated_filing,
    parsed_filing_path,
)
from sec_agent.ingest.sections import Paragraph
from sec_agent.schemas import Chunk, Filing, Section

# Tokens are counted with the embedding model's own tokenizer, because its limit is what matters.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# The design's range is 300 to 500 tokens. The model reads 512 including its two special tokens,
# so 500 leaves no chunk truncated. The maximum always holds; the minimum yields to a section or
# table shorter than it and to the rare leftover that fits nowhere else.
MAX_TOKENS = 500
MIN_TOKENS = 300
# The design's "about 10%" overlap, taken from the maximum chunk size.
OVERLAP_TOKENS = 50
# A split table repeats its title and column labels, since a figure means nothing without them.
# A heading row longer than this would crowd out the figures. Chosen by hand.
TABLE_HEADER_MAX_TOKENS = 100
# Stanley's non-GAAP reconciliations sit at the start of its MD&A, but the text before them is
# too short for the store to locate them. They are the only unlocated tables in the pinned
# filings that belong to one of the five sections, and this phrase picks out exactly those.
MDNA_TABLE_MARKER = "non-gaap"
# Words with the whitespace after them, so that joining them restores the text exactly.
WORD = re.compile(r"\S+\s*")

TokenCounter = Callable[[str], int]


def load_token_counter() -> TokenCounter:
    """The embedding model's tokenizer. The first call downloads its small tokenizer file."""
    # Imported here because tests inject a simple counter and should not need the library.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(EMBEDDING_MODEL)
    # A truncating tokenizer would report at most its limit, hiding every oversized chunk.
    tokenizer.no_truncation()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


@dataclass(frozen=True)
class _Piece:
    """A run of text from one paragraph, numbered by its place in the section."""

    source: int
    text: str
    tokens: int


@dataclass(frozen=True)
class _Draft:
    """A chunk being assembled. The first `overlap` pieces repeat the previous chunk's text."""

    pieces: tuple[_Piece, ...]
    overlap: int

    @property
    def tokens(self) -> int:
        return _total(self.pieces)

    @property
    def fresh(self) -> tuple[_Piece, ...]:
        return self.pieces[self.overlap :]

    @property
    def text(self) -> str:
        parts = [self.pieces[0].text]
        for previous, piece in pairwise(self.pieces):
            parts.append(" " if piece.source == previous.source else "\n\n")
            parts.append(piece.text)
        return "".join(parts)


def _total(pieces: Iterable[_Piece]) -> int:
    return sum(piece.tokens for piece in pieces)


def _largest_fit(available: int, limit: int, size_of: Callable[[int], int]) -> int:
    """The largest n from 1 to `available` with size_of(n) within the limit, or 0."""
    return bisect_right(range(1, available + 1), limit, key=size_of)


def _cut_at_limit(text: str, limit: int, count: TokenCounter) -> tuple[str, str]:
    """The last resort for text with no usable boundary: cut it at the limit, between words."""
    atoms = WORD.findall(text)
    # One enormous "word" has no space to cut at, so it is cut between characters.
    if len(atoms) == 1:
        atoms = list(text)
    # Always take something, so that splitting a long text is certain to finish.
    taken = max(1, _largest_fit(len(atoms), limit, lambda n: count("".join(atoms[:n]))))
    return "".join(atoms[:taken]).rstrip(), "".join(atoms[taken:]).lstrip()


def _take_sentences(
    piece: _Piece, room: int, count: TokenCounter, *, must_fit: bool
) -> tuple[_Piece | None, _Piece | None]:
    """Split a piece into what fits in `room` and the rest, at a sentence boundary if possible.

    `must_fit` means nothing else is in the chunk, so a first sentence too long for the room is
    cut at the limit. Otherwise it is left whole for the next chunk.
    """
    sentences = SENTENCE_SPLIT.split(piece.text)
    taken = used = 0
    for sentence in sentences:
        tokens = count(sentence)
        if used + tokens > room:
            break
        taken += 1
        used += tokens
    if taken:
        head, rest = " ".join(sentences[:taken]), " ".join(sentences[taken:])
    elif must_fit:
        head, rest = _cut_at_limit(piece.text, room, count)
    else:
        return None, piece

    def build(text: str) -> _Piece | None:
        return _Piece(piece.source, text, count(text)) if text else None

    return build(head), build(rest)


def _closing_words(text: str, limit: int, count: TokenCounter) -> str:
    """As many of the text's last words as fit the limit."""
    words = WORD.findall(text)
    taken = _largest_fit(len(words), limit, lambda n: count("".join(words[-n:])))
    return "".join(words[-taken:]) if taken else ""


def _overlap(pieces: Sequence[_Piece], count: TokenCounter) -> tuple[_Piece, ...]:
    """The text the next chunk repeats: whole closing sentences that fit the overlap budget."""
    tail: list[_Piece] = []
    budget = OVERLAP_TOKENS
    for piece in reversed(pieces):
        for sentence in reversed(SENTENCE_SPLIT.split(piece.text)):
            tokens = count(sentence)
            if tokens > budget:
                if tail:
                    return tuple(tail)
                # The last sentence alone exceeds the budget, so repeat its closing words.
                text = _closing_words(piece.text, OVERLAP_TOKENS, count)
                return (_Piece(piece.source, text, count(text)),) if text else ()
            tail.insert(0, _Piece(piece.source, sentence, tokens))
            budget -= tokens
    return tuple(tail)


def _draft_section(paragraphs: Sequence[Paragraph], count: TokenCounter) -> list[_Draft]:
    """Pack one section's paragraphs into chunks, keeping paragraphs whole where it can."""
    queue = deque(
        _Piece(source, paragraph.text, count(paragraph.text))
        for source, paragraph in enumerate(paragraphs)
    )
    drafts: list[_Draft] = []
    current: list[_Piece] = []
    overlap = 0
    while queue:
        piece = queue.popleft()
        used = _total(current)
        if used + piece.tokens <= MAX_TOKENS:
            current.append(piece)
            continue
        if used < MIN_TOKENS:
            # Stopping here would leave a small chunk, so take what fits of this paragraph.
            only_overlap = len(current) == overlap
            head, rest = _take_sentences(piece, MAX_TOKENS - used, count, must_fit=only_overlap)
            if head:
                current.append(head)
            if rest:
                queue.appendleft(rest)
        else:
            # Big enough already, so end the chunk at this paragraph boundary.
            queue.appendleft(piece)
        drafts.append(_Draft(tuple(current), overlap))
        current = list(_overlap(current, count))
        overlap = len(current)
    if len(current) > overlap:
        drafts.append(_Draft(tuple(current), overlap))
    # A small leftover at the end of a section joins the chunk before it, if there is room.
    if len(drafts) > 1 and drafts[-1].tokens < MIN_TOKENS:
        before, last = drafts[-2], drafts[-1]
        if before.tokens + _total(last.fresh) <= MAX_TOKENS:
            drafts[-2:] = [_Draft((*before.pieces, *last.fresh), before.overlap)]
    return drafts


def _chunk_record(
    filing: Filing, section: Section, chunk_id: str, paragraph_id: str, text: str, tokens: int
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paragraph_id=paragraph_id,
        ticker=filing.ticker,
        cik=filing.cik,
        form=filing.form,
        period=filing.period,
        section=section,
        accession=filing.accession,
        text=text,
        token_count=tokens,
    )


def chunk_section(
    filing: Filing, section: Section, paragraphs: Sequence[Paragraph], count: TokenCounter
) -> list[SourcedChunk]:
    """Chunk one section. Each chunk's `paragraph_id` is where its new text begins."""
    chunks = []
    # Several chunks can begin in one long paragraph; this numbers them to keep ids unique.
    started: Counter[str] = Counter()
    for draft in _draft_section(paragraphs, count):
        parent = paragraphs[draft.fresh[0].source].paragraph_id
        text = draft.text
        chunk = _chunk_record(
            filing, section, f"{parent}:{started[parent]}", parent, text, count(text)
        )
        started[parent] += 1
        sources = dict.fromkeys(paragraphs[piece.source].paragraph_id for piece in draft.pieces)
        chunks.append(SourcedChunk(chunk=chunk, paragraph_ids=tuple(sources)))
    return chunks


def _table_section(table: Mapping[str, object]) -> Section:
    """The section a table is filed under. An unlocated one is OTHER unless it is known MD&A."""
    if table["section"]:
        return Section(str(table["section"]))
    return Section.MDNA if MDNA_TABLE_MARKER in str(table["text"]).lower() else Section.OTHER


def _split_table(
    text: str, has_title: bool, captions: Sequence[str], count: TokenCounter
) -> list[str]:
    """A table's text in parts within the limit, split between rows.

    Every part states the table's scale, and repeats its title and column labels when they fit.
    A caption found outside the table's rows is added, in the filing's own words, after the title.
    """
    lines = text.split("\n")
    title_lines = 1 if has_title else 0
    missing = [caption for caption in captions if caption not in text]
    whole = "\n".join([*lines[:title_lines], *missing, *lines[title_lines:]])
    if count(whole) <= MAX_TOKENS:
        return [whole]

    # The first line is the title when there is one, and the next line the column labels.
    header = lines[: title_lines + 1]
    if count("\n".join(header)) > TABLE_HEADER_MAX_TOKENS:
        header = lines[:title_lines]
    if count("\n".join(header)) > TABLE_HEADER_MAX_TOKENS:
        header = []
    title = header[:title_lines]
    labels = header[title_lines:]
    # Captions are kept apart from that header, so one is never dropped along with a long header.
    repeated = [caption for caption in captions if caption not in "\n".join(header)]
    budget = MAX_TOKENS - (count("\n".join([*header, *repeated])) if header or repeated else 0)

    # A row too long for a chunk of its own is cut at the limit, the last resort.
    rows: deque[str] = deque(lines[len(header) :])
    parts: list[list[str]] = []
    current: list[str] = []
    used = 0
    overlap = 0
    while rows:
        row = rows.popleft()
        tokens = count(row)
        if tokens > budget - OVERLAP_TOKENS:
            row, rest = _cut_at_limit(row, budget - OVERLAP_TOKENS, count)
            tokens = count(row)
            if rest:
                rows.appendleft(rest)
        if used + tokens > budget:
            parts.append(current)
            # Repeat the last row when it is short enough, as the overlap between parts.
            last = count(current[-1])
            current = [current[-1]] if last <= OVERLAP_TOKENS else []
            used = last if current else 0
            overlap = len(current)
        current.append(row)
        used += tokens
    if len(current) > overlap:
        parts.append(current)

    def render(part: list[str]) -> str:
        # A caption already among this part's rows needn't appear twice.
        rows_text = "\n".join(part)
        needed = [caption for caption in repeated if caption not in rows_text]
        return "\n".join([*title, *needed, *labels, *part])

    return [render(part) for part in parts]


def chunk_table(
    filing: Filing, table: Mapping[str, object], count: TokenCounter
) -> list[SourcedChunk]:
    """Chunk one unparsed table. Its table id stands in for the parent paragraph id."""
    table_id = str(table["table_id"])
    section = _table_section(table)
    caption = table["scale_caption"]
    captions = str(caption).split("\n") if caption else []
    parts = _split_table(str(table["text"]), bool(table["title"]), captions, count)
    return [
        SourcedChunk(
            chunk=_chunk_record(filing, section, f"{table_id}:{n}", table_id, part, count(part)),
            paragraph_ids=(table_id,),
        )
        for n, part in enumerate(parts)
        if part.strip()
    ]


def chunk_filing(
    filing: Filing,
    parsed: ParsedFiling,
    tables: Iterable[Mapping[str, object]],
    count: TokenCounter,
) -> list[SourcedChunk]:
    """All of a filing's chunks in reading order: each section's text, then its tables."""
    # The same rule the fact store applies: unvalidated text never reaches the index.
    if not parsed.passed:
        raise FilingRejectedError(f"{filing.accession} failed validation; chunking nothing.")
    if parsed.accession != filing.accession:
        raise ValueError(f"parsed {parsed.accession} is not filing {filing.accession}")
    chunks: list[SourcedChunk] = []
    for section, content in parsed.sections.items():
        chunks += chunk_section(filing, section, content.paragraphs, count)
    for table in tables:
        chunks += chunk_table(filing, table, count)
    return chunks


def chunk_corpus(
    corpus: Corpus, parsed_dir: Path, store: FactStore, count: TokenCounter
) -> dict[str, int]:
    """Chunk every pinned filing into the store. Returns the chunk count for each accession.

    Takes the pinned corpus, so the other filings kept for parser tests are never indexed.
    """
    counts: dict[str, int] = {}
    for company in corpus.companies:
        for pinned in company.filings:
            # Without its facts loaded, a filing's unparsed tables would silently go unindexed.
            if not store.is_loaded(pinned.accession):
                raise FilingRejectedError(
                    f"{pinned.accession} has no current facts. Run `sec-agent ingest`."
                )
            path = parsed_filing_path(parsed_dir, company.ticker, pinned.accession)
            chunks = chunk_filing(
                company.to_filing(pinned),
                load_validated_filing(path),
                store.unparsed_tables(pinned.accession),
                count,
            )
            store.replace_chunks(pinned.accession, chunks)
            counts[pinned.accession] = len(chunks)
    return counts
