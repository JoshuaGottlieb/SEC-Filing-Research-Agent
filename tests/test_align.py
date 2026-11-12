"""Paragraph alignment: passage repair, carried-forward text, matching, statuses and changes."""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest

from sec_agent.compare.align import (
    MIN_MATCH_SIMILARITY,
    PERIOD_MISMATCH_PENALTY,
    PERIOD_WORDS,
    RUNNING_HEADER,
    SIMILARITY_RESOLUTION,
    AlignedPair,
    PairLabel,
    Passage,
    PeriodKind,
    PeriodRoll,
    SectionOutcome,
    _assign,
    align_comparison,
    align_filings,
    align_passages,
    align_section,
    changes_for,
    is_carried_forward,
    period_kind,
    prepare_passages,
    section_outcome,
)
from sec_agent.config import REPO_ROOT, get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.audit import TableAudit
from sec_agent.ingest.index import BgeEmbedder
from sec_agent.ingest.pipeline import (
    FilingRejectedError,
    ParsedFiling,
    ParserName,
    load_validated_filing,
    parsed_filing_path,
    save_parsed_filing,
)
from sec_agent.ingest.sections import Paragraph, ParsedSection, SectionStatus
from sec_agent.schemas import ChangeKind, Comparison, ComparisonType, FormType, Section

HEADER = "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS—(Continued)"
YEARLY = PeriodRoll(year_shifts={1}, prior_month="December", current_month="December")
QUARTERLY = PeriodRoll(year_shifts={0}, prior_month="March", current_month="June")
YOY_10Q = PeriodRoll(year_shifts={1}, prior_month="June", current_month="June")
# Q1 to Q2: the earlier filing's quarter is also its year to date.
FIRST_QUARTER = QUARTERLY.model_copy(update={"prior_quarter_is_year_to_date": True})
# Stanley's 52/53-week years: FY2025 ended January 3, 2026, FY2024 December 28, 2024, and its
# Q2 2026 ended July 4, 2026 against June 28, 2025.
STANLEY_YEARLY = PeriodRoll(year_shifts={1, 2}, prior_month="December", current_month="January")
STANLEY_YOY_10Q = PeriodRoll(year_shifts={1}, prior_month="June", current_month="July")


class TopicEmbedder:
    """Texts with the same first word get the same vector; different first words are orthogonal.

    So a pair's cosine similarity is exactly 1 or 0, which keeps the matching in tests exact.
    """

    def __init__(self) -> None:
        self.topics: dict[str, int] = {}
        self.calls: list[list[str]] = []

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            index = self.topics.setdefault(text.split()[0], len(self.topics))
            vectors.append([1.0 if d == index else 0.0 for d in range(64)])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("alignment embeds passages, never queries")


class FixedEmbedder:
    """Returns the vector given for each text, to set an exact similarity."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("alignment embeds passages, never queries")


class FakeChunks:
    """A paragraph-to-chunk lookup standing in for `FactStore`."""

    def __init__(self, chunks: dict[str, tuple[str, ...]]) -> None:
        self.chunks = chunks
        self.asked: list[str] = []

    def chunk_ids_for_paragraph(self, paragraph_id: str) -> tuple[str, ...]:
        self.asked.append(paragraph_id)
        return self.chunks.get(paragraph_id, ())


def paragraphs(*texts: str, prefix: str = "acc:financial_notes") -> tuple[Paragraph, ...]:
    return tuple(
        Paragraph(paragraph_id=f"{prefix}:{n:04d}", text=text) for n, text in enumerate(texts)
    )


def passages(*texts: str, prefix: str = "p") -> list[Passage]:
    return [Passage(paragraph_ids=(f"{prefix}{n}",), text=text) for n, text in enumerate(texts)]


def labels(pairs: Sequence[AlignedPair]) -> list[str]:
    return [pair.label.value for pair in pairs]


# Passage repair


def test_running_header_paragraphs_are_dropped():
    result = prepare_passages(paragraphs("First note.", HEADER, "Second note.", HEADER))
    assert [p.text for p in result] == ["First note.", "Second note."]
    assert [p.paragraph_ids for p in result] == [
        ("acc:financial_notes:0000",),
        ("acc:financial_notes:0002",),
    ]


def test_unaudited_header_and_hyphen_variant_are_dropped():
    result = prepare_passages(
        paragraphs(
            "A.",
            "NOTES TO UNAUDITED CONSOLIDATED FINANCIAL STATEMENTS—(Continued)",
            "B.",
            "Notes to Consolidated Financial Statements - (continued)",
            "C.",
        )
    )
    assert [p.text for p in result] == ["A.", "B.", "C."]


def test_a_header_mentioned_inside_a_sentence_is_kept():
    text = "See NOTES TO CONSOLIDATED FINANCIAL STATEMENTS—(Continued) for details."
    assert [p.text for p in prepare_passages(paragraphs(text))] == [text]


def test_a_sentence_split_around_a_header_is_joined():
    result = prepare_passages(
        paragraphs("The estimated costs", HEADER, "associated with these programs are accrued.")
    )
    assert len(result) == 1
    assert result[0].text == "The estimated costs associated with these programs are accrued."
    assert result[0].paragraph_ids == ("acc:financial_notes:0000", "acc:financial_notes:0002")


@pytest.mark.parametrize("second", ["31, 2022 on a prospective basis.", "$79,136 to property."])
def test_a_split_continuing_with_a_figure_is_joined(second):
    result = prepare_passages(
        paragraphs("Entities may apply this through December", HEADER, second)
    )
    assert len(result) == 1


def test_a_sentence_split_over_two_pages_is_joined_twice():
    result = prepare_passages(paragraphs("One", HEADER, "two", HEADER, "three."))
    assert [p.text for p in result] == ["One two three."]
    assert len(result[0].paragraph_ids) == 3


def test_a_finished_sentence_is_not_joined():
    result = prepare_passages(paragraphs("A complete sentence.", HEADER, "continues oddly."))
    assert len(result) == 2


def test_a_capitalized_paragraph_after_a_header_is_not_joined():
    result = prepare_passages(paragraphs("Inventories", HEADER, "Inventories are valued at cost."))
    assert len(result) == 2


def test_no_join_without_a_header_between():
    # Without a header, an unpunctuated paragraph is as likely to introduce a list.
    result = prepare_passages(paragraphs("The plans include", "lower costs; and"))
    assert len(result) == 2


def test_a_header_first_or_alone_leaves_nothing_to_join():
    assert [p.text for p in prepare_passages(paragraphs(HEADER, "starts lowercase."))] == [
        "starts lowercase."
    ]
    assert prepare_passages(paragraphs(HEADER)) == ()


def test_blank_paragraphs_are_skipped():
    assert [p.text for p in prepare_passages(paragraphs("A.", "   ", "B."))] == ["A.", "B."]


# Carried-forward text


def test_years_moved_on_by_the_gap_are_carried_forward():
    assert is_carried_forward(
        "Accruals were $227 and $221 at December 31, 2024 and 2023, respectively.",
        "Accruals were $227 and $221 at December 31, 2025 and 2024, respectively.",
        YEARLY,
    )


def test_a_changed_figure_is_not_carried_forward():
    assert not is_carried_forward(
        "Accruals were $227 at December 31, 2024.",
        "Accruals were $228 at December 31, 2025.",
        YEARLY,
    )


def test_a_year_moved_by_more_than_the_gap_is_not_carried_forward():
    assert not is_carried_forward("Notes due in 2030.", "Notes due in 2032.", YEARLY)


def test_years_never_roll_between_consecutive_quarters():
    assert not is_carried_forward("As of 2025.", "As of 2026.", QUARTERLY)


def test_the_period_end_month_rolls_between_quarters():
    assert is_carried_forward(
        "No options were granted in the period ended March 29, 2026.",
        "No options were granted in the period ended June 28, 2026.",
        QUARTERLY,
    )


def test_a_month_other_than_the_period_end_does_not_roll():
    assert not is_carried_forward("Paid in April.", "Paid in July.", QUARTERLY)


def test_months_never_roll_when_both_periods_end_in_the_same_month():
    assert not is_carried_forward("Paid in June 2025.", "Paid in July 2026.", YOY_10Q)


def test_the_day_of_a_period_end_date_may_move():
    assert is_carried_forward(
        "For the six months ended June 29, 2025 and June 30, 2024.",
        "For the six months ended June 28, 2026 and June 29, 2025.",
        YOY_10Q,
    )


def test_the_day_of_an_event_date_may_not_move():
    assert not is_carried_forward(
        "The act was signed on July 4, 2025.", "The act was signed on July 7, 2026.", YOY_10Q
    )


def test_a_52_53_week_year_end_may_move_the_calendar_year_by_two():
    assert is_carried_forward(
        "For the years ended December 28, 2024 and December 30, 2023.",
        "For the years ended January 3, 2026 and December 28, 2024.",
        STANLEY_YEARLY,
    )
    # Fiscal-year mentions still roll by one.
    assert is_carried_forward("In fiscal 2024 we sold.", "In fiscal 2025 we sold.", STANLEY_YEARLY)
    assert not is_carried_forward("Due in 2030.", "Due in 2033.", STANLEY_YEARLY)


def test_a_year_earlier_date_in_the_other_period_month_may_move_its_day():
    assert is_carried_forward(
        "For the six months ended June 28, 2025 and June 29, 2024.",
        "For the six months ended July 4, 2026 and June 28, 2025.",
        STANLEY_YOY_10Q,
    )
    assert not is_carried_forward("Signed May 1, 2025.", "Signed May 9, 2026.", STANLEY_YOY_10Q)


def test_punctuation_and_typography_are_ignored():
    assert is_carried_forward(
        "Hershey’s “brands” – chocolate, mints and gum.",
        'Hershey\'s "brands" - chocolate, mints, and gum',
        YEARLY,
    )


@pytest.mark.parametrize(
    ("prior", "current"),
    [
        # Each of these scored 98 or more out of 100 on Hershey, yet changes the meaning.
        ("We have two programs.", "We have four programs."),
        ("This could hurt results.", "This can hurt results."),
        ("Our CEO is Michele Buck.", "Our CEO is Kirk Tanner."),
        ("Costs rose.", "Costs rose sharply."),
        ("Costs rose sharply.", "Costs rose."),
    ],
)
def test_any_other_difference_is_a_modification(prior, current):
    assert not is_carried_forward(prior, current, YEARLY)


def test_period_roll_is_read_from_the_comparison(make_filing):
    yoy = Comparison(
        comparison_id="HSY:yoy-10k",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10K,
        current=make_filing(year=2025),
        prior=make_filing(year=2024),
    )
    assert PeriodRoll.of(yoy) == YEARLY
    seq = Comparison(
        comparison_id="HSY:seq-10q",
        ticker="HSY",
        comparison_type=ComparisonType.SEQ_10Q,
        current=make_filing(FormType.FORM_10Q, 2026, "Q2", date(2026, 6, 28)),
        prior=make_filing(FormType.FORM_10Q, 2026, "Q1", date(2026, 3, 29)),
    )
    assert PeriodRoll.of(seq) == FIRST_QUARTER
    yoy_10q = Comparison(
        comparison_id="HSY:yoy-10q",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10Q,
        current=make_filing(FormType.FORM_10Q, 2026, "Q2", date(2026, 6, 28)),
        prior=make_filing(FormType.FORM_10Q, 2025, "Q2", date(2025, 6, 29)),
    )
    assert not PeriodRoll.of(yoy_10q).prior_quarter_is_year_to_date
    second_to_third = seq.model_copy(
        update={
            "current": make_filing(FormType.FORM_10Q, 2026, "Q3", date(2026, 9, 27)),
            "prior": make_filing(FormType.FORM_10Q, 2026, "Q2", date(2026, 6, 28)),
        }
    )
    assert not PeriodRoll.of(second_to_third).prior_quarter_is_year_to_date
    # Two first quarters a year apart each report every figure once: nothing to separate.
    first_to_first = seq.model_copy(
        update={
            "current": make_filing(FormType.FORM_10Q, 2026, "Q1", date(2026, 3, 29)),
            "prior": make_filing(FormType.FORM_10Q, 2025, "Q1", date(2025, 3, 30)),
        }
    )
    assert not PeriodRoll.of(first_to_first).prior_quarter_is_year_to_date
    stanley = Comparison(
        comparison_id="SWK:yoy-10k",
        ticker="SWK",
        comparison_type=ComparisonType.YOY_10K,
        current=make_filing(year=2025, period_end=date(2026, 1, 3), ticker="SWK"),
        prior=make_filing(year=2024, period_end=date(2024, 12, 28), ticker="SWK"),
    )
    assert PeriodRoll.of(stanley) == STANLEY_YEARLY


# Matching


def test_identical_text_pairs_off_without_embedding():
    embedder = TopicEmbedder()
    pairs = align_passages(
        passages("Alpha one.", "Beta two."), passages("Alpha one.", "Beta two."), embedder, YEARLY
    )
    assert labels(pairs) == ["unchanged", "unchanged"]
    assert all(pair.similarity is None for pair in pairs)
    assert embedder.calls == []


def test_identical_text_pairs_off_in_document_order():
    pairs = align_passages(
        passages("Same.", "Same.", prefix="c"),
        passages("Same.", "Same.", prefix="p"),
        TopicEmbedder(),
        YEARLY,
    )
    assert [(pair.current.paragraph_ids, pair.prior.paragraph_ids) for pair in pairs] == [
        (("c0",), ("p0",)),
        (("c1",), ("p1",)),
    ]


def test_only_the_text_that_differs_is_embedded():
    embedder = TopicEmbedder()
    align_passages(
        passages("Same text.", "Revenue rose 5%."),
        passages("Same text.", "Revenue rose 3%."),
        embedder,
        YEARLY,
    )
    assert embedder.calls == [["Revenue rose 5%.", "Revenue rose 3%."]]


def test_a_changed_paragraph_is_modified_and_keeps_its_similarity():
    pairs = align_passages(
        passages("Revenue rose 5%."), passages("Revenue rose 3%."), TopicEmbedder(), YEARLY
    )
    assert labels(pairs) == ["modified"]
    assert pairs[0].similarity == pytest.approx(1.0)


def test_a_rolled_forward_paragraph_is_unchanged_boilerplate():
    pairs = align_passages(
        passages("Accruals at December 31, 2025 and 2024."),
        passages("Accruals at December 31, 2024 and 2023."),
        TopicEmbedder(),
        YEARLY,
    )
    assert labels(pairs) == ["unchanged"]
    assert pairs[0].similarity == pytest.approx(1.0)


def test_unrelated_paragraphs_are_added_and_removed():
    pairs = align_passages(
        passages("Tariffs rose."), passages("Leases ended."), TopicEmbedder(), YEARLY
    )
    assert labels(pairs) == ["added", "removed"]
    assert pairs[0].current.text == "Tariffs rose."
    assert pairs[1].prior.text == "Leases ended."


def test_the_assignment_matches_each_paragraph_to_its_own_topic():
    pairs = align_passages(
        passages("Leases changed.", "Tariffs changed.", "Pensions are new."),
        passages("Tariffs were low.", "Leases were long."),
        TopicEmbedder(),
        YEARLY,
    )
    assert [
        (p.label.value, p.current and p.current.text, p.prior and p.prior.text) for p in pairs
    ] == [
        ("modified", "Leases changed.", "Leases were long."),
        ("modified", "Tariffs changed.", "Tariffs were low."),
        ("added", "Pensions are new.", None),
    ]


class BatchNoiseEmbedder(FixedEmbedder):
    """Like bge on a CPU: a vector moves a little with where its text falls in the call."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        super().__init__(vectors)
        self.calls: list[list[str]] = []

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # Far below the rounding step, as the real noise is.
        return [
            [v + (1e-7 * k if d == 1 else 0.0) for d, v in enumerate(self.vectors[text])]
            for k, text in enumerate(texts)
        ]


def test_a_repeated_paragraph_takes_its_partners_by_position_not_by_noise():
    # Mattel's quarter and year to date both say the same sentence; the earlier filing's quarter
    # said 10% and its year to date 7%. Both earlier sentences are equally similar to the copies.
    copy, filler = "Vehicles billings rose 7%.", "Other text stays the same."
    quarter, year = "Vehicles billings rose 10% in the quarter.", "Vehicles billings rose 7% ytd."
    vectors = {copy: [1.0, 0.0, 0.0], quarter: [0.9, 0.436, 0.0], year: [0.9, 0.0, 0.436]}
    embedder = BatchNoiseEmbedder(vectors)
    pairs = align_passages(
        passages(copy, filler, copy), passages(quarter, filler, year), embedder, YOY_10Q
    )
    # Each text is embedded once, so the two copies can't come back a few bits apart; with the
    # copy embedded twice, the noise above would cross the pairs.
    assert embedder.calls == [[copy, quarter, year]]
    assert [(p.current.paragraph_ids, p.prior.paragraph_ids) for p in pairs] == [
        (("p0",), ("p0",)),
        (("p1",), ("p1",)),
        (("p2",), ("p2",)),
    ]


def test_position_is_the_place_in_the_whole_section():
    # The later copy closes its section (0.95). Of the two equally similar earlier sentences, the
    # one closing its section (0.975) is the nearer, not the one halfway (0.475), even though
    # among the passages left unmatched the halfway one sits nearer the end of its own filing.
    fillers = [f"Filler{k} stays." for k in range(9)]
    current = passages(*fillers, "Vehicles rose 7%.")
    prior = passages(
        *[f"Gone{k} text." for k in range(9)], "Vehicles rose 10%.", *fillers, "Vehicles rose 9%."
    )
    pairs = align_passages(current, prior, TopicEmbedder(), YOY_10Q)
    assert pairs[9].label is PairLabel.MODIFIED
    assert pairs[9].prior.text == "Vehicles rose 9%."


@pytest.mark.parametrize(
    ("prior_positions", "expected"), [([0.25, 0.75], [0, 1]), ([0.75, 0.25], [1, 0])]
)
def test_a_tie_goes_to_the_pairing_with_closer_positions(prior_positions, expected):
    # The other pairing wins by 0.8 of a rounding step: noise, more than the position term.
    noise = 0.4 * SIMILARITY_RESOLUTION
    favored = np.array([[0.9, 0.9], [0.9, 0.9]])
    for row, column in enumerate(expected):
        favored[row, 1 - column] += noise
    rows, columns = _assign(favored, np.array([0.25, 0.75]), np.array(prior_positions))
    assert (list(rows), list(columns)) == ([0, 1], expected)


@pytest.mark.parametrize("steps", [1, 3, 100_000])
def test_position_never_overrides_a_real_difference_in_similarity(steps):
    # The positions favor crossing as strongly as they can; one step of similarity still wins.
    similarity = np.array([[0.9 + steps * SIMILARITY_RESOLUTION, 0.9], [0.9, 0.9]])
    rows, columns = _assign(similarity, np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert (list(rows), list(columns)) == ([0, 1], [0, 1])


def test_position_is_only_a_tie_break_across_many_pairs():
    # Forty pairs, all at the worst positions, can't outweigh one rounding step of similarity.
    n = 40
    similarity = np.full((n, n), 0.8)
    np.fill_diagonal(similarity, 0.8 + 2 * SIMILARITY_RESOLUTION)
    rows, columns = _assign(similarity, np.linspace(0, 1, n), np.linspace(1, 0, n))
    assert list(columns) == list(range(n))


# A first quarter against a later quarter (HSY-F14: Hershey's Q1 2026 segment paragraph went to
# the Q2 six-month paragraph and left the Q2 quarterly one as added).

Q1_SALES = "Net sales were $3,104.2 million in the first quarter of 2026."
Q2_SALES = "Net sales were $2,787.3 million in the second quarter of 2026."
H1_SALES = "Net sales were $5,891.5 million in the first six months of 2026."


def at_cosine(cosine: float, axis: int) -> list[float]:
    # A unit vector with this cosine to [1, 0, 0, 0], leaning into its own axis.
    vector = [cosine, 0.0, 0.0, 0.0]
    vector[axis] = (1 - cosine**2) ** 0.5
    return vector


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (Q1_SALES, PeriodKind.QUARTER),
        (
            "Tools & Outdoor Segment profit for the first three months of 2026 was $276.0 million.",
            PeriodKind.QUARTER,
        ),
        (
            "The components of lease expense for the three months ended March 29, 2026 were:",
            PeriodKind.QUARTER,
        ),
        (H1_SALES, PeriodKind.YEAR_TO_DATE),
        ("Net sales in the first half of 2026 were $1.99 billion.", PeriodKind.YEAR_TO_DATE),
        (
            "On a year-to-date basis, corporate overhead amounted to $146.3 million.",
            PeriodKind.YEAR_TO_DATE,
        ),
        (
            "Net sales for the nine months ended September 27, 2026 rose 3%.",
            PeriodKind.YEAR_TO_DATE,
        ),
        ("For the three and six months ended July 4, 2026, the Company recognized a charge.", None),
        ("Gross billings represent amounts invoiced to a customer.", None),
    ],
)
def test_the_period_kind_is_read_from_the_paragraph(text, kind):
    assert period_kind(text) is kind


def test_the_period_is_read_from_the_opening_words_only():
    padding = ["Earlier"] * (PERIOD_WORDS - 2)
    assert period_kind(" ".join([*padding, "first", "quarter"])) is PeriodKind.QUARTER
    assert period_kind(" ".join([*padding, "words", "then", "first", "quarter"])) is None


def first_quarter_embedder(quarter: float, year_to_date: float = 0.98) -> FixedEmbedder:
    return FixedEmbedder(
        {
            Q1_SALES: at_cosine(1.0, 1),
            Q2_SALES: at_cosine(quarter, 2),
            H1_SALES: at_cosine(year_to_date, 3),
        }
    )


def test_a_first_quarter_paragraph_pairs_with_the_later_quarter():
    # The six-month paragraph reads closer (0.98 against 0.95), as on Hershey.
    embedder = first_quarter_embedder(quarter=0.95)
    pairs = align_passages(
        passages(Q2_SALES, H1_SALES), passages(Q1_SALES), embedder, FIRST_QUARTER
    )
    assert labels(pairs) == ["modified", "added"]
    assert pairs[0].prior is not None and pairs[0].prior.text == Q1_SALES
    # The recorded similarity is the embeddings', not the penalized one.
    assert pairs[0].similarity == pytest.approx(0.95)


def test_without_a_first_quarter_the_closer_paragraph_wins():
    # In any other comparison both filings report both periods, so nothing is penalized.
    embedder = first_quarter_embedder(quarter=0.95)
    pairs = align_passages(passages(Q2_SALES, H1_SALES), passages(Q1_SALES), embedder, QUARTERLY)
    assert labels(pairs) == ["added", "modified"]


@pytest.mark.parametrize(
    ("gap", "partner"), [(0.03, Q2_SALES), (0.049, Q2_SALES), (0.051, H1_SALES), (0.08, H1_SALES)]
)
def test_the_penalty_only_outweighs_a_smaller_pull(gap, partner):
    # A first-quarter paragraph whose quarterly partner reads much further away keeps the year to
    # date: the penalty is a preference, not a ban.
    assert PERIOD_MISMATCH_PENALTY == 0.05
    embedder = first_quarter_embedder(quarter=0.98 - gap)
    pairs = align_passages(
        passages(Q2_SALES, H1_SALES), passages(Q1_SALES), embedder, FIRST_QUARTER
    )
    matched = [pair for pair in pairs if pair.label is PairLabel.MODIFIED]
    assert len(matched) == 1 and matched[0].current is not None
    assert matched[0].current.text == partner


def test_the_floor_reads_the_unpenalized_similarity():
    # 0.78 clears the floor; 0.78 less the penalty would not.
    assert 0.78 - PERIOD_MISMATCH_PENALTY < MIN_MATCH_SIMILARITY <= 0.78
    embedder = first_quarter_embedder(quarter=0.0, year_to_date=0.78)
    pairs = align_passages(passages(H1_SALES), passages(Q1_SALES), embedder, FIRST_QUARTER)
    assert labels(pairs) == ["modified"]
    assert pairs[0].similarity == pytest.approx(0.78)


def test_a_paragraph_naming_no_period_is_never_penalized():
    plain = "Gross billings represent amounts invoiced to a customer."
    embedder = FixedEmbedder(
        {Q1_SALES: at_cosine(1.0, 1), plain: at_cosine(0.96, 2), Q2_SALES: at_cosine(0.93, 3)}
    )
    pairs = align_passages(passages(plain, Q2_SALES), passages(Q1_SALES), embedder, FIRST_QUARTER)
    assert labels(pairs) == ["modified", "added"]


@pytest.mark.parametrize(
    ("delta", "expected"), [(0.001, ["modified"]), (-0.001, ["added", "removed"])]
)
def test_the_similarity_floor_decides_a_match(delta, expected):
    # Two unit vectors whose cosine is the floor plus or minus a little.
    cosine = MIN_MATCH_SIMILARITY + delta
    vectors = {"Later text.": [1.0, 0.0], "Earlier text.": [cosine, (1 - cosine**2) ** 0.5]}
    pairs = align_passages(
        passages("Later text."), passages("Earlier text."), FixedEmbedder(vectors), YEARLY
    )
    assert labels(pairs) == expected


def test_vectors_are_normalized_before_comparing():
    vectors = {"Later text.": [10.0, 0.0], "Earlier text.": [0.0, 0.0], "Other.": [3.0, 0.0]}
    pairs = align_passages(
        passages("Later text."), passages("Other."), FixedEmbedder(vectors), YEARLY
    )
    assert labels(pairs) == ["modified"]
    assert pairs[0].similarity == pytest.approx(1.0)


def test_a_surplus_copy_of_shared_text_is_unchanged():
    footer = "See Notes to Consolidated Financial Statements."
    embedder = TopicEmbedder()
    pairs = align_passages(passages(footer, footer, footer), passages(footer), embedder, YEARLY)
    assert labels(pairs) == ["unchanged"] * 3
    pairs = align_passages(passages(footer), passages(footer, footer), embedder, YEARLY)
    assert labels(pairs) == ["unchanged", "unchanged"]
    assert embedder.calls == []


def test_removed_paragraphs_follow_the_later_filing_in_their_own_order():
    pairs = align_passages(
        passages("Kept."),
        passages("Gone one.", "Kept.", "Vanished two."),
        TopicEmbedder(),
        YEARLY,
    )
    assert labels(pairs) == ["unchanged", "removed", "removed"]
    assert [pair.prior.text for pair in pairs[1:]] == ["Gone one.", "Vanished two."]


def test_one_empty_side_needs_no_embedding():
    embedder = TopicEmbedder()
    assert labels(align_passages(passages("New."), [], embedder, YEARLY)) == ["added"]
    assert labels(align_passages([], passages("Old."), embedder, YEARLY)) == ["removed"]
    assert align_passages([], [], embedder, YEARLY) == ()
    assert embedder.calls == []


def test_a_pair_must_have_the_passages_its_label_needs():
    passage = Passage(paragraph_ids=("p",), text="x")
    with pytest.raises(ValueError, match="added"):
        AlignedPair(label=PairLabel.ADDED, prior=passage)
    with pytest.raises(ValueError, match="modified"):
        AlignedPair(label=PairLabel.MODIFIED, current=passage)
    with pytest.raises(ValueError, match="removed"):
        AlignedPair(label=PairLabel.REMOVED, current=passage, prior=passage)
    with pytest.raises(ValueError):
        Passage(paragraph_ids=(), text="x")


# Section status


def section(
    status: SectionStatus, *texts: str, name: Section = Section.RISK_FACTORS
) -> ParsedSection:
    return ParsedSection(
        section=name, status=status, paragraphs=paragraphs(*texts, prefix=f"acc:{name.value}")
    )


CONTENT = SectionStatus.CONTENT
NMC = SectionStatus.NO_MATERIAL_CHANGES
XREF = SectionStatus.CROSS_REFERENCE
MISSING = SectionStatus.MISSING


@pytest.mark.parametrize(
    ("current", "prior", "expected"),
    [
        (CONTENT, CONTENT, SectionOutcome.COMPARED),
        (CONTENT, NMC, SectionOutcome.COMPARED),
        (NMC, NMC, SectionOutcome.NO_MATERIAL_CHANGES),
        (NMC, CONTENT, SectionOutcome.NO_MATERIAL_CHANGES),
        (NMC, None, SectionOutcome.NO_MATERIAL_CHANGES),
        (XREF, XREF, SectionOutcome.CROSS_REFERENCE),
        (CONTENT, XREF, SectionOutcome.CROSS_REFERENCE),
        (XREF, CONTENT, SectionOutcome.CROSS_REFERENCE),
        (CONTENT, MISSING, SectionOutcome.MISSING),
        (MISSING, CONTENT, SectionOutcome.MISSING),
        (CONTENT, None, SectionOutcome.MISSING),
        (None, CONTENT, SectionOutcome.MISSING),
        (XREF, None, SectionOutcome.MISSING),
    ],
)
def test_section_outcome(current, prior, expected):
    def build(status):
        return None if status is None else section(status, "Some text.")

    assert section_outcome(build(current), build(prior)) is expected


def test_no_material_changes_reports_no_pairs():
    result = align_section(
        Section.RISK_FACTORS,
        section(NMC, "There have been no material changes to our risk factors."),
        section(CONTENT, "A new risk about tariffs."),
        TopicEmbedder(),
        YOY_10Q,
    )
    assert result.outcome is SectionOutcome.NO_MATERIAL_CHANGES
    assert result.pairs == ()
    assert (result.current_status, result.prior_status) == (NMC, CONTENT)


def test_content_after_no_material_changes_is_all_added():
    result = align_section(
        Section.RISK_FACTORS,
        section(CONTENT, "A new risk about tariffs.", "A new risk about AI."),
        section(NMC, "There have been no material changes to our risk factors."),
        TopicEmbedder(),
        YOY_10Q,
    )
    assert result.outcome is SectionOutcome.COMPARED
    assert labels(result.pairs) == ["added", "added"]


def test_cross_reference_and_missing_report_no_pairs():
    embedder = TopicEmbedder()
    xref = align_section(
        Section.MARKET_RISK,
        section(XREF, "See Item 7."),
        section(XREF, "See Item 7."),
        embedder,
        YEARLY,
    )
    missing = align_section(Section.MDNA, section(CONTENT, "Text."), None, embedder, YEARLY)
    assert (xref.outcome, xref.pairs) == (SectionOutcome.CROSS_REFERENCE, ())
    assert (missing.outcome, missing.pairs, missing.prior_status) == (
        SectionOutcome.MISSING,
        (),
        None,
    )
    assert embedder.calls == []


def test_a_compared_section_counts_its_labels():
    result = align_section(
        Section.FINANCIAL_NOTES,
        section(
            CONTENT, "Kept.", "Leases rose.", "Pensions are new.", name=Section.FINANCIAL_NOTES
        ),
        section(CONTENT, "Kept.", "Leases fell.", "Tariffs ended.", name=Section.FINANCIAL_NOTES),
        TopicEmbedder(),
        YEARLY,
    )
    assert [result.count(label) for label in PairLabel] == [1, 1, 1, 1]


def test_the_header_repair_runs_inside_section_alignment():
    # Each year's page break falls in a different place, but the joined text is the same.
    later = section(CONTENT, "If the fair value of the", HEADER, "reporting unit is low, we test.")
    earlier = section(
        CONTENT, "If the fair value of", HEADER, "the reporting unit is low, we test."
    )
    result = align_section(Section.RISK_FACTORS, later, earlier, TopicEmbedder(), YEARLY)
    assert labels(result.pairs) == ["unchanged"]


# Filings, changes and the whole comparison


def comparison(make_filing) -> Comparison:
    return Comparison(
        comparison_id="HSY:yoy-10k",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10K,
        current=make_filing(year=2025),
        prior=make_filing(year=2024),
    )


def parsed(accession: str, sections: dict[Section, ParsedSection]) -> ParsedFiling:
    return ParsedFiling(
        accession=accession,
        form=FormType.FORM_10K,
        parser=ParserName.EDGARTOOLS,
        layout="edgartools",
        sections=sections,
        coverage=1.0,
        table_audit=TableAudit(tables_total=0, tables_removed=0, prose_rows_preserved=0),
        problems=(),
    )


def filing_sections(accession: str, mdna: tuple[str, ...], notes: tuple[str, ...]):
    return {
        Section.MDNA: ParsedSection(
            section=Section.MDNA,
            status=CONTENT,
            paragraphs=paragraphs(*mdna, prefix=f"{accession}:mdna"),
        ),
        Section.FINANCIAL_NOTES: ParsedSection(
            section=Section.FINANCIAL_NOTES,
            status=CONTENT,
            paragraphs=paragraphs(*notes, prefix=f"{accession}:financial_notes"),
        ),
    }


def test_align_filings_covers_sections_either_filing_has_in_section_order(make_filing):
    comp = comparison(make_filing)
    current = parsed(
        comp.current.accession, filing_sections(comp.current.accession, ("A.",), ("B.",))
    )
    prior_sections = filing_sections(comp.prior.accession, ("A.",), ("B.",))
    prior_sections[Section.BUSINESS] = section(CONTENT, "We make chocolate.", name=Section.BUSINESS)
    prior = parsed(comp.prior.accession, prior_sections)
    result = align_filings(comp, current, prior, TopicEmbedder())
    assert [(s.section, s.outcome) for s in result] == [
        (Section.BUSINESS, SectionOutcome.MISSING),
        (Section.MDNA, SectionOutcome.COMPARED),
        (Section.FINANCIAL_NOTES, SectionOutcome.COMPARED),
    ]


def test_align_filings_refuses_filings_of_another_comparison(make_filing):
    comp = comparison(make_filing)
    current = parsed(comp.current.accession, {})
    other = parsed("someone-else", {})
    with pytest.raises(ValueError, match="HSY:yoy-10k"):
        align_filings(comp, current, other, TopicEmbedder())
    with pytest.raises(ValueError, match="HSY:yoy-10k"):
        align_filings(comp, other, current, TopicEmbedder())


def test_changes_cite_every_chunk_of_a_joined_passage(make_filing):
    comp = comparison(make_filing)
    sections = (
        align_section(
            Section.FINANCIAL_NOTES,
            section(
                CONTENT, "Kept.", "Leases now", HEADER, "end in 2031.", name=Section.FINANCIAL_NOTES
            ),
            section(CONTENT, "Kept.", "Leases end in 2029.", name=Section.FINANCIAL_NOTES),
            TopicEmbedder(),
            YEARLY,
        ),
    )
    notes = "acc:financial_notes"
    chunks = FakeChunks(
        {
            f"{notes}:0001": ("c1", "c2"),
            f"{notes}:0003": ("c2", "c3"),
            f"{notes}:0000": ("c0",),
        }
    )
    (change,) = changes_for(comp, sections, chunks)
    assert change.kind is ChangeKind.MODIFIED
    assert change.change_id == f"HSY:yoy-10k:modified:{notes}:0001"
    assert change.section is Section.FINANCIAL_NOTES
    assert change.current_text == "Leases now end in 2031."
    assert change.prior_text == "Leases end in 2029."
    # Duplicates removed, order kept; the earlier filing used the same ids in this test.
    assert change.current_chunk_ids == ("c1", "c2", "c3")
    assert change.prior_chunk_ids == ("c1", "c2")
    assert f"{notes}:0000" not in chunks.asked  # unchanged text is never looked up


def test_added_and_removed_changes_have_one_side(make_filing):
    comp = comparison(make_filing)
    sections = (
        align_section(
            Section.RISK_FACTORS,
            ParsedSection(
                section=Section.RISK_FACTORS,
                status=CONTENT,
                paragraphs=paragraphs("Tariffs are a new risk.", prefix="cur:risk_factors"),
            ),
            ParsedSection(
                section=Section.RISK_FACTORS,
                status=CONTENT,
                paragraphs=paragraphs("Pandemics were a risk.", prefix="pri:risk_factors"),
            ),
            TopicEmbedder(),
            YEARLY,
        ),
    )
    chunks = FakeChunks({"cur:risk_factors:0000": ("cur-c",), "pri:risk_factors:0000": ("pri-c",)})
    added, removed = changes_for(comp, sections, chunks)
    assert (added.kind, added.change_id) == (
        ChangeKind.ADDED,
        "HSY:yoy-10k:added:cur:risk_factors:0000",
    )
    assert (added.prior_text, added.current_chunk_ids, added.prior_chunk_ids) == (
        None,
        ("cur-c",),
        (),
    )
    assert (removed.kind, removed.change_id) == (
        ChangeKind.REMOVED,
        "HSY:yoy-10k:removed:pri:risk_factors:0000",
    )
    assert (removed.current_text, removed.current_chunk_ids, removed.prior_chunk_ids) == (
        None,
        (),
        ("pri-c",),
    )


def test_align_comparison_reads_the_saved_filings(settings, make_filing):
    comp = comparison(make_filing)
    for accession, mdna in (
        (comp.current.accession, ("Revenue rose 5% in 2025.", "Kept as is.")),
        (comp.prior.accession, ("Revenue rose 3% in 2024.", "Kept as is.")),
    ):
        save_parsed_filing(
            parsed(accession, filing_sections(accession, mdna, ("Accruals at December 31.",))),
            parsed_filing_path(settings.parsed_filings_dir, "HSY", accession),
        )
    result = align_comparison(comp, settings.parsed_filings_dir, TopicEmbedder(), FakeChunks({}))
    assert result.comparison_id == "HSY:yoy-10k"
    assert [(s.section, labels(s.pairs)) for s in result.sections] == [
        (Section.MDNA, ["modified", "unchanged"]),
        (Section.FINANCIAL_NOTES, ["unchanged"]),
    ]
    assert [(c.kind, c.current_text) for c in result.changes] == [
        (ChangeKind.MODIFIED, "Revenue rose 5% in 2025.")
    ]


def test_align_comparison_refuses_a_missing_or_rejected_filing(settings, make_filing):
    comp = comparison(make_filing)
    with pytest.raises(FilingRejectedError):
        align_comparison(comp, settings.parsed_filings_dir, TopicEmbedder(), FakeChunks({}))
    for accession, problems in (
        (comp.current.accession, ("coverage too low",)),
        (comp.prior.accession, ()),
    ):
        record = parsed(accession, {}).model_copy(update={"problems": problems})
        save_parsed_filing(
            record, parsed_filing_path(settings.parsed_filings_dir, "HSY", accession)
        )
    with pytest.raises(FilingRejectedError):
        align_comparison(comp, settings.parsed_filings_dir, TopicEmbedder(), FakeChunks({}))


# The real pinned filings

# Every sentence the Hershey header splits, counted by hand from the printed join points.
HERSHEY_JOINS = {
    ("10-K", 2025, "FY"): 4,
    ("10-K", 2024, "FY"): 8,
    ("10-Q", 2026, "Q2"): 0,
    ("10-Q", 2025, "Q2"): 1,
    ("10-Q", 2026, "Q1"): 1,
}
# Sections that can't be compared paragraph by paragraph in the nine comparisons; every other
# section the filings have is compared.
NOT_COMPARED = {
    ("HSY:yoy-10q", Section.RISK_FACTORS): SectionOutcome.CROSS_REFERENCE,
    ("HSY:seq-10q", Section.RISK_FACTORS): SectionOutcome.CROSS_REFERENCE,
    ("MAT:yoy-10q", Section.RISK_FACTORS): SectionOutcome.NO_MATERIAL_CHANGES,
    ("MAT:seq-10q", Section.RISK_FACTORS): SectionOutcome.NO_MATERIAL_CHANGES,
    ("SWK:yoy-10k", Section.MARKET_RISK): SectionOutcome.CROSS_REFERENCE,
    ("SWK:yoy-10q", Section.MARKET_RISK): SectionOutcome.CROSS_REFERENCE,
    ("SWK:seq-10q", Section.MARKET_RISK): SectionOutcome.CROSS_REFERENCE,
}
_WORDS = re.compile(r"\w+")


class WordCountEmbedder:
    """Word counts hashed into a fixed vector: no model, yet similar paragraphs stay similar."""

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 512
            for word, n in Counter(_WORDS.findall(text.lower())).items():
                vector[zlib.crc32(word.encode()) % 512] += n
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("alignment embeds passages, never queries")


@pytest.fixture(scope="module")
def pinned():
    settings = get_settings()
    with FactStore(settings.duckdb_path, read_only=True) as store:
        yield settings, store, Corpus.load(REPO_ROOT / "config" / "corpus.yaml")


def check_changes(changes, store, comparison):
    assert len({change.change_id for change in changes}) == len(changes)
    for change in changes:
        for text, chunk_ids, filing in (
            (change.current_text, change.current_chunk_ids, comparison.current),
            (change.prior_text, change.prior_chunk_ids, comparison.prior),
        ):
            assert bool(text) == bool(chunk_ids), change.change_id
            assert not (text and RUNNING_HEADER.search(text)), change.change_id
            for chunk_id in chunk_ids:
                chunk = store.get_chunk(chunk_id)
                assert chunk is not None and chunk.accession == filing.accession
                assert chunk.section is change.section


@pytest.mark.corpus
def test_real_header_repair_touches_only_the_hershey_notes(pinned):
    settings, _, corpus = pinned
    for company in corpus.companies:
        for filing in company.filings:
            parsed_filing = load_validated_filing(
                parsed_filing_path(settings.parsed_filings_dir, company.ticker, filing.accession)
            )
            for name, parsed_section in parsed_filing.sections.items():
                result = prepare_passages(parsed_section.paragraphs)
                assert not any(RUNNING_HEADER.fullmatch(p.text) for p in result)
                joins = sum(len(p.paragraph_ids) - 1 for p in result)
                if company.ticker == "HSY" and name is Section.FINANCIAL_NOTES:
                    key = (filing.form.value, filing.fiscal_year, filing.fiscal_period)
                    assert joins == HERSHEY_JOINS[key], key
                else:
                    assert len(result) == len(parsed_section.paragraphs)


@pytest.mark.corpus
def test_real_all_nine_comparisons_align(pinned):
    # The word-count embedder keeps this quick; the real model is tested on one comparison below.
    settings, store, corpus = pinned
    comparisons = [c for company in corpus.companies for c in company.comparisons()]
    assert len(comparisons) == 9
    for comparison in comparisons:
        result = align_comparison(
            comparison, settings.parsed_filings_dir, WordCountEmbedder(), store
        )
        expected_sections = list(Section)[:5]
        if comparison.comparison_type is not ComparisonType.YOY_10K:
            expected_sections.remove(Section.BUSINESS)  # a 10-Q has no business section
        assert [s.section for s in result.sections] == expected_sections
        for alignment in result.sections:
            expected = NOT_COMPARED.get(
                (comparison.comparison_id, alignment.section), SectionOutcome.COMPARED
            )
            assert alignment.outcome is expected, (comparison.comparison_id, alignment.section)
            assert bool(alignment.pairs) == (expected is SectionOutcome.COMPARED)
        assert result.changes
        check_changes(result.changes, store, comparison)


@pytest.mark.corpus
def test_real_hershey_quarter_with_the_real_model_filters_boilerplate(pinned):
    settings, store, corpus = pinned
    comparison = corpus.comparison("HSY", ComparisonType.SEQ_10Q)
    result = align_comparison(comparison, settings.parsed_filings_dir, BgeEmbedder(), store)
    pairs = [pair for alignment in result.sections for pair in alignment.pairs]
    # Boilerplate whose dates rolled forward (March to June) is matched and filtered out.
    rolled = [p for p in pairs if p.label is PairLabel.UNCHANGED and p.current.text != p.prior.text]
    assert len(rolled) >= 10
    assert all(p.similarity is not None for p in rolled)
    assert len(result.changes) < len(pairs) / 2
    # A known edit in the MD&A: the tariff sentence now speaks of refunds.
    assert any(
        change.kind is ChangeKind.MODIFIED
        and change.section is Section.MDNA
        and "tariff refunds" in change.current_text
        and "tariff refunds" not in change.prior_text
        for change in result.changes
    )
    check_changes(result.changes, store, comparison)
