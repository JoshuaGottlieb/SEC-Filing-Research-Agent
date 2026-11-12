from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sec_agent.benchmark import (
    Benchmark,
    JudgeExample,
    LabeledChange,
    Quote,
    RetrievalQuestion,
    SampledChange,
    Side,
    Split,
    StatedCause,
    check_benchmark,
    load_benchmark,
    matching_chunks,
    resolve_quote,
    save_benchmark,
)
from sec_agent.schemas import (
    ChangeKind,
    Chunk,
    Comparison,
    ComparisonType,
    Direction,
    EntailmentVerdict,
    MaterialityCriterion,
    MetricName,
    Section,
)

CURRENT_TEXT = (
    "Gross margin was 33.5% in 2025 compared with 47.3% in 2024. The decrease was driven by "
    "higher commodity and tariff costs."
)
PRIOR_TEXT = "Gross margin was 47.3% in 2024. We completed the acquisition of Sour Strips."


@pytest.fixture
def comparison(make_filing) -> Comparison:
    return Comparison(
        comparison_id="HSY:yoy-10k",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10K,
        current=make_filing(year=2025),
        prior=make_filing(year=2024),
    )


@pytest.fixture
def chunks(comparison):
    def chunk(chunk_id: str, filing, text: str, section: Section = Section.MDNA) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            paragraph_id=chunk_id.rsplit(":", 1)[0],
            ticker="HSY",
            cik="0000047111",
            form=filing.form,
            period=filing.period,
            section=section,
            accession=filing.accession,
            text=text,
            token_count=30,
        )

    cur, pri = comparison.current, comparison.prior
    by_accession = {
        cur.accession: [
            chunk("c:mdna:0001:0", cur, CURRENT_TEXT),
            # The overlap repeats the last sentence at the start of the next chunk.
            chunk("c:mdna:0002:0", cur, "higher commodity and tariff costs. Volume fell 1%."),
            chunk("c:risk:0001:0", cur, "Tariffs could raise our costs.", Section.RISK_FACTORS),
        ],
        pri.accession: [chunk("p:mdna:0001:0", pri, PRIOR_TEXT)],
    }
    return lambda accession: by_accession.get(accession, [])


def quote(text: str, side: Side = Side.CURRENT) -> Quote:
    return Quote(side=side, text=text)


def question(**fields) -> RetrievalQuestion:
    base = {
        "item_id": "HSY-R01",
        "comparison_id": "HSY:yoy-10k",
        "section": Section.MDNA,
        "question": "gross margin decline drivers",
        "quotes": (quote("driven by higher commodity and tariff costs"),),
    }
    return RetrievalQuestion(**{**base, **fields})


def cause(**fields) -> StatedCause:
    base = {
        "item_id": "HSY-S01",
        "comparison_id": "HSY:yoy-10k",
        "section": Section.MDNA,
        "change": "Gross margin fell from 47.3% to 33.5%",
        "metric": MetricName.GROSS_MARGIN,
        "direction": Direction.DOWN,
        "causes": ("higher commodity costs", "tariff costs"),
        "quote": quote("The decrease was driven by higher commodity and tariff costs."),
        "distractors": ("pricing pressure",),
    }
    return StatedCause(**{**base, **fields})


def test_split_follows_the_ticker():
    assert question().split is Split.TUNE
    assert question(item_id="MAT-R01", comparison_id="MAT:yoy-10q").split is Split.TEST


def test_comparison_id_must_name_a_known_comparison_type():
    with pytest.raises(ValidationError):
        question(comparison_id="HSY:qoq")


def test_absent_sides_are_the_sides_without_quotes():
    assert question().absent_sides() == (Side.PRIOR,)
    both = question(quotes=(quote("Gross margin was 33.5%"), quote("47.3% in 2024", Side.PRIOR)))
    assert both.absent_sides() == ()


@pytest.mark.parametrize(
    ("kind", "sides", "ok"),
    [
        (ChangeKind.ADDED, [Side.CURRENT], True),
        (ChangeKind.ADDED, [Side.CURRENT, Side.PRIOR], False),
        (ChangeKind.REMOVED, [Side.PRIOR], True),
        (ChangeKind.REMOVED, [Side.CURRENT], False),
        (ChangeKind.MODIFIED, [Side.CURRENT, Side.PRIOR], True),
        (ChangeKind.MODIFIED, [Side.CURRENT], False),
    ],
)
def test_change_quotes_must_match_its_kind(kind, sides, ok):
    def build():
        return LabeledChange(
            item_id="HSY-C01",
            comparison_id="HSY:yoy-10k",
            section=Section.MDNA,
            kind=kind,
            criterion=MaterialityCriterion.CHANGED_QUANTITY,
            summary="Gross margin fell",
            quotes=tuple(quote("x", side) for side in sides),
        )

    if ok:
        build()
    else:
        with pytest.raises(ValidationError):
            build()


def test_numeric_change_kind_is_not_a_labeled_text_change():
    with pytest.raises(ValidationError):
        LabeledChange(
            item_id="HSY-C01",
            comparison_id="HSY:yoy-10k",
            section=Section.MDNA,
            kind=ChangeKind.NUMERIC,
            criterion=MaterialityCriterion.METRIC_LINKED_SHIFT,
            summary="s",
            quotes=(quote("x"),),
        )


def test_stated_cause_quotes_the_current_filing_and_pairs_metric_with_direction():
    with pytest.raises(ValidationError):
        cause(quote=quote("Gross margin was 47.3% in 2024.", Side.PRIOR))
    with pytest.raises(ValidationError):
        cause(direction=None)
    assert cause(metric=None, direction=None).metric is None


def test_benchmark_refuses_duplicate_ids_and_unknown_causes():
    with pytest.raises(ValidationError, match="duplicate"):
        Benchmark(retrieval=(question(), question()))
    example = JudgeExample(
        item_id="HSY-J01",
        comparison_id="HSY:yoy-10k",
        cause_id="HSY-S09",
        claim="Tariffs lowered gross margin",
        quote=quote("higher commodity and tariff costs"),
        verdict=EntailmentVerdict.STATES_CAUSE,
    )
    with pytest.raises(ValidationError, match="unknown cause"):
        Benchmark(causes=(cause(),), judge=(example,))
    Benchmark(causes=(cause(),), judge=(example.model_copy(update={"cause_id": "HSY-S01"}),))


def test_save_and_load_round_trip(tmp_path: Path):
    sampled = SampledChange(
        item_id="HSY-F01",
        comparison_id="HSY:yoy-10k",
        change_id="HSY:yoy-10k:added:x",
        kind=ChangeKind.ADDED,
        current_text="New text",
    )
    bench = Benchmark(retrieval=(question(),), causes=(cause(),), sampled=(sampled,))
    path = tmp_path / "b.json"
    save_benchmark(bench, path)
    assert load_benchmark(path) == bench


def test_matching_finds_every_overlapping_chunk(chunks, comparison):
    pool = chunks(comparison.current.accession)
    found = matching_chunks("higher commodity and tariff costs", pool)
    assert found == ("c:mdna:0001:0", "c:mdna:0002:0")


def test_matching_ignores_typography_and_case_but_not_a_cut_number(chunks, comparison):
    pool = chunks(comparison.current.accession)
    assert matching_chunks("GROSS MARGIN WAS 33.5%  in 2025", pool) == ("c:mdna:0001:0",)
    assert matching_chunks("3.5% in 2025", pool) == ()


def test_resolve_reads_the_quoted_side_and_respects_the_section(chunks, comparison):
    assert resolve_quote(quote("Sour Strips", Side.PRIOR), comparison, chunks) == ("p:mdna:0001:0",)
    assert resolve_quote(quote("Sour Strips"), comparison, chunks) == ()
    tariffs = quote("Tariffs could raise our costs.")
    assert resolve_quote(tariffs, comparison, chunks, Section.MDNA) == ()
    assert resolve_quote(tariffs, comparison, chunks, Section.RISK_FACTORS) == ("c:risk:0001:0",)


def test_check_reports_each_unusable_label(chunks, comparison):
    bench = Benchmark(
        retrieval=(
            question(),
            question(item_id="HSY-R02", quotes=(quote("Tariffs could raise our costs."),)),
            question(item_id="HSY-R03", comparison_id="HSY:seq-10q"),
        ),
        causes=(cause(quote=quote("The decrease was driven by lower volume.")),),
    )
    problems = check_benchmark(bench, {"HSY:yoy-10k": comparison}, chunks)
    assert len(problems) == 3
    assert problems[0].startswith("HSY-R02: current quote not found in mdna")
    assert problems[1].startswith("HSY-R03: unknown comparison")
    assert problems[2].startswith("HSY-S01: current quote not found")


def test_check_passes_clean_labels(chunks, comparison):
    bench = Benchmark(retrieval=(question(),), causes=(cause(),))
    assert check_benchmark(bench, {"HSY:yoy-10k": comparison}, chunks) == []
