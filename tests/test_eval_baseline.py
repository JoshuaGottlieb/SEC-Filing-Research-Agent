"""T6.5's baseline prompt and scorer (`eval/baseline/`), with no model calls."""

from __future__ import annotations

import json
import sys
from datetime import date
from typing import Any

import pytest

from sec_agent.config import REPO_ROOT
from sec_agent.schemas import (
    Chunk,
    Comparison,
    ComparisonType,
    EntailmentVerdict,
    FiscalPeriod,
    FormType,
    Section,
)

sys.path.insert(0, str(REPO_ROOT / "eval" / "baseline"))
import baseline  # noqa: E402
import score  # noqa: E402

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))
CURRENT_ACC = "0000047111-2025-000001"
PRIOR_ACC = "0000047111-2024-000001"


def chunk(chunk_id: str, text: str, *, prior: bool = False, section=Section.MDNA) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paragraph_id=chunk_id,
        ticker="HSY",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=PRIOR if prior else CURRENT,
        section=section,
        accession=PRIOR_ACC if prior else CURRENT_ACC,
        text=text,
        token_count=10,
    )


@pytest.fixture
def comparison(make_filing) -> Comparison:
    return Comparison(
        comparison_id="HSY:yoy-10k",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10K,
        current=make_filing(year=2025),
        prior=make_filing(year=2024),
    )


class FakeStore:
    """The chunk side of the fact store: `chunks(accession)` and `get_chunk`."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunks(self, accession: str | None = None) -> list[Chunk]:
        return [c for c in self._chunks if accession is None or c.accession == accession]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self._chunks if c.chunk_id == chunk_id), None)

    def find_usd_fact(self, accession: str, value: float, tolerance: float) -> str | None:
        return None


STORE_CHUNKS = [
    # The second chunk starts with the end of the first, as the chunker's overlap does.
    chunk("c-1", "Gross margin decreased due to higher cocoa costs. Volume was lower."),
    chunk("c-2", "Volume was lower. Prices rose 5% in North America."),
    chunk("c-3", "Cocoa futures are volatile.", section=Section.MARKET_RISK),
    chunk("p-1", "Gross margin increased due to price realization.", prior=True),
]


# The baseline's text


def test_overlap_is_kept_once():
    assert baseline.strip_overlap("one two three", "two three four") == " four"
    assert baseline.strip_overlap("abc", "xyz") == "xyz"


def test_section_text_rebuilds_reading_order_without_repeats():
    text = baseline.section_text(STORE_CHUNKS[:2])
    assert text.count("Volume was lower.") == 1
    assert text.startswith("Gross margin decreased") and text.endswith("North America.")


def test_filing_sections_reads_only_the_baseline_sections():
    chunks = [*STORE_CHUNKS[:3], chunk("n-1", "A note.", section=Section.FINANCIAL_NOTES)]
    sections = baseline.filing_sections(chunks)
    assert set(sections) == {"mdna", "market_risk"}


def test_prompt_is_json_with_both_filings_and_the_metrics(comparison):
    table = [{"metric": "gross_margin", "current": "33.5%", "prior": "47.3%"}]
    prompt = json.loads(
        baseline.build_prompt(comparison, STORE_CHUNKS[:3], STORE_CHUNKS[3:], table)
    )
    assert prompt["current"]["filing"] == "10-K for FY2025"
    assert "cocoa" in prompt["current"]["sections"]["mdna"]
    assert "price realization" in prompt["prior"]["sections"]["mdna"]
    assert prompt["metrics"] == table


def test_brief_holds_at_most_the_agents_change_cap():
    change = {"description": "x", "kind": "numeric", "metric": None, "claims": []}
    with pytest.raises(ValueError):
        baseline.BaselineBrief.model_validate({"changes": [change] * (baseline.MAX_CHANGES + 1)})


# What a claim asserts


@pytest.mark.parametrize(
    ("label", "text", "asserted"),
    [
        ("confirmed", "Cocoa costs rose.", True),
        ("stated", "Cocoa costs rose.", True),
        ("plausible", "Margin fell while cocoa costs rose.", False),
        ("inferred", "Margin fell while cocoa costs rose.", False),
        # Causal wording asserts a cause whatever the label says.
        ("inferred", "Margin fell due to cocoa costs.", True),
        ("plausible", "Margin was driven by cocoa costs.", True),
    ],
)
def test_asserted_by_label_or_wording(label, text, asserted):
    assert score.asserts(label, text)[0] is asserted


def test_a_removed_item_is_about_the_prior_period(comparison):
    assert score.periods_for("removed", comparison) == {PRIOR}
    assert score.periods_for("numeric", comparison) == {CURRENT}


# Resolving a baseline quote


@pytest.mark.parametrize(
    ("quote", "side", "section", "expected", "chunk_id"),
    [
        ("due to higher cocoa costs", "current", Section.MDNA, "found", "c-1"),
        ("Cocoa futures are volatile.", "current", Section.MDNA, "wrong_section", "c-3"),
        ("due to price realization", "current", Section.MDNA, "other_filing", "p-1"),
        # Across the boundary between c-1 and c-2: verbatim, but in no single chunk.
        (
            "cocoa costs. Volume was lower. Prices rose 5%",
            "current",
            Section.MDNA,
            "spans_chunks",
            None,
        ),
        ("due to lower cocoa costs", "current", Section.MDNA, "not_found", None),
    ],
)
def test_resolve_quote(comparison, quote, side, section, expected, chunk_id):
    filings = score.Filings(FakeStore(STORE_CHUNKS))
    resolution, found = score.resolve_quote(quote, side, section, comparison, filings)
    assert resolution.value == expected
    if expected == "spans_chunks":
        # Placed in the chunk where it starts, for the period and metadata checks.
        assert found is not None and found.chunk_id == "c-1"
    else:
        assert (found.chunk_id if found else None) == chunk_id


def _baseline_record(*quotes: dict, support: str = "stated", kind: str = "numeric") -> dict:
    return {
        "brief": {
            "changes": [
                {
                    "description": "Gross margin fell",
                    "kind": kind,
                    "metric": "gross_margin",
                    "claims": [
                        {
                            "text": "Higher cocoa costs lowered gross margin.",
                            "support": support,
                            "quotes": list(quotes),
                            "metrics": [],
                        }
                    ],
                }
            ]
        }
    }


class NoMetrics:
    def compare(self, comparison, name):
        return None

    def get_metric(self, metric_id):
        return None


def _score_baseline(comparison, record) -> score.ScoredClaim:
    store = FakeStore(STORE_CHUNKS)
    (claim,) = score.baseline_claims(record, comparison, store, NoMetrics(), score.Filings(store))
    return claim


def test_a_found_quote_is_a_valid_citation(comparison):
    claim = _score_baseline(
        comparison,
        _baseline_record({"filing": "current", "section": "mdna", "text": "due to higher cocoa"}),
    )
    (citation,) = claim.citations
    assert citation.valid and citation.resolution == "found"
    assert claim.asserted and claim.label == "stated"


def test_a_prior_quote_for_a_current_move_is_wrong_period(comparison):
    claim = _score_baseline(
        comparison,
        _baseline_record(
            {"filing": "prior", "section": "mdna", "text": "due to price realization"}
        ),
    )
    (citation,) = claim.citations
    assert citation.verbatim and citation.metadata_ok and citation.wrong_period
    assert not citation.valid


def test_a_quote_given_the_wrong_filing_fails_metadata_and_period(comparison):
    claim = _score_baseline(
        comparison,
        _baseline_record(
            {"filing": "current", "section": "mdna", "text": "due to price realization"}
        ),
    )
    (citation,) = claim.citations
    assert citation.resolution == "other_filing"
    assert not citation.metadata_ok and citation.wrong_period


def test_a_quote_across_chunks_counts_as_verbatim(comparison):
    claim = _score_baseline(
        comparison,
        _baseline_record(
            {
                "filing": "current",
                "section": "mdna",
                "text": "cocoa costs. Volume was lower. Prices rose 5%",
            }
        ),
    )
    (citation,) = claim.citations
    assert citation.resolution == "spans_chunks" and citation.verbatim


def test_an_invented_quote_is_not_verbatim(comparison):
    claim = _score_baseline(
        comparison,
        _baseline_record({"filing": "current", "section": "mdna", "text": "cocoa fell 12%"}),
    )
    (citation,) = claim.citations
    assert citation.resolution == "not_found" and not citation.verbatim


def test_an_untraced_figure_is_reported(comparison):
    record = _baseline_record(
        {"filing": "current", "section": "mdna", "text": "due to higher cocoa costs"}
    )
    record["brief"]["changes"][0]["claims"][0]["text"] = "Cocoa costs rose 40%, cutting margin."
    claim = _score_baseline(comparison, record)
    assert "untraced_figure" in claim.figure_failures


def test_seq_10q_asserted_cause_is_a_base_period_mismatch(make_filing):
    q2 = make_filing(FormType.FORM_10Q, 2026, "Q2", date(2026, 6, 30))
    q1 = make_filing(FormType.FORM_10Q, 2026, "Q1", date(2026, 3, 31))
    seq = Comparison(
        comparison_id="HSY:seq-10q",
        ticker="HSY",
        comparison_type=ComparisonType.SEQ_10Q,
        current=q2,
        prior=q1,
    )
    store = FakeStore([])
    stated = _baseline_record(support="stated")
    hedged = _baseline_record(support="inferred")
    hedged["brief"]["changes"][0]["claims"][0]["text"] = "Margin fell as cocoa costs rose."
    (a,) = score.baseline_claims(stated, seq, store, NoMetrics(), score.Filings(store))
    (b,) = score.baseline_claims(hedged, seq, store, NoMetrics(), score.Filings(store))
    assert a.base_period_mismatch and not b.base_period_mismatch


# The judge step


class FakeJudge:
    def __init__(self, verdicts: list[EntailmentVerdict]) -> None:
        self.verdicts = verdicts
        self.calls = 0

    async def judge(self, claim: str, quote: str):
        verdict = self.verdicts[self.calls]
        self.calls += 1
        return type("Judgment", (), {"verdict": verdict, "reason": "r"})()


def _claim(**fields: Any) -> score.ScoredClaim:
    defaults: dict[str, Any] = {
        "system": "baseline",
        "comparison_id": "HSY:yoy-10k",
        "comparison_type": "yoy-10k",
        "change_key": "k1",
        "change": "Gross margin fell",
        "change_kind": "numeric",
        "metric": "gross_margin",
        "claim_id": "c1",
        "text": "Higher cocoa costs lowered gross margin.",
        "label": "stated",
        "causal_phrases": [],
        "asserted": True,
        "citations": [],
        "metric_ids": [],
        "figure_failures": [],
        "base_period_mismatch": False,
        "says_year_over_year": False,
    }
    return score.ScoredClaim(**{**defaults, **fields})


def _citation(**fields: Any) -> score.ScoredCitation:
    defaults: dict[str, Any] = {
        "quote": "q",
        "chunk_id": "c-1",
        "side": "current",
        "section": "mdna",
        "resolution": "found",
        "failures": [],
        "verbatim": True,
        "metadata_ok": True,
        "wrong_period": False,
    }
    return score.ScoredCitation(**{**defaults, **fields})


async def test_judge_stops_at_the_first_quote_that_states_the_cause():
    claim = _claim(citations=[_citation(), _citation(), _citation()])
    judge = FakeJudge([EntailmentVerdict.IMPLIES, EntailmentVerdict.STATES_CAUSE])
    await score.judge_claims([claim], judge)
    assert claim.supported is True and judge.calls == 2


async def test_unsupported_reasons():
    no_citation = _claim()
    invalid_only = _claim(citations=[_citation(wrong_period=True)])
    implies = _claim(citations=[_citation()])
    hedged = _claim(asserted=False, citations=[_citation()])
    judge = FakeJudge([EntailmentVerdict.IMPLIES])
    await score.judge_claims([no_citation, invalid_only, implies, hedged], judge)
    assert no_citation.unsupported_reason == "no citation"
    assert invalid_only.unsupported_reason == "no valid citation"
    assert implies.unsupported_reason == "no quote states the cause"
    assert hedged.supported is None and judge.calls == 1


async def test_wrong_period_support_needs_a_cause_resting_on_another_period():
    # Rests on last year's text: its only right-period quote does not state the cause.
    rests = _claim(citations=[_citation(wrong_period=True), _citation()])
    # Cites the prior filing for the earlier value, but a current quote states the cause.
    context = _claim(citations=[_citation(wrong_period=True), _citation()])
    # A hedged comparison that quotes the prior filing is correct use.
    hedged = _claim(asserted=False, citations=[_citation(wrong_period=True)])
    only_prior = _claim(citations=[_citation(wrong_period=True)])
    judge = FakeJudge([EntailmentVerdict.IMPLIES, EntailmentVerdict.STATES_CAUSE])
    await score.judge_claims([rests, context, hedged, only_prior], judge)
    assert rests.wrong_period_support is True
    assert context.wrong_period_support is False
    assert hedged.wrong_period_support is False
    assert only_prior.wrong_period_support is True


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("The filing says the year-over-year decline was driven by costs.", True),
        ("Segment income fell 5.9% year over year on higher costs.", True),
        ("Volume declined about 10%, driven primarily by price elasticity.", False),
        ("Higher Q2 net interest expense was due to short-term debt.", False),
    ],
)
def test_says_year_over_year(text, says):
    assert score.says_year_over_year(text) is says


async def test_free_run_settles_only_claims_without_a_valid_citation():
    settled, open_ = _claim(), _claim(citations=[_citation()])
    await score.judge_claims([settled, open_], None)
    assert settled.supported is False and open_.supported is None


# The cause matcher's bookkeeping


def _cause(metric="gross_margin"):
    return score.StatedCause.model_validate(
        {
            "item_id": "HSY-S01",
            "comparison_id": "HSY:yoy-10k",
            "section": "mdna",
            "change": "Gross margin fell",
            "metric": metric,
            "direction": "down",
            "causes": ["higher commodity costs", "lower volume"],
            "quote": {"side": "current", "text": "x"},
            "distractors": ["higher freight"],
        }
    )


def test_match_levels_needs_one_rating_per_item():
    cause = _cause()
    good = score.Match(
        about_the_change=True,
        ratings=[
            score.Rating(id="c1", level="names"),
            score.Rating(id="C2", level="no"),
            score.Rating(id="D1", level="broader"),
            score.Rating(id="D9", level="names"),  # not listed: ignored
        ],
        reason="",
    )
    assert score.match_levels(good, cause) == (["names", "no"], ["broader"])
    missing = good.model_copy(update={"ratings": good.ratings[:2]})
    assert score.match_levels(missing, cause) is None


def test_candidates_are_the_labeled_metric_or_text_changes():
    cause = _cause()
    same = _claim(claim_id="a")
    other_metric = _claim(claim_id="b", metric="sga")
    text = _claim(claim_id="c", metric=None, change_kind="modified")
    other_comparison = _claim(claim_id="d", comparison_id="HSY:yoy-10q")
    found = score.candidates_for(cause, [same, other_metric, text, other_comparison])
    assert [c.claim_id for c in found] == ["a", "c"]


def test_summary_counts_distractors_by_whether_the_claim_asserts():
    cause = _cause()
    asserted = _claim(
        claim_id="a",
        citations=[_citation()],
        supported=True,
        cause_matches=[
            {"cause_id": "HSY-S01", "causes": ["names", "no"], "distractors": ["names"]}
        ],
    )
    hedged = _claim(
        claim_id="b",
        asserted=False,
        label="inferred",
        cause_matches=[
            {"cause_id": "HSY-S01", "causes": ["no", "names"], "distractors": ["names"]}
        ],
    )
    numbers = score.summarize([asserted, hedged], [cause], [])["baseline"]
    assert numbers["labeled causes named"] == "2/2 (100%)"
    assert numbers["distractors asserted as fact"] == 1
    assert numbers["distractors in hedged claims"] == 1
    assert numbers["unsupported asserted claims"] == "0/1 (0%)"


# The hand check

import hand_check  # noqa: E402


def _scored(system: str, n: int, **fields: Any) -> dict:
    claim = {
        "system": system,
        "claim_id": f"{system}-{n}",
        "comparison_id": "HSY:yoy-10k",
        "change": "Gross margin fell",
        "text": f"claim {n}",
        "asserted": True,
        "supported": n % 2 == 0,
        "citations": [
            {
                "side": "current",
                "section": "mdna",
                "quote": "ok",
                "verbatim": True,
                "metadata_ok": True,
                "wrong_period": False,
            },
            {
                "side": "prior",
                "section": "mdna",
                "quote": "old",
                "verbatim": True,
                "metadata_ok": True,
                "wrong_period": True,
            },
        ],
    }
    return {**claim, **fields}


def test_hand_check_sample_is_blind_and_repeatable():
    claims = [_scored(s, n) for s in ("agent", "baseline") for n in range(20)]
    claims.append(_scored("agent", 99, asserted=False))
    claims.append(_scored("agent", 98, supported=None))
    sheet, key = hand_check.sample({"claims": claims}, 15, hand_check.SEED)
    again, _ = hand_check.sample({"claims": list(reversed(claims))}, 15, hand_check.SEED)
    assert sheet == again
    assert len(sheet) == 30
    assert sum(entry["system"] == "agent" for entry in key.values()) == 15
    assert not {"agent-99", "agent-98"} & {entry["claim_id"] for entry in key.values()}
    for row in sheet:
        assert set(row) == {"id", "comparison", "change", "claim", "quotes"}
        # Only valid quotes are shown.
        assert [quote["quote"] for quote in row["quotes"]] == ["ok"]


def test_hand_check_shows_the_agent_rationale_not_its_heading():
    heading = "Changed paragraph in Management's Discussion and Analysis"
    claims = [
        _scored("agent", 0, change=heading, coverage_text=f"{heading}: Tariffs turned material."),
        _scored("agent", 2, change=heading, coverage_text=None),
        _scored("baseline", 0, change="Tariffs turned material."),
    ]
    sheet, _ = hand_check.sample({"claims": claims}, 15, hand_check.SEED)
    assert sorted(row["change"] for row in sheet) == sorted(
        ["Tariffs turned material.", heading, "Tariffs turned material."]
    )


def test_hand_check_tally():
    key = {
        "H01": {"system": "agent", "claim_id": "a", "judge_supported": True},
        "H02": {"system": "agent", "claim_id": "b", "judge_supported": True},
        "H03": {"system": "baseline", "claim_id": "c", "judge_supported": False},
    }
    result = hand_check.tally(key, {"H01": "yes", "H02": "no", "H03": "yes"})
    assert result["agent"]["user and judge agree"] == 1
    assert result["agent"]["judge supported, user did not"] == 1
    assert result["baseline"]["user supported, judge did not"] == 1
    with pytest.raises(SystemExit):
        hand_check.tally(key, {"H01": "yes"})


# Whether the baseline's changes are real

import grounding  # noqa: E402

from sec_agent.schemas import Change, ChangeKind  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Gross margin collapsed from 47.3% to 33.5% as cost of sales rose 31.7%.", -1),
        ("Net sales slipped 0.6% as sales adjustments jumped.", -1),
        ("Segment margin improved from 6.9% to 10.9%.", 1),
        ("Net income swung to a loss, with net margin falling to -1.6%.", -1),
        ("Inventory stood at $1.2 billion.", 0),
    ],
)
def test_grounding_reads_the_first_direction_word(text, expected):
    assert grounding.direction(text) == expected


def test_a_numeric_change_is_grounded_when_its_metric_moved_the_stated_way():
    assert grounding.numeric_verdict("Gross margin fell 2.1 points.", -0.021) == grounding.GROUNDED
    assert grounding.numeric_verdict("Gross margin rose 2.1 points.", -0.021) == grounding.DISAGREES
    assert grounding.numeric_verdict("Gross margin was 48.7%.", -0.021) == grounding.NO_DIRECTION
    assert grounding.numeric_verdict("Segment loss widened.", None) == grounding.NO_VALUES


def test_a_text_change_is_grounded_only_by_a_quote_in_a_changed_passage():
    changed = Change(
        change_id="HSY:yoy-10k:modified:x",
        comparison_id="HSY:yoy-10k",
        kind=ChangeKind.MODIFIED,
        section=Section.MDNA,
        current_text="Tariffs had a material negative impact on our results of operations.",
        prior_text="Tariffs had no material impact on our results of operations.",
    )
    quote = {"filing": "current", "text": "a material negative impact on our results"}
    assert grounding.text_verdict([quote], [changed]) == (grounding.GROUNDED, ["modified"])
    elsewhere = {"filing": "current", "text": "Net sales rose on price realization."}
    assert grounding.text_verdict([elsewhere], [changed]) == (grounding.NOT_IN_CHANGE, [])
    # The right words from the other filing do not count.
    wrong_side = {"filing": "prior", "text": "a material negative impact on our results"}
    assert grounding.text_verdict([wrong_side], [changed])[0] == grounding.NOT_IN_CHANGE
