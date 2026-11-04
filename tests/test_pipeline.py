"""The route every filing takes, and the guarantee that rejected text can't be read."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from sec_agent import cli
from sec_agent.edgar.corpus import Corpus, CorpusCompany, CorpusFiling
from sec_agent.ingest import pipeline
from sec_agent.ingest.pipeline import (
    PARSER_VERSION,
    FilingRejectedError,
    ParserName,
    load_validated_filing,
    parse_filing,
    parsed_filing_path,
    read_parsed_filing,
    save_parsed_filing,
)
from sec_agent.ingest.sections import ParsedSection, SectionStatus
from sec_agent.schemas import FormType, Section

Q = FormType.FORM_10Q


def sentences(label: str, count: int) -> str:
    """`count` paragraphs of prose, each long enough to count and containing the words the
    audit expects to find (results of operations, liquidity, risks, basis of presentation)."""
    return "\n\n".join(
        f"Paragraph {n} of the {label} explains results of operations, liquidity, risks and "
        "basis of presentation, along with what the company reviewed during the period."
        for n in range(count)
    )


def good_items() -> dict[str, str]:
    """edgartools-shaped output for a 10-Q that passes every check. The notes and MD&A are
    long enough to clear the 8,000 and 10,000-character minimums for a 10-Q."""
    return {
        "part_i_item_1": "Item 1. Financial Statements\n\n" + sentences("notes", 60),
        "part_i_item_2": "Item 2. Management's Discussion and Analysis\n\n" + sentences("mdna", 80),
        "part_i_item_3": "Item 3. Market Risk\n\nRefer to Item 2 for this information.",
        "part_ii_item_1a": "Item 1A. Risk Factors\n\nThere are no material changes in our risks.",
    }


def bad_items() -> dict[str, str]:
    """Output that fails: an MD&A of two words, and every other required section missing."""
    return {"part_i_item_2": "Too short."}


def filing_html(items: dict[str, str] | None = None, *, extra: str = "") -> str:
    """HTML containing the items' text as paragraphs, which the fallback can also read."""
    paragraphs = []
    for text in (items or good_items()).values():
        paragraphs += [f"<p>{block}</p>" for block in text.split("\n\n")]
    return f"<html><body>{''.join(paragraphs)}{extra}</body></html>"


def use_edgartools_output(monkeypatch, output) -> None:
    """Replace edgartools with a fixed answer, or with an exception to simulate a crash."""

    def render(stripped_html, form):
        if isinstance(output, Exception):
            raise output
        return output

    monkeypatch.setattr(pipeline, "render_items", render)


def test_filing_that_validates_uses_edgartools(monkeypatch):
    use_edgartools_output(monkeypatch, good_items())
    parsed = parse_filing(filing_html(), Q, "acc-1")
    assert parsed.passed
    assert parsed.parser is ParserName.EDGARTOOLS
    assert parsed.primary_problems == ()


def test_filing_edgartools_misreads_goes_to_the_fallback(monkeypatch):
    use_edgartools_output(monkeypatch, bad_items())
    parsed = parse_filing(filing_html(), Q, "acc-1")
    assert parsed.passed
    assert parsed.parser is ParserName.FALLBACK
    assert parsed.primary_problems


def test_edgartools_crash_goes_to_the_fallback_and_is_recorded(monkeypatch):
    use_edgartools_output(monkeypatch, ValueError("unreadable document"))
    parsed = parse_filing(filing_html(), Q, "acc-1")
    assert parsed.passed
    assert parsed.parser is ParserName.FALLBACK
    assert "ValueError" in parsed.primary_problems[0]


def test_filing_neither_parser_can_read_is_rejected(monkeypatch):
    use_edgartools_output(monkeypatch, bad_items())
    parsed = parse_filing("<html><body><p>A letter to shareholders.</p></body></html>", Q, "acc-1")
    assert not parsed.passed


def attempt(monkeypatch, *, problems: tuple[str, ...], coverage: float, drop_notes: bool = False):
    """A parse record with chosen problems and coverage, to test which failed parse is kept."""
    use_edgartools_output(monkeypatch, good_items())
    parsed = parse_filing(filing_html(), Q, "acc-1")
    sections = dict(parsed.sections)
    if drop_notes:
        sections[Section.FINANCIAL_NOTES] = ParsedSection(
            section=Section.FINANCIAL_NOTES, status=SectionStatus.MISSING
        )
    return parsed.model_copy(
        update={"problems": problems, "coverage": coverage, "sections": sections}
    )


def test_failed_parse_with_more_content_beats_one_with_fewer_problems(monkeypatch):
    """Cosmos Health: the fallback tied on problems, having lost all of its notes."""
    kept = attempt(monkeypatch, problems=("a", "b", "c"), coverage=1.0)
    lost = attempt(monkeypatch, problems=("a",), coverage=0.36, drop_notes=True)
    assert pipeline._is_better(kept, lost)
    assert not pipeline._is_better(lost, kept)


def test_with_equal_content_better_coverage_wins(monkeypatch):
    # The tie-break order is content, then coverage, then problem count.
    high = attempt(monkeypatch, problems=("a", "b"), coverage=0.99)
    low = attempt(monkeypatch, problems=("a",), coverage=0.5)
    assert pipeline._is_better(high, low)


def test_with_equal_content_and_coverage_fewer_problems_wins(monkeypatch):
    fewer = attempt(monkeypatch, problems=("a",), coverage=1.0)
    more = attempt(monkeypatch, problems=("a", "b"), coverage=1.0)
    assert pipeline._is_better(fewer, more)


def test_exact_tie_keeps_the_current_parse(monkeypatch):
    # The challenger must be strictly better to replace the parse already kept.
    first = attempt(monkeypatch, problems=("a",), coverage=1.0)
    second = attempt(monkeypatch, problems=("b",), coverage=1.0)
    assert not pipeline._is_better(second, first)


def test_wrongly_removed_table_rejects_the_filing_under_both_parsers(monkeypatch):
    """Table problems come from stripping, so changing parser doesn't hide them."""
    use_edgartools_output(monkeypatch, good_items())

    # A deliberately broken table stripper deletes a prose sentence. The audit measures it
    # against the original HTML, so it must notice, whichever parser read the items.

    def broken_strip(html):
        result = real_strip(html)
        return result.model_copy(update={"html": html.replace(SENTENCE, "")})

    real_strip = pipeline.strip_data_tables
    monkeypatch.setattr(pipeline, "strip_data_tables", broken_strip)
    table = (
        f"<table><tr><td>Sales</td><td>1</td><td>2</td></tr><tr><td>{SENTENCE}</td></tr></table>"
    )
    parsed = parse_filing(filing_html(extra=table), Q, "acc-1")
    assert not parsed.passed
    assert any("wrongly removed" in problem for problem in parsed.problems)


# A prose sentence of more than 15 words, so the audit counts it if it goes missing.
SENTENCE = (
    "The company recorded a charge in the quarter because a major supplier closed two plants "
    "and moved its remaining production to another region."
)


def test_removed_tables_are_kept_in_the_record(monkeypatch):
    """A removed table's figures may not be in XBRL, so they stay recoverable."""
    use_edgartools_output(monkeypatch, good_items())
    table = "<table><tr><td>Barrels per day</td><td>100</td><td>200</td></tr></table>"
    parsed = parse_filing(filing_html(extra=table), Q, "acc-1")
    assert parsed.removed_tables[0].rows == (("Barrels per day", "100", "200"),)


def test_section_text_reaches_the_record(monkeypatch):
    # Paragraph ids are "<accession>:<section>:<number>", which later citations point to.
    use_edgartools_output(monkeypatch, good_items())
    parsed = parse_filing(filing_html(), Q, "acc-1")
    assert parsed.sections[Section.MDNA].paragraphs[0].paragraph_id == "acc-1:mdna:0000"


# --- Reading records back ---


def test_passed_filing_round_trips(monkeypatch, tmp_path):
    use_edgartools_output(monkeypatch, good_items())
    parsed = parse_filing(filing_html(), Q, "acc-1")
    path = parsed_filing_path(tmp_path, "HSY", "acc-1")
    save_parsed_filing(parsed, path)
    assert load_validated_filing(path) == parsed


def test_rejected_filing_cannot_be_loaded(monkeypatch, tmp_path):
    # The guarantee the whole route exists for: later stages can't read rejected text.
    use_edgartools_output(monkeypatch, bad_items())
    parsed = parse_filing("<html><body><p>Nothing here.</p></body></html>", Q, "acc-1")
    path = tmp_path / "acc-1.json"
    save_parsed_filing(parsed, path)
    with pytest.raises(FilingRejectedError, match="failed validation"):
        load_validated_filing(path)


def test_record_from_an_older_parser_cannot_be_loaded(monkeypatch, tmp_path):
    use_edgartools_output(monkeypatch, good_items())
    parsed = parse_filing(filing_html(), Q, "acc-1")
    path = tmp_path / "acc-1.json"
    save_parsed_filing(parsed.model_copy(update={"parser_version": PARSER_VERSION - 1}), path)
    assert read_parsed_filing(path) is None
    with pytest.raises(FilingRejectedError, match="older parser version"):
        load_validated_filing(path)


def test_missing_record_cannot_be_loaded(tmp_path):
    with pytest.raises(FilingRejectedError, match="missing"):
        load_validated_filing(tmp_path / "absent.json")


# --- `sec-agent ingest` runs the gate ---


@pytest.fixture
def downloaded(monkeypatch, settings, tmp_path):
    """A one-filing corpus with its files already present, so nothing touches the network."""
    filing = CorpusFiling(
        form=Q,
        fiscal_year=2026,
        fiscal_period="Q2",
        period_end="2026-06-30",
        filed_date="2026-08-01",
        accession="acc-1",
        primary_document="doc.htm",
    )
    corpus = Corpus(
        companies=(CorpusCompany(ticker="AAA", company="A", cik="0000000001", filings=(filing,)),)
    )
    corpus_path = tmp_path / "corpus.yaml"
    corpus.save(corpus_path)
    document = settings.raw_filings_dir / "AAA" / "acc-1" / "doc.htm"
    document.parent.mkdir(parents=True)
    document.write_text(filing_html(), encoding="utf-8")
    facts = settings.raw_filings_dir / "AAA" / "companyfacts.json"
    facts.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    record = parsed_filing_path(settings.parsed_filings_dir, "AAA", "acc-1")
    return corpus_path, record


def test_ingest_validates_and_saves_filings(monkeypatch, downloaded):
    corpus_path, record = downloaded
    use_edgartools_output(monkeypatch, good_items())
    result = CliRunner().invoke(cli.app, ["ingest", "--corpus", str(corpus_path)])
    assert result.exit_code == 0, result.output
    assert "1 passed, 0 rejected" in result.output
    assert load_validated_filing(record).passed


def test_ingest_fails_loudly_on_a_rejected_filing(monkeypatch, downloaded, settings):
    corpus_path, record = downloaded
    document = settings.raw_filings_dir / "AAA" / "acc-1" / "doc.htm"
    document.write_text("<html><body><p>Nothing here.</p></body></html>", encoding="utf-8")
    use_edgartools_output(monkeypatch, bad_items())
    result = CliRunner().invoke(cli.app, ["ingest", "--corpus", str(corpus_path)])
    assert result.exit_code == 1
    assert "REJECTED AAA" in result.output
    assert "0 passed, 1 rejected" in result.output
    with pytest.raises(FilingRejectedError):
        load_validated_filing(record)


def test_ingest_does_not_reparse_a_current_record(monkeypatch, downloaded):
    corpus_path, record = downloaded
    use_edgartools_output(monkeypatch, good_items())
    CliRunner().invoke(cli.app, ["ingest", "--corpus", str(corpus_path)])
    # From here, any attempt to parse again fails the test.
    use_edgartools_output(monkeypatch, AssertionError("should not parse again"))
    monkeypatch.setattr(cli, "parse_filing", _fail_if_called)
    result = CliRunner().invoke(cli.app, ["ingest", "--corpus", str(corpus_path)])
    assert result.exit_code == 0, result.output


def _fail_if_called(*args, **kwargs):
    # Stands in for parse_filing in the test above.
    raise AssertionError("should not parse again")
