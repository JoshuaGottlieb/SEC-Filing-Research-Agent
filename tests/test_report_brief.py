"""The brief writer (T5.4): layout, references, the rules it enforces, and the audit trail."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import pytest

from sec_agent.agents.verification import find_causal_language
from sec_agent.report.brief import (
    NO_FINDINGS_SUMMARY,
    REF,
    WRAP_WIDTH,
    BriefError,
    assign_refs,
    audit_trail,
    check_brief,
    escape,
    format_summary_line,
    render_markdown,
    wrap_markdown,
    write_run,
)
from sec_agent.report.names import (
    CHANGE_KIND_NAMES,
    COMPARISON_NAMES,
    LABEL_NAMES,
    METRIC_NAMES,
    SECTION_NAMES,
    metric_name,
)
from sec_agent.schemas import (
    DISCLAIMER,
    Brief,
    Change,
    ChangeKind,
    Citation,
    Claim,
    Comparison,
    ComparisonType,
    Duration,
    EntailmentVerdict,
    ExplanationLabel,
    Filing,
    Finding,
    FiscalPeriod,
    FormType,
    MetricName,
    MetricValue,
    NodeStatus,
    Section,
    TreeNode,
)

CURRENT = FiscalPeriod(fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31))
PRIOR = FiscalPeriod(fiscal_year=2024, fiscal_period="FY", period_end=date(2024, 12, 31))


def filing(period: FiscalPeriod, accession: str) -> Filing:
    return Filing(
        ticker="HSY",
        company="The Hershey Company",
        cik="0000047111",
        form=FormType.FORM_10K,
        period=period,
        accession=accession,
        filed_date=date(period.fiscal_year + 1, 2, 15),
        url=f"https://www.sec.gov/{accession}.htm",
    )


COMPARISON = Comparison(
    comparison_id="HSY-yoy",
    ticker="HSY",
    comparison_type=ComparisonType.YOY_10K,
    current=filing(CURRENT, "acc-2025"),
    prior=filing(PRIOR, "acc-2024"),
)


def cite(quote: str, section: Section = Section.MDNA, period: FiscalPeriod = CURRENT) -> Citation:
    return Citation(
        chunk_id=f"chunk-{abs(hash(quote)) % 1000}",
        accession="acc-2025",
        ticker="HSY",
        form=FormType.FORM_10K,
        period=period,
        section=section,
        quote=quote,
        url="https://www.sec.gov/acc-2025.htm",
    )


def confirmed(claim_id: str, text: str, quote: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        label=ExplanationLabel.CONFIRMED,
        citations=(cite(quote),),
        entailment=EntailmentVerdict.STATES_CAUSE,
    )


def plausible(claim_id: str, text: str, quote: str = "", metric_ids: tuple[str, ...] = ()):
    return Claim(
        claim_id=claim_id,
        text=text,
        label=ExplanationLabel.PLAUSIBLE,
        citations=(cite(quote),) if quote else (),
        metric_ids=metric_ids,
    )


MARGIN = Change(
    change_id="chg-gm", comparison_id="HSY-yoy", kind=ChangeKind.NUMERIC, metric_id="gm-2025"
)
RISK = Change(
    change_id="chg-risk",
    comparison_id="HSY-yoy",
    kind=ChangeKind.ADDED,
    section=Section.RISK_FACTORS,
    current_text="Tariffs on imported cocoa could raise our costs.",
)
REMOVED = Change(
    change_id="chg-gone",
    comparison_id="HSY-yoy",
    kind=ChangeKind.REMOVED,
    section=Section.BUSINESS,
    prior_text="We operate a retail store in Hershey, Pennsylvania.",
)

# The ToT design's example: input costs confirmed, pricing pressure plausible.
MARGIN_FINDING = Finding(
    finding_id="run1:chg-gm",
    change=MARGIN,
    tree_root_id="run1:chg-gm:root",
    claims=(
        plausible(
            "run1:chg-gm:r1:c2",
            "Promotions rose and gross margin fell; the filings do not link the two.",
            "Competitors increased promotional activity during 2025.",
        ),
        confirmed(
            "run1:chg-gm:r1:c1",
            "Higher input costs lowered gross margin, driven by cocoa.",
            "Gross margin decreased due to higher input costs, primarily cocoa.",
        ),
    ),
)
RISK_FINDING = Finding(
    finding_id="run1:chg-risk",
    change=RISK,
    tree_root_id="run1:chg-risk:root",
    claims=(
        plausible(
            "run1:chg-risk:r1:c1",
            "A tariff risk was added; the filings do not link it to a change in costs.",
            "Tariffs on imported cocoa could raise our costs.",
        ),
    ),
)
GONE_FINDING = Finding(
    finding_id="run1:chg-gone",
    change=REMOVED,
    tree_root_id="run1:chg-gone:root",
    review_reason="every candidate was pruned",
)
FINDINGS = (RISK_FINDING, GONE_FINDING, MARGIN_FINDING)  # materiality order


def metric(metric_id: str, period: FiscalPeriod, value: float, name: str = "gross_margin"):
    return MetricValue(
        metric_id=metric_id,
        ticker="HSY",
        name=name,
        period=period,
        duration=Duration.TWELVE_MONTHS,
        value=value,
        unit="ratio",
        input_fact_ids=("f1",),
    )


class FakeMetrics:
    def __init__(self, *values: MetricValue) -> None:
        self._values = {value.metric_id: value for value in values}

    def get_metric(self, metric_id: str) -> MetricValue | None:
        return self._values.get(metric_id)


METRICS = FakeMetrics(metric("gm-2025", CURRENT, 0.335), metric("gm-2024", PRIOR, 0.473))


def summary_for(findings=FINDINGS) -> str:
    refs = assign_refs(findings)
    return "\n".join(
        [
            format_summary_line(
                "Higher input costs lowered gross margin.", [refs.claims["run1:chg-gm:r1:c1"]]
            ),
            format_summary_line(
                "A tariff risk was added to the risk factors.",
                [refs.claims["run1:chg-risk:r1:c1"]],
            ),
            format_summary_line(
                "A removed business paragraph needs review.", [refs.findings["run1:chg-gone"]]
            ),
        ]
    )


def brief(findings=FINDINGS, summary: str | None = None, **fields: Any) -> Brief:
    return Brief(
        run_id="run1",
        comparison=COMPARISON,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        summary=summary if summary is not None else summary_for(findings),
        findings=findings,
        **{"cost_usd": 0.42, **fields},
    )


def contains(markdown: str, text: str) -> bool:
    """Whether `text` appears in the brief, ignoring how it was wrapped and quoted."""
    words = [word for word in markdown.split() if word != ">"]
    return " ".join(text.split()) in " ".join(words)


def statements(markdown: str) -> list[str]:
    """What states something: summary, claim and review bullets, and finding headings.

    A statement is its first line plus the indented lines that continue it when it is wrapped.
    Block quotes are filing text, and the indented lines after a blank line under a claim are its
    evidence, so neither is part of a statement.
    """
    found: list[str] = []
    current: list[str] | None = None
    for line in markdown.splitlines():
        if line.startswith(("- ", "### ")):
            current = [line]
            found.append(line)
        elif current is not None and line.startswith("  ") and not line.lstrip().startswith(">"):
            current.append(line.strip())
            found[-1] = " ".join(current)
        else:
            current = None
    return found


def other_prose(markdown: str) -> list[str]:
    """Lines that are neither part of a statement, nor a block quote, nor blank."""
    lines: list[str] = []
    in_statement = False
    for line in markdown.splitlines():
        if line.startswith(("- ", "### ")):
            in_statement = True
        elif in_statement and line.startswith("  ") and not line.lstrip().startswith(">"):
            pass  # a wrapped statement continues
        else:
            in_statement = False
            if line.strip() and not line.lstrip().startswith(">"):
                lines.append(line)
    return lines


# The plan's done-when.
def test_every_statement_links_to_a_record_in_the_audit_trail():
    report = brief()
    markdown = render_markdown(report, METRICS)
    resolved = audit_trail(report)["refs"]
    lines = statements(markdown)
    assert len(lines) == 3 + 3 + 3 + 1  # summary, finding headings, claims, review items
    for line in lines:
        refs = REF.findall(line)
        assert refs, line
        assert all(ref in resolved for ref in refs), line


def test_the_references_resolve_to_the_full_ids():
    resolved = audit_trail(brief())["refs"]
    claim_ids = {claim.claim_id for finding in FINDINGS for claim in finding.claims}
    finding_ids = {finding.finding_id for finding in FINDINGS}
    assert set(resolved.values()) == claim_ids | finding_ids
    assert all(ref.startswith("C") for ref, value in resolved.items() if value in claim_ids)


def test_findings_are_grouped_by_strongest_label_in_materiality_order():
    markdown = render_markdown(brief(), METRICS)
    confirmed_at = markdown.index("## Changes with a confirmed explanation")
    plausible_at = markdown.index("## Changes with plausible explanations only")
    unresolved_at = markdown.index("## Changes without a supported explanation")
    review_at = markdown.index("## Needs review")
    assert confirmed_at < plausible_at < unresolved_at < review_at
    # The margin finding is third in materiality but has the only confirmed claim.
    assert (
        confirmed_at
        < markdown.index("### [F1] Gross margin change: 33.5% in FY2025")
        < plausible_at
    )
    assert plausible_at < markdown.index("### [F2] New paragraph in Risk Factors") < unresolved_at
    assert unresolved_at < markdown.index("### [F3] Removed paragraph in Business") < review_at


def test_a_findings_claims_show_confirmed_first_with_their_labels():
    markdown = render_markdown(brief(), METRICS)
    first = markdown.index("- **Confirmed** [C1]: Higher input costs lowered gross margin")
    second = markdown.index("- **Plausible** [C2]: Promotions rose and gross margin fell")
    assert first < second


def test_the_disclaimer_is_on_every_brief():
    for report in (brief(), brief(findings=(), summary=NO_FINDINGS_SUMMARY)):
        assert contains(render_markdown(report), escape(DISCLAIMER))
        assert "Not investment advice" in render_markdown(report)


def test_quotes_are_block_quotes_with_their_source():
    markdown = render_markdown(brief(), METRICS)
    assert "  > Gross margin decreased due to higher input costs, primarily cocoa." in markdown
    assert contains(
        markdown,
        "Source: 10-K for FY2025, Management's Discussion and Analysis, "
        "<https://www.sec.gov/acc-2025.htm>",
    )


def test_changed_text_is_quoted_under_its_period():
    markdown = render_markdown(brief(), METRICS)
    assert "After (FY2025):\n\n> Tariffs on imported cocoa could raise our costs." in markdown
    assert "Before (FY2024):\n\n> We operate a retail store in Hershey, Pennsylvania." in markdown


def test_metrics_show_their_name_value_and_period():
    claim = plausible("run1:chg-gm:r1:c3", "Gross margin fell.", metric_ids=("gm-2025", "gm-x"))
    finding = MARGIN_FINDING.model_copy(update={"claims": (claim,)})
    markdown = render_markdown(
        brief(findings=(finding,), summary="Gross margin fell. [C1]"), METRICS
    )
    assert "### [F1] Gross margin change: 33.5% in FY2025" in markdown
    # A metric the lookup does not know is shown by its id rather than dropped.
    assert "Metrics: Gross margin: 33.5% (FY2025); metric gm-x" in markdown


def test_a_finding_with_no_claims_says_so_and_is_in_review():
    markdown = render_markdown(brief(), METRICS)
    assert "No explanation of this change met the evidence standard." in markdown
    assert "- [F3] Removed paragraph in Business: every candidate was pruned" in markdown


def test_a_finding_with_claims_and_a_review_note_is_in_both_places():
    noted = RISK_FINDING.model_copy(update={"review_reason": "the judge failed on one claim"})
    findings = (noted, GONE_FINDING, MARGIN_FINDING)
    markdown = render_markdown(brief(findings=findings), METRICS)
    assert "### [F2] New paragraph in Risk Factors" in markdown
    assert "- [F2] New paragraph in Risk Factors: the judge failed on one claim" in markdown


def test_a_brief_with_nothing_to_review_says_so():
    report = brief(findings=(MARGIN_FINDING,), summary="Input costs. [C1]")
    assert "Nothing in this brief needs review." in render_markdown(report, METRICS)


def test_filing_text_cannot_change_the_markdown():
    hostile = (
        "See [C9](https://evil.example) and ![x](y)\n# Fake heading\n<script>alert(1)</script>"
    )
    risky = RISK.model_copy(update={"current_text": hostile})
    finding = RISK_FINDING.model_copy(update={"change": risky})
    markdown = render_markdown(brief(findings=(finding,), summary="A risk. [C1]"))
    # Escaped brackets: no link and no image can form.
    assert "\\[C9\\](https://evil.example)" in markdown
    assert "[C9](" not in markdown.replace("\\[C9\\](", "")
    assert "!\\[x\\](y)" in markdown
    assert "<script>" not in markdown
    assert not any(line.startswith("# Fake") for line in markdown.splitlines())
    # A reference inside filing text is escaped, so it cannot pose as one of ours.
    assert "C9" not in REF.findall(markdown)
    assert "> \\# Fake heading" in markdown


def test_escape_makes_markdown_inert():
    assert escape("a *b* [c](d) `e` <f> #g |h| ~i~ \\j") == (
        "a \\*b\\* \\[c\\](d) \\`e\\` \\<f\\> \\#g \\|h\\| \\~i\\~ \\\\j"
    )
    assert escape("one\ntwo   three") == "one two three"


def test_causal_wording_in_the_output_is_only_where_it_is_allowed():
    # Gap 13 over the whole brief: outside filing text, causal phrases appear only on confirmed
    # claim lines and on summary lines citing only confirmed claims.
    report = brief()
    markdown = render_markdown(report, METRICS)
    refs = assign_refs(report.findings)
    confirmed_refs = {
        refs.claims[claim.claim_id]
        for finding in report.findings
        for claim in finding.claims
        if claim.label is ExplanationLabel.CONFIRMED
    }
    for statement in statements(markdown):
        if find_causal_language(statement):
            cited = set(REF.findall(statement))
            assert cited and cited <= confirmed_refs, statement
    # Everything else the system wrote (titles, notes, "Metrics:" lines) states no cause.
    for line in other_prose(markdown):
        assert find_causal_language(line) == (), line


def test_a_plausible_claim_worded_causally_stops_the_brief(tmp_path):
    bad = plausible(
        "run1:chg-risk:r1:c1", "Costs rose due to tariffs.", "Tariffs could raise costs."
    )
    finding = RISK_FINDING.model_copy(update={"claims": (bad,)})
    report = brief(findings=(finding,), summary="A risk was added. [C1]")
    problems = check_brief(report)
    assert any("plausible but worded causally (due to)" in problem for problem in problems)
    with pytest.raises(BriefError) as raised:
        write_run(report, tmp_path / "out")
    assert raised.value.problems == problems
    assert not (tmp_path / "out").exists()


def test_a_confirmed_claim_may_be_worded_causally():
    assert check_brief(brief()) == ()


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Gross margin fell.", "cites no claim or finding"),
        ("Gross margin fell. [C9]", "cites unknown C9"),
        ("Margin fell due to promotions. [C2]", "uses causal wording (due to)"),
        ("Margin fell due to costs. [C1] [F1]", "uses causal wording (due to)"),
        ("Investors should buy the shares. [C1]", "reads as investment advice"),
        ("We see a price target of 120. [C1]", "price target"),
        ("Margin fell [C2] as costs rose. [C1]", "has a reference inside its text"),
    ],
)
def test_summary_lines_are_checked(line, expected):
    report = brief(findings=(MARGIN_FINDING,), summary=line)
    assert any(expected in problem for problem in check_brief(report))


def test_a_summary_line_citing_only_confirmed_claims_may_be_causal():
    report = brief(findings=(MARGIN_FINDING,), summary="Margin fell due to input costs. [C1]")
    assert check_brief(report) == ()


def test_an_empty_summary_stops_the_brief():
    assert "the summary is empty" in check_brief(brief(summary="  \n"))


def test_a_brief_with_no_findings_uses_the_fixed_summary(tmp_path):
    assert check_brief(brief(findings=(), summary=NO_FINDINGS_SUMMARY)) == ()
    assert check_brief(brief(findings=(), summary="All quiet.")) != ()
    files = write_run(brief(findings=(), summary=NO_FINDINGS_SUMMARY), tmp_path)
    assert "Nothing in this brief needs review." in files.brief.read_text()


def test_a_causal_review_reason_stops_the_brief():
    noted = GONE_FINDING.model_copy(update={"review_reason": "pruned because costs rose"})
    problems = check_brief(brief(findings=(RISK_FINDING, noted, MARGIN_FINDING)))
    assert any("review reason of run1:chg-gone uses causal wording" in p for p in problems)


def test_repeated_claim_ids_stop_the_brief():
    twin = RISK_FINDING.model_copy(update={"claims": (MARGIN_FINDING.claims[0].model_copy(),)})
    problems = check_brief(brief(findings=(twin, MARGIN_FINDING), summary="x [C1]"))
    assert problems == ("claim ids repeat: run1:chg-gm:r1:c2",)


def test_a_causal_rationale_is_left_out_and_recorded():
    explained = RISK.model_copy(update={"rationale": "Added because tariffs rose."})
    clean = REMOVED.model_copy(update={"rationale": "A disclosed location was dropped."})
    findings = (
        RISK_FINDING.model_copy(update={"change": explained}),
        GONE_FINDING.model_copy(update={"change": clean}),
        MARGIN_FINDING,
    )
    report = brief(findings=findings)
    markdown = render_markdown(report, METRICS)
    assert "Added because tariffs rose" not in markdown
    assert "Why it was flagged: A disclosed location was dropped." in markdown
    assert check_brief(report) == ()
    assert audit_trail(report)["rationales_omitted"] == ["run1:chg-risk"]


def test_the_audit_trail_holds_everything_behind_the_brief():
    root = TreeNode(
        node_id="run1:chg-gm:root", run_id="run1", change_id="chg-gm", depth=0, thought="gm fell"
    )
    pruned = TreeNode(
        node_id="run1:chg-gm:r1:c3",
        run_id="run1",
        change_id="chg-gm",
        parent_id=root.node_id,
        depth=1,
        thought="Currency moves lowered gross margin.",
        status=NodeStatus.PRUNED,
        prune_reason="score below the threshold",
    )
    claim = plausible("run1:chg-gm:r1:c2", "Gross margin fell.", metric_ids=("gm-2025",))
    finding = MARGIN_FINDING.model_copy(update={"claims": (claim,)})
    report = brief(findings=(finding,), summary="Gross margin fell. [C1]")
    trail = audit_trail(
        report,
        trees={finding.finding_id: (root, pruned)},
        labeling_failures={finding.finding_id: ("claim x: used the fixed wording",)},
        metrics=METRICS,
    )
    assert trail["brief"] == json.loads(report.model_dump_json())
    assert trail["refs"] == {"F1": "run1:chg-gm", "C1": "run1:chg-gm:r1:c2"}
    # Pruned branches are kept with their reasons.
    assert trail["trees"]["run1:chg-gm"][1]["prune_reason"] == "score below the threshold"
    assert trail["labeling_failures"] == {"run1:chg-gm": ["claim x: used the fixed wording"]}
    # The change's own metric and the claim's are both stored.
    assert set(trail["metrics"]) == {"gm-2025"}
    assert trail["metrics"]["gm-2025"]["value"] == 0.335
    json.dumps(trail)  # all of it is plain JSON


def test_write_run_writes_both_files_and_replaces_a_rerun(tmp_path):
    report = brief()
    files = write_run(report, tmp_path / "run1", metrics=METRICS)
    assert files.brief.read_text(encoding="utf-8") == render_markdown(report, METRICS)
    assert json.loads(files.audit.read_text(encoding="utf-8"))["refs"]["C1"] == "run1:chg-gm:r1:c1"
    rerun = brief(cost_usd=1.5)
    write_run(rerun, tmp_path / "run1")
    assert "model cost $1.50" in files.brief.read_text(encoding="utf-8")


def test_the_summary_line_round_trips():
    line = format_summary_line("Margin fell.", ["C1", "F2"])
    assert line == "Margin fell. [C1] [F2]"


@pytest.mark.parametrize(
    ("mapping", "enum"),
    [
        (SECTION_NAMES, Section),
        (LABEL_NAMES, ExplanationLabel),
        (CHANGE_KIND_NAMES, ChangeKind),
        (METRIC_NAMES, MetricName),
        (COMPARISON_NAMES, ComparisonType),
    ],
)
def test_every_member_has_a_display_name(mapping, enum):
    # A member added later must get a name here, or this fails before the brief does.
    for member in enum:
        assert mapping[member].strip(), member
    assert set(mapping) == set(enum)


def test_every_section_renders_in_a_citation_source():
    for section in Section:
        claim = plausible("run1:chg-gm:r1:c9", "Gross margin fell.", "Quote text.")
        claim = claim.model_copy(update={"citations": (cite("Quote text.", section=section),)})
        finding = MARGIN_FINDING.model_copy(update={"claims": (claim,)})
        markdown = render_markdown(brief(findings=(finding,), summary="Margin fell. [C1]"))
        assert f"10-K for FY2025, {SECTION_NAMES[section]}," in markdown


def test_a_numeric_heading_falls_back_to_the_metric_id_and_signs_dollars():
    report = brief(findings=(MARGIN_FINDING,), summary="Input costs. [C1]")
    assert "### [F1] Metric change: metric gm-2025" in render_markdown(report)
    loss = metric("gm-2025", CURRENT, -1234.4, name="gross_margin").model_copy(
        update={"unit": "USD"}
    )
    assert "Gross margin change: -$1,234 in FY2025" in render_markdown(report, FakeMetrics(loss))


LONG_QUOTE = (  # Real: HSY FY2025 10-K, paragraph mdna:0036, 451 characters on one line.
    "Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately "
    "1,380 basis points. The decrease was driven by higher commodity and tariff costs, unfavorable "
    "mark-to-market activity on our commodity derivative instruments and lower volume, which more "
    "than offset the benefits from net price realization, supply chain productivity, and net "
    "savings related to our Advancing Agility & Automation Initiative (“AAA Initiative”)."
)


def long_brief() -> Brief:
    claim = confirmed(
        "run1:chg-gm:r1:c1", "Higher commodity costs lowered gross margin.", LONG_QUOTE
    )
    finding = MARGIN_FINDING.model_copy(update={"claims": (claim,)})
    summary = (
        "Gross margin fell sharply over the year, and the filings say that higher commodity costs "
        "lowered it. [C1]"
    )
    return brief(findings=(finding,), summary=summary)


def test_no_line_is_longer_than_the_cap_except_a_heading_or_a_single_token():
    markdown = render_markdown(long_brief(), METRICS)
    for line in markdown.splitlines():
        if len(line) > WRAP_WIDTH:
            content = line.lstrip(" >-")
            assert line.startswith("#") or " " not in content, line


def test_wrapping_splits_no_word():
    # The line breaks the user saw ("th" / "em") came from a fixed-width view of long lines.
    markdown = render_markdown(long_brief(), METRICS)
    assert contains(markdown, LONG_QUOTE)
    for word in ("derivative", "them", "approximately", "Initiative"):
        assert all(
            not line.rstrip().endswith(word[:-1]) or line.rstrip().endswith(word)
            for line in markdown.splitlines()
            if word[:-1] in line
        )


def test_wrapped_lines_keep_their_markdown_structure():
    markdown = render_markdown(long_brief(), METRICS)
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  > Gross margin was"))
    quote = []
    for line in lines[start:]:
        if not line.startswith("  >"):
            break
        quote.append(line)
    # Every wrapped line of a quote under a claim keeps the claim's indent and the quote marker.
    assert len(quote) > 3 and all(line.startswith("  > ") or line == "  >" for line in quote)
    summary = next(i for i, line in enumerate(lines) if line.startswith("- Gross margin fell"))
    assert lines[summary + 1].startswith("  ") and lines[summary + 1].strip().endswith("[C1]")


def test_a_continuation_line_cannot_start_a_list_or_a_heading():
    # One long word fills the first line, so the marker is what lands at the start of the second.
    for marker, guarded in (
        ("- next", "\\- next"),
        ("+ next", "\\+ next"),
        ("2025. next", "2025\\. next"),
        ("=== next", "\\=== next"),
    ):
        lines = wrap_markdown(f"- {'x' * 96} {marker}").splitlines()
        assert lines[1] == f"  {guarded}", lines
    quoted = wrap_markdown(f"  > {'x' * 94} 1) next").splitlines()
    assert quoted[1] == "  > 1\\) next", quoted


def test_a_heading_and_a_long_url_are_never_broken():
    heading = "### [F1] " + "word " * 30
    assert wrap_markdown(heading) == heading
    url = "<https://www.sec.gov/" + "a" * 120 + ">"
    lines = wrap_markdown(f"  > Source: 10-K for FY2025, {url}").splitlines()
    assert url in lines[-1]


def test_a_quote_line_starting_with_a_dash_does_not_become_a_list():
    risky = RISK.model_copy(update={"current_text": "- tariffs\n2025. A new year"})
    finding = RISK_FINDING.model_copy(update={"change": risky})
    markdown = render_markdown(brief(findings=(finding,), summary="A risk. [C1]"))
    assert "> \\- tariffs" in markdown
    assert "> 2025\\. A new year" in markdown


def test_the_summary_list_ends_before_the_next_heading():
    markdown = render_markdown(brief(), METRICS)
    assert "[F3]\n\n## Changes with a confirmed explanation" in markdown


def test_an_unknown_metric_name_is_shown_raw():
    assert metric_name("gross_margin") == "Gross margin"
    assert metric_name("segment_mix") == "segment_mix"
