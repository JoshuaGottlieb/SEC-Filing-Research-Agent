from __future__ import annotations

from datetime import date

import pytest

from sec_agent.config import Settings
from sec_agent.schemas import Filing, FiscalPeriod, FormType


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that never touch the real data directory or the real keys."""
    return Settings(
        ANTHROPIC_API_KEY="test-key",
        SEC_USER_AGENT="Test User test@example.com",
        data_dir=tmp_path,
        max_usd_per_run=1.0,
    )


@pytest.fixture
def make_filing():
    """A factory for Filing records. Defaults to Hershey's FY2025 10-K; override what matters."""

    def build(
        form: FormType = FormType.FORM_10K,
        year: int = 2025,
        fiscal_period: str = "FY",
        period_end: date | None = None,
        ticker: str = "HSY",
    ) -> Filing:
        return Filing(
            ticker=ticker,
            company="The Hershey Company",
            cik="0000047111",
            form=form,
            period=FiscalPeriod(
                fiscal_year=year,
                fiscal_period=fiscal_period,
                period_end=period_end or date(year, 12, 31),
            ),
            accession=f"0000047111-{year}-000001",
            filed_date=date(year + 1, 2, 15),
            url="https://www.sec.gov/Archives/example.htm",
        )

    return build


# --- Saved runs, as the Streamlit app reads them (`report.run_view`, `ui.streamlit_app`) ---------

RUN_ID = "run1"
MARGIN_ID = f"{RUN_ID}:chg-gm"
RISK_ID = f"{RUN_ID}:chg-risk"
GONE_ID = f"{RUN_ID}:chg-gone"
# A rationale and a quote with two dollar amounts: Streamlit would read "$...$" as LaTeX.
DOLLAR_QUOTE = "Net sales rose to $2.8 billion from $2.6 billion, up $118 million."


def _saved_runs(data_dir):
    """Two saved runs under `data_dir/runs`: a full HSY run and an empty MAT run.

    The HSY run has a numeric finding with a confirmed and a plausible claim and a full tree
    (kept, pruned and pending nodes), a risk-factor finding, and a finding with no claims that
    needs review; labeling notes, a rationale dropped for an untraced figure, a change past the
    cap, and a cost ledger with timings.
    """
    import json
    from datetime import UTC, datetime

    from sec_agent.compare.materiality import MaterialityDecision
    from sec_agent.llm.ledger import CostLedger, LedgerEntry
    from sec_agent.report.brief import NotExamined, SkippedChange, audit_trail
    from sec_agent.schemas import (
        Brief,
        Change,
        ChangeKind,
        Citation,
        Claim,
        Comparison,
        ComparisonType,
        Duration,
        EntailmentVerdict,
        EvidenceSourceRef,
        EvidenceSupport,
        ExplanationLabel,
        Finding,
        MetricValue,
        NodeStatus,
        Section,
        TreeNode,
    )
    from sec_agent.timing import Timing, TimingLog

    def filing(ticker, company, year, accession):
        return Filing(
            ticker=ticker,
            company=company,
            cik="0000000001",
            form=FormType.FORM_10K,
            period=FiscalPeriod(
                fiscal_year=year, fiscal_period="FY", period_end=date(year, 12, 31)
            ),
            accession=accession,
            filed_date=date(year + 1, 2, 15),
            url=f"https://www.sec.gov/{accession}.htm",
        )

    def cite(quote, chunk_id="acc-2025:mdna:0001:0", section=Section.MDNA):
        return Citation(
            chunk_id=chunk_id,
            accession="acc-2025",
            ticker="HSY",
            form=FormType.FORM_10K,
            period=FiscalPeriod(
                fiscal_year=2025, fiscal_period="FY", period_end=date(2025, 12, 31)
            ),
            section=section,
            quote=quote,
            url="https://www.sec.gov/acc-2025.htm",
        )

    def metric(metric_id, year, value):
        return MetricValue(
            metric_id=metric_id,
            ticker="HSY",
            name="gross_margin",
            period=FiscalPeriod(
                fiscal_year=year, fiscal_period="FY", period_end=date(year, 12, 31)
            ),
            duration=Duration.TWELVE_MONTHS,
            value=value,
            unit="ratio",
            input_fact_ids=("f1",),
        )

    def node(suffix, depth, thought, parent=None, status=NodeStatus.KEPT, **fields):
        return TreeNode(
            node_id=f"{MARGIN_ID}:{suffix}",
            run_id=RUN_ID,
            change_id="chg-gm",
            parent_id=None if parent is None else f"{MARGIN_ID}:{parent}",
            depth=depth,
            thought=thought,
            status=status,
            **fields,
        )

    hsy = Comparison(
        comparison_id="HSY-yoy",
        ticker="HSY",
        comparison_type=ComparisonType.YOY_10K,
        current=filing("HSY", "The Hershey Company", 2025, "acc-2025"),
        prior=filing("HSY", "The Hershey Company", 2024, "acc-2024"),
    )
    cocoa = "Gross margin decreased due to higher input costs, primarily cocoa."
    margin = Finding(
        finding_id=MARGIN_ID,
        change=Change(
            change_id="chg-gm", comparison_id="HSY-yoy", kind=ChangeKind.NUMERIC, metric_id="gm-25"
        ),
        tree_root_id=f"{MARGIN_ID}:root",
        claims=(
            Claim(
                claim_id=f"{MARGIN_ID}:r1:c2",
                text="Promotions rose and gross margin fell; the filings do not link the two.",
                label=ExplanationLabel.PLAUSIBLE,
                metric_ids=("gm-25", "gm-24"),
            ),
            Claim(
                claim_id=f"{MARGIN_ID}:r1:c1",
                text="Higher input costs lowered gross margin.",
                label=ExplanationLabel.CONFIRMED,
                citations=(cite(cocoa),),
                entailment=EntailmentVerdict.STATES_CAUSE,
            ),
        ),
    )
    risk = Finding(
        finding_id=RISK_ID,
        change=Change(
            change_id="chg-risk",
            comparison_id="HSY-yoy",
            kind=ChangeKind.ADDED,
            section=Section.RISK_FACTORS,
            current_text="Tariffs on imported cocoa could raise our costs.",
            rationale=DOLLAR_QUOTE,
        ),
        claims=(
            Claim(
                claim_id=f"{RISK_ID}:r1:c1",
                text="Sales grew; the filings do not link it to the new tariff risk.",
                label=ExplanationLabel.PLAUSIBLE,
                citations=(cite(DOLLAR_QUOTE, "acc-2025:risk_factors:0002:0"),),
            ),
        ),
    )
    gone = Finding(
        finding_id=GONE_ID,
        change=Change(
            change_id="chg-gone",
            comparison_id="HSY-yoy",
            kind=ChangeKind.REMOVED,
            section=Section.BUSINESS,
            prior_text="We operate a retail store in Hershey, Pennsylvania.",
        ),
        review_reason="every candidate was pruned",
    )
    tree = (
        node("root", 0, "numeric change: gross margin fell 1,380 bps"),
        node("r1:c1", 1, 'Cocoa costs rose, "sharply", and lowered margin.', "root", score=1.0),
        node("r1:c2", 1, "Promotional spending rose.", "root", score=0.8),
        node(
            "r1:c3",
            1,
            "A plant closure lowered margin.",
            "root",
            status=NodeStatus.PRUNED,
            score=0.2,
            prune_reason="below the pre-screen threshold",
        ),
        node(
            "r1:c1:e1",
            2,
            "MD&A on input costs",
            "r1:c1",
            score=0.9,
            source=EvidenceSourceRef(section=Section.MDNA),
            evidence=EvidenceSupport.STATES_CAUSE,
            citations=(cite(cocoa),),
        ),
        node(
            "r1:c1:e2",
            2,
            "Risk factors on cocoa",
            "r1:c1",
            status=NodeStatus.PRUNED,
            score=0.1,
            source=EvidenceSourceRef(section=Section.RISK_FACTORS),
            evidence=EvidenceSupport.NONE,
            prune_reason="no evidence after one reformulated search",
        ),
        node(
            "r1:c3:e1",
            2,
            "Notes on plant closures",
            "r1:c3",
            status=NodeStatus.PRUNED,
            prune_reason="its parent was pruned",
        ),
        node("r1:c2:e1", 2, "Search everywhere for promotions", "r1:c2", status=NodeStatus.PENDING),
    )

    class Metrics:
        values = {
            m.metric_id: m for m in (metric("gm-25", 2025, 0.335), metric("gm-24", 2024, 0.473))
        }

        def get_metric(self, metric_id):
            return self.values.get(metric_id)

    hsy_brief = Brief(
        run_id=RUN_ID,
        comparison=hsy,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        summary="- Higher input costs lowered gross margin. [C1]\n- A tariff risk was added. [C3]",
        findings=(risk, gone, margin),
        cost_usd=0.42,
    )
    trail = audit_trail(
        hsy_brief,
        trees={MARGIN_ID: tree},
        labeling_failures={
            MARGIN_ID: (
                f"claim {MARGIN_ID}:r1:c3: the judge found every quote unrelated to it, so it is "
                "left out of the brief",
                f"claim {MARGIN_ID}:r1:c2: confirmed as written, but no wording at the quote's "
                "level could be written, so it is left unconfirmed",
            ),
            RISK_ID: (f"rework: {RISK_ID}:r1:c1:e1: the critic failed: the reply was cut off",),
        },
        metrics=Metrics(),
        not_examined=NotExamined(
            max_changes=3,
            changes=(
                SkippedChange(
                    change_id="HSY-yoy:yoy-10k:modified:acc-2025:mdna:0044",
                    materiality_rank=4,
                    importance=3,
                ),
            ),
        ),
        untraced_rationales=(
            MaterialityDecision(
                change_id="HSY-yoy:yoy-10k:modified:acc-2025:mdna:0073",
                material=True,
                importance=3,
                criterion=None,
                rationale="Corporate expense rose, reversing a 12.4% decline.",
                untraced_figures=("12.4%",),
            ),
        ),
        summary_used_template=False,
    )
    runs = data_dir / "runs"
    folder = runs / RUN_ID
    folder.mkdir(parents=True)
    (folder / "audit.json").write_text(json.dumps(trail), encoding="utf-8")
    ledger = CostLedger(10.0, path=folder / "ledger.jsonl")
    ledger.record(
        LedgerEntry(
            "critic",
            "claude-haiku-5-5",
            10,
            5,
            0,
            0,
            0.02,
            False,
            elapsed_s=2.0,
            stage="search",
            change_id="chg-gm",
        )
    )
    TimingLog(folder / "timings.jsonl").add(Timing("search", "chg-gm", 30.0))

    mat = Comparison(
        comparison_id="MAT-yoy",
        ticker="MAT",
        comparison_type=ComparisonType.YOY_10K,
        current=filing("MAT", "Mattel, Inc.", 2025, "mat-2025"),
        prior=filing("MAT", "Mattel, Inc.", 2024, "mat-2024"),
    )
    empty = Brief(
        run_id="run-mat",
        comparison=mat,
        generated_at=datetime(2026, 11, 9, 12, 0, tzinfo=UTC),
        summary="No material changes were found.",
    )
    (runs / "run-mat").mkdir()
    (runs / "run-mat" / "audit.json").write_text(json.dumps(audit_trail(empty)), encoding="utf-8")
    # A run folder with no audit trail (stopped before its brief) is not listed.
    (runs / "stopped").mkdir()
    (runs / "stopped" / "ledger.jsonl").write_text("", encoding="utf-8")
    return runs


@pytest.fixture
def saved_runs(tmp_path):
    """The runs folder holding the two saved runs described in `_saved_runs`."""
    return _saved_runs(tmp_path)
