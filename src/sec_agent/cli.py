"""Command line interface."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated

import duckdb
import typer

from sec_agent.config import REPO_ROOT, Settings, get_settings
from sec_agent.edgar.client import EdgarClient
from sec_agent.edgar.corpus import (
    CORPUS_TICKERS,
    Corpus,
    CorpusCompany,
    CorpusFiling,
    discover_corpus,
    download_corpus,
)
from sec_agent.finance.fact_store import FactStore
from sec_agent.ingest.pipeline import (
    ParsedFiling,
    parse_filing,
    parsed_filing_path,
    read_parsed_filing,
    save_parsed_filing,
)
from sec_agent.schemas import ComparisonType

# The pinned list of filings the project analyzes. Other corpus files are used for parser tests.
CORPUS_PATH = REPO_ROOT / "config" / "corpus.yaml"

CorpusOption = Annotated[Path, typer.Option("--corpus", help="Corpus file to use.")]

app = typer.Typer(no_args_is_help=True, help="SEC filing research agent.")


@app.callback()
def main() -> None:
    """SEC filing research agent."""


@app.command()
def doctor() -> None:
    """Check configuration and local storage. Never prints secret values."""
    settings = get_settings()
    failures: list[str] = []

    # Prints one [ok]/[FAIL] line and remembers failures so the exit code can reflect them.

    def report(name: str, passed: bool, detail: str = "") -> None:
        typer.echo(f"[{'ok' if passed else 'FAIL'}] {name}{f': {detail}' if detail else ''}")
        if not passed:
            failures.append(name)

    report("ANTHROPIC_API_KEY set", settings.anthropic_api_key is not None)
    report(
        "SEC_USER_AGENT has a contact email",
        bool(settings.sec_user_agent and "@" in settings.sec_user_agent),
    )
    report("strong model", True, settings.model_strong)
    report("worker model", True, settings.model_worker)
    report("per-run budget", True, f"${settings.max_usd_per_run:.2f}")

    settings.ensure_dirs()
    report("data directory", True, str(settings.data_dir))
    # Real write tests, because DuckDB and SQLite can fail on some shared folders.
    with tempfile.TemporaryDirectory(dir=settings.data_dir) as scratch:
        report("DuckDB read and write", _check_duckdb(Path(scratch)))
        report("SQLite WAL read and write", _check_sqlite(Path(scratch)))

    if failures:
        typer.echo(f"{len(failures)} check(s) failed.")
        raise typer.Exit(code=1)
    typer.echo("All checks passed.")


@app.command("discover-corpus")
def discover_corpus_command(
    corpus_path: CorpusOption = CORPUS_PATH,
    tickers: Annotated[
        list[str] | None, typer.Option("--ticker", help="Repeat for each company.")
    ] = None,
) -> None:
    """Find the latest eligible filings on EDGAR and pin them in a corpus file."""
    settings = get_settings()
    corpus = discover_corpus(
        EdgarClient(settings.require_sec_user_agent()),
        tuple(tickers) if tickers else CORPUS_TICKERS,
    )
    corpus.save(corpus_path)
    for company in corpus.companies:
        typer.echo(f"{company.ticker} ({company.company}, CIK {company.cik})")
        for filing in company.filings:
            # Show the filer's own tag when the derived period differs, so the fix is visible.
            corrected = (
                f"  (filer tagged {filing.filer_fiscal_period} {filing.filer_fiscal_year})"
                if filing.filer_fiscal_year is not None
                else ""
            )
            typer.echo(
                f"  {filing.form.value}  {filing.fiscal_period} {filing.fiscal_year}  "
                f"ended {filing.period_end}  filed {filing.filed_date}  {filing.accession}"
                f"{corrected}"
            )
    typer.echo(f"Wrote {corpus_path}")


def _read_html(settings: Settings, company: CorpusCompany, filing: CorpusFiling) -> str:
    """The downloaded filing's HTML. Untrusted text: it is only parsed, never executed."""
    path = settings.raw_filings_dir / company.ticker / filing.accession / filing.primary_document
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_pinned_filing(
    settings: Settings, company: CorpusCompany, filing: CorpusFiling
) -> ParsedFiling:
    return parse_filing(_read_html(settings, company, filing), filing.form, filing.accession)


@app.command()
def ingest(corpus_path: CorpusOption = CORPUS_PATH) -> None:
    """Download filings, validate each one, and load facts for those that pass.

    Exits 1 if any filing is rejected. Later stages can only read filings that passed, so
    nothing unvalidated can reach the index or the fact store.
    """
    settings = get_settings()
    settings.ensure_dirs()
    corpus = Corpus.load(corpus_path)
    client = EdgarClient(settings.require_sec_user_agent())
    downloaded, skipped = download_corpus(client, corpus, settings.raw_filings_dir)
    typer.echo(f"Downloaded {downloaded} file(s), {skipped} already present.")

    # For each filing: reuse a saved parse record if current, else parse; then load or reject.
    passed = rejected = 0
    with FactStore(settings.duckdb_path) as store:
        for company in corpus.companies:
            for filing in company.filings:
                record = parsed_filing_path(
                    settings.parsed_filings_dir, company.ticker, filing.accession
                )
                parsed = read_parsed_filing(record)
                if parsed is None:
                    parsed = _parse_pinned_filing(settings, company, filing)
                    save_parsed_filing(parsed, record)
                if parsed.passed:
                    passed += 1
                    if not store.is_loaded(filing.accession):
                        store.load_filing(
                            company, filing, _read_html(settings, company, filing), parsed
                        )
                    continue
                rejected += 1
                # A filing that once passed may fail under tighter rules, so drop its old facts.
                store.remove_filing(filing.accession)
                typer.echo(f"REJECTED {company.ticker} {filing.form.value} {filing.accession}")
                for problem in parsed.problems:
                    typer.echo(f"  {problem}")
        for company in corpus.companies:
            typer.echo(
                f"{company.ticker}: {store.count('xbrl_facts', company.ticker)} XBRL facts, "
                f"{store.count('table_facts', company.ticker)} table facts, "
                f"{store.count('unparsed_tables', company.ticker)} tables waiting as text."
            )
    typer.echo(f"Validated {passed + rejected} filing(s): {passed} passed, {rejected} rejected.")
    # A non-zero exit lets scripts and CI notice rejections.
    if rejected:
        raise typer.Exit(code=1)


@app.command()
def analyze(  # noqa: PLR0913 -- one argument per option
    ticker: Annotated[str, typer.Argument(help="Company ticker, such as HSY.")],
    comparison: Annotated[
        ComparisonType, typer.Option("--comparison", help="Which two filings to compare.")
    ],
    max_changes: Annotated[
        int | None,
        typer.Option("--max-changes", min=1, help="Most material changes to explore (default 10)."),
    ] = None,
    max_usd: Annotated[
        float | None,
        typer.Option(
            "--max-usd", min=0.01, help="Hard stop on model spend (default: the per-run budget)."
        ),
    ] = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Run id; a fresh one by default.")
    ] = None,
    corpus_path: CorpusOption = CORPUS_PATH,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Print the cost estimate and go ahead without asking."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Continue the stopped run with this id from its checkpoint."),
    ] = None,
) -> None:
    """Compare two filings and write a cited brief to the runs folder.

    Shows a cost estimate and asks before spending anything, unless `--yes` is given. Exits 1
    with no brief if the estimate is declined, the budget cannot cover the materiality step, the
    materiality step hits the budget, the brief breaks a rule, or a run cannot be resumed. A
    stopped run continues from its last finished step with `--resume RUN_ID`.
    """
    if resume is not None and run_id is not None:
        raise typer.BadParameter("give --resume or --run-id, not both")
    # Imported here: the workflow loads torch and the search models, which other commands
    # (doctor, search-facts) do not need.
    from sec_agent.graph import (
        DEFAULT_MAX_CHANGES,
        CostEstimate,
        ResumeError,
        RunDeclined,
        describe_estimate,
        run_analysis,
    )
    from sec_agent.llm.ledger import BudgetExceededError
    from sec_agent.report.brief import BriefError

    def confirm(estimate: CostEstimate) -> bool:
        for line in describe_estimate(estimate):
            typer.echo(line)
        return yes or typer.confirm("Go ahead?", default=False)

    started: list[str] = []

    def announce(new_run_id: str) -> None:
        started.append(new_run_id)
        typer.echo(f"Run {new_run_id}")

    def resume_hint() -> str:
        name = started[0] if started else "RUN_ID"
        return f"sec-agent analyze {ticker.upper()} --comparison {comparison.value} --resume {name}"

    settings = get_settings()
    settings.ensure_dirs()
    try:
        report = asyncio.run(
            run_analysis(
                settings,
                ticker.upper(),
                comparison,
                max_changes=max_changes or DEFAULT_MAX_CHANGES,
                max_usd=max_usd,
                run_id=resume or run_id,
                corpus_path=corpus_path,
                confirm=confirm,
                resume=resume is not None,
                announce=announce,
            )
        )
    except KeyboardInterrupt as error:
        typer.echo(f"Stopped. Continue from the last finished step with: {resume_hint()}")
        raise typer.Exit(code=130) from error
    except BudgetExceededError as error:
        typer.echo(f"No brief written: {error}")
        typer.echo(f"To continue with a higher --max-usd: {resume_hint()} --max-usd AMOUNT")
        raise typer.Exit(code=1) from error
    except ResumeError as error:
        # Nothing ran, and a finished run's brief is already there, so "no brief" would mislead.
        typer.echo(f"Not run: {error}")
        raise typer.Exit(code=1) from error
    except (BriefError, RunDeclined) as error:
        typer.echo(f"No brief written: {error}")
        raise typer.Exit(code=1) from error
    typer.echo(f"Brief: {report.brief}")
    typer.echo(f"Audit trail: {report.audit}")
    typer.echo(
        f"Run {report.run_id}: {report.selected} change(s) selected, {report.searched} searched, "
        f"{report.not_examined} not examined, {report.node_evaluations} node evaluations, "
        f"model cost ${report.cost_usd:.4f} (estimated ${report.estimated_usd:.2f})."
    )


@app.command("search-facts")
def search_facts(
    ticker: str,
    query: Annotated[str | None, typer.Argument(help="Words that must appear in a fact.")] = None,
    source: Annotated[str | None, typer.Option(help="`xbrl` or `table`.")] = None,
    limit: int = 25,
) -> None:
    """Search the fact store the way the Financial Agent will."""
    with FactStore(get_settings().duckdb_path) as store:
        for fact in store.search(ticker, query, source=source, limit=limit):
            detail = f" | {fact.column_label}" if fact.column_label else ""
            detail += f" | under: {fact.heading}" if fact.heading else ""
            typer.echo(
                f"[{fact.source}] {fact.end_date} {fact.duration or '-':8} "
                f"{fact.value:>20,.2f} {fact.unit or '':8} {fact.name}{detail}"
            )


@app.command("parse-report")
def parse_report(corpus_path: CorpusOption = CORPUS_PATH) -> None:
    """Parse and audit every pinned filing from scratch. Exits 1 on any problem.

    Unlike `ingest`, this ignores saved records and loads nothing, so it measures the parser.
    """
    settings = get_settings()
    corpus = Corpus.load(corpus_path)
    problem_count = 0
    for company in corpus.companies:
        for filing in company.filings:
            parsed = _parse_pinned_filing(settings, company, filing)
            audit = parsed.table_audit
            problem_count += len(parsed.problems)
            typer.echo(
                f"{company.ticker} {filing.form.value} {filing.fiscal_period} "
                f"{filing.fiscal_year}: {'ok' if parsed.passed else 'PROBLEMS'}  "
                f"[{parsed.parser.value}, {parsed.layout}]"
            )
            typer.echo(
                f"  tables: {audit.tables_removed} of {audit.tables_total} removed, "
                f"{audit.prose_rows_preserved} prose rows kept; "
                f"sentence coverage {parsed.coverage:.0%}"
            )
            for section in parsed.sections.values():
                note = f"  [{section.note}]" if section.note else ""
                typer.echo(
                    f"  {section.section.value:16} {section.status.value:20} "
                    f"{section.char_count:7} chars {len(section.paragraphs):5} paras{note}"
                )
            for problem in parsed.primary_problems:
                typer.echo(f"  edgartools rejected: {problem}")
            for problem in parsed.problems:
                typer.echo(f"  PROBLEM: {problem}")
    if problem_count:
        typer.echo(f"{problem_count} problem(s) found.")
        raise typer.Exit(code=1)
    typer.echo("All filings passed.")


def _check_duckdb(directory: Path) -> bool:
    """Create a table, insert a row and read it back."""
    with duckdb.connect(directory / "check.duckdb") as db:
        db.execute("CREATE TABLE t (x INTEGER); INSERT INTO t VALUES (42)")
        return db.execute("SELECT x FROM t").fetchone() == (42,)


def _check_sqlite(directory: Path) -> bool:
    """The same round trip in SQLite's WAL mode, which the tree store and checkpoints use."""
    db = sqlite3.connect(directory / "check.sqlite")
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE t (x INTEGER)")
        db.execute("INSERT INTO t VALUES (42)")
        db.commit()
        return db.execute("SELECT x FROM t").fetchone() == (42,)
    finally:
        db.close()


if __name__ == "__main__":
    app()
