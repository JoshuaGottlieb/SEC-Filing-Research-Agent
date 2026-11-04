"""The one route from a downloaded filing to text that later stages may use.

Every filing is parsed with edgartools and validated. If validation fails, the fallback
parser gets a turn under the same checks. A filing that fails both is recorded as
rejected, and `load_validated_filing` refuses it, so bad text can't reach the index.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sec_agent.ingest.audit import (
    MIN_SENTENCE_COVERAGE,
    TableAudit,
    audit_tables,
    note_numbering_problems,
    section_ending_problems,
    section_problems,
    sentence_coverage,
)
from sec_agent.ingest.fallback import parse_fallback
from sec_agent.ingest.sections import (
    REQUIRED_CONTENT_CHARS,
    ParsedSection,
    SectionStatus,
    qa_problems,
    render_items,
    sections_from_items,
)
from sec_agent.ingest.tables import StripResult, TableRecord, strip_data_tables
from sec_agent.schemas import FormType, Section

# Bump when parsing or validation changes, so older records are redone, not trusted.
# Saved records carry the version they were made with; read_parsed_filing ignores any other.
PARSER_VERSION = 5


class FilingRejectedError(RuntimeError):
    """A filing failed validation, or its record is out of date."""


class ParserName(StrEnum):
    """Which parser produced the text that was kept."""

    EDGARTOOLS = "edgartools"
    FALLBACK = "fallback"


class ParsedFiling(BaseModel):
    """One filing's parsed sections and the result of validating them. Saved as JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parser_version: int = PARSER_VERSION
    accession: str
    form: FormType
    parser: ParserName
    # How the text was found: "edgartools", or one of the fallback layouts.
    layout: str
    sections: dict[Section, ParsedSection]
    # Share of sampled sentences that reached a parsed item (see audit.sentence_coverage).
    coverage: float
    table_audit: TableAudit
    # Why the filing was rejected. Empty means it passed every check.
    problems: tuple[str, ...]
    # When the fallback ran, why edgartools' output was rejected.
    primary_problems: tuple[str, ...] = ()
    # Every data table taken out of the text, so the fact parser can read them later.
    removed_tables: tuple[TableRecord, ...] = ()

    @property
    def passed(self) -> bool:
        # A filing is accepted only if no check found a problem.
        return not self.problems


def _build(
    html: str,
    strip: StripResult,
    form: FormType,
    accession: str,
    parser: ParserName,
    layout: str,
    rendered: dict[str, str],
    primary_problems: tuple[str, ...] = (),
) -> ParsedFiling:
    sections = sections_from_items(rendered, form, accession)
    coverage = sentence_coverage(strip.html, rendered)
    audit = audit_tables(html, strip, sections)
    # Every gate check runs, and every problem is recorded, so a rejection says everything
    # that is wrong rather than only the first thing found.
    problems = [
        *qa_problems(form, sections),
        *section_problems(form, sections, rendered),
        *note_numbering_problems(form, sections, rendered),
        *section_ending_problems(strip.html, sections, rendered),
        *audit.problems,
    ]
    if coverage < MIN_SENTENCE_COVERAGE:
        problems.append(f"only {coverage:.0%} of sampled sentences reached a parsed item")
    return ParsedFiling(
        accession=accession,
        form=form,
        parser=parser,
        layout=layout,
        sections=sections,
        coverage=coverage,
        table_audit=audit,
        problems=tuple(problems),
        primary_problems=primary_problems,
        removed_tables=strip.removed,
    )


def _required_sections_met(parsed: ParsedFiling) -> int:
    """How many of the required sections have real content at or above their minimum length."""
    return sum(
        1
        for section, minimum in REQUIRED_CONTENT_CHARS[parsed.form].items()
        if parsed.sections[section].status is SectionStatus.CONTENT
        and parsed.sections[section].char_count >= minimum
    )


def _is_better(candidate: ParsedFiling, current: ParsedFiling) -> bool:
    """Which of two failed parses to keep on record. Content matters more than problem count.

    Cosmos Health's fallback output had lost its notes, and it won a tie on number of problems
    against an `edgartools` output that had them.
    """

    # Compared in order: required sections met, then sentence coverage (rounded so tiny
    # differences don't decide it), then fewer problems.
    def score(parsed: ParsedFiling) -> tuple[int, float, int]:
        return (_required_sections_met(parsed), round(parsed.coverage, 2), -len(parsed.problems))

    return score(candidate) > score(current)


def parse_filing(html: str, form: FormType, accession: str) -> ParsedFiling:
    """Parse and validate one filing, using the fallback parser if edgartools fails."""
    # Step 1: take the data tables out. Figures come from XBRL and the table fact parser.
    strip = strip_data_tables(html)
    primary_problems: tuple[str, ...]
    primary: ParsedFiling | None = None
    try:
        primary = _build(
            html,
            strip,
            form,
            accession,
            ParserName.EDGARTOOLS,
            "edgartools",
            render_items(strip.html, form),
        )
    # edgartools is third-party code reading arbitrary filer HTML. Any failure there
    # is a reason to try the fallback, and it's kept as a recorded problem.
    except Exception as error:  # noqa: BLE001
        primary_problems = (f"edgartools raised {type(error).__name__}: {error}",)
    else:
        if primary.passed:
            return primary
        primary_problems = primary.problems

    # Step 2: edgartools failed or was rejected, so give the fallback parser a turn under the
    # same checks.
    result = parse_fallback(strip.html, form)
    fallback = _build(
        html,
        strip,
        form,
        accession,
        ParserName.FALLBACK,
        result.layout,
        result.rendered,
        primary_problems=primary_problems,
    )
    # Step 3: both failed. If edgartools crashed there's nothing to compare; otherwise keep
    # whichever failed parse is better, so the record shows the most useful attempt.
    if fallback.passed or primary is None:
        return fallback
    return fallback if _is_better(fallback, primary) else primary


def parsed_filing_path(parsed_dir: Path, ticker: str, accession: str) -> Path:
    """Where a filing's parse record lives: <parsed dir>/<ticker>/<accession>.json."""
    return parsed_dir / ticker / f"{accession}.json"


def save_parsed_filing(parsed: ParsedFiling, path: Path) -> None:
    """Write a record, rejected or not. Rejections are saved too, so they aren't retried."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(parsed.model_dump_json(), encoding="utf-8")


def read_parsed_filing(path: Path) -> ParsedFiling | None:
    """Read a saved record. None when it's absent or from an older parser version."""
    if not path.exists():
        return None
    parsed = ParsedFiling.model_validate_json(path.read_text(encoding="utf-8"))
    return parsed if parsed.parser_version == PARSER_VERSION else None


def load_validated_filing(path: Path) -> ParsedFiling:
    """What later stages call. Raises unless the filing is current and passed validation."""
    parsed = read_parsed_filing(path)
    if parsed is None:
        raise FilingRejectedError(
            f"{path} is missing or from an older parser version. Run `sec-agent ingest`."
        )
    if not parsed.passed:
        raise FilingRejectedError(
            f"{parsed.accession} failed validation: " + "; ".join(parsed.problems[:5])
        )
    return parsed
