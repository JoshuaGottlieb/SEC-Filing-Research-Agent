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
