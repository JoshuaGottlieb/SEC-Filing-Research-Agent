# Research brief: HSY: 10-K for FY2025 against FY2024

Comparison: year over year (10-K). Run HSY-yoy-10k-t81-20261115, generated
2026-11-16T02:38:47+00:00, model cost $0.86.

> **Not investment advice.** This brief is a first-pass research aid generated from SEC filings. It
> is not investment advice and makes no recommendation or price prediction. Verify findings against
> the cited filings before relying on them.

References such as [C1] (a claim) and [F1] (a change) point to `audit.json`, which holds the full
record behind every statement: the claim, its quotes, its metrics and its Tree-of-Thought tree.

## Summary

- Gross margin fell from 47.3% in FY2024 to 33.5% in FY2025, and the cost ratio rose from 52.7% to
  66\.5%; this brief did not confirm a link between these figures and the cost of sales change.
  [C25]
- Higher commodity and tariff costs, unfavorable commodity derivative mark-to-market activity, and
  lower volume lowered gross margin. [C2] [C14] [C15]
- Higher commodity and tariff costs and lower sales volume also lowered operating margin, while
  investments in advertising and related consumer marketing raised SM&A expenses. [C3] [C4] [C17]
- Higher income taxes lowered net income. [C18]

## Changes with a confirmed explanation

### [F1] Gross margin change: 33.5% in FY2025

Why it was flagged: Gross margin fell -1,377 bps year over year, a very large compression that
materially changes the profitability picture.

- **Confirmed** [C1]: Tariffs raised cost of sales in 2025.

  > Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase
  > of $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable
  > costs, primarily related to $736.6 million in higher commodity costs, $287.2 million in higher
  > supply chain costs, including tariffs
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

- **Confirmed** [C2]: Higher commodity and tariff costs, unfavorable commodity derivative
  mark-to-market activity, and lower volume lowered gross margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Revenue growth: 4.4% (FY2025)

### [F2] Operating margin change: 12.3% in FY2025

Why it was flagged: Operating margin declined -1,354 bps, a major deterioration in operating
profitability.

- **Confirmed** [C3]: Higher commodity and tariff costs lowered operating margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors noted
  > above in gross margin.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Confirmed** [C4]: Lower sales volume lowered operating margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors noted
  > above in gross margin.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

### [F3] Changed paragraph in Risk Factors

Why it was flagged: Commodity derivative mark-to-market swung from $563.0 million favorable in 2024
to $491.0 million unfavorable in 2025, adding to cost of sales. This ties directly to the flagged
gross margin move of -1,377 bps and the cost of goods sold move of +31.7%.

Before (FY2024):

> Although we use forward contracts and commodity futures and options contracts to hedge commodity
> prices where possible, commodity price increases ultimately result in corresponding increases in
> our raw material and energy costs. For example, our cost of sales during the year ended 2024
> compared to the same period of 2023 experienced an incremental $563.0 million of favorable
> mark-to-market activity on our commodity derivative instruments intended to economically hedge
> future years’ commodity purchases, more than offsetting higher commodity costs. During the year
> ended 2024, market prices for the majority of our exchange traded commodities increased,
> including cocoa, which has increased approximately 130% since the beginning of the year.

After (FY2025):

> Although we use forward contracts and commodity futures and options contracts to hedge commodity
> prices where possible, commodity price increases ultimately result in corresponding increases in
> our raw material and energy costs. For the year ended December 31, 2025, in addition to higher
> commodity costs, our cost of sales increased compared to the same period of 2024 as a result of
> $491.0 million of unfavorable mark-to-market activity on our commodity derivative instruments
> intended to economically hedge future years’ commodity purchases. During the year ended 2025,
> market prices for the majority of our exchange traded commodities remained volatile, including
> cocoa which has decreased from record highs but remains structurally elevated.

- **Confirmed** [C5]: Higher supply chain costs, including tariffs, increased cost of sales.

  > Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase
  > of $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable
  > costs, primarily related to $736.6 million in higher commodity costs, $287.2 million in higher
  > supply chain costs, including tariffs, as well as $491.0 million of unfavorable mark-to-market
  > activity on our commodity derivative instruments intended to economically hedge future years’
  > commodity purchases (See Item 7A - Quantitative and Qualitative Disclosures About Market Risk
  > for more information).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Cost of sales to revenue: 66.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Confirmed** [C6]: Higher commodity and tariff costs lowered gross margin, more than offsetting
  net price realization.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024)

- **Plausible** [C7]: Mark-to-market activity on commodity derivatives was unfavorable in 2025, and
  gross margin fell from 47.3% in 2024 to 33.5% in 2025; this brief did not confirm a link to gross
  margin, cost of sales.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Plausible** [C8]: SM&A expenses rose from $2,373.6 million in 2024 to $2,460.6 million in 2025,
  and operating margin fell from 25.9% to 12.3%; this brief did not confirm a link to gross margin
  or cost of sales.

  > SM&A expenses were $2,460.6 million in 2025 compared to $2,373.6 million in 2024, an increase
  > of $87.0 million, or 3.7%. The increase was driven by higher compensation and benefit costs and
  > investments in advertising and related consumer marketing expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: SG&A expense: $2,460,569,000 (FY2025); SG&A expense: $2,373,621,000 (FY2024); Operating
  margin: 12.3% (FY2025); Operating margin: 25.9% (FY2024)

### [F4] Changed paragraph in Management's Discussion and Analysis

Why it was flagged: Cost of sales rose to $7,769.9 million from $5,901.4 million, up 31.7%. This
includes $736.6 million in higher commodity costs and $491.0 million of unfavorable mark-to-market
activity, matching the flagged +31.7% COGS move.

Before (FY2024):

> Cost of sales were $6,167.2 million in 2023 compared with $5,920.5 million in 2022, an increase
> of $246.7 million, or 4.2%. The increase included $356.2 million of unfavorable costs driven by
> higher supply chain costs, including higher labor costs partially offset by lower logistics
> costs, and unfavorable mix. The increase was further driven by an incremental $97.7 million of
> unfavorable mark-to-market activity on our commodity derivative instruments intended to
> economically hedge future years’ commodity purchases. These increases were partially offset by
> $207.2 million of favorable supply chain productivity and price realization.

After (FY2025):

> Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase of
> $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable costs,
> primarily related to $736.6 million in higher commodity costs, $287.2 million in higher supply
> chain costs, including tariffs, as well as $491.0 million of unfavorable mark-to-market activity
> on our commodity derivative instruments intended to economically hedge future years’ commodity
> purchases (See Item 7A - Quantitative and Qualitative Disclosures About Market Risk for more
> information). The increase was partially offset by $96.6 million of favorable cost savings due to
> lower sales volume and lower business realignment costs.

- **Confirmed** [C9]: Unfavorable mark-to-market activity on commodity derivatives intended to
  hedge future commodity purchases increased cost of sales.

  > Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase
  > of $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable
  > costs, primarily related to $736.6 million in higher commodity costs, $287.2 million in higher
  > supply chain costs, including tariffs, as well as $491.0 million of unfavorable mark-to-market
  > activity on our commodity derivative instruments intended to economically hedge future years’
  > commodity purchases
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross profit: $3,922,691,000 (FY2025); Gross profit: $5,300,888,000 (FY2024); Operating
  income: $1,441,528,000 (FY2025); Operating income: $2,898,232,000 (FY2024)

- **Confirmed** [C10]: Higher commodity and supply chain costs, including tariffs, raised cost of
  sales.

  > Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase
  > of $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable
  > costs, primarily related to $736.6 million in higher commodity costs, $287.2 million in higher
  > supply chain costs, including tariffs
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Cost of sales to revenue: 66.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024);
  Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024)

- **Confirmed** [C11]: Higher commodity costs lowered gross margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Plausible** [C12]: SM&A expenses rose from $2,373.6 million in 2024 to $2,460.6 million in
  2025, and operating margin fell from 25.9% to 12.3%; this brief did not confirm a link to cost of
  sales.

  > SM&A expenses, excluding advertising and related consumer marketing, increased approximately
  > 4\.2% in 2025 driven by higher compensation costs, partially offset by net savings related to
  > our AAA Initiative.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > SM&A expenses were $2,460.6 million in 2025 compared to $2,373.6 million in 2024, an increase
  > of $87.0 million, or 3.7%. The increase was driven by higher compensation and benefit costs and
  > investments in advertising and related consumer marketing expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: SG&A expense: $2,460,569,000 (FY2025); SG&A expense: $2,373,621,000 (FY2024); Operating
  margin: 12.3% (FY2025); Operating margin: 25.9% (FY2024)

- **Plausible** [C13]: Cost of sales rose from 52.7% of revenue in FY2024 to 66.5% in FY2025, while
  volume fell about 1% on price elasticity in North America Confectionery and International; this
  brief did not confirm a link to cost of sales.

  > The increase was partially offset by a volume decrease of approximately 1%, primarily driven by
  > price elasticity impacts within the North America Confectionery and International segments,
  > partially offset by strong results in North America Salty Snacks.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Cost of sales to revenue: 66.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024);
  Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024)

### [F5] Changed paragraph in Management's Discussion and Analysis

Why it was flagged: Gross margin fell to 33.5% in 2025 from 47.3% in 2024, a decrease of
approximately 1,380 basis points (flagged at -1,377 bps). This is a major margin swing.

Before (FY2024):

> Gross margin was 44.8% in 2023 compared with 43.2% in 2022, an increase of 160 basis points. The
> increase was driven by favorable price realization and increased supply chain productivity. The
> increase was partially offset by unfavorable activity on our mark-to-market impact from commodity
> derivative instruments, higher supply chain costs, including higher labor costs and increased
> waste. The increase was further driven by unfavorable mix and foreign exchange rates.

After (FY2025):

> Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
> basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
> mark-to-market activity on our commodity derivative instruments and lower volume, which more than
> offset the benefits from net price realization, supply chain productivity, and net savings
> related to our Advancing Agility & Automation Initiative (“AAA Initiative”).

- **Confirmed** [C14]: Higher commodity and tariff costs lowered gross margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Confirmed** [C15]: Unfavorable mark-to-market activity on commodity derivative instruments
  lowered gross margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

### [F6] Changed paragraph in Management's Discussion and Analysis

Why it was flagged: Operating profit fell to $1,441.5 million from $2,898.2 million, a decrease of
50\.3%, and operating margin dropped to 12.3% from 25.9%. This matches the flagged operating\_margin
move of -1,354 bps.

Before (FY2024):

> Operating profit was $2,560.9 million in 2023 compared to $2,260.8 million in 2022, an increase
> of $300.1 million, or 13.3%. The increase was predominantly due to higher gross profit, partially
> offset by higher SM&A expenses, as noted above. Operating profit margin decreased to 22.9% in
> 2023 from 21.7% in 2022 by the same factors noted above in gross margin.

After (FY2025):

> Operating profit was $1,441.5 million in 2025 compared to $2,898.2 million in 2024, a decrease of
> $1,456.7 million, or 50.3%. The decrease was predominantly due to lower gross profit and higher
> SM&A expenses, partially offset by lower business realignment expenses, as noted above. Operating
> profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors noted above in
> gross margin.

- **Confirmed** [C16]: Higher commodity and tariff costs lowered gross margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Cost of sales to revenue: 66.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024);
  Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024)

- **Confirmed** [C17]: Investments in advertising and related consumer marketing raised SM&A
  expenses.

  > SM&A expenses were $2,460.6 million in 2025 compared to $2,373.6 million in 2024, an increase
  > of $87.0 million, or 3.7%. The increase was driven by higher compensation and benefit costs and
  > investments in advertising and related consumer marketing expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024)

### [F7] Changed paragraph in Management's Discussion and Analysis

Why it was flagged: Net income fell to $883.3 million from $2,221.2 million, a decrease of 60.2%,
and diluted EPS dropped to $4.34 from $10.92. This matches the flagged net\_margin move of -1,227
bps.

Before (FY2024):

> Net income was $2,221.2 million in 2024 compared to $1,861.8 million in 2023, an increase of
> $359.4 million, or 19.3%. Earnings Per Share (“EPS”)-diluted was $10.92 in 2024 compared to $9.06
> in 2023, an increase of $1.86, or 20.5%. The increase in both net income and EPS-diluted was
> driven primarily by higher gross profit, lower SM&A expenses and lower income taxes, partially
> offset by higher business realignment costs and higher other income and expenses. Our 2024
> EPS-diluted benefited from lower weighted-average shares outstanding as a result of share
> repurchases pursuant to our Board-approved repurchase programs.

After (FY2025):

> Net income was $883.3 million in 2025 compared to $2,221.2 million in 2024, a decrease of
> $1,337.9 million, or 60.2%. Earnings Per Share (“EPS”)-diluted was $4.34 in 2025 compared to
> $10.92 in 2024, a decrease of $6.58, or 60.3%. The decrease in both net income and EPS-diluted
> was driven by lower gross profit, higher SM&A expenses, higher interest expense, and higher
> income taxes, partially offset by lower business realignment costs and lower other expenses.

- **Confirmed** [C18]: Higher income taxes lowered net income.

  > Net income was $883.3 million in 2025 compared to $2,221.2 million in 2024, a decrease of
  > $1,337.9 million, or 60.2%. Earnings Per Share (“EPS”)-diluted was $4.34 in 2025 compared to
  > $10.92 in 2024, a decrease of $6.58, or 60.3%. The decrease in both net income and EPS-diluted
  > was driven by lower gross profit, higher SM&A expenses, higher interest expense, and higher
  > income taxes, partially offset by lower business realignment costs and lower other expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Net margin: 7.6% (FY2025); Net margin: 19.8% (FY2024)

- **Confirmed** [C19]: Higher SM&A spending, including advertising and consumer marketing
  investments, raised selling, marketing and administrative expenses.

  > SM&A expenses were $2,460.6 million in 2025 compared to $2,373.6 million in 2024, an increase
  > of $87.0 million, or 3.7%. The increase was driven by higher compensation and benefit costs and
  > investments in advertising and related consumer marketing expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Operating margin: 12.3%
  (FY2025); Operating margin: 25.9% (FY2024)

- **Plausible** [C20]: Cost of sales rose 31.7% to $7,769.9 million in 2025 and gross margin fell
  from 47.3% to 33.5%; this brief did not confirm a link from tariffs or cocoa costs to gross
  profit, selling, marketing, or net income.

  > Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase
  > of $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable
  > costs, primarily related to $736.6 million in higher commodity costs, $287.2 million in higher
  > supply chain costs, including tariffs
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Plausible** [C21]: Net interest expense rose from $165.7 million in 2024 to $190.2 million in
  2025, and net margin fell from 19.8% to 7.6%; this brief did not confirm a link to the change in
  gross profit, selling, marketing, net income.

  > Net interest expense was $190.2 million in 2025 compared to $165.7 million in 2024, an increase
  > of $24.5 million, or 14.8%. The increase was primarily due to higher long-term debt balances in
  > 2025 compared to 2024, driven by the February 2025 debt issuance.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > The decrease in both net income and EPS-diluted was driven by lower gross profit, higher SM&A
  > expenses, higher interest expense, and higher income taxes, partially offset by lower business
  > realignment costs and lower other expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Net margin: 7.6% (FY2025); Net margin: 19.8% (FY2024)

- **Plausible** [C22]: Gross margin fell from 47.3% in FY2024 to 33.5% in FY2025 while higher
  commodity and tariff costs were cited alongside net price realization; this brief did not confirm
  a link to gross profit, selling, marketing, net income.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024); Gross profit: $3,922,691,000 (FY2025);
  Gross profit: $5,300,888,000 (FY2024)

### [F8] Cost of sales change: $7,769,885,000 in FY2025

Why it was flagged: Cost of goods sold rose +31.7%, a significant cost increase tied directly to
the margin compression.

- **Confirmed** [C23]: Higher commodity and supply chain costs, including tariffs, raised cost of
  sales.

  > Cost of sales were $7,769.9 million in 2025 compared to $5,901.4 million in 2024, an increase
  > of $1,868.5 million, or 31.7%. The increase was driven by $1,965.1 million of unfavorable
  > costs, primarily related to $736.6 million in higher commodity costs, $287.2 million in higher
  > supply chain costs, including tariffs
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Plausible** [C24]: Commodity and tariff costs rose and gross margin fell from 47.3% to 33.5% in
  FY2025; this brief did not confirm a link to the cost of sales change.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > For the year ended December 31, 2025, in addition to higher commodity costs, our cost of sales
  > increased compared to the same period of 2024 as a result of $491.0 million of unfavorable
  > mark-to-market activity on our commodity derivative instruments intended to economically hedge
  > future years’ commodity purchases.
  >
  > Source: 10-K for FY2025, Risk Factors,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Plausible** [C25]: Gross margin fell from 47.3% in FY2024 to 33.5% in FY2025, and the cost
  ratio rose from 52.7% to 66.5%; this brief did not confirm a link to cost of sales.

  > The decrease was primarily due to higher commodity costs, higher supply chain costs, and
  > unfavorable product mix.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

### [F9] Changed paragraph in Management's Discussion and Analysis

Why it was flagged: Tariffs went from having no material impact in 2024 to a material negative
impact on 2025 results of operations and commodity prices, with ongoing exposure to Canada, Mexico
and other countries. This is a materially escalated risk.

Before (FY2024):

> Furthermore, certain geopolitical events, specifically the conflict between Russia and Ukraine,
> as well as the imposition of tariffs on U.S. imports and retaliatory tariffs in response, have
> increased global economic and political uncertainty. For the year ended December 31, 2024,
> neither the conflict between Russia and Ukraine, nor the imposition of tariffs on U.S. imports
> and retaliatory tariffs, had a material impact on our commodity prices or supply availability.
> However, we are continuing to monitor each of these events for any significant escalation or
> expansion of economic or supply chain disruptions or broader inflationary costs, which may result
> in material adverse effects on our results of operations.

After (FY2025):

> Furthermore, changes in global trade policies, including tariffs on U.S. imports, continue to
> increase global economic and political uncertainty. For the year ended December 31, 2025, the
> imposition of tariffs on U.S. imports and retaliatory tariffs, had a material negative impact on
> our results of operations and commodity prices. We are continuing to monitor the ongoing
> negotiations related to tariffs, specifically, goods imported into the U.S. from Canada, Mexico
> and other countries, as well as export markets, in which we have significant business operations,
> all of which may result in material adverse effects on our results of operations. The scope and
> length of tariffs, including their effects on the broader economy and our business, remain
> uncertain. These outcomes may be influenced by factors such as continued U.S. negotiations with
> impacted countries, retaliatory measures from other nations, possible tariff exemptions, public
> sentiment toward U.S. products and companies, and the domestic availability of lower-cost
> alternatives.

- **Confirmed** [C26]: Higher list prices raised net sales.

  > The net sales increase reflects a favorable price realization of approximately 6% primarily due
  > to higher list prices across all three segments
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

- **Confirmed** [C27]: Higher commodity and tariff costs lowered gross margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Cost of sales to revenue: 66.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024);
  Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024)

## Changes with plausible explanations only

### [F10] Net margin change: 7.6% in FY2025

Why it was flagged: Net margin dropped -1,227 bps, a large decline in bottom-line profitability.

- **Plausible** [C28]: Gross margin fell from 47.3% in 2024 to 33.5% in 2025, and filing text cites
  higher commodity and tariff costs among the factors; this brief did not confirm a link to net
  margin.

  > Gross margin was 33.5% in 2025 compared with 47.3% in 2024, a decrease of approximately 1,380
  > basis points. The decrease was driven by higher commodity and tariff costs, unfavorable
  > mark-to-market activity on our commodity derivative instruments and lower volume, which more
  > than offset the benefits from net price realization, supply chain productivity, and net savings
  > related to our Advancing Agility & Automation Initiative (“AAA Initiative”).
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > Operating profit margin decreased to 12.3% in 2025 from 25.9% in 2024 by the same factors noted
  > above in gross margin.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > CONSOLIDATED STATEMENTS OF INCOME
  > (in thousands, except per share amounts)
  > For the years ended December 31, \| 2025 \| 2024 \| 2023
  > Net sales \| $ \| 11,692,576 \| $ \| 11,202,263 \| $ \| 11,164,992
  > \[...\]
  > Gross profit \| 3,922,691 \| 5,300,888 \| 4,997,816
  >
  > Source: 10-K for FY2025, Notes to the Financial Statements,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Gross margin: 33.5% (FY2025); Gross margin: 47.3% (FY2024); Cost of sales to revenue:
  66\.5% (FY2025); Cost of sales to revenue: 52.7% (FY2024)

- **Plausible** [C29]: SM&A rose 3.7% to $2,460.6 million in 2025, citing compensation and
  advertising investments, while operating margin fell from 25.9% to 12.3%; this brief did not
  confirm a link to net margin.

  > SM&A expenses were $2,460.6 million in 2025 compared to $2,373.6 million in 2024, an increase
  > of $87.0 million, or 3.7%. The increase was driven by higher compensation and benefit costs and
  > investments in advertising and related consumer marketing expenses.
  >
  > Source: 10-K for FY2025, Management's Discussion and Analysis,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > Advertising expense is also charged to expense as incurred and totaled $611,951 in 2025,
  > $600,094 in 2024 and $604,853 in 2023.
  >
  > Source: 10-K for FY2025, Notes to the Financial Statements,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Operating margin: 12.3% (FY2025); Operating margin: 25.9% (FY2024)

- **Plausible** [C30]: FY2025 table rows show unrecognized tax benefit changes of 58,457 thousand
  and valuation allowance changes of 28,842 thousand; operating income fell from $2,898 million to
  $1,442 million. This brief did not confirm a link to net margin.

  > NOTES TO CONSOLIDATED FINANCIAL STATEMENTS—(Continued)
  > (amounts in thousands, except share data or if otherwise indicated)
  > For the year ended December 31, \| 2025
  > Amount ($) \| Percent (%)
  > \[...\]
  > Worldwide changes in unrecognized tax benefits \| 58,457 \| 4.8
  >
  > Source: 10-K for FY2025, Notes to the Financial Statements,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  > NOTES TO CONSOLIDATED FINANCIAL STATEMENTS—(Continued)
  > (amounts in thousands, except share data or if otherwise indicated)
  > For the year ended December 31, \| 2025
  > Amount ($) \| Percent (%)
  > \[...\]
  > Changes in valuation allowance \| 28,842 \| 2.4
  >
  > Source: 10-K for FY2025, Notes to the Financial Statements,
  > <https://www.sec.gov/Archives/edgar/data/47111/000162828026008586/hsy-20251231.htm>

  Metrics: Operating income: $1,441,528,000 (FY2025); Operating income: $2,898,232,000 (FY2024)

## Needs review

Nothing in this brief needs review.

## Not examined

This run explores at most 10 material changes, in order of materiality, so 42 further material
changes were not examined. audit.json lists them under not_examined, with their rank and
importance.
