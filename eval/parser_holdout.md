# Parser validation results

Measured on 2026-11-03. Figures are directional: the held-out set is 5 companies.

## Method

**Companies.** Drawn at random from EDGAR's ticker list (seed 20261103), excluding the 8
companies used in development. Seven draws were needed: FinVolution files no 10-K and JAB
Acquisition has fewer than two. Selected: Avnet, Albemarle, National Vision, CVD Equipment,
Flexsteel, for 25 filings in `config/parser_holdout.yaml`.

**Frozen rules.** Parser files were fingerprinted before the draw and confirmed unchanged
after the run. The section results below describe those frozen rules. Some problems were fixed
afterward, as listed under "Changes since the frozen run", which doesn't change these numbers.

**Scoring.** A filing passes the gate, or doesn't. Then its section boundaries were read by
hand and compared with the other parser's output, because passing doesn't prove correctness.

## Section parsing on unseen filings

| Measure | Result | 95% interval |
|---|---|---|
| Filings accepted by the gate | 18 of 25 (72%) | 51% to 88% |
| Companies with every filing accepted | 3 of 5 | 15% to 95% |
| Accepted filings with a materially wrong section | 0 of 18 | up to 19% |
| Accepted filings with small noise at a section's edge | 6 of 18 | |
| Rejected filings that appear correct (false alarms) | 7 of 7 | |
| Filings needing the fallback parser | 14 of 25 | |

Filings from one company resemble each other, so the filing-level intervals are too narrow.

**False alarms.**
- National Vision (5 filings): a long balance-sheet row label with values counts as a
  paragraph made mostly of numbers.
- CVD Equipment (2 filings): Item 10 holds director biographies, about 22,000 characters,
  over the 20,000 limit for a normally short item.

**Edge noise.** Albemarle's 10-Q notes start with four table-of-contents lines (3 filings).
Avnet's 10-Q notes end with forward-looking statements (3 filings).

**Other weaknesses found.**
- Albemarle's leftover-table flag cleared only because the fallback renders kept tables row
  by row, so the check stopped firing. The table itself remains in the text.
- Flexsteel tags its March 2026 10-Q as fiscal 2025 Q3, the same as March 2025, so corpus
  selection silently chose the older filing.
- Boundaries were read at each section's first and last paragraph. Errors in the middle
  would only show up in the comparison between parsers.

## Changes since the frozen run

These filings helped shape the fixes, so the results here don't replace the rates above. The
second held-out test, below, gives the fair rate for the updated rules.

| Problem | Fix | Result |
|---|---|---|
| National Vision's long balance-sheet row | A row with only numbers beside its label is a data row, however long the label | 5 filings pass, and the row no longer reaches the text |
| CVD Equipment's long Item 10 | A long, normally short item fails only if it holds MD&A or statement headings | 2 filings pass, and a real swollen Item 14 from Chevron still fails |
| Flexsteel's stale quarter | Fiscal periods derived from 10-K dates, keeping the filer's tag when it differs | March 2026 is selected, with labels Q3 2026, Q3 2025, and Q2 2026 |

All 25 filings in `config/parser_holdout.yaml` now pass. That file still holds the filings
that were measured, so it pins Flexsteel's older quarters. The newly selected Flexsteel filings
were run through the gate separately, and they pass.

**Fiscal labels.** Derived periods were compared with filers' tags on 694 filings from 18
companies, including Walmart, Target, Costco, Microsoft, and Deere for unusual fiscal years. They
differed on 11 filings. Each was read and the filer's tag was wrong, and all 7 of Flexsteel's were
among them. No filing was left out.

## Table figures

Parsing doesn't read XBRL tags, so tagged cells act as an answer key.

| Check | Development (8 companies) | Held-out (5 companies) |
|---|---|---|
| Tagged cells, parser frozen before held-out run | | 99.23% of 4,940 |
| Tagged cells, final rules: value | 100% of 12,174 | 100% of 4,448 (no longer blind) |
| Tagged cells, final rules: period end | 99.43% | 100% (no longer blind) |
| Same row and period in two filings agree (untagged) | 98.3% of 658 | 100% of 58 |

The answer key overstates accuracy on untagged tables, which are the actual target.

| Hand audit of untagged facts, against the raw table | Result |
|---|---|
| First, before dollars required evidence | About 12 of 45 wrong, some by a factor of 1,000 |
| Second, after the rules were tightened | 60 of 60 correct, 14 of them dashes |
| Third, only searchable facts, no dashes | 40 of 40 numbers correct; 2 with misleading labels |

The 2 misleading facts led to a new rule that sends such tables to text. With 0 numeric errors
in the last 100 facts, the 95% upper bound on the numeric error rate is about 3.6%. The rules
were adjusted between audits, and these samples were drawn from filings used to adjust them.

**Coverage.** About 30% of data tables parse cleanly. The rest wait in `unparsed_tables` for
the chunker. Share counts, per-share figures, and non-dollar quantities such as Chevron's
production volumes aren't facts yet, by design, and stay reachable as text.

## Second held-out test: updated rules

Same method with a new seed (20261104), excluding every company used so far. No draw was
skipped. Selected: Western Digital, United Bancshares, MKS, Cosmos Health, Ingram Micro, for 25
filings in `config/parser_holdout2.yaml`. Code was fingerprinted before the draw and confirmed
unchanged after the run. Nothing found here has been fixed.

| Measure | Result | 95% interval |
|---|---|---|
| Filings accepted by the gate | 17 of 25 (68%) | 46% to 85% |
| Companies with every filing accepted | 3 of 5 | 15% to 95% |
| Accepted filings with a materially wrong section | 0 of 17 | up to 20% |
| Accepted filings with noise inside a section | 2 of 17 | |

Accepted filings across both tests had no materially wrong section: 0 of 35, up to 10%. The rules
changed between the tests, and the same limits on reading apply.

| The 8 rejections | Filings | Was rejecting them right? |
|---|---|---|
| United Bancshares 10-K: MD&A and statements are in Exhibit 13, which isn't read | 2 | Yes |
| Cosmos Health 10-Q, Q2 2025: both parsers lost the notes | 1 | Yes |
| Cosmos Health 10-Q, Q2 and Q1 2026: `edgartools` looks right, but its output was discarded | 2 | Only for the output it chose |
| United Bancshares 10-Q: a row label holds a sentence, which was dropped with its figures | 3 | No, sections look right |

**Why Cosmos Health's good output was discarded.** Two rules combine. The title check fails a
section whose text is only "Not applicable" with no heading. Then, with both parsers at two
problems, the tie goes to the fallback, even though it had lost 150,000 characters of notes.

**Noise inside a section.** Western Digital's 10-K notes carry a running header, "(Continued)",
merged into 46 of about 540 paragraphs (2 filings). The gate doesn't check for this.

### Table figures on these filings

| Check | Result |
|---|---|
| Tagged cells, value and period end | 99.44% of 3,039 |
| Same row and period in two filings agree (untagged) | 94.5% of 109 |
| Hand audit of 36 searchable facts, no dashes | 36 of 36 values right; 35 of 36 fully right |
| Data tables that parsed cleanly | 18.9% of 1,155, below 29% to 35% earlier |

Problems found:
- **Percentage columns:** MKS's "Percentage" column reads as dollars when no % sign appears. The
  check against XBRL tags moved those tables to text, and no such fact is in the database, but an
  untagged table of this shape would get through.
- **Fiscal year ends:** a bare-year column is matched to the nearest date in the filing, so
  Western Digital's fiscal 2025 reads as June 30, 2025, not June 27. The tagged-cell check allows
  10 days, so it can't see this.
- **Disagreements between filings:** those traced were restated values or a shared label with
  opposite signs, not misreads. Not all were traced.

### After this test

Four problems found here were fixed afterward. These filings shaped the fixes, so 68% remains
the fair rate for the rules that were tested, and no fair rate exists yet for the current ones.

| Problem | Fix | Result |
|---|---|---|
| Bare-year columns matched to any nearby date | Only dates that close a reporting period count | 0 of 686 bare-year facts misdated, was 9 |
| "Percentage" column read as dollars | Treated as a percent, with or without a % sign | Tagged-cell value accuracy on these filings 100%, was 99.47% |
| Title check failed on "Not applicable" | Sections of 1,200 characters or fewer are exempt | 2 Cosmos Health 10-Qs now pass using `edgartools` |
| Tie between failed parses went to the fallback | Keep the one with more required sections, then better coverage | Rejection records now keep the fuller output |

The gate now accepts 19 of these 25 filings. The other 6 are the United Bancshares filings and
Cosmos Health's Q2 2025 10-Q, which are listed as known rejections in the real-filing tests.

### What this shows

The rate didn't improve: 68% against 72%, within the noise for samples this small. The fixes
removed two causes of false alarms, and these companies exposed others. Rejections are now also
partly real, which the first test didn't show. Across both tests the gate failed safe, so the
main cost is coverage: filings that could have been used are refused.

## Third held-out test: current rules

Measured on 2026-11-04 with the scripts in [ingestion/](ingestion/README.md). Same method with a
new seed (20261105), excluding every company used so far. Two draws were skipped: UP Fintech files
no 10-K or 10-Q, and SSHT S&T has fewer than two 10-Ks. Selected: RxSight, Comstock, Autoscope,
Everus, AT&T, for 25 filings in `config/parser_holdout3.yaml`. The rule files matched
`ingestion/holdout3.sha256` before the draw and after the run. Nothing found here has been fixed.

**This is the first test where the gate accepted wrong sections.** The acceptance rate rose, but
acceptance isn't correctness, so read the two rows together.

| Measure | Result | 95% interval |
|---|---|---|
| Filings accepted by the gate | 24 of 25 (96%) | 80% to 100% |
| Companies with every filing accepted | 4 of 5 | 28% to 100% |
| Accepted filings with a materially wrong section | 4 of 24 (17%) | 5% to 37% |
| Accepted and with no material error | 20 of 25 (80%) | 59% to 93% |
| Accepted filings with noise at a section's edge | 3 of 24 | |
| Filings needing the fallback parser | 3 of 25 | 3% to 31% |

| The 4 wrong sections | What happened | How it was found |
|---|---|---|
| Everus 10-K, 2025 and 2024: notes | `edgartools` ended Item 8 early and filed the last notes under signatures. About 100 sentences are missing, including the segment note (18% and 20% of the notes' sentences) | Parser comparison |
| AT&T 10-K, 2025 and 2024: market risk | About 15,000 of 19,500 characters are management's report and the audit reports, which sit between Item 7A and Item 8 | Reading the last paragraph. Both parsers agree, so the comparison stayed quiet |

The gate missed both because each check was satisfied. Everus's notes still run 89,000 to 102,000
characters, and sentence coverage counts the signature section as parsed. No check looks for audit
reports under market risk.

**The one rejection.** Comstock's 2024 10-K keeps two share issuance tables with a single number
column in its text. The check is right about the tables, and the sections look right, so this
is closer to a false alarm.

**Smaller problems.**
- AT&T's 10-Q sections end with the banner "AT&T INC." (3 filings).
- Comstock's 10-Q says "No new risk factors", which is labeled `content`, not `no_material_changes` (3 filings).
- The draw can pick a stale filer. Autoscope's last filing here is from 2022, as with United Bancshares.

### Table figures on these filings

| Check | Result |
|---|---|
| Tagged cells, value and period end | 100% of 2,441 |
| Same row, period and duration in two filings agree (untagged) | 98.0% of 99 |
| Hand audit of 36 searchable facts, no dashes | 35 of 36 right |
| Data tables that parsed cleanly | 21.2% of 1,173 |

**The wrong fact.** AT&T's Q3 2025 10-Q reports 19,200 thousand prepaid subscribers, and the store
has $19.2 billion. The parser doesn't read the header "(in 000s)" as a scale, so it used the page
banner "Dollars in millions". All 22 untagged facts in that filing's 3 tables of this shape are
wrong. No other filing in the 115 stores facts from a table with that header. The answer key can't
see this, because those cells have no tags.

**Traced signals.** Two untagged values matched XBRL only for another period, and both were
coincidences: a lease payment and treasury stock. AT&T's 2 disagreements between filings are one
row, for a quarter and for nine months, likely recast with its new segments, though that isn't proven.
Comstock has 0 table facts: all 209 of its tables wait for the chunker, mostly for lack of a
stated scale.

### What three tests show

| | First | Second | Third |
|---|---|---|---|
| Filings accepted | 18 of 25 (72%) | 17 of 25 (68%) | 24 of 25 (96%) |
| Accepted with a materially wrong section | 0 of 18 | 0 of 17 | 4 of 24 |
| Blind tagged-cell accuracy | 99.23% | 99.44% | 100% |

- **The rate is too noisy to show improvement.** The rules improved between tests, but so did the
  draw's luck: this one seems to have fewer unusual layouts, with the fallback used on 3 of 25
  filings, against 14 of 25 in the first.
- **The gate doesn't always fail safe.** The earlier claim rested on 0 of 35, and it no longer
  holds. Across all three tests, 4 of 59 accepted filings had a wrong section (2% to 17%), from
  two companies. The rules changed along the way, so this is rough.

**Re-scored comparison.** `cross_filing_agreement.py` now also requires the same duration, because
the older comparison could match a three-month figure to a nine-month one. On current rules it gives
development 98.8% of 859, first test 100% of 54, second 95.3% of 129, and third 98.0% of 99. The
tables above keep the numbers as measured at the time.

### After the third test

These filings shaped the fixes below, so the third test's rates describe the rules it tested, and
no fair rate exists for the current ones. The thresholds were set on all 115 filings.

| New check | What it looks for | On 115 filings, both parsers |
|---|---|---|
| Foreign headings in market risk | Audit or management reports under market risk | Flags AT&T's 2 10-Ks, on both parsers |
| Signatures (since removed) | A section that opens with other text or holds note headings | Flags Everus's 2 10-Ks, on `edgartools` |
| Note numbering | A skipped number, the next note in another item, or a 10-K citing a later note | Flags Everus's 2 10-Ks, plus a United Bancshares 10-K already rejected |
| Section ending | More than 2,000 characters of prose before the next item, not parsed elsewhere | Flags Everus's 2 10-Ks at about 21,700 characters. The most anywhere else is 296 |

Results: Everus's 10-Ks now use the fallback and keep Notes 12 to 15. AT&T's 10-Ks are rejected,
because both parsers misfile the reports, and `ingest` now removes a rejected filing's facts from
the store. No other filing changed route. A colon at a section's end isn't used as evidence: it
would have flagged about 14 sections, nearly all ending just before a removed table.

**Limits.** Note numbering can't read lettered notes (Stanley, Autoscope) or unnumbered ones. The
ending check needs the expected next item heading, so it says nothing on layouts without one (GE,
JPMorgan's 10-Q). It also misses a loss under 2,000 characters.

**Hershey, Mattel, Stanley.** All 15 filings pass, with no section affected. Hershey's and Mattel's
note numbers run complete, and Stanley's letters run to the last in the filing (S in the 10-Ks).
Real-filing tests now pin these endings. Hershey's notes do hold noise the gate allows: the banner
"NOTES TO CONSOLIDATED FINANCIAL STATEMENTS—(Continued)" is its own paragraph (42 of 309 in the
FY2025 10-K), and it can split a sentence across a page break.

## Fourth held-out test: current rules

Measured on 2026-11-04 with the scripts in [ingestion/](ingestion/README.md). Same method with a new
seed (20261106), excluding every company used so far. Three draws were skipped because they file no
10-K or 10-Q: GEMZ, Linkage Global and PMA Graphene. Selected: Jabil, CapForce, United Natural Foods,
First America Resources and Regis, for 25 filings in `config/parser_holdout4.yaml`. The rule files
matched `ingestion/holdout4.sha256` before the draw and after the run, and nothing found here has
been fixed.

These are the only fair numbers for the current rules. They include the changes made just before
the draw: the three checks and the ending check above, the removal of two gate checks, and the
scale fix below.

| Measure | Result | 95% interval |
|---|---|---|
| Filings accepted by the gate | 17 of 25 (68%) | 46% to 85% |
| Companies with every filing accepted | 3 of 5 | 15% to 95% |
| Accepted filings with a materially wrong section | 0 of 17 | 0% to 20% |
| Accepted and with no material error | 17 of 25 (68%) | 46% to 85% |
| Filings where the fallback ran | 11 of 25 (44%) | 24% to 65% |
| Fallback succeeded | 3 of 25 (12%) | 3% to 31% |
| Both parsers failed | 8 of 25 (32%) | 15% to 54% |

**How the 17 were scored.** Every section's first and last paragraphs were read, every parser
disagreement was traced, and each filing's note headings were checked against the raw filing. All
note bodies are in the section. This can't rule out a loss in the middle of a non-note section.

**The 8 rejections look like false alarms, so they cost coverage and not accuracy.**

| Filings | Why | Reading |
|---|---|---|
| CapForce: 2 10-Ks and 1 10-Q | A row label in the equity statement reads as a sentence, "Reclassification of preferred stock par value... see Note 3", and is removed with its figures | Same cause as United Bancshares. The notes run through the last one |
| First America Resources: all 5 | Sections are below length floors set for much larger companies, and "Not required for Smaller Reporting Companies" counts as a short risk factors section | The sections read correctly, except that the FY2024 10-K's risk factors parsed empty. The filing there says only "Not required for Smaller Reporting Companies" |

**Smaller problems, none material.**
- Jabil's 10-Qs (3) end the notes with a forward-looking statement that the filer places on its own page before the Item 2 heading.
- Jabil's 10-Ks (2) begin the notes with the Item 15 document list, because the statements are in Item 15.
- Regis's filings (5) use rule lines as separators, so 83 to 117 paragraphs in each have a heading of underscores. Regis's 10-Ks also keep the "(Continued)" banner, joined to the next sentence.
- CapForce's 10-Qs (2) label "not required" market risk as `content`.

### Table figures on these filings

| Check | Result |
|---|---|
| Tagged cells, value | 100% of 2,938 |
| Tagged cells, period end | 99.80% (6 wrong) |
| Same row, period and duration in two filings agree (untagged) | 97.3% of 74 |
| Hand audit of 27 searchable facts, no dashes | 27 of 27 right (87% to 100%) |
| Data tables that parsed cleanly | 25.1% of 1,013 |

The audit could only sample three companies: CapForce has no table facts, and First America's
filings were all rejected. No clean table in these filings uses shorthand for its scale, so this
test doesn't exercise the scale fix on new data.

**Traced signals.**
- The 6 period errors are 2 interest-rate cells in each of 3 Regis 10-Qs. The filer's column says "Fiscal Year 2026", but the tag says June 30, 2025. Comparing with the tags marked those tables not clean, so none of these figures are stored as facts.
- Jabil's 2 disagreements are SG&A for 2024 and 2025 ($1,071 against $1,160 million, and $1,015 against $1,122 million). Both 10-Ks show the figures as parsed, so this looks like a recast, though that isn't proven.

### What four tests show

| | First | Second | Third | Fourth |
|---|---|---|---|---|
| Filings accepted | 18 of 25 (72%) | 17 of 25 (68%) | 24 of 25 (96%) | 17 of 25 (68%) |
| Accepted with a materially wrong section | 0 of 18 | 0 of 17 | 4 of 24 | 0 of 17 |
| Blind tagged-cell value accuracy | 99.23% | 99.44% | 100% | 100% |

- **The acceptance rate still isn't showing improvement.** It moves with the draw, and tiny filers drive much of it.
- **Zero wrong sections here is weak evidence.** The interval runs to 20%, and the wrong sections in the third test came from layouts this draw may simply not contain.
- **The largest remaining cost is false rejection,** from row-label sentences and from length floors that don't fit small companies.

Fallback use under the current rules, with the current checks:

| Filings | Fallback ran | It succeeded | Both failed |
|---|---|---|---|
| 100 random draws, from 20 companies | 36 (36%, 27% to 46%) | 19 | 17 |
| All 140 | 47 (34%, 26% to 42%) | 30 | 17 |
| 10-Ks in the draws | 19 of 40 (48%) | | |
| 10-Qs in the draws | 17 of 60 (28%) | | |

The random-draw figures describe rules partly built from those filings, apart from the 25 above.

## Scale fix and removed checks

**Scale.** AT&T's Q3 2025 10-Q labels subscriber tables "(in 000s)". The parser couldn't read that,
so it used the page banner "Dollars in millions" and stored 90,255 thousand subscribers as $90
billion. It now reads shorthand as a scale: 000s, 000's, (000), $000, (000,000), MM, bn and $M.
Those tables state no dollars, so they now go to the chunker as text. Shorthand attached to one
quantity, such as "Shares (000s)", marks the table not clean, since it says nothing about the rest.

Across 6,620 tables in the 115 earlier filings, exactly 3 changed from clean to not clean, removing
the 22 wrong facts. No other table's facts changed, and 55 tables that were already not clean have
a more specific reason. Tagged-cell accuracy can't confirm the fix, because these cells have no tags.

**Removed gate checks.** Both removals changed no decision on the 115 filings.

| Check | Why it could go | What's lost |
|---|---|---|
| Short item holds main content | It never failed in 230 parses | The only check for a long item that absorbs MD&A or statements, as Chevron's Item 14 did before tables were stripped row by row |
| Signatures | It fired only on Everus, where numbering and the ending check also fire, and the ending check counts signature text as left out | A note filed under signatures with under 2,000 characters left out, on a filer without numbered notes |

Eight redundant tests were also removed, along with the tests of these two checks.
