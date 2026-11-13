# Benchmark labels (T6.1)

Hand-confirmed labels for the evaluations in Phase 6. The record types and the quote resolver are in
`src/sec_agent/benchmark.py`; `check.py` validates a label file and `draw_sample.py` draws the
false-positive sample.

## Splits

Hershey (`HSY`) is the **tuning** set: thresholds such as `MIN_SCORE`, `MIN_MATCH_SIMILARITY`,
`MIN_SEASONAL_SHARE`, the scan's flag levels and the materiality rubric may be adjusted against it.
Mattel (`MAT`) and Stanley Black & Decker (`SWK`) are the **test** set: they give the reported
numbers and must not be used to tune anything. With a set this small the results are directional.

## What is labeled

| Kind | Per company | Used by |
|---|---|---|
| Retrieval questions | 10 (8 answered in both filings, 2 where the topic is absent from one) | T6.2 recall@5 with and without the reranker, `MIN_SCORE`, T2.4 redundancy |
| Material changes | 8 | T6.3 missed changes |
| Stated causes | 3 | T6.4 known-cause survival in the beam, T6.5 |
| Judge examples | 9 (one per verdict for each stated cause) | T5.2 agreement check |
| Sampled changes | 14 HSY, 13 MAT, 13 SWK | T6.3 false positives |

Each company has three comparisons: `yoy-10k` (FY2025 10-K against FY2024), `yoy-10q` (Q2 2026
10-Q against Q2 2025) and `seq-10q` (Q2 2026 against Q1 2026). Items are spread across all three.

## How the drafts were made

Drafted by Claude (Claude Code, Opus 5.5) from the **indexed chunk text** of the pinned filings,
read section by section straight from the chunk store. None of the agent's retrieval, alignment,
materiality, Tree-of-Thought or verification code was run to produce a draft, so a label cannot
inherit the agent's mistakes. Two consequences to keep in mind:

- Labels can only point at text the index holds. Text the parser lost is out of reach for both the
  agent and the labels; ingestion is measured separately (`docs/How_SEC_Filing_Ingestion_Is_Tested.md`).
- The drafter is the same model family the agent uses for materiality and the judge, so the two
  could share blind spots. The user's confirmation of every label is the check on that, and the
  review log records every edit and rejection.

The **sampled changes** are the exception: they are T3.1 alignment output by design, drawn by
`draw_sample.py` with seed `20261111` (fixed on 2026-11-11, before any label was made). Alignment
decides only which paragraphs pair up, never whether a change matters.

### Rules the drafts follow

- **Quotes are verbatim** from one chunk of the named filing (`side` is `current` or `prior` in the
  comparison). A quote is a sentence or clause long enough to be unique, not a whole paragraph.
  `check.py` confirms each quote appears in its filing using the citation checker's matching.
- **Retrieval questions** read like the searches the agent issues: short, specific, in an analyst's
  words, not copied from the answer ("drivers of the gross margin decline", "LesserEvil acquisition
  price"). `quotes` lists every passage in each filing that answers the question, one quote per
  passage. `section` is the filter a caller would pass, or omitted for the whole filing. For an
  absent-topic question, the topic appears in one filing only, and the other side has no quote.
  Questions avoid pure table figures, which are not in chunk text.
- **Material changes** meet one of the three criteria (`new_or_removed_risk`, `changed_quantity`,
  `metric_linked_shift`) and would matter to a fundamental analyst. Excluded: period roll-forwards
  ("2024" becoming "2025" with nothing else changed), reworded boilerplate and typography.
  `modified` quotes both versions; `added` only the current, `removed` only the prior.
- **Stated causes** are changes whose cause the current filing states in so many words ("driven
  by", "due to", "as a result of"). `causes` lists each stated cause briefly; `distractors` lists two
  or three plausible causes the filings do not state for this change. Some distractors are
  contradicted by the filing rather than merely unstated; the item's note says so. 10-Qs explain
  changes against the same quarter a year earlier, never the previous quarter, so `seq-10q` has no
  stated causes: the filings give the agent nothing to confirm there, and an explanation of a
  sequential move can at best be `plausible`.
- **Judge examples**, three per stated cause, follow the judge's rule (T5.2): the quote **states the
  cause** when it names the claimed cause or a particular case of it; it only **implies** it when it
  names a broader cause, or supports the claim without naming a cause; it is **unrelated** when it
  does not bear on the claim. The `implies` and `unrelated` quotes come from the same filings.

## Review and freezing

The user confirms every label on a review page (accept, edit or reject, with a note). For a
retrieval question the scored unit is the chunk a quote locates, not the quote itself: a question
whose chunk answers it is accepted even when the highlighted sentence does not, and the note names
the answering sentence, which replaces the quote at freezing so the anchor cannot also select a
neighbouring chunk through the ~10% overlap. Confirmed
items, with edits applied, are written to `HSY.json` (tuning) and `test.json` (Mattel and
Stanley); every decision, including rejections, goes to `review_log.json`. Once frozen, a label
file changes only to fix a quote that a later re-parse moved, and the log records why.

`freeze.py` does the freezing from `drafts/` and `review_log.json`, and rerunning it gives the same
files. It refuses a draft whose hash differs from the one recorded at review.

### Outcome (frozen 2026-11-12)

All 130 items were reviewed and none was edited. Three were rejected:

- `HSY-C03`: an AI risk factor too general to count as a material change.
- `HSY-R03` and `SWK-R06`: risk factors that only speculate (AI regulation, the share repurchase
  program) rather than say anything substantive.

Rejected items were not replaced, because a replacement would need its own review. Both rejected
questions were absent-topic questions, so Hershey and Stanley each keep one.

| File | Retrieval (absent-topic) | Changes | Causes | Judge | Sampled (material) |
|---|---|---|---|---|---|
| `HSY.json` (tune) | 9 (1) | 7 | 3 | 9 | 14 (3) |
| `test.json` (MAT + SWK) | 19 (3) | 16 | 6 | 18 | 26 (5) |

In five Hershey retrieval questions (`R01`, `R02`, `R04`, `R05`, `R08`), the highlighted sentence
did not answer the question, or answered only part of it, but its chunk did. The reviewer's
answering sentence replaced the quote, and each replacement locates the same single chunk.

PR #42 (exact ties in alignment broken by position) changed none of the 40 sampled changes.

#### Redraw after the seq-10q pairing fix

The first review of `HSY-F14` showed that alignment paired the Q1 2026 North America
Confectionery net sales paragraph with Q2's six-month paragraph rather than the quarterly one. A Q1
10-Q has one paragraph per segment where a Q2 10-Q has two (quarter and year to date). PR #45
fixed this pairing for comparisons against a Q1 10-Q. The sample was then redrawn with the same seed
and counts. It still covers 1,904 changes; 38 of 40 picks were unchanged, and `HSY-F14` and `MAT-F07`
were replaced and reviewed again. Both new changes are not material. `review_log.json` keeps the
earlier decisions under `superseded`, with the change each one referred to.

The reviewer's reason applies to `seq-10q` generally. These MD&A paragraphs compare each period with
the same period a year earlier, so a Q2 paragraph set against a Q1 paragraph compares two
year-over-year statements about different periods. The filings hold no quarter-over-quarter
figure for these items. Treating such a pair as a sequential change is a gap in the evidence,
which falls to retrieval and change analysis, not to alignment. This matches the finding that
10-Qs state no causes for sequential moves.

#### Label fixes after the judge agreement check (2026-11-12)

The first T5.2 agreement run on Hershey disagreed with two `states_cause` labels, `HSY-J01` and
`HSY-J07`. Each quote began "The decrease (increase) was driven by…" without naming what changed;
the subject was in the sentence before. Read alone, such a quote only implies the claim's cause,
and the judge said so. With the user's approval, each quote now starts with the preceding sentence
from the same chunk (gross margin for `J01`, SM&A for `J07`). Claims and verdicts are unchanged.
`review_log.json` records both under `label_fixes`, and `freeze.py` applies and checks them. No test
quote had this problem: every Mattel and Stanley `states_cause` quote names the metric it explains.
