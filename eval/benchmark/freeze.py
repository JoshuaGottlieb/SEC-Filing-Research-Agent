"""Freeze the reviewed labels: apply `review_log.json` to the drafts and write the label files.

Run from the repository root:

    uv run --frozen python eval/benchmark/freeze.py

Reads `drafts/{HSY,MAT,SWK,sampled}.json` and `review_log.json`, and writes `HSY.json` (tuning)
and `test.json` (Mattel and Stanley). Rerunning it gives the same files. It stops without writing
if a draft no longer matches the hash recorded when it was reviewed, if an item has no decision or
a decision names no item, or if a swapped quote no longer locates the chunk the reviewer accepted.

What the review changes:

- A rejected item is dropped. Accepted items keep the draft's fields, since no item was edited.
- A sampled change takes the reviewer's material / not material label.
- The reviewer's note is appended to the item's note, marked "Reviewer:".
- Retrieval quotes in `QUOTE_SWAPS` are replaced by the sentence the reviewer named. The scored
  unit is the chunk, which the swap keeps (it is checked), so only the anchor text changes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from sec_agent.benchmark import (
    Benchmark,
    Item,
    Quote,
    RetrievalQuestion,
    SampledChange,
    Split,
    load_benchmark,
    resolve_quote,
    save_benchmark,
)
from sec_agent.config import get_settings
from sec_agent.edgar.corpus import Corpus
from sec_agent.finance.fact_store import FactStore

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DRAFTS = ("HSY.json", "MAT.json", "SWK.json", "sampled.json")
KINDS = ("retrieval", "changes", "causes", "judge", "sampled")

# Retrieval questions whose highlighted sentence did not answer (or answered only part of) the
# question while its chunk did. Each entry names the quote by the chunk the reviewer accepted and
# gives the answering sentence from the reviewer's note.
QUOTE_SWAPS: dict[str, tuple[str, str]] = {
    "HSY-R01": (
        "0001628280-26-008586:mdna:0034:0",
        "The decrease was driven by higher commodity and tariff costs, unfavorable mark-to-market "
        "activity on our commodity derivative instruments and lower volume",
    ),
    "HSY-R02": (
        "0001628280-26-008586:market_risk:0023:0",
        "Higher cocoa prices to the consumer have lowered consumption and a second consecutive "
        "surplus is expected in the 2025 – 2026 growing season.",
    ),
    "HSY-R04": (
        "0001628280-26-008586:business:0033:0",
        "Collective bargaining agreements covered approximately 5,570 employees, or approximately "
        "28% of the Company’s employees worldwide.",
    ),
    "HSY-R05": (
        "0001628280-26-050900:mdna:0014:0",
        "Net sales were $2,787.3 million in the second quarter of 2026 compared to $2,614.7 "
        "million in the same period of 2025, an increase of $172.6 million, or 6.6%.",
    ),
    "HSY-R08": (
        "0001628280-26-050900:financial_notes:0097:0",
        "The 2025 effective tax rate was higher due to the impact of tax reserves and foreign rate "
        "differentials related to mark-to-market activity.",
    ),
}


def load_drafts(log: dict) -> Benchmark:
    merged: dict[str, list] = {kind: [] for kind in KINDS}
    for name in DRAFTS:
        path = HERE / "drafts" / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != log["drafts_sha256"][name]:
            raise SystemExit(f"{path} changed after review (sha256 {digest[:12]})")
        draft = load_benchmark(path)
        for kind in KINDS:
            merged[kind].extend(getattr(draft, kind))
    return Benchmark(**merged)


def with_note(item: Item, note: str | None) -> dict:
    if not note:
        return {}
    joined = f"{item.note}\nReviewer: {note}" if item.note else f"Reviewer: {note}"
    return {"note": joined}


def swap_quote(question: RetrievalQuestion, chunk_id: str, text: str) -> RetrievalQuestion:
    targets = [quote for quote in question.quotes if quote.chunk_id == chunk_id]
    if len(targets) != 1:
        raise SystemExit(f"{question.item_id}: no single quote on {chunk_id}")
    old = targets[0]
    new = Quote(side=old.side, text=text, chunk_id=chunk_id)
    quotes = tuple(new if quote is old else quote for quote in question.quotes)
    return question.model_copy(update={"quotes": quotes})


def apply(drafts: Benchmark, log: dict) -> Benchmark:
    decisions = {entry["item_id"]: entry for entry in log["decisions"]}
    ids = {item.item_id for item in drafts.items()}
    if ids != set(decisions):
        missing, unknown = sorted(ids - set(decisions)), sorted(set(decisions) - ids)
        raise SystemExit(f"no decision for {missing}; decision for unknown items {unknown}")
    frozen: dict[str, list] = {kind: [] for kind in KINDS}
    for kind in KINDS:
        for item in getattr(drafts, kind):
            decision = decisions[item.item_id]
            if decision["decision"] == "reject":
                continue
            update = with_note(item, decision.get("note"))
            if isinstance(item, SampledChange):
                update["material"] = decision["material"]
            item = item.model_copy(update=update)
            if item.item_id in QUOTE_SWAPS:
                item = swap_quote(item, *QUOTE_SWAPS[item.item_id])
            frozen[kind].append(item)
    return Benchmark(**frozen)


def check_swaps(benchmark: Benchmark) -> None:
    """Each swapped sentence must locate exactly the chunk the reviewer accepted."""
    corpus = Corpus.load(REPO_ROOT / "config" / "corpus.yaml")
    comparisons = {c.comparison_id: c for co in corpus.companies for c in co.comparisons()}
    questions = {q.item_id: q for q in benchmark.retrieval}
    with FactStore(get_settings().duckdb_path, read_only=True) as store:
        for item_id, (chunk_id, text) in QUOTE_SWAPS.items():
            question = questions[item_id]
            quote = next(q for q in question.quotes if q.text == text)
            found = resolve_quote(
                quote, comparisons[question.comparison_id], store.chunks, question.section
            )
            if found != (chunk_id,):
                raise SystemExit(f"{item_id}: swapped quote locates {found}, not {chunk_id}")


def split(benchmark: Benchmark, which: Split) -> Benchmark:
    return Benchmark(
        **{
            kind: tuple(item for item in getattr(benchmark, kind) if item.split is which)
            for kind in KINDS
        }
    )


def main() -> int:
    log = json.loads((HERE / "review_log.json").read_text())
    frozen = apply(load_drafts(log), log)
    check_swaps(frozen)
    for name, which in (("HSY.json", Split.TUNE), ("test.json", Split.TEST)):
        part = split(frozen, which)
        save_benchmark(part, HERE / name)
        counts = ", ".join(f"{len(getattr(part, kind))} {kind}" for kind in KINDS)
        print(f"{name}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
