"""T6.5: the user's blinded hand check of asserted claims (free: no model calls).

Run from the repository root, on a `score.py` output scored with the judge (not `--free`):

    uv run --frozen --no-sync python eval/baseline/hand_check.py sample scores.json sheet.json
    uv run --frozen --no-sync python eval/baseline/hand_check.py tally sheet.json answers.json

The unsupported-claims rate rests on the T5.2 judge, which the agent's confirmed claims already
passed, so on that measure the agent scores well partly by construction. This is the check that
does not: the user (the user's decision, 2026-11-15) rates about 15 asserted claims from each
system, mixed and with the source hidden, against the claim's valid quotes.

`sample` draws claims that the judge settled with a valid quote to read (claims with no valid
quote are unsupported whatever anyone reads), with a fixed seed, and writes the sheet: an id, the
change, the claim and its quotes, with no system, label or verdict. The key goes beside it in
`<sheet>.key.json`; the user should not open it. Claims are shuffled together, and each claim's
wording is shown as written: an agent claim can read differently from a baseline one, which the
check cannot hide.

`tally` reads the user's answers, `{"H01": "yes", "H02": "no", ...}`: "yes" when a quote states
that the claim's cause produced this change, "no" otherwise. The question is the judge's, plus
"this change": the judge sees only the claim and the quote. It reports, for each system, how
many claims the user found supported and how often the user and the judge agree.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Fixed before any claim was seen (2026-11-15), like the benchmark's sample seed.
SEED = 20261115
PER_SYSTEM = 15
ANSWERS = ("yes", "no")


def sample(scores: dict, per_system: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    picked = []
    for system in ("agent", "baseline"):
        pool = [
            claim
            for claim in scores["claims"]
            if claim["system"] == system
            and claim["asserted"]
            and claim["supported"] is not None
            and any(_valid(citation) for citation in claim["citations"])
        ]
        # Sorted first so the draw depends only on the seed, not on file order.
        pool.sort(key=lambda claim: claim["claim_id"])
        picked += rng.sample(pool, min(per_system, len(pool)))
    rng.shuffle(picked)
    sheet, key = [], {}
    for n, claim in enumerate(picked, 1):
        item_id = f"H{n:02d}"
        sheet.append(
            {
                "id": item_id,
                "comparison": claim["comparison_id"],
                "change": _change(claim),
                "claim": claim["text"],
                "quotes": [
                    {"filing": citation["side"], "section": citation["section"], "quote": q}
                    for citation in claim["citations"]
                    if _valid(citation) and (q := citation["quote"])
                ],
            }
        )
        key[item_id] = {
            "system": claim["system"],
            "claim_id": claim["claim_id"],
            "judge_supported": claim["supported"],
        }
    return sheet, key


def _change(claim: dict) -> str:
    """The change as one sentence for either system. The agent's heading names only a text
    change's kind and section ("Changed paragraph in Management's Discussion and Analysis"),
    which would mark its claims as the agent's, so its materiality rationale is shown instead,
    the counterpart of the baseline's description. `score.py` saves it in `coverage_text` as
    "<heading>: <rationale>"."""
    heading, described = claim["change"], claim.get("coverage_text")
    prefix = f"{heading}: "
    if described and described.startswith(prefix):
        return described[len(prefix) :]
    return heading


def _valid(citation: dict) -> bool:
    return citation["verbatim"] and citation["metadata_ok"] and not citation["wrong_period"]


def tally(key: dict, answers: dict) -> dict:
    missing = sorted(set(key) - set(answers))
    bad = sorted(item for item, answer in answers.items() if answer not in ANSWERS)
    if missing or bad:
        raise SystemExit(f"missing answers: {missing}; answers not yes/no: {bad}")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item_id, entry in key.items():
        row = counts[entry["system"]]
        human = answers[item_id] == "yes"
        row["rated"] += 1
        row["user: supported"] += human
        row["judge: supported"] += entry["judge_supported"]
        row["user and judge agree"] += human == entry["judge_supported"]
        row["judge supported, user did not"] += entry["judge_supported"] and not human
        row["user supported, judge did not"] += human and not entry["judge_supported"]
    return {system: dict(row) for system, row in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    draw = commands.add_parser("sample")
    draw.add_argument("scores", type=Path)
    draw.add_argument("sheet", type=Path)
    draw.add_argument("--per-system", type=int, default=PER_SYSTEM)
    count = commands.add_parser("tally")
    count.add_argument("sheet", type=Path)
    count.add_argument("answers", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        scores = json.loads(args.scores.read_text())
        if scores.get("free_only"):
            raise SystemExit("score with the judge first: a --free run settles too few claims")
        sheet, key = sample(scores, args.per_system, SEED)
        args.sheet.write_text(json.dumps(sheet, indent=2, ensure_ascii=False) + "\n")
        key_path = args.sheet.with_suffix(".key.json")
        key_path.write_text(json.dumps(key, indent=2) + "\n")
        print(f"{len(sheet)} claims -> {args.sheet} (key: {key_path})")
    else:
        key = json.loads(args.sheet.with_suffix(".key.json").read_text())
        answers = json.loads(args.answers.read_text())
        print(json.dumps(tally(key, answers), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
