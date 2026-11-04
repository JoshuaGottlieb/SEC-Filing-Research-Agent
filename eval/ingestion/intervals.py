"""Print exact 95% intervals, for results from small samples.

python eval/ingestion/intervals.py 17/25 3/5 0/17
"""

from __future__ import annotations

import typer
from _common import exact_interval


def main(results: list[str]) -> None:
    # Each argument is "successes/trials", for example 17/25.
    for result in results:
        successes, trials = (int(part) for part in result.split("/"))
        low, high = exact_interval(successes, trials)
        typer.echo(f"{result}: {successes / trials:.0%}, 95% interval {low:.1%} to {high:.1%}")


if __name__ == "__main__":
    typer.run(main)
