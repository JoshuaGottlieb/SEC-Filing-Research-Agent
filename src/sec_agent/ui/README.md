# The Streamlit app

`sec-agent app` opens a Streamlit app over saved runs. It reads each run's `audit.json`, `ledger.jsonl` and `timings.jsonl` and never calls a model, so browsing is free.

```bash
uv run --frozen sec-agent app             # opens http://localhost:8501; --port to change it
```

It shows the runs in `<data dir>/runs/`. Pick a company, a comparison and a run; the newest is the default.

To browse the committed sample runs, point the data directory at them: `SEC_AGENT_DATA_DIR=samples uv run --frozen sec-agent app`. Without a full setup, use the light install in [samples/README.md](../../../samples/README.md).

## The five tabs

- **Brief:** the summary and each change with its labeled claims. Filter by strongest label and by section. The changed text shows both versions side by side, with the words that differ marked. Expand a quote to see the full passage it came from, with the quote highlighted, and a link to the filing on SEC.gov.
- **Evidence trees:** one change's Tree-of-Thought tree, drawn top-down: the change, then one card per candidate explanation with its score and label.
  - A card's evidence searches show as a row of coloured chips, one per search: the colour is what the search found, solid if kept, hollow if pruned.
  - Open a candidate's branch to see each search as its own box, stacked under it.
  - Pruned branches are dashed, with the reason on hover and in the node table, and can be hidden.
  - Candidates that reached the brief are outlined in their label's colour.
  - Below the tree, pick a candidate to see its searches and the quotes they found.
- **Replay audit:** an animated replay of the run, step by step:
  - detecting the changes, with paragraphs from this run aligned and their changed sentences marked;
  - ranking them: the changes explored, then some of those past the cap;
  - for each change, the candidates proposed and pre-screened, each candidate's evidence searches, labeling and rework rounds;
  - the summary.

  The tree grows as the run built it, with the run's clock, running cost and model calls, and a note on what each step does. Play, pause, step, change the speed, scrub, or jump to a change. It is built from the saved timings, ledger and audit trail; runs saved before timings were recorded replay without a clock.
- **Review queue:** what to check by hand. Findings that need review, explanations left out of the brief or annotated by labeling (with the reason and the evidence found), searches that did not finish, rationales left out, and the changes past the `--max-changes` cap. Nothing in it stopped the run.
- **Cost and time:** the run's cost, wall time and calls by step and model, as `sec-agent eval cost` reports them.

## Limits

- The app shows saved runs only. Starting a run from the app is a future enhancement and is not currently included; start runs with `sec-agent analyze`.
- The full passages come from the DuckDB store, opened read-only. While `ingest` or `index` holds it for writing, the app shows the quotes alone. The light install on the samples has no store, so it shows the quotes alone too.

## The code

- `streamlit_app.py`: the app and its tabs.
- `replay_page.py`, `replay.html`: the animated replay page, with a run's data filled in.
- The data behind each view comes from `report/run_view.py`, `report/replay.py` and `report/run_cost.py`, which hold no Streamlit code, so tests check them directly.
