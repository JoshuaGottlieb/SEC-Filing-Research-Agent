# Sample runs

These are the agent's real outputs: nine full runs, three companies by three comparisons, saved exactly as the pipeline wrote them. You can read them on GitHub, or browse them in the Streamlit app with a light install: no ingestion, no models to download, no API key and no paid calls.

They come from the full-corpus batch of 2026-11-15 to 16. Each run explored its 10 most material changes under a $1.50 cap. Hershey (HSY) is the company the settings were tuned on. Mattel (MAT) and Stanley Black & Decker (SWK) are the held-out test set, run once (see [EVALUATION.md](../EVALUATION.md)).

| Run | Comparison | Periods | Confirmed | Plausible | Cost | Time |
|---|---|---|---|---|---|---|
| [HSY-yoy-10k](runs/HSY-yoy-10k-t81-20261115/brief.md) | year over year (10-K) | FY2025 against FY2024 | 18 | 12 | $0.86 | 33 min |
| [HSY-yoy-10q](runs/HSY-yoy-10q-t81-20261115/brief.md) | year over year (10-Q) | Q2 2026 against Q2 2025 | 14 | 17 | $0.55 | 22 min |
| [HSY-seq-10q](runs/HSY-seq-10q-t81-20261115/brief.md) | sequential quarters (10-Q) | Q2 2026 against Q1 2026 | 18 | 14 | $0.68 | 31 min |
| [MAT-yoy-10k](runs/MAT-yoy-10k-t81-20261115/brief.md) | year over year (10-K) | FY2025 against FY2024 | 11 | 25 | $0.85 | 41 min |
| [MAT-yoy-10q](runs/MAT-yoy-10q-t81-20261115/brief.md) | year over year (10-Q) | Q2 2026 against Q2 2025 | 16 | 21 | $0.89 | 36 min |
| [MAT-seq-10q](runs/MAT-seq-10q-t81-20261115/brief.md) | sequential quarters (10-Q) | Q2 2026 against Q1 2026 | 2 | 21 | $0.57 | 30 min |
| [SWK-yoy-10k](runs/SWK-yoy-10k-t81-20261115/brief.md) | year over year (10-K) | FY2025 against FY2024 | 13 | 18 | $0.88 | 35 min |
| [SWK-yoy-10q](runs/SWK-yoy-10q-t81-20261115/brief.md) | year over year (10-Q) | Q2 2026 against Q2 2025 | 15 | 23 | $1.04 | 42 min |
| [SWK-seq-10q](runs/SWK-seq-10q-t81-20261115/brief.md) | sequential quarters (10-Q) | Q2 2026 against Q1 2026 | 12 | 26 | $1.29 | 47 min |

Confirmed and plausible count the claims in each brief. Cost is model spend; time is wall-clock time on a 4-core laptop with no GPU.

Each run folder holds:

- `brief.md`: the research brief, the end product. Every claim carries its quote and SEC link.
- `audit.json`: the full record behind it: every claim, quote, metric and Tree-of-Thought tree (pruned branches included), the changes not examined, and the review notes.
- `ledger.jsonl`: one line per model call, with its role, model, tokens and cost.
- `timings.jsonl`: one line per step, with its duration.

## Read a brief on GitHub

Click a run in the table. GitHub renders `brief.md` directly. Start with [HSY-yoy-10k](runs/HSY-yoy-10k-t81-20261115/brief.md), the run the demo video uses.

## Browse the runs in the app (about 5 minutes, no API key)

The app shows each run's brief, its evidence trees, a step-by-step replay, the review queue and the cost and time. It reads only the files in this folder and never calls a model.

You need Python 3.12 and git. The install is about 730 MB, mostly Streamlit and pandas. The full setup in the main README needs about 2.5 GB and a 15-minute indexing step.

**1. Fetch only the folders the app needs** (`src/`, `config/` and `samples/`, about 17 MB):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/JoshuaGottlieb/SEC-Filing-Research-Agent.git
cd SEC-Filing-Research-Agent
git sparse-checkout set src config samples
```

No git? Download the repository as a ZIP from its GitHub page (Code, then Download ZIP) and unzip it. It is small; only the full setup is heavy.

**2. Install the viewer's packages** in a fresh virtual environment:

```bash
python3.12 -m venv .venv-viewer
source .venv-viewer/bin/activate            # Windows: .venv-viewer\Scripts\activate
pip install -r samples/requirements-viewer.txt
```

**3. Start the app on the samples:**

```bash
SEC_AGENT_DATA_DIR=samples PYTHONPATH=src streamlit run src/sec_agent/ui/streamlit_app.py
```

On Windows (PowerShell):

```powershell
$env:SEC_AGENT_DATA_DIR = "samples"; $env:PYTHONPATH = "src"
streamlit run src/sec_agent/ui/streamlit_app.py
```

The app opens at <http://localhost:8501> on the first company and comparison in alphabetical order: Hershey, sequential quarters. To see the run the demo video uses, set **Comparison** in the sidebar to **Year over year (10-K)**. The **Company** and **Comparison** menus switch between the other runs.

**What to look at:**

- **Brief:** the summary, then each change with its claims. Open a claim to see its quote and the SEC link.
- **Evidence trees:** each change's Tree-of-Thought search. Dashed cards are pruned explanations; hover a box for its text and score.
- **Replay audit:** the run step by step, rebuilt from its timings, ledger and audit trail. Press play.
- **Review queue:** everything left for a person: changes past the 10-change cap, explanations left out of the brief with their evidence, and searches that did not finish.
- **Cost and time:** spend and time by step, by change and by model.

**One difference from a full setup:** a claim shows only its quote, not the full passage around it. The passages live in the chunk store, which only ingestion builds. With a full setup, copy these run folders into your data directory's `runs/` folder and `sec-agent app` shows the passages too.
