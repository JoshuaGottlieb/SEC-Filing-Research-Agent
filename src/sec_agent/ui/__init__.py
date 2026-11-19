"""The Streamlit app, started by `sec-agent app` (T7.1 to T7.3).

It shows saved runs only: the brief, evidence trees, a step-by-step replay, the review queue
(design gap 12) and the cost and time. It never calls a model. The data behind each view comes
from `report/`.

- `streamlit_app.py`: the app.
- `replay_page.py`, `replay.html`: the animated replay page with a run's data filled in.
"""
