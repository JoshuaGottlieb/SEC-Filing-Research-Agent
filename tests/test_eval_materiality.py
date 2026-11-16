"""T6.3's materiality scorer (`eval/materiality/materiality.py`), with no model calls."""

from __future__ import annotations

import sys

from sec_agent.benchmark import Benchmark, LabeledChange, Quote, SampledChange, Side
from sec_agent.compare.materiality import MaterialityDecision
from sec_agent.config import REPO_ROOT
from sec_agent.schemas import ChangeKind, MaterialityCriterion, RankedChanges, Section

sys.path.insert(0, str(REPO_ROOT / "eval" / "materiality"))
import materiality  # noqa: E402

CMP = "MAT:yoy-10k"
RISK = "Tariffs on goods imported from China could materially increase our product costs."
OLD = "Tariffs did not have a material impact on our results."


def change(change_id: str, *, current: str | None = None, prior: str | None = None, cmp=CMP):
    kind = (
        ChangeKind.MODIFIED
        if current and prior
        else (ChangeKind.ADDED if current else ChangeKind.REMOVED)
    )
    return materiality.Change(
        change_id=change_id,
        comparison_id=cmp,
        kind=kind,
        section=Section.RISK_FACTORS,
        current_text=current,
        prior_text=prior,
    )


def decision(change_id: str, importance: int) -> MaterialityDecision:
    material = importance >= materiality.MIN_MATERIAL_IMPORTANCE
    return MaterialityDecision(
        change_id=change_id,
        material=material,
        importance=importance,
        criterion=MaterialityCriterion.NEW_OR_REMOVED_RISK if importance else None,
        rationale="why",
    )


def label(item_id: str, *quotes: Quote, kind=ChangeKind.ADDED, cmp=CMP) -> LabeledChange:
    return LabeledChange(
        item_id=item_id,
        comparison_id=cmp,
        section=Section.RISK_FACTORS,
        kind=kind,
        criterion=MaterialityCriterion.NEW_OR_REMOVED_RISK,
        summary="a new tariff risk",
        quotes=quotes,
    )


def sample(item_id: str, change_id: str, material: bool | None, current: str) -> SampledChange:
    return SampledChange(
        item_id=item_id,
        comparison_id=CMP,
        change_id=change_id,
        kind=ChangeKind.ADDED,
        section=Section.RISK_FACTORS,
        current_text=current,
        material=material,
    )


def test_holds_matches_either_way_on_the_quoted_side_only():
    wide = change("c1", current=f"Intro sentence. {RISK} More text.")
    assert materiality.holds(wide, Side.CURRENT, RISK) == "quote_in_change"
    assert materiality.holds(wide, Side.PRIOR, RISK) is None
    narrow = change("c2", current=RISK)
    assert materiality.holds(narrow, Side.CURRENT, f"{RISK} And a second sentence.") == (
        "change_in_quote"
    )


def test_holds_uses_the_checkers_boundaries():
    # "5%" must not match inside "1.5%", as in the citation checker.
    assert materiality.holds(change("c", current="Costs rose 1.5% in the year."), Side.CURRENT,
                             "5% in the year") is None  # fmt: skip


def test_label_found_missed_and_not_aligned():
    changes = [change("c1", current=RISK), change("c2", current="Unrelated new text here.")]
    found = materiality.score_label(
        label("L1", Quote(side=Side.CURRENT, text=RISK)), changes, {"c1": decision("c1", 4)}
    )
    assert found["status"] == "found" and found["best_importance"] == 4
    missed = materiality.score_label(
        label("L1", Quote(side=Side.CURRENT, text=RISK)), changes, {"c1": decision("c1", 2)}
    )
    assert missed["status"] == "missed" and missed["best_importance"] == 2
    absent = materiality.score_label(
        label("L2", Quote(side=Side.CURRENT, text="Text no change holds.")), changes, {}
    )
    assert absent["status"] == "not_aligned" and absent["matches"] == []


def test_label_found_if_any_matching_change_is_material():
    # A modified label can be held by a removed and an added paragraph that did not pair.
    changes = [change("c1", prior=OLD), change("c2", current=RISK)]
    result = materiality.score_label(
        label("L1", Quote(side=Side.CURRENT, text=RISK), Quote(side=Side.PRIOR, text=OLD),
              kind=ChangeKind.MODIFIED),
        changes,
        {"c1": decision("c1", 0), "c2": decision("c2", 3)},
    )  # fmt: skip
    assert result["status"] == "found" and len(result["matches"]) == 2


def test_label_ignores_other_comparisons():
    other = change("c1", current=RISK, cmp="MAT:seq-10q")
    result = materiality.score_label(
        label("L1", Quote(side=Side.CURRENT, text=RISK)), [other], {"c1": decision("c1", 5)}
    )
    assert result["status"] == "not_aligned"


def test_sample_status_and_drift():
    run = {"c1": change("c1", current=RISK)}
    decided = materiality.score_sample(
        sample("F1", "c1", False, RISK), run, {"c1": decision("c1", 3)}
    )
    assert decided["status"] == "decided" and decided["called_material"] is True
    drifted = materiality.score_sample(
        sample("F1", "c1", False, "older text"), run, {"c1": decision("c1", 3)}
    )
    assert drifted["status"] == "text_changed"
    assert materiality.score_sample(sample("F2", "gone", False, RISK), run, {})["status"] == (
        "not_in_run"
    )


def test_score_counts_at_each_cutoff():
    changes = [
        change("c1", current=RISK),
        change("f1", current="Sampled one."),
        change("f2", current="Sampled two."),
        change("f3", current="Sampled three."),
    ]
    decisions = [decision("c1", 2), decision("f1", 1), decision("f2", 3), decision("f3", 4)]
    record = {
        "comparison_id": CMP,
        "text_changes": [c.model_dump(mode="json") for c in changes],
        "result": {
            "comparison_id": CMP,
            "ranked": RankedChanges(comparison_id=CMP, changes=()).model_dump(mode="json"),
            "decisions": [d.model_dump(mode="json") for d in decisions],
            "unresolved": [],
            "calls": 1,
            "cost_usd": 0.0,
        },
    }
    benchmark = Benchmark(
        changes=(label("L1", Quote(side=Side.CURRENT, text=RISK)),),
        sampled=(
            sample("F1", "f1", False, "Sampled one."),
            sample("F2", "f2", False, "Sampled two."),
            sample("F3", "f3", True, "Sampled three."),
        ),
    )
    summary = materiality.score(benchmark, [record])["summary"]
    assert summary["labeled_found"] == "0/1"
    assert summary["sampled_false_positives"] == "1/2"
    assert summary["sampled_material_found"] == "1/1"
    assert summary["by_importance_cutoff"]["1"] == {
        "labeled_found": "1/1",
        "sampled_false_positives": "2/2",
        "sampled_material_found": "1/1",
    }
    assert summary["by_importance_cutoff"]["2"]["labeled_found"] == "1/1"
    assert summary["by_importance_cutoff"]["3"]["sampled_false_positives"] == "1/2"


def test_tickers_come_from_the_labels():
    benchmark = Benchmark(
        changes=(label("L1", Quote(side=Side.CURRENT, text=RISK), cmp="SWK:yoy-10q"),),
        sampled=(sample("F1", "c", False, RISK),),
    )
    assert materiality.tickers_of(benchmark) == ["MAT", "SWK"]
