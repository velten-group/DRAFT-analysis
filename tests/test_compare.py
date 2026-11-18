"""Supp. Fig. 6 comparison: another term may be shown only where the GO score is tied at the top-10 cut."""
import pandas as pd

from draft_analysis import compare


def _full() -> dict:
    # adjusted p: a = 1e-5 (unique); b, c, d tie at 0.5
    return {"T0": pd.DataFrame({"Term": ["a", "b", "c", "d"], "Adjusted P-value": [1e-5, 0.5, 0.5, 0.5]})}


def _panel(terms) -> pd.DataFrame:
    adj = {"a": 1e-5, "b": 0.5, "c": 0.5, "d": 0.5, "x": 1e-5}
    return pd.DataFrame({"Term": terms, "Adjusted P-value": [adj[t] for t in terms],
                         "score": compare.go_score([adj[t] for t in terms]), "topic": "T0", "panel_title": "Topic0"})


def test_another_tied_term_is_accepted():
    assert compare.tied_go_terms(_panel(["c", "a"]), _panel(["b", "a"]), _full()) == []


def test_an_untied_term_must_match():
    problems = compare.tied_go_terms(_panel(["b", "x"]), _panel(["b", "a"]), _full())
    assert problems and "untied" in problems[0]


def test_scores_must_match():
    assert compare.tied_go_terms(_panel(["a", "a"]), _panel(["b", "a"]), _full()) == ["T0: scores differ"]
