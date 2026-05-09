import pandas as pd

from analysis.score_generator import calculate_risk_scores, calculate_score_group
from core.models import ConfirmedFieldMapping


def test_positive_fields_produce_higher_scores_when_values_are_higher():
    df = pd.DataFrame({"spend": [10, 50, 100]})
    mappings = [ConfirmedFieldMapping(name="spend", inferred_type="numeric", role="monetary_value", polarity="positive")]
    score = calculate_score_group(df, mappings, "value_score")
    assert score.iloc[2] > score.iloc[0]


def test_negative_fields_produce_lower_health_scores_when_values_are_higher():
    df = pd.DataFrame({"refund_count": [0, 2, 10]})
    mappings = [ConfirmedFieldMapping(name="refund_count", inferred_type="numeric", role="return_refund", polarity="negative")]
    raw, health = calculate_risk_scores(df, mappings)
    assert raw.iloc[2] > raw.iloc[0]
    assert health.iloc[2] < health.iloc[0]


def test_missing_score_groups_return_none():
    df = pd.DataFrame({"age": [20, 30, 40]})
    mappings = [ConfirmedFieldMapping(name="age", inferred_type="numeric", role="age", polarity="neutral")]
    assert calculate_score_group(df, mappings, "value_score") is None

