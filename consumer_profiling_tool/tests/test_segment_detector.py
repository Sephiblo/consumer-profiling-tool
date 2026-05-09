import pandas as pd

from analysis.score_generator import generate_customer_scores
from analysis.segment_profiling import rank_segments
from core.models import ConfirmedFieldMapping


def _mappings():
    return [
        ConfirmedFieldMapping(name="tier", inferred_type="categorical", role="existing_segment", polarity="neutral"),
        ConfirmedFieldMapping(name="spend", inferred_type="numeric", role="monetary_value", polarity="positive"),
        ConfirmedFieldMapping(name="orders", inferred_type="numeric", role="purchase_frequency", polarity="positive"),
        ConfirmedFieldMapping(name="refunds", inferred_type="numeric", role="return_refund", polarity="negative"),
    ]


def test_tier1_can_be_highest_if_metrics_support_it():
    df = pd.DataFrame({"tier": ["tier1", "tier1", "tier5", "tier5"], "spend": [1000, 900, 10, 20], "orders": [20, 18, 1, 2], "refunds": [0, 0, 5, 4]})
    ranking, _ = rank_segments(generate_customer_scores(df, _mappings()))
    assert ranking.iloc[0]["Segment"] == "tier1"


def test_tier5_can_be_highest_if_metrics_support_it_and_not_alphabetical():
    df = pd.DataFrame({"tier": ["tier1", "tier1", "tier5", "tier5"], "spend": [10, 20, 1000, 900], "orders": [1, 2, 20, 18], "refunds": [5, 4, 0, 0]})
    ranking, _ = rank_segments(generate_customer_scores(df, _mappings()))
    assert ranking.iloc[0]["Segment"] == "tier5"

