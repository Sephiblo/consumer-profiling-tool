import pandas as pd

from analysis.quality_analysis import analyze_data_quality
from core.models import ConfirmedFieldMapping
from schema_detection.type_detector import detect_field_types


def test_business_rule_quality_checks_find_cross_field_issues():
    df = pd.DataFrame(
        {
            "frequency": [0, 2, 1],
            "monetary": [100, 0, 50],
            "session_count": [0, 1, 0],
            "clicks": [3, 0, 0],
            "cart_abandon_rate": [120, 50, -1],
            "age": [35, 150, 25],
            "total_spend": [10, -5, 0],
            "return_count": [2, 0, 4],
            "order_count": [1, 0, 3],
        }
    )
    mappings = [
        ConfirmedFieldMapping(name="frequency", inferred_type="numeric", role="purchase_frequency", polarity="positive"),
        ConfirmedFieldMapping(name="monetary", inferred_type="numeric", role="monetary_value", polarity="positive"),
        ConfirmedFieldMapping(name="session_count", inferred_type="numeric", role="session_activity", polarity="positive"),
        ConfirmedFieldMapping(name="clicks", inferred_type="numeric", role="click_activity", polarity="positive"),
        ConfirmedFieldMapping(name="cart_abandon_rate", inferred_type="numeric", role="cart_abandonment", polarity="negative"),
        ConfirmedFieldMapping(name="age", inferred_type="numeric", role="age", polarity="neutral"),
        ConfirmedFieldMapping(name="total_spend", inferred_type="numeric", role="monetary_value", polarity="positive"),
        ConfirmedFieldMapping(name="return_count", inferred_type="numeric", role="return_refund", polarity="negative"),
        ConfirmedFieldMapping(name="order_count", inferred_type="numeric", role="purchase_frequency", polarity="positive"),
    ]

    quality = analyze_data_quality(df, detect_field_types(df), mappings)
    issues = "\n".join(quality["business_rule_issues"])

    assert "frequency is 0 but monetary is positive" in issues
    assert "session_count is 0 but clicks is positive" in issues
    assert "cart_abandon_rate is outside the 0-100 percentage range" in issues
    assert "age is outside the reasonable 0-120 age range" in issues
    assert "total_spend contains negative spend/value" in issues
    assert "return_count is greater than order_count" in issues


def test_short_age_keyword_does_not_match_page_views():
    df = pd.DataFrame({"page_views": [10, 200, 500]})
    quality = analyze_data_quality(df, detect_field_types(df))
    issues = "\n".join(quality["potential_data_issues"])

    assert "page_views is outside the reasonable 0-120 age range" not in issues
