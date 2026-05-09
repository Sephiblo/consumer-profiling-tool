import pandas as pd

from schema_detection.semantic_mapper import map_field_semantics
from schema_detection.type_detector import detect_field_types


def test_semantic_mapper_maps_required_v2_examples():
    df = pd.DataFrame(
        {
            "total_spend": [10, 20, 30],
            "orders_12m": [1, 2, 3],
            "days_since_last_order": [5, 20, 30],
            "email_open_rate": [0.2, 0.5, 0.7],
            "refund_count": [0, 1, 2],
            "tier_level": ["tier1", "tier2", "tier1"],
            "age_group": ["18-24", "25-34", "25-34"],
            "city": ["London", "Manchester", "London"],
            "price_sensitivity": ["high", "low", "medium"],
            "company_size": ["small", "enterprise", "mid-market"],
            "decision_role": ["buyer", "influencer", "user"],
        }
    )
    mapped = {profile.name: profile for profile in map_field_semantics(detect_field_types(df), df)}

    assert mapped["total_spend"].suggested_role == "monetary_value"
    assert mapped["orders_12m"].suggested_role == "purchase_frequency"
    assert mapped["days_since_last_order"].suggested_role == "recency"
    assert mapped["days_since_last_order"].suggested_polarity == "negative"
    assert mapped["email_open_rate"].suggested_role in {"email_engagement", "engagement"}
    assert mapped["refund_count"].suggested_role in {"return_refund", "risk_or_friction"}
    assert mapped["refund_count"].suggested_polarity == "negative"
    assert mapped["tier_level"].suggested_role == "existing_segment"
    assert mapped["age_group"].suggested_role == "age"
    assert mapped["city"].suggested_role == "city"
    assert mapped["price_sensitivity"].suggested_role == "price_sensitivity"
    assert mapped["company_size"].suggested_role == "company_size"
    assert mapped["decision_role"].suggested_role == "decision_role"

