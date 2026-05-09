from core.models import ConfirmedFieldMapping
from schema_detection.coverage_assessor import assess_profile_coverage


def test_behavioural_dataset_has_strong_behavioural_and_no_demographic_psychographic():
    mappings = [
        ConfirmedFieldMapping(name="monetary", inferred_type="numeric", role="monetary_value", polarity="positive"),
        ConfirmedFieldMapping(name="frequency", inferred_type="numeric", role="purchase_frequency", polarity="positive"),
        ConfirmedFieldMapping(name="recency", inferred_type="numeric", role="recency", polarity="negative"),
        ConfirmedFieldMapping(name="sessions", inferred_type="numeric", role="session_activity", polarity="positive"),
        ConfirmedFieldMapping(name="returns", inferred_type="numeric", role="return_refund", polarity="negative"),
    ]
    coverage = assess_profile_coverage(mappings)
    behavioural = next(item for item in coverage.dimensions if item.dimension == "Behavioural, digital and transactional")
    demographic = next(item for item in coverage.dimensions if item.dimension == "Demographic and socioeconomic")
    psychographic = next(item for item in coverage.dimensions if item.dimension == "Psychographic and motivational")
    assert behavioural.analysis_depth == "strong"
    assert not demographic.available
    assert psychographic.proxy_only


def test_demographic_geographic_dataset_has_coverage():
    mappings = [
        ConfirmedFieldMapping(name="age", inferred_type="numeric", role="age", polarity="neutral"),
        ConfirmedFieldMapping(name="gender", inferred_type="categorical", role="gender", polarity="neutral"),
        ConfirmedFieldMapping(name="city", inferred_type="categorical", role="city", polarity="neutral"),
        ConfirmedFieldMapping(name="income", inferred_type="numeric", role="income", polarity="neutral"),
    ]
    coverage = assess_profile_coverage(mappings)
    assert any(item.dimension == "Demographic and socioeconomic" and item.available for item in coverage.dimensions)
    assert any(item.dimension == "Geographic and environmental" and item.available for item in coverage.dimensions)


def test_b2b_dataset_returns_icp_coverage():
    mappings = [
        ConfirmedFieldMapping(name="industry", inferred_type="categorical", role="industry", polarity="neutral"),
        ConfirmedFieldMapping(name="company_size", inferred_type="categorical", role="company_size", polarity="positive"),
        ConfirmedFieldMapping(name="contract_value", inferred_type="numeric", role="contract_value", polarity="positive"),
    ]
    coverage = assess_profile_coverage(mappings)
    b2b = next(item for item in coverage.dimensions if item.dimension == "B2B firmographic and decision-role")
    assert b2b.available

