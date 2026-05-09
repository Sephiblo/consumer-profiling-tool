from core.models import ConfirmedFieldMapping
from schema_detection.b2b_b2c_detector import detect_profile_mode


def test_b2b_dataset_returns_b2b_mode():
    mappings = [
        ConfirmedFieldMapping(name="company_size", inferred_type="categorical", role="company_size", polarity="positive"),
        ConfirmedFieldMapping(name="industry", inferred_type="categorical", role="industry", polarity="neutral"),
        ConfirmedFieldMapping(name="decision_role", inferred_type="categorical", role="decision_role", polarity="neutral"),
    ]
    assert detect_profile_mode(mappings).mode == "b2b"

