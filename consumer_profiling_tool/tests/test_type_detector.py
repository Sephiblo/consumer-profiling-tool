import pandas as pd

from schema_detection.type_detector import detect_field_types


def test_type_detector_detects_v2_field_types():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004"],
            "total_spend": [100.0, 200.0, 150.0, 250.0],
            "segment": ["A", "B", "A", "B"],
            "responded": [1, 0, 1, 0],
            "signup_date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
            "feedback_text": [
                "This is a long response about delivery quality and service expectations.",
                "Another long free text answer about price sensitivity and product fit.",
                "Detailed complaint narrative with many words and context.",
                "A long survey answer about lifestyle and buying motivations.",
            ],
            "email_open_rate": ["10%", "20%", "30%", "40%"],
            "email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com"],
        }
    )
    profiles = {profile.name: profile for profile in detect_field_types(df)}

    assert profiles["customer_id"].is_id_like
    assert profiles["total_spend"].is_numeric
    assert profiles["segment"].is_categorical
    assert profiles["responded"].is_binary
    assert profiles["signup_date"].is_datetime_like
    assert profiles["feedback_text"].is_text_like
    assert profiles["email_open_rate"].inferred_type == "percentage_rate"
    assert profiles["email"].is_sensitive_candidate

