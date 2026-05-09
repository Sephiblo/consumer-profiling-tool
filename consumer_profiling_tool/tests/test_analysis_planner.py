from analysis.analysis_planner import build_analysis_plan
from core.models import ConfirmedFieldMapping
from schema_detection.coverage_assessor import assess_profile_coverage


def test_planner_skips_rfm_and_supports_clustering_without_segment():
    mappings = [ConfirmedFieldMapping(name="sessions", inferred_type="numeric", role="session_activity", polarity="positive")]
    plan = build_analysis_plan(mappings, assess_profile_coverage(mappings))
    assert "RFM value-lifecycle analysis" in plan.skipped_analyses
    assert "Automatic behavioural clustering" in plan.supported_analyses


def test_planner_supports_response_model_only_if_binary_target_exists():
    mappings = [ConfirmedFieldMapping(name="response", inferred_type="binary", role="binary_target", polarity="positive")]
    plan = build_analysis_plan(mappings, assess_profile_coverage(mappings))
    assert "Response prediction model" in plan.supported_analyses


def test_planner_reports_proxy_motivation_without_psychographic_fields():
    mappings = [ConfirmedFieldMapping(name="clicks", inferred_type="numeric", role="click_activity", polarity="positive")]
    plan = build_analysis_plan(mappings, assess_profile_coverage(mappings))
    assert "Psychographic/motivation interpretation" in plan.proxy_analyses

