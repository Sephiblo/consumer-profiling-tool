"""Theory-enhanced schema-flexible consumer profiling Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.analysis_planner import build_analysis_plan
from analysis.b2b_icp_analysis import run_b2b_icp_analysis
from analysis.clustering import generate_behavioural_clusters
from analysis.demographic_analysis import run_demographic_analysis
from analysis.eda_engine import run_eda
from analysis.funnel_analysis import run_funnel_analysis
from analysis.geographic_analysis import run_geographic_analysis
from analysis.lifecycle_analysis import run_lifecycle_analysis
from analysis.negative_persona import run_negative_persona_analysis
from analysis.psychographic_analysis import run_psychographic_analysis
from analysis.quality_analysis import analyze_data_quality
from analysis.recommendation_engine import data_collection_recommendations, generate_segment_recommendations
from analysis.response_model import run_response_model
from analysis.rfm_analysis import run_rfm_analysis
from analysis.roi_analysis import run_roi_analysis
from analysis.score_generator import generate_customer_scores
from analysis.segment_profiling import metric_comparison_by_segment, rank_segments
from core.constants import FIELD_ROLES, POLARITIES
from core.models import ConfirmedFieldMapping
from core.privacy import scan_privacy
from preprocessing.cleaner import clean_dataframe
from preprocessing.identity_resolution import analyze_identity_fields
from preprocessing.missingness_analyzer import analyze_missingness
from reporting.export import analysis_summary_json, coverage_json, mapping_json, scored_customers_csv
from reporting.report_generator import generate_markdown_report
from schema_detection.b2b_b2c_detector import detect_profile_mode
from schema_detection.coverage_assessor import assess_profile_coverage
from schema_detection.semantic_mapper import map_field_semantics
from schema_detection.segment_detector import detect_existing_segment
from schema_detection.type_detector import detect_field_types
from theory.theory_reporter import generate_theory_narrative


st.set_page_config(page_title="Theory-Enhanced Consumer Profiling", layout="wide")

SAMPLE_PATH = Path("/mnt/data/模拟用客户数据.csv")


def read_csv_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV with UTF-8 then GB18030 fallback."""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="gb18030")


def build_mapping_editor_rows(semantic_profiles) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Column": profile.name,
                "Type": profile.inferred_type,
                "Suggested Role": profile.suggested_role,
                "Role Confidence": profile.role_confidence,
                "Polarity": profile.suggested_polarity,
                "Polarity Confidence": profile.polarity_confidence,
                "Sensitive?": profile.is_sensitive_candidate,
                "Proxy?": profile.is_proxy_inference,
                "Reasons": "; ".join(profile.reasons),
                "User Role": profile.suggested_role,
                "User Polarity": profile.suggested_polarity,
            }
            for profile in semantic_profiles
        ]
    )


def confirmed_mappings_from_editor(editor_df: pd.DataFrame) -> list[ConfirmedFieldMapping]:
    mappings: list[ConfirmedFieldMapping] = []
    for _, row in editor_df.iterrows():
        mappings.append(
            ConfirmedFieldMapping(
                name=row["Column"],
                inferred_type=row["Type"],
                role=row["User Role"],
                role_confidence=float(row["Role Confidence"]),
                polarity=row["User Polarity"],
                polarity_confidence=float(row["Polarity Confidence"]),
                is_sensitive_candidate=bool(row["Sensitive?"]),
                is_proxy_inference=bool(row["Proxy?"]),
                reasons=[text for text in str(row["Reasons"]).split("; ") if text],
            )
        )
    return mappings


def run_pipeline(
    df: pd.DataFrame,
    mappings: list[ConfirmedFieldMapping],
    type_profiles,
    force_clusters: bool = False,
) -> dict[str, object]:
    cleaned_df, cleaning_meta = clean_dataframe(df, type_profiles)
    quality_report = analyze_data_quality(cleaned_df, type_profiles, mappings)
    missingness = analyze_missingness(df)
    privacy_scan = scan_privacy(df, type_profiles)
    identity = analyze_identity_fields(df, mappings)
    coverage = assess_profile_coverage(mappings)
    mode_result = detect_profile_mode(mappings)
    plan = build_analysis_plan(mappings, coverage)
    segment_detection = detect_existing_segment(cleaned_df, type_profiles, mappings)

    use_existing_segment = (
        not force_clusters
        and segment_detection.field_name is not None
        and any(mapping.role == "existing_segment" for mapping in mappings)
    )
    cluster_result = None
    cluster_labels = None
    if not use_existing_segment:
        cluster_result = generate_behavioural_clusters(cleaned_df, mappings, mode_result.mode)
        cluster_labels = cluster_result["labels"]

    scored_df = generate_customer_scores(cleaned_df, mappings, cluster_labels)
    response_result, probabilities = run_response_model(cleaned_df, mappings)
    if probabilities is not None:
        scored_df["predicted_response_probability"] = probabilities

    if use_existing_segment:
        segment_profile, interpretations = rank_segments(scored_df, "_original_segment")
        active_segment_column = "_original_segment"
    else:
        segment_profile = cluster_result["profile_table"] if cluster_result else pd.DataFrame()
        interpretations = cluster_result["explanations"] if cluster_result else []
        active_segment_column = "_generated_cluster" if "_generated_cluster" in scored_df.columns else ""

    recommendations = generate_segment_recommendations(scored_df, active_segment_column) if active_segment_column else []
    negative_personas = run_negative_persona_analysis(scored_df, active_segment_column)
    roi_result = run_roi_analysis(cleaned_df, scored_df, mappings, active_segment_column)
    theory_narrative = generate_theory_narrative(coverage, mode_result, plan.proxy_analyses)
    demographic = run_demographic_analysis(cleaned_df, mappings, active_segment_column)
    geographic = run_geographic_analysis(cleaned_df, mappings)
    psychographic = run_psychographic_analysis(cleaned_df, mappings)
    b2b_icp = run_b2b_icp_analysis(cleaned_df, mappings, mode_result)
    lifecycle = run_lifecycle_analysis(scored_df, mappings)
    funnel = run_funnel_analysis(cleaned_df, mappings)
    rfm = run_rfm_analysis(scored_df, mappings)
    eda = run_eda(scored_df, mappings, active_segment_column if active_segment_column else None)
    report = generate_markdown_report(
        quality_report=quality_report,
        privacy_scan=privacy_scan,
        mappings=mappings,
        coverage=coverage,
        analysis_plan=plan,
        mode_result=mode_result,
        theory_narrative=theory_narrative,
        scored_df=scored_df,
        segment_profile=segment_profile,
        interpretations=interpretations,
        recommendations=recommendations,
        negative_personas=negative_personas,
        roi_result=roi_result,
    )

    return {
        "cleaned_df": cleaned_df,
        "cleaning_meta": cleaning_meta,
        "quality_report": quality_report,
        "missingness": missingness,
        "privacy_scan": privacy_scan,
        "identity": identity,
        "coverage": coverage,
        "mode_result": mode_result,
        "analysis_plan": plan,
        "segment_detection": segment_detection,
        "cluster_result": cluster_result,
        "scored_df": scored_df,
        "response_result": response_result,
        "segment_profile": segment_profile,
        "interpretations": interpretations,
        "recommendations": recommendations,
        "negative_personas": negative_personas,
        "roi_result": roi_result,
        "theory_narrative": theory_narrative,
        "demographic": demographic,
        "geographic": geographic,
        "psychographic": psychographic,
        "b2b_icp": b2b_icp,
        "lifecycle": lifecycle,
        "funnel": funnel,
        "rfm": rfm,
        "eda": eda,
        "active_segment_column": active_segment_column,
        "data_collection_recommendations": data_collection_recommendations(coverage.missing_pillars),
        "report": report,
    }


def show_upload() -> None:
    st.header("1. Upload and Dataset Preview / 上传和数据预览")
    uploaded_file = st.file_uploader("Upload arbitrary customer/account CSV", type=["csv"])
    sample_loaded = SAMPLE_PATH.exists() and st.button("Load sample dataset / 加载示例数据")

    df = None
    if uploaded_file is not None:
        df = read_csv_upload(uploaded_file)
    elif sample_loaded:
        df = pd.read_csv(SAMPLE_PATH)

    if df is not None:
        type_profiles = detect_field_types(df)
        semantic_profiles = map_field_semantics(type_profiles, df)
        quality_report = analyze_data_quality(df, type_profiles)
        privacy_scan = scan_privacy(df, type_profiles)
        st.session_state["raw_df"] = df
        st.session_state["type_profiles"] = type_profiles
        st.session_state["semantic_profiles"] = semantic_profiles
        st.session_state["quality_report_initial"] = quality_report
        st.session_state["privacy_scan_initial"] = privacy_scan
        st.session_state["mapping_editor"] = build_mapping_editor_rows(semantic_profiles)

    if "raw_df" not in st.session_state:
        st.info("Upload a CSV to begin. The app does not require fixed column names.")
        return

    df = st.session_state["raw_df"]
    quality_report = st.session_state["quality_report_initial"]
    cols = st.columns(4)
    cols[0].metric("Rows", f"{quality_report['row_count']:,}")
    cols[1].metric("Columns", f"{quality_report['column_count']:,}")
    cols[2].metric("Duplicate rows", f"{quality_report['duplicate_row_count']:,}")
    cols[3].metric("Quality warnings", len(quality_report["potential_data_issues"]))
    for issue in quality_report["potential_data_issues"][:5]:
        st.warning(issue)
    st.dataframe(df.head(20), use_container_width=True)


def show_mapping() -> None:
    st.header("2. Field Mapping and Privacy Review / 字段映射和隐私审查")
    if "mapping_editor" not in st.session_state:
        st.info("Upload a CSV first.")
        return
    privacy_scan = st.session_state["privacy_scan_initial"]
    st.info(privacy_scan.privacy_notice)
    if privacy_scan.flagged_fields:
        st.write("Privacy-flagged fields:", ", ".join(flag.name for flag in privacy_scan.flagged_fields))

    edited = st.data_editor(
        st.session_state["mapping_editor"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "User Role": st.column_config.SelectboxColumn("User Role", options=FIELD_ROLES, required=True),
            "User Polarity": st.column_config.SelectboxColumn("User Polarity", options=POLARITIES, required=True),
            "Reasons": st.column_config.TextColumn("Reasons", width="large"),
        },
        disabled=[
            "Column",
            "Type",
            "Suggested Role",
            "Role Confidence",
            "Polarity",
            "Polarity Confidence",
            "Sensitive?",
            "Proxy?",
            "Reasons",
        ],
    )
    st.session_state["mapping_editor"] = edited
    force_clusters = st.checkbox("Generate new behavioural clusters even if existing segment exists / 强制生成新聚类", value=False)
    if st.button("Confirm mapping and run theory-enhanced profiling", type="primary"):
        mappings = confirmed_mappings_from_editor(edited)
        st.session_state["confirmed_mappings"] = mappings
        try:
            st.session_state["pipeline"] = run_pipeline(
                st.session_state["raw_df"],
                mappings,
                st.session_state["type_profiles"],
                force_clusters=force_clusters,
            )
            st.success("Profiling completed.")
        except Exception as exc:
            st.error(f"Profiling failed: {exc}")


def require_pipeline():
    pipeline = st.session_state.get("pipeline")
    if not pipeline:
        st.info("Confirm mapping and run profiling first.")
    return pipeline


def show_coverage() -> None:
    st.header("3. Profile Coverage Matrix / 画像覆盖度矩阵")
    pipeline = require_pipeline()
    if not pipeline:
        return
    coverage = pipeline["coverage"]
    st.metric("Data completeness score", f"{coverage.data_completeness_score:.1f}")
    st.write(coverage.summary)
    st.dataframe(pd.DataFrame([item.model_dump() for item in coverage.dimensions]), use_container_width=True)


def show_analysis_availability() -> None:
    st.header("4. Analysis Availability / 分析可用性")
    pipeline = require_pipeline()
    if not pipeline:
        return
    plan = pipeline["analysis_plan"]
    cols = st.columns(3)
    with cols[0]:
        st.subheader("Supported")
        for item in plan.supported_analyses:
            st.write(f"[OK] {item}")
    with cols[1]:
        st.subheader("Skipped")
        for name, reason in plan.skipped_analyses.items():
            st.write(f"[Skipped] {name}: {reason}")
    with cols[2]:
        st.subheader("Proxy")
        for name, reason in plan.proxy_analyses.items():
            st.write(f"[Proxy] {name}: {reason}")
    for warning in plan.warnings:
        st.warning(warning)


def show_dashboard() -> None:
    st.header("5. Dashboard Overview / 总览")
    pipeline = require_pipeline()
    if not pipeline:
        return
    scored = pipeline["scored_df"]
    mappings = st.session_state["confirmed_mappings"]
    mode = pipeline["mode_result"]
    segment_column = pipeline["active_segment_column"]
    id_field = pipeline["identity"].suggested_primary_id or "Not detected"
    segment_field = next((m.name for m in mappings if m.role == "existing_segment"), "Generated clusters")
    cols = st.columns(4)
    cols[0].metric("Total records", f"{len(scored):,}")
    cols[1].metric("ID field", id_field)
    cols[2].metric("Mode", mode.mode)
    cols[3].metric("Segment field", segment_field)
    cols = st.columns(4)
    cols[0].metric("Segments/clusters", scored[segment_column].nunique() if segment_column in scored else 0)
    cols[1].metric("Available pillars", len(pipeline["coverage"].available_pillars))
    cols[2].metric("Completeness", f"{pipeline['coverage'].data_completeness_score:.1f}")
    cols[3].metric("Mean risk", _metric_mean(scored, "risk_score_raw"))
    cols = st.columns(3)
    cols[0].metric("Mean value", _metric_mean(scored, "value_score"))
    cols[1].metric("Mean engagement", _metric_mean(scored, "engagement_score"))
    cols[2].metric("Mean B2B fit", _metric_mean(scored, "b2b_account_fit_score"))


def _metric_mean(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or not df[column].notna().any():
        return "N/A"
    return f"{df[column].mean():.1f}"


def show_segment_cluster() -> None:
    st.header("6. Segment/Cluster Profiling / 分段或聚类画像")
    pipeline = require_pipeline()
    if not pipeline:
        return
    scored = pipeline["scored_df"]
    segment_column = pipeline["active_segment_column"]
    table = pipeline["segment_profile"]
    for text in pipeline["interpretations"]:
        st.write(text)
    if not table.empty:
        st.dataframe(table, use_container_width=True)
    if segment_column and segment_column in scored:
        counts = scored[segment_column].value_counts().reset_index()
        counts.columns = [segment_column, "count"]
        st.plotly_chart(px.bar(counts, x=segment_column, y="count", title="Segment/cluster size"), use_container_width=True)
        if "Composite Score" in table.columns and table["Composite Score"].notna().any():
            st.plotly_chart(px.bar(table, x="Segment", y="Composite Score", title="Data-derived composite score"), use_container_width=True)
        comparison = metric_comparison_by_segment(scored, segment_column)
        if not comparison.empty:
            melted = comparison.melt(id_vars=[segment_column], var_name="metric", value_name="score")
            st.plotly_chart(px.bar(melted, x=segment_column, y="score", color="metric", barmode="group"), use_container_width=True)


def show_theory_analysis() -> None:
    st.header("7. Theory-Based Profile Analysis / 理论画像分析")
    pipeline = require_pipeline()
    if not pipeline:
        return
    st.markdown(pipeline["theory_narrative"])
    for label, key in [
        ("Demographic analysis", "demographic"),
        ("Geographic analysis", "geographic"),
        ("Psychographic analysis", "psychographic"),
        ("B2B ICP/persona analysis", "b2b_icp"),
        ("Lifecycle analysis", "lifecycle"),
        ("Funnel analysis", "funnel"),
        ("RFM analysis", "rfm"),
    ]:
        st.subheader(label)
        result = pipeline[key]
        st.write(result.get("summary", "No summary."))
        table = result.get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(table, use_container_width=True)
        tables = result.get("tables")
        if isinstance(tables, dict):
            for name, item in list(tables.items())[:5]:
                st.write(name)
                st.dataframe(item, use_container_width=True)


def show_strategy() -> None:
    st.header("8. Business Strategy Recommendations / 商业策略建议")
    pipeline = require_pipeline()
    if not pipeline:
        return
    st.subheader("Priority groups")
    for item in pipeline["recommendations"]:
        st.write(f"**{item['segment']}**")
        st.write(f"Recommendation: {item['recommendation']}")
        st.write(f"Evidence: {item['evidence_from_data']}")
        st.write(f"Confidence: {item['confidence_level']}")
        st.caption(f"Assumptions: {item['assumptions']} | Limitation: {item['limitation']}")
    st.subheader("Negative persona warnings")
    st.dataframe(pipeline["negative_personas"], use_container_width=True)
    st.subheader("ROI and marketing effectiveness")
    st.write(pipeline["roi_result"]["summary"])
    roi_table = pipeline["roi_result"].get("table")
    if isinstance(roi_table, pd.DataFrame) and not roi_table.empty:
        st.dataframe(roi_table, use_container_width=True)
    st.subheader("Data collection priorities")
    for item in pipeline["data_collection_recommendations"]:
        st.write(f"- {item}")


def show_customer_explorer() -> None:
    st.header("9. Customer Score Explorer / 客户评分浏览")
    pipeline = require_pipeline()
    if not pipeline:
        return
    scored = pipeline["scored_df"].copy()
    filtered = scored.copy()
    segment_column = pipeline["active_segment_column"]
    cols = st.columns(5)
    if segment_column and segment_column in filtered:
        selected = cols[0].multiselect("Segment/cluster", sorted(filtered[segment_column].dropna().astype(str).unique()))
        if selected:
            filtered = filtered[filtered[segment_column].astype(str).isin(selected)]
    if "recommended_profile_type" in filtered:
        selected_profile = cols[1].multiselect("Persona type", sorted(filtered["recommended_profile_type"].dropna().astype(str).unique()))
        if selected_profile:
            filtered = filtered[filtered["recommended_profile_type"].isin(selected_profile)]
    if cols[2].checkbox("High value"):
        filtered = filtered[pd.to_numeric(filtered.get("value_score"), errors="coerce") >= 70]
    if cols[3].checkbox("High risk"):
        filtered = filtered[pd.to_numeric(filtered.get("risk_score_raw"), errors="coerce") >= 70]
    if cols[4].checkbox("Negative persona"):
        filtered = filtered[filtered.get("negative_persona_candidate", False)]
    preferred = [
        "_customer_profile_id",
        "_original_segment",
        "_generated_cluster",
        "value_score",
        "frequency_loyalty_score",
        "engagement_score",
        "conversion_score",
        "risk_score_raw",
        "risk_score_health",
        "b2b_account_fit_score",
        "profile_quality_score",
        "predicted_response_probability",
        "negative_persona_candidate",
        "recommended_profile_type",
        "recommended_action",
        "recommendation_evidence",
        "recommendation_confidence",
    ]
    st.dataframe(filtered[[column for column in preferred if column in filtered]].head(1000), use_container_width=True)


def show_export() -> None:
    st.header("10. Export / 导出")
    pipeline = require_pipeline()
    if not pipeline:
        return
    include_sensitive = st.checkbox("Include sensitive-looking raw fields in scored CSV export / 导出中包含敏感原始字段", value=False)
    cols = st.columns(5)
    cols[0].download_button("Scored CSV", scored_customers_csv(pipeline["scored_df"], include_sensitive), "scored_customers.csv", "text/csv")
    cols[1].download_button("Mapping JSON", mapping_json(st.session_state["confirmed_mappings"]), "field_mapping.json", "application/json")
    cols[2].download_button("Coverage JSON", coverage_json(pipeline["coverage"]), "profile_coverage.json", "application/json")
    cols[3].download_button(
        "Analysis JSON",
        analysis_summary_json(
            pipeline["analysis_plan"],
            pipeline["quality_report"],
            {"mode": pipeline["mode_result"].model_dump(), "identity": pipeline["identity"].model_dump()},
        ),
        "analysis_summary.json",
        "application/json",
    )
    cols[4].download_button("Markdown report", pipeline["report"], "consumer_profile_report.md", "text/markdown")


def main() -> None:
    st.title("Theory-Enhanced Schema-Flexible Consumer Profiling Tool")
    st.caption("Schema-flexible field detection, theory coverage, B2C/B2B profiling, scoring, personas, and recommendations")
    show_upload()
    st.divider()
    show_mapping()
    st.divider()
    tabs = st.tabs(
        [
            "Coverage",
            "Analysis availability",
            "Dashboard",
            "Segment/cluster",
            "Theory analysis",
            "Strategy",
            "Customer explorer",
            "Export",
        ]
    )
    with tabs[0]:
        show_coverage()
    with tabs[1]:
        show_analysis_availability()
    with tabs[2]:
        show_dashboard()
    with tabs[3]:
        show_segment_cluster()
    with tabs[4]:
        show_theory_analysis()
    with tabs[5]:
        show_strategy()
    with tabs[6]:
        show_customer_explorer()
    with tabs[7]:
        show_export()


if __name__ == "__main__":
    main()

