"""Automatic behavioural and account-fit clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.constants import B2B_ROLES, PSYCHOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping

CLUSTER_ROLES = {
    "monetary_value",
    "purchase_frequency",
    "avg_order_value",
    "profitability",
    "clv_or_ltv",
    "recency",
    "engagement",
    "session_activity",
    "page_view",
    "click_activity",
    "email_engagement",
    "product_interest",
    "conversion_or_response",
    "risk_or_friction",
    "cart_abandonment",
    "return_refund",
    "complaint_support",
} | PSYCHOGRAPHIC_ROLES | B2B_ROLES


def _selected_features(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping], include_sensitive: bool = False) -> list[str]:
    return [
        mapping.name
        for mapping in mappings
        if mapping.name in df.columns
        and mapping.role in CLUSTER_ROLES
        and mapping.role not in {"binary_target", "numeric_target"}
        and not mapping.is_sensitive_candidate or False
    ] if include_sensitive else [
        mapping.name
        for mapping in mappings
        if mapping.name in df.columns
        and mapping.role in CLUSTER_ROLES
        and mapping.role not in {"binary_target", "numeric_target"}
        and not mapping.is_sensitive_candidate
    ]


def _preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers)


def generate_behavioural_clusters(
    df: pd.DataFrame,
    mappings: list[ConfirmedFieldMapping],
    mode: str = "b2c",
    random_state: int = 42,
) -> dict[str, object]:
    features = _selected_features(df, mappings)
    if not features or len(df) < 3:
        labels = pd.Series(["Cluster 1"] * len(df), index=df.index)
        return {
            "labels": labels,
            "k": 1,
            "silhouette_score": None,
            "profile_table": pd.DataFrame({"Cluster": ["Cluster 1"], "Count": [len(df)], "Share": [1.0]}),
            "explanations": ["No reliable clustering features were available."],
            "feature_names": features,
        }

    matrix = _preprocessor(df, features).fit_transform(df[features])
    row_count = len(df)
    candidate_ks = [min(3, row_count - 1)] if row_count < 100 else [k for k in [3, 4, 5, 6] if k < row_count]
    candidate_ks = sorted(set(k for k in candidate_ks if k >= 2))
    if not candidate_ks:
        candidate_ks = [2]

    best_score = -1.0
    best_labels: np.ndarray | None = None
    best_k = candidate_ks[0]
    for k in candidate_ks:
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(matrix)
            score = silhouette_score(matrix, labels) if len(set(labels)) > 1 else -1
            if score > best_score:
                best_score, best_labels, best_k = score, labels, k
        except Exception:
            continue
    if best_labels is None:
        best_k = min(4, max(2, row_count - 1))
        best_labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(matrix)
        best_score = np.nan

    named, table, explanations = name_clusters(df, pd.Series(best_labels, index=df.index), mappings, features, mode)
    return {
        "labels": named,
        "k": int(best_k),
        "silhouette_score": None if pd.isna(best_score) else round(float(best_score), 3),
        "profile_table": table,
        "explanations": explanations,
        "feature_names": features,
    }


def name_clusters(
    df: pd.DataFrame,
    cluster_ids: pd.Series,
    mappings: list[ConfirmedFieldMapping],
    features: list[str],
    mode: str,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    mapping_by_name = {mapping.name: mapping for mapping in mappings}
    numeric_features = [feature for feature in features if pd.api.types.is_numeric_dtype(df[feature])]
    overall = df[numeric_features].mean(numeric_only=True) if numeric_features else pd.Series(dtype=float)
    rows: list[dict[str, object]] = []
    lookup: dict[int, str] = {}
    explanations: list[str] = []
    for cluster_id, index in cluster_ids.groupby(cluster_ids).groups.items():
        group = df.loc[index]
        traits: list[str] = []
        flags = {"value": False, "engagement": False, "risk": False, "b2b": False, "psychographic": False}
        for feature in numeric_features:
            base = overall.get(feature)
            if pd.isna(base) or base == 0:
                continue
            mean = group[feature].mean()
            delta = (mean - base) / abs(base)
            if abs(delta) >= 0.15:
                direction = "higher" if delta > 0 else "lower"
                traits.append(f"{direction} {feature}")
                role = mapping_by_name[feature].role
                flags["value"] = flags["value"] or (role in {"monetary_value", "avg_order_value", "contract_value"} and delta > 0)
                flags["engagement"] = flags["engagement"] or (role in {"engagement", "session_activity", "page_view", "click_activity"} and delta > 0)
                flags["risk"] = flags["risk"] or (role in {"risk_or_friction", "cart_abandonment", "return_refund", "complaint_support", "recency"} and delta > 0)
                flags["b2b"] = flags["b2b"] or (role in B2B_ROLES and delta > 0)
                flags["psychographic"] = flags["psychographic"] or (role in PSYCHOGRAPHIC_ROLES and delta > 0)
        name = _cluster_name(flags, int(cluster_id), mode)
        lookup[int(cluster_id)] = name
        explanations.append(
            f"{name}: " + ("; ".join(traits[:3]) if traits else "no strong distinguishing numeric traits detected.")
        )
        row = {
            "Cluster": name,
            "Count": int(len(group)),
            "Share": round(len(group) / max(len(df), 1), 4),
            "Top distinguishing features": "; ".join(traits[:3]) if traits else "Limited signal",
            "Business interpretation": _cluster_interpretation(name),
            "Recommended action": _cluster_action(name),
            "Confidence/limitations": "Suggestive clustering based on available non-sensitive features.",
        }
        rows.append(row)
    return cluster_ids.map(lambda item: lookup[int(item)]), pd.DataFrame(rows), explanations


def _cluster_name(flags: dict[str, bool], cluster_id: int, mode: str) -> str:
    if mode in {"b2b", "mixed_b2b_b2c"} and flags["b2b"] and flags["value"]:
        return "B2B High-Fit Strategic Accounts"
    if mode in {"b2b", "mixed_b2b_b2c"} and flags["b2b"]:
        return "B2B Small Low-Fit Accounts"
    if flags["value"] and flags["engagement"] and not flags["risk"]:
        return "High-Value Loyal Customers"
    if flags["engagement"] and not flags["value"]:
        return "Active Browsers with Low Conversion"
    if flags["value"] and flags["risk"]:
        return "At-Risk Former Buyers"
    if flags["risk"]:
        return "High-Intent Cart Abandoners"
    if flags["psychographic"]:
        return "Price-Sensitive Promotion Responders"
    return f"Behavioural Cluster {cluster_id + 1}"


def _cluster_interpretation(name: str) -> str:
    if "B2B" in name:
        return "Account-level profile should be used for ICP and account-based marketing decisions."
    if "At-Risk" in name:
        return "This group combines value potential with lapse or friction risk."
    if "Active Browsers" in name:
        return "This group shows behavioural intent but needs conversion-friction diagnosis."
    return "This group should be interpreted as a behavioural pattern, not a causal explanation."


def _cluster_action(name: str) -> str:
    if "Strategic" in name:
        return "Prioritise executive outreach, renewal planning, and expansion plays."
    if "At-Risk" in name:
        return "Use win-back, service recovery, and reason-for-lapse research."
    if "Cart" in name or "Browsers" in name:
        return "Improve funnel trust, checkout, reminders, and conversion nudges."
    return "Use lifecycle messaging and collect richer motivation/value data."

