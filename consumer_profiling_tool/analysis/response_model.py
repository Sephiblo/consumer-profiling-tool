"""Optional binary response prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.models import ConfirmedFieldMapping, ResponseModelResult


def run_response_model(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> tuple[ResponseModelResult | None, pd.Series | None]:
    """Fit an interpretable logistic response model when a binary target exists."""
    target_mapping = next((mapping for mapping in mappings if mapping.role == "binary_target"), None)
    if not target_mapping or target_mapping.name not in df.columns:
        return None, None

    target = _binary_target(df[target_mapping.name])
    if target.nunique(dropna=True) != 2:
        return (
            ResponseModelResult(
                target_field=target_mapping.name,
                metrics={},
                warnings=["Target field is not cleanly binary after coercion."],
            ),
            None,
        )

    excluded_roles = {"customer_id", "free_text", "ignore", "existing_segment", "date_or_time", "binary_target", "numeric_target"}
    feature_columns = [
        mapping.name
        for mapping in mappings
        if mapping.name in df.columns and mapping.name != target_mapping.name and mapping.role not in excluded_roles
        and not mapping.is_sensitive_candidate
    ]
    if not feature_columns:
        return (
            ResponseModelResult(
                target_field=target_mapping.name,
                metrics={},
                warnings=["No usable feature columns were available for response modelling."],
            ),
            None,
        )

    valid_target = target.notna()
    X = df.loc[valid_target, feature_columns]
    y = target.loc[valid_target].astype(int)
    if len(y) < 20 or y.nunique() != 2:
        return (
            ResponseModelResult(
                target_field=target_mapping.name,
                metrics={},
                warnings=["Not enough valid binary target rows were available for response modelling."],
            ),
            None,
        )
    positive_rate = y.mean()
    warnings: list[str] = []
    if positive_rate < 0.1 or positive_rate > 0.9:
        warnings.append("Target is severely imbalanced; model metrics may be unstable.")

    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))

    preprocessor = ColumnTransformer(transformers)
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    if len(df) >= 200:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        }
    else:
        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        }
        warnings.append("Dataset has fewer than 200 rows; metrics are in-sample and should be treated cautiously.")

    all_probabilities = pd.Series(
        model.predict_proba(df[feature_columns])[:, 1],
        index=df.index,
        name="predicted_response_probability",
    )
    positive, negative = _logistic_drivers(model, numeric_features, categorical_features)

    return (
        ResponseModelResult(
            target_field=target_mapping.name,
            metrics={key: (round(value, 4) if value is not None else None) for key, value in metrics.items()},
            warnings=warnings,
            top_positive_drivers=positive,
            top_negative_drivers=negative,
        ),
        all_probabilities,
    )


def _binary_target(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "responded": 1,
        "converted": 1,
        "1": 1,
        "1.0": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "not responded": 0,
        "not converted": 0,
        "0": 0,
        "0.0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(pd.to_numeric(series, errors="coerce"))


def _logistic_drivers(model: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    try:
        preprocessor = model.named_steps["preprocess"]
        feature_names: list[str] = []
        feature_names.extend(numeric_features)
        if categorical_features:
            encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
            feature_names.extend(encoder.get_feature_names_out(categorical_features).tolist())
        coefs = model.named_steps["model"].coef_[0]
        rows = sorted(zip(feature_names, coefs), key=lambda item: item[1], reverse=True)
        positive = [{"feature": name, "coefficient": round(float(coef), 4)} for name, coef in rows[:8]]
        negative = [{"feature": name, "coefficient": round(float(coef), 4)} for name, coef in rows[-8:]]
        return positive, negative
    except Exception:
        return [], []
