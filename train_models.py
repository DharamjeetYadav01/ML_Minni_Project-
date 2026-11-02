
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

from xgboost import XGBRegressor


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("chandigarh_cleaned.csv")

print("\n========== MACHINE LEARNING ==========")
print("Dataset Shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGETS
# ==========================================

categorical_features = [
    "day_of_week",
    "route_id",
    "route_name",
    "origin_stop",
    "destination_stop",
    "weather",
    "special_event",
    "hour_category"
]

numerical_features = [
    "hour",
    "is_weekend",
    "is_holiday",
    "scheduled_buses",
    "actual_buses",
    "temperature_c",
    "rainfall_mm",
    "bus_capacity"
]

features = categorical_features + numerical_features

X = df[features]

# Regression target
y_reg = df["passenger_count"]

# Classification target
y_clf = df["crowd_level"]


# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================

numeric_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 4. REGRESSION TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 5. RANDOM FOREST REGRESSOR
# ==========================================

print("\n========== RANDOM FOREST ==========")

rf_model = Pipeline(steps=[
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 4))


# ==========================================
# 6. XGBOOST REGRESSOR
# ==========================================

print("\n========== XGBOOST ==========")

xgb_model = Pipeline(steps=[
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )
])

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)

print("MAE:", round(xgb_mae, 2))
print("RMSE:", round(xgb_rmse, 2))
print("R2 Score:", round(xgb_r2, 4))


# ==========================================
# 7. DECISION TREE CLASSIFIER
# ==========================================

print("\n========== DECISION TREE CLASSIFIER ==========")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

dt_model = Pipeline(steps=[
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
    )
])

dt_model.fit(
    X_train_c,
    y_train_c
)

dt_predictions = dt_model.predict(
    X_test_c
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test_c,
            dt_predictions
        ),
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test_c,
        dt_predictions,
        zero_division=0
    )
)

class_labels = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_c,
        dt_predictions,
        labels=class_labels
    )
)


# ==========================================
# 8. K-MEANS CLUSTERING
# ==========================================

print("\n========== K-MEANS CLUSTERING ==========")

cluster_features = [
    "hour",
    "passenger_count",
    "scheduled_buses",
    "actual_buses",
    "occupancy_rate"
]

# Keep valid numerical observations.
# Imputer also handles any remaining missing values.

cluster_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "kmeans",
        KMeans(
            n_clusters=4,
            random_state=42,
            n_init=10
        )
    )
])

cluster_data = df[cluster_features].replace(
    [np.inf, -np.inf],
    np.nan
)

cluster_labels = cluster_pipeline.fit_predict(
    cluster_data
)

df["demand_cluster"] = cluster_labels

print("\nCluster Distribution:")

print(
    df["demand_cluster"].value_counts().sort_index()
)

print("\nCluster Profiles:")

print(
    df.groupby("demand_cluster")[
        [
            "passenger_count",
            "hour",
            "occupancy_rate"
        ]
    ].mean().round(2)
)

# Silhouette score on a sample for evaluation

sample_size = min(1000, len(cluster_data))

sample_indices = np.random.default_rng(42).choice(
    len(cluster_data),
    size=sample_size,
    replace=False
)

sample_transformed = cluster_pipeline.named_steps[
    "scaler"
].transform(
    cluster_pipeline.named_steps[
        "imputer"
    ].transform(
        cluster_data.iloc[sample_indices]
    )
)

sample_labels = cluster_labels[sample_indices]

silhouette = silhouette_score(
    sample_transformed,
    sample_labels
)

print("\nSilhouette Score:", round(silhouette, 4))


# ==========================================
# 9. SAVE TRAINED MODELS
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    rf_model,
    "models/demand_rf.pkl"
)

joblib.dump(
    xgb_model,
    "models/demand_xgb.pkl"
)

joblib.dump(
    dt_model,
    "models/crowd_classifier.pkl"
)

joblib.dump(
    cluster_pipeline,
    "models/kmeans_model.pkl"
)

print("\nAll four models saved successfully!")


# ==========================================
# 10. SAVE MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "MAE": [
        rf_mae,
        xgb_mae
    ],
    "RMSE": [
        rf_rmse,
        xgb_rmse
    ],
    "R2 Score": [
        rf_r2,
        xgb_r2
    ]
})

results.to_csv(
    "model_comparison.csv",
    index=False
)

# Save classifier metrics separately

report = classification_report(
    y_test_c,
    dt_predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    "classification_report.csv"
)

# Save cluster-labelled data

df.to_csv(
    "chandigarh_clustered.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results)

print("\nMACHINE LEARNING TRAINING COMPLETED!")
