
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("chandigarh_public_transport.csv")

print("\n========== DATA PREPROCESSING ==========")
print("Original Dataset Shape:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates().copy()


# ==========================================
# 3. HANDLE EMPTY STRINGS
# ==========================================

df = df.replace(r"^\s*$", np.nan, regex=True)


# ==========================================
# 4. CLEAN CATEGORICAL COLUMNS
# ==========================================

categorical_cols = [
    "day_of_week",
    "route_id",
    "route_name",
    "origin_stop",
    "destination_stop",
    "weather",
    "special_event"
]

for col in categorical_cols:
    df[col] = df[col].astype("string").str.strip()

df["weather"] = df["weather"].str.title()
df["day_of_week"] = df["day_of_week"].str.title()


# ==========================================
# 5. DATE CONVERSION
# ==========================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Recover day of week from date

df["day_of_week"] = df["date"].dt.day_name().combine_first(
    df["day_of_week"]
)

# Correct weekend indicator where date is available

valid_dates = df["date"].notna()

df.loc[valid_dates, "is_weekend"] = (
    df.loc[valid_dates, "date"].dt.dayofweek >= 5
).astype(int)


# ==========================================
# 6. NUMERICAL CONVERSION
# ==========================================

numeric_cols = [
    "hour",
    "is_weekend",
    "is_holiday",
    "scheduled_buses",
    "actual_buses",
    "temperature_c",
    "rainfall_mm",
    "passenger_count",
    "bus_capacity"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ==========================================
# 7. HANDLE INVALID VALUES
# ==========================================

# Passenger count

df.loc[
    (df["passenger_count"] < 0) |
    (df["passenger_count"] > 200),
    "passenger_count"
] = np.nan

# Bus capacity

df.loc[
    (df["bus_capacity"] <= 0) |
    (df["bus_capacity"] > 100),
    "bus_capacity"
] = np.nan

# Temperature

df.loc[
    (df["temperature_c"] < -10) |
    (df["temperature_c"] > 55),
    "temperature_c"
] = np.nan

# Rainfall

df.loc[
    (df["rainfall_mm"] < 0) |
    (df["rainfall_mm"] > 100),
    "rainfall_mm"
] = np.nan

# Actual buses

df.loc[
    (df["actual_buses"] < 0) |
    (df["actual_buses"] > 20),
    "actual_buses"
] = np.nan

# Scheduled buses

df.loc[
    (df["scheduled_buses"] <= 0) |
    (df["scheduled_buses"] > 20),
    "scheduled_buses"
] = np.nan

# Hour

df.loc[
    (df["hour"] < 0) |
    (df["hour"] > 23),
    "hour"
] = np.nan


# ==========================================
# 8. HANDLE MISSING VALUES
# ==========================================

# Do not impute the target variable

df = df.dropna(
    subset=["passenger_count"]
).copy()

# Numerical columns

numeric_impute_cols = [
    "hour",
    "temperature_c",
    "rainfall_mm",
    "actual_buses",
    "scheduled_buses",
    "bus_capacity",
    "is_weekend",
    "is_holiday"
]

for col in numeric_impute_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )

# Categorical columns

for col in categorical_cols:
    if df[col].notna().any():
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )


# ==========================================
# 9. FEATURE ENGINEERING
# ==========================================

df["hour_category"] = pd.cut(
    df["hour"],
    bins=[0, 7, 11, 16, 20, 24],
    labels=[
        "Early Morning",
        "Morning Peak",
        "Afternoon",
        "Evening Peak",
        "Night"
    ],
    right=False
)

# Total available passenger capacity

df["available_capacity"] = (
    df["bus_capacity"] * df["actual_buses"]
)

# Avoid division by zero

df["occupancy_rate"] = np.where(
    df["available_capacity"] > 0,
    df["passenger_count"] / df["available_capacity"],
    np.nan
)

# Remove infinity

df["occupancy_rate"] = df[
    "occupancy_rate"
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ==========================================
# 10. CROWD LEVEL CLASSIFICATION
# ==========================================

def classify_crowd(rate):

    if pd.isna(rate):
        return "Critical"

    elif rate < 0.50:
        return "Low"

    elif rate < 0.80:
        return "Medium"

    elif rate < 1.00:
        return "High"

    else:
        return "Critical"


df["crowd_level"] = df[
    "occupancy_rate"
].apply(classify_crowd)


# ==========================================
# 11. FINAL DATASET CHECK
# ==========================================

print("\n========== PREPROCESSING RESULTS ==========")

print("Final Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nDuplicate Records:", df.duplicated().sum())

print("\nCrowd Distribution:")
print(df["crowd_level"].value_counts())

print("\nOccupancy Statistics:")
print(df["occupancy_rate"].describe())


# ==========================================
# 12. SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    "chandigarh_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
