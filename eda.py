
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("chandigarh_public_transport.csv")

print("\n========== DATASET OVERVIEW ==========")

# --------------------------------
# 2. Dataset Shape
# --------------------------------

print("\nDataset Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# --------------------------------
# 3. First and Last Records
# --------------------------------

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== LAST 5 RECORDS ==========")
print(df.tail())

# --------------------------------
# 4. Dataset Information
# --------------------------------

print("\n========== DATASET INFORMATION ==========")
df.info()

# --------------------------------
# 5. Column Names
# --------------------------------

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# --------------------------------
# 6. Missing Values
# --------------------------------

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing[missing > 0])

print("\nTotal Missing Values:", df.isnull().sum().sum())

# --------------------------------
# 7. Duplicate Records
# --------------------------------

print("\n========== DUPLICATE RECORDS ==========")

print("Duplicate Rows:", df.duplicated().sum())

# --------------------------------
# 8. Statistical Summary
# --------------------------------

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

# --------------------------------
# 9. Categorical Column Summary
# --------------------------------

print("\n========== CATEGORICAL SUMMARY ==========")

print(df.describe(include="object"))

# --------------------------------
# 10. Unique Values
# --------------------------------

print("\n========== UNIQUE VALUES ==========")

for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique()} unique values")

# --------------------------------
# 11. Numerical Distribution
# --------------------------------

print("\n========== NUMERICAL DISTRIBUTION ==========")

print(df.select_dtypes(include=np.number).hist(
    figsize=(14, 10),
    bins=20
))

plt.tight_layout()
plt.show()

# --------------------------------
# 12. Passenger Demand Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["passenger_count"].dropna(),
    bins=30,
    kde=True
)

plt.title("Passenger Demand Distribution")
plt.xlabel("Passenger Count")
plt.ylabel("Frequency")

plt.show()

# --------------------------------
# 13. Route-wise Passenger Demand
# --------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="route_id",
    y="passenger_count",
    errorbar=None
)

plt.title("Average Passenger Demand by Route")
plt.xlabel("Route")
plt.ylabel("Average Passengers")

plt.show()

# --------------------------------
# 14. Hour-wise Passenger Demand
# --------------------------------

hourly_demand = df.groupby("hour")["passenger_count"].mean()

plt.figure(figsize=(10, 5))

hourly_demand.plot(kind="line", marker="o")

plt.title("Passenger Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Passengers")
plt.grid()

plt.show()

print("\n========== EDA COMPLETED ==========")
