import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# CONFIGURATION
DATA_FOLDER = "./"
PATTERN = "features_*.csv"
TARGET = "taxi_time"

MIN_VARIANCE = 1e-6
CORR_THRESHOLD = 0.95

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

FINAL_SIZE = 2500


# LOAD DATA
files = sorted(glob.glob(os.path.join(DATA_FOLDER, PATTERN)))
if not files:
    raise ValueError("No feature files found.")

print("\nLoading Raw Feature Files")
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Combined dataset rows: {len(df)}\n")


# BASIC CLEANING — DROP NaN ROWS
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET not in numeric_cols:
    raise ValueError(f"TARGET '{TARGET}' not found.")

numeric_features = [c for c in numeric_cols if c != TARGET]

print("Removing Missing Value Rows")
clean_df = df.dropna(axis=0)
print(f"Rows after dropping NaNs: {len(clean_df)}\n")


# STEP 1 — WINSORISE FEATURES AND TARGET BEFORE STRATIFICATION

print("Winsorising Features (X)")
X = clean_df[numeric_features].copy()
X_wins = X.copy()

for col in numeric_features:
    lo = X[col].quantile(WINSOR_LOWER)
    hi = X[col].quantile(WINSOR_UPPER)
    X_wins[col] = np.clip(X[col], lo, hi)

X = X_wins

print("Winsorising TARGET (y)")
y = clean_df[TARGET].copy()
y_lo = y.quantile(WINSOR_LOWER)
y_hi = y.quantile(WINSOR_UPPER)

print(f"Target winsor bounds: [{y_lo:.2f}, {y_hi:.2f}]")
y = np.clip(y, y_lo, y_hi)

clean_df[numeric_features] = X
clean_df[TARGET] = y


# STEP 2 — STRATIFIED DOWNSAMPLING BY TAXI-TIME QUANTILES

print("\nStratified Sampling by Taxi-Time Quantiles")

# 10-bin quantile stratification
clean_df["time_bin"] = pd.qcut(clean_df[TARGET], q=10, duplicates="drop")

groups = clean_df.groupby("time_bin")
counts = groups.size()

frac = FINAL_SIZE / len(clean_df)

sampled = groups.apply(lambda g: g.sample(frac=frac, random_state=42))
sampled = sampled.reset_index(drop=True)

print(f"Final sampled rows: {len(sampled)}\n")

clean_df = sampled.drop(columns=["time_bin"], errors="ignore")


# STEP 3 — LOW VARIANCE FILTERING

print("Variance Screening")
X = clean_df[numeric_features].copy()
y = clean_df[TARGET].copy()

variances = X.var()
low_var = variances[variances < MIN_VARIANCE].index.tolist()

print(f"Removed low-variance (<{MIN_VARIANCE}) features: {low_var}")
X = X.drop(columns=low_var, errors="ignore")
numeric_features = X.columns.tolist()


# STEP 4 — CORRELATION PRUNING

print("\nCorrelation Screening")
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
print(f"Removed correlated features (>|{CORR_THRESHOLD}|): {to_drop}")

X = X.drop(columns=to_drop, errors="ignore")
numeric_features = X.columns.tolist()


# STEP 5 — SAVE CLEANED DATASET

cleaned_df = X.copy()
cleaned_df[TARGET] = y
cleaned_df.to_csv("features_cleaned.csv", index=False)

print("Cleaning Complete")
print(f"Output file: features_cleaned.csv")
print(f"Rows: {len(cleaned_df)}")
print(f"Features: {len(numeric_features)}")


# OPTIONAL: VARIANCE PLOT
plt.figure(figsize=(12,4))
plt.title("Feature Variance (After Cleaning)")
plt.bar(range(len(numeric_features)), cleaned_df[numeric_features].var())
plt.xticks(range(len(numeric_features)), numeric_features, rotation=90)
plt.tight_layout()
plt.savefig("feature_variance_plot.png")
plt.close()
