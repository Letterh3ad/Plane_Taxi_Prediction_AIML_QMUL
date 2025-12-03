import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# CONFIG
DATA_FOLDER = "./"
PATTERN = "features_*.csv"
TARGET = "taxi_time"

# Variance threshold for removing extremely low-variance features
MIN_VARIANCE = 1e-6

# Correlation threshold for removing redundant features
CORR_THRESHOLD = 0.95

# Winsorisation limits for extreme outliers
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


# LOAD RAW DATA
files = sorted(glob.glob(os.path.join(DATA_FOLDER, PATTERN)))
if not files:
    raise ValueError("No feature files found.")

print("\n Loading Raw Feature Files ")
dfs = []
for fname in files:
    print(f"  Loaded: {fname}")
    dfs.append(pd.read_csv(fname))

df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"\nCombined dataset rows: {len(df)}")

# Keep only numeric features + target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET not in numeric_cols:
    raise ValueError("TARGET column not found or not numeric.")

numeric_features = [c for c in numeric_cols if c != TARGET]

X = df[numeric_features].copy()
y = df[TARGET].copy()


# 1. HANDLE MISSING VALUES
print("\n=== Removing Missing Value Rows ===")
missing_before = df.isna().sum().sum()
print(f"Missing values before: {missing_before}")

# Drop any row containing NaN in features or target
clean_df = df.dropna(axis=0, how="any")

print(f"Rows before drop: {len(df)}")
print(f"Rows after drop: {len(clean_df)}")
print(f"Dropped rows: {len(df) - len(clean_df)}")

# Re-split into X and y after drop
X = clean_df[numeric_features].copy()
y = clean_df[TARGET].copy()


# 2. REMOVE NEAR-ZERO VARIANCE FEATURES
print("\n Variance Screening ")
variances = X.var()

low_var_features = variances[variances < MIN_VARIANCE].index.tolist()
print(f"Features removed due to low variance (<{MIN_VARIANCE}): {low_var_features}")

X = X.drop(columns=low_var_features, errors="ignore")

numeric_features = [c for c in X.columns]


# 3. WINSORISATION OF OUTLIERS
print("\n Winsorising Outliers ")
X_wins = X.copy()

for col in numeric_features:
    lower = X[col].quantile(WINSOR_LOWER)
    upper = X[col].quantile(WINSOR_UPPER)
    X_wins[col] = np.clip(X[col], lower, upper)

X = X_wins


# 4. REMOVE HIGHLY CORRELATED FEATURES
print("\n Correlation Pruning ")

corr_matrix = X.corr().abs()

# Upper triangle mask
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]

print(f"Correlated features removed (>|{CORR_THRESHOLD}|): {to_drop}")

X = X.drop(columns=to_drop, errors="ignore")

numeric_features = [c for c in X.columns]


# 5. FINAL CLEANED DATASET
cleaned_df = X.copy()
cleaned_df[TARGET] = y

cleaned_df.to_csv("features_cleaned.csv", index=False)

print("\n Cleaning Complete ")
print(f"Output file: features_cleaned.csv")
print(f"Remaining features: {len(numeric_features)}")
print(f"Total rows: {len(cleaned_df)}")


# 6. OPTIONAL: PLOT VARIANCE BEFORE/AFTER
plt.figure(figsize=(12,4))
plt.title("Feature Variance (After Cleaning)")
plt.bar(range(len(numeric_features)), cleaned_df[numeric_features].var())
plt.xticks(range(len(numeric_features)), numeric_features, rotation=90)
plt.tight_layout()
plt.savefig("feature_variance_plot.png")
