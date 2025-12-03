import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# CONFIG

DATASETS = [
    "features_cleaned.csv"
]

TARGET = "taxi_time"
TEST_SIZE = 0.30
VAL_SPLIT = 1/3
VAR_THRESHOLD = 0.95


# LOAD & CONCAT MULTIPLE DATASETS

dfs = []
for fname in DATASETS:
    print(f"Loading {fname}")
    dfs.append(pd.read_csv(fname))

df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Final dataset rows: {len(df)}")


# CLEANING

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [c for c in numeric_cols if c != TARGET]

X = df[numeric_features].copy()
y = df[TARGET].copy()

X = X.fillna(X.mean())


# TRAIN / VAL / TEST SPLIT

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=True, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=VAL_SPLIT, shuffle=True, random_state=42
)


# SCALING

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
X_full_scaled  = scaler.transform(X)


# SAVE SCALED DATASETS

def save_scaled(name, X_scaled, y_vals):
    df_scaled = pd.DataFrame(X_scaled, columns=numeric_features)
    df_scaled[TARGET] = y_vals.values
    df_scaled.to_csv(name, index=False)

save_scaled("scaled_train.csv", X_train_scaled, y_train)
save_scaled("scaled_val.csv",   X_val_scaled,   y_val)
save_scaled("scaled_test.csv",  X_test_scaled,  y_test)
save_scaled("scaled_full.csv",  X_full_scaled,  y)

print("Scaled datasets written.")


# PCA (fit only on TRAIN scaled)

pca_full = PCA().fit(X_train_scaled)
expl_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

n_components = np.argmax(cum_var >= VAR_THRESHOLD) + 1
print(f"PCA components retained (≥{VAR_THRESHOLD*100}% variance): {n_components}")

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)
X_full_pca  = pca.transform(X_full_scaled)


# SAVE PCA-REDUCED DATASETS

pc_cols = [f"PC{i+1}" for i in range(n_components)]

def save_reduced(name, X_red, y_vals):
    df_red = pd.DataFrame(X_red, columns=pc_cols)
    df_red[TARGET] = y_vals.values
    df_red.to_csv(name, index=False)

save_reduced("reduced_train.csv", X_train_pca, y_train)
save_reduced("reduced_val.csv",   X_val_pca,   y_val)
save_reduced("reduced_test.csv",  X_test_pca,  y_test)
save_reduced("reduced_full.csv",  X_full_pca,  y)

print("Reduced datasets written.")


# NEW: EXPORT BOTH FULL-SCALED AND PCA-REDUCED VERSIONS OF THE DATASET

df_scaled_full = pd.DataFrame(X_full_scaled, columns=numeric_features)
df_scaled_full[TARGET] = y.values
df_scaled_full.to_csv("combined_scaled.csv", index=False)

df_reduced_full = pd.DataFrame(X_full_pca, columns=pc_cols)
df_reduced_full[TARGET] = y.values
df_reduced_full.to_csv("combined_reduced.csv", index=False)

print("Combined scaled and reduced datasets written.")


# FEATURE IMPORTANCE + VARIANCE TABLE

loadings = pd.DataFrame(
    pca.components_.T,
    columns=pc_cols,
    index=numeric_features
)

feature_importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
feature_importance.to_csv("feature_importance.csv")

cum_var_table = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(expl_var))],
    "ExplainedVariance": expl_var,
    "CumulativeVariance": cum_var
})
cum_var_table.to_csv("pca_cumulative_variance_table.csv", index=False)


# PLOTS

plt.figure(figsize=(10,6))
plt.bar(range(1, len(cum_var)+1), cum_var)
plt.axhline(VAR_THRESHOLD, color='r', linestyle='--')
plt.title("Cumulative Variance by Principal Component")
plt.xlabel("PC")
plt.ylabel("Cumulative Explained Variance")
plt.grid(axis="y")
plt.savefig("pca_cumulative_variance_bar.png", dpi=200)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(VAR_THRESHOLD, color='r', linestyle='--')
plt.title("Cumulative Explained Variance")
plt.xlabel("PC count")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.savefig("pca_scree_plot.png", dpi=200)

print("\nAll processing complete.")
