import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# LOAD AND CLEAN FEATURES DATA

df = pd.read_csv("features.csv")

# Identify target (taxi_time)
TARGET = "taxi_time"

# Identify numerical columns (PCA only works on numeric)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target from PCA feature list
numeric_features = [c for c in numeric_cols if c != TARGET]

X = df[numeric_features]
y = df[TARGET]

# Handle missing values (mean impute)
X = X.fillna(X.mean())

# TRAIN / VAL / TEST SPLIT (70 / 20 / 10)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, shuffle=True, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1/3), shuffle=True, random_state=42
)

# STANDARDISE (fit only on training)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Also scale the full dataset (transform only)
X_full_scaled = scaler.transform(X)

# PCA FIT ON TRAINING ONLY

pca = PCA().fit(X_train_scaled)

expl_var = pca.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

# DETERMINE NUMBER OF COMPONENTS (retain 95% variance)

n_components = np.argmax(cum_var >= 0.95) + 1
print(f"Selected number of components: {n_components}")

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)
X_full_pca = pca.transform(X_full_scaled)

# FEATURE IMPORTANCE (ABSOLUTE PCA LOADINGS)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(n_components)],
    index=numeric_features
)

# Importance = sum of absolute loadings across retained PCs
feature_importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
feature_importance = feature_importance.to_frame("importance")

feature_importance.to_csv("feature_importance.csv")

print("\nTop Features Driving PCA:")
print(feature_importance.head(10))

# CUMULATIVE VARIANCE TABLE

cum_var_table = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(expl_var))],
    "ExplainedVariance": expl_var,
    "CumulativeVariance": cum_var
})

cum_var_table.to_csv("pca_cumulative_variance_table.csv", index=False)

print("\nCumulative Variance Table:")
print(cum_var_table)

# BAR CHART OF CUMULATIVE VARIANCE

plt.figure(figsize=(10,6))
plt.bar(range(1, len(cum_var)+1), cum_var)
plt.axhline(0.95, color='r', linestyle='--')
plt.title("Cumulative Variance by Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.grid(axis="y")
plt.savefig("pca_cumulative_variance_bar.png", dpi=200)

# SAVE REDUCED SPLIT DATASETS

train_out = pd.DataFrame(X_train_pca, columns=[f"PC{i+1}" for i in range(n_components)])
train_out[TARGET] = y_train.values
train_out.to_csv("train_reduced.csv", index=False)

val_out = pd.DataFrame(X_val_pca, columns=[f"PC{i+1}" for i in range(n_components)])
val_out[TARGET] = y_val.values
val_out.to_csv("validation_reduced.csv", index=False)

test_out = pd.DataFrame(X_test_pca, columns=[f"PC{i+1}" for i in range(n_components)])
test_out[TARGET] = y_test.values
test_out.to_csv("test_reduced.csv", index=False)

# SAVE FULL DATASET (NO SPLIT)

full_out = pd.DataFrame(X_full_pca, columns=[f"PC{i+1}" for i in range(n_components)])
full_out[TARGET] = y.values
full_out.to_csv("full_reduced.csv", index=False)

# OPTIONAL: save scaled full dataset
full_scaled_df = pd.DataFrame(X_full_scaled, columns=numeric_features)
full_scaled_df[TARGET] = y.values
full_scaled_df.to_csv("full_scaled.csv", index=False)

# SCREE PLOT

plt.figure(figsize=(8,5))
plt.plot(range(1, len(expl_var)+1), cum_var, marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.savefig("pca_scree_plot.png", dpi=200)

print("\nPCA analysis complete.")
print("Files written:")
print("- train_reduced.csv")
print("- validation_reduced.csv")
print("- test_reduced.csv")
print("- full_reduced.csv")
print("- full_scaled.csv")
print("- feature_importance.csv")
print("- pca_scree_plot.png")
print("- pca_cumulative_variance_bar.png")
print("- pca_cumulative_variance_table.csv")
