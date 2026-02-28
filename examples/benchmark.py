import time

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from striders import Striders
from ucimlrepo import fetch_ucirepo


def benchmark(x, y_pred, model):
    """
    Runs a comparison between TreeSHAP (Reference) and Striders (Rust Engine).
    Measures execution time, Speed-up, Fidelity (R^2), and Additive Property.
    """
    # --- 1. Reference: TreeSHAP (Standard Explanation) ---
    start_time = time.time()
    explainer_tree = shap.TreeExplainer(model)
    shap_tree = explainer_tree.shap_values(x)
    tree_shap_time = time.time() - start_time
    print(f"TreeSHAP execution time: {tree_shap_time:.4f}s")

    # --- 2. Target: Striders (High-Performance Rust Engine) ---
    # Parameters: m_landmarks (complexity), sigma (kernel width), lambda_reg (regularization)
    strider_model = Striders(m_landmarks=50, sigma=1.0, lambda_reg=0.01)

    start_time = time.time()
    # Fit the surrogate model to the black-box predictions
    strider_model.fit(x, y_pred)
    # Generate SHAP values and internal surrogate predictions
    strider_preds, strider_shaps = strider_model.explain(x)
    strider_time = time.time() - start_time

    # Calculate Fidelity: How well the Rust surrogate mimics the original model
    r2 = r2_score(y_pred, strider_preds)

    print(f"Striders execution time: {strider_time:.4f}s")
    print(f"Speed up: {tree_shap_time / strider_time:.1f}x")
    print(f"Fidelity (R^2): {r2:.4f}")

    # --- 3. Consistency Check: Additive Property ---
    # For Kernel-based methods: Sum(SHAP) + E[y] should approximately equal the prediction
    expected_value = np.mean(y_pred)
    sample_idx = 0
    sum_shaps = np.sum(strider_shaps[sample_idx]) + expected_value
    actual_pred = strider_preds[sample_idx]

    print(f"Additive Check (Sample {sample_idx}):")
    print(f"   - Sum of SHAPs + Base: {sum_shaps:.4f}")
    print(f"   - Striders Prediction: {actual_pred:.4f}")

    return shap_tree, strider_shaps


# =================================================================
# SCENARIO 1: Regression (California Housing Dataset)
# =================================================================
print("\n--- Testing Scenario: Regression (California Housing) ---")
housing = fetch_california_housing()
x_raw = pd.DataFrame(housing.data, columns=housing.feature_names)
y_raw = housing.target

# Standardizing inputs is critical for Kernel/RBF methods
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw).astype(np.float32)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
rf.fit(x_scaled, y_raw)
y_pred = rf.predict(x_scaled).astype(np.float32)

shap_tree, strider_shaps = benchmark(x_scaled, y_pred, rf)

# Pearson Correlation: Measures how similar the explanations are to TreeSHAP
corr, _ = pearsonr(shap_tree.flatten(), strider_shaps.flatten())
print(f"   - Pearson Correlation with TreeSHAP: {corr:.4f}")


# =================================================================
# SCENARIO 2: Classification (UCI Credit Default Dataset)
# =================================================================
print("\n--- Testing Scenario: Classification (Credit Default) ---")
credit_default = fetch_ucirepo(id=350)
x_raw = credit_default.data.features
y_true = credit_default.data.targets.values.ravel()

# Handling missing values and scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw.fillna(x_raw.mean())).astype(np.float32)

clf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
clf.fit(x_scaled, y_true)

# For classification, we explain the probability of the positive class (Class 1)
y_pred_proba = clf.predict_proba(x_scaled)[:, 1].astype(np.float32)

shap_tree, strider_shaps = benchmark(x_scaled, y_pred_proba, clf)

# Note: TreeExplainer returns [class_0_shaps, class_1_shaps] for classifiers.
# We compare our results with class_1 (index 1) to match y_pred_proba.
if isinstance(shap_tree, list):
    # Older SHAP versions return a list
    shap_tree_pos = shap_tree[1]
else:
    # Modern SHAP versions return (N, P, 2) array
    shap_tree_pos = shap_tree[:, :, 1]

corr, _ = pearsonr(shap_tree_pos.flatten(), strider_shaps.flatten())
print(f"   - Pearson Correlation with TreeSHAP: {corr:.4f}")
