import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, f_oneway
# ---------------------------------------------------------
# 1. LOAD MODEL + DATA
# ---------------------------------------------------------
best_model_path = r"C:\Users\saumy\OneDrive\Desktop\tbi_pipeline_final_codes\notebooks\output\models\best_model_final.pkl"
best_model_instance = joblib.load(best_model_path)

filepath = r"../data/processed/df17nov.csv"
df = pd.read_csv(filepath)

target_col = 'FIM_change'
X = df.drop(columns=[target_col])
y = df[target_col]

# Ensure feature alignment
model_features = best_model_instance.feature_names_in_
X = X[model_features]

# ---------------------------------------------------------
# 2. CREATE AGE GROUPS
# ---------------------------------------------------------
age_bins = [0, 30, 50, 100]
age_labels = ['<30', '30-50', '50+']
df['AgeGroup'] = pd.cut(df['AGENoPHI'], bins=age_bins, labels=age_labels)

# ---------------------------------------------------------
# 3. ORIGINAL MODEL PREDICTIONS
# ---------------------------------------------------------
df['y_pred_original'] = best_model_instance.predict(X)
df['Residual_original'] = df['y_pred_original'] - y

# ---------------------------------------------------------
# 4. COMPUTE MEAN RESIDUAL PER AGE GROUP
# ---------------------------------------------------------
age_bias = (
    df.groupby('AgeGroup')['Residual_original']
    .mean()
    .rename("MeanResidual")
)

print("\n=== Mean residual (bias) before calibration ===")
print(age_bias)
print()

# ---------------------------------------------------------
# 5. APPLY GROUP-WISE CALIBRATION (SAFE VERSION)
# ---------------------------------------------------------
def calibrate_row(row, bias_series):
    if pd.isna(row['AgeGroup']):
        # Leave prediction unchanged if AgeGroup is missing
        return row['y_pred_original']
    else:
        return row['y_pred_original'] - bias_series[row['AgeGroup']]

df['y_pred_calibrated'] = df.apply(lambda row: calibrate_row(row, age_bias), axis=1)
df['Residual_calibrated'] = df['y_pred_calibrated'] - y

# ---------------------------------------------------------
# 6. RE-RUN AGEGROUP ANOVA TO CHECK IMPROVEMENT
# ---------------------------------------------------------
groups_original = [
    df[df['AgeGroup'] == grp]['Residual_original']
    for grp in age_labels
]

groups_calibrated = [
    df[df['AgeGroup'] == grp]['Residual_calibrated']
    for grp in age_labels
]

f_orig, p_orig = f_oneway(*groups_original)
f_cal, p_cal = f_oneway(*groups_calibrated)

print("=== ANOVA Before Calibration ===")
print(f"F = {f_orig:.3f}, p = {p_orig:.4f}\n")

print("=== ANOVA After Calibration ===")
print(f"F = {f_cal:.3f}, p = {p_cal:.4f}\n")

# ---------------------------------------------------------
# 7. SAVE OUTPUTS
# ---------------------------------------------------------
df[['y_pred_original', 'y_pred_calibrated', 'Residual_original', 'Residual_calibrated']].to_csv(
    "output/agegroup_calibration_output.csv",
    index=False
)

age_bias.to_csv("output/agegroup_bias_values.csv")

print("Saved calibrated predictions and bias values.")
print("\n Calibration script finished successfully.")
