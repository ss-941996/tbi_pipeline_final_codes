# ----------------------------------------------------
# Fairness Residual Analysis Script (Python .py-safe)
# ----------------------------------------------------

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, f_oneway


# ====================================================
# 1. LOAD MODEL + DATA
# ====================================================

best_model_path = r"C:\Users\saumy\OneDrive\Desktop\tbi_pipeline_final_codes\notebooks\output\models\best_model_final.pkl"
best_model_instance = joblib.load(best_model_path)

filepath = r"../data/processed/df17nov.csv"
df = pd.read_csv(filepath)

# Define features & target
target_col = 'FIM_change'
X = df.drop(columns=[target_col])
y = df[target_col]

# Align features (important for running .py outside notebook)
model_features = best_model_instance.feature_names_in_
X = X[model_features]


# ====================================================
# 2. DEFINE SUBGROUPS
# ====================================================

# Age groups
age_bins = [0, 30, 50, 100]
age_labels = ['<30', '30-50', '50+']
df['AgeGroup'] = pd.cut(df['AGENoPHI'], bins=age_bins, labels=age_labels)

# Sex
df['Sex'] = df['SexF'].map({1: 'Female', 2: 'Male'})

# Employment categories
emp_map = {
    5: 'Employed', 2: 'Full-Time Student', 3: 'Part-Time Student',
    7: 'House/Family Care', 10: 'Unemployed: Looking', 11: 'Volunteer Work',
    12: 'Retired: Disability', 13: 'Unemployed: Not Looking',
    14: 'Hospitalized Without Pay', 15: 'Retired: Other',
    16: 'On Leave: No Pay', 55: 'Other', 666: 'Did Not Exist',
    777: 'Refused', 888: 'Not Applicable', 9: 'Retired: Age-related'
}
df['Employment'] = df['Emp1'].map(emp_map).fillna('Unknown')

# Income categories
earn_map = {
    1: '0-9,999', 2: '10,000-19,999', 3: '20,000-29,999', 4: '30,000-39,999',
    5: '40,000-49,999', 6: '50,000-59,999', 7: '60,000-69,999',
    8: '70,000-79,999', 9: '80,000-89,999', 10: '90,000-99,999',
    11: '100,000+', 666: 'Did Not Exist', 777: 'Refused',
    888: 'Not Applicable', 999: 'Unknown'
}
df['Income'] = df['Earn'].map(earn_map).fillna('Unknown')

# Education categories
edu_map = {
    1: '1 year or less', 2: '2 years', 3: '3 years', 4: '4 years', 5: '5 years',
    6: '6 years', 7: '7 years', 8: '8 years', 9: '9 years', 10: '10 years',
    11: '11-12 years', 12: 'HS Diploma', 13: 'Associate Work',
    14: 'Associate Degree', 15: 'Bachelors Work', 16: 'Bachelors Degree',
    17: 'Masters Work', 18: 'Masters Degree', 19: 'Doctoral Work',
    20: 'Doctoral Degree', 21: 'Other', 666: 'Did Not Exist', 999: 'Unknown'
}
df['Education'] = df['EduYears'].map(edu_map).fillna('Unknown')

# Subgroup dictionary
subgroups = {
    'AgeGroup': df['AgeGroup'],
    'Sex': df['Sex'],
    'Employment': df['Employment'],
    'Income': df['Income'],
    'Education': df['Education']
}


# ====================================================
# 3. PREDICTIONS & RESIDUALS
# ====================================================

y_pred = best_model_instance.predict(X)
residuals = y_pred - y

df['y_pred'] = y_pred
df['Residuals'] = residuals


# ====================================================
# 4. METRICS PER SUBGROUP
# ====================================================

metrics_list = []

for grp_name, grp_col in subgroups.items():
    for category in grp_col.unique():

        idx = grp_col == category
        y_true_grp = y[idx]
        y_pred_grp = y_pred[idx]
        resid_grp = residuals[idx]

        if len(y_true_grp) == 0:
            continue

        metrics_list.append({
            'Subgroup': grp_name,
            'Category': category,
            'N': len(y_true_grp),
            'MAE': mean_absolute_error(y_true_grp, y_pred_grp),
            'MSE': mean_squared_error(y_true_grp, y_pred_grp),
            'RMSE': np.sqrt(mean_squared_error(y_true_grp, y_pred_grp)),
            'R2': r2_score(y_true_grp, y_pred_grp),
            'MeanResidual': resid_grp.mean(),
            'StdResidual': resid_grp.std()
        })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)


# ====================================================
# 5. VISUALIZATIONS (saved automatically)
# ====================================================

sns.set(style="whitegrid", palette="tab20", font_scale=1.1)
os.makedirs("output/figures", exist_ok=True)

for grp_name in subgroups.keys():
    df_grp = metrics_df[metrics_df['Subgroup'] == grp_name]

    # ----- MAE Plot -----
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Category', y='MAE', data=df_grp, palette="tab20")
    plt.title(f'MAE by {grp_name}')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel(grp_name)
    plt.xticks(rotation=45)
    plt.savefig(f"output/figures/residual_analysis_{grp_name}_MAE.png",
                bbox_inches='tight', dpi=300)
    plt.close()

    # ----- Residuals Plot -----
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Category', y='MeanResidual', data=df_grp, palette="tab20")
    plt.title(f'Mean Residuals by {grp_name} (Bias Check)')
    plt.ylabel('Mean Residual (Pred - True)')
    plt.xlabel(grp_name)
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(rotation=45)
    plt.savefig(f"output/figures/residual_analysis_{grp_name}_Residuals.png",
                bbox_inches='tight', dpi=300)
    plt.close()


# ====================================================
# 6. CONFIRMATORY STATS TESTS
# ====================================================

print("\n=== Confirmatory tests for bias ===\n")

for grp_name, grp_col in subgroups.items():
    categories = grp_col.dropna().unique()
    groups = [residuals[grp_col == c] for c in categories]

    if len(categories) == 2:
        cat1, cat2 = categories
        t_stat, p_val = ttest_ind(groups[0], groups[1], equal_var=False)
        print(f"{grp_name} ({cat1} vs {cat2}): t={t_stat:.3f}, p={p_val:.4f}")

    elif len(categories) > 2:
        f_stat, p_val = f_oneway(*groups)
        cats = ", ".join([str(c) for c in categories])
        print(f"{grp_name} (ANOVA across: {cats}): F={f_stat:.3f}, p={p_val:.4f}")

    else:
        print(f"{grp_name}: Not enough categories to test.")
# ====================================================
# 7. best model parms
# ====================================================


best_model_path = r"C:\Users\saumy\OneDrive\Desktop\tbi_pipeline_final_codes\notebooks\output\models\best_model_final.pkl"
best_model_instance = joblib.load(best_model_path)
params = best_model_instance.get_params()
for k, v in params.items():
    print(f"{k}: {v}")
# ====================================================
# 7. caliberated and non caliberated model perf comparisons 
# ====================================================


from sklearn.utils import resample

# Original and calibrated predictions
y_true = y.to_numpy()
y_pred_orig = df['y_pred_original'].to_numpy()
y_pred_cal  = df['y_pred_calibrated'].to_numpy()

# Bootstrap parameters
B = 5000
mae_diffs, r2_diffs = [], []

for _ in range(B):
    # Resample indices with replacement
    idx = resample(np.arange(len(y_true)), replace=True)
    y_boot = y_true[idx]
    y_orig_boot = y_pred_orig[idx]
    y_cal_boot  = y_pred_cal[idx]
    
    # Compute metrics
    mae_orig = mean_absolute_error(y_boot, y_orig_boot)
    mae_cal  = mean_absolute_error(y_boot, y_cal_boot)
    r2_orig  = r2_score(y_boot, y_orig_boot)
    r2_cal   = r2_score(y_boot, y_cal_boot)
    
    # Store differences (calibrated - original)
    mae_diffs.append(mae_cal - mae_orig)
    r2_diffs.append(r2_cal - r2_orig)

# Confidence intervals
mae_ci = (np.percentile(mae_diffs, 2.5), np.percentile(mae_diffs, 97.5))
r2_ci  = (np.percentile(r2_diffs, 2.5), np.percentile(r2_diffs, 97.5))

print("Bootstrap CI for MAE difference (Cal - Orig):", mae_ci)
print("Bootstrap CI for R² difference (Cal - Orig):", r2_ci)

