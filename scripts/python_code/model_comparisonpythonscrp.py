# 01_modeling_pipeline_comprehensive.py
# ============================================================================
# Purpose: Complete modeling pipeline for TBI functional outcome prediction
# - Load cleaned data from R preprocessing
# - Train 6 models with nested CV and Optuna hyperparameter tuning
# - Perform comprehensive feature importance analysis (PI, Built-in, SHAP)
# - Statistical comparison with bootstrapped confidence intervals
# - Save all results and visualizations
#
# Author: saumya sharma
# Date: 17-11-2025
# Input: data/processed/cleaned_data_for_modeling.csv
# Output: Models, metrics, importance scores, and visualizations
# ============================================================================

# ============================================================================
# 0. INITIAL SETUP
# ============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import optuna
import shap
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create output directories
os.makedirs("output/models", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

print("="*80)
print("TBI FUNCTIONAL OUTCOME PREDICTION - MODELING PIPELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Cleaned Data")
print("="*80)

df = pd.read_csv("../data/processed/df17nov.csv")
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

X_full = df.drop(columns=['Mod1Id', 'FIM_change'])
y_full = df['FIM_change']

print(f"Features: {X_full.shape[1]} variables")
print(f"Target: FIM_change (range: {y_full.min():.1f} to {y_full.max():.1f})")
print(f"Mean FIM_change: {y_full.mean():.2f} ± {y_full.std():.2f}")

# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Train/Test Split")
print("="*80)

X_NCV, X_FINAL_TEST, y_NCV, y_FINAL_TEST = train_test_split(
    X_full, y_full, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
)

print(f"Training Set (NCV): {X_NCV.shape[0]} samples")
print(f"Test Set (Holdout): {X_FINAL_TEST.shape[0]} samples")

# ============================================================================
# 3. DEFINE MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Model Definitions")
print("="*80)

MODEL_DEFINITIONS = {
    'MeanPredictor': DummyRegressor(strategy='mean'),
    'HuberRegressor': HuberRegressor(max_iter=200),
    'XGBRegressor': XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
    'LGBMRegressor': LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1),
    'HistGBMRegressor': HistGradientBoostingRegressor(random_state=RANDOM_SEED),
}

print("Models to train:")
for i, model_name in enumerate(MODEL_DEFINITIONS.keys(), 1):
    print(f"  {i}. {model_name}")

# ============================================================================
# 4. OPTUNA OBJECTIVE FUNCTIONS
# ============================================================================
def objective_huber(trial, X_train, y_train, inner_cv):
    params = {
        'alpha': trial.suggest_float('alpha', 0.0, 1.0),
        'epsilon': trial.suggest_float('epsilon', 1.0, 2.0),
        'max_iter': 1000
    }
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('huber', HuberRegressor(**params))
    ])
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=1)
    return -np.mean(scores)

def objective_xgb(trial, X_train, y_train, inner_cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(**params, n_jobs=1))
    ])
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=1)
    return -np.mean(scores)

def objective_lgbm(trial, X_train, y_train, inner_cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'random_state': RANDOM_SEED,
        'verbosity': -1
    }
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('lgbm', LGBMRegressor(**params, n_jobs=1))
    ])
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=1)
    return -np.mean(scores)

def objective_histgbm(trial, X_train, y_train, inner_cv):
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'random_state': RANDOM_SEED
    }
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('histgbm', HistGradientBoostingRegressor(**params))
    ])
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring='neg_mean_squared_error', cv=inner_cv, n_jobs=1)
    return -np.mean(scores)

OBJECTIVE_MAP = {
    'HuberRegressor': objective_huber,
    'XGBRegressor': objective_xgb,
    'LGBMRegressor': objective_lgbm,
    'HistGBMRegressor': objective_histgbm,
}

# ============================================================================
# 5. NESTED CROSS-VALIDATION WITH OPTUNA
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Nested Cross-Validation (5 Outer × 3 Inner Folds)")
print("="*80)

all_model_results = {}
final_models_for_comparison = {}

for model_name, base_model in MODEL_DEFINITIONS.items():
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")
    
    # Outer CV setup
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    outer_results = []
    best_models = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_NCV, y_NCV), 1):
        print(f"\nOuter Fold {fold_idx}/5...")
        
        X_train, X_test = X_NCV.iloc[train_idx], X_NCV.iloc[test_idx]
        y_train, y_test = y_NCV.iloc[train_idx], y_NCV.iloc[test_idx]
        
        # Inner CV for hyperparameter tuning
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        
        if model_name == 'MeanPredictor':
            # Baseline model - no tuning needed
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('mean', DummyRegressor(strategy='mean'))
            ])
            best_params = {}
        
        else:
            # Hyperparameter tuning with Optuna
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
            )
            
            objective_with_data = lambda trial: OBJECTIVE_MAP[model_name](
                trial, X_train, y_train, inner_cv
            )
            
            study.optimize(objective_with_data, n_trials=20, show_progress_bar=False)
            best_params = study.best_params
            
            # Cast integer parameters
            int_params = ['n_estimators', 'max_depth', 'num_leaves', 'max_iter',
                         'hist_max_iter', 'hist_max_depth', 'lgbm_n_estimators',
                         'lgbm_max_depth', 'lgbm_num_leaves', 'xgb_n_estimators',
                         'xgb_max_depth']
            for k in int_params:
                if k in best_params:
                    best_params[k] = int(best_params[k])
            
            # Build pipeline with best parameters
            if model_name == 'HuberRegressor':
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('huber', HuberRegressor(**best_params, max_iter=1000))
                ])
            
            elif model_name == 'XGBRegressor':
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('xgb', XGBRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=0))
                ])
            
            elif model_name == 'LGBMRegressor':
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler()),
                    ('lgbm', LGBMRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1))
                ])
            
            elif model_name == 'HistGBMRegressor':
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('histgbm', HistGradientBoostingRegressor(**best_params, random_state=RANDOM_SEED))
                ])
            
            
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        best_models.append(pipeline)
        
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        outer_results.append({
            'fold': fold_idx,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'params': best_params
        })
        
        print(f"  Fold {fold_idx}: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
    
    # Aggregate results
    outer_df = pd.DataFrame(outer_results)
    all_model_results[model_name] = {
        'models': best_models,
        'results': outer_df,
        'mean_mse': outer_df['mse'].mean(),
        'std_mse': outer_df['mse'].std(),
        'mean_mae': outer_df['mae'].mean(),
        'std_mae': outer_df['mae'].std(),
        'mean_rmse': outer_df['rmse'].mean(),
        'std_rmse': outer_df['rmse'].std(),
        'mean_r2': outer_df['r2'].mean(),
        'std_r2': outer_df['r2'].std()
    }
    
    print(f"\n{model_name} - NCV Results:")
    print(f"  R²:   {all_model_results[model_name]['mean_r2']:.3f} ± {all_model_results[model_name]['std_r2']:.3f}")
    print(f"  MAE:  {all_model_results[model_name]['mean_mae']:.3f} ± {all_model_results[model_name]['std_mae']:.3f}")
    print(f"  RMSE: {all_model_results[model_name]['mean_rmse']:.3f} ± {all_model_results[model_name]['std_rmse']:.3f}")
    print(f"  MSE:  {all_model_results[model_name]['mean_mse']:.3f} ± {all_model_results[model_name]['std_mse']:.3f}")
    
    # Train final model on entire NCV set with best parameters from best fold
    best_fold_result = outer_df.sort_values('r2', ascending=False).iloc[0]
    best_fold_params = best_fold_result['params']
    
    # Reconstruct pipeline with best parameters
    if model_name == 'MeanPredictor':
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('mean', DummyRegressor(strategy='mean'))
        ])
    elif model_name == 'HuberRegressor':
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('huber', HuberRegressor(**best_fold_params, max_iter=1000))
        ])
    elif model_name == 'XGBRegressor':
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(**best_fold_params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=0))
        ])
    elif model_name == 'LGBMRegressor':
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('lgbm', LGBMRegressor(**best_fold_params, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1))
        ])
    elif model_name == 'HistGBMRegressor':
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('histgbm', HistGradientBoostingRegressor(**best_fold_params, random_state=RANDOM_SEED))
        ])
    
        
    
    final_pipeline.fit(X_NCV, y_NCV)
    final_models_for_comparison[model_name] = final_pipeline

# ============================================================================
# 6. MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Model Performance Comparison")
print("="*80)

comparison_list = []
for model_name, info in all_model_results.items():
    comparison_list.append({
        'Model': model_name,
        'Num_Features': len(X_full.columns),
        'Mean_MSE_NCV': info['mean_mse'],
        'Std_MSE_NCV': info['std_mse'],
        'Mean_MAE_NCV': info['mean_mae'],
        'Std_MAE_NCV': info['std_mae'],
        'Mean_RMSE_NCV': info['mean_rmse'],
        'Std_RMSE_NCV': info['std_rmse'],
        'Mean_R2_NCV': info['mean_r2'],
        'Std_R2_NCV': info['std_r2']
    })

comparison_df = pd.DataFrame(comparison_list).sort_values('Mean_R2_NCV', ascending=False)
comparison_df.to_csv("output/tables/model_comparison_table.csv", index=False)

print("\nModel Performance Summary (sorted by R²):")
print(comparison_df[['Model', 'Mean_R2_NCV', 'Mean_MAE_NCV', 'Mean_RMSE_NCV']].to_string(index=False))

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Mean R²:   {comparison_df.iloc[0]['Mean_R2_NCV']:.3f} ± {comparison_df.iloc[0]['Std_R2_NCV']:.3f}")
print(f"   Mean MAE:  {comparison_df.iloc[0]['Mean_MAE_NCV']:.3f} ± {comparison_df.iloc[0]['Std_MAE_NCV']:.3f}")

# ============================================================================
# 7. FINAL EVALUATION ON HOLDOUT TEST SET
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Final Evaluation on Holdout Test Set")
print("="*80)

best_model_instance = final_models_for_comparison[best_model_name]
y_pred_test = best_model_instance.predict(X_FINAL_TEST)

mse_test = mean_squared_error(y_FINAL_TEST, y_pred_test)
mae_test = mean_absolute_error(y_FINAL_TEST, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_FINAL_TEST, y_pred_test)

print(f"\n{best_model_name} Performance on Holdout Test Set:")
print(f"  R²:   {r2_test:.3f}")
print(f"  MSE:  {mse_test:.3f}")
print(f"  MAE:  {mae_test:.3f}")
print(f"  RMSE: {rmse_test:.3f}")

# Save best model
joblib.dump(best_model_instance, "output/models/best_model_final.pkl")
joblib.dump(list(X_full.columns), "output/models/feature_list.pkl")
print("\n✓ Best model and feature list saved")

# ============================================================================
# 8. BOOTSTRAPPED CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Statistical Comparison (Bootstrapped 95% CI, n=1000)")
print("="*80)

def calculate_ci_difference_regression(y_true, y_pred1, y_pred2, alpha=0.05, n_bootstrap=1000, random_state=42):
    """Calculate bootstrapped confidence intervals for model comparison"""
    np.random.seed(random_state)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    r2_diffs = []
    mae_diffs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r2_diff = r2_score(y_true[idx], y_pred1[idx]) - r2_score(y_true[idx], y_pred2[idx])
        mae_diff = mean_absolute_error(y_true[idx], y_pred2[idx]) - mean_absolute_error(y_true[idx], y_pred1[idx])
        r2_diffs.append(r2_diff)
        mae_diffs.append(mae_diff)
    
    return {
        'R2_diff_mean': np.mean(r2_diffs),
        'R2_CI_lower': np.percentile(r2_diffs, alpha/2*100),
        'R2_CI_upper': np.percentile(r2_diffs, (1-alpha/2)*100),
        'R2_significant': 'Yes' if np.percentile(r2_diffs, alpha/2*100) > 0 else 'No',
        'MAE_diff_mean': np.mean(mae_diffs),
        'MAE_CI_lower': np.percentile(mae_diffs, alpha/2*100),
        'MAE_CI_upper': np.percentile(mae_diffs, (1-alpha/2)*100),
        'MAE_significant': 'Yes' if np.percentile(mae_diffs, alpha/2*100) > 0 else 'No'
    }

# Compare best model against all others
pred_best = best_model_instance.predict(X_FINAL_TEST)

ci_results = []
for compare_name, model_instance in final_models_for_comparison.items():
    if compare_name == best_model_name:
        continue
    
    pred_compare = model_instance.predict(X_FINAL_TEST)
    ci_stats = calculate_ci_difference_regression(y_FINAL_TEST, pred_best, pred_compare)
    
    ci_results.append({
        'Comparison': f"{best_model_name} vs {compare_name}",
        'R2_diff_mean': ci_stats['R2_diff_mean'],
        'R2_CI': f"[{ci_stats['R2_CI_lower']:.4f}, {ci_stats['R2_CI_upper']:.4f}]",
        'R2_significant': ci_stats['R2_significant'],
        'MAE_diff_mean': ci_stats['MAE_diff_mean'],
        'MAE_CI': f"[{ci_stats['MAE_CI_lower']:.4f}, {ci_stats['MAE_CI_upper']:.4f}]",
        'MAE_significant': ci_stats['MAE_significant']
    })

ci_df = pd.DataFrame(ci_results)
ci_df.to_csv("output/tables/bootstrapped_ci_comparison.csv", index=False)

print("\nBootstrapped CI Comparison Results:")
print(ci_df.to_string(index=False))

# ============================================================================
# 9. COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Comprehensive Feature Importance Analysis")
print("="*80)

def compute_feature_importance(model, X, y, X_test, y_test, model_name, top_n=20, seed=42):
    """
    Compute three types of feature importance:
    1. Permutation Importance (on test set)
    2. Built-in Model Feature Importance (if available)
    3. SHAP values (with TreeExplainer fallback to KernelExplainer)
    """
    print(f"\nComputing feature importance for {model_name}...")
    
    # Extract the actual model from pipeline
    if hasattr(model, 'named_steps'):
        # It's a pipeline - get the last step
        model_obj = model.steps[-1][1]
    else:
        model_obj = model
    
    # -------------------------------------------------------------------------
    # 1. PERMUTATION IMPORTANCE 
    # -------------------------------------------------------------------------
    print("  1️⃣ Permutation Importance...")
    perm_imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=seed,
        n_jobs=-1
    )
    
    perm_imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_imp.importances_mean,
        'importance_std': perm_imp.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print(f"     ✓ Computed (Top 5: {', '.join(perm_imp_df.head(5)['feature'].values)})")
    
    # -------------------------------------------------------------------------
    # 2. BUILT-IN MODEL FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    print("  2️⃣ Built-in Feature Importance...")
    builtin_imp = None
    
    # Check if model has feature_importances_ attribute
    if hasattr(model_obj, 'feature_importances_'):
        builtin_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': model_obj.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"     ✓ Computed (Top 5: {', '.join(builtin_imp.head(5)['feature'].values)})")
    
    # For stacked models, try to get from final estimator
    elif hasattr(model_obj, 'final_estimator_') and hasattr(model_obj.final_estimator_, 'feature_importances_'):
        builtin_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': model_obj.final_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"     ✓ Computed from final estimator")
    
    # For linear models (Huber), use coefficients
    elif hasattr(model_obj, 'coef_'):
        builtin_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model_obj.coef_)
        }).sort_values('importance', ascending=False)
        print(f"     ✓ Computed from coefficients")
    
    else:
        print(f"     ⚠ Not available for {model_name}")
    
    # -------------------------------------------------------------------------
    # 3. SHAP VALUES (with TreeExplainer → generic Explainer fallback)
    # -------------------------------------------------------------------------
    print("  3️⃣ SHAP Values...")
    
    X_sample = X.sample(n=min(100, len(X)), random_state=seed)
    X_background = X.sample(n=min(50, len(X)), random_state=seed)
    
    shap_values = None
    shap_imp_df = None
    
    try:
        # Try TreeExplainer first (fast for tree-based models)
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_sample)
        print("     ✓ SHAP computed using TreeExplainer")
    
    except Exception as e:
        print(f"     TreeExplainer failed: {str(e)[:50]}...")
        print("     → Falling back to generic shap.Explainer")
        
        try:
            # Safe prediction wrapper
            def model_predict(data):
                return model.predict(data)
            
            # Fallback to generic SHAP explainer (Kernel-based)
            explainer = shap.Explainer(model_predict, X_background)
            shap_obj = explainer(X_sample)
            shap_values = shap_obj.values
            print("     ✓ SHAP computed using fallback Explainer (Kernel-based)")
        
        except Exception as e2:
            print(f"     ❌ SHAP computation failed: {str(e2)[:50]}")
            shap_values = None
    
    # Build SHAP importance dataframe
    if shap_values is not None:
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-output case
            shap_array = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_array = shap_values
        
        # Ensure 2D array
        if shap_array.ndim == 3:
            shap_array = shap_array.mean(axis=0)
        
        shap_imp_df = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_array).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"     ✓ Top 5 SHAP features: {', '.join(shap_imp_df.head(5)['feature'].values)}")
    
    return {
        'permutation': perm_imp_df,
        'builtin': builtin_imp,
        'shap': shap_imp_df,
        'shap_values': shap_values,
        'X_sample': X_sample
    }


# Compute importance for best model
importance_results = compute_feature_importance(
    model=best_model_instance,
    X=X_NCV,
    y=y_NCV,
    X_test=X_FINAL_TEST,
    y_test=y_FINAL_TEST,
    model_name=best_model_name,
    top_n=20,
    seed=RANDOM_SEED
)

# Save importance results
importance_results['permutation'].to_csv("output/tables/permutation_importance.csv", index=False)
print("\n✓ Permutation importance saved")

if importance_results['builtin'] is not None:
    importance_results['builtin'].to_csv("output/tables/builtin_importance.csv", index=False)
    print("✓ Built-in importance saved")

if importance_results['shap'] is not None:
    importance_results['shap'].to_csv("output/tables/shap_importance.csv", index=False)
    print("✓ SHAP importance saved")

# Print top features from each method
print("\n" + "="*80)
print("Top 15 Features by Each Importance Method")
print("="*80)

print("\n📊 Permutation Importance (Top 15):")
print(importance_results['permutation'].head(15)[['feature', 'importance_mean']].to_string(index=False))

if importance_results['builtin'] is not None:
    print("\n📊 Built-in Model Importance (Top 15):")
    print(importance_results['builtin'].head(15)[['feature', 'importance']].to_string(index=False))

if importance_results['shap'] is not None:
    print("\n📊 SHAP Importance (Top 15):")
    print(importance_results['shap'].head(15)[['feature', 'mean_abs_shap']].to_string(index=False))

# ============================================================================
# 10. DOMAIN-WISE FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Domain-Wise Feature Importance Analysis")
print("="*80)

# Define domain mapping (customize based on your features)
# This is a template - you'll need to update based on your actual feature names
DOMAIN_MAPPING = {
    'Demographics': ['AGE', 'GENDER', 'RACE', 'ETHNIC', 'MARITAL', 'EDUC'],
    'Injury_Severity': ['GCS', 'PTA', 'TFC', 'LOC', 'INJURY'],
    'Functional_Baseline': ['FIM', 'DRS', 'MOTOR', 'COGN'],
    'Medical_History': ['DIABETES', 'HYPERTENSION', 'CARDIAC', 'NEURO'],
    'Treatment': ['LOS', 'REHAB', 'THERAPY', 'INTERVENTION'],
    'Other': []  # Catch-all for features not matching above
}

def assign_domain(feature_name):
    """Assign feature to domain based on substring matching"""
    feature_upper = feature_name.upper()
    for domain, keywords in DOMAIN_MAPPING.items():
        if domain == 'Other':
            continue
        for keyword in keywords:
            if keyword in feature_upper:
                return domain
    return 'Other'

# Add domain column to importance dataframes
for imp_type in ['permutation', 'builtin', 'shap']:
    if importance_results[imp_type] is not None:
        importance_results[imp_type]['domain'] = importance_results[imp_type]['feature'].apply(assign_domain)

# Aggregate importance by domain
domain_importance = {}

if importance_results['permutation'] is not None:
    domain_importance['permutation'] = importance_results['permutation'].groupby('domain')['importance_mean'].sum().sort_values(ascending=False)

if importance_results['builtin'] is not None:
    domain_importance['builtin'] = importance_results['builtin'].groupby('domain')['importance'].sum().sort_values(ascending=False)

if importance_results['shap'] is not None:
    domain_importance['shap'] = importance_results['shap'].groupby('domain')['mean_abs_shap'].sum().sort_values(ascending=False)

# Print domain-wise importance
print("\nDomain-wise Feature Importance:")
for imp_type, domain_imp in domain_importance.items():
    print(f"\n{imp_type.upper()}:")
    print(domain_imp.to_string())

# Save domain importance
domain_importance_df = pd.DataFrame(domain_importance)
domain_importance_df.to_csv("output/tables/domain_importance.csv")
print("\n✓ Domain-wise importance saved")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Generating Visualizations")
print("="*80)

# -------------------------------------------------------------------------
# 11a. Model Performance Comparison Plot
# -------------------------------------------------------------------------
print("\n  Plotting model performance comparison...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

plot_df = comparison_df[comparison_df['Model'] != 'MeanPredictor']
metrics = [
    ('Mean_R2_NCV', 'Mean R² (NCV)', 'green'),
    ('Mean_MAE_NCV', 'Mean Absolute Error (NCV)', 'orange'),
    ('Mean_MSE_NCV', 'Mean Squared Error (NCV)', 'red')
]

for i, (metric, title, color) in enumerate(metrics):
    sns.barplot(x='Model', y=metric, data=plot_df, ax=axes[i], color=color)
    
    # Add error bars
    std_metric = metric.replace('Mean', 'Std')
    axes[i].errorbar(
        x=range(len(plot_df)),
        y=plot_df[metric],
        yerr=plot_df[std_metric],
        fmt='none',
        color='black',
        capsize=5
    )
    
    axes[i].set_xlabel('Model', fontsize=12)
    axes[i].set_ylabel(title, fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_title(title, fontsize=14)

plt.tight_layout()
plt.savefig("output/figures/model_comparison_barplots.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Model comparison plot saved")

# -------------------------------------------------------------------------
# 11b. Permutation Importance Plot
# -------------------------------------------------------------------------
print("  Plotting permutation importance...")

plt.figure(figsize=(12, 8))
top_20_perm = importance_results['permutation'].head(20)
sns.barplot(
    x='importance_mean',
    y='feature',
    data=top_20_perm,
    palette='viridis'
)
plt.errorbar(
    x=top_20_perm['importance_mean'],
    y=range(len(top_20_perm)),
    xerr=top_20_perm['importance_std'],
    fmt='none',
    color='black',
    capsize=3
)
plt.xlabel('Permutation Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title(f'Top 20 Features - Permutation Importance ({best_model_name})', fontsize=14)
plt.tight_layout()
plt.savefig("output/figures/permutation_importance_top20.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Permutation importance plot saved")

# -------------------------------------------------------------------------
# 11c. Built-in Importance Plot (if available)
# -------------------------------------------------------------------------
if importance_results['builtin'] is not None:
    print("  Plotting built-in importance...")
    
    plt.figure(figsize=(12, 8))
    top_20_builtin = importance_results['builtin'].head(20)
    
    # Color by sign for linear models, otherwise single color
    if 'coef_' in str(type(best_model_instance.steps[-1][1])):
        colors = ['red' if x < 0 else 'green' for x in top_20_builtin['importance']]
    else:
        colors = 'magma'
    
    sns.barplot(
        x='importance',
        y='feature',
        data=top_20_builtin,
        palette=colors if isinstance(colors, list) else colors
    )
    plt.xlabel('Built-in Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top 20 Features - Built-in Importance ({best_model_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig("output/figures/builtin_importance_top20.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Built-in importance plot saved")

# -------------------------------------------------------------------------
# 11d. SHAP Summary Plot
# -------------------------------------------------------------------------
if importance_results['shap_values'] is not None:
    print("  Plotting SHAP summary...")
    
    # SHAP summary plot (dot plot)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        importance_results['shap_values'],
        importance_results['X_sample'],
        show=False,
        max_display=20
    )
    plt.title(f'SHAP Summary Plot ({best_model_name})', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("output/figures/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP bar plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        importance_results['shap_values'],
        importance_results['X_sample'],
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title(f'SHAP Feature Importance ({best_model_name})', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("output/figures/shap_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ SHAP plots saved")

# -------------------------------------------------------------------------
# 11e. Domain Importance Plot
# -------------------------------------------------------------------------
print("  Plotting domain-wise importance...")

fig, axes = plt.subplots(1, len(domain_importance), figsize=(6*len(domain_importance), 6))
if len(domain_importance) == 1:
    axes = [axes]

for i, (imp_type, domain_imp) in enumerate(domain_importance.items()):
    domain_imp.plot(kind='barh', ax=axes[i], color='steelblue')
    axes[i].set_xlabel('Total Importance', fontsize=12)
    axes[i].set_ylabel('Domain', fontsize=12)
    axes[i].set_title(f'Domain Importance - {imp_type.upper()}', fontsize=14)
    axes[i].invert_yaxis()

plt.tight_layout()
plt.savefig("output/figures/domain_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Domain importance plot saved")

# -------------------------------------------------------------------------
# 11f. Predicted vs Actual Plot
# -------------------------------------------------------------------------
print("  Plotting predicted vs actual...")

plt.figure(figsize=(10, 8))
plt.scatter(y_FINAL_TEST, y_pred_test, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_FINAL_TEST.min(), y_FINAL_TEST.max()],
         [y_FINAL_TEST.min(), y_FINAL_TEST.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual FIM Change', fontsize=12)
plt.ylabel('Predicted FIM Change', fontsize=12)
plt.title(f'Predicted vs Actual FIM Change ({best_model_name})\nTest Set R² = {r2_test:.3f}',
          fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Predicted vs actual plot saved")

# -------------------------------------------------------------------------
# 11g. Residual Plot
# -------------------------------------------------------------------------
print("  Plotting residuals...")

residuals = y_FINAL_TEST - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Residual scatter
axes[0].scatter(y_pred_test, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted FIM Change', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Residual Plot', fontsize=14)
axes[0].grid(alpha=0.3)

# Residual histogram
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output/figures/residual_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Residual plots saved")

# ============================================================================
# 12. FINAL SUMMARY & EXPORT
# ============================================================================
print("\n" + "="*80)
print("STEP 11: Final Summary")
print("="*80)

# Create comprehensive summary
summary = {
    'Best_Model': best_model_name,
    'Num_Features': len(X_full.columns),
    'Training_Samples': len(X_NCV),
    'Test_Samples': len(X_FINAL_TEST),
    'NCV_R2_mean': comparison_df.iloc[0]['Mean_R2_NCV'],
    'NCV_R2_std': comparison_df.iloc[0]['Std_R2_NCV'],
    'NCV_MAE_mean': comparison_df.iloc[0]['Mean_MAE_NCV'],
    'NCV_MAE_std': comparison_df.iloc[0]['Std_MAE_NCV'],
    'Test_R2': r2_test,
    'Test_MSE': mse_test,
    'Test_MAE': mae_test,
    'Test_RMSE': rmse_test
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("output/tables/final_summary.csv", index=False)

print("\n📊 MODELING PIPELINE COMPLETE!")
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"\nNested Cross-Validation (5-fold):")
print(f"  R²:   {summary['NCV_R2_mean']:.3f} ± {summary['NCV_R2_std']:.3f}")
print(f"  MAE:  {summary['NCV_MAE_mean']:.3f} ± {summary['NCV_MAE_std']:.3f}")
print(f"\nFinal Test Set Performance:")
print(f"  R²:   {summary['Test_R2']:.3f}")
print(f"  MSE:  {summary['Test_MSE']:.3f}")
print(f"  MAE:  {summary['Test_MAE']:.3f}")
print(f"  RMSE: {summary['Test_RMSE']:.3f}")

print("\n" + "="*80)
print("OUTPUT FILES")
print("="*80)
print("\n📁 Models:")
print("  • output/models/best_model_final.pkl")
print("  • output/models/feature_list.pkl")
print("\n📁 Tables:")
print("  • output/tables/model_comparison_table.csv")
print("  • output/tables/bootstrapped_ci_comparison.csv")
print("  • output/tables/permutation_importance.csv")
if importance_results['builtin'] is not None:
    print("  • output/tables/builtin_importance.csv")
if importance_results['shap'] is not None:
    print("  • output/tables/shap_importance.csv")
print("  • output/tables/domain_importance.csv")
print("  • output/tables/final_summary.csv")
print("\n📁 Figures:")
print("  • output/figures/model_comparison_barplots.png")
print("  • output/figures/permutation_importance_top20.png")
if importance_results['builtin'] is not None:
    print("  • output/figures/builtin_importance_top20.png")
if importance_results['shap_values'] is not None:
    print("  • output/figures/shap_summary_plot.png")
    print("  • output/figures/shap_importance_bar.png")
print("  • output/figures/domain_importance.png")
print("  • output/figures/predicted_vs_actual.png")
print("  • output/figures/residual_analysis.png")

print("\n" + "="*80)
print("✅ PIPELINE EXECUTION COMPLETE")
print("="*80)