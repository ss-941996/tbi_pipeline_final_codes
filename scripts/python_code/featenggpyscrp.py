# ═══════════════════════════════════════════════════════════════════
# SIMPLE PIPELINE: TEST BEST MODEL ON RAW vs RAW+ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════════
# This takes your already-trained best model (XGBoost with tuned hyperparameters)
# and evaluates it on two datasets:
#   Dataset A: Raw features only
#   Dataset B: Raw features + Engineered features
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING CLASS
# ═══════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Adds engineered features while keeping ALL original features.
    Fit on training data, transform on both train and test.
    """
    
    def __init__(self):
        self.median_values = {}
    
    def fit(self, X):
        """Learn parameters from training data only"""
        X_clean = X.copy()
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.median_values = X_clean.median().to_dict()
        return self
    
    def transform(self, X):
        """Add engineered features to original dataset"""
        X_eng = X.copy()
        
        # ─────────────────────────────────────────────────────
        # FIM Composites
        # ─────────────────────────────────────────────────────
        comm_cols = ['FIMCompA', 'FIMExpressA']
        if all(col in X_eng.columns for col in comm_cols):
            X_eng['FIM_Communication_Score'] = X_eng[comm_cols].mean(axis=1)
        
        social_cols = ['FIMSocialA', 'FIMProbSlvA']
        if all(col in X_eng.columns for col in social_cols):
            X_eng['FIM_Social_Cognition'] = X_eng[social_cols].mean(axis=1)
        
        # ─────────────────────────────────────────────────────
        # Efficiency ratios (avoid division by zero)
        # ─────────────────────────────────────────────────────
        if 'LOSRehabNoInt' in X_eng.columns:
            los_safe = X_eng['LOSRehabNoInt'] + 1
            
            if 'FIMMOTA' in X_eng.columns:
                X_eng['FIM_Motor_Efficiency'] = X_eng['FIMMOTA'] / los_safe
            
            if 'FIMCOGA' in X_eng.columns:
                X_eng['FIM_Cognitive_Efficiency'] = X_eng['FIMCOGA'] / los_safe
            
            if 'FIM_Communication_Score' in X_eng.columns:
                X_eng['Communication_per_Day'] = X_eng['FIM_Communication_Score'] / los_safe
            
            if 'FIM_Social_Cognition' in X_eng.columns:
                X_eng['SocialCog_per_Day'] = X_eng['FIM_Social_Cognition'] / los_safe
        
        # ─────────────────────────────────────────────────────
        # Cognitive-Motor interactions
        # ─────────────────────────────────────────────────────
        if 'FIMCOGA' in X_eng.columns and 'FIMMOTA' in X_eng.columns:
            X_eng['FIM_CogMot_Ratio'] = X_eng['FIMCOGA'] / (X_eng['FIMMOTA'] + 1)
            X_eng['FIM_CogMot_Product'] = X_eng['FIMCOGA'] * X_eng['FIMMOTA']
        
        # ─────────────────────────────────────────────────────
        # Cognitive items average
        # ─────────────────────────────────────────────────────
        fim_cog_items = ['FIMCompA', 'FIMExpressA', 'FIMProbSlvA', 'FIMSocialA']
        if all(col in X_eng.columns for col in fim_cog_items):
            X_eng['FIM_CogItems_Avg'] = X_eng[fim_cog_items].mean(axis=1)
        
        # ─────────────────────────────────────────────────────
        # Clean data
        # ─────────────────────────────────────────────────────
        X_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_eng.fillna(self.median_values, inplace=True)
        X_eng.columns = X_eng.columns.str.replace('[<>[\],]', '_', regex=True)
        
        return X_eng
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


def prepare_raw_features(X, median_values=None):
    """Clean raw features without adding new ones"""
    X_raw = X.copy()
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if median_values is None:
        X_raw.fillna(X_raw.median(), inplace=True)
    else:
        X_raw.fillna(median_values, inplace=True)
    
    X_raw.columns = X_raw.columns.str.replace('[<>[\],]', '_', regex=True)
    return X_raw


# ═══════════════════════════════════════════════════════════════════
# 2. EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred):
    """Calculate all regression metrics"""
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mse': mean_squared_error(y_true, y_pred)
    }


def bootstrap_comparison(y_true, y_pred_raw, y_pred_eng, n_bootstrap=1000, random_state=42):
    """
    Bootstrap evaluation to compare two sets of predictions.
    Returns metrics for both models and statistical tests.
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    metrics_raw = {'r2': [], 'mae': [], 'rmse': []}
    metrics_eng = {'r2': [], 'mae': [], 'rmse': []}
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_bs = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_raw_bs = y_pred_raw[indices]
        y_pred_eng_bs = y_pred_eng[indices]
        
        # Raw model metrics
        metrics_raw['r2'].append(r2_score(y_true_bs, y_pred_raw_bs))
        metrics_raw['mae'].append(mean_absolute_error(y_true_bs, y_pred_raw_bs))
        metrics_raw['rmse'].append(np.sqrt(mean_squared_error(y_true_bs, y_pred_raw_bs)))
        
        # Engineered model metrics
        metrics_eng['r2'].append(r2_score(y_true_bs, y_pred_eng_bs))
        metrics_eng['mae'].append(mean_absolute_error(y_true_bs, y_pred_eng_bs))
        metrics_eng['rmse'].append(np.sqrt(mean_squared_error(y_true_bs, y_pred_eng_bs)))
    
    # Calculate confidence intervals and statistics
    results = {}
    for metric in ['r2', 'mae', 'rmse']:
        raw_vals = np.array(metrics_raw[metric])
        eng_vals = np.array(metrics_eng[metric])
        diff_vals = eng_vals - raw_vals
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(eng_vals, raw_vals)
        
        results[metric] = {
            'raw_mean': np.mean(raw_vals),
            'raw_ci': np.percentile(raw_vals, [2.5, 97.5]),
            'eng_mean': np.mean(eng_vals),
            'eng_ci': np.percentile(eng_vals, [2.5, 97.5]),
            'diff_mean': np.mean(diff_vals),
            'diff_ci': np.percentile(diff_vals, [2.5, 97.5]),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════
# 3. MAIN COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════════

def compare_raw_vs_engineered(
    filepath,
    best_model_pipeline,
    target_col='FIM_change',
    test_size=0.2,
    n_bootstrap=1000,
    random_state=42
):
    """
    Compare best model's performance on:
    - Dataset A: Raw features only
    - Dataset B: Raw features + Engineered features
    
    Parameters
    ----------
    filepath : str
        Path to your data CSV
    best_model_pipeline : Pipeline or model
         already-trained best model (e.g., from final_models_for_comparison['XGBRegressor'])
    target_col : str
        Target variable name
    test_size : float
        Proportion for test set
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int
        Random seed
    
    Returns
    -------
    dict : Contains all comparison results, models, and predictions
    """
    
    print("=" * 80)
    print("FEATURE ENGINEERING COMPARISON: RAW vs RAW+ENGINEERED")
    print("Using your best trained model from the original pipeline")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────
    # Step 1: Load and split data
    # ─────────────────────────────────────────────────────
    print("\n📊 Step 1: Loading and splitting data...")
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=['Mod1Id', target_col] if 'Mod1Id' in df.columns else [target_col])
    y = df[target_col]
    
    # Split data BEFORE feature engineering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Original features: {X_train.shape[1]} columns")
    
    # ─────────────────────────────────────────────────────
    # Step 2: Prepare Dataset A (RAW features only)
    # ─────────────────────────────────────────────────────
    print("\n🔧 Step 2: Preparing Dataset A (Raw features only)...")
    
    # Clean training data and get medians
    X_train_raw_temp = X_train.copy()
    X_train_raw_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_medians = X_train_raw_temp.median().to_dict()
    
    # Apply to train and test
    X_train_raw = prepare_raw_features(X_train, raw_medians)
    X_test_raw = prepare_raw_features(X_test, raw_medians)
    
    print(f"   Dataset A features: {X_train_raw.shape[1]} columns")
    
    # ─────────────────────────────────────────────────────
    # Step 3: Prepare Dataset B (RAW + ENGINEERED)
    # ─────────────────────────────────────────────────────
    print("\n⚙️  Step 3: Preparing Dataset B (Raw + Engineered features)...")
    
    fe = FeatureEngineer()
    X_train_eng = fe.fit_transform(X_train)
    X_test_eng = fe.transform(X_test)
    
    new_features = [col for col in X_train_eng.columns if col not in X_train_raw.columns]
    
    print(f"   Dataset B features: {X_train_eng.shape[1]} columns")
    print(f"   New features added: {len(new_features)}")
    print(f"\n   📝 Engineered features:")
    for feat in new_features:
        print(f"      • {feat}")
    
    # ─────────────────────────────────────────────────────
    # Step 4: Clone and retrain model on both datasets
    # ─────────────────────────────────────────────────────
    print("\n🤖 Step 4: Training models on both datasets...")
    print("   (Cloning your best model with same hyperparameters)")
    
    # Clone the model to ensure independent training
    model_raw = clone(best_model_pipeline)
    model_eng = clone(best_model_pipeline)
    
    print("   Training Model A on Dataset A (Raw features)...")
    model_raw.fit(X_train_raw, y_train)
    
    print("   Training Model B on Dataset B (Raw + Engineered features)...")
    model_eng.fit(X_train_eng, y_train)
    
    # ─────────────────────────────────────────────────────
    # Step 5: Predict on test set
    # ─────────────────────────────────────────────────────
    print("\n🎯 Step 5: Evaluating on test set...")
    
    y_pred_raw = model_raw.predict(X_test_raw)
    y_pred_eng = model_eng.predict(X_test_eng)
    
    metrics_raw = calculate_metrics(y_test, y_pred_raw)
    metrics_eng = calculate_metrics(y_test, y_pred_eng)
    
    print("\n   📊 Test Set Results:")
    print("   " + "-" * 75)
    print(f"   {'Metric':<15} {'Model A (Raw)':>18} {'Model B (Raw+Eng)':>18} {'Δ':>10} {'Better'}")
    print("   " + "-" * 75)
    
    for metric in ['r2', 'mae', 'rmse', 'mse']:
        raw_val = metrics_raw[metric]
        eng_val = metrics_eng[metric]
        diff = eng_val - raw_val
        
        # Determine which is better
        if metric == 'r2':
            better = "B ✓" if diff > 0 else "A ✓"
        else:  # Lower is better for error metrics
            better = "B ✓" if diff < 0 else "A ✓"
        
        print(f"   {metric.upper():<15} {raw_val:>18.4f} {eng_val:>18.4f} {diff:>9.4f}  {better}")
    
    # ─────────────────────────────────────────────────────
    # Step 6: Bootstrap validation with statistical testing
    # ─────────────────────────────────────────────────────
    print(f"\n🔬 Step 6: Bootstrap validation ({n_bootstrap} iterations)...")
    print("   This may take a moment...")
    
    bootstrap_results = bootstrap_comparison(
        y_test, y_pred_raw, y_pred_eng, 
        n_bootstrap=n_bootstrap, 
        random_state=random_state
    )
    
    print("\n   📉 Bootstrap Results with 95% Confidence Intervals:")
    print("   " + "-" * 80)
    
    for metric in ['r2', 'mae', 'rmse']:
        res = bootstrap_results[metric]
        print(f"\n   {metric.upper()}:")
        print(f"      Model A (Raw):          {res['raw_mean']:>7.4f}  "
              f"(95% CI: [{res['raw_ci'][0]:>7.4f}, {res['raw_ci'][1]:>7.4f}])")
        print(f"      Model B (Raw+Eng):      {res['eng_mean']:>7.4f}  "
              f"(95% CI: [{res['eng_ci'][0]:>7.4f}, {res['eng_ci'][1]:>7.4f}])")
        print(f"      Difference (B - A):     {res['diff_mean']:>7.4f}  "
              f"(95% CI: [{res['diff_ci'][0]:>7.4f}, {res['diff_ci'][1]:>7.4f}])")
    
    print("\n   🔍 Statistical Significance Tests (Paired t-tests):")
    print("   " + "-" * 80)
    print(f"   {'Metric':<8} {'Mean Δ':>10} {'t-statistic':>12} {'p-value':>10} {'Significant'}")
    print("   " + "-" * 80)
    
    for metric in ['r2', 'mae', 'rmse']:
        res = bootstrap_results[metric]
        sig_marker = "Yes ***" if res['significant'] else "No"
        print(f"   {metric.upper():<8} {res['diff_mean']:>10.4f} {res['t_statistic']:>12.3f} "
              f"{res['p_value']:>10.4f}   {sig_marker}")
    
    # ─────────────────────────────────────────────────────
    # Step 7: Final recommendation
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📋 SUMMARY AND RECOMMENDATION")
    print("=" * 80)
    
    r2_res = bootstrap_results['r2']
    mae_res = bootstrap_results['mae']
    
    print(f"\n   Performance Differences (Model B - Model A):")
    print(f"   • R² change:    {r2_res['diff_mean']:+.4f} (p = {r2_res['p_value']:.4f})")
    print(f"   • MAE change:   {mae_res['diff_mean']:+.4f} (p = {mae_res['p_value']:.4f})")
    
    if r2_res['diff_mean'] > 0 and r2_res['significant']:
        print("\n   ✅ MODEL B (Raw + Engineered Features) is SIGNIFICANTLY BETTER")
        print(f"      R² improved by {r2_res['diff_mean']:.4f} (95% CI: [{r2_res['diff_ci'][0]:.4f}, {r2_res['diff_ci'][1]:.4f}])")
        if mae_res['diff_mean'] < 0:
            print(f"      MAE reduced by {abs(mae_res['diff_mean']):.4f}")
        print("\n   💡 Recommendation: Use Dataset B with engineered features")
        print(f"      The {len(new_features)} engineered features provide significant value")
        
    elif r2_res['diff_mean'] < 0 and r2_res['significant']:
        print("\n   ⚠️  MODEL A (Raw Features Only) is SIGNIFICANTLY BETTER")
        print(f"      R² decreased by {abs(r2_res['diff_mean']):.4f} when adding features")
        print("\n   💡 Recommendation: Use Dataset A (raw features only)")
        print("      Engineered features may be adding noise or causing overfitting")
        
    else:
        print("\n   ➖ NO SIGNIFICANT DIFFERENCE between models")
        print(f"      R² difference: {r2_res['diff_mean']:.4f} (p = {r2_res['p_value']:.4f})")
        print("\n   💡 Recommendation: Use Dataset A (raw features) - Occam's Razor")
        print("      When performance is similar, prefer the simpler model")
    
    # Check if CI crosses zero
    if r2_res['diff_ci'][0] < 0 < r2_res['diff_ci'][1]:
        print("\n   ⚠️  Note: 95% CI for R² difference includes zero")
        print("      This suggests high uncertainty in the performance difference")
    
    print("\n" + "=" * 80)
    
    return {
        'test_metrics': {
            'raw': metrics_raw,
            'engineered': metrics_eng
        },
        'bootstrap_results': bootstrap_results,
        'predictions': {
            'y_test': y_test,
            'y_pred_raw': y_pred_raw,
            'y_pred_eng': y_pred_eng
        },
        'models': {
            'model_raw': model_raw,
            'model_eng': model_eng
        },
        'datasets': {
            'X_train_raw': X_train_raw,
            'X_train_eng': X_train_eng,
            'X_test_raw': X_test_raw,
            'X_test_eng': X_test_eng
        },
        'new_features': new_features
    }

# ========================================================================
# LOAD BEST MODEL FROM DISK
# ========================================================================

import joblib

best_model_path = r"C:\Users\saumy\OneDrive\Desktop\tbi_pipeline_final_codes\notebooks\output\models\best_model_final.pkl"
best_model_instance = joblib.load(best_model_path)

print("Best model loaded successfully!")

# ========================================================================
# RUN RAW vs ENGINEERED FEATURE COMPARISON
# ========================================================================

filepath = "../data/processed/df17nov.csv"

results = compare_raw_vs_engineered(
    filepath=filepath,
    best_model_pipeline=best_model_instance,
    target_col='FIM_change',
    test_size=0.2,
    n_bootstrap=1000,
    random_state=42
)

print("\nFinal Comparison Results:")
print(results)
