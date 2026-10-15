import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import optuna
import pickle
import os

# Suppress optuna logs to keep terminal clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_preprocess(data_path):
    print("⏳ Loading high-scale dataset...")
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])
        
    categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    existing_cats = [col for col in categorical_cols if col in df.columns]
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    X = df.drop(columns=['price'])
    y = df['price']
    return X, y

def train_and_evaluate():
    X, y = load_and_preprocess("data/Clean_Dataset.csv")
    
    # Save feature names for FastAPI alignment
    feature_columns = list(X.columns)
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

    # Train-test split (using smaller validation subset for fast tuning)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tune_tr, X_tune_val, y_tune_tr, y_tune_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    print("🧪 [1/3] Initiating Optuna Hyperparameter Optimization for XGBoost...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'n_jobs': -1,
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tune_tr, y_tune_tr)
        preds = model.predict(X_tune_val)
        return r2_score(y_tune_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5) # 5 rounds to keep it quick but effective
    print(f"✨ Best XGBoost Parameters Found: {study.best_params}")

    # Train Final Best XGBoost
    print("\n🚀 [2/3] Training Optimized XGBoost Regressor...")
    best_xgb = xgb.XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
    best_xgb.fit(X_train, y_train)
    xgb_preds = best_xgb.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)

    # Train Baseline Random Forest for comparison
    print("\n🌲 [3/3] Training Comparison Random Forest Regressor (Baseline)...")
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42) # Restricted to ensure your RAM doesn't crash
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)

    # --- Side-by-Side Model Comparison ---
    print("\n📊 ==========================================")
    print("📋         MODEL COMPARISON MATRIX            ")
    print("==============================================")
    print(f"💡 XGBoost (Tuned)     -> R²: {xgb_r2:.4f} | MAE: ₹{xgb_mae:.2f}")
    print(f"💡 Random Forest       -> R²: {rf_r2:.4f} | MAE: ₹{rf_mae:.2f}")
    print("==============================================")

    # Select and serialize the true engineering winner
    if xgb_r2 > rf_r2:
        print("🏆 Winner: XGBoost selected for production deployment!")
        winner_model = best_xgb
    else:
        print("🏆 Winner: Random Forest selected for production deployment!")
        winner_model = rf

    with open('artifacts/flight_model.pkl', 'wb') as f:
        pickle.dump(winner_model, f)
    print("✅ Best Model Artifacts successfully mounted into 'artifacts/' folder!")

if __name__ == "__main__":
    train_and_evaluate()