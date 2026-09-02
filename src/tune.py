import pandas as pd
import optuna
import os
import joblib
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective(trial, model_name, X, y):

    if model_name == 'AdaBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 9)
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                              max_depth=max_depth, random_state=42, eval_metric='logloss')
        
    elif model_name == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 100)
        model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, 
                               num_leaves=num_leaves, random_state=42, verbose=-1)
        
    elif model_name == 'CatBoost':
        iterations = trial.suggest_int('iterations', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        depth = trial.suggest_int('depth', 3, 9)
        model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, 
                                   depth=depth, random_state=42, verbose=0)


    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean()

def tune_models(data_dir="data", output_dir="models", n_trials=10):
    print("🚀 Loading preprocessed data for Tuning...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    
    models = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in models:
        print(f"\n⚙️ Tuning {model_name} with Optuna...")

        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        
        study = optuna.create_study(direction='maximize')

        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train), n_trials=n_trials)
        
        print(f"✅ Best ROC-AUC for {model_name}: {study.best_value:.4f}")
        print(f"🔧 Best Params: {study.best_params}")
        

        if model_name == 'AdaBoost':
            best_model = AdaBoostClassifier(**study.best_params, random_state=42)
        elif model_name == 'XGBoost':
            best_model = XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
        elif model_name == 'LightGBM':
            best_model = LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
        elif model_name == 'CatBoost':
            best_model = CatBoostClassifier(**study.best_params, random_state=42, verbose=0)
            
        best_model.fit(X_train, y_train)
        

        joblib.dump(best_model, f"{output_dir}/{model_name}_tuned.pkl")
        
    print("\n📁 Tuning complete. All tuned models saved in 'models/'.")

if __name__ == "__main__":
    tune_models(n_trials=10)