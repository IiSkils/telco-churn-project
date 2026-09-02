import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate

def train_baseline_models(data_dir="data", output_dir="models"):
    print("🚀 Loading preprocessed data for Baseline Training...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()


    models = {
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }

    print("⚖️ Evaluating Baseline Models using Cross-Validation...")
    results = []
    
    os.makedirs(output_dir, exist_ok=True)


    for name, model in models.items():
        print(f"Training and validating {name}...")

        cv_results = cross_validate(model, X_train, y_train, cv=5, 
                                    scoring=['accuracy', 'f1', 'roc_auc'])
        
        mean_acc = np.mean(cv_results['test_accuracy'])
        mean_f1 = np.mean(cv_results['test_f1'])
        mean_auc = np.mean(cv_results['test_roc_auc'])
        
        results.append({
            "Model": name, 
            "Accuracy": mean_acc, 
            "F1_Score": mean_f1, 
            "ROC_AUC": mean_auc
        })
        

        model.fit(X_train, y_train)
        joblib.dump(model, f"{output_dir}/{name}_baseline.pkl")


    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
    print("\n🏆 Baseline Models Leaderboard (Ranked by ROC-AUC):")
    print(results_df.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/baseline_leaderboard.csv", index=False)
    print("\n📁 Baseline training complete. Models saved in 'models/' and leaderboard in 'outputs/'.")

if __name__ == "__main__":
    train_baseline_models()