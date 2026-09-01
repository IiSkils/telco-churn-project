import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate

def compare_and_select_best(data_dir="data", model_dir="models", output_dir="outputs"):
    print("🚀 Loading preprocessed data and tuned models...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()

    # تحميل النماذج التي تم تحسينها بواسطة Optuna
    models = {
        "AdaBoost": joblib.load(f"{model_dir}/AdaBoost_tuned.pkl"),
        "XGBoost": joblib.load(f"{model_dir}/XGBoost_tuned.pkl"),
        "LightGBM": joblib.load(f"{model_dir}/LightGBM_tuned.pkl"),
        "CatBoost": joblib.load(f"{model_dir}/CatBoost_tuned.pkl")
    }

    results = []
    best_model_name = ""
    best_score = 0
    best_model = None

    print("⚖️ Running final Cross-Validation for the official leaderboard...")
    for name, model in models.items():
        # استخدام ROC-AUC كمعيار أساسي بسبب عدم توازن البيانات
        cv_results = cross_validate(model, X_train, y_train, cv=5, 
                                    scoring=['roc_auc', 'f1', 'accuracy'])
        
        mean_auc = np.mean(cv_results['test_roc_auc'])
        mean_f1 = np.mean(cv_results['test_f1'])
        mean_acc = np.mean(cv_results['test_accuracy'])
        
        results.append({
            "Model": name, 
            "ROC-AUC": mean_auc, 
            "F1-Score": mean_f1, 
            "Accuracy": mean_acc
        })
        
        # تحديد الفائز بناءً على ROC-AUC
        if mean_auc > best_score:
            best_score = mean_auc
            best_model_name = name
            best_model = model

    # إنشاء وطباعة جدول الترتيب
    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("\n🏆 Final Tuned Models Leaderboard:")
    print(df_results.to_string(index=False))

    # حفظ الموديل الفائز بالاسم المطلوب في التكليف
    joblib.dump(best_model, f"{model_dir}/best_model.pkl")
    print(f"\n✅ Winner Selected: {best_model_name} (Saved as best_model.pkl)")

    # رسم المخطط البياني وحفظه
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="ROC-AUC", y="Model", hue="Model", data=df_results, palette="viridis", legend=False)
    plt.title("Model Comparison (Cross-Validation ROC-AUC)")
    plt.xlabel("ROC-AUC Score")
    plt.ylabel("Algorithm")
    # ضبط حدود المحور السيني لإظهار الفروق الدقيقة بوضوح
    plt.xlim(0.82, 0.86)
    plt.tight_layout()
    
    chart_path = f"{output_dir}/model_comparison.png"
    plt.savefig(chart_path)
    print(f"📊 Comparison chart saved to {chart_path}")

if __name__ == "__main__":
    compare_and_select_best()