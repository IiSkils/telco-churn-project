import matplotlib
matplotlib.use('Agg')
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_models(data_dir="data", model_dir="models", output_dir="outputs"):
    print("🚀 Loading test data...")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()


    best_model_name = "CatBoost"
    worst_model_name = "LightGBM"

    print(f"⚖️ Loading Best ({best_model_name}) and Worst ({worst_model_name}) models...")
    best_model = joblib.load(f"{model_dir}/best_model.pkl")
    worst_model = joblib.load(f"{model_dir}/{worst_model_name}_tuned.pkl")

    os.makedirs(output_dir, exist_ok=True)


    for name, model, is_best in [(best_model_name, best_model, True), (worst_model_name, worst_model, False)]:
        print(f"\n📊 Evaluating {name}...")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if is_best else 'Reds')
        plt.title(f"Confusion Matrix - {name} ({'Best' if is_best else 'Worst'})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{name.lower()}.png")
        plt.close()


    print(f"\n🌟 Extracting Feature Importance for {best_model_name}...")
    importances = best_model.feature_importances_
    feature_names = X_test.columns
    
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title(f"Top 10 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_best_model.png")
    plt.close()

    print(f"✅ Evaluation complete. All required figures saved to '{output_dir}/'.")

if __name__ == "__main__":
    evaluate_models()