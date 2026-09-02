# 📞 Telco Customer Churn Predictor (Team 3)
**Developed by:** Hazem Ahmed & Seif Eldeen Mohamed

## 🎯 Business Objective
Customer acquisition costs are significantly higher than retention. Our goal is to build an End-to-End ML Pipeline that identifies customers at high risk of churning *before* they leave, allowing the business to proactively allocate retention budgets.

## 🧠 AI Strategy & Model Selection
Since the dataset is highly imbalanced (~73% stay, ~27% leave), relying on standard **Accuracy** is deceptive. A model guessing "everyone stays" would be 73% accurate but useless for business operations. 

Therefore, we evaluated our Boosting models based on **ROC-AUC and F1-Score** to perfectly balance Precision and Recall.

*   **Models Evaluated:** AdaBoost, XGBoost, LightGBM, CatBoost.
*   **Tuning:** Optuna + Nested Cross-Validation.
*   **The Winner:** **CatBoost** achieved the highest cross-validated ROC-AUC, providing the most reliable predictions for real-world financial decisions without triggering false alarms.

## 🚀 Impact
We deployed the finalized CatBoost model into an interactive Streamlit Web Dashboard. CRM agents can simply input a customer ID and instantly receive an AI-driven churn probability analysis, saving hundreds of hours of manual data review.