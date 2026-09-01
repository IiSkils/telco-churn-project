import streamlit as st
import pandas as pd
import joblib

# 1. إعدادات الصفحة
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📞 Telco Customer Churn Predictor (Team 3)")

# 2. تحميل الموديل والبيانات في الذاكرة (Caching) لتسريع التطبيق
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data
def load_data():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    return X_test, y_test

model = load_model()
X_test, y_test = load_data()

# 3. إنشاء التبويبات (Tabs)
tab1, tab2 = st.tabs(["🏆 Model Leaderboard", "🔮 Prediction"])

# ================= التبويب الأول =================
with tab1:
    st.header("Model Performance Leaderboard")
    
    # جدول الترتيب النهائي بناءً على نتائج الـ Cross-Validation
    results = {
        "Model": ["CatBoost", "AdaBoost", "XGBoost", "LightGBM"],
        "ROC-AUC": [0.8489, 0.8483, 0.8449, 0.8409],
        "F1-Score": [0.5859, 0.5606, 0.5258, 0.5562],
        "Accuracy": [0.8060, 0.8016, 0.7964, 0.8001]
    }
    st.dataframe(pd.DataFrame(results).set_index("Model"), use_container_width=True)
    
    st.subheader("Visual Comparisons")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("outputs/model_comparison.png", caption="Model Comparison (ROC-AUC)")
        st.image("outputs/feature_importance_best_model.png", caption="Top 10 Feature Importances (CatBoost)")
        
    with col2:
        st.image("outputs/confusion_matrix_catboost.png", caption="Best Model (CatBoost) - Confusion Matrix")
        st.image("outputs/confusion_matrix_lightgbm.png", caption="Worst Model (LightGBM) - Confusion Matrix")

# ================= التبويب الثاني =================
with tab2:
    st.header("Predict Customer Churn")
    st.write("اختر عميلاً من بيانات الاختبار (Test Data) لتوقع ما إذا كان سيغادر الشركة أم لا.")
    
    # اختيار عميل عشوائي بناءً على رقم الصف
    customer_index = st.slider("Select Customer Index:", 0, len(X_test)-1, 0)
    
    customer_data = X_test.iloc[[customer_index]]
    actual_churn = y_test.iloc[customer_index].values[0]
    
    st.subheader("Customer Profile (Encoded Features)")
    st.dataframe(customer_data)
    
    if st.button("Predict Churn"):
        prediction = model.predict(customer_data)[0]
        probability = model.predict_proba(customer_data)[0][1]
        
        st.markdown("### Results:")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Low Risk of Churn. (Probability: {probability:.2%})")
            
        st.info(f"Actual Status in Database: {'Churned (1)' if actual_churn == 1 else 'Stayed (0)'}")