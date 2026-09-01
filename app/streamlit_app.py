import streamlit as st
import pandas as pd
import joblib
import random
import time

# 1. إعدادات الصفحة
st.set_page_config(page_title="Telco Churn Predictor", page_icon="🚀", layout="wide")

# ================= القائمة الجانبية (Sidebar) =================
with st.sidebar:
    st.title("🏆 The Dream Team")
    st.markdown("### 👨‍💻 Developed By:")
    st.markdown("- **Hazem Ahmed**")
    st.markdown("- **Saif ElDeen Mohamed**")
    st.divider()
    st.info("🤖 **Model Used:** CatBoost Classifier (Tuned via Optuna)")
    st.success("🎯 **Project Goal:** End-to-end ML Pipeline for Telco Customer Churn Prediction.")

# العنوان الرئيسي
st.title("🚀 Telco Customer Churn Predictor")
st.markdown("---")

# 2. تحميل الموديل والبيانات في الذاكرة لتسريع التطبيق
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

# 3. إنشاء التبويبات
tab1, tab2 = st.tabs(["🔮 Prediction Hub", "📊 Model Leaderboard"])

# ================= التبويب الأول: التوقع (Prediction) =================
with tab1:
    st.markdown("### 🎯 How would you like to select a customer?")
    
    # تهيئة المتغير العشوائي في الذاكرة
    if 'random_idx' not in st.session_state:
        st.session_state.random_idx = 0

    # خيارات متعددة لاختيار العميل
    selection_method = st.radio(
        "Selection Method",
        ["🔍 Search by ID", "📋 Browse List", "🎲 Random Customer"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.write("") 

    if selection_method == "🔍 Search by ID":
        customer_index = st.number_input(f"Enter Customer ID (0 to {len(X_test)-1}):", min_value=0, max_value=len(X_test)-1, value=0)
    
    elif selection_method == "📋 Browse List":
        customer_index = st.selectbox(
            "Select a customer from the database:",
            options=range(len(X_test)),
            format_func=lambda x: f"Customer #{x} — Tenure: {X_test.iloc[x].get('tenure', 0)}m | Monthly: ${X_test.iloc[x].get('MonthlyCharges', 0):.2f}"
        )
        
    else:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🎲 Pick Random Customer", use_container_width=True):
                st.session_state.random_idx = random.randint(0, len(X_test)-1)
        customer_index = st.session_state.random_idx
        with col2:
            st.info(f"Currently viewing random customer ID: **#{customer_index}**")

    # جلب بيانات العميل المختار
    customer_data = X_test.iloc[[customer_index]]
    actual_churn = y_test.iloc[customer_index].values[0]
    
    st.subheader("👤 Customer Profile Overview")
    
    # عرض الأرقام الأساسية
    tenure = customer_data['tenure'].values[0] if 'tenure' in customer_data.columns else 0
    monthly = customer_data['MonthlyCharges'].values[0] if 'MonthlyCharges' in customer_data.columns else 0
    total = customer_data['TotalCharges'].values[0] if 'TotalCharges' in customer_data.columns else 0
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("🗓️ Tenure (Months)", int(tenure))
    m_col2.metric("💳 Monthly Charges", f"${monthly:.2f}")
    m_col3.metric("💰 Total Charges", f"${total:.2f}")
    
    # تنظيف البيانات وجعلها مفهومة للبشر 
    display_df = customer_data.T
    display_df.columns = ["Value"]
    
    def format_value(val, col_name):
        if col_name in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            return val
        if val == 1 or val == 1.0:
            return "✅ Yes"
        elif val == 0 or val == 0.0:
            return "❌ No"
        return val

    display_df['Value'] = [format_value(val, idx) for idx, val in zip(display_df.index, display_df['Value'])]
    clean_indices = [idx.replace('_', ' ').title() for idx in display_df.index]
    display_df.index = clean_indices
    
    with st.expander("📄 View Full Customer Details", expanded=True):
        st.dataframe(display_df, use_container_width=True, height=250)
    
    st.divider()
    
    # زر التوقع والتأثيرات البصرية
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        with st.spinner("🧠 AI is analyzing customer profile..."):
            time.sleep(1) 
            prediction = model.predict(customer_data)[0]
            probability = model.predict_proba(customer_data)[0][1]
        
        st.markdown("### 📊 AI Analysis Results:")
        
        # شريط التقدم المرئي للنسبة
        st.progress(float(probability), text=f"Churn Probability: {probability:.2%}")
        
        if prediction == 1:
            st.error("⚠️ **High Risk of Churn!** This customer is likely to leave.")
        else:
            st.success("✅ **Low Risk of Churn.** This customer is likely to stay.")
            
        st.info(f"📂 **Actual Status in Database:** {'Churned (Left the company)' if actual_churn == 1 else 'Stayed (Active)'}")

# ================= التبويب الثاني: التقييم والمقارنات =================
with tab2:
    st.header("Model Performance Leaderboard")
    
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