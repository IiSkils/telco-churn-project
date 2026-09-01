import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data(input_path="data/telco_churn.csv", output_dir="data"):
    print("🚀 Starting Data Preprocessing...")
    df = pd.read_csv(input_path)
    
    # 1. معالجة عمود التكلفة الإجمالية (تحويل المسافات الفارغة إلى NaN ثم حذفها)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # 2. حذف عمود الـ ID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # 3. تشفير الهدف (Churn) إلى 0 و 1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # فصل الميزات عن الهدف
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 4. تشفير باقي البيانات النصية (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # 5. التقسيم مع الحفاظ على التوازن (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Split complete -> Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"✅ Total Features after Encoding: {X_train.shape[1]}")
    
    # 6. حفظ البيانات المعالجة
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print("📁 Processed files saved successfully in 'data/' folder.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()