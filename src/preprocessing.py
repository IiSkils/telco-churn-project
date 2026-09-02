import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_data(input_path="data/telco_churn.csv", output_dir="data"):
    print("🚀 Starting Data Preprocessing...")
    df = pd.read_csv(input_path)
    

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    

    X = pd.get_dummies(X, drop_first=True)
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Split complete -> Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"✅ Total Features after Encoding: {X_train.shape[1]}")
    

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print("📁 Processed files saved successfully in 'data/' folder.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()