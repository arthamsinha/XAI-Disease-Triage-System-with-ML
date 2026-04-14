import os
import requests
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap

# Configuration
BASE_URL = "https://raw.githubusercontent.com/sohamvsonar/Disease-Prediction-and-Medical-Recommendation-System/main/dataset/"
FILES = {
    "Training.csv": "Training.csv",
    "Symptom-severity.csv": "Symptom-severity.csv",
    "description.csv": "description.csv",
    "precautions_df.csv": "precautions_df.csv",
    "medications.csv": "medications.csv"
}

DATA_DIR = "data"
MODEL_DIR = "model"

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for filename, remote_name in FILES.items():
        local_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            url = BASE_URL + remote_name
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {filename} from {url}")

def train():
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    
    # Remove unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Preprocessing
    # The dataset has 132 symptoms + 1 prognosis column
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    feature_columns = X.columns.tolist()
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Model 1: XGBoost (Primary)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Model 2: Random Forest (Secondary)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_xgb = xgb_model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
        "precision": float(precision_score(y_test, y_pred_xgb, average='weighted')),
        "recall": float(recall_score(y_test, y_pred_xgb, average='weighted')),
        "f1": float(f1_score(y_test, y_pred_xgb, average='weighted'))
    }
    
    # Confusion Matrix (Top 6 classes)
    cm = confusion_matrix(y_test, y_pred_xgb)
    class_counts = pd.Series(y).value_counts().index[:6].tolist()
    class_indices = le.transform(class_counts)
    
    # Exporting a representative confusion matrix for the top 6 classes
    # We will pick the top 6 classes and their inter-prediction counts
    top_6_cm = cm[np.ix_(class_indices, class_indices)].tolist()
    metrics["confusion_matrix_top6"] = {
        "classes": class_counts,
        "matrix": top_6_cm
    }
    
    # SHAP Analysis
    print("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_results = explainer.shap_values(X_test)
    
    # XGBoost 2.0+ can return a 3D array (samples, features, classes) or a list of 2D arrays
    if isinstance(shap_results, list):
        # List of 2D arrays (one per class)
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_results], axis=0)
    elif len(shap_results.shape) == 3:
        # 3D array (samples, features, classes)
        # We want mean absolute SHAP across samples and then averaged across classes
        mean_shap = np.abs(shap_results).mean(axis=(0, 2))
    else:
        # 2D array (samples, features) - binary or regression
        mean_shap = np.abs(shap_results).mean(axis=0)
        
    shap_df = pd.DataFrame({'feature': feature_columns, 'importance': mean_shap.tolist()})
    top_8_shap = shap_df.sort_values(by='importance', ascending=False).head(8)
    
    metrics["shap_top8"] = top_8_shap.to_dict(orient='records')
    
    # Save Outputs
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print("Saving models and metrics...")
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "model_xgb.pkl"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "model_rf.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=4)
        
    # Summary Table
    print("\n" + "="*30)
    print(" TRAINING SUMMARY")
    print("="*30)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("-" * 30)
    print("Top 5 SHAP Features:")
    for i, row in top_8_shap.head(5).iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")
    print("="*30)

if __name__ == "__main__":
    download_data()
    train()
