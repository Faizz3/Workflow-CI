import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import os

def run_training():
    mlflow.autolog()
    
    # Memuat data (untuk CI kita gunakan sklearn datasets agar self-contained)
    data = load_wine()
    X = pd.DataFrame(StandardScaler().fit_transform(data.data), columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="MLProject_CI_Model"):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        print("Model berhasil dilatih via MLProject!")

if __name__ == "__main__":
    run_training()
