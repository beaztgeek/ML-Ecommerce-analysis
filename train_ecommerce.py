import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# --- CONFIGURATION ---
DATA_PATH = "datas/online_shoppers_intention.csv"
MODEL_NAME = "sale_model.pkl"
RANDOM_STATE = 42

def preprocess_data(df):
    """Nettoyage et Feature Engineering."""
    # Conversion de la cible en binaire (XGBoost requirement)
    y = df['Revenue'].astype(int)
    
    # Feature Engineering : Temps moyen par page produit
    X = df.drop('Revenue', axis=1).copy()
    X['Time_Per_Product_Page'] = X['ProductRelated_Duration'] / (X['ProductRelated'] + 1)
    
    # Encodage des variables catégorielles (One-Hot)
    X_encoded = pd.get_dummies(X, columns=['Month', 'VisitorType', 'Weekend'])
    
    return X_encoded, y

def train_model(X_train, y_train):
    """Initialisation et entraînement du classifieur XGBoost."""
    # Calcul du poids pour gérer le déséquilibre des classes (Class Imbalance)
    # Ratio : sessions_sans_achat / sessions_avec_achat
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio, # Ajustement automatique du poids
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.85):
    """Évaluation avec un seuil de décision personnalisé."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > threshold).astype(int)
    
    print("\n[EVALUATION] Decision Threshold:", threshold)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds))
    
    # Feature Importance analysis
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    print("\nTop 5 Purchase Drivers:")
    print(importances.sort_values(ascending=False).head(5))

def main():
    # 1. Loading
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Pipeline
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 3. Training
    print("Training XGBoost model...")
    model = train_model(X_train, y_train)
    
    # 4. Evaluation
    evaluate_model(model, X_test, y_test)
    
    # 5. Export
    with open(MODEL_NAME, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel exported successfully to {MODEL_NAME}")

if __name__ == "__main__":
    main()