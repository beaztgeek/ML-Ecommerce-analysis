import pickle
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "sale_model.pkl"

def load_inference_model(path):
    """Charge le modele exporte pour l'inference."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file '{path}' not found.")
        return None

def predict_intent(model, page_values=0.0, month='Jan', exit_rate=0.0, product_duration=0):
    """
    Simule une session utilisateur et calcule la probabilite de conversion.
    """
    # 1. Preparation du vecteur d'entree
    feature_names = model.feature_names_in_
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # 2. Assignation des variables
    input_df['PageValues'] = page_values
    input_df['ExitRates'] = exit_rate
    input_df['ProductRelated_Duration'] = product_duration
    # Feature engineering indispensable pour la coherence avec l'entrainement
    input_df['Time_Per_Product_Page'] = product_duration / 5  
    
    # One-Hot Encoding du mois
    month_col = f'Month_{month}'
    if month_col in feature_names:
        input_df[month_col] = 1
        
    # 3. Calcul de la probabilite
    probability = model.predict_proba(input_df)[0, 1]
    
    # 4. Sortie console standard
    print(f"REPORT - Month: {month} | PageValue: {page_values}")
    print(f"Probability: {probability:.4f}")
    
    if probability > 0.85:
        print("Verdict: HIGH_INTENT")
    elif probability > 0.40:
        print("Verdict: POTENTIAL")
    else:
        print("Verdict: LOW_INTENT")
    print("-" * 40)

if __name__ == "__main__":
    trained_model = load_inference_model(MODEL_PATH)
    
    if trained_model:
        # Scenario 1 : Intention elevee (Q4 + Panier)
        predict_intent(trained_model, page_values=35.0, month='Nov', exit_rate=0.005, product_duration=500)
        
        # Scenario 2 : Consultation simple (Q1 + Pas de panier)
        predict_intent(trained_model, page_values=0.0, month='Feb', exit_rate=0.05, product_duration=50)
        
        # Scenario 3 : Profil intermediaire
        predict_intent(trained_model, page_values=12.0, month='May', exit_rate=0.015, product_duration=250)