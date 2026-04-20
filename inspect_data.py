import pandas as pd

def quick_inspect(file_path="datas/online_shoppers_intention.csv"):
    """
    Analyse de premier niveau du dataset e-commerce.
    """
    df = pd.read_csv(file_path)
    
    print(f"--- Dataset Loaded: {file_path} ---")
    print(f"Total sessions: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    
    # Vérification des valeurs manquantes (essentiel pour GitHub)
    null_counts = df.isnull().sum().sum()
    print(f"Missing values: {null_counts}")
    
    print("\n--- Target Distribution (Revenue) ---")
    print(df['Revenue'].value_counts(normalize=True))

if __name__ == "__main__":
    quick_inspect()