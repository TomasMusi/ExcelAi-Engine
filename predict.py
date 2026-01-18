import yaml
import joblib
import pandas as pd
from src.processing import Preprocessor

def run_inference():
    # Load artifacts
    try:
        model = joblib.load("models/sales_model.joblib")
        preprocessor = joblib.load("models/preprocessor.joblib")
    except FileNotFoundError:
        print("Error: Models not found. Run main.py first.")
        return

    print("\n--- SalesProphet AI: Interactive Prediction ---")
    day = input("Enter Day (e.g., Pondeli): ")
    cat = input("Enter Category (e.g., Elektronika): ")
    price = float(input("Enter Price: "))
    promo = input("Promo (Ano/Ne): ")

    # Prepare input
    data = pd.DataFrame([[day, cat, price, promo]], 
                        columns=['Den_v_tydnu', 'Kategorie', 'Cena_za_kus', 'Promo_akce'])

    # Clean and encode
    data = preprocessor.clean_data(data)
    for col in preprocessor.categorical_cols:
        data[col] = preprocessor.encoders[col].transform(data[col])

    # Predict
    prediction = model.predict(data)
    print(f"\nResult: Predicted Sales -> {prediction[0]:.2f} units\n")

if __name__ == "__main__":
    run_inference()