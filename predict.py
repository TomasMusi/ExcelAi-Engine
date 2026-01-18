import yaml
import joblib
import pandas as pd
import logging
import os
from datetime import datetime

# Configure logging to show only messages
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_inference():
    # 1. Load the configuration
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found in configs/config.yaml")
        return

    # 2. Load the saved model and the robust preprocessor
    try:
        model = joblib.load("models/sales_model.joblib")
        preprocessor = joblib.load("models/preprocessor.joblib")
    except FileNotFoundError:
        logging.error("Model files not found. Please run 'python3 main.py' first to train the model.")
        return

    print("\n" + "="*40)
    print("   SalesProphet AI: Prediction Tool")
    print("="*40)
    
    try:
        # 3. Get user input
        print("\nEnter details for prediction (Case insensitive):")
        day = input("Day (e.g., Pondeli, Utery...): ")
        category = input("Category (e.g., Elektronika, Obleceni...): ")
        price = float(input("Price per unit (e.g., 500): "))
        promo = input("Promo active? (Ano/Ne): ")

        # 4. Create DataFrame for input
        input_data = pd.DataFrame([[day, category, price, promo]], 
                                columns=['Den_v_tydnu', 'Kategorie', 'Cena_za_kus', 'Promo_akce'])

        # 5. Robust Preprocessing
        cleaned_data = preprocessor.clean_data(input_data)
        encoded_data = preprocessor.transform_data(cleaned_data)

        # 6. Generate Prediction
        prediction = model.predict(encoded_data)[0]

        print("\n" + "-"*40)
        print(f"PREDICTED SALES: {prediction:.2f} units")
        print("-"*40 + "\n")

        # 7. EXPORT TO EXCEL
        output_path = config['data_paths']['output']
        
        # Prepare the new entry with a timestamp
        new_entry = input_data.copy()
        new_entry['Predicted_Units'] = round(prediction, 2)
        new_entry['Predicted_At'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if file exists to append or create new
        if os.path.exists(output_path) and output_path.endswith('.xlsx'):
            existing_df = pd.read_excel(output_path, engine='openpyxl')
            final_df = pd.concat([existing_df, new_entry], ignore_index=True)
        else:
            final_df = new_entry

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Excel
        final_df.to_excel(output_path, index=False, engine='openpyxl')
        logging.info(f"Result saved to: {output_path}")

    except ValueError:
        logging.error("Invalid input! Please ensure Price is a number.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_inference()