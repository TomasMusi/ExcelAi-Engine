import yaml
import logging
import os
import joblib
from src.loader import DataLoader
from src.processing import Preprocessor
from src.engine import SalesEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    # Load configuration
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found in configs/config.yaml")
        return

    # 1. Data Loading and Preprocessing
    loader = DataLoader(config['data_paths']['input'])
    df = loader.load_data()

    preprocessor = Preprocessor(config['features']['categorical'])
    
    df_clean = preprocessor.clean_data(df)

    preprocessor.fit_encoders(df_clean)
    df_final = preprocessor.transform_data(df_clean)

    # Split into features (X) and target (y)
    target_col = config['features']['target']
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]

    # 2. Model Training
    engine = SalesEngine(config['model_params'])
    score = engine.train(X, y)
    logging.info(f"Successfully trained. R2 Score: {score:.4f}")

    # 3. Model Persistence (Saving)
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Save the trained AI model
    model_path = "models/sales_model.joblib"
    engine.save_model(model_path)

    # 4. Save Preprocessor (Mappings)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    logging.info("Preprocessor (robust mappings) saved successfully.")

if __name__ == "__main__":
    main()