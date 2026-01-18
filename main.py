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
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Data Loading and Preprocessing
    loader = DataLoader(config['data_paths']['input'])
    df = loader.load_data()

    preprocessor = Preprocessor(config['features']['categorical'])
    df_clean = preprocessor.clean_data(df)
    df_final = preprocessor.encode_features(df_clean)

    # Split into features (X) and target (y)
    target_col = config['features']['target']
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]

    # 2. Model Training
    engine = SalesEngine(config['model_params'])
    score = engine.train(X, y)
    logging.info(f"Successfully trained. R2 Score: {score:.4f}")

    # 3. Model Persistence (Saving)
    # Create the models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    model_path = "models/sales_model.joblib"
    engine.save_model(model_path)

    # 4. Save Preprocessor (Encoders)
    # This is crucial for matching text labels to the same numbers during prediction
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    logging.info("Preprocessor (encoders) saved successfully.")

if __name__ == "__main__":
    main()