from sklearn.ensemble import RandomForestRegressor
import logging # For logging
import joblib # For saving and loading models

class SalesEngine:
    def __init__(self, model_params):
        self.model = RandomForestRegressor(**model_params, n_jobs=-1)

    def train(self, X, y):
        logging.info("Training AI model (Random Forest)...")
        self.model.fit(X, y)
        return self.model.score(X, y)

    def save_model(self, path):
        """Save trained model to file."""
        joblib.dump(self.model, path)
        logging.info(f"Model was saved to: {path}")

    def load_model(self, path):
        """Loads model from file."""
        self.model = joblib.load(path)
        logging.info(f"Model was loaded from: {path}")

    def predict(self, X_input):
        return self.model.predict(X_input)