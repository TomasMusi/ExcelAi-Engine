import pandas as pd
import logging

class Preprocessor:
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        # This is the attribute the error is complaining about:
        self.mappings = {col: {} for col in categorical_cols}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in self.categorical_cols:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        return df_clean

    def fit_encoders(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            unique_vals = df[col].unique()
            self.mappings[col] = {val: i for i, val in enumerate(unique_vals)}
            logging.info(f"Feature '{col}' encoded.")

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col in self.categorical_cols:
            df_encoded[col] = df_encoded[col].apply(
                lambda x: self.mappings[col].get(x, -1)
            )
        return df_encoded