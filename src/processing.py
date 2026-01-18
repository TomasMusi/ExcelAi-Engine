import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.encoders = {col: LabelEncoder() for col in categorical_cols}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in self.categorical_cols:
            if df_clean[col].dtype == 'object':
                # Deletes spaces and converts to lowercase
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        return df_clean

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_cols:
            df[col] = self.encoders[col].fit_transform(df[col])
        return df