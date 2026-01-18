import pandas as pd
import logging
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            logging.error(f"File {self.file_path} not found!")
            raise FileNotFoundError

        # Determine file if CSV or Excel based on extension
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(self.file_path)
        else:
            return pd.read_excel(self.file_path, engine='openpyxl')