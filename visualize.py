import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import os

def generate_insights():
    # 1. Load Data and Config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    output_path = config['data_paths']['output']
    
    if not os.path.exists(output_path):
        print(f"Error: {output_path} not found. Run predict.py a few times first!")
        return

    df = pd.read_excel(output_path)
    
    # Create a folder for plots
    os.makedirs("data/plots", exist_ok=True)

    # --- CHART 1: Price vs. Predicted Sales ---
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Cena_za_kus', y='Predicted_Units', scatter_kws={'alpha':0.5})
    plt.title('Price Sensitivity Analysis')
    plt.xlabel('Price (CZK)')
    plt.ylabel('Predicted Units')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('data/plots/price_sensitivity.png')
    print("Saved: data/plots/price_sensitivity.png")

    # --- CHART 2: Feature Importance ---
    # We load the model to see what it learned
    model = joblib.load("models/sales_model.joblib")
    
    # Get feature names (assuming the same order as your training)
    features = config['features']['categorical'] + ['Cena_za_kus']
    importances = model.feature_importances_
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
    plt.title('What Drives Sales? (AI Feature Importance)')
    plt.tight_layout()
    plt.savefig('data/plots/feature_importance.png')
    print("Saved: data/plots/feature_importance.png")

    plt.show()

if __name__ == "__main__":
    generate_insights()