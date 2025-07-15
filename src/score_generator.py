import json
import pandas as pd
from feature_engineering import extract_features
import joblib
import os


def generate_scores(input_json_path, output_json_path):
    # Load transaction JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Feature extraction
    features_df = extract_features(df)

    # Load model and scaler from root folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(root_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(root_dir, 'scaler.pkl'))

    # Prepare features (drop wallet column)
    X = scaler.transform(features_df.drop(columns=['userWallet']))

    # Predict credit scores
    scores = model.predict(X)

    # Convert numpy int64 to Python int before JSON dump
    result = {wallet: int(round(score)) for wallet, score in zip(features_df['userWallet'], scores)}

    # Save wallet scores to output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Wallet scores saved successfully to {output_json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeFi Credit Score Generator")
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output wallet_score.json')
    args = parser.parse_args()

    generate_scores(args.input, args.output)
