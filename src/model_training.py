import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from feature_engineering import extract_features


def train_model(input_json_path):
    # Load transaction data
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Feature extraction
    features_df = extract_features(df)

    # Basic rule-based credit score labeling
    def label_score(row):
        score = 500
        if row['repay_borrow_ratio'] > 0.5:
            score += 200
        if row['liquidation_count'] > 0:
            score -= 200
        if row['deposit_count'] > 5:
            score += 100
        if row['active_days'] > 60:
            score += 100
        if row['borrow_count'] == 0:
            score -= 100
        if row['action_variety'] >= 3:
            score += 100
        return max(0, min(1000, score))

    features_df['score'] = features_df.apply(label_score, axis=1)

    X = features_df.drop(columns=['userWallet', 'score'])
    y = features_df['score']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save model and scaler in ROOT folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    joblib.dump(model, os.path.join(root_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(root_dir, 'scaler.pkl'))

    print("Model and Scaler saved successfully in project root!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Credit Scoring Model")
    parser.add_argument('--input', default='../data/user-wallet-transactions.json', help='Input transaction JSON path')
    args = parser.parse_args()

    train_model(args.input)
