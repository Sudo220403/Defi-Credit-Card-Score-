Credit Scoring Model – Aave V2 Protocol
Problem Statement
The goal is to develop a robust machine learning model that assigns a credit score between 0 and 1000 to DeFi wallets based solely on historical transaction behavior from Aave V2 protocol. Higher scores indicate reliable and responsible DeFi usage, while lower scores highlight risky or bot-like behavior.
Project Architecture

user-wallet-transactions.json
       │
       ▼
 feature_engineering.py → Extracts wallet-level features (borrow count, repay ratio, etc.)
       │
       ▼
 model_training.py → Trains Random Forest model on rule-based labels
       │
       ▼
 score_generator.py → Predicts credit score (0-1000) for each wallet

Features Engineered:
- Total transaction count
- Deposit count, Borrow count, Repay count, Liquidation count
- Ratio of repay/borrow
- Number of unique action types
- Active wallet days (first to last transaction)
- Amount-weighted behavior features (e.g., average transaction amount)
Modeling Approach:
• Labeling Strategy: Rule-based scores used for training (bonus points for repay behavior, penalties for liquidation, etc.)
• Model Used: Random Forest Regressor (scikit-learn)
• Scaling: Features are standardized using StandardScaler
How to Run
1)Install dependencies:
pip install -r requirements.txt

2)Train model:
python src/model_training.py --input data/user-wallet-transactions.json
3)Generate scores:
python src/score_generator.py --input data/user-wallet-transactions.json --output wallet_score.json
Project Structure

├── data/
│   └── user-wallet-transactions.json
├── src/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── score_generator.py
│   └── utils.py
├── model.pkl
├── scaler.pkl
├── wallet_score.json
├── README.md
