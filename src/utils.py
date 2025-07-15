
import pandas as pd
import matplotlib.pyplot as plt

def load_json_to_df(json_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_score_distribution(scores, bins=10):
    plt.figure(figsize=(8,5))
    plt.hist(scores, bins=bins, edgecolor='black')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.title('Wallet Credit Score Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()