import pandas as pd

def extract_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    grouped = df.groupby('userWallet')

    features = []
    for user, group in grouped:
        actions = group['action'].value_counts().to_dict()
        total_tx = len(group)
        active_days = (group['timestamp'].max() - group['timestamp'].min()).days + 1

        repay_count = actions.get('repay', 0)
        borrow_count = actions.get('borrow', 0)
        repay_borrow_ratio = repay_count / borrow_count if borrow_count else 0

        liquidation_count = actions.get('liquidationcall', 0)
        deposit_count = actions.get('deposit', 0)

        features.append({
            'userWallet': user,
            'total_tx': total_tx,
            'repay_borrow_ratio': repay_borrow_ratio,
            'liquidation_count': liquidation_count,
            'deposit_count': deposit_count,
            'borrow_count': borrow_count,
            'active_days': active_days,
            'action_variety': len(actions),
        })

    return pd.DataFrame(features)
