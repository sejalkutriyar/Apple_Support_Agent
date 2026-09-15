import pandas as pd

def load_data(path="data/apple_support_sample.csv"):
    """Load the AppleSupport conversation sample dataset."""
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Total rows:", len(df))
    print("\nColumns:", df.columns.tolist())
    print("\nSample row:")
    print(df.iloc[0])
    print("\nDM-redirect breakdown:")
    print(df['is_dm_redirect'].value_counts())