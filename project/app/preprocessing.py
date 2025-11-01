import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(user_input):
    """Clean, balance, and split the dataset."""
    df = pd.DataFrame(user_input)

    scaler = StandardScaler()
    X_scaled = scaler.transform(df)

    return X_scaled
