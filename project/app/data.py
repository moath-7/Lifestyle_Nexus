import pandas as pd

def load_data(path="/home/momo/Lifestyle_Nexus/data/health_lifestyle_dataset-20k.csv"):
    """Load and return the health lifestyle dataset."""
    df = pd.read_csv(path)
    df_clean = df.drop(columns=['id'])
    df_clean['gender'] = df_clean['gender'].map({'Male': 0, 'Female': 1})
    return df_clean
