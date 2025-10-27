import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    """Clean, balance, and split the dataset."""
    X = df.drop(columns=['disease_risk'])
    y = df['disease_risk']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_resampled, test_size=0.2, random_state=42
    )


    return X_train, X_test, y_train, y_test
