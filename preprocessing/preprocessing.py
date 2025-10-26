import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def clean_data(self, df):
        """Basic data cleaning"""
        df_clean = df.copy()
        df_clean = df_clean.drop(columns=['id'])
        df_clean['gender'] = df_clean['gender'].map({'Male': 0, 'Female': 1})
        return df_clean

    def create_features(self, df):
        """Create additional features"""
        df['activity_level'] = pd.cut(
            df['daily_steps'],
            bins=[0, 5000, 10000, 15000, 20000],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
        return df

    def prepare_training_data(self, df, target_col='disease_risk'):
        """Prepare data for model training"""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        X_scaled = self.scaler.fit_transform(X_resampled)

        return train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
