import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.console = Console()

    def load_data(self):
        """Load and perform initial data exploration"""
        self.data = pd.read_csv(self.filepath)
        return self.data

    def explore_data(self):
        """Perform basic data exploration"""
        print("Dataset Shape:", self.data.shape)
        print("\nMissing Values:\n", self.data.isnull().sum())
        print("\nDuplicate Rows:", self.data.duplicated().sum())

    def plot_water_intake_by_gender(self):
        """Plot gender comparison by water intake"""
        low_water = self.data[self.data['water_intake_l'] < 2]
        high_water = self.data[self.data['water_intake_l'] >= 2]

        low_gender = low_water['gender'].value_counts()
        high_gender = high_water['gender'].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        axes[0].pie(low_gender, labels=low_gender.index, autopct='%1.1f%%',
                   colors=['#66b3ff','#ffb3e6'])
        axes[1].pie(high_gender, labels=high_gender.index, autopct='%1.1f%%',
                   colors=['#66b3ff','#ffb3e6'])
        plt.show()
