from data.data import DataLoader
from preprocessing.preprocessing import DataPreprocessor
from models.model import HealthRiskModel
import pandas as pd
from rich.console import Console

def main():
    console = Console()

    # Load and explore data
    data_loader = DataLoader('data/health_lifestyle_dataset-20k.csv')
    df = data_loader.load_data()
    data_loader.explore_data()

    # Preprocess data
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_features(df_clean)
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(df_features)

    # Train and evaluate models
    models = ['lightgbm', 'random_forest', 'logistic']
    results = {}

    for model_type in models:
        console.print(f"\n[bold blue]Training {model_type} model...[/bold blue]")
        model = HealthRiskModel(model_type=model_type)
        model.train(X_train, y_train, optimize=True)
        results[model_type] = model.evaluate(X_test, y_test)
        console.print(f"\n{model_type} Results:")
        console.print(f"Accuracy: {results[model_type]['accuracy']:.3f}")
        console.print(results[model_type]['classification_report'])

if __name__ == "__main__":
    main()
