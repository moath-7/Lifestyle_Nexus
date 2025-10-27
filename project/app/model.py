import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from scipy.stats import randint, uniform
from project.app.data import load_data
from project.app.preprocessing import preprocess_data

def train_lightgbm(X_train, y_train):
    """Train a LightGBM model with tuned hyperparameters."""
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(20, 60),
        'max_depth': randint(3, 15),
        'min_child_samples': randint(20, 100),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }

    model = LGBMClassifier(random_state=42)
    f1_scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        scoring=f1_scorer,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    print("Best parameters (LightGBM):", random_search.best_params_)
    best_model = random_search.best_estimator_
    joblib.dump(best_model, "/Lifestyle_Nexus/models/lightgbm_model.pkl")
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy and report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nTuned LightGBM Model Performance (Balanced Data):")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return acc

def load_model():
    model = joblib.load("/Lifestyle_Nexus/models/lightgbm_model.pkl")
    return model


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    train_lightgbm(X_train, y_train)

if __name__ == "__main__":
    main()
