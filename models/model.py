from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

class HealthRiskModel:
    def __init__(self, model_type='lightgbm'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.best_model = None

    def _initialize_model(self):
        if self.model_type == 'lightgbm':
            return LGBMClassifier(random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=42)
        elif self.model_type == 'logistic':
            return LogisticRegression(max_iter=2000)

    def train(self, X_train, y_train, optimize=True):
        if optimize:
            param_dist = self._get_param_distribution()
            random_search = RandomizedSearchCV(
                self.model, param_dist, n_iter=50, cv=5,
                n_jobs=-1, random_state=42
            )
            random_search.fit(X_train, y_train)
            self.best_model = random_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)
            self.best_model = self.model

    def predict(self, X):
        return self.best_model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
