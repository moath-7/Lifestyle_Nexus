from data import load_data
from preprocessing import preprocess_data
from model import train_lightgbm, evaluate_model

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_lightgbm(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if _name_ == "_main_":
    main()
