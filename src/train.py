import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


DATA_PATH = "data/customers.csv"
MODEL_PATH = "model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    return X, y


def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if f1 < 0.70:
      raise ValueError("Model failed quality gate: F1 < 0.70")

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

    return accuracy, f1


if __name__ == "__main__":
    train_model()