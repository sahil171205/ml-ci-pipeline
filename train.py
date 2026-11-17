
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train ML model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    # Save trained model
    joblib.dump(model, "model.pkl")

    return accuracy

if __name__ == "__main__":
    train_model()
