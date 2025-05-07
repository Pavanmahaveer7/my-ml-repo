from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).parent / "iris.pkl"


def train_model(random_state: int = 42):
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    joblib.dump(clf, MODEL_PATH)
    return accuracy_score(yte, clf.predict(Xte))


def predict(sample):
    """sample = list[float] length 4"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained – call train_model() first")
    clf = joblib.load(MODEL_PATH)
    return clf.predict([sample])[0]
