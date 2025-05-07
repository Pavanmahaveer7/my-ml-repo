from src.iris_model import train_model

def test_train_reaches_minimum_accuracy():
    acc = train_model()
    assert acc >= 0.9          # simple quality gate
