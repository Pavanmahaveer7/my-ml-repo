from src import iris_model as m

def test_full_pipeline_smoke(tmp_path):
    # train → predict → no exceptions
    m.MODEL_PATH = tmp_path / "model.pkl"   # keep workspace clean
    m.train_model()
    pred = m.predict([5.1, 3.5, 1.4, 0.2])
    assert pred in (0, 1, 2)
