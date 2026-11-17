
import os
from train import train_model

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy >= 0.8, "Model accuracy is too low"

def test_model_file():
    train_model()
    assert os.path.exists("model.pkl"), "Model file not found"
