"""
Test cases for the semester classifier module.
"""

import pytest
import torch
from src.model.classifier import SemesterClassifier
from transformers import BatchEncoding


@pytest.fixture
def classifier():
    return SemesterClassifier()


def test_classifier_initialization(classifier):
    assert classifier.device in [torch.device("cuda"), torch.device("cpu")]
    assert classifier.tokenizer is not None
    assert classifier.model is not None


def test_preprocess_text(classifier):
    text = "This is Semester 1 of B.Com"
    inputs = classifier.preprocess_text(text)

    # Accept both dict and BatchEncoding (from transformers)
    assert isinstance(
        inputs, (dict, BatchEncoding)
    ), "Inputs should be a dict or BatchEncoding"
    assert all(
        key in inputs for key in ["input_ids", "attention_mask"]
    ), "Missing required keys"
    assert isinstance(inputs["input_ids"], torch.Tensor), "input_ids should be a tensor"
    assert isinstance(
        inputs["attention_mask"], torch.Tensor
    ), "attention_mask should be a tensor"


def test_predict(classifier):
    text = "This is Semester 1 of B.Com"
    result = classifier.predict(text)

    assert isinstance(result, dict)
    assert "has_semester_info" in result
    assert "confidence" in result
    assert isinstance(result["has_semester_info"], bool)
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1


def test_save_and_load_model(classifier, tmp_path):
    # Save model
    model_path = str(tmp_path / "test_model")
    classifier.save_model(model_path)

    # Create new classifier and load model
    new_classifier = SemesterClassifier()
    new_classifier.load_model(model_path)

    # Test if loaded model works
    text = "This is Semester 1 of B.Com"
    original_result = classifier.predict(text)
    loaded_result = new_classifier.predict(text)

    assert original_result["has_semester_info"] == loaded_result["has_semester_info"]
    # Note: Confidence scores might differ slightly due to model initialization
    assert abs(original_result["confidence"] - loaded_result["confidence"]) < 0.1
