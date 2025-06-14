"""
Classifier module for semester information detection and classification.
"""

from typing import Dict, List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SemesterClassifier:
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the semester classifier.

        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification: has semester info or not
        ).to(self.device)

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input.

        Args:
            text (str): Input text

        Returns:
            Dict[str, torch.Tensor]: Tokenized and processed text
        """
        return self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

    def predict(self, text: str) -> Dict:
        """
        Predict whether text contains semester information.

        Args:
            text (str): Input text

        Returns:
            Dict: Prediction results with confidence scores
        """
        inputs = self.preprocess_text(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

        return {"has_semester_info": bool(prediction), "confidence": confidence}

    def train(
        self, train_data: List[Dict], validation_data: Optional[List[Dict]] = None
    ):
        """
        Train the classifier on provided data.

        Args:
            train_data (List[Dict]): List of training examples
            validation_data (Optional[List[Dict]]): List of validation examples
        """
        # Training implementation will be added based on the training data format
        pass

    def save_model(self, path: str):
        """
        Save the trained model.

        Args:
            path (str): Path to save the model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        """
        Load a trained model.

        Args:
            path (str): Path to the saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
