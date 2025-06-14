"""
Trainer script for semester information classifier.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from src.pdf_processor import PDFProcessor
from src.ocr_engine import OCREngine
from .classifier import SemesterClassifier
from sklearn.metrics import accuracy_score
import torch

TRAIN_DIR = "data/training"
TEST_DIR = "data/testing"
MODEL_SAVE_PATH = "models/best_semester_classifier"

# Helper to extract labels from OCR
# For demo, we assume: if semester info is found, label=1 else 0


def extract_labeled_data(pdf_dir):
    pdf_processor = PDFProcessor()
    ocr_engine = OCREngine()
    data = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc=f"Processing {pdf_dir}"):
        page_data = pdf_processor.process_pdf(str(pdf_path))
        for page_key, page_info in page_data.items():
            ocr_result = ocr_engine.process_image(page_info["image"])
            label = (
                1
                if ocr_result["has_semester_info"] and ocr_result["semester_entries"]
                else 0
            )
            data.append({"text": ocr_result["extracted_text"], "label": label})
    return data


def train_and_evaluate():
    print("Extracting training data...")
    train_data = extract_labeled_data(TRAIN_DIR)
    print("Extracting testing data...")
    test_data = extract_labeled_data(TEST_DIR)

    classifier = SemesterClassifier()
    optimizer = torch.optim.Adam(classifier.model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Prepare data
    X_train = [d["text"] for d in train_data]
    y_train = [d["label"] for d in train_data]
    X_test = [d["text"] for d in test_data]
    y_test = [d["label"] for d in test_data]

    # Training loop (simple, 1 epoch for demo)
    classifier.model.train()
    for i, (text, label) in enumerate(
        tqdm(zip(X_train, y_train), total=len(X_train), desc="Training")
    ):
        inputs = classifier.preprocess_text(text)
        labels = torch.tensor([label]).to(classifier.device)
        outputs = classifier.model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    classifier.model.eval()
    y_pred = []
    with torch.no_grad():
        for text in tqdm(X_test, desc="Evaluating"):
            pred = classifier.predict(text)
            y_pred.append(int(pred["has_semester_info"]))
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Save best model
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(MODEL_SAVE_PATH)
    print(f"Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
