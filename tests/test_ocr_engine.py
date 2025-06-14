"""
Test cases for the OCR engine module.
"""

import pytest
import numpy as np
from src.ocr_engine import OCREngine


@pytest.fixture
def ocr_engine():
    return OCREngine()


@pytest.fixture
def sample_image():
    # Create a sample image with text
    image = np.zeros((100, 300), dtype=np.uint8)
    # Add some white text on black background
    image[40:60, 50:250] = 255
    return image


def test_ocr_engine_initialization(ocr_engine):
    assert ocr_engine.lang == "eng"
    assert len(ocr_engine.semester_patterns) > 0


def test_detect_semester_info(ocr_engine):
    # Test with semester information
    text = "This is Semester 1 of B.Com program"
    result = ocr_engine.detect_semester_info(text)
    assert result["has_semester_info"]
    assert "semester" in result["semester_details"].lower()

    # Test without semester information
    text = "This is a regular text without semester info"
    result = ocr_engine.detect_semester_info(text)
    assert not result["has_semester_info"]
    assert result["semester_details"] is None


def test_process_image(ocr_engine, sample_image):
    result = ocr_engine.process_image(sample_image)
    assert isinstance(result, dict)
    assert "extracted_text" in result
    assert "has_semester_info" in result
    assert "semester_details" in result
    assert "confidence" in result


def test_semester_patterns(ocr_engine):
    test_cases = [
        ("Semester 1", True),
        ("Sem 2", True),
        ("3rd Semester", True),
        ("4th Sem", True),
        ("No semester here", False),
        ("B.Com Semester 1", True),
        ("Bachelor of Commerce Sem 2", True),
    ]

    for text, expected in test_cases:
        result = ocr_engine.detect_semester_info(text)
        assert result["has_semester_info"] == expected, f"Failed for text: {text}"
