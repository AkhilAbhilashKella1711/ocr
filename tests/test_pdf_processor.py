"""
Test cases for the PDF processor module.
"""

import pytest
import numpy as np
from pathlib import Path
from src.pdf_processor import PDFProcessor


@pytest.fixture
def pdf_processor():
    return PDFProcessor()


@pytest.fixture
def sample_pdf_path(tmp_path):
    # Create a dummy PDF file for testing
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    return str(pdf_path)


def test_pdf_processor_initialization(pdf_processor):
    assert pdf_processor.dpi == 300


def test_convert_pdf_to_images_nonexistent_file(pdf_processor):
    with pytest.raises(FileNotFoundError):
        pdf_processor.convert_pdf_to_images("nonexistent.pdf")


def test_preprocess_image(pdf_processor):
    # Create a sample image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    processed = pdf_processor.preprocess_image(image)

    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == 2  # Should be grayscale
    assert processed.dtype == np.uint8


def test_process_pdf_invalid_file(pdf_processor):
    with pytest.raises(FileNotFoundError):
        pdf_processor.process_pdf("nonexistent.pdf")


def test_save_processed_images(pdf_processor, tmp_path):
    # Create sample processed images
    images = {
        "page_1": {
            "image": np.zeros((100, 100), dtype=np.uint8),
            "page_number": 1,
            "original_size": (100, 100),
        }
    }

    output_dir = str(tmp_path / "output")
    pdf_processor.save_processed_images(images, output_dir)

    # Check if the output file exists
    output_file = Path(output_dir) / "page_1.png"
    assert output_file.exists()
