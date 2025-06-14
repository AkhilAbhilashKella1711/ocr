"""
PDF Processor module for handling PDF files and converting them to images.
"""

import os
from typing import Dict, List, Optional, Tuple
import pdf2image
import cv2
import numpy as np
from pathlib import Path


class PDFProcessor:
    def __init__(self, dpi: int = 300):
        """
        Initialize the PDF processor.

        Args:
            dpi (int): DPI for PDF to image conversion
        """
        self.dpi = dpi

    def convert_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            List[np.ndarray]: List of images, one for each page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
            return [np.array(image) for image in images]
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)

        return denoised

    def process_pdf(self, pdf_path: str) -> Dict[str, Dict]:
        """
        Process a PDF file and return page-wise information.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict[str, Dict]: Dictionary containing page-wise information
        """
        images = self.convert_pdf_to_images(pdf_path)
        results = {}

        for idx, image in enumerate(images, 1):
            processed_image = self.preprocess_image(image)
            results[f"page_{idx}"] = {
                "image": processed_image,
                "page_number": idx,
                "original_size": image.shape[:2],
            }

        return results

    def save_processed_images(self, images: Dict[str, Dict], output_dir: str) -> None:
        """
        Save processed images to disk.

        Args:
            images (Dict[str, Dict]): Dictionary of processed images
            output_dir (str): Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for page_key, page_data in images.items():
            image = page_data["image"]
            output_file = output_path / f"{page_key}.png"
            cv2.imwrite(str(output_file), image)
