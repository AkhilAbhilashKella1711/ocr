"""
OCR Engine module for extracting text from images using Tesseract OCR.
"""

import pytesseract
from typing import Dict, List, Optional, Set
import cv2
import numpy as np
import re


class OCREngine:
    def __init__(self, lang: str = "eng"):
        """
        Initialize the OCR engine.

        Args:
            lang (str): Language for OCR (default: 'eng')
        """
        self.lang = lang
        self.semester_patterns = [
            r"(?:1st|2nd|3rd|[4-8]th)\s*semester",
            r"(?:1st|2nd|3rd|[4-8]th)\s*sem",
            r"semester\s*(?:1st|2nd|3rd|[4-8]th)",
            r"sem\s*(?:1st|2nd|3rd|[4-8]th)",
            r"semester\s*[1-8]",
            r"sem\s*[1-8]",
            r"[1-8]\s*semester",
            r"[1-8]\s*sem",
            r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s+semester\b",
            r"\b(?:i|ii|iii|iv|v|vi|vii|viii)\s*(?:st|nd|rd|th)?\s*sem(?:ester)?\b",  # Roman numerals
            r"\b(?:[1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?\b",  # e.g., 1st sem, 2nd semester
        ]

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image (np.ndarray): Input image

        Returns:
            str: Extracted text
        """
        try:
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
        except Exception as e:
            raise Exception(f"Error in OCR processing: {str(e)}")

    def normalize_semester_text(self, text: str) -> str:
        """
        Normalize semester text to a standard format.

        Args:
            text (str): Raw semester text

        Returns:
            str: Normalized semester text
        """
        # Convert to lowercase and remove extra spaces
        text = " ".join(text.lower().split())

        # Replace variations with standard format
        replacements = {
            r"sem\s*(\d+)": r"Semester \1",
            r"(\d+)\s*sem": r"Semester \1",
            r"(\d+)(?:st|nd|rd|th)\s*semester": r"Semester \1",
            r"semester\s*(\d+)(?:st|nd|rd|th)": r"Semester \1",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text

    def detect_semester_info(self, text: str) -> Dict:
        """
        Detect semester information from extracted text.

        Args:
            text (str): Extracted text from image

        Returns:
            Dict: Dictionary containing semester information with multiple entries
        """
        text = text.lower()
        semester_info = {
            "has_semester_info": False,
            "semester_entries": [],
            "confidence": 0.0,
        }

        # Find all semester matches
        all_matches = []
        for pattern in self.semester_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            all_matches.extend(matches)

        if all_matches:
            semester_info["has_semester_info"] = True
            semester_info["confidence"] = 0.9

            # Track unique semester numbers to avoid duplicates
            seen_semesters: Set[str] = set()

            # Process each match
            for match in all_matches:
                semester_text = self.normalize_semester_text(match.group())

                # Extract semester number
                sem_num_match = re.search(r"\d+", semester_text)
                if not sem_num_match:
                    continue

                sem_num = sem_num_match.group()
                if sem_num in seen_semesters:
                    continue

                seen_semesters.add(sem_num)

                # Look for course information near this semester mention
                # Search in a window of 200 characters before and after the match
                start_pos = max(0, match.start() - 200)
                end_pos = min(len(text), match.end() + 200)
                context = text[start_pos:end_pos]

                course_info = None
                course_patterns = [
                    r"b\.?\s*tech(?:nology)?",
                    r"bachelor\s*of\s*technology",
                    r"b\.?\s*com(?:merce)?",
                    r"bachelor\s*of\s*commerce",
                    r"mca",
                    r"master\s*of\s*computer\s*applications",
                    r"bba",
                    r"bachelor\s*of\s*business\s*administration",
                    r"mba",
                    r"master\s*of\s*business\s*administration",
                    r"b\.sc",
                    r"bachelor\s*of\s*science",
                    r"m\.sc",
                    r"master\s*of\s*science",
                    r"ba",
                    r"bachelor\s*of\s*arts",
                    r"ma",
                    r"master\s*of\s*arts",
                    r"ph(?:\.?d)",
                    r"doctor\s*of\s*philosophy",
                    r"engineering",
                    r"engg",
                    r"diploma",
                    # Add more course patterns as needed
                ]

                for pattern in course_patterns:
                    course_match = re.search(pattern, context, re.IGNORECASE)
                    if course_match:
                        course_name = course_match.group()
                        course_name = course_name.replace(r"\.?\s*", " ").strip()
                        course_name = course_name.replace("\\", "")
                        course_info = course_name
                        break

                # Create semester entry
                entry = {
                    "semester": semester_text,
                    "course": course_info,
                    "semester_number": int(sem_num),
                }
                semester_info["semester_entries"].append(entry)

            # Sort entries by semester number
            semester_info["semester_entries"].sort(key=lambda x: x["semester_number"])

        return semester_info

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Process an image to extract text and detect semester information.

        Args:
            image (np.ndarray): Input image

        Returns:
            Dict: Dictionary containing extracted text and semester information
        """
        text = self.extract_text(image)
        semester_info = self.detect_semester_info(text)

        return {"extracted_text": text, **semester_info}
