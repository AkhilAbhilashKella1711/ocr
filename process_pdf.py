"""
Script to process a PDF file and extract semester information.
"""

import os
from src.pdf_processor import PDFProcessor
from src.ocr_engine import OCREngine
from pathlib import Path


def format_semester_entry(entry: dict) -> str:
    """
    Format a single semester entry for display.

    Args:
        entry (dict): Semester entry containing semester and course information

    Returns:
        str: Formatted semester entry
    """
    semester = entry["semester"]
    course = entry["course"] if entry["course"] else "Course not specified"
    return f"{semester} - {course}"


def process_single_pdf(pdf_path: str) -> None:
    """
    Process a single PDF file and display semester information for each page.

    Args:
        pdf_path (str): Path to the PDF file
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return

    print(f"\nProcessing PDF: {Path(pdf_path).name}")
    print("-" * 50)

    # Initialize processors
    pdf_processor = PDFProcessor(dpi=300)  # Higher DPI for better accuracy
    ocr_engine = OCREngine()

    try:
        # Process PDF and get images
        print("Converting PDF to images...")
        page_data = pdf_processor.process_pdf(pdf_path)

        # Process each page
        print("\nAnalyzing pages for semester information...")
        print("-" * 50)

        for page_key, page_info in page_data.items():
            # Extract text and detect semester info
            ocr_result = ocr_engine.process_image(page_info["image"])
            page_num = page_info["page_number"]

            if ocr_result["has_semester_info"] and ocr_result["semester_entries"]:
                print(f"\nPage {page_num}:")
                for entry in ocr_result["semester_entries"]:
                    entry_str = format_semester_entry(entry)
                    print(f"  • {entry_str}")
            else:
                print(f"\nPage {page_num}: NA")

        print("\n" + "-" * 50)
        print("Processing complete!")

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    # Process the specific PDF file
    pdf_file = "data/training/Application_724649266.pdf"
    process_single_pdf(pdf_file)
