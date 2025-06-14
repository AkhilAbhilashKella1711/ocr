"""
Helper functions for the semester information extractor.
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
import json


def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict, output_path: str) -> None:
    """
    Save processing results to a JSON file.

    Args:
        results (Dict): Results to save
        output_path (str): Path to save the results
    """
    ensure_directory(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(input_path: str) -> Dict:
    """
    Load processing results from a JSON file.

    Args:
        input_path (str): Path to the results file

    Returns:
        Dict: Loaded results
    """
    with open(input_path, "r") as f:
        return json.load(f)


def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files in a directory.

    Args:
        directory (str): Directory to search

    Returns:
        List[str]: List of PDF file paths
    """
    return [str(f) for f in Path(directory).glob("**/*.pdf")]


def format_semester_output(page_results: Dict) -> str:
    """
    Format semester information for display.

    Args:
        page_results (Dict): Results for a single page

    Returns:
        str: Formatted output string
    """
    if not page_results["has_semester_info"]:
        return f"Page {page_results['page_number']}: No semester information found"

    return (
        f"Page {page_results['page_number']}: "
        f"Found semester information - {page_results['semester_details']} "
        f"(Confidence: {page_results['confidence']:.2f})"
    )
