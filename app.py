import streamlit as st
import tempfile
import os
import logging
from pathlib import Path
from src.pdf_processor import PDFProcessor
from src.ocr_engine import OCREngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Cache the PDF processor and OCR engine
@st.cache_resource
def get_pdf_processor():
    return PDFProcessor(dpi=300)


@st.cache_resource
def get_ocr_engine():
    return OCREngine()


def process_pdf(pdf_path: str) -> list:
    """Process a single PDF file and return semester information."""
    try:
        logger.info(f"Starting to process PDF: {pdf_path}")

        # Initialize processors
        pdf_processor = PDFProcessor(dpi=300)
        ocr_engine = OCREngine()

        # Convert PDF to images
        logger.info("Converting PDF to images...")
        page_data = pdf_processor.process_pdf(pdf_path)
        logger.info(f"Found {len(page_data)} pages")

        results = []
        for page_key, page_info in page_data.items():
            try:
                page_num = page_info["page_number"]
                logger.info(f"Processing page {page_num}")

                # Process image with OCR
                ocr_result = ocr_engine.process_image(page_info["image"])
                logger.info(f"OCR result for page {page_num}: {ocr_result}")

                # Extract semester information
                if ocr_result.get("semester_entries"):
                    entries = []
                    for entry in ocr_result["semester_entries"]:
                        semester = entry.get("semester", "Unknown")
                        course = entry.get("course", "Course not specified")
                        entries.append(f"{semester} - {course}")
                    if entries:  # Only add if we found entries
                        results.append((page_num, entries))
                    else:
                        results.append((page_num, ["No semester information found"]))
                else:
                    results.append((page_num, ["No semester information found"]))

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                results.append((page_num, [f"Error: {str(e)}"]))

        return sorted(results, key=lambda x: x[0])  # Sort by page number

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise


# Set page config
st.set_page_config(page_title="Semester Info Extractor", page_icon="📄", layout="wide")

# Title and description
st.title("📄 Semester Information Extractor")
st.write(
    "Upload a PDF file to extract and display semester information from each page."
)

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
            logger.info(f"Saved uploaded file to: {pdf_path}")

        # Show processing message
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Process the PDF
            results = process_pdf(pdf_path)

            if results:
                st.success("Processing complete!")
                st.subheader("Results:")

                # Display results
                for page_num, entries in results:
                    with st.expander(f"Page {page_num}", expanded=True):
                        for entry in entries:
                            st.markdown(f"• {entry}")
            else:
                st.warning("No semester information found in the PDF.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main processing: {str(e)}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
            logger.info("Cleaned up temporary file")
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")
