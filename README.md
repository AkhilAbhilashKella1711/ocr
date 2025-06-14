# Semester Information Extractor

This project is an AI-powered application designed to extract and analyze semester information from PDF documents. It utilizes OCR (Optical Character Recognition) to convert PDF pages into images, extracts text, and then applies a trained model to identify and categorize semester and course details. The application includes modules for data handling, model training, testing, and a user-friendly web interface built with Streamlit.

## Features

- **PDF Processing**: Converts PDF pages into images for OCR processing.
- **OCR Engine**: Extracts text from images using Tesseract OCR.
- **Semester Information Detection**: Identifies semester numbers and associated course details using advanced regex patterns.
- **Text Classification Model**: A deep learning model for classifying text related to semesters (currently achieving high accuracy).
- **Streamlit Web UI**: An interactive web interface for uploading PDFs and viewing extracted semester information.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**: This project is developed with Python 3.8 and above.
- **pip**: Python package installer (usually comes with Python).
- **Virtualenv** (optional but recommended): For managing project dependencies.
  ```bash
  pip install virtualenv
  ```
- **Tesseract OCR Engine**: Tesseract is required for the OCR functionality.

  - **macOS (using Homebrew)**:
    ```bash
    brew install tesseract
    ```
  - **Linux (Debian/Ubuntu)**:
    ```bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr
    ```
  - **Windows**: Download the installer from [Tesseract-OCR GitHub](https://tesseract-ocr.github.io/tessdoc/Downloads.html).

## Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

If you haven't already, clone the repository to your local machine:

```bash
git clone <repository_url> # Replace <repository_url> with the actual URL
cd ocr # Or whatever your project directory is named
```

### 2. Set Up Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required Python packages using pip:

```bash
pip install -r requirements.txt
pip install opencv-python pytesseract # Ensure these are explicitly installed if not in requirements.txt
```

### 4. Prepare Data (Optional)

For model training and testing, you might need specific PDF files. Place your PDF files in a `data/` directory at the project root. (e.g., `data/train_pdfs/` and `data/test_pdfs/` if applicable). The current `app.py` is designed to process any uploaded PDF.

## Usage

### 1. Training the Model

The training script is located at `src/model/trainer.py`. This script trains the text classifier model.

```bash
python src/model/trainer.py
```

Upon successful completion, the best-performing model will be saved to `models/best_semester_classifier`. The terminal output will show the training progress and final accuracy (e.g., `Test Accuracy: 1.0000`).

### 2. Testing the Model

To run the unit tests for the PDF processor, OCR engine, and classifier modules:

```bash
python -m unittest discover tests
```

This command will run all tests located in the `tests/` directory. You should see output indicating which tests passed or failed.

### 3. Running the Streamlit UI

To launch the interactive web application:

First, ensure your virtual environment is active:

```bash
source venv/bin/activate
```

Then, run the Streamlit application:

```bash
streamlit run app.py --server.port 8501 --server.address localhost --server.maxUploadSize 200 --server.enableCORS false --server.enableXsrfProtection false
```

After running the command, open your web browser and navigate to:
[http://localhost:8501](http://localhost:8501)

You can then upload a PDF file, and the application will extract and display any detected semester information.

## Troubleshooting

- **`command not found: streamlit`**: This usually means your virtual environment is not activated, or Streamlit is not installed in the currently active environment. Ensure you run `source venv/bin/activate` before running Streamlit commands.
- **`Unable to import 'pytesseract'` or `No module named 'cv2'`**: These errors indicate that `pytesseract` or `opencv-python` are not correctly installed. Run `pip install opencv-python pytesseract` within your activated virtual environment. Also, ensure Tesseract OCR is installed on your system.
- **Application crashes after PDF upload**: Check the terminal output where Streamlit is running for detailed error messages. Ensure your PDF files are not corrupted and that the OCR engine has necessary permissions to process temporary files. The logging in `app.py` can provide more insights.

---
